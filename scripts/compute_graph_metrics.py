#!/usr/bin/env python3
"""
compute_graph_metrics.py — Citation-graph centralities + Louvain communities
for the active library (WS-A of the agentic-retrieval upgrade,
docs/agentic_retrieval_upgrade_plan.md).

Reads (READ-ONLY) <data_root>/data/citation_graph.json — node key = chroma
metadata.source — and computes on the DIRECTED citation graph
([citing, cited] edges):
    pagerank (alpha=0.85), HITS (hub + authority), exact betweenness
plus Louvain communities on the UNDIRECTED projection (resolution 1.0,
fixed seed → deterministic partition; communities numbered by size
descending, so id 0 = the largest cluster).

Outputs:
  1. <data_root>/data/graph_metrics.json — per-source metrics (the 6
     computed fields + the provenance triple citation_count / rcr /
     in_corpus_cited_by carried through from the graph nodes), plus
     graph-level stats (community count, modularity, per-metric maxima
     for downstream normalization) and an `aliases` map (FTS stem →
     source) so BM25-channel entries — whose `source` is the sanitized
     parent_store stem ("10068468_md") — resolve to the same record as
     the dense channel ("10068468.md"). Ambiguous stems (sanitize
     collisions) are dropped: a wrong boost is worse than no boost.
  2. Chroma metadata upsert (Phase D pattern from eph_rag's
     build_citation_graph_full.py): adds pagerank/authority/hub/
     betweenness/community_id/community_size to every chunk of a
     matching source. Brief write-locks during the upsert are
     acceptable (precedent).

Consumers: src/reranker.py (provenance boost, env BIB_RAG_PROVENANCE) and
src/agent_tools.py check_evidence_coverage (community concentration check).

Usage:
    eph-rag scripts/compute_graph_metrics.py               # compute + chroma upsert
    eph-rag scripts/compute_graph_metrics.py --no-chroma   # metrics file only
"""
import os
import re
import sys
import json
import math
import time
import argparse
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from kb_config import parse_kb_arg, get_config, print_config  # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _sanitize(source: str) -> str:
    """parent_store filename-stem rule (ParentStoreManager._safe_filename)."""
    return re.sub(r"[^\w\-]", "_", source)[:100]


def _dist_line(name: str, values) -> str:
    vals = sorted(v for v in values if v is not None)
    n = len(vals)
    if not n:
        return f"  {name:<14} (no values)"
    med = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
    nz = sum(1 for v in vals if v > 0)
    return (f"  {name:<14} min={vals[0]:.3g} p50={med:.3g} "
            f"mean={sum(vals)/n:.3g} max={vals[-1]:.3g} nonzero={nz}/{n}")


def main():
    argv = parse_kb_arg()
    ap = argparse.ArgumentParser(
        description="Citation-graph centralities + Louvain communities "
                    "→ data/graph_metrics.json (+ optional chroma upsert)")
    ap.add_argument("--no-chroma", action="store_true",
                    help="compute + write graph_metrics.json only; skip the "
                         "chroma metadata upsert")
    args = ap.parse_args(argv)

    cfg = get_config()
    print_config()
    print()

    graph_path = os.path.join(cfg["data_dir"], "citation_graph.json")
    if not os.path.exists(graph_path):
        log(f"NO_CITATION_GRAPH: {graph_path} not found — nothing to "
            f"compute. (build it with scripts/build_reference_graph.py --icite)")
        sys.exit(1)

    with open(graph_path, encoding="utf-8") as f:
        g = json.load(f)
    nodes, edges = g["nodes"], g["edges"]
    log(f"graph: {len(nodes)} nodes, {len(edges)} edges "
        f"({g.get('edge_def', '')})")

    import networkx as nx
    G = nx.DiGraph()
    G.add_nodes_from(nodes.keys())
    G.add_edges_from((s, t) for s, t in edges)  # [citing, cited]

    # -- centralities on the directed graph --------------------------------
    log("pagerank (alpha=0.85) ...")
    pr = nx.pagerank(G, alpha=0.85)

    log("HITS (hub/authority) ...")
    hits_mode = "converged"
    try:
        hub, auth = nx.hits(G, max_iter=200, normalized=True)
    except nx.PowerIterationFailedConvergence:
        try:
            log("  HITS did not converge in 200 iters — retrying max_iter=1000")
            hub, auth = nx.hits(G, max_iter=1000, normalized=True)
            hits_mode = "converged_max_iter_1000"
        except nx.PowerIterationFailedConvergence:
            # last-resort fallback: degree ratios (same [0,1]-ish scale,
            # same semantics direction — authority ∝ being cited)
            log("  HITS failed — falling back to normalized in/out-degree")
            n_in = dict(G.in_degree())
            n_out = dict(G.out_degree())
            si = sum(n_in.values()) or 1
            so = sum(n_out.values()) or 1
            auth = {s: n_in[s] / si for s in G}
            hub = {s: n_out[s] / so for s in G}
            hits_mode = "degree_fallback"

    log("betweenness (exact Brandes — may take a few minutes on 3k nodes) ...")
    t0 = time.time()
    bt = nx.betweenness_centrality(G, normalized=True)
    log(f"betweenness done in {time.time()-t0:.0f}s")

    # -- Louvain communities on the undirected projection -------------------
    GU = G.to_undirected()  # reciprocal citation pairs merge into one edge
    log("louvain communities (undirected projection, resolution=1.0, seed=7) ...")
    comms = nx.community.louvain_communities(
        GU, weight=None, resolution=1.0, seed=7)
    # deterministic ids: largest first, ties broken by smallest source name
    comms.sort(key=lambda c: (-len(c), min(c)))
    comm_of, comm_size = {}, {}
    for cid, comm in enumerate(comms):
        for s in comm:
            comm_of[s] = cid
            comm_size[s] = len(comm)
    modularity = nx.community.modularity(GU, comms, resolution=1.0)
    log(f"communities: {len(comms)} (modularity {modularity:.4f})")

    # -- per-source records ---------------------------------------------------
    metrics = {}
    for s, nd in nodes.items():
        metrics[s] = {
            "pagerank": round(pr.get(s, 0.0), 8),
            "authority": round(auth.get(s, 0.0), 8),
            "hub": round(hub.get(s, 0.0), 8),
            "betweenness": round(bt.get(s, 0.0), 8),
            "community_id": comm_of.get(s, -1),
            "community_size": comm_size.get(s, 0),
            # provenance triple carried through from the graph nodes so the
            # reranker has ONE lookup for every boost signal
            "citation_count": int(nd.get("citation_count") or 0),
            "rcr": round(float(nd.get("rcr") or 0), 3),
            "in_corpus_cited_by": int(nd.get("in_corpus_cited_by") or 0),
        }

    # -- FTS-stem aliases (BM25-channel source domain → graph key) ----------
    stem_map = defaultdict(list)
    for s in nodes:
        stem_map[_sanitize(s)].append(s)
    aliases = {st: v[0] for st, v in stem_map.items() if len(v) == 1}
    dropped = len(stem_map) - len(aliases)
    log(f"aliases: {len(aliases)} stems resolvable, {dropped} ambiguous dropped")

    max_cc = max((m["citation_count"] for m in metrics.values()), default=0)
    out = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "graph_source": f"{graph_path} (n={len(nodes)}, e={len(edges)}, read-only)",
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "n_communities": len(comms),
        "modularity": round(modularity, 4),
        "resolution": 1.0,
        "louvain_seed": 7,
        "hits_mode": hits_mode,
        "max_pagerank": max(pr.values(), default=0.0),
        "max_authority": max(auth.values(), default=0.0),
        "max_hub": max(hub.values(), default=0.0),
        "max_betweenness": max(bt.values(), default=0.0),
        "max_citation_count": max_cc,
        "max_log1p_citation": math.log1p(max_cc) if max_cc > 0 else 0.0,
        "metrics": metrics,
        "aliases": aliases,
    }
    out_path = os.path.join(cfg["data_dir"], "graph_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    log(f"graph_metrics.json → {out_path} "
        f"({os.path.getsize(out_path)/1e6:.1f} MB)")

    # -- distribution stats ---------------------------------------------------
    print("\n─── metric distributions ───")
    print(_dist_line("pagerank", pr.values()))
    print(_dist_line("authority", auth.values()))
    print(_dist_line("hub", hub.values()))
    print(_dist_line("betweenness", bt.values()))
    print(_dist_line("citation_count", (m["citation_count"] for m in metrics.values())))
    sizes = sorted((len(c) for c in comms), reverse=True)
    singles = sum(1 for s in sizes if s == 1)
    print(f"  communities    n={len(comms)} modularity={modularity:.4f} "
          f"largest={sizes[:10]} singletons={singles}")

    print("\n─── top-10 by pagerank ───")
    for s, v in sorted(pr.items(), key=lambda kv: -kv[1])[:10]:
        nd = nodes[s]
        print(f"  {v:.6f}  {s}  [{nd.get('year', '')}] "
              f"{(nd.get('title') or '')[:66]}  (cited {metrics[s]['citation_count']}, "
              f"comm {comm_of[s]})")
    print("─── top-10 by authority ───")
    for s, v in sorted(auth.items(), key=lambda kv: -kv[1])[:10]:
        nd = nodes[s]
        print(f"  {v:.6f}  {s}  [{nd.get('year', '')}] "
              f"{(nd.get('title') or '')[:66]}  (in-corpus cited_by "
              f"{metrics[s]['in_corpus_cited_by']}, comm {comm_of[s]})")
    print()

    if args.no_chroma:
        log("--no-chroma: skipping metadata upsert")
        log("GRAPH-METRICS-COMPLETE (metrics only)")
        return

    # -- Phase D: chroma metadata upsert (build_citation_graph_full pattern) --
    import chromadb
    col = chromadb.PersistentClient(path=cfg["chroma_path"]).get_collection(
        cfg["collection_name"])
    total = col.count()
    log(f"chroma upsert: {total} chunks @ {cfg['chroma_path']} "
        f"(collection '{cfg['collection_name']}')")
    page, offset = 5000, 0
    updated = scanned = 0
    seen_sources = set()
    t0 = time.time()
    while True:
        r = col.get(limit=page, offset=offset, include=["metadatas"])
        ids = r["ids"]
        if not ids:
            break
        upd_ids, upd_metas = [], []
        for cid_, m in zip(ids, r["metadatas"] or []):
            m = m or {}
            rec = metrics.get(m.get("source", ""))
            if rec:
                nm = dict(m)
                nm["pagerank"] = rec["pagerank"]
                nm["authority"] = rec["authority"]
                nm["hub"] = rec["hub"]
                nm["betweenness"] = rec["betweenness"]
                nm["community_id"] = rec["community_id"]
                nm["community_size"] = rec["community_size"]
                upd_ids.append(cid_)
                upd_metas.append(nm)
                seen_sources.add(m.get("source", ""))
        if upd_ids:
            col.update(ids=upd_ids, metadatas=upd_metas)
            updated += len(upd_ids)
        scanned += len(ids)
        offset += page
        if (offset // page) % 20 == 0:
            log(f"  {scanned}/{total} scanned, {updated} updated "
                f"({time.time()-t0:.0f}s)")
    log(f"Phase D DONE: {updated}/{scanned} chunks updated across "
        f"{len(seen_sources)}/{len(nodes)} sources ({time.time()-t0:.0f}s)")
    log("GRAPH-METRICS-COMPLETE")


if __name__ == "__main__":
    main()