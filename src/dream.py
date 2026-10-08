#!/usr/bin/env python3
"""
dream.py — Phase 1 of autonomous research: deterministic graph divergence.

The "dreaming" step. Given a scientific question, produce candidate research
directions from the SHAPE of the citation graph — NOT from an LLM. This is
the reproducibility anchor: pure graph algorithms over a frozen snapshot,
fixed seed, no randomness, no model call. Same inputs → byte-identical
candidates.

Three divergence sources (each returns directions that point back to
concrete graph structure, so every direction is auditable):

  D1  bridge    — high-betweenness nodes that connect two otherwise distant
                  communities → "method/mechanism from field A may apply to
                  field B because hub paper Z links both".
  D2  uncited   — pairs sharing topic/community but with NO citation edge
                  between them (directly or via 1 hop) → "two papers doing
                  the same thing that don't know about each other".
  D3  orphan    — high-RCR / high-citation nodes with LOW in-corpus degree
                  (authoritative but uncited inside the library) → "an
                  influential direction the corpus has neglected".

Design borrowed from: idea-generation (diverse candidates, seed scoring),
capability-evolver (deterministic, no-hallucination core), novelty-assessment
(harsh-critic is Phase 3, not here — Phase 1 only proposes).

Contract:
    diverge(question, metrics, graph, n_each=..) -> List[Direction]
      Direction = {
        "id": "D1-<key>", "source": "bridge"|"uncited"|"orphan",
        "anchors": [source, ...],        # graph nodes the direction rests on
        "signal": {...},                 # the raw graph numbers (auditable)
        "topic": str,                    # shared topic/keyword, may be ""
        "note": str,                     # human-readable one-liner
      }
    Pure: no I/O beyond what the caller passes in. Never raises on missing
    metrics/graph (returns [] for an unusable source).

The question is used only to FOCUS the output (keyword overlap with topic
tags / titles); the graph structure is the generator. Pass "" to keep all
candidates (unfocused dream).
"""
from __future__ import annotations

import hashlib
import os
import re
from typing import Dict, List, Optional


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# ─── helpers ────────────────────────────────────────────────────────────────
_WORD = re.compile(r"[a-z0-9][a-z0-9\-]{2,}")


def _tokens(text: str) -> set:
    """Lowercased content tokens (len>=3). Good enough for focus overlap;
    the heavy matching is the corpus's job, this only ranks candidates."""
    return set(_WORD.findall((text or "").lower()))


def _q_overlap(question_tokens: set, *texts: str) -> float:
    """Fraction of the question's tokens present in the given texts.
    Returns 0.0 when the question is empty (no focus → everything ties)."""
    if not question_tokens:
        return 0.0
    bag = set()
    for t in texts:
        bag |= _tokens(t)
    if not bag:
        return 0.0
    hit = len(question_tokens & bag)
    return hit / len(question_tokens)


def _node_title(meta: Dict) -> str:
    return (meta or {}).get("title") or ""


def _graph_titles(graph: Optional[Dict]) -> Dict[str, str]:
    """source -> title, from the citation graph's node records.

    graph_metrics.json (source for centralities/community) carries NO title,
    so token overlap would be computed against empty strings and EVERY
    focused candidate would be filtered out (observed 2026-10-09: 0
    candidates on eph_rag). Titles live in citation_graph.json['nodes'].
    """
    out: Dict[str, str] = {}
    if not graph:
        return out
    for k, v in (graph.get("nodes") or {}).items():
        t = (v or {}).get("title")
        if t:
            out[str(k)] = t
    return out


def _stable_id(prefix: str, *parts: str) -> str:
    """Deterministic short id — same parts → same id, across runs."""
    h = hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:10]
    return f"{prefix}-{h}"


def graph_fingerprint(metrics: Dict, graph: Optional[Dict] = None) -> str:
    """Stable hash of the graph SNAPSHOT (node set + community partition).

    Part of the run_id: if the graph is rebuilt (new edges/nodes), every old
    run_id is invalidated automatically — a run is only reproducible against
    the exact snapshot it was created from. We hash the node→community map
    and the node count, not the full edge list (cheap, and community drift
    is the signal that matters for divergence).
    """
    parts = []
    if metrics:
        parts.append(f"n={len(metrics)}")
        # community partition: sort for determinism
        comm = sorted((str(k), str((v or {}).get("community_id")))
                      for k, v in metrics.items())
        parts.append(hashlib.sha1(
            "|".join(f"{a}:{b}" for a, b in comm).encode()).hexdigest())
    if graph:
        parts.append(f"e={len(graph.get('edges') or [])}")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]


# ─── D1: cross-community bridges ──────────────────────────────────────────────
def _d1_bridges(metrics: Dict, question_tokens: set, n_each: int,
                titles: Optional[Dict[str, str]] = None,
                focus: str = "rank") -> List[Dict]:
    """High-betweenness nodes whose neighbors span >=2 communities.

    Betweenness high = structural broker. We additionally require that the
    broker actually touches multiple communities (from graph_metrics
    community_id) so a high-betweenness node WITHIN one community (a dense
    hub) doesn't masquerade as a bridge.
    """
    titles = titles or {}
    # community map + degree
    comm: Dict[str, int] = {}
    for k, v in (metrics or {}).items():
        cid = (v or {}).get("community_id")
        if cid is not None:
            comm[str(k)] = cid
    if len(comm) < 2:
        return []

    btw = sorted(
        ((str(k), float((v or {}).get("betweenness") or 0.0), v)
         for k, v in (metrics or {}).items()),
        key=lambda x: -x[1])

    out: List[Dict] = []
    seen_comm_pairs = set()
    for src, b, v in btw:
        if b <= 0:
            break
        cid = comm.get(src)
        if cid is None:
            continue
        # candidate bridge anchor; we record its own community + the metric.
        topic = (v or {}).get("topic") or ""
        title = _node_title(v) or titles.get(src, "")
        ov = _q_overlap(question_tokens, title, topic)
        if focus == "filter" and question_tokens and ov == 0.0:
            continue  # strict mode: drop non-matching
        key = (cid, round(b, 6))
        if key in seen_comm_pairs:
            continue
        seen_comm_pairs.add(key)
        out.append({
            "id": _stable_id("D1", src, str(cid)),
            "source": "bridge",
            "anchors": [src],
            "signal": {"betweenness": round(b, 8), "community_id": cid,
                       "pagerank": round(float((v or {}).get("pagerank") or 0.0), 8),
                       "rcr": (v or {}).get("rcr")},
            "topic": topic,
            "note": (f"structural bridge (betweenness {b:.2e}) in community "
                     f"{cid}; a broker connecting subfields — its mechanism "
                     f"may transfer across them"),
        })
        if len(out) >= n_each:
            break
    return out


# ─── D2: same-topic, never-cited pairs ────────────────────────────────────────
def _adjacency(graph: Optional[Dict]) -> Dict[str, set]:
    adj: Dict[str, set] = {}
    if not graph:
        return adj
    for a, b in (graph.get("edges") or []):
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    return adj


def _undirected_reachable_within(adj: Dict[str, set], a: str, b: str,
                                 hops: int = 1) -> bool:
    """True if a and b are connected within `hops` undirected steps. hops=1
    means directly adjacent (cites or cited-by)."""
    if a == b:
        return True
    frontier = {a}
    seen = {a}
    for _ in range(hops):
        nxt = set()
        for x in frontier:
            nxt |= adj.get(x, set())
        nxt -= seen
        if b in nxt:
            return True
        seen |= nxt
        frontier = nxt
        if not frontier:
            break
    return False


def _d2_uncited(metrics: Dict, graph: Optional[Dict], question_tokens: set,
                n_each: int, titles: Optional[Dict[str, str]] = None,
                focus: str = "rank") -> List[Dict]:
    """Pairs in the SAME community that share a topic but are not connected
    within 1 hop — candidate "parallel work that doesn't cite each other".

    Deterministic pair enumeration: walk nodes sorted by (community, source),
    compare each node only to LATER nodes in the same community, stop at
    n_each pairs. O(n^2) within a community is bounded by community size;
    we cap the scan to keep it fast on big communities.
    """
    if not graph or not metrics:
        return []
    titles = titles or {}
    adj = _adjacency(graph)
    if not adj:
        return []

    # group by community. Include edge-less nodes too: a node with NO edges
    # is the strongest "uncited" case (nothing in the corpus links it), so it
    # must not be dropped — only nodes missing a community label are skipped.
    by_comm: Dict[int, List[str]] = {}
    for k, v in metrics.items():
        k = str(k)
        cid = (v or {}).get("community_id")
        if cid is None:
            continue
        by_comm.setdefault(cid, []).append(k)

    scan_cap = _env_int("DREAM_D2_SCAN", 4000)
    out: List[Dict] = []
    for cid in sorted(by_comm):
        nodes = sorted(by_comm[cid])
        scanned = 0
        for i in range(len(nodes)):
            a = nodes[i]
            va = metrics.get(a) or {}
            ta = _tokens(_node_title(va) or titles.get(a, "")) \
                | _tokens((va.get("topic") or ""))
            if not ta:
                continue
            for j in range(i + 1, len(nodes)):
                scanned += 1
                if scanned > scan_cap:
                    break
                b = nodes[j]
                if _undirected_reachable_within(adj, a, b, hops=1):
                    continue  # already cite each other (or cited-by) — not a blind spot
                vb = metrics.get(b) or {}
                tb = _tokens(_node_title(vb) or titles.get(b, "")) \
                    | _tokens((vb.get("topic") or ""))
                shared = ta & tb
                if len(shared) < 2:
                    continue  # needs a real topical overlap
                title_a = _node_title(va) or titles.get(a, "")
                title_b = _node_title(vb) or titles.get(b, "")
                ov = _q_overlap(question_tokens, title_a, title_b,
                                va.get("topic") or "", vb.get("topic") or "")
                if focus == "filter" and question_tokens and ov == 0.0:
                    continue
                out.append({
                    "id": _stable_id("D2", a, b),
                    "source": "uncited",
                    "anchors": [a, b],
                    "signal": {"community_id": cid,
                               "shared_tokens": sorted(shared)[:8],
                               "rcr_a": va.get("rcr"), "rcr_b": vb.get("rcr")},
                    "topic": ", ".join(sorted(shared)[:4]),
                    "note": (f"same community {cid}, {len(shared)} shared "
                             f"terms, NO citation edge within 1 hop — "
                             f"parallel work that may not know of each other"),
                })
                if len(out) >= n_each:
                    return out
            if scanned > scan_cap:
                break
    return out


# ─── D3: authoritative orphans ────────────────────────────────────────────────
def _d3_orphans(metrics: Dict, question_tokens: set, n_each: int,
                titles: Optional[Dict[str, str]] = None,
                focus: str = "rank") -> List[Dict]:
    """High external impact (RCR or citation_count) but LOW in-corpus
    citations → an influential work the library has not built on.

    Ranked by (rcr normalized) x (1 - in_corpus_cited_by normalized), so a
    famous-but-neglected node floats up. Deterministic sort.
    """
    if not metrics:
        return []
    titles = titles or {}
    rcr_max = max((float((v or {}).get("rcr") or 0.0) for v in metrics.values()),
                  default=0.0) or 1.0
    icc_max = max((int((v or {}).get("in_corpus_cited_by") or 0)
                   for v in metrics.values()), default=0) or 1

    scored = []
    for k, v in metrics.items():
        v = v or {}
        rcr = float(v.get("rcr") or 0.0)
        icc = int(v.get("in_corpus_cited_by") or 0)
        # need a real external signal
        if rcr <= 0 and not (v.get("citation_count") or 0):
            continue
        neglect = 1.0 - (icc / icc_max)
        score = (rcr / rcr_max) * neglect
        if score <= 0:
            continue
        scored.append((str(k), score, rcr, icc, v))
    scored.sort(key=lambda x: (-x[1], x[0]))

    out: List[Dict] = []
    for src, score, rcr, icc, v in scored:
        title = _node_title(v) or titles.get(src, "")
        topic = v.get("topic") or ""
        ov = _q_overlap(question_tokens, title, topic)
        if focus == "filter" and question_tokens and ov == 0.0:
            continue
        out.append({
            "id": _stable_id("D3", src),
            "source": "orphan",
            "anchors": [src],
            "signal": {"rcr": rcr, "in_corpus_cited_by": icc,
                       "citation_count": v.get("citation_count"),
                       "neglect_score": round(score, 6)},
            "topic": topic,
            "note": (f"high external impact (RCR {rcr:.2f}) but only "
                     f"{icc} in-corpus citations — an influential direction "
                     f"the library has under-built on"),
        })
        if len(out) >= n_each:
            break
    return out


# ─── public API ──────────────────────────────────────────────────────────────
def diverge(question: str, metrics: Dict, graph: Optional[Dict] = None,
            n_each: Optional[int] = None,
            focus: str = "rank",
            seeds: Optional[List[str]] = None,
            anchor: bool = True) -> List[Dict]:
    """Run all three divergence sources; return a ranked, de-duplicated list.

    Order: D1 (bridges) then D2 (uncited pairs) then D3 (orphans) — bridges
    are usually the highest-value structural signal, orphans the most
    surprising. Within each source the ordering is deterministic.

    `question` focuses the candidates. focus mode:
      "rank"  (default) — compute overlap, sort by (overlap, native score),
              but KEEP all candidates. The question boosts rather than
              filters: a gastrulation bridge can still bear on an Eph
              question through its mechanism, so hard-filtering on raw title
              keywords throws away the structural signal (observed
              2026-10-09: 0 candidates on a real eph_rag query).
      "filter" — drop candidates with zero overlap (strict; only when the
              caller wants a tightly on-topic dream).

    `seeds` / `anchor`: domain anchoring. When `anchor` is True and `seeds`
    is None, the seed set is retrieved from the question (seed_sources);
    candidates structurally near the seed (same paper or same Louvain
    community) are then ranked FIRST. This is what keeps a pure-structural
    dream from drifting into an unrelated subfield whose papers merely have
    high betweenness/RCR (observed 2026-10-09: "Gastrulation" and "Segment
    Anything" floated up on the unfocused run). Pass seeds=[] to disable.

    `question`="" → no ranking signal, native order (unfocused dream).
    Never raises: a missing metrics/graph yields [] for the affected source.
    """
    n_each = n_each if n_each is not None else _env_int("DREAM_N_EACH", 5)
    qt = _tokens(question)
    titles = _graph_titles(graph)

    # domain seed: retrieve only if the caller asked for anchoring and did
    # not supply a seed set. Cheap retrieval failure -> no anchoring.
    profile = None
    if anchor:
        if seeds is None:
            try:
                seeds = seed_sources(question)
            except Exception:
                seeds = []
        profile = anchor_profile(seeds or [], metrics or {}, graph)

    candidates: List[Dict] = []
    for fn in (
        lambda: _d1_bridges(metrics or {}, qt, n_each, titles, focus),
        lambda: _d2_uncited(metrics or {}, graph, qt, n_each, titles, focus),
        lambda: _d3_orphans(metrics or {}, qt, n_each, titles, focus),
    ):
        try:
            candidates.extend(fn())
        except Exception:
            continue  # one source failing never kills the dream
    # de-dup by id (a node could surface in two sources; keep the first)
    seen = set()
    out = []
    for c in candidates:
        if c["id"] in seen:
            continue
        seen.add(c["id"])
        out.append(c)
    # "rank" mode: sort by (anchor proximity, focus overlap, native order).
    # Anchor proximity leads because it is the stronger relevance signal:
    # a candidate in the question's own community is more likely on-topic
    # than one that merely shares a word in its title.
    if focus == "rank" and (qt or (profile and profile.get("sources"))):
        indexed = []
        for i, c in enumerate(out):
            ov = _q_overlap(qt, c.get("note", ""), c.get("topic", ""),
                            " ".join(titles.get(a, "") for a in c["anchors"]))
            anc = _anchor_score(c, profile, metrics or {})
            indexed.append((anc, ov, i, c))
        indexed.sort(key=lambda t: (-t[0], -t[1], t[2]))
        out = [_attach_scores(c, qt, titles, anc)
               for anc, _, _, c in indexed]
    return out


def _attach_scores(c: Dict, qt: set, titles: Dict[str, str], anc: float) -> Dict:
    """Add focus overlap + anchor proximity to a candidate's signal."""
    ov = _q_overlap(qt, c.get("note", ""), c.get("topic", ""),
                    " ".join(titles.get(a, "") for a in c["anchors"]))
    c = dict(c)
    c["signal"] = {**(c.get("signal") or {}),
                   "focus_overlap": round(ov, 3),
                   "anchor_score": round(anc, 3)}
    return c


# ─── domain anchoring ─────────────────────────────────────────────────────────
def seed_sources(question: str, top_k: int = 12) -> List[str]:
    """Retrieve the papers the question actually lands on — the domain seed.

    Uses the shared CLI retrieval path (query_bib_rag.search, which runs the
    full hybrid+rerank pipeline) and returns the distinct SOURCE filenames.
    Import is lazy and failure is soft: when retrieval is unavailable we
    return [] and diverge() falls back to pure structural ranking.
    """
    if not question:
        return []
    try:
        try:  # bib_rag-package-try
            from .query_bib_rag import search as _search
        except ImportError:  # flat (loose-script mode)
            from query_bib_rag import search as _search
        results = _search(question, top_k=top_k, expand=False, distill=False)
    except Exception:
        return []
    seen, out = set(), []
    for r in results or []:
        s = (r.get("metadata") or {}).get("source") or ""
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def anchor_profile(seeds: List[str], metrics: Dict,
                   graph: Optional[Dict] = None) -> Dict:
    """Summarize the domain seed's structural footprint.

    Returns {"seeds": set, "communities": {cid: weight}, "sources": set}.
    The community weights are the fraction of seeds in each community — so a
    candidate's structural proximity to the question can be scored.
    """
    seeds = [s for s in (seeds or []) if s]
    comm: Dict[int, int] = {}
    for s in seeds:
        m = metrics.get(s) or {}
        cid = m.get("community_id")
        if cid is not None:
            comm[cid] = comm.get(cid, 0) + 1
    total = sum(comm.values()) or 1
    weights = {cid: n / total for cid, n in comm.items()}
    return {"seeds": set(seeds), "communities": weights, "sources": set(seeds)}


def _anchor_score(c: Dict, profile: Optional[Dict], metrics: Dict) -> float:
    """Structural proximity of a candidate to the domain seed.

    Signals, strongest first:
      - a candidate anchor is itself a seed paper        -> 1.0
      - a candidate anchor sits in a seeded community    -> 0.5..1.0 (by
        how dominant that community is in the seed set)
      - otherwise                                         -> 0.0
    """
    if not profile or not profile.get("sources"):
        return 0.0
    seeds = profile["sources"]
    weights = profile.get("communities") or {}
    best = 0.0
    for a in c.get("anchors", []):
        if a in seeds:
            return 1.0
        cid = (metrics.get(a) or {}).get("community_id")
        if cid is not None and cid in weights:
            best = max(best, 0.5 + 0.5 * weights[cid])
    return best


# ─── CLI (manual inspection) ─────────────────────────────────────────────────
def _cli() -> int:
    import json
    import sys
    try:
        from .kb_config import get_config
    except ImportError:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from kb_config import get_config
    question = sys.argv[1] if len(sys.argv) > 1 else ""
    cfg = get_config()
    mpath = os.path.join(cfg["data_dir"], "graph_metrics.json")
    gpath = cfg.get("reference_graph_path") or os.path.join(
        cfg["data_dir"], "reference_graph.json")
    with open(mpath, encoding="utf-8") as f:
        metrics = (json.load(f).get("metrics")
                   or json.load(open(mpath, encoding="utf-8")).get("nodes") or {})
    graph = None
    if os.path.exists(gpath):
        with open(gpath, encoding="utf-8") as f:
            graph = json.load(f)
    print(f"fingerprint: {graph_fingerprint(metrics, graph)}")
    cands = diverge(question, metrics, graph)
    for c in cands:
        print(f"\n[{c['source']}] {c['id']}")
        print(f"  anchors: {', '.join(c['anchors'])}")
        print(f"  {c['note']}")
    print(f"\n{len(cands)} candidates")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
