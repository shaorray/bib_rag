#!/usr/bin/env python3
"""
related_papers.py — "I just read X, what should I read next?"

Scores every library paper against a query paper using three signals and
returns the top-k with a human-readable reason per hit:

  topic overlap   Jaccard over canonical topic keywords (chroma `topics`)
  embedding sim   cosine over parent title+lead embeddings (bge-m3), cached
                  in <data_root>/data/paper_embeddings.npz
  graph signal    in-library citation links + bibliographic coupling from
                  reference_graph.json (author-year edges)

CLI:
    <name>-rag scripts/related_papers.py 25925582.md --k 8
    <name>-rag scripts/related_papers.py "E-cadherin junctions as active..." --k 5

Agent tool: BibRagTools.find_related_papers (agent_tools.py).
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

try:
    from .kb_config import get_config
except ImportError:  # direct script execution
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from kb_config import get_config

try:
    from .reference_graph import load_graph, _resolve_source, biblio_coupling
except ImportError:
    from reference_graph import load_graph, _resolve_source, biblio_coupling

_CFG = get_config()

W_TOPIC, W_EMB, W_GRAPH = 0.40, 0.35, 0.25


# ---------------------------------------------------------------------------
# paper-level data collection (one chroma scroll + one parent_store read each)
# ---------------------------------------------------------------------------

def _chroma_stamp() -> str:
    """Cheap change-detection stamp: sqlite size only. The file's mtime bumps
    on every chroma client open/close (WAL checkpoint), so it cannot be part
    of the stamp — size changes only on real writes."""
    return str(os.stat(_CFG["chroma_sqlite"]).st_size)


def collect_papers(col, use_cache: bool = True) -> Dict[str, dict]:
    """source -> {title, year, topics, article_type}.
    Cached at <data_root>/data/papers_meta.json; cache invalidated when the
    chroma sqlite file changes (the 481k-chunk scroll costs ~2 min cold)."""
    cache = Path(_CFG["data_root"]) / "data" / "papers_meta.json"
    stamp = _chroma_stamp()
    if use_cache and cache.exists():
        try:
            d = json.loads(cache.read_text())
            if d.get("chroma_stamp") == stamp:
                return d["papers"]
        except Exception:
            pass
    if col is None:
        import chromadb
        col = chromadb.PersistentClient(path=_CFG["chroma_path"]).get_collection(
            _CFG["collection_name"])
    papers = {}
    offset, page = 0, 5000
    while True:
        r = col.get(include=["metadatas"], limit=page, offset=offset)
        ids, metas = r["ids"] or [], r["metadatas"] or []
        if not ids:
            break
        for m in metas:
            s = m.get("source", "")
            if not s or s in papers:
                continue
            try:
                tps = json.loads(m.get("topics") or "[]")
            except Exception:
                tps = []
            papers[s] = {"title": m.get("title", "") or "",
                         "year": str(m.get("year", "") or ""),
                         "topics": tps,
                         "article_type": m.get("article_type", "") or ""}
        offset += page
    if use_cache and papers:
        cache.write_text(json.dumps({"chroma_stamp": stamp, "papers": papers}))
    return papers


def _lead_text(content: str, max_words: int = 600) -> str:
    w = (content or "").split()
    return " ".join(w[:max_words])


def _parent_key(source: str) -> str:
    return source[:-3] + "_md" if source.endswith(".md") else source + "_md"


def parent_text(source: str) -> str:
    store = _CFG["parent_store_dir"]
    stem = source[:-3] if source.endswith(".md") else source
    # parent_store naming: raw filename with .md->_md, else <stem>_md
    for cand in (f"{stem}_md.json", f"{source[:-3] if source.endswith('.md') else source}.json"):
        p = os.path.join(store, cand)
        if os.path.exists(p):
            try:
                d = json.load(open(p))
                if isinstance(d, list) and d:
                    return _lead_text(d[0].get("content", ""))
                if isinstance(d, dict):
                    return _lead_text(d.get("content", ""))
            except Exception:
                pass
    return ""


def embed_texts(texts: Dict[str, str], cache_path: Optional[Path]) -> Dict[str, List[float]]:
    """Embed {key: text} with bge-m3; persist cache npz keyed by text hash."""
    import hashlib
    import numpy as np

    def h(t):
        return hashlib.md5(t.encode()).hexdigest()[:12]

    want = {k: h(v) for k, v in texts.items() if v}
    old = {}
    if cache_path and cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        saved = dict(zip(list(z["hashes"]), list(z["emb"])))
        old = {k: saved[hash_] for k, hash_ in want.items() if hash_ in saved}
    todo = {k: v for k, v in texts.items() if v and k not in old}
    if todo:
        import requests
        keys = list(todo)
        vecs = []
        for i in range(0, len(keys), 256):
            batch = [texts[k] for k in keys[i:i + 256]]
            r = requests.post(_CFG["embed_url"],
                              json={"input": batch, "model": "bge-m3"}, timeout=300)
            vecs.extend(d["embedding"] for d in r.json()["data"])
        for k, v in zip(keys, vecs):
            old[k] = v
        if cache_path:
            hashes = []
            embs = []
            seen = set()
            for k, hash_ in want.items():
                if hash_ in seen or k not in old:
                    continue
                seen.add(hash_)
                hashes.append(hash_)
                embs.append(old[k])
            np.savez(cache_path, hashes=hashes, emb=np.array(embs, dtype=np.float32))
    return old


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def score_related(query_src: str, papers: Dict[str, dict],
                  graph: Optional[Dict]) -> List[dict]:
    """Full scoring sweep for one query paper. Returns ranked candidates."""
    if query_src not in papers:
        return []
    q = papers[query_src]
    q_set = set(q["topics"])

    # graph paper ids are '<stem>_md'; chroma sources are '<stem>.md'.
    # Build the mapping once so graph hits join the papers dict.
    def gkey_to_src(k: str) -> Optional[str]:
        if k in papers:
            return k
        if k.endswith("_md") and k[:-3] + ".md" in papers:
            return k[:-3] + ".md"
        return None

    # graph channel: forward snowball + biblio coupling (author-year based)
    fw, coupling = {}, {}
    if graph:
        r = snowball_quiet(graph, query_src, limit=300)
        for m in r:
            s = gkey_to_src(m["source"])
            if s:
                fw[s] = 1.0
        for c in biblio_coupling(graph, query_src, limit=300):
            s = gkey_to_src(c["source"])
            if s:
                coupling[s] = c["coupling"]

    # embedding channel: compare query lead text against candidate leads
    texts = {query_src: parent_text(query_src)}
    for s in papers:
        if s != query_src:
            texts[s] = parent_text(s)
    emb = embed_texts(texts, Path(_CFG["data_root"]) / "data" / "paper_embeddings.npz")
    qv = _norm(emb.get(query_src))
    cands: List[dict] = []
    for s, p in papers.items():
        if s == query_src:
            continue
        # topic jaccard
        tps = set(p["topics"])
        jac = len(q_set & tps) / len(q_set | tps) if (q_set or tps) else 0.0
        # embedding cos
        ev = _norm(emb.get(s))
        cos = float(qv @ ev) if (qv is not None and ev is not None) else 0.0
        gsig = max(fw.get(s, 0.0), 0.6 * coupling.get(s, 0.0))
        score = W_TOPIC * jac + W_EMB * cos + W_GRAPH * gsig
        if score <= 0.05:
            continue
        reasons = []
        shared = q_set & tps
        if shared:
            reasons.append("shares " + ", ".join(sorted(shared)[:3]))
        if s in fw:
            reasons.append("cites it" if _cites(graph, s, query_src)
                           else "same citation neighbourhood")
        if coupling.get(s, 0) >= 0.15:
            reasons.append(f"bibliographic coupling {coupling[s]:.2f}")
        cands.append({"source": s, "title": p["title"], "year": p["year"],
                      "article_type": p["article_type"],
                      "topics": p["topics"],
                      "topic_j": round(jac, 3), "emb_cos": round(cos, 3),
                      "graph": round(gsig, 3),
                      "score": round(score, 4),
                      "why": "; ".join(reasons) or "embedding similarity"})
    return cands


def _norm(v):
    import numpy as np
    if v is None:
        return None
    a = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(a)
    return a / n if n else None


def _cites(graph, src, target) -> bool:
    """Does `src` cite `target`? Match edge (author, year) against the
    TARGET paper's real authors/year — never the citing paper's own name."""
    tm = graph["papers"].get(target, {})
    t_surnames = {a.split()[0].split(",")[0].lower()
                  for a in (tm.get("authors", "") or "").split(";") if a.strip()}
    t_year = str(tm.get("year", "") or "")
    for e in graph.get("edges", []):
        if e["from"] != src:
            continue
        if str(e.get("to_year", "") or "") != t_year:
            continue
        if any(a and e.get("to_author", "").lower().startswith(a[:4])
               for a in t_surnames):
            return True
    return False


def snowball_quiet(graph, source, limit=300):
    from reference_graph import snowball
    r = snowball(graph, source, "forward", limit=limit)
    return r.get("matches", [])


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------

def find_related(query: str, k: int = 8) -> dict:
    """Agent/CLI entry. query = source name, PMID-stem, or title fragment."""
    # cache-first: skip the 2-min chroma scroll entirely on a valid stamp.
    # force_cache_load: import chromadb at all costs only when cache is stale.
    papers = collect_papers(None)
    if not papers:
        import chromadb
        col = chromadb.PersistentClient(path=_CFG["chroma_path"]).get_collection(
            _CFG["collection_name"])
        papers = collect_papers(col, use_cache=False)
    graph = load_graph()

    qsrc = _resolve(query, papers, graph)
    if qsrc is None:
        return {"error": f"paper not found in library: {query}", "matches": []}

    t0 = time.time()
    cands = score_related(qsrc, papers, graph)
    cands.sort(key=lambda d: -d["score"])
    return {"query": qsrc, "title": papers[qsrc]["title"],
            "matches": cands[:k]}


def _resolve(query: str, papers: Dict[str, dict], graph: Optional[Dict]):
    """source | PMID-stem | title fragment -> source key."""
    if query in papers:
        return query
    stem = query[:-3] if query.endswith(".md") else query
    for cand in (f"{stem}.md", stem):
        if cand in papers:
            return cand
    if graph:
        r = _resolve_source(graph, query)
        if r and (r in papers or f"{r.replace('_md','')}.md" in papers):
            return r if r in papers else f"{r.replace('_md', '')}.md"
    ql = query.lower()
    hits = [s for s, p in papers.items() if ql and ql in p["title"].lower()]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:  # prefer the highest-chroma-visibility (first sorted)
        return sorted(hits)[0]
    return None


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Find papers related to one paper")
    ap.add_argument("query", help="source, PMID-stem, or title fragment")
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()
    t0 = time.time()
    out = find_related(args.query, args.k)
    if "error" in out:
        print(out["error"])
        sys.exit(1)
    print(f"query: {out['query']} — {out['title'][:80]}")
    for i, m in enumerate(out["matches"], 1):
        print(f"{i:2}. [{m['article_type'] or '?':12}] {m['score']:.3f} "
              f"{m['source'][:44]:44} {m['year']}")
        print(f"      {m['title'][:90]}")
        print(f"      why: {m['why']}")
    print(f"({len(out['matches'])} results, {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()