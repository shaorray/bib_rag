#!/usr/bin/env python3
"""
reference_graph.py — Citation-graph extraction & snowballing for bib_rag.

Borrowed mechanism (Corvus citation snowballing, see
/Disk_bot/notes/citation_rag/02_Corvus.md): follow the reference lists of
papers to discover related work — a retrieval channel ORTHOGONAL to
embedding/BM25 similarity.

bib_rag truncates text at the REFERENCES header (chunking.truncate_at_references)
so the reference list was previously discarded. This module recovers a
lightweight citation graph WITHOUT re-running ingestion:

  - extract at query time is too slow → build a one-time graph file at
    <data_root>/data/reference_graph.json from the parent_store contents
    (which DO retain the truncation boundary... but many sources keep
    in-text citation tails). Two extraction sources, in priority order:
      1. parent content tails: anything after a REFERENCES-ish header that
         survived truncation (older builds) — parsed as numbered/author-year
         entries;
      2. in-text citations "(Author, 2020; Author2 & Author3, 2021)" across
         ALL parent content → author-year edges.

Graph schema (v1, file-based; SQLite if it outgrows):
  { "version": 1,
    "built_at": iso,
    "papers": { "<source>": {"title":…, "year":…, "doi":…} },
    "edges":  [ {"from": source, "to_raw": "raw ref string",
                 "to_author": "Parker", "to_year": "2021",
                 "to_title_hint": "…", "direction": "cited-by|cites"} ] }

Snowballing tools then answer:
  - who cites X / who does X cite (within the library)
  - "find papers similar to X via shared references" (bibliographic coupling)
"""
from __future__ import annotations

import os
import re
import sys
import json
from typing import Dict, List, Optional, Set, Tuple

try:
    from .kb_config import get_config
except ImportError:  # direct script execution (scripts/build_reference_graph.py)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from kb_config import get_config

try:
    from .identifiers import normalize_doi
except ImportError:  # src/ on sys.path directly (CLI/tests)
    from identifiers import normalize_doi

# ---------------------------------------------------------------------------
# Extraction regexes
# ---------------------------------------------------------------------------

# Header that starts a reference section (mirrors chunking.truncate_at_references)
_REFS_HEADER_RE = re.compile(
    r"^#*\s*(REFERENCES?|BIBLIOGRAPHY)\s*$", re.I | re.M)

# In-text citation: (Parker, 2021) / (Parker & Chen, 2021) / (Parker et al., 2021; Wu, 2019)
_INTEXT_RE = re.compile(
    r"\(([A-Z][A-Za-z'\-]+(?:\s*(?:&|and)\s*[A-Z][A-Za-z'\-]+)?(?:\s+et\s+al\.)?),?\s+"
    r"((?:19|20)\d{2})[a-z]?\)")

# Reference-entry leading pattern: "Parker, J. (2021). Title..." or "1. Parker..."
_REF_ENTRY_RE = re.compile(
    r"([A-Z][A-Za-z'\-]+),?\s+[A-Z]\.?.{0,40}?\(((?:19|20)\d{2})[a-z]?\)\s*\.?\s*([^.]{15,180})\.")

_GRAPH_VERSION = 1


def _iter_parent_store():
    cfg = get_config()
    store = cfg["parent_store_dir"]
    if not os.path.isdir(store):
        return
    for fname in sorted(os.listdir(store)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(store, fname)
        try:
            with open(path, encoding="utf-8") as f:
                parents = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(parents, list) or not parents:
            continue
        meta = parents[0].get("meta", {})
        yield fname[:-5], parents, {
            "title": meta.get("title", ""),
            "year": meta.get("year", ""),
            "doi": meta.get("doi", ""),
            "authors": meta.get("authors", ""),
        }


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_reference_graph(out_path: Optional[str] = None,
                          progress: bool = False) -> Dict:
    """Scan the whole parent store, extract citation edges, write graph JSON.

    Two edge sources:
      A. in-text author-year citations in body sections → direction "cites"
      B. surviving reference-list entries → direction "cites" (stronger:
         includes a title hint)
    Returns the graph dict.
    """
    cfg = get_config()
    if out_path is None:
        out_path = cfg.get("reference_graph_path") or os.path.join(
            cfg["data_dir"], "reference_graph.json")

    papers: Dict[str, dict] = {}
    edges: List[dict] = []
    seen_edges: Set[Tuple[str, str, str]] = set()

    for source, parents, meta in _iter_parent_store():
        papers[source] = meta

        # A. in-text citations (cheap, all sections)
        full_text = "\n\n".join(p.get("content", "") for p in parents)
        # Remove any surviving reference list first to avoid double-counting
        m = _REFS_HEADER_RE.search(full_text)
        body = full_text[:m.start()] if m else full_text
        tail = full_text[m.end():] if m else ""

        for author, year in _INTEXT_RE.findall(body):
            key = (source, author, year)
            if key not in seen_edges:
                seen_edges.add(key)
                edges.append({"from": source, "to_raw": f"{author} ({year})",
                              "to_author": author, "to_year": year,
                              "to_title_hint": "", "direction": "cites"})

        # B. surviving reference entries (title hints!)
        for author, year, hint in _REF_ENTRY_RE.findall(tail):
            hint = hint.strip()
            key = (source, hint.lower()[:80], year)
            if key in seen_edges:
                continue
            seen_edges.add(key)
            edges.append({"from": source, "to_raw": f"{author} ({year}). {hint}",
                          "to_author": author, "to_year": year,
                          "to_title_hint": hint[:200], "direction": "cites"})

        if progress and len(papers) % 200 == 0:
            print(f"  [refgraph] {len(papers)} papers, {len(edges)} edges")

    graph = {"version": _GRAPH_VERSION,
             "built_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
             "papers": papers,
             "edges": edges}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False)
    os.replace(tmp, out_path)
    if progress:
        print(f"[refgraph] {len(papers)} papers, {len(edges)} edges → {out_path}")
    return graph


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

def load_graph() -> Optional[Dict]:
    cfg = get_config()
    path = cfg.get("reference_graph_path") or os.path.join(
        cfg["data_dir"], "reference_graph.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _resolve_source(graph: Dict, source_or_title: str) -> Optional[str]:
    """Accept a source filename, a title, a DOI/arXiv/PMID, or a fuzzy
    prefix; return the canonical source key.

    DOI matching uses identifiers.normalize_doi on BOTH sides, so
    "https://doi.org/10.1016/..." in the graph metadata matches a bare
    "10.1016/..." query (seerai identifier-normalization mechanism).
    """
    if source_or_title in graph["papers"]:
        return source_or_title
    q = source_or_title.lower().strip()
    # exact title match
    for src, meta in graph["papers"].items():
        if meta.get("title", "").lower() == q:
            return src
    # identifier match (DOI first — canonical, strongest signal)
    q_doi = normalize_doi(source_or_title)
    if q_doi:
        for src, meta in graph["papers"].items():
            m_doi = normalize_doi(meta.get("doi", ""))
            if m_doi and m_doi == q_doi:
                return src
    # containment match (title or filename)
    hits = [src for src, meta in graph["papers"].items()
            if q in src.lower() or (meta.get("title") and q in meta["title"].lower())]
    if len(hits) == 1:
        return hits[0]
    if hits:
        return hits[0]  # ambiguous → closest first, caller sees the count
    # filename convention bridge: chroma `source` = "x.md" but graph keys are
    # parent-store stems "x_md" (and vice versa). Agents pass what search
    # results show (Parent ID contains "x.md#..."), so try both variants.
    for variant in (q.replace(".md", "_md"), q.replace("_md", ".md"),
                    q.replace(".md", "_md.json").replace(".json", "")):
        if variant in graph["papers"]:
            return variant
    v_hits = [src for src in graph["papers"]
              if variant_ok(q, src)]
    if len(v_hits) == 1:
        return v_hits[0]
    return None


def variant_ok(q: str, src: str) -> bool:
    """Fuzzy containment tolerant of the .md <-> _md convention."""
    qn = q.lower().replace(".md", "_md").replace(".json", "")
    sn = src.lower().replace(".md", "_md").replace(".json", "")
    return qn in sn or sn in qn


def snowball(graph: Dict, source_or_title: str,
             direction: str = "forward", limit: int = 10) -> Dict:
    """Who cites this paper (forward) / whom this paper cites (backward)
    WITHIN the library. Returns {'query':…, 'direction':…, 'matches':[…]}.
    In-text edges carry only author-year — match resolution maps them to
    library papers by title hint (when present) or author+year."""
    src = _resolve_source(graph, source_or_title)
    if src is None:
        return {"error": f"source not found: {source_or_title}",
                "matches": []}

    papers = graph["papers"]
    by_source = {s: m for s, m in papers.items()}
    matches: List[dict] = []

    if direction == "forward":  # papers IN the library citing `src`
        qmeta = papers.get(src, {})
        title_tokens = _title_key(qmeta.get("title", ""))
        # first-author surnames of the QUERIED paper ("Lecuit T; Yap AS"
        # parent-store format) — in-text edges are matched against THESE,
        # never against the citing paper's own name (self-mention trap).
        q_surnames = {a.split()[0].split(",")[0].lower()
                      for a in (qmeta.get("authors", "") or "").split(";")
                      if a.strip()}
        for e in graph["edges"]:
            if e["from"] == src or (not e["to_author"] and not e.get("to_title_hint")):
                continue
            hit = None
            # strong: title hint overlaps target title
            if e.get("to_title_hint") and title_tokens:
                hint_tokens = _title_key(e["to_title_hint"])
                inter = len(hint_tokens & title_tokens)
                if title_tokens and inter >= max(2, len(title_tokens) // 3):
                    hit = e["from"]
            # weaker: edge (author, year) matches the QUERY paper's
            # first-author surname + year. Skip when the query paper has no
            # usable author metadata (filename-style titles, see meta_audit).
            q_year = str(qmeta.get("year", "") or "")
            if hit is None and q_surnames and e["to_year"] \
                    and (e["to_year"] == q_year or not q_year):
                for a in q_surnames:
                    if a and e["to_author"].lower().startswith(a[:4]):
                        hit = e["from"]
                        break
            if hit:
                matches.append({"source": hit,
                                "title": by_source.get(hit, {}).get("title", ""),
                                "year": by_source.get(hit, {}).get("year", ""),
                                "via": e.get("to_title_hint", "") or f"{e['to_author']} ({e['to_year']})"})
    else:  # backward: papers `src` cites that ARE in the library
        for e in graph["edges"]:
            if e["from"] != src:
                continue
            # try resolving the raw ref to a library paper
            target = None
            if e.get("to_title_hint"):
                t_tokens = _title_key(e["to_title_hint"])
                best, best_j = None, 0.0
                for s2, m2 in by_source.items():
                    if s2 == src:
                        continue
                    tt = _title_key(m2.get("title", ""))
                    if not tt:
                        continue
                    j = len(t_tokens & tt) / max(1, len(t_tokens | tt))
                    if j > best_j:
                        best, best_j = s2, j
                if best is not None and best_j >= 0.5:
                    target = best
            matches.append({
                "raw_ref": e.get("to_raw", ""),
                "resolved_source": target,
                "resolved_title": by_source.get(target, {}).get("title", "") if target else "",
                "in_library": target is not None,
            })

    # dedup + cap
    seen = set()
    deduped = []
    for m in matches:
        k = m.get("source") or m.get("raw_ref", "")
        if k and k not in seen:
            seen.add(k)
            deduped.append(m)
    return {"query": src, "direction": direction,
            "matches": deduped[:limit]}


def biblio_coupling(graph: Dict, source_or_title: str, limit: int = 8) -> List[dict]:
    """Papers sharing the most cited references with `src` (backward snowball
    similarity). Rank by overlap of author-year citation sets."""
    src = _resolve_source(graph, src_key := source_or_title)
    if src is None:
        return []
    mine = {(e["to_author"], e["to_year"]) for e in graph["edges"] if e["from"] == src}
    if not mine:
        return []
    scored = []
    for other in graph["papers"]:
        if other == src:
            continue
        theirs = {(e["to_author"], e["to_year"]) for e in graph["edges"] if e["from"] == other}
        if not theirs:
            continue
        j = len(mine & theirs) / len(mine | theirs)
        if j > 0:
            scored.append({"source": other,
                           "title": graph["papers"][other].get("title", ""),
                           "year": graph["papers"][other].get("year", ""),
                           "coupling": round(j, 3)})
    scored.sort(key=lambda d: -d["coupling"])
    return scored[:limit]


def _title_key(title: str) -> Set[str]:
    t = (title or "").lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    stop = {"a", "an", "the", "of", "in", "on", "and", "or", "for", "to", "by", "with"}
    return {w for w in t.split() if len(w) > 1 and w not in stop}