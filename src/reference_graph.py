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

Graph schema (v2, file-based; SQLite if it outgrows):
  { "version": 2,
    "built_at": iso,
    "papers": { "<source>": {"title":…, "year":…, "doi":…} },
    "edges":  [ { "from": source, "to_raw": "raw ref string",
                 "to_author": "Parker", "to_year": "2021",
                 "to_title_hint": "…", "direction": "cited-by|cites"},
                # v2: PubMed/iCite-verified edges (exact, no heuristics)
                { "from": citing_source, "to": cited_source,
                 "to_raw": "iCite", "direction": "cites", "resolved": true} ] }

v2 merge: when <data_root>/data/citation_graph.json exists (built by the
icite pipeline: build_citation_graph*.py / rebuild_graph_from_cache.py,
source-key space "12351647.md"), its [citing, cited] pairs are converted
to parent-store stem keys and injected as RESOLVED edges. The heuristic
extraction only covers author-year citation styles (60% of this corpus
uses numbered "[1,2]" styles whose reference lists are truncated away at
ingest) — the iCite edges are the ground-truth backstop.

Snowballing tools then answer:
  - who cites X / who does X cite (within the library)
  - "find papers similar to X via shared references" (bibliographic coupling)
"""
from __future__ import annotations

import os
import re
import sys
import json
import time
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

_GRAPH_VERSION = 2


def _icite_to_stem(source: str) -> str:
    """chroma source ("12351647.md") -> parent-store stem ("12351647_md").

    Mirrors chunking.save_parent_store: re.sub(r'[^\\w\\-]', '_', source)[:100].
    """
    return re.sub(r"[^\w\-]", "_", source)[:100]


def _merge_icite_edges(papers: Dict, edges: List[dict],
                       progress: bool = False) -> int:
    """Inject PubMed/iCite-verified citation edges as RESOLVED edges.

    Source of truth: <data_root>/data/citation_graph.json — source-key space
    ("12351647.md"), edges = [[citing, cited], ...]. That file is produced by
    the icite pipeline (build_citation_graph_full.py → rebuild_graph_from_
    cache.py) and is absent in libraries without PubMed corpora (geo_rag) —
    merge is a silent no-op there.
    """
    cfg = get_config()
    path = os.path.join(cfg["data_dir"], "citation_graph.json")
    if not os.path.exists(path):
        return 0
    try:
        with open(path, encoding="utf-8") as f:
            ic = json.load(f)
    except (OSError, json.JSONDecodeError):
        return 0
    ic_edges = ic.get("edges") or []
    if not isinstance(ic_edges, list):
        return 0

    # key conversion: ".md" chroma space -> parent-store stem space
    known = set(papers)  # stems collected from the parent store scan
    seen: Set[Tuple[str, str]] = set()
    injected = 0
    for e in ic_edges:
        if not (isinstance(e, (list, tuple)) and len(e) == 2):
            continue
        citing, cited = _icite_to_stem(str(e[0])), _icite_to_stem(str(e[1]))
        if citing == cited or citing not in known or cited not in known:
            continue
        if (citing, cited) in seen:
            continue
        seen.add((citing, cited))
        edges.append({"from": citing, "to": cited, "to_raw": "iCite",
                      "direction": "cites", "resolved": True})
        injected += 1
    if progress and injected:
        print(f"  [refgraph] icite merge: {injected} resolved edges injected")
    return injected


# ---------------------------------------------------------------------------
# iCite (PubMed) verified citation graph — generic fetch pipeline
# (distilled from eph_rag/scripts/build_citation_graph_full.py +
#  rebuild_graph_from_cache.py, 2026-08-31; corpus-agnostic version)
# ---------------------------------------------------------------------------

_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_ICITE_API = "https://icite.od.nih.gov/api/pubs"
_NCBI_THROTTLE = 0.35          # s between eutils calls (NCBI courtesy limit)
_UA = "bib_rag-icite-builder/1.0"


def _ncbi_get(url: str, retries: int = 3, timeout: int = 30,
              throttle: Optional[list] = None):
    """Throttled GET returning parsed JSON, or None after final retry."""
    import urllib.request
    _t = throttle if throttle is not None else [0.0]
    for attempt in range(retries):
        wait = _NCBI_THROTTLE - (time.time() - _t[0])
        if wait > 0:
            time.sleep(wait)
        _t[0] = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except Exception:
            if attempt == retries - 1:
                return None
            time.sleep(2 * (attempt + 1))
    return None


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", (t or "").lower()).strip()


def _title_sim(a: str, b: str) -> float:
    A, B = set(_norm_title(a).split()), set(_norm_title(b).split())
    if not A or not B:
        return 0.0
    return len(A & B) / max(len(A), len(B))


def _icite_corpus_registry() -> Dict[str, dict]:
    """All parent_store sources with meta (source key = chroma .md form).

    Key = the parent JSON's own `source` field (chroma metadata.source,
    e.g. "12351647.md"), NOT the parent-store filename stem — sanitization
    is lossy, the source field is authoritative. Filename-stem fallback
    only for legacy entries lacking the field."""
    registry = {}
    for stem, parents, meta in _iter_parent_store():
        src = (parents[0].get("source") or "").strip()
        if not src:
            src = f"{stem}.md"
        pm = str(meta.get("pmid") or "").strip()
        if not re.fullmatch(r"\d+", pm):
            # digit-filename convention: source "12351647.md" -> PMID
            base = src[:-3] if src.endswith(".md") else src
            if re.fullmatch(r"\d+", base):
                pm = base
        registry[src] = {
            "pmid": pm,
            "title": (meta.get("title") or "").strip(),
            "doi": (meta.get("doi") or "").strip(),
            "year": str(meta.get("year") or "").strip(),
            "journal": (meta.get("journal") or "").strip(),
        }
    return registry


def build_icite_graph(progress: bool = False,
                      resolve_limit: int = 0) -> Optional[Dict]:
    """Fetch PubMed/iCite-verified citation edges for the ACTIVE library.

    Phase A  resolve PMIDs for every parent_store source lacking one
             (esearch DOI-first, title+year fallback with 0.85 title-sim
             gate) — resumable cache data/icite_pmid_resolution.json
    Phase B  fetch iCite records for all corpus PMIDs in batches of 100 —
             resumable cache data/icite_corpus_icite.json
    Phase C  build the source-key graph (cross-expanding duplicate PMIDs),
             write data/citation_graph.json (+ .csv mirror), then rebuild
             reference_graph.json so _merge_icite_edges picks it up.

    resolve_limit>0 caps Phase A network calls (dry-run probing).
    Returns the written citation-graph dict, or None when the corpus has
    no resolvable PMIDs.
    """
    import urllib.parse
    cfg = get_config()
    data_dir = cfg["data_dir"]
    os.makedirs(data_dir, exist_ok=True)
    res_cache = os.path.join(data_dir, "icite_pmid_resolution.json")
    icite_cache = os.path.join(data_dir, "icite_corpus_icite.json")
    out_path = os.path.join(data_dir, "citation_graph.json")

    registry = _icite_corpus_registry()
    if progress:
        print(f"[icite] registry: {len(registry)} sources; "
              f"with PMID {sum(1 for v in registry.values() if v['pmid'])}")
    if not registry:
        return None

    def _log(msg):
        if progress:
            print(f"[icite] {msg}", flush=True)

    # ---- Phase A: PMID resolution (resumable) ----
    res = {}
    if os.path.exists(res_cache):
        try:
            res = json.load(open(res_cache, encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            res = {}
    for s, r in res.items():          # cache first, then meta fallback
        if s in registry and not registry[s]["pmid"] and r.get("pmid"):
            registry[s]["pmid"] = str(r["pmid"])
    todo = [s for s, v in registry.items() if not v["pmid"] and s not in res]
    if resolve_limit:
        todo = todo[:resolve_limit]
    _log(f"PMID resolution todo: {len(todo)} (cached: {len(res)})")
    throttle = [0.0]
    for i, s in enumerate(todo, 1):
        v = registry[s]
        pmid, method = "", ""
        if v["doi"]:
            term = f"{v['doi']}[DOI]"
            url = (f"{_EUTILS}/esearch.fcgi?db=pubmed&retmode=json&term="
                   + urllib.parse.quote(term))
            d = _ncbi_get(url, throttle=throttle)
            ids = (d or {}).get("esearchresult", {}).get("idlist", [])
            if len(ids) == 1:
                pmid, method = ids[0], "doi"
            elif len(ids) > 1:
                pmid, method = ids[0], "doi_multi"
        if not pmid and v["title"]:
            year = v["year"][:4] if re.match(r"(19|20)\d{2}", v["year"]) else ""
            term = f'"{v["title"]}"[Title]' + (f" AND {year}[pdat]" if year else "")
            url = (f"{_EUTILS}/esearch.fcgi?db=pubmed&retmode=json&term="
                   + urllib.parse.quote(term))
            d = _ncbi_get(url, throttle=throttle)
            ids = (d or {}).get("esearchresult", {}).get("idlist", [])[:3]
            if ids:
                su = (f"{_EUTILS}/esummary.fcgi?db=pubmed&retmode=json&id="
                      + ",".join(ids))
                d2 = _ncbi_get(su, throttle=throttle)
                best, best_sim = "", 0.0
                for uid in (d2 or {}).get("result", {}).get("uids", []):
                    p = d2["result"][uid]
                    sim = _title_sim(v["title"], p.get("title", ""))
                    if sim > best_sim:
                        best, best_sim = uid, sim
                if best and best_sim >= 0.85:
                    pmid, method = best, f"title:{best_sim:.2f}"
        res[s] = {"pmid": pmid, "method": method}
        if pmid:
            registry[s]["pmid"] = pmid
        if i % 100 == 0 or i == len(todo):
            with open(res_cache, "w", encoding="utf-8") as f:
                json.dump(res, f)
            _log(f"resolution {i}/{len(todo)} "
                 f"resolved_total={sum(1 for x in res.values() if x['pmid'])}")
    with open(res_cache, "w", encoding="utf-8") as f:
        json.dump(res, f)
    resolved = sum(1 for v in registry.values() if v["pmid"])
    _log(f"Phase A done: corpus PMIDs {resolved}/{len(registry)}")

    # ---- Phase B: iCite fetch (resumable, batches of 100) ----
    icite = {}
    if os.path.exists(icite_cache):
        try:
            icite = {int(k): v for k, v in json.load(
                open(icite_cache, encoding="utf-8")).items()}
        except (OSError, json.JSONDecodeError):
            icite = {}
    all_pm = sorted({int(v["pmid"]) for v in registry.values() if v["pmid"]})
    fetch = [p for p in all_pm if p not in icite]
    _log(f"iCite fetch: {len(fetch)} new PMIDs (cached: {len(icite)})")
    import urllib.request
    B = 100
    for i in range(0, len(fetch), B):
        batch = fetch[i:i + B]
        qs = ",".join(str(p) for p in batch)
        for attempt in range(3):
            try:
                url = f"{_ICITE_API}?pmids={qs}"
                req = urllib.request.Request(url, headers={"User-Agent": _UA})
                with urllib.request.urlopen(req, timeout=90) as r:
                    d = json.load(r)
                for p in d.get("data", []):
                    icite[int(p["pmid"])] = {
                        "year": p.get("year"),
                        "journal": p.get("journal"),
                        "citation_count": p.get("citation_count"),
                        "rcr": p.get("relative_citation_ratio"),
                        "cited_by": p.get("cited_by") or [],
                        "references": p.get("references") or [],
                    }
                break
            except Exception:
                if attempt == 2:
                    _log(f"  iCite batch {i // B + 1} failed")
                time.sleep(3 * (attempt + 1))
        if (i // B) % 5 == 0:
            with open(icite_cache, "w", encoding="utf-8") as f:
                json.dump({str(k): v for k, v in icite.items()}, f)
    with open(icite_cache, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in icite.items()}, f)
    _log(f"Phase B done: iCite records {len(icite)}")

    # ---- Phase C: source-key graph ----
    pm2src = {}
    for s, v in registry.items():
        if v["pmid"]:
            pm2src.setdefault(int(v["pmid"]), []).append(s)
    edges, edges_pmid = set(), set()
    for P, v in icite.items():
        if P not in pm2src:
            continue
        for c in v["cited_by"]:
            C = int(c)
            if C in pm2src and C != P:
                edges_pmid.add((C, P))
                for a in pm2src[C]:
                    for b in pm2src[P]:
                        if a != b:
                            edges.add((a, b))
        for r_ in v["references"]:
            R = int(r_)
            if R in pm2src and R != P:
                edges_pmid.add((P, R))
                for a in pm2src[P]:
                    for b in pm2src[R]:
                        if a != b:
                            edges.add((a, b))
    in_deg, out_deg = {}, {}
    for a, b in edges:
        in_deg[b] = in_deg.get(b, 0) + 1
        out_deg[a] = out_deg.get(a, 0) + 1
    nodes = {}
    for s, v in registry.items():
        P = int(v["pmid"]) if v["pmid"] else None
        ic = icite.get(P) if P is not None else None
        ic = ic or {}
        nodes[s] = {"pmid": P, "doi": v["doi"], "title": v["title"][:90],
                    "year": v["year"], "journal": v["journal"][:60],
                    "citation_count": ic.get("citation_count"),
                    "rcr": ic.get("rcr"),
                    "in_corpus_cited_by": in_deg.get(s, 0),
                    "in_corpus_cites": out_deg.get(s, 0)}
    graph = {"generated": __import__("datetime").datetime.now().isoformat(
                 timespec="seconds"),
             "scope": "full corpus",
             "key_def": "node key = chroma metadata.source (md filename) — "
                        "direct join for retrieval results",
             "n_nodes": len(nodes),
             "n_with_pmid": sum(1 for n in nodes.values() if n["pmid"]),
             "n_edges": len(edges),
             "edge_def": "edges: [citing_source, cited_source]; "
                         "edges_pmid mirror: [citing_pmid, cited_pmid]",
             "nodes": nodes,
             "edges": sorted([list(e) for e in edges]),
             "edges_pmid": sorted([list(e) for e in edges_pmid])}
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=1)
    os.replace(tmp, out_path)
    _log(f"Phase C done: {len(nodes)} nodes / {len(edges)} edges "
         f"({sum(1 for n in nodes.values() if n['pmid'])} with pmid)")

    # csv mirror (same rows as eph scripts)
    import csv as _csv
    src2pm = {s: (int(v["pmid"]) if v["pmid"] else "")
              for s, v in registry.items()}
    csv_path = os.path.join(data_dir, "citation_edges.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["citing_source", "cited_source", "citing_pmid",
                    "cited_pmid", "citing_title", "cited_title"])
        for a, b in sorted(edges):
            w.writerow([a, b, src2pm.get(a, ""), src2pm.get(b, ""),
                        nodes.get(a, {}).get("title", "")[:70],
                        nodes.get(b, {}).get("title", "")[:70]])
    _log(f"csv mirror written: {csv_path}")

    # rebuild reference_graph.json so _merge_icite_edges consumes the update
    build_reference_graph(progress=progress)
    return graph


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

    icite_injected = _merge_icite_edges(papers, edges, progress=progress)
    n_resolved = sum(1 for e in edges if e.get("resolved"))
    if progress:
        src = f"icite injected {icite_injected}, " if icite_injected else ""
        print(f"[refgraph] {len(papers)} papers, {len(edges)} edges "
              f"({src}heuristic {len(edges) - n_resolved}) → {out_path}")

    graph = {"version": _GRAPH_VERSION,
             "built_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
             "papers": papers,
             "edges": edges}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False)
    os.replace(tmp, out_path)
    return graph


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

def load_graph() -> Optional[Dict]:
    """Load the citation graph, cached by file mtime (called per query —
    re-reading a multi-MB JSON on every search is waste; the graph only
    changes when build_reference_graph runs)."""
    cfg = get_config()
    path = cfg.get("reference_graph_path") or os.path.join(
        cfg["data_dir"], "reference_graph.json")
    if not os.path.exists(path):
        return None
    try:
        mtime = os.path.getmtime(path)
        cache = getattr(load_graph, "_cache", None)
        if cache and cache[0] == path and cache[1] == mtime:
            return cache[2]
        with open(path, encoding="utf-8") as f:
            graph = json.load(f)
        load_graph._cache = (path, mtime, graph)  # type: ignore[attr-defined]
        return graph
    except (OSError, json.JSONDecodeError):
        return None


def neighbors(graph: Dict, source_or_title: str, limit: int = 30) -> Dict[str, Dict]:
    """1-hop citation neighborhood of a paper, resolved to LIBRARY sources.

    Both directions:
      forward  ("cited-by") — library papers whose in-text citations match
                 `src`'s first-author surname + year (snowball forward
                 logic, reused);
      backward ("cites")     — library papers resolving the (author, year)
                 targets of `src`'s own citations. v1 graphs carry no
                 title hints (in-text extraction), so resolution is the
                 author-year heuristic: edge surname prefix (≥4 chars)
                 matches a candidate's first-author surname AND years
                 are equal.

    Returns {resolved_source: {"via": …, "direction": "cites"|"cited-by"}}
    — empty when the source isn't in the graph. Never raises.
    """
    try:
        src = _resolve_source(graph, source_or_title)
        if src is None:
            return {}
        papers = graph["papers"]

        def _surname_year(meta: Dict) -> Tuple[Set[str], str]:
            surnames: Set[str] = set()
            for a in (meta.get("authors", "") or "").split(";"):
                a = a.strip()
                if a:
                    surnames.add(a.split()[0].split(",")[0].lower())
            return surnames, str(meta.get("year", "") or "")

        out: Dict[str, Dict] = {}

        # forward: who cites src (snowball's matcher does the work)
        fwd = snowball(graph, src, "forward", limit=limit)
        for m in fwd.get("matches", []):
            s = m.get("source")
            if s and s != src:
                out.setdefault(s, {"via": m.get("via", ""),
                                   "direction": "cited-by"})

        # backward: whom src cites — resolve (author, year) edges against
        # library first-author surnames + years.
        # v2: resolved (iCite) edges first — exact targets, no heuristics
        for e in graph["edges"]:
            if e.get("resolved") and e.get("from") == src:
                tgt = e.get("to")
                if tgt and tgt != src and tgt not in out:
                    out[tgt] = {"via": "iCite (PubMed verified)",
                                "direction": "cites"}
        year_index: Dict[str, List[str]] = {}
        for s, m in papers.items():
            _, y = _surname_year(m)
            if y:
                year_index.setdefault(y, []).append(s)
        for e in graph["edges"]:
            if e.get("resolved") or e.get("from") != src:
                continue
            ta = (e.get("to_author") or "").lower()
            ta_base = re.split(r"\s+et\s+al\.?$|\s*&\s*|\s+and\s+", ta)[0]
            ta_base = ta_base.strip().rstrip(",")
            ty = e.get("to_year") or ""
            if len(ta_base) < 3 or not ty:
                continue
            for cand in year_index.get(ty, []):
                if cand == src or cand in out:
                    continue
                csurn, _ = _surname_year(papers.get(cand, {}))
                if any(cs.startswith(ta_base[:4]) for cs in csurn if cs):
                    out[cand] = {"via": e.get("to_raw", ""),
                                 "direction": "cites"}
                    if len(out) >= limit * 2:
                        return out
        return out
    except Exception:
        # citation boost is an enhancement, never a dependency
        return {}


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
        # v2: pass 1 — resolved (iCite-verified) edges, exact reverse lookup
        for e in graph["edges"]:
            if e.get("resolved") and e.get("to") == src:
                hit = e["from"]
                if hit != src:
                    matches.append({"source": hit,
                                    "title": by_source.get(hit, {}).get("title", ""),
                                    "year": by_source.get(hit, {}).get("year", ""),
                                    "via": "iCite (PubMed verified)"})
        # v2: pass 2 — heuristic author-year matching (existing logic)
        for e in graph["edges"]:
            if e.get("resolved") or e["from"] == src or \
                    (not e.get("to_author") and not e.get("to_title_hint")):
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
        # v2: pass 1 — resolved (iCite-verified) edges: exact targets
        for e in graph["edges"]:
            if e.get("resolved") and e.get("from") == src:
                target = e.get("to")
                matches.append({
                    "raw_ref": "iCite (PubMed verified)",
                    "resolved_source": target,
                    "resolved_title": by_source.get(target, {}).get("title", "") if target else "",
                    "in_library": target is not None,
                })
        # v2: pass 2 — heuristic reference resolution (existing logic)
        for e in graph["edges"]:
            if e.get("resolved") or e["from"] != src:
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
    similarity). Rank by overlap of cited-target sets.

    v2: resolved (iCite-verified) edges contribute their exact target source
    (strong signal); heuristic edges contribute (author, year) pairs. A single
    pass over edges builds per-source cited sets (was O(P*E) full scans)."""
    src = _resolve_source(graph, source_or_title)
    if src is None:
        return []
    # one pass over edges: source -> set of cited-target keys
    cited: Dict[str, set] = {}
    for e in graph["edges"]:
        f = e.get("from")
        if not f:
            continue
        if e.get("resolved"):
            key = ("src", e.get("to"))          # exact target (iCite)
        else:
            key = (e.get("to_author"), e.get("to_year"))
        cited.setdefault(f, set()).add(key)
    mine = cited.get(src) or set()
    if not mine:
        return []
    scored = []
    for other, theirs in cited.items():
        if other == src or not theirs:
            continue
        j = len(mine & theirs) / len(mine | theirs)
        if j > 0:
            scored.append({"source": other,
                           "title": graph["papers"].get(other, {}).get("title", ""),
                           "year": graph["papers"].get(other, {}).get("year", ""),
                           "coupling": round(j, 3)})
    scored.sort(key=lambda d: -d["coupling"])
    return scored[:limit]


def _title_key(title: str) -> Set[str]:
    t = (title or "").lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    stop = {"a", "an", "the", "of", "in", "on", "and", "or", "for", "to", "by", "with"}
    return {w for w in t.split() if len(w) > 1 and w not in stop}