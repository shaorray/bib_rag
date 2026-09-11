#!/usr/bin/env python3
"""
match_references.py — Resolve raw reference-graph edges to corpus papers (LOCAL only).

WS-B of the agentic-retrieval upgrade: reference_graph.json v2 mixes
(a) heuristic in-text-citation edges  {from, to_raw, to_author, to_year,
 to_title_hint}  and (b) iCite/Crossref-verified edges {from, to, resolved}.
The raw (a) edges point at citation STRINGS ("Daniel and Reynolds (1995)"),
not corpus ids — this script resolves them against the corpus registry by
author-surname + year, without any network call.

Reality of the current eph_rag build (verified 2026-08-31):
  * to_title_hint is EMPTY on every raw edge (reference lists are truncated
    away at ingest; in-text citations are the only extraction source) →
    title-similarity tiers from the original plan are inapplicable; tiers
    are adapted to surname+year evidence (documented in the report).
  * 17,595 edges are already verified — passed through, keys converted to
    the chroma '.md' key space.

Key spaces (see src/reference_graph.py):
  reference_graph papers/edges  → parent-store STEM keys
      ("10068468.md" → "10068468_md"; title-style names truncated at 100
       chars, so the reverse map is built from citation_graph.json node
       keys, not by string transform — 0 collisions verified).
  citation_graph.json nodes     → chroma '.md' source keys (join target).
  Output uses '.md' keys throughout; every from/to is validated to join
  100% against citation_graph.json nodes.

Matching model (per unique (to_author, to_year, to_title_hint) target):
  candidates   = corpus papers whose author-surname set contains the
                 target's primary surname AND |year - to_year| <= tol
  surname comp = 1.0 primary==first-author surname, else 0.6 (any-author)
  year comp    = 1.0 exact, 0.5 off-by-one (± --year-tol)
  coauthor comp= +1.0 secondary surname also in author set (two-surname
                 targets "X and Y"), -1.0 if absent, 0 if no secondary
  score        = 0.55*surname + 0.35*year + 0.10*coauthor   (max 1.0)
  tiers        strong: score >= 0.90 and clear margin (gap > 0.10) over
                 runner-up;  weak: score >= 0.55 (incl. ties → ambiguous);
                 unmatched: no candidate / score < 0.55
  ambiguity    runner-up within 0.10 of best → ambiguous=True, pick is a
                 deterministic tiebreak (citation_count, in_corpus_cited_by,
                 key order) — consumers should treat ambiguous as
                 "needs verification".

Usage:
  python3 scripts/match_references.py                    # full run + spot-check
  python3 scripts/match_references.py --data-root DIR
  python3 scripts/match_references.py --spot-check 20 --unmatched-sample 20
Pure stdlib; no network; never writes reference_graph.json itself.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import random
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict

# --------------------------------------------------------------------------
# Tunables (CLI-overridable)
# --------------------------------------------------------------------------
W_SURNAME, W_YEAR, W_COAUTHOR = 0.55, 0.35, 0.10
ANY_AUTHOR_SURNAME = 0.6     # surname component when primary is a non-first author
YEAR_OFF_SCORE = 0.5         # year component when |diff| == 1
COAUTHOR_MISS = -1.0         # secondary surname cited but absent from paper

DEFAULT_STRONG_MIN = 0.90
DEFAULT_WEAK_MIN = 0.55
DEFAULT_AMBIG_GAP = 0.10     # runner-up within this of best → ambiguous
DEFAULT_YEAR_TOL = 1

DEFAULT_DATA_ROOT = "/Disk_bot/RAG/eph_rag"
DEFAULT_SEED = 20260831

_YEAR_RE = re.compile(r"((?:19|20)\d{2})")

# particles that may prefix a surname ("De Rooij", "van der Meer")
_PARTICLES = {"de", "van", "von", "der", "den", "del", "della", "di", "da",
              "la", "le", "du", "ter", "ten", "op", "'t"}


# --------------------------------------------------------------------------
# Normalization helpers
# --------------------------------------------------------------------------
def fold(text: str) -> str:
    """NFKD-fold to ascii-ish, lowercase, keep alphanumerics only."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", text.lower())


def clean_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\n", " ")).strip()


def parse_year(value) -> int | None:
    m = _YEAR_RE.search(str(value or ""))
    return int(m.group(1)) if m else None


# --------------------------------------------------------------------------
# Registry-side author parsing
# --------------------------------------------------------------------------
def _segment_surname(seg: str) -> list[str]:
    """Surnames from ONE author segment.

    Handles "Surname,Given", "Surname Initials" (PubMed), "De Rooij F",
    "Surname". Returns 1-2 forms: the full particle-joined surname and the
    capitalized head token alone ("derooij", "rooij") when particles lead.
    """
    seg = clean_ws(seg)
    if not seg:
        return []
    if "," in seg:
        seg = seg.split(",")[0].strip()
        if not seg:
            return []
    toks = seg.split()
    if not toks:
        return []
    # collect leading particles
    parts: list[str] = []
    i = 0
    while i < len(toks) and toks[i].lower().strip(".") in _PARTICLES:
        parts.append(toks[i])
        i += 1
    if i >= len(toks):
        # all particles, no head — take the first token as best guess
        return [fold(toks[0])]
    head = toks[i]
    # stop at initials ("Woodward WA" → Woodward; "De Rooij F" → De Rooij)
    if re.fullmatch(r"[A-Z]{1,3}\.?", head):
        return [fold(toks[0])] if not parts else [fold("".join(parts) + head)]
    surname = "".join(parts) + head if parts else head
    out = [fold(surname)]
    if parts:
        out.append(fold(head))  # "rooij" also indexed
    return out


def parse_registry_authors(authors: str) -> tuple[str, set[str], bool]:
    """(first_surname, surname_set, ok) from a registry authors string.

    Formats in this corpus:
      "Surname,Given; Surname,Given; ..."      (Zotero/PubMed semicolon)
      "Surname Initials; Surname Initials"     (PubMed semicolon)
      "Surname,Given"                          (single author, comma)
      "Surname1, Given1, Surname2, Given2..."  (Zotero flattened pairs —
                                                surnames at EVEN indices;
                                                verified on this corpus)
      "Surname Initials"                       (single author, space)
    """
    a = clean_ws(authors or "")
    if not a or a.strip(",; ") == "":
        return "", set(), False
    segments: list[str] = []
    if ";" in a:
        segments = [s for s in a.split(";") if s.strip()]
    elif "," in a:
        parts = [p.strip() for p in a.split(",") if p.strip()]
        # Zotero flattened pairs → even indices are surnames (0-based).
        # Verified against this corpus: odd parts are always given names
        # ("Barry M.", "Sarah", "T. J."), never "Surname Initials".
        segments = parts[0::2]
    else:
        segments = [a]
    if not segments:
        return "", set(), False
    surnames: list[str] = []
    for seg in segments:
        surnames.extend(_segment_surname(seg))
    surnames = [s for s in surnames if s and len(s) > 1]
    if not surnames:
        return "", set(), False
    return surnames[0], set(surnames), True


def surname_from_filename(md_key: str) -> str:
    """Fallback first surname from title-style filenames.

    "Abdul-Wajid et al. - 2015 - Title.md" → abdulwajid
    "Autorino_2026_Tissue organization....md" → autorino
    """
    base = md_key[:-3] if md_key.endswith(".md") else md_key
    m = re.match(r"^\s*([A-Za-zÀ-ÿ'\-]+)(?:\s+et\s+al\.)?\s*-", base)
    if not m:
        m = re.match(r"^\s*([A-Za-zÀ-ÿ'\-]+)_\d{4}_", base)
    if m:
        s = fold(m.group(1))
        if len(s) > 1:
            return s
    return ""


# --------------------------------------------------------------------------
# Target-side parsing (citation strings)
# --------------------------------------------------------------------------
def parse_target_author(to_author: str) -> tuple[str, str]:
    """(primary, secondary) normalized surnames from "Daniel and Reynolds" /
    "Gebbink et al." / "Smith". secondary="" when absent."""
    a = clean_ws(to_author or "")
    a = re.sub(r"\bet\s+\.?\s*al\.?", "", a)          # drop "et al."
    a = re.sub(r"\s+and\s+|\s*&\s+", " § ", a)        # explicit separator
    a = re.sub(r",.*$", "", a)                        # anything after a comma
    parts = [p for p in a.split(" § ") if p.strip()]
    surnames = []
    for p in parts[:2]:
        toks = p.split()
        if not toks:
            continue
        # particles: join to head ("van der Meer" → vandermeer)
        i = 0
        while i < len(toks) - 1 and toks[i].lower().strip(".") in _PARTICLES:
            i += 1
        s = fold("".join(toks[: i + 1])) if i else fold(toks[0])
        if len(s) > 1:
            surnames.append(s)
    primary = surnames[0] if surnames else ""
    secondary = surnames[1] if len(surnames) > 1 else ""
    return primary, secondary


def title_similarity(a: str, b: str) -> float:
    """Token-set overlap ∪ difflib ratio — used only when a title hint
    exists (kept for future builds / plan fidelity)."""
    if not a or not b:
        return 0.0
    ta = {fold(t) for t in a.split() if fold(t)}
    tb = {fold(t) for t in b.split() if fold(t)}
    if not ta or not tb:
        return 0.0
    overlap = len(ta & tb) / max(1, min(len(ta), len(tb)))
    ratio = difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()
    return 0.6 * overlap + 0.4 * ratio


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_stem_map(citation_nodes: dict) -> dict[str, str]:
    """{stem → '.md' key} from citation-graph node keys (authoritative)."""
    stem2md = {}
    collisions = []
    for md in citation_nodes:
        stem = re.sub(r"[^\w\-]", "_", md)[:100]
        if stem in stem2md and stem2md[stem] != md:
            collisions.append(stem)
        stem2md[stem] = md
    if collisions:
        raise SystemExit(f"FATAL: {len(collisions)} stem collisions — "
                         f"cannot invert key map: {collisions[:3]}")
    return stem2md


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------
def build_registry(data_root: str, refgraph_papers: dict, citation_nodes: dict,
                   incremental: dict, papers_meta: dict, stem2md: dict,
                   verbose: bool = True) -> tuple[dict, dict]:
    """registry: {'.md' key: {title, year, authors_raw, first_surname,
    surname_set, citation_count, in_corpus_cited_by, fields_from}}.

    Priority chain per field (first non-empty wins):
      title : refgraph papers → citation node → incremental → papers_meta
              → parent_store meta → filename
      year  : refgraph papers → citation node → incremental → papers_meta
              → parent_store meta → filename
      authors: refgraph papers → incremental → parent_store meta → filename
    parent_store is read lazily, only for papers with holes."""
    parent_dir = os.path.join(data_root, "parent_store")
    reg: dict[str, dict] = {}
    provenance = Counter()
    need_parent: list[str] = []

    for md, node in citation_nodes.items():
        stem = re.sub(r"[^\w\-]", "_", md)[:100]
        rp = refgraph_papers.get(stem, {})
        inc = incremental.get(md, {})
        pm = papers_meta.get(md, {})

        title = (rp.get("title") or "").strip() \
            or (node.get("title") or "").strip() \
            or (inc.get("title") or "").strip() \
            or (pm.get("title") or "").strip()
        year = parse_year(rp.get("year")) \
            or parse_year(node.get("year")) \
            or parse_year(inc.get("year")) \
            or parse_year(pm.get("year"))
        authors = clean_ws(rp.get("authors") or "") \
            or clean_ws(inc.get("authors") or "")

        first, sset, ok = parse_registry_authors(authors)
        if not title or not year or not ok:
            need_parent.append(md)
        reg[md] = {
            "title": title, "year": year, "authors_raw": authors,
            "first_surname": first, "surname_set": sset,
            "citation_count": node.get("citation_count") or 0,
            "in_corpus_cited_by": node.get("in_corpus_cited_by") or 0,
            "fields_from": {
                "title": "refgraph" if (rp.get("title") or "").strip() else
                         ("citation_node" if (node.get("title") or "").strip()
                          else "other"),
                "authors": "refgraph" if (rp.get("authors") or "").strip() else
                           ("incremental" if (inc.get("authors") or "").strip()
                            else "other"),
            },
        }

    # lazy parent_store fallback for papers with holes
    for md in need_parent:
        r = reg[md]
        ppath = os.path.join(parent_dir, re.sub(r"[^\w\-]", "_", md)[:100] + ".json")
        pmeta = {}
        try:
            with open(ppath, encoding="utf-8") as f:
                chunks = json.load(f)
            if isinstance(chunks, list) and chunks:
                pmeta = chunks[0].get("meta") or {}
        except (OSError, json.JSONDecodeError):
            pass
        if not r["title"]:
            r["title"] = (pmeta.get("title") or "").strip()
            if r["title"]:
                provenance["title:parent_store"] += 1
        if not r["year"]:
            r["year"] = parse_year(pmeta.get("year"))
            if r["year"]:
                provenance["year:parent_store"] += 1
        if not r["surname_set"]:
            a = clean_ws(pmeta.get("authors") or "")
            if a:
                first, sset, ok = parse_registry_authors(a)
                if ok:
                    r["authors_raw"] = r["authors_raw"] or a
                    r["first_surname"], r["surname_set"] = first, sset
                    provenance["authors:parent_store"] += 1
        if not r["first_surname"]:
            s = surname_from_filename(md)
            if s:
                r["first_surname"] = s
                r["surname_set"] = r["surname_set"] or {s}
                provenance["authors:filename"] += 1
        if not r["year"]:
            y = parse_year(md)
            if y:
                r["year"] = y
                provenance["year:filename"] += 1

    cov = {
        "papers": len(reg),
        "with_title": sum(1 for r in reg.values() if r["title"]),
        "with_year": sum(1 for r in reg.values() if r["year"]),
        "with_authors_raw": sum(1 for r in reg.values() if r["authors_raw"]),
        "with_first_surname": sum(1 for r in reg.values() if r["first_surname"]),
        "with_surname_set": sum(1 for r in reg.values() if r["surname_set"]),
        "title_source": dict(Counter(r["fields_from"]["title"] for r in reg.values())),
        "authors_source": dict(Counter(r["fields_from"]["authors"] for r in reg.values())),
        "parent_store_fallback": dict(provenance),
    }
    if verbose:
        print("[registry] coverage:")
        for k, v in cov.items():
            print(f"  {k}: {v}")
    return reg, cov


def build_surname_index(registry: dict) -> dict[str, dict[str, bool]]:
    """{surname → {md_key: is_first_author}} over ALL authors."""
    idx: dict[str, dict[str, bool]] = defaultdict(dict)
    for md, r in registry.items():
        first = r["first_surname"]
        for s in r["surname_set"]:
            idx[s][md] = (s == first)
    return dict(idx)


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------
def match_target(primary: str, secondary: str, year: int, hint: str,
                 surname_index: dict, registry: dict,
                 year_tol: int, strong_min: float, weak_min: float,
                 ambig_gap: float) -> dict:
    """Resolve one unique target → {to, score, tier, method, ambiguous,
    runner_up, n_candidates}."""
    if not primary or not year:
        return {"to": None, "score": 0.0, "tier": "unmatched",
                "method": "bad_target", "ambiguous": False,
                "runner_up": None, "n_candidates": 0,
                "reason": "unparseable author/year"}
    cands = surname_index.get(primary)
    if not cands:
        return {"to": None, "score": 0.0, "tier": "unmatched",
                "method": "no_candidate", "ambiguous": False,
                "runner_up": None, "n_candidates": 0,
                "reason": "primary surname not in corpus"}
    scored = []
    for md, is_first in cands.items():
        r = registry[md]
        if not r["year"]:
            continue
        ydiff = abs(r["year"] - year)
        if ydiff > year_tol:
            continue
        sc = 1.0 if is_first else ANY_AUTHOR_SURNAME
        yc = 1.0 if ydiff == 0 else YEAR_OFF_SCORE
        co = 0.0
        if secondary:
            co = 1.0 if secondary in r["surname_set"] else COAUTHOR_MISS
        score = W_SURNAME * sc + W_YEAR * yc + W_COAUTHOR * co
        # title-hint similarity (future builds; empty hints score 0 weight)
        if hint:
            ts = title_similarity(hint, r["title"])
            score = 0.7 * score + 0.3 * ts
        scored.append((score, md, ydiff, is_first, co > 0))
    if not scored:
        return {"to": None, "score": 0.0, "tier": "unmatched",
                "method": "no_candidate", "ambiguous": False,
                "runner_up": None, "n_candidates": 0,
                "reason": "surname in corpus but no paper within ±%d years"
                          % year_tol}
    # deterministic ranking: score, then citation_count, then
    # in_corpus_cited_by, then lexicographic key (min for stability)
    scored.sort(key=lambda t: (-t[0], -registry[t[1]]["citation_count"],
                               -registry[t[1]]["in_corpus_cited_by"], t[1]))
    best_score, best_md, ydiff, is_first, coauth = scored[0]
    runner = scored[1][1] if len(scored) > 1 else None
    runner_score = scored[1][0] if len(scored) > 1 else -1.0
    ambiguous = (runner is not None
                 and (best_score - runner_score) < ambig_gap)
    if best_score >= strong_min and not ambiguous:
        tier = "strong"
    elif best_score >= weak_min:
        tier = "weak"
    else:
        return {"to": None, "score": round(best_score, 3), "tier": "unmatched",
                "method": "low_score", "ambiguous": False,
                "runner_up": runner, "n_candidates": len(scored),
                "reason": f"best candidate score {best_score:.2f} < {weak_min}"}
    parts = [("first_author" if is_first else "any_author"),
             ("year" if ydiff == 0 else f"year_off{ydiff}")]
    if secondary:
        parts.append("coauthor_match" if coauth else "coauthor_miss")
    if ambiguous:
        parts.append("tiebreak")
    return {"to": best_md, "score": round(best_score, 3), "tier": tier,
            "method": "+".join(parts), "ambiguous": ambiguous,
            "runner_up": runner, "n_candidates": len(scored), "reason": ""}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Resolve raw reference-graph edges to corpus papers "
                    "(local surname+year matching, no network).")
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out", default=None,
                    help="default <data-root>/data/reference_graph_matched.json")
    ap.add_argument("--year-tol", type=int, default=DEFAULT_YEAR_TOL)
    ap.add_argument("--strong-min", type=float, default=DEFAULT_STRONG_MIN)
    ap.add_argument("--weak-min", type=float, default=DEFAULT_WEAK_MIN)
    ap.add_argument("--ambig-gap", type=float, default=DEFAULT_AMBIG_GAP)
    ap.add_argument("--spot-check", type=int, default=10,
                    help="matched pairs to print side-by-side (0 = none)")
    ap.add_argument("--unmatched-sample", type=int, default=10,
                    help="unmatched targets to print with reasons")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    data = os.path.join(args.data_root, "data")
    rg_path = os.path.join(data, "reference_graph.json")
    cg_path = os.path.join(data, "citation_graph.json")

    rg = load_json(rg_path)
    cg = load_json(cg_path)
    inc_path = os.path.join(data, "incremental_metadata.json")
    pm_path = os.path.join(data, "papers_meta.json")
    incremental = load_json(inc_path) if os.path.exists(inc_path) else {}
    papers_meta = (load_json(pm_path).get("papers", {})
                   if os.path.exists(pm_path) else {})

    nodes = cg["nodes"]
    stem2md = build_stem_map(nodes)
    refgraph_papers = rg.get("papers", {})

    # sanity: every refgraph paper stem must map to a citation-graph node
    unmapped = [s for s in refgraph_papers if s not in stem2md]
    if unmapped:
        print(f"FATAL: {len(unmapped)} refgraph paper stems have no "
              f"citation-graph node: {unmapped[:3]}", file=sys.stderr)
        return 2

    registry, cov = build_registry(args.data_root, refgraph_papers, nodes,
                                   incremental, papers_meta, stem2md,
                                   verbose=not args.quiet)
    surname_index = build_surname_index(registry)

    edges_in = rg.get("edges", [])
    raw = [e for e in edges_in if not e.get("resolved")]
    resolved = [e for e in edges_in if e.get("resolved")]

    # ---- dedupe raw edges by unique target, match once ----
    targets: dict[tuple, dict] = {}
    target_edges: dict[tuple, list[int]] = defaultdict(list)
    for i, e in enumerate(raw):
        key = (clean_ws(e.get("to_author", "")), str(e.get("to_year", "")),
               clean_ws(e.get("to_title_hint") or ""))
        target_edges[key].append(i)
    if not args.quiet:
        print(f"[targets] raw edges {len(raw)} → unique targets {len(target_edges)}")

    for key in target_edges:
        to_author, to_year_s, hint = key
        primary, secondary = parse_target_author(to_author)
        year = parse_year(to_year_s)
        targets[key] = match_target(primary, secondary, year, hint,
                                    surname_index, registry,
                                    args.year_tol, args.strong_min,
                                    args.weak_min, args.ambig_gap)

    # ---- emit edges (original order), pass verified through ----
    out_edges: list[dict] = []
    tier_rank = {"strong": 0, "weak": 1, "unmatched": 2}
    seen_pairs: dict[tuple, int] = {}   # (from, to) → index in out_edges
    dropped_dup = 0
    replaced_by_verified = 0
    self_loops = 0
    from_join_fail = []
    to_join_fail = []
    raw_tier_prededupe = Counter()

    def rank(rec: dict) -> tuple:
        # lower is better: tier, then score, then provenance (verified wins ties)
        return (tier_rank[rec["tier"]], -rec["score"],
                0 if rec.get("resolved") else 1)

    def emit(rec: dict):
        nonlocal dropped_dup, replaced_by_verified, self_loops
        if rec["from"] not in nodes:
            from_join_fail.append(rec["from"])
            return
        if rec.get("to") is not None and rec["to"] not in nodes:
            to_join_fail.append(rec["to"])
            return
        if not rec.get("resolved"):
            raw_tier_prededupe[rec["tier"]] += 1
        if rec.get("to") is not None:
            if rec["to"] == rec["from"]:
                self_loops += 1
                rec = dict(rec, to=None, tier="unmatched",
                           method=rec["method"] + "+self_loop_filtered",
                           ambiguous=False)
            else:
                pair = (rec["from"], rec["to"])
                if pair in seen_pairs:
                    j = seen_pairs[pair]
                    prev = out_edges[j]
                    # keep the better-ranked record (verified beats an
                    # equal-score heuristic match)
                    if rank(rec) < rank(prev):
                        if rec.get("resolved") and not prev.get("resolved"):
                            replaced_by_verified += 1
                        out_edges[j] = rec
                    dropped_dup += 1
                    return
                seen_pairs[pair] = len(out_edges)
        out_edges.append(rec)

    for e in edges_in:
        from_md = stem2md.get(e.get("from"))
        if from_md is None:
            from_join_fail.append(e.get("from"))
            continue
        if e.get("resolved"):
            to_md = stem2md.get(e.get("to"))
            emit({"from": from_md, "to": to_md, "to_raw": e.get("to_raw", ""),
                  "to_author": "", "to_year": "",
                  "score": 1.0, "method": "verified", "tier": "strong",
                  "ambiguous": False, "resolved": True})
        else:
            key = (clean_ws(e.get("to_author", "")), str(e.get("to_year", "")),
                   clean_ws(e.get("to_title_hint") or ""))
            m = targets[key]
            emit({"from": from_md, "to": m["to"], "to_raw": e.get("to_raw", ""),
                  "to_author": e.get("to_author", ""), "to_year": e.get("to_year", ""),
                  "score": m["score"], "method": m["method"], "tier": m["tier"],
                  "ambiguous": m["ambiguous"]})

    # ---- join validation (hard gate) ----
    from_ok = not from_join_fail
    to_ok = not to_join_fail
    n_from = sum(1 for e in out_edges if e["from"] in nodes)
    n_to_nn = sum(1 for e in out_edges if e.get("to") is not None)
    n_to_ok = sum(1 for e in out_edges
                  if e.get("to") is not None and e["to"] in nodes)
    if not args.quiet:
        print(f"[join] from-keys valid: {n_from}/{len(out_edges)}"
              f"{' FAIL ' + str(from_join_fail[:3]) if not from_ok else ''}")
        print(f"[join] to!=null valid:  {n_to_ok}/{n_to_nn}"
              f"{' FAIL ' + str(to_join_fail[:3]) if not to_ok else ''}")
    if not from_ok or not to_ok or n_from != len(out_edges) or n_to_ok != n_to_nn:
        print("FATAL: join validation failed — output NOT written", file=sys.stderr)
        return 3

    # ---- stats ----
    tier_counts = Counter(e["tier"] for e in out_edges)
    raw_out = [e for e in out_edges if not e.get("resolved")]
    n_verified = sum(1 for e in out_edges if e.get("resolved"))
    raw_tier = Counter(e["tier"] for e in raw_out)
    tgt_tier = Counter(m["tier"] for m in targets.values())
    tgt_ambig = sum(1 for m in targets.values() if m["ambiguous"])
    tgt_reasons = Counter(m["method"] for m in targets.values() if not m["to"])
    method_counts = Counter(e["method"] for e in raw_out if e["tier"] != "unmatched")
    unmatched_methods = Counter(e["method"] for e in raw_out if e["tier"] == "unmatched")

    stats = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "data_root": args.data_root,
        "input": {
            "reference_graph_version": rg.get("version"),
            "total_edges": len(edges_in),
            "raw_edges": len(raw),
            "resolved_edges": len(resolved),
            "unique_raw_targets": len(targets),
        },
        "registry_coverage": cov,
        "parameters": {
            "year_tolerance": args.year_tol,
            "strong_min": args.strong_min,
            "weak_min": args.weak_min,
            "ambiguity_gap": args.ambig_gap,
            "weights": {"surname": W_SURNAME, "year": W_YEAR,
                        "coauthor": W_COAUTHOR},
            "any_author_surname": ANY_AUTHOR_SURNAME,
            "year_off_score": YEAR_OFF_SCORE,
        },
        "edges_out": {
            "total": len(out_edges),
            "verified_passthrough": n_verified,
            "raw_matched_strong": raw_tier.get("strong", 0),
            "raw_matched_weak": raw_tier.get("weak", 0),
            "raw_unmatched": raw_tier.get("unmatched", 0),
            "raw_edge_tiers_prededupe": dict(raw_tier_prededupe),
            "raw_edge_match_rate": round(
                (raw_tier.get("strong", 0) + raw_tier.get("weak", 0)) / max(1, len(raw)), 4),
            "raw_edge_match_rate_prededupe": round(
                (raw_tier_prededupe.get("strong", 0) + raw_tier_prededupe.get("weak", 0))
                / max(1, len(raw)), 4),
            "duplicates_dropped": dropped_dup,
            "heuristic_replaced_by_verified": replaced_by_verified,
            "self_loops_filtered": self_loops,
        },
        "unique_targets": {
            "total": len(targets),
            "strong": tgt_tier.get("strong", 0),
            "weak": tgt_tier.get("weak", 0),
            "unmatched": tgt_tier.get("unmatched", 0),
            "ambiguous": tgt_ambig,
            "unmatched_reasons": dict(tgt_reasons.most_common()),
            "target_match_rate": round(
                (tgt_tier.get("strong", 0) + tgt_tier.get("weak", 0)) / max(1, len(targets)), 4),
        },
        "methods_matched": dict(method_counts.most_common()),
        "unmatched_reasons": dict(unmatched_methods.most_common()),
        "join_validation": {
            "from_keys_in_citation_nodes": f"{n_from}/{len(out_edges)}",
            "to_nonnull_in_citation_nodes": f"{n_to_ok}/{n_to_nn}",
            "passed": True,
        },
    }

    out_path = args.out or os.path.join(data, "reference_graph_matched.json")
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"stats": stats, "edges": out_edges}, f,
                  ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, out_path)
    if not args.quiet:
        print(f"[out] wrote {out_path} ({len(out_edges)} edges, "
              f"{os.path.getsize(out_path)/1e6:.1f} MB)")

    # ---- tier/method summary ----
    print("\n===== MATCH SUMMARY =====")
    print(f"raw edges:            {len(raw)}")
    print(f"unique targets:       {len(targets)}")
    print(f"targets strong:       {tgt_tier.get('strong', 0)}")
    print(f"targets weak:         {tgt_tier.get('weak', 0)} "
          f"(ambiguous: {tgt_ambig})")
    print(f"targets unmatched:    {tgt_tier.get('unmatched', 0)}")
    print(f"target match rate:    {stats['unique_targets']['target_match_rate']:.1%}")
    print(f"raw-edge strong/weak/unmatched: "
          f"{raw_tier.get('strong', 0)}/{raw_tier.get('weak', 0)}/{raw_tier.get('unmatched', 0)}")
    print(f"verified passthrough: {sum(1 for e in out_edges if e.get('resolved'))}")
    print(f"self-loops filtered:  {self_loops}   dup pairs dropped: {dropped_dup}")
    print("methods:", dict(method_counts.most_common(8)))
    print("unmatched reasons:", dict(unmatched_methods.most_common(8)))

    # ---- spot-check: matched pairs, side by side ----
    if args.spot_check:
        rng = random.Random(args.seed)
        matched_keys = [k for k, m in targets.items() if m["to"]]
        sample = rng.sample(matched_keys, min(args.spot_check, len(matched_keys)))
        print(f"\n===== SPOT-CHECK: {len(sample)} MATCHED PAIRS =====")
        for k in sample:
            to_author, to_year_s, _ = k
            m = targets[k]
            r = registry[m["to"]]
            n_auth = len(r["surname_set"]) or 1
            print(f"\nTARGET : {to_author} ({to_year_s})"
                  f"   [{m['tier']} score={m['score']} ambiguous={m['ambiguous']}]")
            print(f"  method: {m['method']}  | candidates: {m['n_candidates']}")
            print(f"  CORPUS: {m['to']}")
            print(f"  title : {r['title'][:100]}")
            print(f"  1st surname: {r['first_surname']!r}  "
                  f"({n_auth} author-surnames)  year: {r['year']}")

    # ---- unmatched sample with reasons ----
    if args.unmatched_sample:
        rng = random.Random(args.seed + 1)
        un_keys = [k for k, m in targets.items() if not m["to"]]
        sample = rng.sample(un_keys, min(args.unmatched_sample, len(un_keys)))
        print(f"\n===== UNMATCHED SAMPLE: {len(sample)} =====")
        for k in sample:
            to_author, to_year_s, _ = k
            m = targets[k]
            primary, _ = parse_target_author(to_author)
            in_corpus = primary in surname_index
            yrs = sorted({registry[md]['year'] for md in surname_index.get(primary, {})
                          if registry[md]['year']}) if in_corpus else []
            near = [y for y in yrs if abs(y - (parse_year(to_year_s) or 0)) <= 5]
            print(f"  {to_author} ({to_year_s})  → {m['reason']}")
            print(f"      surname '{primary}' in corpus: {in_corpus}"
                  + (f"; corpus years ±5: {near[:8]}" if in_corpus else ""))

    if not args.quiet:
        print(f"\n[done] {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())