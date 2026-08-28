#!/usr/bin/env python3
"""
zotero_match.py — Title-similarity & DOI verification for Zotero hits.

Problem it solves (paper-qa mechanism, see /Disk_bot/notes/citation_rag/01):
bib_rag_writer_debate.search_zotero() took items[0] from a Zotero search and
trusted it. A fuzzy title hit with a low actual similarity produces a
" Zhang-guan-li-dai " citation (right paper family, wrong paper). paper-qa
verifies every citation against the source's DOI/title before allowing it.

This module provides:
  - title_similarity(a, b)  → normalized-token Jaccard + containment blend
  - verify_zotero_hit(query_title, query_doi, hit) → (ok, score, reason)
  - pick_best_hit(candidates, query_title, query_doi) → best candidate or None

All zero-LLM. Env tunables:
  ZOTERO_MATCH_MIN_SIM   (default 0.55) — min blended similarity to accept
  ZOTERO_MATCH_MIN_DOI_PRE  (default 8)  — min shared DOI prefix chars
"""
from __future__ import annotations

import os
import re
from typing import Dict, Optional, Tuple

# --- tunables -----------------------------------------------------------

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


MIN_SIM = _env_float("ZOTERO_MATCH_MIN_SIM", 0.55)
MIN_DOI_PREFIX = _env_int("ZOTERO_MATCH_MIN_DOI_PREFIX", 8)

# Same normalization philosophy as citation_guard.normalize, but tailored to
# paper titles: strip "et al", years, journal-y suffixes, punctuation.
_STOP = frozenset("""
a an the and or of in on for from by with to at is are was were be been
study studies analysis approach method methods using based novel new
""".split())


def _title_tokens(title: str) -> set:
    t = (title or "").lower()
    t = re.sub(r"et\s+al\.?", " ", t)
    t = re.sub(r"\b(19|20)\d{2}\b", " ", t)          # years
    t = re.sub(r"[^a-z0-9\s\-]", " ", t)
    toks = re.findall(r"[a-z][a-z\-]*\d*|\d+", t)
    return {w for w in toks if len(w) > 1 and w not in _STOP}


def title_similarity(query: str, cand: str) -> float:
    """Blended similarity in [0,1]:
       0.5 * Jaccard + 0.5 * containment(smaller/larger set).
    Containment rewards the case where the query is a truncated version of
    the candidate title (common with Zotero's 'et al - year - ' prefixes)."""
    q, c = _title_tokens(query), _title_tokens(cand)
    if not q or not c:
        return 0.0
    inter = len(q & c)
    jac = inter / len(q | c)
    contain = inter / min(len(q), len(c))
    return 0.5 * jac + 0.5 * contain


def doi_match(query_doi: str, cand_doi: str) -> Optional[bool]:
    """True/False when both DOIs are present (canonical comparison via
    identifiers.normalize_doi — strips doi:/https://doi.org/ prefixes and
    version suffixes first); None when either side is missing.

    Two-tier: exact canonical equality → True; shared registrant prefix
    (≥8 chars) → True (suffix drift tolerated); clear divergence → False.
    """
    from identifiers import normalize_doi, doi_prefix_agree
    nq, nc = normalize_doi(query_doi), normalize_doi(cand_doi)
    if not nq or not nc:
        return None
    if nq == nc:
        return True
    return doi_prefix_agree(nq, nc, min_prefix=MIN_DOI_PREFIX)


def verify_zotero_hit(query_title: str, query_doi: str,
                      hit: Dict[str, str]) -> Tuple[bool, float, str]:
    """Verify one Zotero candidate against the query.

    Returns (ok, score, reason):
      ok=True  → hit is plausibly the same paper
      ok=False → reject (score < threshold, or DOI conflict)
    DOI logic: if both DOIs present and they conflict → reject regardless of
    title score; if they agree → accept even at lower title similarity.
    """
    sim = title_similarity(query_title, hit.get("title", ""))
    dm = doi_match(query_doi, hit.get("doi", ""))
    if dm is False:
        return False, sim, "doi-conflict"
    if dm is True:
        return True, sim, "doi-agrees"
    if sim >= MIN_SIM:
        return True, sim, "title-match"
    return False, sim, "below-threshold"


def pick_best_hit(candidates: list, query_title: str,
                  query_doi: str = "") -> Optional[Dict]:
    """Scan candidates (list of {key, title, doi, ...}), return the first
    verified hit in *score order*, or None when nothing verifies.

    This replaces the old blind `items[0]` pickup.
    """
    scored = []
    for cand in candidates:
        if not isinstance(cand, dict) or not cand.get("key"):
            continue
        ok, score, reason = verify_zotero_hit(query_title, query_doi, cand)
        scored.append((ok, score, reason, cand))
    # DOI-agreeing hits first, then by score
    scored.sort(key=lambda t: (not (t[2] == "doi-agrees"), -t[1]))
    for ok, _score, _reason, cand in scored:
        if ok:
            return cand
    return None