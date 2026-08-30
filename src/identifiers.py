#!/usr/bin/env python3
"""
identifiers.py — Canonical normalization for scholarly identifiers.

Borrowed mechanism (seerai identifier normalization, see
/Disk_bot/notes/zotero_RAG/05): DOI / arXiv / PMID strings arrive in many
shapes — "https://doi.org/10.1016/j.ydbio.2021.01.002",
"doi:10.1016/j.ydbio.2021.01.002", "10.1016/J.YDBIO.2021.01.002V2",
"arXiv:2103.12345v2". Comparing them raw breaks joins between the reference
graph, Zotero hits, and parent_store metadata. One canonical form per
identifier type fixes the key space everywhere.

Canonical forms:
  doi   → "10.xxxx/suffix"   (lowercased, prefixes & version suffixes stripped)
  arxiv → "2103.12345"       (version stripped, arXiv:/abs/ prefix stripped)
  pmid  → digits only
  pmcid → "PMC3452677"       (PMC prefix required; bare digits → PMID)

All zero-LLM, pure functions.
"""
from __future__ import annotations

import os
import re
import unicodedata
from typing import Optional, Tuple

# --- DOI -------------------------------------------------------------------

# Anything that wraps a DOI: https://doi.org/, http://dx.doi.org/, doi:, DOI:
_DOI_PREFIX_RE = re.compile(
    r"^\s*(?:https?://(?:dx\.)?doi\.org/|doi:\s*|DOI:\s*)+", re.I)

# A DOI starts with the registrant "10.<num>/" — anything before it is noise.
_DOI_BODY_RE = re.compile(r"(10\.\d{4,9}/\S+)", re.I)
# Version/annotation suffixes some sources append: .v2 / v3 / ?download=true
_DOI_TRAIL_RE = re.compile(r"[)\]>,.;:!?'\"\s]+$")   # trailing punctuation

# Full-width CJK variants (１０．２０１０３／ｊ．ｓｔｘｂ．…) — PDF text
# extraction of Chinese journals keeps the glyphs full-width. NFKC folds
# them to ASCII digits/punctuation so the DOI regex matches.
def has_fullwidth(s: str) -> bool:
    """True when the string contains full-width chars that NFKC would fold."""
    if not s:
        return False
    return any(unicodedata.normalize("NFKC", ch) != ch for ch in s)


# Suffixes that mark a truncated junk extraction (never a real DOI suffix).
_JUNK_DOI_SUFFIX = {"journal", "journal."}


def extract_doi(raw: str) -> str:
    """Best-effort CLEAN DOI from a mangled raw string; '' when none found.

    Handles the shapes PDF-to-metadata pipelines leave behind:
      '10.1016/x](https://doi.org/10.1016/x)'   markdown link tail
      '10.1038/x**'                             emphasis debris
      '10.1038/](https://doi.org/10.1038/y)'    bare half truncated, link OK
      '10.1371/journal.'                        truncated junk
      '１０．２０１０３／ｊ．…'                    full-width CJK glyphs
    Candidates are collected from doi.org links AND bare 10.x/ runs; the
    longest plausible candidate wins (truncated prefixes lose to the full
    form inside the link). Caller must still verify (Crossref) before
    trusting the result — this is extraction, not validation.
    """
    raw = unicodedata.normalize("NFKC", str(raw or ""))
    cands: list = []
    cands += re.findall(r"(?:dx\.)?doi\.org/(10\.[^\s\)\]\*]+)", raw)
    cands += re.findall(r"(10\.\d{4,5}/[^\s\)\]\*]+)", raw)
    # cut URL/query debris some sources glue on (?urlappend=…&jav=…)
    cands = [re.split(r"[?&]", c)[0] for c in cands]
    cands = [c.rstrip(".,;)") for c in cands]
    good = [c for c in cands
            if len(c.split("/", 1)[1]) >= 5
            and c.split("/", 1)[1].lower() not in _JUNK_DOI_SUFFIX]
    return max(good, key=len) if good else ""


def normalize_doi(raw: str) -> Optional[str]:
    """Extract & canonicalize a DOI from arbitrary input. None when absent.

    "https://doi.org/10.1016/J.YDBIO.2021.01.002" → "10.1016/j.ydbio.2021.01.002"
    "doi: 10.1016/j.ydbio.2021.01.002v2"          → "10.1016/j.ydbio.2021.01.002"
    """
    if not raw:
        return None
    # Fold full-width CJK glyph variants (１０．２０１０３ → 10.20103) before
    # any regex runs — Chinese-journal PDFs extract with full-width digits.
    raw = unicodedata.normalize("NFKC", str(raw))
    s = raw.strip()
    # strip wrapping prefixes
    s = _DOI_PREFIX_RE.sub("", s)
    # find the actual doi body anywhere in the string
    m = _DOI_BODY_RE.search(s)
    if not m:
        return None
    doi = m.group(1).lower().rstrip(".")
    doi = _DOI_TRAIL_RE.sub("", doi)
    # strip version suffix: ...002v2 / ...002.v2 at the very end. Only a "v"
    # preceded by a digit (optionally via a dot) is a version marker —
    # Oxford-style DOIs like 10.1093/nar/gkv370 legitimately end in "v<digits>".
    doi = re.sub(r"(?<=\d)\.?v\d+$", "", doi)
    return doi or None


def doi_equal(a: str, b: str) -> bool:
    """True when both raw strings normalize to the same DOI.
    False when either side has no DOI (caller decides what that means)."""
    na, nb = normalize_doi(a), normalize_doi(b)
    if not na or not nb:
        return False
    return na == nb


# --- arXiv -------------------------------------------------------------------

_ARXIV_RE = re.compile(
    r"(?:arxiv[:\s/]*|\barxiv\.org/abs/)?(\d{4}\.\d{4,5})(v\d+)?", re.I)
# old-style ids: math/0703123, cs/0112017
_ARXIV_OLD_RE = re.compile(
    r"(?:arxiv[:\s/]*)?([a-z\-]+/\d{7})(v\d+)?", re.I)


def normalize_arxiv(raw: str) -> Optional[str]:
    """Canonicalize an arXiv id: strip arXiv:/abs/ and version.
    "arXiv:2103.12345v2" → "2103.12345"; "math/0703123" → "math/0703123"."""
    if not raw:
        return None
    s = str(raw).strip()
    m = _ARXIV_RE.search(s) or _ARXIV_OLD_RE.search(s)
    if not m:
        return None
    # group(1) = id, group(2) = version suffix (stripped)
    return m.group(1).lower()


# --- PMID -------------------------------------------------------------------

_PMID_RE = re.compile(r"^\s*(?:pmid[:\s]*)?(\d{1,9})\s*$", re.I)


def normalize_pmid(raw: str) -> Optional[str]:
    """Extract & canonicalize a PMID: digits only, pmid: prefix stripped."""
    if raw is None:
        return None
    m = _PMID_RE.match(str(raw).strip())
    return m.group(1) if m else None


# --- PMCID -------------------------------------------------------------------

_PMCID_RE = re.compile(r"\bPMC\s?(\d{5,9})\b", re.I)


def normalize_pmcid(raw: str) -> Optional[str]:
    """Extract & canonicalize a PMCID: canonical form is the bare `PMC<digits>`.

    "PMCID: PMC3452677" / "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3452677/"
    → "PMC3452677". Distinguishing PMID vs PMCID matters: both are digit
    strings, so a bare number is treated as PMID (never PMC) and PMCID only
    matches when the PMC prefix is present.
    """
    if raw is None:
        return None
    m = _PMCID_RE.search(str(raw))
    return f"PMC{m.group(1)}" if m else None


def detect_identifier(raw: str) -> Tuple[str, Optional[str]]:
    """Classify raw text → (kind, canonical) where kind ∈ {doi, arxiv, pmcid,
    pmid, unknown}. First match wins (DOI before arXiv before PMCID before
    PMID — the PMC prefix must be checked before the bare-digit PMID rule)."""
    doi = normalize_doi(raw)
    if doi:
        return "doi", doi
    ax = normalize_arxiv(raw)
    if ax:
        return "arxiv", ax
    pc = normalize_pmcid(raw)
    if pc:
        return "pmcid", pc
    pm = normalize_pmid(raw)
    if pm:
        return "pmid", pm
    return "unknown", None


# --- DOI prefix comparison (kept compatible with zotero_match.doi_match) ----

def doi_prefix_agree(a: str, b: str, min_prefix: int = 8) -> Optional[bool]:
    """Prefix comparison on CANONICAL DOIs. Returns None when either side is
    missing. A bare `10.xxxx/` registrant prefix is only ~8-11 chars, and
    journal-mate DOIs share much more ('10.1016/j.ydbio.' = 18 chars), so a
    fixed-char prefix can never separate same-journal papers. The safe rule:
      - identical (after version-suffix strip) → agree;
      - shared prefix ending exactly at a '/' boundary (truncated string,
        e.g. metadata cut at 20 chars) → agree;
      - anything else (differing item id) → disagree.
    """
    na, nb = normalize_doi(a), normalize_doi(b)
    if not na or not nb:
        return None
    if na == nb:
        return True
    body_a = re.sub(r"(?<=\d)[._-]?v\d+$", "", na)
    body_b = re.sub(r"(?<=\d)[._-]?v\d+$", "", nb)
    if body_a == body_b:
        return True
    # truncated metadata: the shorter DOI is a strict PREFIX of the longer one
    # AND the cut lands at a segment boundary ('...2021.01' vs '...2021.01.002')
    short, long_ = (body_a, body_b) if len(body_a) <= len(body_b) else (body_b, body_a)
    if long_.startswith(short) and (len(short) >= max(min_prefix, 8)):
        return True
    # otherwise a shared mid-item prefix means DIFFERENT papers
    return False