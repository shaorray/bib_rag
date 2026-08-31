#!/usr/bin/env python3
"""meta_audit.py — Paper metadata proof-reader and genuine-info fetcher.

Validates every paper's `meta` block in parent_store/*.json against multiple
ground-truth sources, and for papers that fail proof-reading, fetches the
genuine metadata (DOI, title, authors, year, journal, PMID/PMCID) into a
concrete `suggested_fix` payload. Dry-run by default; --apply writes back.

Ground-truth sources (priority order):
  1. My Library.bib (BIB_RAG_BIB_PATH env / --bib flag) — canonical Zotero export (title, DOI, authors, year, journal, key)
  2. Zotero local API (:23119) — authoritative for this user's library (optional, graceful)
  3. Crossref                  — DOI verification + title/author/year search to find genuine DOI
  4. PubMed E-utilities        — PMID/PMCID backfill + biomedical corroboration
  5. OpenAlex                  — secondary cross-check for DOI existence + title similarity

Pass criteria (Strict):
  - DOI resolves on Crossref (200 with a title)
  - Title Jaccard similarity >= 0.85 between meta.title (stripped) and Crossref title
  - |meta.year - crossref_year| <= 1

Output (consistent with existing repo conventions):
  - outputs/meta_audit_report_<DATE>.csv    per-paper results
  - data/meta_audit_summary_<DATE>.json     machine-readable summary
  - data/meta_audit_log_<DATE>.json         detailed evidence trace
  - data/meta_audit_report_<DATE>.md        human-readable report

Usage:
  python3 scripts/meta_audit.py                       # full audit, dry-run
  python3 scripts/meta_audit.py --limit 10             # first 10 files (test)
  python3 scripts/meta_audit.py --apply               # write fixes (implies backup)
  python3 scripts/meta_audit.py --resume              # skip already-audited files
  python3 scripts/meta_audit.py --offline             # .bib + Zotero only, no external APIs
  python3 scripts/meta_audit.py --no-pubmed --no-openalex   # selective source disable
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys

import os

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ (bib_utils, zotero_access)
from kb_config import get_config
from library_config import get_setting as _lib_setting

_CFG = get_config()

# ---------------------------------------------------------------------------
# Defaults (overridable via CLI)
# ---------------------------------------------------------------------------

DEFAULT_PARENT_DIR = Path(_CFG["parent_store_dir"])
DEFAULT_BIB_PATH = Path(_lib_setting(_CFG["data_root"], "bib_path", "My Library.bib"))
DEFAULT_ZOTERO_URL = "http://localhost:23119"
DEFAULT_CROSSREF_MAILTO = "bib-rag@example.com"
DEFAULT_DATA_DIR = Path(_CFG["data_dir"])
DEFAULT_OUT_DIR = Path(_CFG["outputs_dir"])

DATE_TAG = datetime.now().strftime("%Y-%m-%d")

# Pass/fail thresholds
TITLE_JACCARD_PASS = 0.85      # strict pass
TITLE_JACCARD_MATCH = 0.80     # threshold for accepting a search-result match
YEAR_TOLERANCE_PASS = 1        # |year diff| <= 1 passes
YEAR_TOLERANCE_SEARCH = 2      # search window for crossref/pubmed

# Filename-identity thresholds (blind-spot fix 2026-08-29)
IDENTITY_TITLE_PASS = 0.30     # fname-title vs meta-title Jaccard below this = identity mismatch
ORACLE_TITLE_PASS = 0.30       # candidate title vs fname-title Jaccard floor (fix validation)

# Rate limits (seconds between requests)
CROSSREF_MIN_INTERVAL = 1.0 / 50   # polite pool allows 50/s; be conservative
PUBMED_MIN_INTERVAL = 0.34         # 3 req/s without API key
OPENALEX_MIN_INTERVAL = 0.1         # 10 req/s polite

HTTP_TIMEOUT = 10

log = logging.getLogger("meta_audit")


# ---------------------------------------------------------------------------
# Shared helpers live in bib_utils.py (normalize, filename_to_key, bib parsing)
# ---------------------------------------------------------------------------

from bib_utils import (
    BibIndex, extract_abstract, extract_year_from_content, filename_to_key,
    is_doi_like, is_fake_doi, jaccard, normalize, normalize_doi,
    parse_bib_entries, strip_author_year_prefix, title_tokens,
)

import zotero_access  # noqa: E402  (scripts/ sibling; MCP-first Zotero access)


# ---------------------------------------------------------------------------
# Filename identity + authors quality (blind-spot fix 2026-08-29)
#
# The audit previously verified meta INTERNAL consistency only (DOI↔Crossref),
# so (a) a missing/initials-only `authors` field never failed a file, and
# (b) a file whose meta block belonged to a DIFFERENT paper (Zotero export
# swap) passed as long as that wrong paper's DOI resolved. Both are now
# checked: the parent_store FILENAME (author-year-title, exported from Zotero)
# is the identity oracle.
# ---------------------------------------------------------------------------

# an author "part" is real if it contains a run of >= 2 unicode letters
_AUTHOR_PART_RE = re.compile(r"[^\W\d_]{2,}", re.UNICODE)


def authors_quality(authors) -> str:
    """Classify a meta.authors string: ok | missing | initials_only | partial.

    - 'T', 'M; B', ','        → initials_only (no part has a real name)
    - 'Doe,Jane; , , ,'       → partial (some parts are garbage)
    - 'St-Laurent, Martin-Hugues' → ok (hyphenated surnames are fine)
    """
    if not authors or not str(authors).strip():
        return "missing"
    parts = [p.strip() for p in str(authors).split(";") if p.strip()]
    if not parts:
        return "missing"
    bad = sum(1 for p in parts if not _AUTHOR_PART_RE.search(p))
    if bad == 0:
        return "ok"
    if bad == len(parts):
        return "initials_only"
    return "partial"


def _filename_identity(fname: str) -> Optional[Tuple[str, str, str, str]]:
    """Parse a parent_store filename → (surname_norm, year, title_segment).

    Handles both libraries' filename shapes:
      - eph:  `Wan_et_al__-_2019_-_LTMG_a_novel_..._md`
      - geo:  `Ali_等_-_2017_-_Part-load_..._md`, `Fajardy和Mac_Dowell_-_2020_-_...`,
              `RE507_-_Wagner_等_-_2019_-_...` (legacy RE### prefix),
              `2025_-_Sustainable_..._md` (year-first: no surname),
              `10_1111_2F..._md` (DOI-shaped: no identity → None)

    Returns None only for DOI-shaped stems; year-first stems yield
    ('', year, title). Surname is the FIRST author's (et_al / 等 / 和 / and
    separators split off); title segment is everything after the year.
    Returns (surname_norm, year, title_segment, surname_raw) — the raw
    surname keeps compound-separator characters for registry author search.
    """
    stem = re.sub(r"\.json$", "", fname)
    stem = re.sub(r"_md$", "", stem)
    stem = re.sub(r"^RE\d+_?-*_?", "", stem)      # legacy geo RE### prefix
    if re.match(r"^10_\d", stem):                 # DOI-shaped: no identity
        return None
    ym = re.search(r"(19\d{2}|20[0-3]\d)", stem)
    if not ym:
        return None
    year = ym.group(1)
    pre = stem[: ym.start()].strip("_- ")
    tseg = stem[ym.end():].strip("_- ").replace("_", " ").strip("- ")
    surname = ""
    surname_raw = ""
    if pre and not re.match(r"^\d+$", pre):
        # separators may lack the trailing underscore (`Wan_et_al`, `Ali_等`)
        pre2 = re.split(r"_et_al_?|_等_?|和|_and_?", pre)[0].strip("_- ")
        if pre2:
            surname = normalize(pre2)
            surname_raw = pre2
    return (surname, year, tseg, surname_raw)


_JUNK_TITLE_RE = [
    r"^https?://", r"^www\.", r"^pubs\.acs\.org", r"^10\.\d",
    r"^(research|review|original|rapid)\s+(article|paper|communication|communication)\s*$",
    r"^received:.*accepted", r"^editorial board", r"^contents",
    r"^full length article", r"^open access", r"^article\s*$",
    r"^online access available", r"^sciencedirect", r"^elsevier",
    r"^this article has been retracted", r"^retraction",
    r"^table\s+\d", r"^fig(ure)?\.?\s*\d",
    r"^answers?\s+to", r"^accepted (author )?manuscript",
    r"^author['’]?s accepted manuscript",
]


def _pmid_from_filename(fname: str) -> Optional[str]:
    """Filename IS a PMID (eph library: `29031557_md.json`, `10068468_md.json`).

    Zotero exported these items with the PubMed ID as the item name — the
    strongest possible identity oracle for the file: one esummary call names
    the exact paper. Returns None for author+year / DOI / descriptive stems.
    """
    m = re.match(r"^(\d{6,9})(?:_md)?\.json$", fname)
    return m.group(1) if m else None


def _doi_from_filename(fname: str) -> List[str]:
    """Filename IS a DOI (geo library: `10_1111_2Fgcb_12437_md.json`).

    Zotero's filesystem export escapes `.`→`_` and `/`→`_2F` (percent-style,
    2F = hex for `/`); some exports omit the `2F` marker and collapse the
    slash to a plain `_`. Returns candidate DOIs in priority order — the
    caller verifies each against Crossref and the one that resolves is the
    paper. Empty for non-DOI stems.
    """
    stem = re.sub(r"\.json$", "", fname)
    stem = re.sub(r"_md$", "", stem)
    if not re.match(r"^10_", stem):
        return []
    if "2F" in stem:
        return [stem.replace("_2F", "/").replace("_", ".")]
    # no 2F marker: first `_` separates `10` from the registrant code, the
    # next `_` is the suffix slash (`10_1038_s41597...` → `10.1038/s41597...`)
    parts = stem.split("_", 2)
    if len(parts) == 3:
        return [f"{parts[0]}.{parts[1]}/{parts[2].replace('_', '.')}"]
    return []


def extract_title_from_content(data: list) -> str:
    """Pull a plausible paper title out of the document head.

    Used when BOTH the meta title (junk noise) and the filename title segment
    (truncated away) are unusable as registry search keys — searching on the
    junk title self-confirms (the search finds the paper the junk text
    belongs to). The content's first markdown heading is the real title far
    more often than not.
    """
    _SECTION_HEAD_RE = re.compile(
        r"^(\d+[\.\)]?\s+)?"
        r"(material and methods?|methods?|results?|introduction|abstract|"
        r"discussion|conclusions?|references?|experimental( procedures|"
        r" section)?|background|related work|acknowledg(e)?ments?|"
        r"supplementary (materials?|data|information)|abstracts?|"
        r"author contributions?|declaration of interests?|funding|"
        r"supporting information|appendix|glossary|abbreviations?)\s*$",
        re.IGNORECASE,
    )
    head_lines: List[str] = []
    for sec in data[:3]:
        head_lines.extend(str(sec.get("content", "") or "").split("\n"))

    def _plausible(s: str) -> bool:
        t = re.sub(r"\*+", "", s).strip()
        if len(title_tokens(t)) < 3:
            return False
        return not _SECTION_HEAD_RE.match(t)

    # pass 1: markdown headings (`## **Title**`, `# Title`)
    for ln in head_lines:
        s = ln.strip()
        if not s:
            continue
        m = re.match(r"^#{1,6}\s*\*{0,2}(.+?)\*{0,2}\s*$", s)
        if m and _plausible(m.group(1)):
            return m.group(1).strip()
    # pass 2: standalone bold lines
    for ln in head_lines:
        s = ln.strip()
        if not s:
            continue
        m = re.match(r"^\*\*(.+)\*\*$", s)
        if m and _plausible(m.group(1)):
            return m.group(1).strip()
    return ""


def is_junk_title(title: str) -> bool:
    """True if a meta title carries no real bibliographic signal (journal
    header noise, submission timestamps, URLs) — the filename title should be
    used for registry search instead."""
    if not title or not str(title).strip():
        return True
    s = re.sub(r"[*_`#]", "", str(title)).strip()
    if len(title_tokens(s)) < 3:
        return True
    low = s.lower()
    return any(re.match(p, low) for p in _JUNK_TITLE_RE)


def _content_supports_identity(data: list, f_title: str) -> float:
    """Fraction of filename-title tokens found in the document head.

    High → the indexed content IS the filename's paper (the meta block was
    swapped/stolen → fixable by re-fetching genuine meta).
    Low → the content is a different paper entirely (the md document itself
    is wrong → needs re-index, no meta fix can help).

    Plural-folded: 'cadherin' matches content 'cadherins' (else singular
    paper titles lose one token per plural occurrence and marginal cases
    flip to content_swap).
    """
    if not f_title:
        return 0.0
    ft = title_tokens(f_title[:60])
    if not ft:
        return 0.0
    head = " ".join(str(sec.get("content", "") or "")[:2500] for sec in data[:3])
    head = unicodedata.normalize("NFKD", head.lower())
    head = "".join(c for c in head if not unicodedata.combining(c))
    head_toks = set(re.findall(r"[a-z0-9]+", head))
    head_folded = head_toks | {t[: -1] for t in head_toks if t.endswith("s")}
    hits = sum(1 for t in ft if t in head_folded)
    return hits / len(ft)


def _identity_assessment(fname: str, meta: Dict[str, Any],
                         data: Optional[list] = None) -> Dict[str, Any]:
    """Shared local identity check (used by audit_file AND --only-defects).

    Compares the parent_store FILENAME identity (surname, year, title —
    exported from Zotero, trustworthy) against the meta block, and classifies
    authors quality. Returns a dict with:
      aq, identity_mismatch, meta_swap, content_swap, identity_flags, oracle,
      tsim, support, fname_title
    """
    aq = authors_quality(meta.get("authors", ""))
    out = {
        "aq": aq, "identity_mismatch": False, "meta_swap": False,
        "content_swap": False, "identity_flags": [], "oracle": None,
        "tsim": 1.0, "support": 0.0, "fname_title": "",
    }
    fident = _filename_identity(fname)
    if fident is None:
        return out
    f_sur, f_year, f_title, f_sur_raw = fident
    out["fname_title"] = f_title
    if f_sur or f_title:
        out["oracle"] = (f_sur, f_year, f_title, f_sur_raw)

    m_sur = ""
    if meta.get("authors"):
        first = str(meta["authors"]).split(";")[0]
        m_sur = normalize(first.split(",")[0].strip())
    sur_mm = bool(f_sur) and bool(m_sur) and f_sur != m_sur
    year_mm = False
    try:
        my = str(meta.get("year", "") or "").strip()
        year_mm = bool(my) and abs(int(f_year) - int(my)) > 2
    except (TypeError, ValueError):
        year_mm = False

    clean_t = strip_author_year_prefix(meta.get("title", "") or "")
    ft_toks = title_tokens(f_title[:60])
    mt_toks = title_tokens((clean_t or meta.get("title", "") or "")[:60])
    tsim = jaccard(ft_toks, mt_toks) if (ft_toks and mt_toks) else 1.0
    out["tsim"] = round(tsim, 3)

    if (len(ft_toks) >= 3 and len(mt_toks) >= 3
            and tsim < IDENTITY_TITLE_PASS and (sur_mm or year_mm)):
        out["identity_mismatch"] = True
        flags = [f"identity:title_sim={tsim:.2f}"]
        if sur_mm:
            flags.append(f"identity:surname({f_sur}!={m_sur})")
        if year_mm:
            flags.append(f"identity:year({f_year}!={meta.get('year')})")
        support = _content_supports_identity(data or [], f_title)
        out["support"] = round(support, 2)
        # meta_swap (fixable): content clearly IS the filename paper.
        # 3-token titles need a higher support bar (2/3 tokens = 0.67) since
        # one accidental token hit is 0.33; >=4-token titles trust 0.35.
        # meta_swap is the safer error: a wrong fetch candidate is rejected
        # by the filename oracle → plain fail, nothing written.
        if (len(ft_toks) >= 4 and support >= 0.35) or (
                len(ft_toks) == 3 and support >= 0.67):
            out["meta_swap"] = True
            flags.append(f"identity:meta_swap(content_support={support:.2f})")
        else:
            out["content_swap"] = True
            flags.append(f"identity:content_swap(content_support={support:.2f})")
        out["identity_flags"] = flags
    return out


def _title_search_sim(claimed_title: str, result_title: str) -> float:
    """Similarity between a claimed title and a registry result title.

    PREFIX-RECALL semantics, shared by search-time best-pick AND the
    confidence gate: filenames truncate at 255 bytes, so a perfect registry
    match reads ~0.67 under plain Jaccard. Recall (claimed tokens covered by
    the result) stays 1.0 under pure truncation; a wrong paper loses recall.
    The symmetric precision floor (>= 0.5) guards short-result false hits.
    """
    ct = title_tokens((claimed_title or "")[:60])
    rt = title_tokens((result_title or "")[:60])
    if not ct or not rt:
        return 0.0
    inter = len(ct & rt)
    recall = inter / len(ct)
    precision = inter / len(rt)
    if recall >= TITLE_JACCARD_MATCH and precision >= 0.5:
        return recall
    return jaccard(ct, rt)


def _oracle_rejects(fix: Dict[str, str], oracle) -> bool:
    """True if a suggested fix belongs to a DIFFERENT paper than the filename.

    oracle: (surname_norm, year, title, surname_raw?) — accepts both the
    4-tuple from _identity_assessment and legacy 3-tuples.

    Gates (each applied only when the oracle side is non-empty):
      - title Jaccard(fix.title, fname title) >= ORACLE_TITLE_PASS
      - |fix.year - fname year| <= 2
      - fix first-author surname == fname surname, unless the title match is
        strong (>= 0.50) — journal-first filenames carry the journal name
        where the surname should be.
    """
    o_sur = oracle[0]
    o_year = oracle[1]
    o_title = oracle[2]
    if not fix.get("title"):
        return True
    if o_title:
        ot = title_tokens(o_title[:60])
        ft = title_tokens(str(fix["title"])[:60])
        if ot and ft:
            tsim = jaccard(ot, ft)
            if tsim < ORACLE_TITLE_PASS:
                return True
            if o_sur and fix.get("authors"):
                fs = normalize(str(fix["authors"]).split(";")[0].split(",")[0])
                if fs and fs != o_sur and tsim < 0.50:
                    # compound-surname containment: registries may report
                    # only one segment (`Guitart` vs filename glue
                    # `bussoniguitart`) — treat containment as a match.
                    if fs not in o_sur and o_sur not in fs:
                        return True
    if o_year and fix.get("year"):
        try:
            if abs(int(o_year) - int(str(fix["year"]).strip())) > 2:
                return True
        except (TypeError, ValueError):
            pass
    return False


def _local_defect_check(fp: Path) -> bool:
    """Cheap local pre-scan (no network): does this file need the full audit?

    True when authors are missing/initials/partial, the title is junk noise,
    or the filename identity disagrees with the meta block.
    """
    try:
        data = json.loads(fp.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return True
    if not isinstance(data, list) or not data:
        return True
    meta = data[0].get("meta", {}) or {}
    if not isinstance(meta, dict):
        return True
    ident = _identity_assessment(fp.name, meta, data)
    if ident["aq"] != "ok":
        return True
    if ident["identity_mismatch"]:
        return True
    if is_junk_title(meta.get("title", "")):
        f_title = ident["fname_title"]
        if not f_title or len(title_tokens(f_title[:60])) >= 4:
            return True
    return False


# ---------------------------------------------------------------------------
# API clients (urllib-based; graceful degradation)
# ---------------------------------------------------------------------------

class CrossrefClient:
    """Crossref API client. DOI verification + bibliographic search."""

    def __init__(self, mailto: str, enabled: bool = True):
        self.mailto = mailto
        self.enabled = enabled
        self._last = 0.0
        self.calls = 0
        self.errors = 0

    def _throttle(self):
        now = time.time()
        wait = CROSSREF_MIN_INTERVAL - (now - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.time()

    def _get(self, url: str) -> Optional[dict]:
        if not self.enabled:
            return None
        self._throttle()
        req = urllib.request.Request(url)
        req.add_header("User-Agent", f"bib_rag_audit/1.0 (mailto:{self.mailto})")
        try:
            self.calls += 1
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
                return json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, TimeoutError) as ex:
            self.errors += 1
            log.debug("crossref GET %s failed: %s", url, ex)
            return None

    def verify_doi(self, doi: str) -> Optional[Dict[str, Any]]:
        """Verify a DOI resolves. Returns {doi, title, year, journal, authors} or None."""
        nd = normalize_doi(doi)
        if not nd:
            return None
        url = f"https://api.crossref.org/works/{urllib.parse.quote(nd)}?mailto={self.mailto}"
        d = self._get(url)
        if not d or d.get("status") != "ok":
            return None
        m = d.get("message", {})
        title = (m.get("title") or [""])[0]
        year = (m.get("issued", {}).get("date-parts") or [[None]])[0][0]
        journal = (m.get("container-title") or [""])[0]
        authors = "; ".join(
            f"{a.get('family','')},{a.get('given','')}"
            for a in m.get("author", [])
        )
        return {
            "doi": nd,
            "title": title,
            "year": str(year) if year else "",
            "journal": journal,
            "authors": authors,
        }

    def search(self, title: str, author_lastname: str = "",
               year: str = "") -> List[Dict[str, Any]]:
        """Search Crossref by bibliographic query. Returns top-5 results."""
        if not self.enabled or not title:
            return []
        params = {"query.bibliographic": title, "rows": "5", "mailto": self.mailto}
        if author_lastname:
            params["query.author"] = author_lastname
        if year:
            try:
                y = int(year)
                params["filter"] = f"from-pub-date:{y-YEAR_TOLERANCE_SEARCH},until-pub-date:{y+YEAR_TOLERANCE_SEARCH}"
            except ValueError:
                pass
        url = "https://api.crossref.org/works?" + urllib.parse.urlencode(params)
        d = self._get(url)
        if not d or d.get("status") != "ok":
            return []
        out = []
        for it in d.get("message", {}).get("items", []):
            t = (it.get("title") or [""])[0]
            y = (it.get("issued", {}).get("date-parts") or [[None]])[0][0]
            j = (it.get("container-title") or [""])[0]
            a = "; ".join(
                f"{x.get('family','')},{x.get('given','')}"
                for x in it.get("author", [])
            )
            out.append({
                "doi": it.get("DOI", ""),
                "title": t,
                "year": str(y) if y else "",
                "journal": j,
                "authors": a,
                "type": it.get("type", ""),
            })
        return out


class PubmedClient:
    """PubMed E-utilities client. PMID/PMCID backfill + corroboration."""

    def __init__(self, api_key: str = "", enabled: bool = True):
        self.api_key = api_key
        self.enabled = enabled
        self._last = 0.0
        self.calls = 0
        self.errors = 0

    def _throttle(self):
        interval = 0.1 if self.api_key else PUBMED_MIN_INTERVAL
        now = time.time()
        wait = interval - (now - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.time()

    def _get_xml(self, url: str) -> Optional[ET.Element]:
        if not self.enabled:
            return None
        self._throttle()
        try:
            self.calls += 1
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
                return ET.parse(r).getroot()
        except (urllib.error.URLError, ET.ParseError, TimeoutError) as ex:
            self.errors += 1
            log.debug("pubmed GET %s failed: %s", url, ex)
            return None

    def search(self, title: str, author: str = "") -> List[str]:
        """Search PubMed by title (+ optional author). Returns list of PMIDs."""
        if not self.enabled or not title:
            return []
        # Use title as a phrase query for precision
        term = f"{title}[Title]"
        if author:
            term += f" AND {author}[Author]"
        params = {
            "db": "pubmed", "term": term, "retmax": "3",
            "tool": "bib_rag_audit", "email": "bib-rag@example.com",
        }
        if self.api_key:
            params["api_key"] = self.api_key
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + urllib.parse.urlencode(params)
        root = self._get_xml(url)
        if root is None:
            return []
        return [id_elem.text for id_elem in root.findall(".//Id") if id_elem.text]

    def summary(self, pmid: str) -> Optional[Dict[str, Any]]:
        """Fetch PubMed summary for a PMID."""
        params = {
            "db": "pubmed", "id": pmid,
            "tool": "bib_rag_audit", "email": "bib-rag@example.com",
        }
        if self.api_key:
            params["api_key"] = self.api_key
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?" + urllib.parse.urlencode(params)
        root = self._get_xml(url)
        if root is None:
            return None
        doc = root.find(".//DocSum")
        if doc is None:
            return None
        title = doc.findtext('Item[@Name="Title"]') or ""
        pubdate = doc.findtext('Item[@Name="PubDate"]') or ""
        journal = doc.findtext('Item[@Name="Source"]') or ""
        year = ""
        m = re.search(r"(19\d{2}|20[0-3]\d)", pubdate)
        if m:
            year = m.group(1)
        authors = []
        author_list = doc.find('Item[@Name="AuthorList"]')
        if author_list is not None:
            authors = [a.text or "" for a in author_list.findall("Item")]
        doi = ""
        pmcid = ""
        for aid in doc.findall('.//Item[@Name="ArticleIds"]/*'):
            name = aid.get("Name", "")
            if name == "doi":
                doi = aid.text or ""
            elif name == "pmc":
                pmcid = (aid.text or "").replace("pmc-id:", "").replace("PMC", "PMC").strip()
        return {
            "pmid": pmid,
            "pmcid": pmcid,
            "title": title,
            "year": year,
            "authors": "; ".join(authors),
            "journal": journal,
            "doi": doi,
        }


class OpenAlexClient:
    """OpenAlex API client. Secondary DOI verification + title search."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._last = 0.0
        self.calls = 0
        self.errors = 0

    def _throttle(self):
        now = time.time()
        wait = OPENALEX_MIN_INTERVAL - (now - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.time()

    def _get(self, url: str) -> Optional[dict]:
        if not self.enabled:
            return None
        self._throttle()
        try:
            self.calls += 1
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "bib_rag_audit/1.0")
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
                return json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as ex:
            self.errors += 1
            log.debug("openalex GET %s failed: %s", url, ex)
            return None

    def verify_doi(self, doi: str) -> Optional[Dict[str, Any]]:
        nd = normalize_doi(doi)
        if not nd:
            return None
        url = f"https://api.openalex.org/works/doi:{urllib.parse.quote(nd)}"
        d = self._get(url)
        if not d:
            return None
        title = d.get("title", "") or d.get("display_name", "") or ""
        year = d.get("publication_year", "")
        venue = (d.get("primary_location") or {}).get("source", {}) or {}
        journal = venue.get("display_name", "")
        authors = []
        for a in d.get("authorships", []):
            au = (a.get("author") or {})
            name = au.get("display_name", "")
            if name:
                authors.append(name)
        return {
            "doi": nd,
            "title": title,
            "year": str(year) if year else "",
            "journal": journal,
            "authors": "; ".join(authors),
        }

    def search(self, title: str) -> List[Dict[str, Any]]:
        if not self.enabled or not title:
            return []
        url = "https://api.openalex.org/works?search=" + urllib.parse.quote(title) + "&per-page=5"
        d = self._get(url)
        if not d:
            return []
        out = []
        for it in d.get("results", []):
            t = it.get("title", "") or it.get("display_name", "") or ""
            y = it.get("publication_year", "")
            venue = (it.get("primary_location") or {}).get("source", {}) or {}
            j = venue.get("display_name", "")
            dois = it.get("doi", "") or ""
            if dois:
                dois = dois.replace("https://doi.org/", "")
            out.append({
                "doi": dois,
                "title": t,
                "year": str(y) if y else "",
                "journal": j,
                "authors": "",
            })
        return out


class ZoteroClient:
    """Zotero corroboration source.

    Delegates to zotero_access (scripts/zotero_access.py): the Zotero MCP
    server (`zotero-mcp`) is the primary path, the local HTTP API the
    fallback. The `base_url` argument is kept for CLI/UX compatibility and
    logging only.
    """

    # circuit breakers: (a) 5 consecutive raised failures, or (b) an >=80%
    # failure rate after 10 calls (covers timeouts swallowed by zotero_access
    # into empty results — a hung desktop API must not add its 5s timeout to
    # every remaining paper). Empty results DO count as failures for (b):
    # a healthy Zotero serving this library answers most title searches.
    MAX_CONSECUTIVE_FAILURES = 5
    MIN_CALLS_FOR_RATE = 10
    FAILURE_RATE_TRIP = 0.8

    def __init__(self, base_url: str, enabled: bool = True):
        self.base_url = base_url.rstrip("/")
        self.enabled = enabled
        self.calls = 0
        self.errors = 0
        self.available = False
        self._consecutive_failures = 0
        self._tripped = False
        if enabled:
            self._check()

    def _check(self):
        try:
            self.available = zotero_access.available()
        except Exception:
            self.available = False
        if not self.available:
            log.info("zotero not available at %s — skipping", self.base_url)

    def _record_failure(self):
        self.errors += 1
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
            self._tripped = True
            log.info("zotero: %d consecutive failures — disabling for this run",
                     self._consecutive_failures)
        elif (self.calls >= self.MIN_CALLS_FOR_RATE
                and self.errors / self.calls >= self.FAILURE_RATE_TRIP):
            self._tripped = True
            log.info("zotero: %d/%d calls failed — disabling for this run",
                     self.errors, self.calls)

    def search(self, title: str) -> Optional[Dict[str, Any]]:
        if not self.enabled or not self.available or not title:
            return None
        if self._tripped:
            return None
        self.calls += 1
        try:
            items = zotero_access.zotero_search(title[:200], limit=3)
        except Exception as ex:
            self._record_failure()
            log.debug("zotero search failed: %s", ex)
            return None
        if not items:
            self._record_failure()
            return None
        self._consecutive_failures = 0
        # Pick the first item with a matching title
        ps = title_tokens(title)
        for it in items:
            t = it.get("title", "")
            if not t:
                continue
            if jaccard(title_tokens(t), ps) >= 0.70:
                full = zotero_access.zotero_item(it["key"]) or it
                return {
                    "key": full.get("key", ""),
                    "title": full.get("title", ""),
                    "doi": full.get("doi", ""),
                    "year": full.get("year", ""),
                    "authors": full.get("authors", ""),
                    "journal": full.get("journal", ""),
                }
        return None


# ---------------------------------------------------------------------------
# Auditor — orchestrates per-paper validation + fix-fetch
# ---------------------------------------------------------------------------

class Auditor:
    def __init__(
        self,
        bib_index: BibIndex,
        crossref: CrossrefClient,
        pubmed: PubmedClient,
        openalex: OpenAlexClient,
        zotero: ZoteroClient,
    ):
        self.bib = bib_index
        self.crossref = crossref
        self.pubmed = pubmed
        self.openalex = openalex
        self.zotero = zotero

    def audit_file(self, fp: Path) -> Dict[str, Any]:
        """Run the full audit pipeline on one parent_store file.

        Returns a result dict with keys:
          file, status (pass/fail/unverified), field_checks, claimed_meta,
          evidence, suggested_fix, confidence, sources_agreeing, notes
        """
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as ex:
            return {
                "file": fp.name, "status": "error",
                "error": str(ex), "field_checks": {},
                "claimed_meta": {}, "evidence": {}, "suggested_fix": None,
                "confidence": "none", "sources_agreeing": 0,
            }
        if not isinstance(data, list) or not data:
            return {
                "file": fp.name, "status": "error",
                "error": "not a non-empty list", "field_checks": {},
                "claimed_meta": {}, "evidence": {}, "suggested_fix": None,
                "confidence": "none", "sources_agreeing": 0,
            }
        meta = data[0].get("meta", {}) or {}
        claimed = {
            "title": meta.get("title", ""),
            "authors": meta.get("authors", ""),
            "year": meta.get("year", ""),
            "journal": meta.get("journal", ""),
            "doi": meta.get("doi", ""),
            "pmid": meta.get("pmid", ""),
            "pmcid": meta.get("pmcid", ""),
            "key": meta.get("key", ""),
        }

        # ---- Step 0: local identity + authors-quality assessment ----
        # The filename (exported from Zotero) is the identity oracle; a meta
        # block belonging to a different paper must fail even if its DOI
        # resolves. Junk titles (journal headers) carry no signal — use the
        # filename title as the search key instead.
        ident = _identity_assessment(fp.name, meta, data)
        authors_check = ident["aq"]
        oracle = ident["oracle"]
        fname_title = ident["fname_title"]
        pmid_gold = None
        fname_pmid = _pmid_from_filename(fp.name)
        if fname_pmid and self.pubmed.enabled:
            # PMID-shaped filename: one esummary call returns the GOLD record
            # for this exact paper — stronger than any title matching. No
            # oracle (pmid needs no fuzzy gates), no search key guessing.
            pmid_gold = self.pubmed.summary(fname_pmid)
        doi_gold = None
        if not fname_pmid and self.crossref.enabled:
            # DOI-shaped filename: the filename decodes to the paper's own
            # DOI — one verify call returns the GOLD record. Same authority
            # as pmid_gold: no oracle, no fuzzy gates.
            for cand_doi in _doi_from_filename(fp.name):
                resp = self.crossref.verify_doi(cand_doi)
                if resp and resp.get("title"):
                    doi_gold = dict(resp)
                    doi_gold["doi"] = cand_doi
                    break
        meta_title_junk = is_junk_title(claimed["title"])
        if meta_title_junk and fname_title:
            claimed["title"] = fname_title.replace("_", " ")
            clean_title = claimed["title"]
        elif meta_title_junk and not fname_title and not pmid_gold:
            # BOTH search keys unusable: the meta title is junk noise AND the
            # filename title segment was truncated away. Searching on the junk
            # title self-confirms (finds the paper the junk text belongs to —
            # retraction notices, CME answer pages). The document head's first
            # heading is the real title; require the result to be supported
            # by the content below. The claimed DOI comes from the SAME
            # untrustworthy block — clear it so the junk record is never
            # fetched, verified, and offered back as a "fix" candidate.
            # The filename itself is DESCRIPTIVE here (no title segment after
            # the year ⇒ not the Zotero Author_year_Title export shape), so
            # its pre-year segment is not a real surname either — a garbage
            # oracle surname would reject the CORRECT fix author, and a
            # garbage search hint (chicken_cadherin_spinal) would zero out
            # the Crossref results. Neutralize the whole filename identity.
            claimed["doi"] = ""
            oracle = None
            content_title = extract_title_from_content(data)
            if content_title:
                claimed["title"] = content_title
                clean_title = content_title
            else:
                clean_title = ""
        else:
            clean_title = strip_author_year_prefix(claimed["title"])
        if ident["identity_mismatch"] and not ident["meta_swap"]:
            # content itself is a different paper: no meta fix can help
            return {
                "file": fp.name, "status": "fail",
                "field_checks": {
                    "doi": "content_swap",
                    "title": "content_swap",
                    "year": "content_swap",
                    "authors": authors_check,
                },
                "identity_flags": ident["identity_flags"],
                "claimed_meta": claimed, "clean_title": "",
                "evidence": {}, "suggested_fix": None,
                "confidence": "none", "sources_agreeing": 0,
                "needs_reindex": True,
            }
        if ident["identity_mismatch"] and ident["meta_swap"]:
            # meta block stolen by another paper but content is the right one:
            # drop the stolen values so no registry verifies the WRONG doi,
            # and the stolen year does not narrow the search window off-target.
            claimed["doi"] = ""
            if oracle:
                claimed["year"] = oracle[1]
            if fname_title:
                claimed["title"] = fname_title.replace("_", " ")
                clean_title = claimed["title"]
            else:
                clean_title = strip_author_year_prefix(claimed["title"])
        fname_key = filename_to_key(fp.name)
        if oracle is None and meta_title_junk and not fname_title and not pmid_gold:
            # descriptive filename (junk-title branch neutralized the oracle
            # above): its pre-year segment is topic words, not a surname.
            fname_key = None
        if (fname_key and oracle and oracle[0] and fname_key[0]
                and not fname_key[0].isascii()):
            # geo-style filenames parse with CJK separators baked into the
            # lastname (`ali_等_-`) — the identity oracle carries the clean
            # surname+year; use it so registry search hints stay clean.
            fname_key = (oracle[0], oracle[1], normalize((oracle[2] or "")[:50]))
        if fname_key and not fname_key[1]:
            # filename lacks year — try from content
            y = extract_year_from_content(data)
            if y:
                fname_key = (fname_key[0], y, fname_key[2])
        paper_abstract = extract_abstract(data)
        paper_abstract_norm = normalize(paper_abstract[:200]) if paper_abstract else ""

        # ---- Step 1: DOI verification on Crossref ----
        crossref_resp = None
        openalex_resp = None
        doi_field_check = "missing"
        if claimed["doi"]:
            if is_fake_doi(claimed["doi"]):
                doi_field_check = "fake_pattern"
            else:
                if self.crossref.enabled:
                    crossref_resp = self.crossref.verify_doi(claimed["doi"])
                    if crossref_resp:
                        doi_field_check = "resolves"
                    else:
                        # Fallback to OpenAlex for verification
                        if self.openalex.enabled:
                            openalex_resp = self.openalex.verify_doi(claimed["doi"])
                            doi_field_check = "resolves_oa" if openalex_resp else "not_found"
                        else:
                            doi_field_check = "not_found"
                elif self.openalex.enabled:
                    openalex_resp = self.openalex.verify_doi(claimed["doi"])
                    doi_field_check = "resolves_oa" if openalex_resp else "not_found"
                else:
                    doi_field_check = "unverified"
        # ---- Title + year checks against Crossref (if DOI resolved) ----
        title_field_check = "no_ref"
        year_field_check = "no_ref"
        if crossref_resp:
            cr_title = crossref_resp["title"]
            claimed_title_tokens = title_tokens(clean_title or claimed["title"])
            cr_title_tokens = title_tokens(cr_title)
            title_sim = jaccard(claimed_title_tokens, cr_title_tokens)
            if title_sim >= TITLE_JACCARD_PASS:
                title_field_check = "match"
            else:
                title_field_check = f"mismatch(sim={title_sim:.2f})"

            try:
                cr_year = int(crossref_resp["year"]) if crossref_resp["year"] else None
                cl_year = int(claimed["year"]) if claimed["year"] else None
                if cr_year is None or cl_year is None:
                    year_field_check = "missing_year"
                elif abs(cl_year - cr_year) <= YEAR_TOLERANCE_PASS:
                    year_field_check = "match"
                else:
                    year_field_check = f"mismatch(|{cl_year - cr_year}|)"
            except ValueError:
                year_field_check = "bad_year"
        elif openalex_resp:
            # Use OpenAlex as the reference if Crossref unavailable
            oa_title = openalex_resp["title"]
            claimed_title_tokens = title_tokens(clean_title or claimed["title"])
            oa_title_tokens = title_tokens(oa_title)
            title_sim = jaccard(claimed_title_tokens, oa_title_tokens)
            if title_sim >= TITLE_JACCARD_PASS:
                title_field_check = "match_oa"
            else:
                title_field_check = f"mismatch_oa(sim={title_sim:.2f})"
            try:
                oa_year = int(openalex_resp["year"]) if openalex_resp["year"] else None
                cl_year = int(claimed["year"]) if claimed["year"] else None
                if oa_year is None or cl_year is None:
                    year_field_check = "missing_year"
                elif abs(cl_year - oa_year) <= YEAR_TOLERANCE_PASS:
                    year_field_check = "match_oa"
                else:
                    year_field_check = f"mismatch_oa(|{cl_year - oa_year}|)"
            except ValueError:
                year_field_check = "bad_year"

        # ---- Offline mode: verify against the .bib index instead ----
        # When no external verifier is enabled, the local .bib is the ground
        # truth: a paper whose claimed DOI/title/year match a bib entry passes.
        offline_ref = None
        if not self.crossref.enabled and not self.openalex.enabled:
            if claimed["doi"] and not is_fake_doi(claimed["doi"]):
                offline_ref = self.bib.by_doi(claimed["doi"])
            if offline_ref is None and clean_title:
                offline_ref, _st = self.bib.by_title(
                    clean_title, claimed["year"] or "",
                    paper_abstract_norm=paper_abstract_norm,
                )
            if offline_ref is not None:
                bib_title = offline_ref.title or ""
                bib_year = offline_ref.year or ""
                bib_doi = normalize_doi(offline_ref.doi or "")
                if claimed["doi"] and normalize_doi(claimed["doi"]) == bib_doi:
                    doi_field_check = "match_bib"
                elif claimed["doi"]:
                    doi_field_check = "mismatch_bib"
                else:
                    doi_field_check = "missing"
                claimed_title_tokens = title_tokens(clean_title or claimed["title"])
                bib_title_tokens = title_tokens(bib_title)
                sim = jaccard(claimed_title_tokens, bib_title_tokens)
                if sim >= TITLE_JACCARD_PASS:
                    title_field_check = "match_bib"
                else:
                    title_field_check = f"mismatch_bib(sim={sim:.2f})"
                try:
                    by = int(bib_year) if bib_year else None
                    cy = int(claimed["year"]) if claimed["year"] else None
                    if by is None or cy is None:
                        year_field_check = "missing_year"
                    elif abs(cy - by) <= YEAR_TOLERANCE_PASS:
                        year_field_check = "match_bib"
                    else:
                        year_field_check = f"mismatch_bib(|{cy - by}|)"
                except ValueError:
                    year_field_check = "bad_year"

        field_checks = {
            "doi": doi_field_check,
            "title": title_field_check,
            "year": year_field_check,
            "authors": authors_check,
        }

        # ---- Determine pass/fail ----
        doi_ok = doi_field_check in ("resolves", "resolves_oa", "match_bib")
        title_ok = title_field_check in ("match", "match_oa", "match_bib")
        year_ok = year_field_check in ("match", "match_oa", "match_bib")
        authors_ok = authors_check == "ok"
        identity_ok = not ident["identity_mismatch"]
        if offline_ref is not None:
            # .bib is the ground truth in offline mode
            status = "pass" if (doi_ok and title_ok and year_ok
                                and authors_ok and identity_ok) else "fail"
        elif (not self.crossref.enabled and not self.openalex.enabled
                and not claimed["doi"]):
            status = "unverified"
        elif doi_ok and title_ok and year_ok and authors_ok and identity_ok:
            status = "pass"
        else:
            status = "fail"

        # ---- Step 2: For failures, fetch genuine info ----
        suggested_fix = None
        confidence = "none"
        sources_agreeing = 0
        evidence: Dict[str, Any] = {"crossref": crossref_resp, "openalex": openalex_resp}
        fix: Optional[Dict[str, str]] = None

        if status == "fail":
            gold = pmid_gold or doi_gold
            if gold:
                # Gold record from the PMID/DOI filename — no search, no
                # fuzzy matching, no oracle gate. This IS the paper.
                suggested_fix = {
                    "doi": gold.get("doi", ""),
                    "title": gold.get("title", ""),
                    "authors": gold.get("authors", ""),
                    "year": gold.get("year", ""),
                    "journal": gold.get("journal", ""),
                    "pmid": (pmid_gold or {}).get("pmid", ""),
                    "pmcid": (pmid_gold or {}).get("pmcid", ""),
                    "key": "",
                }
                confidence = "high"
                sources_agreeing = 2   # filename ID + registry record
                if pmid_gold:
                    evidence["pubmed_pmid_gold"] = pmid_gold
                else:
                    evidence["crossref_doi_gold"] = doi_gold
            else:
                fix, conf, n = self._fetch_genuine(
                    claimed, clean_title, fname_key, paper_abstract_norm,
                    crossref_resp, openalex_resp, oracle=oracle,
                )
                suggested_fix = fix
                confidence = conf
                sources_agreeing = n
            if fix and ident["identity_mismatch"] and not gold:
                # the mismatch itself is the defect being fixed — a fix that
                # agrees with the filename oracle RESOLVES the identity flag
                identity_ok = True

        return {
            "file": fp.name,
            "status": status,
            "field_checks": field_checks,
            "identity_flags": ident["identity_flags"],
            "needs_reindex": ident["content_swap"],
            "pmid_gold": bool(pmid_gold),
            "doi_gold": bool(doi_gold),
            "claimed_meta": claimed,
            "clean_title": clean_title,
            "evidence": evidence,
            "suggested_fix": suggested_fix,
            "confidence": confidence,
            "sources_agreeing": sources_agreeing,
        }

    def _fetch_genuine(
        self,
        claimed: Dict[str, str],
        clean_title: str,
        fname_key: Optional[Tuple[str, str, str]],
        paper_abstract_norm: str,
        crossref_resp: Optional[Dict[str, Any]],
        openalex_resp: Optional[Dict[str, Any]],
        oracle: Optional[Tuple[str, str, str, str]] = None,
    ) -> Tuple[Optional[Dict[str, str]], str, int]:
        """For a failing paper, gather genuine metadata from all sources.

        Returns (suggested_fix payload or None, confidence, sources_agreeing).

        oracle: (surname_norm, year, title, surname_raw) parsed from the
        parent_store filename — the identity oracle. When present, candidate
        fixes that describe a DIFFERENT paper than the filename are rejected
        outright (_oracle_rejects): a swapped meta block must not be replaced
        by ANOTHER wrong paper's data.
        """
        # Collect candidate fixes from each source, keyed by normalized DOI
        candidates: Dict[str, Dict[str, Any]] = {}
        # Track per-source agreement on DOI
        source_dois: Dict[str, str] = {}

        # --- Source 1: .bib (filename triple-match → title reverse search) ---
        bib_entry = None
        bib_match_status = "no_match"
        if fname_key:
            bib_entry, bib_match_status = self.bib.by_filename_key(
                fname_key,
                paper_title_norm=normalize((clean_title or claimed["title"])[:50]),
                paper_abstract_norm=paper_abstract_norm,
            )
        if not bib_entry and clean_title:
            bib_entry, bib_match_status = self.bib.by_title(
                clean_title or claimed["title"],
                claimed["year"] or (fname_key[1] if fname_key else ""),
                paper_abstract_norm=paper_abstract_norm,
            )
        if bib_entry and bib_entry.doi:
            nd = normalize_doi(bib_entry.doi)
            cand = candidates.setdefault(nd, {"doi": bib_entry.doi})
            cand.update({
                "title": bib_entry.title,
                "authors": bib_entry.author_full,
                "year": bib_entry.year,
                "journal": bib_entry.journal,
                "key": bib_entry.key,
            })
            source_dois["bib"] = nd

        # --- Source 2: Crossref ---
        # If claimed DOI resolved but title/year wrong, crossref_resp has the truth.
        if crossref_resp and crossref_resp.get("title"):
            nd = normalize_doi(crossref_resp["doi"])
            cand = candidates.setdefault(nd, {"doi": crossref_resp["doi"]})
            cand.setdefault("title", crossref_resp["title"])
            cand.setdefault("year", crossref_resp["year"])
            cand.setdefault("journal", crossref_resp["journal"])
            cand.setdefault("authors", crossref_resp["authors"])
            source_dois["crossref_verify"] = nd
        # Crossref search (by title + author + year) to find genuine DOI
        if self.crossref.enabled and clean_title:
            author_hint = ""
            if (oracle and len(oracle) > 3 and oracle[3]
                    and any(sepc in oracle[3] for sepc in "_-")):
                # compound surname: search with the RAW filename form
                # (`Bussoni_Guitart`) — the normalized glue
                # (`bussoniguitart`) never hits Crossref's author index;
                # the raw form feeds the last-segment retry below.
                author_hint = oracle[3]
            elif fname_key:
                author_hint = fname_key[0]
            elif claimed["authors"]:
                # Take first author's last name
                first = claimed["authors"].split(",")[0]
                author_hint = first.strip().split()[-1] if first.strip() else ""
            results = self.crossref.search(
                clean_title or claimed["title"],
                author_hint,
                claimed["year"] or (fname_key[1] if fname_key else ""),
            )
            if not results and author_hint and any(
                    sepc in author_hint for sepc in "_-"):
                # compound surnames parse as one glued token
                # (`Bussoni_Guitart` → `bussoniguitart`) which Crossref's
                # author index never matches; retry with the LAST segment
                # (registry `family` field convention: `Guitart`).
                retry_hint = re.split(r"[_\-]", author_hint)[-1]
                if retry_hint and retry_hint != author_hint:
                    results = self.crossref.search(
                        clean_title or claimed["title"],
                        retry_hint,
                        claimed["year"] or (fname_key[1] if fname_key else ""),
                    )
            # Pick best match by title similarity + year tolerance.
            # PREFIX-RECALL semantics: a parent_store filename truncates at
            # 255 bytes, so the search key (often the filename title) can be
            # a PREFIX of the registry title — Jaccard then reads ~0.67 for a
            # PERFECT match (10 claimed tokens inside 15 registry tokens) and
            # the correct record was being rejected. Recall (claimed tokens
            # covered by the result) stays 1.0 under pure truncation; a wrong
            # paper loses recall. Symmetric check guards the reverse case.
            best = None
            best_sim = 0.0
            claimed_title_tokens = title_tokens(clean_title or claimed["title"])
            for r in results:
                rt = title_tokens(r["title"])
                if not rt or not claimed_title_tokens:
                    continue
                # PREFIX-RECALL semantics — see _title_search_sim.
                sim = _title_search_sim(clean_title or claimed["title"], r["title"])
                year_ok = False
                if claimed["year"] and r["year"]:
                    try:
                        year_ok = abs(int(claimed["year"]) - int(r["year"])) <= YEAR_TOLERANCE_SEARCH
                    except ValueError:
                        year_ok = claimed["year"] == r["year"]
                if sim >= TITLE_JACCARD_MATCH and (year_ok or not claimed["year"]):
                    if sim > best_sim:
                        best = r
                        best_sim = sim
            if best:
                nd = normalize_doi(best["doi"])
                cand = candidates.setdefault(nd, {"doi": best["doi"]})
                cand.setdefault("title", best["title"])
                cand.setdefault("year", best["year"])
                cand.setdefault("journal", best["journal"])
                cand.setdefault("authors", best["authors"])
                source_dois["crossref_search"] = nd

        # --- Source 3: PubMed (title search → summary for PMID/PMCID) ---
        if self.pubmed.enabled and clean_title:
            author_hint = ""
            if claimed["authors"]:
                first = claimed["authors"].split(",")[0]
                author_hint = first.strip().split()[-1] if first.strip() else ""
            pmids = self.pubmed.search(clean_title or claimed["title"], author_hint)
            pubmed_summary = None
            # The first esearch hit is often a NEIGHBOR paper (relevance
            # ranking, not exact matching) — walk the top 5 and keep the
            # summary whose title actually matches the claimed one.
            for pmid in pmids[:5]:
                cand_sum = self.pubmed.summary(pmid)
                if not cand_sum or not cand_sum.get("title"):
                    continue
                pt = title_tokens(cand_sum["title"])
                ct = title_tokens(clean_title or claimed["title"])
                if jaccard(pt, ct) >= TITLE_JACCARD_MATCH:
                    pubmed_summary = cand_sum
                    break
            if pubmed_summary and pubmed_summary.get("title"):
                # Validate title match
                pt = title_tokens(pubmed_summary["title"])
                ct = title_tokens(clean_title or claimed["title"])
                if jaccard(pt, ct) >= TITLE_JACCARD_MATCH:
                    nd = normalize_doi(pubmed_summary["doi"]) if pubmed_summary["doi"] else None
                    if nd:
                        cand = candidates.setdefault(nd, {"doi": pubmed_summary["doi"]})
                        cand["pmid"] = pubmed_summary["pmid"]
                        cand["pmcid"] = pubmed_summary.get("pmcid", "")
                        cand.setdefault("title", pubmed_summary["title"])
                        cand.setdefault("year", pubmed_summary["year"])
                        cand.setdefault("authors", pubmed_summary["authors"])
                        source_dois["pubmed"] = nd
                    else:
                        # No DOI from PubMed but we got PMID/PMCID — attach to any existing candidate
                        if candidates:
                            first_key = next(iter(candidates))
                            candidates[first_key]["pmid"] = pubmed_summary["pmid"]
                            candidates[first_key]["pmcid"] = pubmed_summary.get("pmcid", "")
                        source_dois["pubmed_pmid"] = pubmed_summary["pmid"]

        # --- Source 4: Zotero local API (corroboration) ---
        if self.zotero.enabled and self.zotero.available and clean_title:
            zot = self.zotero.search(clean_title or claimed["title"])
            if zot:
                nd = normalize_doi(zot.get("doi", "")) if zot.get("doi") else None
                if nd:
                    cand = candidates.setdefault(nd, {"doi": zot["doi"]})
                    cand.setdefault("title", zot["title"])
                    cand.setdefault("year", zot["year"])
                    cand.setdefault("journal", zot["journal"])
                    cand.setdefault("authors", zot["authors"])
                    cand["key"] = zot["key"]
                    source_dois["zotero"] = nd
                else:
                    # Attach key to existing candidate
                    if candidates:
                        first_key = next(iter(candidates))
                        candidates[first_key]["key"] = zot["key"]
                    source_dois["zotero_key"] = zot["key"]

        # --- Source 5: OpenAlex (secondary cross-check) ---
        if self.openalex.enabled and openalex_resp and openalex_resp.get("title"):
            nd = normalize_doi(openalex_resp["doi"])
            cand = candidates.setdefault(nd, {"doi": openalex_resp["doi"]})
            cand.setdefault("title", openalex_resp["title"])
            cand.setdefault("year", openalex_resp["year"])
            cand.setdefault("journal", openalex_resp["journal"])
            cand.setdefault("authors", openalex_resp["authors"])
            source_dois["openalex_verify"] = nd

        # --- Merge + confidence scoring ---
        if not candidates:
            return None, "none", 0

        # Oracle gate: reject candidates belonging to a different paper than
        # the filename (swapped-meta files; wrong-paper search matches).
        if oracle:
            for nd in list(candidates.keys()):
                if _oracle_rejects(candidates[nd], oracle):
                    log.debug("oracle rejected candidate %s for %s",
                              nd, oracle[2][:40] if oracle[2] else "")
                    del candidates[nd]
                    # also drop DOI agreements pointing at the rejected record
                    for k in [k for k, v in source_dois.items() if v == nd]:
                        del source_dois[k]
            if not candidates:
                return None, "none", 0

        # Pick the candidate with the most source agreements.
        # Count DISTINCT source families agreeing on each DOI — crossref_verify
        # and crossref_search are two queries to the same API, not two
        # independent sources, so they must not double-count. Non-DOI fallback
        # values (PMIDs, Zotero keys) do not count as DOI agreement.
        SOURCE_FAMILY = {
            "bib": "bib",
            "crossref_verify": "crossref", "crossref_search": "crossref",
            "pubmed": "pubmed", "pubmed_pmid": "pubmed",
            "zotero": "zotero", "zotero_key": "zotero",
            "openalex_verify": "openalex",
        }
        family_dois: Dict[str, set] = defaultdict(set)
        for src, nd in source_dois.items():
            if nd and is_doi_like(nd):
                family_dois[SOURCE_FAMILY.get(src, src)].add(nd)
        doi_source_counts: Dict[str, int] = defaultdict(int)
        for fam, dois in family_dois.items():
            for nd in dois:
                doi_source_counts[nd] += 1
        if not doi_source_counts:
            # No DOI agreement — pick first candidate by insertion order
            best_nd = next(iter(candidates))
            n_agree = 0
        else:
            best_nd = max(doi_source_counts, key=doi_source_counts.get)
            n_agree = doi_source_counts[best_nd]

        fix = candidates[best_nd].copy()
        # Normalize field names for output
        out = {
            "doi": fix.get("doi", ""),
            "title": fix.get("title", ""),
            "authors": fix.get("authors", ""),
            "year": fix.get("year", ""),
            "journal": fix.get("journal", ""),
            "pmid": fix.get("pmid", ""),
            "pmcid": fix.get("pmcid", ""),
            "key": fix.get("key", ""),
        }
        # Confidence: HIGH if ≥2 sources agree on DOI; MEDIUM if 1 (DOI verified);
        # LOW only for pure-search matches with title disagreement.
        # NOTE: when the DOI was *verified* by Crossref/OpenAlex (crossref_verify /
        # openalex_verify keys), the DOI is known-correct, so the title returned
        # from that DOI is authoritative — the claimed title being different is the
        # very corruption we are fixing, NOT a signal of a wrong search match.
        doi_verified = ("crossref_verify" in source_dois
                        or "openalex_verify" in source_dois)
        fix_doi_matches_verified = False
        if doi_verified:
            for k in ("crossref_verify", "openalex_verify"):
                if source_dois.get(k) == best_nd:
                    fix_doi_matches_verified = True
                    break
        if n_agree >= 2:
            confidence = "high"
        elif n_agree == 1:
            if fix_doi_matches_verified:
                # DOI was verified by resolution; title from that DOI is authoritative.
                confidence = "medium"
            elif clean_title and out["title"]:
                # Pure-search match: check title similarity to guard wrong
                # matches — prefix-recall semantics, same as search-time pick.
                if _title_search_sim(clean_title, out["title"]) < TITLE_JACCARD_MATCH:
                    confidence = "low"
                else:
                    confidence = "medium"
            else:
                confidence = "medium"
        else:
            confidence = "low"

        return out, confidence, n_agree


# ---------------------------------------------------------------------------
# Reporter — write CSV / JSON / MD outputs
# ---------------------------------------------------------------------------

class Reporter:
    def __init__(self, out_dir: Path, data_dir: Path, date_tag: str):
        self.out_dir = out_dir
        self.data_dir = data_dir
        self.date_tag = date_tag
        self.csv_path = out_dir / f"meta_audit_report_{date_tag}.csv"
        self.summary_path = data_dir / f"meta_audit_summary_{date_tag}.json"
        self.log_path = data_dir / f"meta_audit_log_{date_tag}.json"
        self.md_path = data_dir / f"meta_audit_report_{date_tag}.md"

    def write_csv(self, rows: List[Dict[str, Any]]):
        fields = [
            "json_file", "status", "doi_check", "title_check", "year_check",
            "authors_check", "identity_flags", "needs_reindex",
            "claimed_doi", "claimed_title", "claimed_year", "claimed_journal",
            "fix_doi", "fix_title", "fix_authors", "fix_year",
            "fix_journal", "fix_pmid", "fix_pmcid", "fix_key",
            "confidence", "sources_agreeing",
        ]
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with open(self.csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                fix = r.get("suggested_fix") or {}
                claimed = r.get("claimed_meta") or {}
                fc = r.get("field_checks", {})
                w.writerow({
                    "json_file": r["file"],
                    "status": r["status"],
                    "doi_check": fc.get("doi", ""),
                    "title_check": fc.get("title", ""),
                    "year_check": fc.get("year", ""),
                    "authors_check": fc.get("authors", ""),
                    "identity_flags": ";".join(r.get("identity_flags", []) or []),
                    "needs_reindex": bool(r.get("needs_reindex")),
                    "claimed_doi": claimed.get("doi", ""),
                    "claimed_title": (claimed.get("title", "") or "")[:80],
                    "claimed_year": claimed.get("year", ""),
                    "claimed_journal": (claimed.get("journal", "") or "")[:60],
                    "fix_doi": fix.get("doi", ""),
                    "fix_title": (fix.get("title", "") or "")[:80],
                    "fix_authors": (fix.get("authors", "") or "")[:80],
                    "fix_year": fix.get("year", ""),
                    "fix_journal": (fix.get("journal", "") or "")[:60],
                    "fix_pmid": fix.get("pmid", ""),
                    "fix_pmcid": fix.get("pmcid", ""),
                    "fix_key": fix.get("key", ""),
                    "confidence": r.get("confidence", ""),
                    "sources_agreeing": r.get("sources_agreeing", 0),
                })

    def write_summary(self, results: List[Dict[str, Any]], elapsed: float,
                      source_usage: Dict[str, Any]):
        n_total = len(results)
        n_pass = sum(1 for r in results if r["status"] == "pass")
        n_fail = sum(1 for r in results if r["status"] == "fail")
        n_unverified = sum(1 for r in results if r["status"] == "unverified")
        n_error = sum(1 for r in results if r["status"] == "error")
        n_high = sum(1 for r in results if r.get("confidence") == "high")
        n_medium = sum(1 for r in results if r.get("confidence") == "medium")
        n_low = sum(1 for r in results if r.get("confidence") == "low")

        field_fail: Dict[str, int] = {"doi": 0, "title": 0, "year": 0}
        for r in results:
            if r["status"] != "fail":
                continue
            fc = r.get("field_checks", {})
            for k in field_fail:
                v = fc.get(k, "")
                if v not in ("resolves", "resolves_oa", "match", "match_oa",
                             "match_bib", "no_ref"):
                    field_fail[k] += 1

        top_failing = sorted(
            [r for r in results if r["status"] == "fail"],
            key=lambda r: r.get("sources_agreeing", 0),
            reverse=True,
        )[:20]

        n_aq_bad = sum(1 for r in results
                       if r.get("field_checks", {}).get("authors", "ok") != "ok")
        n_identity = sum(1 for r in results
                         if r.get("identity_flags")
                         and not r.get("pmid_gold", False))
        n_reindex = sum(1 for r in results if r.get("needs_reindex"))
        n_meta_swap = sum(1 for r in results
                          if any("meta_swap" in f for f in (r.get("identity_flags") or [])))

        summary = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "elapsed_seconds": round(elapsed, 1),
            "n_total": n_total,
            "n_pass": n_pass,
            "n_fail": n_fail,
            "n_unverified": n_unverified,
            "n_error": n_error,
            "n_fix_high": n_high,
            "n_fix_medium": n_medium,
            "n_fix_low": n_low,
            "field_fail_counts": field_fail,
            "authors_defects": n_aq_bad,
            "identity_mismatches": n_identity,
            "meta_swaps": n_meta_swap,
            "needs_reindex": n_reindex,
            "source_usage": source_usage,
            "top_failing_files": [
                {"file": r["file"], "confidence": r.get("confidence"),
                 "sources_agreeing": r.get("sources_agreeing")}
                for r in top_failing
            ],
        }
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    def write_log(self, results: List[Dict[str, Any]]):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "log_entries": results,
        }
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def write_md(self, results: List[Dict[str, Any]], elapsed: float):
        n_total = len(results)
        n_pass = sum(1 for r in results if r["status"] == "pass")
        n_fail = sum(1 for r in results if r["status"] == "fail")
        n_unverified = sum(1 for r in results if r["status"] == "unverified")
        pass_rate = (n_pass * 100.0 / n_total) if n_total else 0.0
        n_high = sum(1 for r in results if r.get("confidence") == "high")
        n_medium = sum(1 for r in results if r.get("confidence") == "medium")
        n_low = sum(1 for r in results if r.get("confidence") == "low")
        lines = [
            f"# Meta Audit Report — {self.date_tag}",
            "",
            f"Generated: {datetime.now().isoformat(timespec='seconds')}",
            f"Elapsed: {elapsed:.1f}s",
            "",
            "## Summary",
            "",
            f"- Total files audited: **{n_total}**",
            f"- Pass: **{n_pass}** ({pass_rate:.1f}%)",
            f"- Fail: **{n_fail}**",
            f"- Unverified: **{n_unverified}**",
            "",
            "## Fix Confidence Distribution",
            "",
            f"- HIGH (≥2 sources agree): **{n_high}**",
            f"- MEDIUM (1 source): **{n_medium}**",
            f"- LOW (title mismatch / conflict): **{n_low}**",
            "",
            "## Top Failing Files",
            "",
            "| File | Confidence | Sources | DOI check | Title check | Year check |",
            "|---|---|---|---|---|---|",
        ]
        fails = [r for r in results if r["status"] == "fail"]
        # Sort by confidence (high first), then sources_agreeing
        conf_order = {"high": 0, "medium": 1, "low": 2, "none": 3}
        fails.sort(key=lambda r: (conf_order.get(r.get("confidence", "none"), 9),
                                  -r.get("sources_agreeing", 0)))
        for r in fails[:30]:
            fc = r.get("field_checks", {})
            lines.append(
                f"| {r['file'][:50]} | {r.get('confidence','')} | "
                f"{r.get('sources_agreeing',0)} | {fc.get('doi','')} | "
                f"{fc.get('title','')[:25]} | {fc.get('year','')} |"
            )
        lines.append("")
        with open(self.md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Apply — write fixes back to parent_store (with backup)
# ---------------------------------------------------------------------------

def apply_fixes(results: List[Dict[str, Any]], parent_dir: Path,
                backup_dir: Path, min_confidence: str = "medium") -> Dict[str, int]:
    """Write suggested_fix back to parent_store/*.json (with backup).

    Only applies fixes with confidence >= min_confidence.
    Returns {applied, skipped, errors}.
    """
    conf_rank = {"high": 3, "medium": 2, "low": 1, "none": 0}
    min_rank = conf_rank.get(min_confidence, 2)
    backup_dir.mkdir(parents=True, exist_ok=True)
    applied = 0
    skipped = 0
    errors = 0
    for r in results:
        fix = r.get("suggested_fix")
        conf = r.get("confidence", "none")
        if not fix or conf_rank.get(conf, 0) < min_rank:
            skipped += 1
            continue
        fp = parent_dir / r["file"]
        try:
            # Backup (only if not already backed up)
            bp = backup_dir / r["file"]
            if not bp.exists():
                bp.write_text(fp.read_text(encoding="utf-8"), encoding="utf-8")
            data = json.loads(fp.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                skipped += 1
                continue
            # Update meta in every chunk (all chunks share meta — update all for safety)
            identity_fix = bool(r.get("identity_flags"))
            for sec in data:
                meta = sec.setdefault("meta", {})
                if identity_fix:
                    # swapped meta: overwrite ALL biblio fields — empty fix
                    # values must CLEAR the stolen ones (pmid/pmcid/journal)
                    for k in ("doi", "title", "authors", "year", "journal",
                              "pmid", "pmcid"):
                        meta[k] = fix.get(k, "")
                    if fix.get("key"):
                        meta["key"] = fix["key"]
                    continue
                if fix.get("doi"):
                    meta["doi"] = fix["doi"]
                if fix.get("title"):
                    meta["title"] = fix["title"]
                if fix.get("authors"):
                    meta["authors"] = fix["authors"]
                if fix.get("year"):
                    meta["year"] = fix["year"]
                if fix.get("journal"):
                    meta["journal"] = fix["journal"]
                if fix.get("pmid"):
                    meta["pmid"] = fix["pmid"]
                if fix.get("pmcid"):
                    meta["pmcid"] = fix["pmcid"]
                if fix.get("key"):
                    meta["key"] = fix["key"]
            fp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            applied += 1
        except Exception as ex:
            log.warning("apply failed for %s: %s", r["file"], ex)
            errors += 1
    return {"applied": applied, "skipped": skipped, "errors": errors}


# ---------------------------------------------------------------------------
# Resume — skip already-audited files
# ---------------------------------------------------------------------------

def load_audited_files(data_dir: Path) -> set:
    """Return filenames that PASSED in the most recent run's log.

    Only `pass` entries are skipped on --resume: failures must be re-audited
    (they may have been fixed by a subsequent --apply, or still need fixing).
    """
    logs = sorted(data_dir.glob("meta_audit_log_*.json"))
    if not logs:
        return set()
    try:
        d = json.loads(logs[-1].read_text(encoding="utf-8"))
        return {
            e["file"]
            for e in d.get("log_entries", [])
            if e.get("status") == "pass" and "file" in e
        }
    except Exception:
        return set()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Audit paper metadata in parent_store and fetch genuine info for failures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--parent-dir", type=Path, default=DEFAULT_PARENT_DIR)
    ap.add_argument("--bib", type=Path, default=DEFAULT_BIB_PATH)
    ap.add_argument("--zotero-url", default=DEFAULT_ZOTERO_URL)
    ap.add_argument("--crossref-mailto", default=DEFAULT_CROSSREF_MAILTO)
    ap.add_argument("--pubmed-api-key", default="")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--limit", type=int, default=0, help="audit only first N files")
    ap.add_argument("--sample", type=int, default=0, help="random sample of N files")
    ap.add_argument("--offline", action="store_true",
                    help="no external APIs (.bib + local Zotero only)")
    ap.add_argument("--no-crossref", action="store_true")
    ap.add_argument("--no-pubmed", action="store_true")
    ap.add_argument("--no-openalex", action="store_true")
    ap.add_argument("--no-zotero", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="skip files that PASSED in the previous run's log")
    ap.add_argument("--only-defects", action="store_true",
                    help="audit only files flagged by the local pre-scan "
                         "(authors missing/initials, junk titles, filename/"
                         "meta identity mismatch) — skips clean files, saves "
                         "API calls")
    ap.add_argument("--apply", action="store_true", help="write fixes back (implies backup)")
    ap.add_argument("--backup-dir", type=Path, default=None,
                    help="default: data/parent_store_backup_metafix_<DATE>")
    ap.add_argument("--apply-min-confidence", default="medium",
                    choices=["high", "medium", "low"],
                    help="minimum confidence to apply (default: medium)")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    if args.limit and args.sample:
        ap.error("--limit and --sample are mutually exclusive")

    # Defaults. --offline disables the external APIs only; the LOCAL Zotero
    # client stays enabled (it makes no network call outside this machine).
    use_crossref = not args.offline and not args.no_crossref
    use_pubmed = not args.offline and not args.no_pubmed
    use_openalex = not args.offline and not args.no_openalex
    use_zotero = not args.no_zotero

    # Load .bib
    if not args.bib.exists():
        log.error("bib file not found: %s", args.bib)
        return 1
    if not args.parent_dir.exists():
        log.error("parent dir not found: %s", args.parent_dir)
        return 1
    log.info("parsing %s ...", args.bib)
    bib_entries = parse_bib_entries(args.bib)
    bib_index = BibIndex(bib_entries)
    log.info("  %d entries, %d with DOI", len(bib_entries), len(bib_index.doi_index))

    # Init clients
    crossref = CrossrefClient(args.crossref_mailto, enabled=use_crossref)
    pubmed = PubmedClient(args.pubmed_api_key, enabled=use_pubmed)
    openalex = OpenAlexClient(enabled=use_openalex)
    zotero = ZoteroClient(args.zotero_url, enabled=use_zotero)
    if use_zotero and zotero.available:
        log.info("zotero available at %s", args.zotero_url)

    auditor = Auditor(bib_index, crossref, pubmed, openalex, zotero)
    reporter = Reporter(args.out_dir, args.data_dir, DATE_TAG)

    # Enumerate files
    files = sorted(args.parent_dir.glob("*.json"))
    if args.limit:
        files = files[: args.limit]
    elif args.sample:
        import random
        random.seed(42)
        files = sorted(random.sample(files, min(args.sample, len(files))))
    if args.resume:
        audited = load_audited_files(args.data_dir)
        before = len(files)
        files = [f for f in files if f.name not in audited]
        log.info("resume: %d files passed previously, %d to audit", before - len(files), len(files))

    if args.only_defects:
        before = len(files)
        files = [f for f in files if _local_defect_check(f)]
        log.info("only-defects: %d/%d files flagged for audit", len(files), before)

    log.info("auditing %d files (mode=%s)...", len(files),
             "apply" if args.apply else "dry-run")

    results: List[Dict[str, Any]] = []
    t0 = time.time()
    for i, fp in enumerate(files, 1):
        if i % 50 == 0 or args.verbose:
            log.info("[%d/%d] %s", i, len(files), fp.name)
        try:
            r = auditor.audit_file(fp)
        except Exception as ex:
            log.exception("audit failed for %s", fp.name)
            r = {
                "file": fp.name, "status": "error", "error": str(ex),
                "field_checks": {}, "claimed_meta": {}, "evidence": {},
                "suggested_fix": None, "confidence": "none",
                "sources_agreeing": 0,
            }
        results.append(r)
    elapsed = time.time() - t0

    # Source usage stats
    source_usage = {
        "crossref": {"calls": crossref.calls, "errors": crossref.errors,
                      "enabled": crossref.enabled},
        "pubmed": {"calls": pubmed.calls, "errors": pubmed.errors,
                   "enabled": pubmed.enabled},
        "openalex": {"calls": openalex.calls, "errors": openalex.errors,
                     "enabled": openalex.enabled},
        "zotero": {"calls": zotero.calls, "errors": zotero.errors,
                   "enabled": zotero.enabled, "available": zotero.available},
        "bib": {"entries": len(bib_entries), "with_doi": len(bib_index.doi_index)},
    }

    # Write reports
    reporter.write_csv(results)
    reporter.write_log(results)
    reporter.write_summary(results, elapsed, source_usage)
    reporter.write_md(results, elapsed)

    # Content-swap files: no meta fix can help — write the reindex list
    reindex = [r["file"] for r in results if r.get("needs_reindex")]
    if reindex:
        p = args.data_dir / f"meta_audit_reindex_{DATE_TAG}.txt"
        p.write_text("\n".join(reindex) + "\n", encoding="utf-8")
        log.info("%d files need REINDEX (content swap): %s", len(reindex), p)

    # Apply (if requested)
    apply_summary = None
    if args.apply:
        backup_dir = args.backup_dir or (args.data_dir / f"parent_store_backup_metafix_{DATE_TAG}")
        apply_summary = apply_fixes(
            results, args.parent_dir, backup_dir,
            min_confidence=args.apply_min_confidence,
        )
        log.info("applied %d fixes, skipped %d, errors %d (backup: %s)",
                 apply_summary["applied"], apply_summary["skipped"],
                 apply_summary["errors"], backup_dir)
        # Write apply log
        apply_log = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "backup_dir": str(backup_dir),
            "min_confidence": args.apply_min_confidence,
            **apply_summary,
        }
        (args.data_dir / f"meta_audit_apply_log_{DATE_TAG}.json").write_text(
            json.dumps(apply_log, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # Print summary
    n_total = len(results)
    n_pass = sum(1 for r in results if r["status"] == "pass")
    n_fail = sum(1 for r in results if r["status"] == "fail")
    n_unverified = sum(1 for r in results if r["status"] == "unverified")
    n_error = sum(1 for r in results if r["status"] == "error")
    n_high = sum(1 for r in results if r.get("confidence") == "high")
    n_medium = sum(1 for r in results if r.get("confidence") == "medium")
    n_low = sum(1 for r in results if r.get("confidence") == "low")
    print()
    print(f"=== meta_audit done ({elapsed:.1f}s, {n_total} files) ===")
    print(f"  pass:       {n_pass}")
    print(f"  fail:       {n_fail}")
    print(f"  unverified: {n_unverified}")
    print(f"  error:      {n_error}")
    print(f"  fix HIGH:   {n_high}")
    print(f"  fix MEDIUM: {n_medium}")
    print(f"  fix LOW:    {n_low}")
    print(f"  → {reporter.csv_path}")
    print(f"  → {reporter.summary_path}")
    print(f"  → {reporter.md_path}")
    if apply_summary:
        print(f"  applied:    {apply_summary['applied']}")
        print(f"  skipped:    {apply_summary['skipped']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())