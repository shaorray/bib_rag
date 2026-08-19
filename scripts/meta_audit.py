#!/usr/bin/env python3
"""meta_audit.py — Paper metadata proof-reader and genuine-info fetcher.

Validates every paper's `meta` block in parent_store/*.json against multiple
ground-truth sources, and for papers that fail proof-reading, fetches the
genuine metadata (DOI, title, authors, year, journal, PMID/PMCID) into a
concrete `suggested_fix` payload. Dry-run by default; --apply writes back.

Ground-truth sources (priority order):
  1. /Disk_bot/My Library.bib  — canonical Zotero export (title, DOI, authors, year, journal, key)
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
import shutil
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

# ---------------------------------------------------------------------------
# Defaults (overridable via CLI)
# ---------------------------------------------------------------------------

DEFAULT_PARENT_DIR = Path("/Disk_bot/Eph/bib_rag/parent_store")
DEFAULT_BIB_PATH = Path("/Disk_bot/My Library.bib")
DEFAULT_ZOTERO_URL = "http://localhost:23119"
DEFAULT_CROSSREF_MAILTO = "bib-rag@example.com"
DEFAULT_DATA_DIR = Path("/Disk_bot/Eph/bib_rag/data")
DEFAULT_OUT_DIR = Path("/Disk_bot/Eph/bib_rag/outputs")

DATE_TAG = datetime.now().strftime("%Y-%m-%d")

# Pass/fail thresholds
TITLE_JACCARD_PASS = 0.85      # strict pass
TITLE_JACCARD_MATCH = 0.80     # threshold for accepting a search-result match
YEAR_TOLERANCE_PASS = 1        # |year diff| <= 1 passes
YEAR_TOLERANCE_SEARCH = 2      # search window for crossref/pubmed

# Rate limits (seconds between requests)
CROSSREF_MIN_INTERVAL = 1.0 / 50   # polite pool allows 50/s; be conservative
PUBMED_MIN_INTERVAL = 0.34         # 3 req/s without API key
OPENALEX_MIN_INTERVAL = 0.1         # 10 req/s polite
ZOTERO_MIN_INTERVAL = 0.1

HTTP_TIMEOUT = 10
ZOTERO_TIMEOUT = 3

# Known fake-DOI patterns (from data/doi_health_report.md analysis)
FAKE_DOI_PATTERNS = [
    r"\(SICI\)",                  # Wiley URL truncation
    r"^BibKey:",                  # placeholder leaked from front-matter injection
    r"https?://",                 # URL-as-DOI
    r"^doi:\s*10\.\S+[<>]",       # markdown pollution
]

log = logging.getLogger("meta_audit")


# ---------------------------------------------------------------------------
# Normalization helpers (reused from scripts/bib_to_parent_store.py conventions)
# ---------------------------------------------------------------------------

def normalize(s: str) -> str:
    """NFKD + strip combining + lowercase + drop non-alphanumeric."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def normalize_doi(d: str) -> str:
    """Strip URL prefix / punctuation from a DOI."""
    if not d:
        return ""
    s = d.strip().lower()
    s = re.sub(r"^https?://(dx\.)?doi\.org/", "", s)
    s = re.sub(r"^doi:\s*", "", s)
    s = s.rstrip("/.,;)")
    return s


def title_tokens(s: str) -> set:
    """Tokenize a title for Jaccard: lowercase + keep alphanumeric tokens."""
    if not s:
        return set()
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    tokens = re.findall(r"[a-z0-9]+", s)
    # Drop very short noise tokens (a, an, the, of ...)
    return {t for t in tokens if len(t) > 2}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def strip_author_year_prefix(title: str) -> str:
    """Strip `Lastname et al. - YEAR - ` or `Lastname - YEAR - ` prefix.

    Mirrors scripts/bib_to_parent_store.py:normalize_paper_title.
    """
    if not title:
        return ""
    if title.startswith("http"):
        return ""
    m = re.match(r"^(.+?)\s+et\s+al\.?\s*-\s*(?:\d{4}\s*-\s*)?(.+)$", title)
    if m:
        return m.group(2).strip()
    m = re.match(r"^[A-Za-z][A-Za-z0-9_-]+\s*-\s*(?:\d{4}\s*-\s*)?(.+)$", title)
    if m:
        return m.group(1).strip()
    # Strip leading "YYYY - " (filename-as-title case)
    m = re.match(r"^\d{4}\s*-\s*(.+)$", title)
    if m:
        return m.group(1).strip()
    return title.strip()


def filename_to_key(fname: str) -> Optional[Tuple[str, str, str]]:
    """Parse parent_store filename → (lastname, year, title_prefix_norm).

    Adapted from scripts/bib_to_parent_store.py:filename_to_key.
    """
    # Format 1: Lastname_et_al__-_<year>_<title>
    m = re.match(r"([\w-]+?)_et_al__-?-?(\d{4})_(.+)", fname, re.UNICODE)
    if m:
        lastname = m.group(1).lower()
        year = m.group(2)
        title_part = m.group(3).replace("_md.json", "").replace("_md", "").replace("_", " ")
        return (lastname, year, normalize(title_part[:50]))
    # Format 2: Lastname_and_Lastname_<year>_<title> or Lastname_<year>_<title>
    m = re.match(r"([\w-]+?)_(\d{4})_(.+)", fname, re.UNICODE)
    if m:
        lastname_raw = m.group(1)
        if "and" not in lastname_raw.lower():
            title_part = m.group(3).replace("_md.json", "").replace("_md", "").replace("_", " ")
            return (lastname_raw.lower(), m.group(2), normalize(title_part[:50]))
        first_author = lastname_raw.split("_and_")[0].lower()
        if first_author and re.match(r"^[A-Za-z]", first_author):
            title_part = m.group(3).replace("_md.json", "").replace("_md", "").replace("_", " ")
            return (first_author, m.group(2), normalize(title_part[:50]))
    # Format 3: Lastname_YYYY
    m = re.match(r"([\w-]+?)_(\d{4})", fname, re.UNICODE)
    if m:
        return (m.group(1).lower(), m.group(2), "")
    return None


def extract_year_from_content(data: list) -> str:
    """Fallback: extract publication year from paper content (first 2000 chars)."""
    for sec in data:
        c = sec.get("content", "")[:2000]
        for m in re.finditer(
            r"(?:©|Copyright|published|received|preprint|Cite this as)[\s\S]{0,40}?(19\d{2}|20[0-3]\d)",
            c, re.IGNORECASE,
        ):
            return m.group(1)
        m = re.search(r"10\.1101/(19\d{2}|20[0-3]\d)", c)
        if m:
            return m.group(1)
    return ""


def extract_abstract(data: list) -> str:
    """Pull abstract-ish section from content (for disambiguation)."""
    for sec in data:
        c = sec.get("content", "")
        m = re.search(
            r"(?si)##\s*\*?\*?(?:Abstract|Summary|Background|Introduction|Research\s+briefing)[:\s]*\n*(.*?)(?=\n##\s|\Z)",
            c,
        )
        if m:
            cand = m.group(1).strip()
            if len(cand) > 100:
                return cand
    return ""


def is_fake_doi(doi: str) -> bool:
    if not doi:
        return False
    for pat in FAKE_DOI_PATTERNS:
        if re.search(pat, doi):
            return True
    return False


# ---------------------------------------------------------------------------
# BibParser — load .bib into doi_index + title_index
# ---------------------------------------------------------------------------

class BibEntry:
    __slots__ = (
        "key", "doi", "title", "author_first_lastname", "author_full",
        "year", "abstract", "journal",
        "title_norm", "author_norm", "abstract_norm",
    )

    def __init__(self, key, doi, title, author_first_lastname, author_full,
                 year, abstract, journal):
        self.key = key
        self.doi = doi
        self.title = title
        self.author_first_lastname = author_first_lastname
        self.author_full = author_full
        self.year = year
        self.abstract = abstract
        self.journal = journal
        self.title_norm = normalize(title[:50])
        self.author_norm = normalize(author_first_lastname)
        self.abstract_norm = normalize(abstract[:200])

    def to_fix_payload(self) -> Dict[str, str]:
        return {
            "doi": self.doi,
            "title": self.title,
            "authors": self.author_full,
            "year": self.year,
            "journal": self.journal,
            "key": self.key,
        }


def parse_bib_entries(bib_path: Path) -> List[BibEntry]:
    """Parse BibTeX file into BibEntry list.

    Reuses the balanced-brace approach from scripts/bib_to_parent_store.py:parse_bib_entries,
    extended with journaltitle extraction.
    """
    content = bib_path.read_text(encoding="utf-8")
    entries_raw = re.split(r"\n(?=@\w+\{)", content)
    result: List[BibEntry] = []
    for e in entries_raw:
        m = re.match(r"^@(\w+)\{([^,]+),", e.strip())
        if not m:
            continue
        key = m.group(2).strip()

        def balanced_field(name):
            fm = re.search(
                r"\b" + re.escape(name) + r"\s*=\s*\{",
                e, re.IGNORECASE,
            )
            if not fm:
                fm = re.search(
                    r"\b" + re.escape(name) + r"\s*=\s*\"",
                    e, re.IGNORECASE,
                )
                if not fm:
                    return ""
                k = fm.end()
                end = e.find('"', k)
                return e[k:end] if end >= 0 else ""
            k = fm.end()
            depth = 1
            j = k
            buf = []
            while j < len(e) and depth > 0:
                c = e[j]
                if c == "{":
                    depth += 1
                    buf.append(c)
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        break
                    buf.append(c)
                else:
                    buf.append(c)
                j += 1
            inner = "".join(buf)
            # Strip one layer of nested {X} braces (Zotero convention)
            inner = re.sub(r"\{([^{}]*)\}", r"\1", inner)
            return inner.strip()

        doi = balanced_field("doi")
        title = balanced_field("title")
        author_str = balanced_field("author")
        journal = balanced_field("journaltitle") or balanced_field("journal")

        # year: prefer year field, then date field, then key
        year = ""
        m_year = re.search(r"^year\s*=\s*\{(\d{4})\}", e, re.MULTILINE)
        if m_year:
            year = m_year.group(1)
        if not year:
            m_date = re.search(r"^date\s*=\s*\{(\d{4})", e, re.MULTILINE)
            if m_date:
                year = m_date.group(1)
        if not year:
            m_key = re.search(r"(\d{4})", key)
            if m_key:
                year = m_key.group(1)

        # authors
        author_full = ""
        author_first_lastname = ""
        if author_str:
            author_str_clean = re.sub(r"[{}]", "", author_str)
            author_full = re.sub(r"\s+and\s+", ", ", author_str_clean).strip()
            first = re.split(r"\s+and\s+", author_str_clean)[0]
            if "," in first:
                author_first_lastname = first.split(",")[0].strip()
            else:
                parts = first.split()
                author_first_lastname = parts[-1] if parts else ""

        abstract = balanced_field("abstract")

        result.append(BibEntry(
            key=key, doi=doi, title=title,
            author_first_lastname=author_first_lastname,
            author_full=author_full, year=year,
            abstract=abstract, journal=journal,
        ))
    return result


class BibIndex:
    """Indexes over BibEntry list for fast lookup."""

    def __init__(self, entries: List[BibEntry]):
        self.entries = entries
        self.doi_index: Dict[str, BibEntry] = {}
        self.title_index: Dict[str, List[BibEntry]] = defaultdict(list)
        self.lastname_year_index: Dict[Tuple[str, str], List[BibEntry]] = defaultdict(list)
        for be in entries:
            if be.doi:
                nd = normalize_doi(be.doi)
                if nd and nd not in self.doi_index:
                    self.doi_index[nd] = be
            if be.title_norm:
                self.title_index[be.title_norm[:25]].append(be)
            if be.author_first_lastname and be.year:
                self.lastname_year_index[
                    (normalize(be.author_first_lastname), be.year)
                ].append(be)

    def by_doi(self, doi: str) -> Optional[BibEntry]:
        nd = normalize_doi(doi)
        return self.doi_index.get(nd) if nd else None

    def by_filename_key(self, fname_key: Tuple[str, str, str],
                        paper_title_norm: str = "",
                        paper_abstract_norm: str = "") -> Tuple[Optional[BibEntry], str]:
        """Triple-match (lastname + year + title-prefix overlap).

        Returns (entry, status) where status ∈ {matched, multi_match, no_match, no_year}.
        """
        lastname, year, title_norm = fname_key
        if not year:
            return None, "no_year"
        candidates = [
            be for be in self.lastname_year_index.get((lastname, year), [])
            if be.doi and be.title_norm
            and (
                be.title_norm[:25] in title_norm
                or title_norm[:25] in be.title_norm
            )
        ]
        if not candidates:
            return None, "no_match"
        if len(candidates) == 1:
            return candidates[0], "matched"
        # multi_match: disambiguate by title+abstract overlap
        def score(be):
            s = len(set(be.title_norm[:25]) & set(title_norm[:25]))
            if paper_abstract_norm and be.abstract_norm:
                if (be.abstract_norm[:30] in paper_abstract_norm
                        or paper_abstract_norm[:30] in be.abstract_norm):
                    s += 10
            return s
        best = max(candidates, key=score)
        return best, "multi_match"

    def by_title(self, paper_title: str, paper_year: str = "",
                 paper_abstract_norm: str = "") -> Tuple[Optional[BibEntry], str]:
        """Reverse title search (no_match fallback)."""
        if not paper_title or len(paper_title) < 15:
            return None, "title_too_short"
        paper_title_norm = normalize(paper_title[:80])
        candidates = []
        for be in self.entries:
            if not be.doi or not be.title_norm:
                continue
            if paper_year and be.year:
                try:
                    if abs(int(paper_year) - int(be.year)) > YEAR_TOLERANCE_PASS:
                        continue
                except ValueError:
                    if paper_year != be.year:
                        continue
            if (be.title_norm[:20] in paper_title_norm
                    or paper_title_norm[:20] in be.title_norm):
                candidates.append(be)
        if not candidates:
            return None, "no_match"
        if len(candidates) == 1:
            return candidates[0], "title_matched"
        def score(be):
            s = len(set(be.title_norm[:20]) & set(paper_title_norm[:20]))
            if paper_abstract_norm and be.abstract_norm:
                if (be.abstract_norm[:30] in paper_abstract_norm
                        or paper_abstract_norm[:30] in be.abstract_norm):
                    s += 10
            return s
        best = max(candidates, key=score)
        return best, "multi_match"


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
    """Zotero local API client. Optional corroboration source."""

    def __init__(self, base_url: str, enabled: bool = True):
        self.base_url = base_url.rstrip("/")
        self.enabled = enabled
        self._last = 0.0
        self.calls = 0
        self.errors = 0
        self.available = False
        if enabled:
            self._check()

    def _check(self):
        try:
            req = urllib.request.Request(f"{self.base_url}/api/users/0/items?limit=1")
            with urllib.request.urlopen(req, timeout=ZOTERO_TIMEOUT) as r:
                json.loads(r.read())
            self.available = True
        except Exception:
            self.available = False
            log.info("zotero not available at %s — skipping", self.base_url)

    def _throttle(self):
        now = time.time()
        wait = ZOTERO_MIN_INTERVAL - (now - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.time()

    def search(self, title: str) -> Optional[Dict[str, Any]]:
        if not self.enabled or not self.available or not title:
            return None
        self._throttle()
        url = f"{self.base_url}/api/users/0/items?q={urllib.parse.quote(title[:200])}&limit=3&itemType=-attachment"
        try:
            self.calls += 1
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=ZOTERO_TIMEOUT) as r:
                items = json.loads(r.read().decode("utf-8"))
        except Exception as ex:
            self.errors += 1
            log.debug("zotero search failed: %s", ex)
            return None
        if not items:
            return None
        # Pick the first item with a matching title
        for it in items:
            data = it.get("data", {})
            t = data.get("title", "")
            if not t:
                continue
            ts = title_tokens(t)
            ps = title_tokens(title)
            if jaccard(ts, ps) >= 0.70:
                return {
                    "key": data.get("key", ""),
                    "title": t,
                    "doi": data.get("DOI", ""),
                    "year": (data.get("date", "") or "")[:4],
                    "authors": "; ".join(
                        (c.get("lastName", "") + "," + c.get("firstName", ""))
                        for c in data.get("creators", [])
                    ),
                    "journal": data.get("publicationTitle", "") or data.get("journalTitle", ""),
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

        # Paper-side derived fields
        clean_title = strip_author_year_prefix(claimed["title"])
        fname_key = filename_to_key(fp.name)
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

        field_checks = {
            "doi": doi_field_check,
            "title": title_field_check,
            "year": year_field_check,
        }

        # ---- Determine pass/fail ----
        doi_ok = doi_field_check in ("resolves", "resolves_oa")
        title_ok = title_field_check in ("match", "match_oa")
        year_ok = year_field_check in ("match", "match_oa")
        # If both external verifiers were offline and DOI missing → unverified
        if (not self.crossref.enabled and not self.openalex.enabled
                and not claimed["doi"]):
            status = "unverified"
        elif doi_ok and title_ok and year_ok:
            status = "pass"
        else:
            status = "fail"

        # ---- Step 2: For failures, fetch genuine info ----
        suggested_fix = None
        confidence = "none"
        sources_agreeing = 0
        evidence: Dict[str, Any] = {"crossref": crossref_resp, "openalex": openalex_resp}

        if status == "fail":
            fix, conf, n = self._fetch_genuine(
                claimed, clean_title, fname_key, paper_abstract_norm,
                crossref_resp, openalex_resp,
            )
            suggested_fix = fix
            confidence = conf
            sources_agreeing = n

        return {
            "file": fp.name,
            "status": status,
            "field_checks": field_checks,
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
    ) -> Tuple[Optional[Dict[str, str]], str, int]:
        """For a failing paper, gather genuine metadata from all sources.

        Returns (suggested_fix payload or None, confidence, sources_agreeing).
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
            if fname_key:
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
            # Pick best match by title Jaccard + year tolerance
            best = None
            best_sim = 0.0
            claimed_title_tokens = title_tokens(clean_title or claimed["title"])
            for r in results:
                rt = title_tokens(r["title"])
                sim = jaccard(claimed_title_tokens, rt)
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
            for pmid in pmids[:1]:
                pubmed_summary = self.pubmed.summary(pmid)
                if pubmed_summary:
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

        # Pick the candidate with the most source agreements
        # Count distinct sources agreeing on each DOI
        doi_source_counts: Dict[str, int] = defaultdict(int)
        for src, nd in source_dois.items():
            if nd:
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
                # Pure-search match: check title similarity to guard wrong matches.
                ct = title_tokens(clean_title)
                ft = title_tokens(out["title"])
                if jaccard(ct, ft) < TITLE_JACCARD_MATCH:
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
                fc = r.get("field_checks") or {}
                w.writerow({
                    "json_file": r["file"],
                    "status": r["status"],
                    "doi_check": fc.get("doi", ""),
                    "title_check": fc.get("title", ""),
                    "year_check": fc.get("year", ""),
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
                if v not in ("resolves", "resolves_oa", "match", "match_oa", "no_ref"):
                    field_fail[k] += 1

        top_failing = sorted(
            [r for r in results if r["status"] == "fail"],
            key=lambda r: r.get("sources_agreeing", 0),
            reverse=True,
        )[:20]

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
            for sec in data:
                meta = sec.setdefault("meta", {})
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
    """Find the latest meta_audit_log_*.json and return set of audited filenames."""
    logs = sorted(data_dir.glob("meta_audit_log_*.json"))
    if not logs:
        return set()
    try:
        d = json.loads(logs[-1].read_text(encoding="utf-8"))
        return {e["file"] for e in d.get("log_entries", []) if "file" in e}
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
    ap.add_argument("--offline", action="store_true", help="skip all external APIs (.bib + Zotero only)")
    ap.add_argument("--no-crossref", action="store_true")
    ap.add_argument("--no-pubmed", action="store_true")
    ap.add_argument("--no-openalex", action="store_true")
    ap.add_argument("--no-zotero", action="store_true")
    ap.add_argument("--resume", action="store_true", help="skip files in the previous run's log")
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

    # Defaults
    use_crossref = not args.offline and not args.no_crossref
    use_pubmed = not args.offline and not args.no_pubmed
    use_openalex = not args.offline and not args.no_openalex
    use_zotero = not args.offline and not args.no_zotero

    # Load .bib
    if not args.bib.exists():
        log.error("bib file not found: %s", args.bib)
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
    if args.sample:
        import random
        random.seed(42)
        files = sorted(random.sample(files, min(args.sample, len(files))))
    if args.resume:
        audited = load_audited_files(args.data_dir)
        before = len(files)
        files = [f for f in files if f.name not in audited]
        log.info("resume: %d files already audited, %d remaining", before - len(files), len(files))

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