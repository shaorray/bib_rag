#!/usr/bin/env python3
"""bib_utils.py — shared helpers for the scripts/ utilities.

Single home for the normalization / filename-parsing / .bib-parsing logic that
was previously copy-pasted across scripts/bib_to_parent_store.py and
scripts/meta_audit.py (where it had already drifted — see the `_et_al` regex
typo that broke format-1 matching in meta_audit).

Keep this file dependency-free: it must be importable from any script run as
`python3 -B scripts/<tool>.py` (sys.path[0] = scripts/).
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Normalization
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
    """Strip URL prefix / punctuation from a DOI. Returns '' for empty input.

    Version-suffix rule matches src/identifiers.normalize_doi (canonical):
    strip a trailing v<digits> ONLY when preceded by a digit (optionally via
    a dot) — Oxford-style DOIs like 10.1093/nar/gkv370 legitimately end in
    "v<digits>" and must survive. Without this rule, versioned DOIs
    (...002v2) normalized here would never equality-match the canonical
    form used everywhere in src/ (bind_zotero / bib_to_parent_store do
    exactly that compare).
    """
    if not d:
        return ""
    s = d.strip().lower()
    s = re.sub(r"^https?://(dx\.)?doi\.org/", "", s)
    s = re.sub(r"^doi:\s*", "", s)
    s = s.rstrip("/.,;)")
    s = re.sub(r"(?<=\d)\.?v\d+$", "", s)
    return s


def is_doi_like(s: str) -> bool:
    """True if s looks like a real DOI (starts with the 10.xxxx/ scheme).

    Used to distinguish genuine DOI agreement from PMID / Zotero-key fallback
    values when counting source agreement.
    """
    return bool(s) and re.match(r"^10\.\d{4,}/", s) is not None


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


def strip_trailing_md(s: str) -> str:
    """Strip a trailing `_md` / `_md.json` marker from a parent_store filename.

    Only strips at the END of the string — a title that legitimately contains
    '_md' in the middle is left intact (the old `.replace('_md', '')` mangled
    those titles).
    """
    return re.sub(r"_md(?:\.json)?$", "", s)


def strip_author_year_prefix(title: str) -> str:
    """Strip `Lastname et al. - YEAR - ` or `Lastname - YEAR - ` prefix.

    Also handles a leading `YYYY - ` (filename-as-title case). A URL-as-title
    is treated as empty (it carries no bibliographic info).
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
    m = re.match(r"^\d{4}\s*-\s*(.+)$", title)
    if m:
        return m.group(1).strip()
    return title.strip()


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def filename_to_key(fname: str) -> Optional[Tuple[str, str, str]]:
    """Parse parent_store filename → (lastname, year, title_prefix_norm).

    Supports 3 formats:
    - `Lastname_et_al__-_<year>_<title>` (multi-author)
    - `Lastname_<year>_<title>` / `Lastname_and_Lastname_<year>_<title>`
    - `Lastname_<year>` (short)

    NOTE: the format-1 separator is `et_al__-_<year>` — TWO underscores, a
    hyphen, a LITERAL underscore, then the year. A previous copy of this
    function used `__-?-?` (no literal underscore), which never matched the
    real filenames and silently fell through to format 2 with a garbage
    lastname like `abdul-wajid_et_al__-`.
    """
    # Format 1: multi-author + et_al
    m = re.match(r"([\w-]+?)_et_al__-_-?(\d{4})_(.+)", fname, re.UNICODE)
    if m:
        lastname = m.group(1).lower()
        year = m.group(2)
        title_part = strip_trailing_md(m.group(3)).replace("_", " ")
        return (lastname, year, normalize(title_part[:50]))
    # Format 2: single author (Lastname_<year>_<title> or Lastname_and_Lastname_<year>_<title>)
    m = re.match(r"([\w-]+?)_(\d{4})_(.+)", fname, re.UNICODE)
    if m:
        lastname_raw = m.group(1)
        if "and" not in lastname_raw.lower():
            # Plain single author: lastname is the whole group
            title_part = strip_trailing_md(m.group(3)).replace("_", " ")
            return (lastname_raw.lower(), m.group(2), normalize(title_part[:50]))
        # Multi-author shorthand: use the first author
        first_author = lastname_raw.split("_and_")[0].lower()
        if first_author and re.match(r"^[A-Za-z]", first_author):
            title_part = strip_trailing_md(m.group(3)).replace("_", " ")
            return (first_author, m.group(2), normalize(title_part[:50]))
    # Format 3: short Lastname_YYYY
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


# ---------------------------------------------------------------------------
# Fake-DOI detection (from data/doi_health_report.md analysis)
# ---------------------------------------------------------------------------

FAKE_DOI_PATTERNS = [
    r"\(SICI\)",                  # Wiley URL truncation
    r"^BibKey:",                  # placeholder leaked from front-matter injection
    r"https?://",                 # URL-as-DOI
    r"^doi:\s*10\.\S+[<>]",       # markdown pollution
]


def is_fake_doi(doi: str) -> bool:
    if not doi:
        return False
    for pat in FAKE_DOI_PATTERNS:
        if re.search(pat, doi):
            return True
    return False


# ---------------------------------------------------------------------------
# .bib parsing
# ---------------------------------------------------------------------------

class BibEntry:
    """One BibTeX entry, with precomputed normalized fields.

    Implements dict-style `[]` / `.get()` access so legacy code that treats
    entries as dicts (scripts/bib_to_parent_store.py) keeps working.
    """

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

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def get(self, k: str, default: Any = None) -> Any:
        return getattr(self, k, default)

    def to_fix_payload(self) -> Dict[str, str]:
        return {
            "doi": self.doi,
            "title": self.title,
            "authors": self.author_full,
            "year": self.year,
            "journal": self.journal,
            "key": self.key,
        }


def _balanced_field(entry_text: str, name: str) -> str:
    """Extract field `name = { ... }` (or `name = "..."`) with balanced braces."""
    fm = re.search(r"\b" + re.escape(name) + r"\s*=\s*\{", entry_text, re.IGNORECASE)
    if not fm:
        fm = re.search(r"\b" + re.escape(name) + r"\s*=\s*\"", entry_text, re.IGNORECASE)
        if not fm:
            return ""
        k = fm.end()
        end = entry_text.find('"', k)
        return entry_text[k:end] if end >= 0 else ""
    k = fm.end()
    depth = 1
    j = k
    buf = []
    while j < len(entry_text) and depth > 0:
        c = entry_text[j]
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


def _parse_authors(author_str: str) -> Tuple[str, str]:
    """Return (author_full, author_first_lastname) from a BibTeX author field."""
    if not author_str:
        return "", ""
    author_str_clean = re.sub(r"[{}]", "", author_str)
    author_full = re.sub(r"\s+and\s+", ", ", author_str_clean).strip()
    first = re.split(r"\s+and\s+", author_str_clean)[0]
    if "," in first:
        first_lastname = first.split(",")[0].strip()
    else:
        parts = first.split()
        first_lastname = parts[-1] if parts else ""
    return author_full, first_lastname


def parse_bib_entries(bib_path: Path) -> List[BibEntry]:
    """Parse a BibTeX file into a list of BibEntry.

    Uses balanced-brace field extraction. Handles tab-indented Zotero exports:
    the year/date regexes anchor on `^\s*` (a previous copy used bare `^year`,
    which never matched tab-indented lines, silently degrading year extraction
    to the entry-key fallback).
    """
    content = bib_path.read_text(encoding="utf-8")
    entries_raw = re.split(r"\n(?=@\w+\{)", content)
    result: List[BibEntry] = []
    for e in entries_raw:
        m = re.match(r"^@(\w+)\{([^,]+),", e.strip())
        if not m:
            continue
        key = m.group(2).strip()

        doi = _balanced_field(e, "doi")
        title = _balanced_field(e, "title")
        author_str = _balanced_field(e, "author")
        journal = _balanced_field(e, "journaltitle") or _balanced_field(e, "journal")
        abstract = _balanced_field(e, "abstract")
        author_full, first_lastname = _parse_authors(author_str)

        # year: prefer year field, then date field, then key
        year = ""
        m_year = re.search(r"^\s*year\s*=\s*\{(\d{4})\}", e, re.MULTILINE)
        if m_year:
            year = m_year.group(1)
        if not year:
            m_date = re.search(r"^\s*date\s*=\s*\{(\d{4})", e, re.MULTILINE)
            if m_date:
                year = m_date.group(1)
        if not year:
            m_key = re.search(r"(\d{4})", key)
            if m_key:
                year = m_key.group(1)

        result.append(BibEntry(
            key=key, doi=doi, title=title,
            author_first_lastname=first_lastname,
            author_full=author_full, year=year,
            abstract=abstract, journal=journal,
        ))
    return result


class BibIndex:
    """Indexes over a list of BibEntry for fast lookup."""

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

        Both sides are normalized before the lastname comparison — filenames
        like `Boström_...` or `Abdul-Wajid_...` must match bib lastnames
        `bostrom` / `abdulwajid`.
        """
        lastname, year, title_norm = fname_key
        if not year:
            return None, "no_year"
        key = (normalize(lastname), year)
        candidates = [
            be for be in self.lastname_year_index.get(key, [])
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
                    if abs(int(paper_year) - int(be.year)) > 1:
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
