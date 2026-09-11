#!/usr/bin/env python3
"""
enrich_meta.py — Fill missing paper metadata (DOI/PMID/title/year/journal/authors)
at add-time, by querying Crossref/PubMed BEFORE indexing.

Pipeline position (add_papers.py):
    PDF → MD → [THIS MODULE] → index_single_paper.index_paper()

What it does with one markdown file:
  1. Scan the text head (~first 6000 chars) for DOI / PMID / PMCID identifiers
     (broader patterns than chunking.extract_meta: tolerates spaces after the
     colon, dx.doi.org, trailing punctuation).
  2. If a DOI was found → Crossref verify_doi() → title/year/journal/authors.
     Then one PubMed ID-converter call → PMID/PMCID (when the paper is in PMC).
  3. If no DOI → Crossref title search (query.bibliographic) gated by title
     similarity ≥ TITLE_SIM_ACCEPT (prefix-recall semantics, same scorer
     family as meta_audit) → if a confident match exists, adopt its DOI and
     repeat step 2's PubMed call for the PMID.
  4. Write the merged metadata as a FRONT-MATTER BLOCK at the top of the .md:

        Title: ...
        Authors: ...
        Year: ...
        Journal: ...
        doi: ...
        PMID: ...
        PMCID: ...

     Downstream readers of this block (no changes needed on their side):
       - chunking.extract_meta       (first-line title heuristic skips these
                                      labelled lines; doi:/PMID: regexes hit)
       - hybrid_search._meta_from_frontmatter  (Title:/Authors:/Year:/Journal:)
     Values scraped from the original text are preserved over registry values
     (never overwrite a value the document itself states), EXCEPT that registry
     values fill empty fields. Idempotent: a second run replaces the previous
     block instead of stacking a second one.

Reuses (single source of truth, no code duplication):
  - scripts/metadata/meta_audit.py :: CrossrefClient, PubmedClient  (throttled,
    error-tolerant, with call/error counters)
  - scripts/bib_utils.py            :: title_tokens, jaccard, is_doi_like, and
                                       the canonical title gates (is_junk_title
                                       mode="search", title_search_sim) merged
                                       here 2026-09-10 from this module's old
                                       behavior-forked copies
  - src/identifiers.py              :: normalize_doi
The import is deferred until first use so that --no-enrich runs and library
imports of this module never pay the meta_audit import cost (it is heavy).

Offline behaviour: every network failure degrades to "keep what regex found"
— enrichment never blocks indexing. In that spirit this module RAISES NOTHING:
all failures return partial results.

CLI (standalone, for back-filling existing md files without re-indexing):
    python3 -B src/enrich_meta.py file1.md file2.md
    python3 -B src/enrich_meta.py --kb geo_rag /path/to/md_dir/   # whole dir
    python3 -B src/enrich_meta.py --dry-run --verbose file.md     # no writes
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: how much of the document head to scan for identifiers (DOI/PMID live in
#: the first pages; scanning more just risks matching reference-list DOIs)
SCAN_CHARS = 6000

#: accept a Crossref title-search hit only at this prefix-recall similarity
#: (meta_audit uses 0.80 for the same job; 0.75 trades a little precision
#: for recall on truncated filename-derived titles)
TITLE_SIM_ACCEPT = 0.75

#: (min-title-token floor and the junk-title classes live in
#: bib_utils.is_junk_title — the canonical gate, mode="search" here)

# Broader identifier patterns than chunking.extract_meta's:
#   - tolerate whitespace after "doi:" / "DOI:"
#   - accept dx.doi.org as well as doi.org
#   - case-insensitive
#   - DOI body: 10.xxxx/anything-but-whitespace, trailing punctuation AND
#       markdown residue ('**') stripped, then canonicalized via
#       identifiers.normalize_doi (handles full-width CJK digits too)
#   - PMID: anchored to a LINE START labelled 'PMID:' — body-text digits and
#     in-reference PMIDs (PMC export footer style) are not line-starts
_RE_DOI = re.compile(
    r"(?:doi\s*[:=]\s*|https?://(?:dx\.)?doi\.org/)"
    r"(10\.\d{4,9}/[^\s\"<>\]]+)",
    re.IGNORECASE)
_RE_PMID = re.compile(r"^\**\s*PMID[ :\-]+\**(\d{5,9})", re.IGNORECASE | re.MULTILINE)
_RE_PMCID = re.compile(r"(PMC\d{4,8})", re.IGNORECASE)


def _normalize_doi(raw: str) -> str:
    """Canonicalize a DOI string via src/identifiers (lazy import), with a
    guaranteed markdown-residue strip (**, ), ], backticks) on the tail."""
    try:
        from identifiers import normalize_doi
        nd = normalize_doi(raw) or ""
    except Exception:
        nd = raw.strip()
    nd = re.sub(r"[\*`\)\]']+$", "", nd.strip()).rstrip(".,;")
    return nd.lower()

#: the front-matter block this module writes. ORDER MATTERS: chunking's
#: title heuristic reads the FIRST non-empty line and rejects lines starting
#: with these labels, so putting "Title:" first keeps that heuristic off,
#: and hybrid_search's anchored regexes (^Title:) find their fields.
_FM_FIELDS = ["Title", "Authors", "Year", "Journal", "doi", "PMID", "PMCID"]

#: A front-matter block written by THIS module sits at the very top of the
#: file: optional blank lines, then consecutive label lines, then a blank
#: separator. Both the prev-parse and the strip must ONLY recognize that
#: shape — a naive `^label:.*` MULTILINE scan treats deep-body reference
#: lines like `doi: 10.1111/j.1464-410X.2008.07987.x` (line ~912 of a
#: EuropePMC-style md) as front-matter, hijacking the paper's identity with
#: its LAST reference's record (found live 2026-09-10 on the JMedLife
#: horseshoe-kidney review) and deleting body lines on strip.
_FM_BLOCK_RE = re.compile(
    r"\A\s*^(?:Title|Authors|Year|Journal|doi|PMID|PMCID|BibKey):.*"
    r"(?:\n(?:(?:Title|Authors|Year|Journal|doi|PMID|PMCID|BibKey):.*|\s*))*\n*",
    re.MULTILINE)

# ---------------------------------------------------------------------------
# Local scanning (no network)
# ---------------------------------------------------------------------------

def scan_identifiers(text_head: str) -> Dict[str, str]:
    """Extract DOI/PMID/PMCID from the document head with tolerant regexes."""
    out: Dict[str, str] = {}
    m = _RE_DOI.search(text_head)
    if m:
        doi = m.group(1).rstrip(".,;)")   # sentence-ending punctuation
        out["doi"] = doi
    m = _RE_PMID.search(text_head)
    if m:
        out["pmid"] = m.group(1)
    m = _RE_PMCID.search(text_head)
    if m:
        out["pmcid"] = m.group(1).upper()
    return out


def local_meta(md_path: Path) -> Dict[str, str]:
    """Everything extractable WITHOUT network: identifiers + title/year guess.

    Mirrors chunking.extract_meta's title fallback (filename stem when the
    first line is not a plausible title) so enrichment has a search key even
    for md files whose body lost the title (two-column PDFs do this).
    """
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    head = text[:SCAN_CHARS]

    meta: Dict[str, str] = {}
    meta.update(scan_identifiers(head))

    # year: prefer the FILENAME's year (curated corpora encode it:
    # 'Lastname_1975_Title' / 'Ren et al. - 2020 - ...'), else first 4-digit
    # in the head (unreliable: matches citation years like 1914)
    m = re.search(r"(?:^|[\s_\-.])((?:19|20)\d{2})(?:[\s_\-.]|$)", md_path.stem)
    if m:
        meta["year"] = m.group(1)
    else:
        m = re.search(r"\b((?:19|20)\d{2})\b", head)
        if m:
            meta["year"] = m.group(1)

    # title: first plausible line, else filename stem (same rule as chunking)
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if lines:
        first = lines[0]
        _junk_prefixes = ("#", "title:", "authors:", "year:", "journal:",
                          "doi:", "pmid:", "pmcid:", "bibkey:", "keywords:")
        _fl = first.lower()
        looks_labelled = (any(_fl.startswith(p) for p in _junk_prefixes)
                          or _RE_DOI.search(first) or _RE_PMID.search(first))
        if (20 < len(first) < 300 and not looks_labelled):
            meta["title"] = first
        else:
            name = re.sub(r"^\+[\w-]*\+", "", md_path.stem)   # "+KB+ " prefix
            name = re.sub(r"\.md$|\.pdf$", "", name).strip()
            if 10 < len(name) <= 200:
                meta["title"] = name
    return meta


# ---------------------------------------------------------------------------
# Network enrichment (deferred imports — meta_audit is heavy)
# ---------------------------------------------------------------------------

_clients: Dict[str, Any] = {}


def _get_clients():
    """Lazy singleton CrossrefClient/PubmedClient borrowed from meta_audit."""
    if "crossref" in _clients:
        return _clients
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))                       # src/ (identifiers)
    sys.path.insert(0, str(here.parent / "scripts"))    # bib_utils
    sys.path.insert(0, str(here.parent / "scripts" / "metadata"))  # meta_audit
    from meta_audit import CrossrefClient, PubmedClient, DEFAULT_CROSSREF_MAILTO
    _clients["crossref"] = CrossrefClient(mailto=DEFAULT_CROSSREF_MAILTO)
    _clients["pubmed"] = PubmedClient()
    return _clients


def _title_sim(claimed: str, result: str) -> float:
    """Search-mode prefix-recall similarity — canonical impl in bib_utils."""
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent / "scripts"))
    from bib_utils import title_search_sim
    return title_search_sim(claimed, result, accept=TITLE_SIM_ACCEPT)


def _crossref_doi_lookup(doi: str) -> Optional[Dict[str, str]]:
    """Crossref record for a known DOI → title/year/journal/authors (+doi)."""
    try:
        cl = _get_clients()
        rec = cl["crossref"].verify_doi(doi)
        if not rec:
            return None
        return {k: str(v) for k, v in rec.items() if v}
    except Exception:
        return None


def _crossref_title_search(title: str, year: str = "") -> Optional[Dict[str, str]]:
    """Best Crossref hit for a title, gated by similarity. None if no match."""
    try:
        cl = _get_clients()
        results = cl["crossref"].search(title, year=year) or []
    except Exception:
        return None
    best, best_sim = None, 0.0
    for r in results:
        if not r.get("doi"):
            continue
        sim = _title_sim(title, r.get("title") or "")
        if sim > best_sim:
            best, best_sim = r, sim
    if best and best_sim >= TITLE_SIM_ACCEPT:
        return {k: str(v) for k, v in best.items() if v}
    return None


def _pubmed_pmid_for_doi(doi: str) -> Optional[str]:
    """PMID for a DOI via PubMed esearch (title-free, exact)."""
    try:
        cl = _get_clients()
        # PubmedClient.search is title-based; use its raw esearch instead:
        import urllib.parse, urllib.request, xml.etree.ElementTree as ET
        params = {
            "db": "pubmed", "term": f"{doi}[DOI]",
            "retmax": "1", "tool": "bib_rag_enrich", "email": "bib-rag@example.com",
        }
        url = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
               + urllib.parse.urlencode(params))
        cl["pubmed"]._throttle()
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as r:
            root = ET.parse(r).getroot()
        ids = [e.text for e in root.findall(".//Id") if e and e.text]
        return ids[0] if ids else None
    except Exception:
        return None


def _pubmed_pmcid_for_pmid(pmid: str) -> Optional[str]:
    """PMCID for a PMID via esummary (only called when a PMID is known)."""
    try:
        cl = _get_clients()
        rec = cl["pubmed"].summary(pmid)
        if rec and rec.get("pmcid"):
            return rec["pmcid"]
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Merge policy
# ---------------------------------------------------------------------------

def _looks_like_filename(s: str) -> bool:
    """A title that is really just the md filename stem (Author123_Journal
    style): no spaces + underscores/camel-case digits, or starts with the
    T_A/T_C/T_D tier prefixes used by curated corpora."""
    if not s:
        return False
    if re.match(r"^[A-Z]_[A-Z]_", s) or re.match(r"^T_[A-Z]_", s):
        return True
    if " " not in s and ("_" in s or re.search(r"\d{3,}", s)):
        return True
    return False


def _is_junk_value(k: str, v: str) -> bool:
    """A text-scraped value that is demonstrably garbage (the chunker's
    first-4-digit year rule and journal regex pick up body text):
      year    — outside plausible publication range (pre-1900 is body noise)
      journal — contains newlines, is shorter than 4 chars, or looks like a
                sentence fragment (starts lowercase / > 80 chars)
    """
    if not v:
        return False
    if k == "year":
        try:
            return not (1900 <= int(v) <= 2035)
        except ValueError:
            return True
    if k == "journal":
        return ("\n" in v or len(v) < 4 or len(v) > 80
                or v[:1].islower() or "This study" in v)
    return False


def _merge(local: Dict[str, str], remote: Optional[Dict[str, str]],
           search_hit: Optional[Dict[str, str]]) -> Dict[str, str]:
    """Registry-verified values replace demonstrably junk local values;
    otherwise document-stated values win; registry fills blanks. The
    title rule: a filename-derived stem loses to a registry title."""
    merged = dict(local)
    for src in (search_hit, remote):
        if not src:
            continue
        for k, v in src.items():
            if not v:
                continue
            if k == "title" and _looks_like_filename(merged.get("title", "")):
                merged["title"] = str(v)
            elif _is_junk_value(k, merged.get(k, "")):
                merged[k] = str(v)   # verified beats garbage
            elif not merged.get(k):
                merged[k] = str(v)
    return merged


# ---------------------------------------------------------------------------
# Front-matter I/O
# ---------------------------------------------------------------------------

def format_frontmatter(meta: Dict[str, str]) -> str:
    """The block written to the top of the md. Only non-empty fields.

    Field formats MATCH chunking.extract_meta's regexes exactly so the
    indexer picks them up without touching shared code:
      - 'doi:' written WITHOUT a trailing space ('doi:10.x/yyy') — the
        chunker's pattern (?:doi:|DOI:|...)(10\.\S+) does not tolerate one.
      - 'PMID:'/'PMCID:' WITH a space — those patterns use \s*.
    """
    lines = []
    for f in _FM_FIELDS:
        v = (meta.get(f.lower()) or meta.get(f) or "").strip()
        if not v:
            continue
        sep = "" if f == "doi" else " "
        lines.append(f"{f}:{sep}{v}")
    return "\n".join(lines) + "\n\n" if lines else ""


def upsert_frontmatter(md_path: Path, meta: Dict[str, str]) -> bool:
    """Write/replace the front-matter block. Returns True if file changed."""
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    body = _FM_BLOCK_RE.sub("", text, count=1).lstrip("\n")
    block = format_frontmatter(meta)
    if not block:
        return False
    new = block + body
    if new == text:
        return False
    md_path.write_text(new, encoding="utf-8")
    return True


def _strip_existing_fm(text: str) -> str:
    return _FM_BLOCK_RE.sub("", text, count=1).lstrip("\n")


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def _is_junk_title(s: str) -> bool:
    """Search-mode title gate — canonical impl in bib_utils."""
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent / "scripts"))
    from bib_utils import is_junk_title
    return is_junk_title(s, mode="search")


def enrich_md(md_path: Path, verbose: bool = False) -> Dict[str, str]:
    """Enrich one markdown file's metadata in place. Returns final meta.

    Never raises: network/parsing failures degrade to local-only results.
    """
    md_path = Path(md_path)
    loc = local_meta(md_path)

    # read existing front-matter (a re-run keeps previous gains). ONLY the
    # leading block written by upsert_frontmatter counts — deep-body `doi:`
    # reference lines are NOT front-matter (citation-hijack defense).
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    prev: Dict[str, str] = {}
    m = _FM_BLOCK_RE.match(text)
    if m:
        for line in m.group(0).split("\n"):
            mm = re.match(r"^(Title|Authors|Year|Journal|doi|PMID|PMCID|BibKey):\s*(.+)$", line)
            if mm:
                prev[mm.group(1).lower()] = mm.group(2).strip()
    for k, v in prev.items():
        if v and not loc.get(k):
            loc[k] = v

    # ---- resolve DOI -------------------------------------------------
    doi = _normalize_doi(loc.get("doi", ""))
    loc["doi"] = doi
    remote = None
    if doi:
        remote = _crossref_doi_lookup(doi)

    # PMCID embedded in the filename (curated corpora: t_PMC11891615.pdf)
    if not loc.get("pmcid"):
        m = re.search(r"(PMC\d{4,8})", md_path.stem, re.IGNORECASE)
        if m:
            loc["pmcid"] = m.group(1).upper()

    # ---- title search only with a plausible key ----------------------
    # Citation-hijack defense: only search when the title is a REAL title
    # (not a journal header / junk banner). Then, the hit is adopted ONLY
    # if its Crossref record's title actually matches our key — a mismatch
    # means the search found a DIFFERENT paper (a reference this article
    # cites), which must not steal its identity.
    search_hit = None
    title_key = loc.get("title", "")
    # (single gate now: canonical is_junk_title(search) already enforces the
    #  >=3 real-token minimum the old explicit len(_tokens(...)) check did)
    if not remote and title_key and not _is_junk_title(title_key) \
            and not _looks_like_filename(title_key):
        cand = _crossref_title_search(title_key, loc.get("year", ""))
        if cand and cand.get("doi"):
            rec = _crossref_doi_lookup(_normalize_doi(cand["doi"]))
            if rec and _title_sim(title_key, rec.get("title", "")) >= TITLE_SIM_ACCEPT:
                search_hit = cand
                nd = _normalize_doi(cand["doi"])
                if nd:
                    loc.setdefault("doi", nd)
                    remote = rec

    # A Crossref DOI-verified record is AUTHORITATIVE for year/journal/
    # authors (the text-scan's first-4-digit year and journal regex are
    # demonstrably unreliable); title only replaces junk/filename titles.
    if remote:
        for k in ("year", "journal", "authors"):
            if remote.get(k):
                loc[k] = remote[k]
        if remote.get("title") and (
                not loc.get("title") or _is_junk_title(loc["title"])
                or _looks_like_filename(loc["title"])):
            loc["title"] = remote["title"]

    merged = _merge(local=loc, remote=remote, search_hit=search_hit)

    # ---- PMID/PMCID via NCBI ID converter (doi or pmcid → ids) -------
    conv = _ncbi_idconv(merged.get("doi", ""), merged.get("pmcid", ""),
                        merged.get("pmid", ""))
    if conv:
        for k in ("pmid", "pmcid", "doi"):
            if conv.get(k) and not merged.get(k):
                merged[k] = conv[k]

    # ---- verify an un-anchored PMID against the document title -------
    # A PMID that survived front-matter persistence (or a PMC-footer line)
    # must belong to THIS paper: esummary's title has to match ours. When
    # there is no DOI-verified record and no usable title, keep it (a
    # curated corpus that labelled the file by PMID knows what it did).
    if merged.get("pmid") and not remote and merged.get("title") \
            and not _looks_like_filename(merged["title"]) \
            and not _is_junk_title(merged["title"]):
        summ = _pubmed_summary(merged["pmid"])
        if summ and summ.get("title") \
                and _title_sim(merged["title"], summ["title"]) < 0.5:
            merged.pop("pmid", None)
            merged.pop("pmcid", None)

    # ---- scrub junk that no registry value replaced -------------------
    # better an empty field than a demonstrably wrong one
    for k in ("year", "journal"):
        if _is_junk_value(k, merged.get(k, "")):
            merged.pop(k, None)
    if merged.get("doi") and not re.match(r"^10\.\d{4,9}/", merged["doi"]):
        merged.pop("doi", None)

    if verbose:
        print(f"    [enrich] {md_path.name}")
        print(f"      local : { {k: v[:40] for k, v in loc.items()} }")
        if remote:
            print(f"      remote: { {k: v[:40] for k, v in remote.items()} }")
        if search_hit:
            print(f"      search: { {k: v[:40] for k, v in search_hit.items()} }")

    upsert_frontmatter(md_path, merged)
    return merged


def _ncbi_idconv(doi: str = "", pmcid: str = "", pmid: str = "") -> Optional[Dict[str, str]]:
    """NCBI ID Converter: any of doi/pmcid/pmid → the other ids. One call."""
    have = [x for x in (pmid, pmcid, doi) if x]
    if not have:
        return None
    try:
        import urllib.parse, urllib.request
        ids = ",".join(have)   # idconv expects comma-separated ids
        url = ("https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
               "?ids=" + urllib.parse.quote(ids)
               + "&format=json&tool=bib_rag_enrich&email=bib-rag@example.com")
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as r:
            d = json.loads(r.read().decode())
        # multiple ids → multiple records; merge them (fields we lack win)
        out: Dict[str, str] = {}
        for rec in d.get("records") or []:
            for k in ("pmid", "pmcid", "doi"):
                if rec.get(k) and not out.get(k):
                    out[k] = str(rec[k])
        return out or None
    except Exception:
        return None


def _pubmed_summary(pmid: str) -> Optional[Dict[str, str]]:
    """PubMed esummary for a PMID (title etc.) — via meta_audit's client."""
    try:
        cl = _get_clients()
        return cl["pubmed"].summary(pmid)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _argv_kb() -> List[str]:
    """Strip --kb so argparse never sees it (mirrors add_papers.py)."""
    argv, kb = [], None
    for i, a in enumerate(sys.argv):
        if a == "--kb" and i + 1 < len(sys.argv):
            kb = sys.argv[i + 1]
        elif kb and sys.argv[i - 1] == "--kb":
            continue
        else:
            argv.append(a)
    if kb:
        os.environ["BIB_RAG_KB_NAME"] = kb
    return argv


def main():
    import argparse
    argv = _argv_kb()
    ap = argparse.ArgumentParser(
        description="Enrich md metadata (DOI/PMID/title/...) via Crossref/PubMed",
        epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="md file(s) or directory of .md files")
    ap.add_argument("--dry-run", action="store_true", help="no writes, print what would happen")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv[1:])

    mds: List[Path] = []
    for p in args.inputs:
        path = Path(p)
        if path.is_file() and path.suffix.lower() == ".md":
            mds.append(path)
        elif path.is_dir():
            mds.extend(sorted(path.glob("*.md")))

    if not mds:
        print("No markdown files found.")
        return 1

    print(f"🔎 enrich_meta: {len(mds)} file(s)")
    filled = 0
    for md in mds:
        before = local_meta(md)
        if args.dry_run:
            print(f"  [dry-run] {md.name}: local scan → { {k: v[:40] for k, v in before.items()} }")
            continue
        meta = enrich_md(md, verbose=args.verbose)
        gained = [k for k in ("doi", "pmid", "pmcid", "title", "year", "journal", "authors")
                  if meta.get(k) and not before.get(k)]
        print(f"  ✅ {md.name}: {'+' + ','.join(gained) if gained else 'no new fields'}")
        if gained:
            filled += 1
    print(f"\n{'(dry-run, no files written)' if args.dry_run else f'✅ Enriched {filled}/{len(mds)} file(s) with new fields'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())