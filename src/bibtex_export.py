#!/usr/bin/env python3
"""
bibtex_export.py — DOI → BibTeX export with missing-field completion.

Borrowed mechanism (paper-qa doi_to_bibtex + format_bibtex — see the
citation-management cross-survey notes): Crossref's official
`/transform/application/x-bibtex` endpoint renders a DOI into a full BibTeX
entry server-side; locally we then (a) rewrite the citation key into the
`lastname_year_firsttitleword` convention the rest of this toolkit uses,
(b) fill missing fields (author/journal/year/title) from parent_store
metadata when Crossref returns an incomplete entry, and (c) optionally
render an ASCII-safe key (diacritics stripped).

Design constraints:
  - The ONLY network call is the Crossref transform GET. Everything else is
    pure string work — zero LLM, graceful degradation on any failure
    (returns None / skips the DOI, never raises through the caller).
  - Reuses identifiers.normalize_doi so the key space matches zotero_match
    and the reference graph.
  - Offline path: bibtex_from_meta() synthesizes a minimal entry from
    parent_store metadata alone (no network) — quality flag included.

Usage:
    from bibtex_export import doi_to_bibtex, bibtex_from_meta, export_answers_bib
    entry = doi_to_bibtex("10.1016/j.ydbio.2021.01.002", mailto="you@example.com")

CLI:
    python3 bibtex_export.py --dois 10.1016/j.ydbio.2021.01.002 10.1038/...
        [--meta-dir <parent_store>] [--out refs.bib]
"""
from __future__ import annotations

import os
import re
import sys
import json
import time
import argparse
import unicodedata
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Tuple

try:
    from .identifiers import normalize_doi
except ImportError:  # src/ on sys.path directly (CLI/tests)
    try:  # bib_rag-package-try
        from .identifiers import normalize_doi
    except ImportError:  # flat (loose-script mode)
        from identifiers import normalize_doi

try:
    from .kb_config import get_config
except ImportError:
    try:  # bib_rag-package-try
        from .kb_config import get_config
    except ImportError:  # flat (loose-script mode)
        from kb_config import get_config

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

CROSSREF_TIMEOUT = _T = 15            # seconds per HTTP call
CROSSREF_MIN_INTERVAL = 1.0           # polite-pool throttle between calls
MAX_RETRIES = 2


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default) or default


# ---------------------------------------------------------------------------
# Key generation (paper-qa create_bibtex_key, adapted)
# ---------------------------------------------------------------------------

def _ascii_fold(text: str) -> str:
    """Strip diacritics (NFKD → drop combining marks), keep ASCII."""
    return "".join(
        c for c in unicodedata.normalize("NFD", text or "")
        if unicodedata.category(c) != "Mn"
    )


def make_bibtex_key(author_lastname: str, year: str, title: str,
                    existing: Optional[set] = None) -> str:
    """`lastname_year_firsttitleword` — Zotero/Better-BibTeX convention this
    repo already uses in filenames. Collision → a/b/c suffix (seerai rule)."""
    last = re.sub(r"[^a-zA-Z]", "", _ascii_fold(author_lastname or "unknown")).lower() or "unknown"
    y = re.sub(r"\D", "", str(year or ""))[:4] or "nd"
    # word chars incl. digits: gene names like Ephb1/EphrinB2 stay whole
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9]*|\d+", _ascii_fold(title or ""))
    first = (words[0] if words else "untitled").lower()
    key = f"{last}_{y}_{first}"
    taken = existing or set()
    if key not in taken:
        return key
    for suffix in "abcdefghij":
        cand = f"{key}{suffix}"
        if cand not in taken:
            return cand
    return f"{key}_{abs(hash(key)) % 10000}"


# ---------------------------------------------------------------------------
# Crossref transform endpoint (paper-qa doi_to_bibtex)
# ---------------------------------------------------------------------------

def _crossref_transform(doi: str, mailto: str) -> Optional[str]:
    """GET https://api.crossref.org/works/<doi>/transform/application/x-bibtex.
    Returns the rendered BibTeX string or None on any failure."""
    url = (f"https://api.crossref.org/works/{urllib.parse.quote(doi, safe='')}"
           f"/transform/application/x-bibtex?mailto={urllib.parse.quote(mailto)}")
    req = urllib.request.Request(url)
    req.add_header("User-Agent",
                   f"bib_rag_bibtex_export/1.0 (mailto:{mailto})")
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=_T) as r:
                body = r.read().decode("utf-8", errors="replace").strip()
            if body.startswith("@"):
                return body
        except Exception:
            if attempt == MAX_RETRIES - 1:
                return None
            time.sleep(1.5 * (attempt + 1))  # linear backoff, polite pool
    return None


def _rewrite_key(bibtex: str, new_key: str) -> str:
    """Replace the entry key Crossref generated with our canonical one."""
    m = re.match(r"(@\w+\s*\{\s*)([^,\s]+)(\s*,)", bibtex)
    if not m:
        return bibtex
    return bibtex[:m.start(2)] + new_key + bibtex[m.end(2):]


_BIBTEX_FIELD_RE = re.compile(
    r"^\s*(\w+)\s*=\s*([\{\"']).*?\2\s*,?\s*$", re.I | re.S | re.M)


def bibtex_fields(bibtex: str) -> Dict[str, str]:
    """Best-effort field extraction from a rendered BibTeX string. Field
    values are brace-balanced (Crossref nests braces in protective wraps)."""
    out: Dict[str, str] = {}
    for m in re.finditer(r"(\w+)\s*=\s*", bibtex):
        start = m.end()
        rest = bibtex[m.end():]
        if not rest or rest[0] not in "{\"'":
            continue
        open_ch = rest[0]
        close_ch = "}" if open_ch == "{" else open_ch
        depth = 1 if open_ch == "{" else 0
        i = 1
        while i < len(rest) and depth > 0:
            c = rest[i]
            if c == "{" and open_ch == "{":
                depth += 1
            elif c == "}" and open_ch == "{":
                depth -= 1
                if depth == 0:
                    break
            elif c == close_ch and open_ch != "{":
                break
            i += 1
        if open_ch == "{":
            out[m.group(1).lower()] = rest[1:i].strip()
        else:
            out[m.group(1).lower()] = rest[1:i].strip()
    return out


def _fill_missing_fields(bibtex: str, meta: Dict[str, str]) -> str:
    """Insert meta-sourced values for fields Crossref omitted
    (paper-qa missing_replacements, string-level — no pybtex dependency)."""
    fields = bibtex_fields(bibtex)
    fills = []
    for field in ("author", "journal", "year", "title"):
        if fields.get(field):
            continue
        val = (meta.get(field) or "").strip()
        if not val:
            continue
        if field == "author" and "," not in val and ";" in val:
            # 'Last, First; Last, First' expected; pass through as-is otherwise
            val = " and ".join(p.strip() for p in val.split(";"))
        fills.append((field, val))
    if not fills:
        return bibtex
    # append before the closing brace of the entry
    idx = bibtex.rstrip().rfind("}")
    if idx <= 0:
        return bibtex
    add = "".join(f",\n  {f} = {{{v}}}" for f, v in fills)
    return bibtex[:idx].rstrip().rstrip(",") + add + "\n" + bibtex[idx:]


def doi_to_bibtex(doi: str, mailto: str = "",
                  meta: Optional[Dict[str, str]] = None,
                  existing_keys: Optional[set] = None) -> Optional[str]:
    """DOI → complete BibTeX entry (Crossref transform + local completion).

    Returns None when the DOI does not resolve or Crossref is unreachable.
    `meta` (optional {author, journal, year, title}) fills missing fields.
    """
    nd = normalize_doi(doi)
    if not nd:
        return None
    mailto = mailto or _env("CROSSREF_MAILTO", "bib_rag@example.org")
    raw = _crossref_transform(nd, mailto)
    if not raw:
        return None
    # canonical key: first author lastname + year + first title word
    fields = bibtex_fields(raw)
    year = (fields.get("year") or (meta or {}).get("year") or "")
    author_lead = re.split(r"\s+and\s+", fields.get("author", ""))[0]
    last = author_lastname(author_lead=author_lead, meta=meta or {})
    key = make_bibtex_key(last, year, fields.get("title", ""), existing_keys)
    out = _rewrite_key(raw, key)
    if meta:
        out = _fill_missing_fields(out, meta)
    return out


def author_lastname(author_lead: str = "", meta: Optional[Dict[str, str]] = None) -> str:
    """Pull the surname out of a Crossref author string ('Last, First') or
    fall back to the parent filename stored in meta['source']."""
    meta = meta or {}
    if author_lead:
        lead = author_lead.replace("{", "").replace("}", "").strip()
        if "," in lead:
            return lead.split(",")[0]
        return lead.split()[-1] if lead.split() else ""
    src = meta.get("source", "")
    m = re.match(r"^([A-Za-z][\w'\-]*?)[ _\-]+", os.path.basename(src))
    return m.group(1) if m else ""


# ---------------------------------------------------------------------------
# Offline path: synthesize from parent_store metadata alone
# ---------------------------------------------------------------------------

def bibtex_from_meta(meta: Dict[str, str],
                     existing_keys: Optional[set] = None) -> Optional[str]:
    """Minimal @article/@misc entry from parent_store metadata. No network.
    PubMed-style author lists ('Last FM; Last FM') are converted to BibTeX
    'Last, FM and Last, FM'. Returns None when there is not even a title."""
    title = (meta.get("title") or "").strip()
    if not title:
        return None
    year = (meta.get("year") or "").strip()
    authors = (meta.get("authors") or meta.get("author") or "").strip()
    if authors and ";" in authors:
        # PubMed format 'Paulson AF; Fang X' → BibTeX 'Paulson, AF and Fang, X'
        parts = []
        for p in (x.strip() for x in authors.split(";")):
            if not p:
                continue
            toks = p.split()
            if len(toks) >= 2:          # 'Last FM' → 'Last, FM'
                parts.append(f"{toks[0]}, {' '.join(toks[1:])}")
            else:
                parts.append(p)
        authors = " and ".join(parts)
    journal_raw = meta.get("journal", "") or ""
    journal = " ".join(journal_raw.split())  # collapse newlines/whitespace
    # Scrape-noise guard: a real journal name is short and has no fragment
    # junk. 'Research article Neuroscience S' came from a multi-line HTML
    # scrape — when the collapsed string doesn't look like a journal, drop it
    # (the entry falls back to @misc; Crossref online mode would fill it).
    # Word-count and stopword checks catch concatenations of page fragments
    # that pass the character-class regex.
    _JOURNAL_STOP = ("research", "article", "preprint", "published", "volume",
                     "issue", "copyright", "license", "available", "download")
    if journal and (
            len(journal) > 60
            or len(journal.split()) > 5
            or any(w in journal.lower().split() for w in _JOURNAL_STOP)
            or not re.match(r"^[A-Za-z][A-Za-z0-9 .&'()-]+$", journal)):
        journal = ""
    doi = normalize_doi(meta.get("doi", "") or "")
    last = author_lastname(meta=meta)
    key = make_bibtex_key(last, year, title, existing_keys)
    etype = "article" if journal else "misc"
    if authors:
        authors = bib_author_cleanup(authors)
    fields = [f"  title = {{{title}}}"]
    if authors:
        fields.append(f"  author = {{{authors}}}")
    if journal:
        fields.append(f"  journal = {{{journal}}}")
    if year:
        fields.append(f"  year = {{{year}}}")
    if doi:
        fields.append(f"  doi = {{{doi}}}")
    lines = [f"@{etype}{{{key},"] + [f + "," for f in fields[:-1]] + [fields[-1], "}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Batch export from parent_store (per-paper meta extraction)
# ---------------------------------------------------------------------------

def load_paper_meta(parent_store_dir: str, source: str) -> Dict[str, str]:
    """{title, year, journal, authors, doi} for a source file, from its
    parent_store JSON (first parent carries the meta). Accepts the source
    filename ('paper.md' → 'paper_md.json') or the store filename directly
    ('paper_md.json'); tries both candidate paths."""
    stem = os.path.splitext(os.path.basename(source))[0]
    safe = re.sub(r"[^\w\-]", "_", stem)[:100]
    candidates = [os.path.join(parent_store_dir, f"{safe}.json")]
    # source '10068468.md' → sanitized '10068468' → store may be '10068468_md.json'
    if not safe.endswith("_md"):
        alt = os.path.join(parent_store_dir, f"{safe}_md.json")
    else:
        alt = os.path.join(parent_store_dir, f"{safe[:-3]}.json")
    path = None
    for cand in (os.path.join(parent_store_dir, f"{safe}.json"), alt):
        if os.path.exists(cand):
            path = cand
            break
    if path is None:
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            parents = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    meta = (parents[0].get("meta", {}) or {}) if parents else {}
    # Filename-derived fallbacks for known-bad meta shapes (metadata pipeline
    # fills these properly via bind_zotero / backfill; until then the .bib
    # must not inherit them): titles like "Last et al. - 2019 - Real Title"
    # or "Last - 2005 - Real Title", years scraped wrong (eLife posts the
    # submission year).
    title = meta.get("title", "") or ""
    # Zotero filename convention heads: "Surname et al. - YYYY - ",
    # "Surname - YYYY - ", or bare "Surname - Title" (preprints). The head
    # token contains no spaces, so real titles beginning with a capitalized
    # word after " - " are stripped only when the stored title actually
    # matches the source filename head (i.e. the scrape copied the filename).
    def _strip_head(t: str) -> str:
        m = re.match(r"^[A-Za-z][\w'\-]* et al\. - (?:\d{4} - )?", t) \
            or re.match(r"^[A-Za-z][\w'\-]* - \d{4} - ", t)
        if m:
            return t[m.end():].strip()
        # "Surname - Title" without year: only strip when a meta.key exists
        # (bound record ⇒ title from the filename is plausible) — otherwise
        # a legit title like "Eph - signaling" would be mangled.
        m2 = re.match(r"^([A-Za-z][\w'\-]*) - (?=[A-Z])(.+)$", t)
        if m2 and meta.get("key"):
            return m2.group(2).strip()
        return t
    title = _strip_head(title)
    year = str(meta.get("year", "") or "")
    if not year:
        m_y = re.search(r"\b(19|20)\d{2}\b", title)
        if m_y:
            year = m_y.group(0)
    return {
        "title": title,
        "year": year,
        "journal": meta.get("journal", "") or "",
        "authors": meta.get("authors", "") or meta.get("author", "") or "",
        "doi": meta.get("doi", "") or "",
        "key": meta.get("key", "") or "",
        "source": source,
    }


def export_answers_bib(sources: List[str], out_path: str, mailto: str = "",
                       offline: bool = False,
                       store_dir: Optional[str] = None) -> Dict:
    """Export a References .bib for the given source filenames.

    online  : Crossref transform per DOI (throttled), metadata fill on top
    offline : synthesized from parent_store meta only
    `store_dir` overrides the active library's parent_store (tests / other KBs).
    Returns {written, skipped, errors, out_path}.
    """
    if store_dir is None:
        cfg = get_config()
        store = cfg["parent_store_dir"]
    else:
        store = store_dir
    existing: set = set()
    entries: List[str] = []
    written = skipped = errors = 0
    for src in sources:
        meta = load_paper_meta(store, src)
        entry = None
        if offline:
            entry = bibtex_from_meta(meta, existing)
        else:
            doi = normalize_doi(meta.get("doi", ""))
            if doi:
                entry = doi_to_bibtex(doi, mailto=mailto, meta=meta,
                                      existing_keys=existing)
            if entry is None:
                entry = bibtex_from_meta(meta, existing)  # graceful fallback
        if not entry:
            skipped += 1
            continue
        key = entry.split("{", 1)[1].split(",", 1)[0]
        existing.add(key)
        entries.append(entry)
        written += 1
    if entries:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(entries) + "\n")
    return {"written": written, "skipped": skipped, "errors": errors,
            "out_path": out_path}


def bib_author_cleanup(authors: str) -> str:
    """Normalize an already-authored BibTeX author field.

    Handles the two real shapes in parent_store meta:
      'A, B; C, D' / 'A, B, C, D' (comma-split display form, e.g. Zotero
      flattened) → 'A, B and C, D'; collapses newlines/extra whitespace.
    Semicolon form is handled by bibtex_from_meta (PubMed style).
    """
    a = " ".join(str(authors or "").split())  # collapse newlines + doubles
    if not a:
        return a
    if " and " in a:
        return a  # already BibTeX form
    if ";" in a:
        return " and ".join(p.strip() for p in a.split(";") if p.strip())
    # 'A, B, C, D' — comma-separated display list; pairwise join.
    parts = [p.strip() for p in a.split(",") if p.strip()]
    if len(parts) < 2:
        return a
    # Heuristic: Zotero flatten produced 'Last,First,Last,First' pairs →
    # recombine into 'Last,First' chunks.
    if len(parts) % 2 == 0:
        pairs = [f"{parts[i]}, {parts[i+1]}" for i in range(0, len(parts), 2)]
        return " and ".join(pairs)
    return " and ".join(parts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="DOI → BibTeX export (bib_rag)")
    ap.add_argument("--dois", nargs="*", help="DOIs to convert")
    ap.add_argument("--meta-dir", help="parent_store dir (for --sources mode)")
    ap.add_argument("--sources", nargs="*", help="source filenames to export")
    ap.add_argument("--out", required=True, help="output .bib path")
    ap.add_argument("--mailto", default=_env("CROSSREF_MAILTO"),
                    help="contact email for Crossref polite pool (or CROSSREF_MAILTO env)")
    ap.add_argument("--offline", action="store_true",
                    help="synthesize from parent_store metadata only (no network)")
    args = ap.parse_args()

    if args.dois:
        ok = 0
        for doi in args.dois:
            entry = doi_to_bibtex(doi, mailto=args.mailto)
            if entry:
                ok += 1
                print(entry, "\n")
            else:
                print(f"% FAILED: {doi}", file=sys.stderr)
        print(f"% {ok}/{len(args.dois)} entries fetched", file=sys.stderr)
        return 0 if ok else 1

    if args.sources:
        if args.meta_dir:
            os.environ["BIB_RAG_PARENT_DIR_OVERRIDE"] = args.meta_dir
        res = export_answers_bib(args.sources, args.out, mailto=args.mailto,
                                 offline=args.offline)
        print(json.dumps(res, ensure_ascii=False))
        return 0 if res["written"] else 1

    ap.error("pass --dois or --sources")


if __name__ == "__main__":
    sys.exit(main())