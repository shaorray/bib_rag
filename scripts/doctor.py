#!/usr/bin/env python3
"""
doctor.py — Runtime self-diagnostics for the active bib_rag library.

Borrowed mechanisms (see the health-check cross-survey notes):
  - research-hub doctor.py: CheckResult dataclass with 4-level status
    (OK/INFO/WARN/FAIL), per-check try/except isolation (one crashing check
    never kills the run), remedy strings attached to every WARN/FAIL,
    offline/unreachable = WARN not FAIL, --strict to surface suppressed
    findings, exit code reflects FAIL only (CI-safe).
  - haiku.rag doctor CLI: index-consistency checks (count parity, parent
    resolution, near-duplicate DOI groups exported for human review) with
    repair commands printed per failure.

Checks (all zero-LLM, mostly offline):
  P0 offline:
    C1  sqlite integrity   — PRAGMA quick_check on chroma + fts databases
    C2  chroma↔FTS drift   — source-set parity via the kb_config sanitize
                             mapping; counts; sampled parent overlap
    C3  parent_store       — JSON loadable, parents listed, source==filename
    C4  reference graph    — file present, edge endpoints closed, orphan
                             paper count, graph↔parent_store set drift
    C5  DOI quality        — junk/truncated DOIs, duplicate-DOI groups
                             (CONFLICT vs same-paper via title similarity);
                             --doi-report writes the full human-review list
    C6  disk space         — data dirs free space
  P1 network (cached 60s, rate-limit friendly):
    C7  Zotero local API   — probe with Allowed-Request header; item count
                             vs parent_store drift (research-hub zotero_drift)
  P2 config:
    C8  endpoints          — embed/LLM URL configured (existence, not probed)
    C9  built indexes      — FTS + reference graph built at all (INFO if not)

Usage:
    python3 scripts/doctor.py                 # all checks, text report
    python3 scripts/doctor.py --json          # machine-readable report
    python3 scripts/doctor.py --strict        # don't suppress known-noise INFO
    python3 scripts/doctor.py --skip-network  # fully offline run
    python3 scripts/doctor.py --doi-report    # also write DOI issue list
Exit code: number of FAILs (0 = healthy), WARN/INFO don't fail.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from kb_config import parse_kb_arg, get_config  # noqa: E402

# ---------------------------------------------------------------------------
# CheckResult (research-hub mechanism)
# ---------------------------------------------------------------------------

OK, INFO, WARN, FAIL = "OK", "INFO", "WARN", "FAIL"
_ICON = {OK: "[OK]", INFO: "[ii]", WARN: "[!!]", FAIL: "[XX]"}
# strict=False demotes these names to INFO (known historical noise)
_KNOWN_NOISE = {"doi_quality"}


@dataclass
class CheckResult:
    name: str
    status: str                      # OK / INFO / WARN / FAIL
    message: str
    remedy: str = ""                 # copy-pasteable fix command
    details: List[str] = field(default_factory=list)

    def demote_if_noise(self, strict: bool) -> "CheckResult":
        if not strict and self.name in _KNOWN_NOISE and self.status == WARN:
            return CheckResult(self.name, INFO, self.message + " (--strict to show)",
                               self.remedy, self.details)
        return self


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _open_ro(path: str) -> Optional[sqlite3.Connection]:
    """Read-only sqlite connection; None when the file doesn't exist."""
    if not os.path.exists(path):
        return None
    try:
        return sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10)
    except sqlite3.Error:
        return None


def _sanitize(source: str) -> str:
    """EXACTLY the kb_config/hybrid_search source→filename mapping."""
    return re.sub(r"[^\w\-]", "_", source)[:100]


def _meta_of(parent_store_dir: str, source: str) -> tuple:
    """(title, year) from a parent_store file's first parent."""
    safe = re.sub(r"[^\w\-]", "_", source)[:100]
    path = os.path.join(parent_store_dir, f"{safe}.json")
    try:
        with open(path, encoding="utf-8") as f:
            parents = json.load(f)
        m = parents[0].get("meta", {}) or {}
        return (m.get("title", "") or "", str(m.get("year", "") or ""))
    except Exception:
        return "", ""


# ---------------------------------------------------------------------------
# C1 — sqlite integrity
# ---------------------------------------------------------------------------

def check_sqlite_integrity(cfg: dict) -> List[CheckResult]:
    out = []
    for label, path in (("chroma", cfg["chroma_sqlite"]),
                        ("fts", cfg["fts_index_path"])):
        if not os.path.exists(path):
            out.append(CheckResult(f"integrity_{label}", WARN,
                                   f"{label} database missing at {path}",
                                   remedy="build it: see scripts/build_fts_index.py / ingest"))
            continue
        conn = _open_ro(path)
        if conn is None:
            out.append(CheckResult(f"integrity_{label}", FAIL,
                                   f"cannot open {path}"))
            continue
        try:
            res = conn.execute("PRAGMA quick_check").fetchone()
            ok = res and res[0] == "ok"
            out.append(CheckResult(
                f"integrity_{label}",
                OK if ok else FAIL,
                f"quick_check {'ok' if ok else res} "
                f"({os.path.getsize(path) / 1e6:.0f} MB)",
                remedy="restore from backup or rebuild the index" if not ok else ""))
        except sqlite3.Error as e:
            out.append(CheckResult(f"integrity_{label}", FAIL,
                                   f"quick_check errored: {e}"))
        finally:
            conn.close()
    return out


# ---------------------------------------------------------------------------
# C2 — chroma ↔ FTS drift
# ---------------------------------------------------------------------------

def check_index_drift(cfg: dict, strict: bool = False) -> List[CheckResult]:
    out = []
    chroma = _open_ro(cfg["chroma_sqlite"])
    if chroma is None:
        return [CheckResult("index_drift", FAIL,
                            "chroma.sqlite3 unreadable — cannot audit drift")]
    try:
        t0 = time.time()
        chroma_sources = {r[0] for r in chroma.execute(
            "SELECT DISTINCT string_value FROM embedding_metadata "
            "WHERE key='source'")}
        n_chunks = chroma.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        chroma_parents = chroma.execute(
            "SELECT COUNT(DISTINCT string_value) FROM embedding_metadata "
            "WHERE key='parent_id'").fetchone()[0]
        out.append(CheckResult(
            "chroma_population", OK,
            f"{n_chunks} chunks, {chroma_parents} parents, "
            f"{len(chroma_sources)} sources"))

        ps_files = {_sanitize(f[:-5]) for f in os.listdir(cfg["parent_store_dir"])
                    if f.endswith(".json")}
        mapped = {_sanitize(s) for s in chroma_sources}

        # A. sources in chroma whose parent_store file was disabled/removed
        disabled_dir = cfg.get("parent_store_disabled_dir", "")
        disabled = ({f[:-5] for f in os.listdir(disabled_dir)
                     if f.endswith(".json")} if os.path.isdir(disabled_dir)
                    else set())
        stale = mapped - ps_files
        stale_active = stale - disabled
        if stale:
            st = FAIL if stale_active else WARN
            out.append(CheckResult(
                "chroma_orphans", st,
                f"{len(stale)} source(s) in chroma with NO parent_store file"
                + (f" ({len(stale_active)} not even in disabled store)" if stale_active else
                   f" — all in parent_store_disabled (expected after remove_paper)"),
                remedy="re-ingest the paper, or re-index via scripts/index_single_paper.py"
                if stale_active else "",
                details=sorted(stale)[:10]))
        else:
            out.append(CheckResult("chroma_orphans", OK,
                                   "every chroma source has a parent_store file"))

        # B. parent_store files never vectorized
        missing = ps_files - mapped
        if missing:
            out.append(CheckResult(
                "ps_unvectorized", WARN,
                f"{len(missing)} parent_store file(s) with NO chroma chunks "
                "(ingested but never embedded)",
                remedy="python3 -B add_papers.py <pdf-or-dir> (incremental via index_single_paper.py)",
                details=sorted(missing)[:10]))
        else:
            out.append(CheckResult("ps_unvectorized", OK,
                                   "every parent_store source is in chroma"))

        # C. FTS coverage — FTS stores the RAW source name (e.g.
        # "10068468.md", from parent JSON's `source` field) while
        # parent_store filenames are sanitized ("10068468_md"), so compare
        # through the same sanitize() used at index time.
        fts = _open_ro(cfg["fts_index_path"])
        if fts is None:
            out.append(CheckResult(
                "fts_coverage", WARN, "FTS index missing — BM25 channel OFF",
                remedy="python3 scripts/build_fts_index.py"))
        else:
            fts_sources = {_sanitize(r[0]) for r in
                           fts.execute("SELECT DISTINCT source FROM child_fts")}
            fts_children = fts.execute("SELECT COUNT(*) FROM child_fts").fetchone()[0]
            gap = ps_files - fts_sources
            out.append(CheckResult(
                "fts_coverage", OK if not gap else WARN,
                f"{fts_sources.__len__()} sources / {fts_children} children in FTS"
                + (f"; {len(gap)} parent_store sources missing" if gap else " — full coverage"),
                remedy="python3 scripts/build_fts_index.py" if gap else "",
                details=sorted(gap)[:10]))
            # chroma↔FTS chunk-count drift (same source, different chunking gen)
            fts_parents = fts.execute(
                "SELECT COUNT(DISTINCT parent_id) FROM child_fts").fetchone()[0]
            if abs(fts_parents - chroma_parents) > max(50, chroma_parents * 0.02):
                out.append(CheckResult(
                    "fts_chroma_parent_parity", WARN,
                    f"parent counts differ: FTS {fts_parents} vs chroma "
                    f"{chroma_parents} (>2% — generational chunking drift; "
                    "BM25/vec dedup still works at parent_id level)",
                    remedy="optional: full re-ingest to realign chunk boundaries"))
            else:
                out.append(CheckResult(
                    "fts_chroma_parent_parity", OK,
                    f"parent counts close: FTS {fts_parents} ≈ chroma {chroma_parents}"))
            fts.close()
    finally:
        chroma.close()
    return out


# ---------------------------------------------------------------------------
# C3 — parent_store consistency
# ---------------------------------------------------------------------------

def check_parent_store(cfg: dict) -> List[CheckResult]:
    out = []
    store = cfg["parent_store_dir"]
    files = sorted(f for f in os.listdir(store) if f.endswith(".json"))
    broken, empty, total_parents = [], [], 0
    for fname in files:
        path = os.path.join(store, fname)
        try:
            with open(path, encoding="utf-8") as f:
                parents = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            broken.append(f"{fname}: {type(e).__name__}")
            continue
        if not isinstance(parents, list) or not parents:
            empty.append(fname)
            continue
        total_parents += len(parents)
    n = len(files)
    if broken:
        out.append(CheckResult("parent_store_integrity", FAIL,
                               f"{len(broken)}/{n} file(s) unreadable/corrupt",
                               remedy="restore from backup or re-ingest these papers",
                               details=broken[:10]))
    elif empty:
        out.append(CheckResult("parent_store_integrity", WARN,
                               f"{len(empty)}/{n} file(s) empty",
                               remedy="re-ingest those papers", details=empty[:10]))
    else:
        out.append(CheckResult("parent_store_integrity", OK,
                               f"{n} files, {total_parents} parents, all parse"))
    return out


# ---------------------------------------------------------------------------
# C4 — reference graph
# ---------------------------------------------------------------------------

def check_reference_graph(cfg: dict) -> List[CheckResult]:
    path = cfg["reference_graph_path"]
    if not os.path.exists(path):
        return [CheckResult("reference_graph", INFO,
                            "not built — snowballing tools unavailable",
                            remedy="python3 scripts/build_reference_graph.py")]
    try:
        with open(path, encoding="utf-8") as f:
            g = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return [CheckResult("reference_graph", FAIL, f"unreadable: {e}",
                            remedy="python3 scripts/build_reference_graph.py")]
    papers, edges = g.get("papers", {}), g.get("edges", [])
    if not papers:
        return [CheckResult("reference_graph", FAIL, "graph is empty",
                            remedy="python3 scripts/build_reference_graph.py")]
    dangling = {e["from"] for e in edges if e.get("from") not in papers}
    if dangling:
        return [CheckResult("reference_graph", FAIL,
                            f"{len(dangling)} edge(s) reference unknown papers",
                            remedy="python3 scripts/build_reference_graph.py",
                            details=sorted(dangling)[:10])]
    with_edges = {e["from"] for e in edges}
    orphans = len(papers) - len(with_edges)
    out = [CheckResult(
        "reference_graph", OK,
        f"{len(papers)} papers, {len(g['edges'])} edges, endpoints closed, "
        f"{orphans} papers without outgoing edges (normal: many lack "
        "parseable reference tails)")]
    # graph ↔ parent_store set parity
    ps_sources = {_sanitize(f[:-5]) for f in os.listdir(cfg["parent_store_dir"])
                   if f.endswith(".json")}
    gp = {_sanitize(s) for s in papers}
    only_graph = gp - ps_sources
    only_ps = ps_sources - gp
    if only_graph or only_ps:
        out.append(CheckResult(
            "refgraph_parity", WARN,
            f"graph↔parent_store drift: {len(only_graph)} graph-only, "
            f"{len(only_ps)} store-only",
            remedy="python3 scripts/build_reference_graph.py",
            details=[f"graph-only: {sorted(only_graph)[:3]}",
                     f"store-only: {sorted(only_ps)[:3]}"]))
    else:
        out.append(CheckResult("refgraph_parity", OK,
                               f"graph papers == parent_store sources ({len(papers)})"))
    return out


# ---------------------------------------------------------------------------
# C5 — DOI quality (uses src/identifiers)
# ---------------------------------------------------------------------------

def check_doi_quality(cfg: dict, doi_report_path: Optional[str] = None,
                      strict: bool = False) -> List[CheckResult]:
    try:
        try:
            from identifiers import normalize_doi
        except ImportError:  # pragma: no cover
            from src.identifiers import normalize_doi
    except ImportError:
        return [CheckResult("doi_quality", INFO,
                            "identifiers module unavailable — skipped")]
    path = cfg["reference_graph_path"]
    if not os.path.exists(path):
        return [CheckResult("doi_quality", INFO,
                            "reference graph missing — no DOI metadata to audit")]
    with open(path, encoding="utf-8") as f:
        g = json.load(f)
    papers = g.get("papers", {})
    from collections import defaultdict
    junk: List[tuple] = []
    by_doi: Dict[str, List[str]] = defaultdict(list)
    for s, m in papers.items():
        raw = m.get("doi", "") or ""
        if not raw:
            continue
        d = normalize_doi(raw)
        if (not d) or re.match(r"^10\.\d+/$", d) \
                or d.rstrip(".").endswith(("journal", "journal.")) \
                or len(d.rstrip("/")) <= 10:
            junk.append((s, raw))
        elif d:
            by_doi[d].append(s)
    dup_groups = {d: ss for d, ss in by_doi.items() if len(ss) > 1}
    n_dup_papers = sum(len(v) for v in dup_groups.values())
    status = OK
    if dup_groups:
        msg = (f"{len(junk)} junk/unparseable DOI(s), {len(dup_groups)} "
               f"duplicate DOI group(s) covering {n_dup_papers} papers")
    else:
        msg = f"{len(junk)} junk/unparseable DOI(s), no duplicates"
    if junk or dup_groups:
        status = WARN
    out = [CheckResult(
        "doi_quality", status, msg,
        remedy="see the DOI issue list in library notes; regenerate with --doi-report"
        if (junk or dup_groups) else "")]
    if junk or dup_groups:
        out[-1].details = (
            [f"junk: {s} → {raw}" for s, raw in sorted(junk)[:5]]
            + [f"dup: {d} × {len(ss)}" for d, ss in sorted(dup_groups.items())[:5]])
    if doi_report_path and (junk or dup_groups):
        _write_doi_report(cfg, doi_report_path, junk, by_doi)
        out.append(CheckResult(
            "doi_report", OK, f"full review list → {doi_report_path}"))
    return out


def _write_doi_report(cfg: dict, out_path: str, junk: List[tuple],
                      by_doi: Dict[str, List[str]]) -> None:
    """Human-review markdown: junk DOIs + duplicate groups with titles/years."""
    def meta_of(src):
        safe = _sanitize(src)
        try:
            with open(os.path.join(cfg["parent_store_dir"], safe + ".json"),
                      encoding="utf-8") as f:
                parents = json.load(f)
            m = parents[0].get("meta", {}) or {}
            return (m.get("title", "") or "", str(m.get("year", "") or ""))
        except Exception:
            return "", ""

    try:
        from zotero_match import title_similarity
    except ImportError:
        from src.zotero_match import title_similarity

    lines = ["# DOI quality report (generated by scripts/doctor.py)", "",
             f"Generated: {__import__('datetime').datetime.now().isoformat(timespec='seconds')}", ""]
    lines.append(f"## Junk / unparseable DOIs ({len(junk)})\n")
    lines.append("| source | raw DOI field | title | year |")
    lines.append("|---|---|---|---|")
    for s, raw in sorted(junk):
        title, year = meta_of(s)
        lines.append(f"| `{s}` | `{raw}` | {title[:70] or '—'} | {year or '—'} |")
    lines.append(f"\n## Duplicate DOI groups ({len(by_doi)} distinct DOIs scanned)\n")
    for d, srcs in sorted(by_doi.items()):
        if len(srcs) < 2:
            continue
        entries = [(s,) + meta_of(s) for s in srcs]
        sims = [title_similarity(a[1], b[1])
                for i, a in enumerate(entries)
                for b in entries[i + 1:] if a[1] and b[1]]
        verdict = ("no-titles" if not sims else
                   ("same-paper" if max(sims) >= 0.6 else "CONFLICT"))
        lines.append(f"### `{d}` — {verdict}")
        for s, title, year in entries:
            lines.append(f"- `{s}` ({year or '—'}) — {title[:80] or '(no title)'}")
        lines.append("")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, out_path)


# ---------------------------------------------------------------------------
# C4b — parent metadata quality (DOI / authors / year) + deterministic repair
# ---------------------------------------------------------------------------

_META_JUNK_JOURNAL = {"as of", "the", "article", "research", "journal"}


def _meta_junk_journal(journal: str, doi: str) -> bool:
    """PDF-debris journal check, with a DOI-based escape hatch.

    Single dictionary words are usually scraped debris — EXCEPT when the
    paper's own DOI registrant confirms them: 'Research' is a real AAAS
    journal (10.34133/research.…), 'Science' likewise (10.1126/science.…).
    A junk call with a matching DOI prefix is a false positive, so the
    DOI vetoes the junk verdict."""
    j = (journal or "").strip()
    if not j or j.isdigit():
        return True
    if j.lower() in _META_JUNK_JOURNAL:
        # a single-word venue must be confirmed by its DOI prefix segment
        # ('research' ∈ 10.34133/research.0390, 'science' ∈ 10.1126/science.…)
        d = (doi or "").strip().lower()
        if d:
            tail = d.split("/", 1)[1] if "/" in d else ""
            if tail.startswith(j.lower() + ".") or tail.startswith(j.lower() + "/"):
                return False   # DOI itself says this is the venue
        return True
    return False
_META_CLEAN_DOI = re.compile(r"^10\.\d{4,5}/\S+$")


def _meta_year_ok(y: str) -> bool:
    """Plausible publication year: 1900..current+1 (not 1850/2030/2050)."""
    import datetime
    if not re.match(r"^(19|20)\d{2}$", y or ""):
        return False
    return 1900 <= int(y) <= datetime.date.today().year + 1


def _rewrite_parent_meta(path: str, chunks: list, updates: dict) -> None:
    """Apply {field: new_value} to every chunk's meta; atomic write."""
    for ch in chunks:
        m = ch.get("meta")
        if isinstance(m, dict):
            m.update(updates)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def check_meta_quality(cfg: dict, repair: bool = False) -> List[CheckResult]:
    """Sanity-audit title/authors/year/journal/doi in every parent file.

    Findings: no-authors, implausible years, junk journals ('As of',
    digits-only), mangled DOIs (markdown link tails, truncations), full-width
    CJK DOIs, filename-form titles.

    repair=True applies DETERMINISTIC local fixes only — operations that
    never need the network and never invent data:
      * full-width DOI (meta or content)  → NFKC fold in place
      * mangled DOI with extractable form → replace with extracted DOI
      * junk journal                      → clear (backfill refills it)
      * out-of-range year (2030/1850/'' ) → clear (backfill refills it)
    In-range but possibly-wrong years (1941) are left alone: only a
    Crossref verification can safely correct those — that is
    scripts/metadata/repair_meta.py, not doctor.
    """
    from identifiers import extract_doi, has_fullwidth, normalize_doi
    import unicodedata

    out: List[CheckResult] = []
    store = cfg["parent_store_dir"]
    files = sorted(f for f in os.listdir(store) if f.endswith(".json"))

    findings = {
        "no_authors": [], "bad_year": [], "junk_journal": [],
        "mangled_doi": [], "fullwidth_doi": [], "filename_title": [],
    }
    repairs = []   # (fname, description) for the meta_repair result

    for fname in files:
        path = os.path.join(store, fname)
        try:
            with open(path, encoding="utf-8") as f:
                chunks = json.load(f)
        except Exception:
            continue
        if not (isinstance(chunks, list) and chunks and isinstance(chunks[0], dict)):
            continue
        meta = chunks[0].get("meta") or {}
        updates: dict = {}

        authors = (meta.get("authors") or "").strip()
        year = str(meta.get("year") or "").strip()
        journal = (meta.get("journal") or "").strip()
        title = (meta.get("title") or "").strip()
        doi_raw = (meta.get("doi") or "").strip()

        if not authors:
            findings["no_authors"].append(fname)
        if not _meta_year_ok(year):
            findings["bad_year"].append((fname, year))
            if repair and year:            # '' is absence, not damage
                updates["year"] = ""
        if journal and _meta_junk_journal(journal, doi_raw):
            findings["junk_journal"].append((fname, journal))
            if repair:
                updates["journal"] = ""

        # --- DOI: full-width fold / mangled extraction / content salvage ---
        if doi_raw and has_fullwidth(doi_raw):
            findings["fullwidth_doi"].append((fname, doi_raw[:60]))
            if repair:
                norm = normalize_doi(doi_raw) or unicodedata.normalize(
                    "NFKC", doi_raw)
                updates["doi"] = norm
        elif doi_raw and not _META_CLEAN_DOI.match(
                unicodedata.normalize("NFKC", doi_raw)):
            best = extract_doi(doi_raw)
            if best:
                findings["mangled_doi"].append((fname, doi_raw[:60]))
                if repair:
                    updates["doi"] = best
            else:
                findings["mangled_doi"].append((fname, doi_raw[:60]))
        elif not doi_raw:
            # No DOI at all: salvage one from the first chunks' content.
            # Chinese-journal PDFs keep it full-width in the body text.
            head = "\n".join((ch.get("content") or "")
                             for ch in chunks[:2])[:2000]
            best = extract_doi(head)
            if best and has_fullwidth(head):
                findings["fullwidth_doi"].append((fname, f"(content) {best}"))
                if repair:
                    updates["doi"] = best

        if title and re.match(r"^(RE\d+ - |\d{6,} )", title):
            findings["filename_title"].append((fname, title[:60]))

        if repair and updates:
            _rewrite_parent_meta(path, chunks, updates)
            repairs.append((fname, ", ".join(sorted(updates))))

    f = findings
    msg = (f"{len(files)} parents: {len(f['no_authors'])} no-authors, "
           f"{len(f['bad_year'])} bad-year, {len(f['mangled_doi'])} mangled-DOI, "
           f"{len(f['junk_journal'])} junk-journal, {len(f['filename_title'])} filename-title"
           + (f", {len(f['fullwidth_doi'])} full-width DOI "
              f"({'repaired' if repair else 'repair available'})"
              if f["fullwidth_doi"] else ""))
    status = OK if not any(f.values()) else WARN
    remedy = ""
    if f["no_authors"] or f["filename_title"] or (f["bad_year"] and not repair) \
            or (f["mangled_doi"] and not repair):
        remedy = "python3 scripts/metadata/repair_meta.py (Crossref-backed fill)"
    if f["fullwidth_doi"] and not repair:
        remedy = (remedy + "; " if remedy else "") + \
            "doctor --repair-meta (deterministic local fixes)"
    out.append(CheckResult("meta_quality", status, msg, remedy=remedy,
                           details=([f"no-authors: {n}" for n in f["no_authors"][:3]]
                                    + [f"bad-year: {n} ({y!r})" for n, y in f["bad_year"][:3]]
                                    + [f"mangled: {n} → {d!r}" for n, d in f["mangled_doi"][:3]]
                                    + [f"junk-journal: {n} ({j!r})" for n, j in f["junk_journal"][:2]])))
    if repairs:
        out.append(CheckResult(
            "meta_repair", OK,
            f"deterministic fixes in {len(repairs)} parent file(s): "
            + "; ".join(f"{n} [{what}]" for n, what in repairs[:3])
            + (" …" if len(repairs) > 3 else "")))
    return out


# ---------------------------------------------------------------------------
# C5b — disk space
# ---------------------------------------------------------------------------

def check_disk(cfg: dict) -> List[CheckResult]:
    out = []
    for label, path in (("data", cfg["data_dir"]),
                        ("parent_store", cfg["parent_store_dir"])):
        if not os.path.isdir(path):
            continue
        try:
            usage = shutil.disk_usage(path)
            free_gb = usage.free / 1e9
            pct = usage.used / usage.total * 100
            st = OK if free_gb > 5 else (WARN if free_gb > 1 else FAIL)
            out.append(CheckResult(
                f"disk_{label}", st,
                f"{free_gb:.1f} GB free ({pct:.0f}% used) on {os.path.realpath(path)}",
                remedy="free space or move BIB_RAG_HOME" if st != OK else ""))
        except OSError:
            continue
    if not out:
        out.append(CheckResult("disk", INFO, "no data dirs found to measure"))
    return out


# ---------------------------------------------------------------------------
# C5b — retracted DOIs (offline, snapshot-based)
# ---------------------------------------------------------------------------

def check_retractions(cfg: dict, strict: bool = False) -> List[CheckResult]:
    """Flag papers whose DOI appears in the Retraction Watch snapshot."""
    try:
        from retraction_watch import load_retractions, check_sources, snapshot_path
    except ImportError:
        from src.retraction_watch import (  # type: ignore
            load_retractions, check_sources, snapshot_path)
    except Exception:
        return [CheckResult("retraction_watch", INFO,
                            "retraction_watch module unavailable — skipped")]
    if os.environ.get("RETRACTION_CHECK", "1") == "0":
        return [CheckResult("retraction_watch", INFO, "disabled via RETRACTION_CHECK=0")]
    snap = snapshot_path()
    if not os.path.exists(snap):
        return [CheckResult(
            "retraction_watch", INFO,
            "no Retraction Watch snapshot — run: "
            "/usr/bin/python3.10 -B src/retraction_watch.py --update",
            remedy="/usr/bin/python3.10 -B src/retraction_watch.py --update")]
    retr = load_retractions(snapshot=snap)
    if not retr:
        return [CheckResult(
            "retraction_watch", WARN,
            f"snapshot exists but empty/unparseable: {snap}",
            remedy=f"re-download: /usr/bin/python3.10 -B src/retraction_watch.py --update")]
    hits = check_sources(store_dir=cfg["parent_store_dir"], retractions=retr)
    if not hits:
        return [CheckResult(
            "retraction_watch", OK,
            f"0 retracted papers among the library's DOIs "
            f"(snapshot: {len(retr):,} DOIs)")]
    def _detail(d: str) -> str:
        try:
            from retraction_watch import retraction_details
        except ImportError:
            from src.retraction_watch import retraction_details  # type: ignore
        info = retraction_details(d, snapshot=snap) or {}
        bits = [info.get("nature", ""), info.get("reason", "").rstrip(";"),
                info.get("date", "").split(" ")[0]]
        return " / ".join(b for b in bits if b)
    shown = sorted(hits.items(), key=lambda kv: kv[1]["doi"])[:3]
    details = "; ".join(
        f"{t[:40]} {h['doi']} [{_detail(h['doi'])}]" for t, h in shown)
    out = [CheckResult(
        "retraction_watch", WARN,
        f"{len(hits)} RETRACTED paper(s) in the library: {details}"
        + (" ..." if len(hits) > 3 else ""),
        remedy="verify on https://retractionwatch.com and remove/quarantine "
               "the paper: move it out of the markdown dir, rebuild indexes")]
    return out


# ---------------------------------------------------------------------------
# C7 — Zotero local API (network, cached)
# ---------------------------------------------------------------------------

def check_zotero(cfg: dict, cache_path: Optional[str] = None) -> List[CheckResult]:
    out = []
    cache = {}
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as f:
                cached = json.load(f)
            if time.time() - cached.get("ts", 0) < 60:
                out.append(CheckResult(
                    "zotero_api", cached.get("status", INFO),
                    f"(cached {int(time.time() - cached['ts'])}s ago) "
                    + cached.get("message", "")))
                return out
        except Exception:
            pass
    try:
        import requests
        r = requests.get("http://localhost:23119/api/users/0/items?limit=1",
                         headers={"Zotero-Allowed-Request": "true"}, timeout=5)
        r.raise_for_status()
        total = r.headers.get("Total-Results")
        n_items = int(total) if total else -1
        msg = f"local API alive; {n_items} items in library" if n_items >= 0 \
            else "local API alive (count unavailable)"
        # drift vs parent_store file count (informational: Zotero holds
        # attachments/notes too, so exact parity is NOT expected)
        if n_items >= 0:
            ps_n = len([f for f in os.listdir(cfg["parent_store_dir"])
                        if f.endswith(".json")])
            diff = n_items - ps_n
            out.append(CheckResult(
                "zotero_api", OK, f"{msg}; parent_store {ps_n} files "
                f"(library has {diff:+d} more items — attachments/notes/ "
                "non-imported refs are normal)"))
        else:
            out.append(CheckResult("zotero_api", OK, msg))
        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump({"ts": time.time(), "status": out[-1].status,
                           "message": out[-1].message}, f)
    except Exception as e:
        out.append(CheckResult(
            "zotero_api", WARN,
            f"local API unreachable ({type(e).__name__}) — Zotero matching "
            "degrades to title-only",
            remedy="start Zotero desktop (localhost:23119)" if "Connection" in str(e) else ""))
    return out


# ---------------------------------------------------------------------------
# C8/C9 — config & built indexes
# ---------------------------------------------------------------------------

def check_endpoints(cfg: dict) -> List[CheckResult]:
    out = []
    for label, url in (("embed", cfg.get("embed_url")),
                       ("llm", os.environ.get("LLM_URL"))):
        if url:
            out.append(CheckResult(f"endpoint_{label}", OK,
                                   f"configured: {url} (not probed)"))
        else:
            out.append(CheckResult(
                f"endpoint_{label}", INFO,
                "not configured in env" + (" (expected — optional)" if label == "llm" else ""),
                remedy="export LLM_URL=... for writer/evaluate" if label == "llm" else ""))
    return out


def check_built_indexes(cfg: dict) -> List[CheckResult]:
    out = []
    fts = cfg["fts_index_path"]
    rg = cfg["reference_graph_path"]
    out.append(CheckResult(
        "fts_built", OK if os.path.exists(fts) else WARN,
        f"{'present' if os.path.exists(fts) else 'missing'}: {fts}",
        remedy="python3 scripts/build_fts_index.py" if not os.path.exists(fts) else ""))
    out.append(CheckResult(
        "reference_graph_built", OK if os.path.exists(rg) else WARN,
        f"{'present' if os.path.exists(rg) else 'missing'}: {rg}",
        remedy="python3 scripts/build_reference_graph.py" if not os.path.exists(rg) else ""))
    return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_doctor(skip_network: bool = False, strict: bool = False,
               doi_report: Optional[str] = None,
               repair_meta: bool = False) -> List[CheckResult]:
    cfg = get_config()
    checks: List[Callable] = [
        lambda: check_built_indexes(cfg),
        lambda: check_endpoints(cfg),
        lambda: check_parent_store(cfg),
        lambda: check_index_drift(cfg, strict),
        lambda: check_reference_graph(cfg),
        lambda: check_doi_quality(cfg, doi_report, strict),
        lambda: check_meta_quality(cfg, repair=repair_meta),
        lambda: check_retractions(cfg, strict),
        lambda: check_disk(cfg),
    ]
    if not skip_network:
        cache = os.path.join(cfg["data_dir"], "doctor_zotero_cache.json")
        checks.append(lambda: check_zotero(cfg, cache))
    results: List[CheckResult] = []
    for fn in checks:
        try:
            results.extend(fn())
        except Exception as e:  # one broken check never kills the doctor
            results.append(CheckResult(
                fn.__name__ if hasattr(fn, "__name__") else "check", WARN,
                f"check itself failed: {type(e).__name__}: {e}"))
    return [r.demote_if_noise(strict) for r in results]


def format_report(results: List[CheckResult], strict: bool) -> str:
    lines = [f"bib_rag doctor — {len(results)} checks"
             + (" (strict)" if strict else ""), "=" * 60]
    counts = {s: 0 for s in (OK, INFO, WARN, FAIL)}
    for r in results:
        counts[r.status] += 1
        lines.append(f"{_ICON[r.status]} {r.name}: {r.message}")
        for d in r.details[:5]:
            lines.append(f"        · {d}")
        if len(r.details) > 5:
            lines.append(f"        … +{len(r.details) - 5} more")
        if r.remedy:
            lines.append(f"        → {r.remedy}")
    lines.append("-" * 60)
    lines.append(f"summary: {counts[OK]} ok, {counts[INFO]} info, "
                 f"{counts[WARN]} warn, {counts[FAIL]} fail")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="bib_rag library health check")
    ap.add_argument("--json", action="store_true", help="JSON report")
    ap.add_argument("--strict", action="store_true",
                    help="surface demoted known-noise findings")
    ap.add_argument("--skip-network", action="store_true",
                    help="offline run (no Zotero probe)")
    ap.add_argument("--doi-report", metavar="PATH",
                    help="write full DOI issue review list to PATH")
    ap.add_argument("--repair-meta", action="store_true",
                    help="fold full-width CJK DOIs in parent files (lossless NFKC)")
    argv = parse_kb_arg()
    args = ap.parse_args(argv)
    results = run_doctor(skip_network=args.skip_network, strict=args.strict,
                         doi_report=args.doi_report,
                         repair_meta=args.repair_meta)
    n_fail = sum(1 for r in results if r.status == FAIL)
    if args.json:
        print(json.dumps({
            "fails": n_fail,
            "checks": [{"name": r.name, "status": r.status,
                        "message": r.message, "remedy": r.remedy,
                        "details": r.details} for r in results],
        }, ensure_ascii=False, indent=2))
    else:
        print(format_report(results, args.strict))
    sys.exit(n_fail)  # CI-safe: only FAILs set a non-zero exit


if __name__ == "__main__":
    main()