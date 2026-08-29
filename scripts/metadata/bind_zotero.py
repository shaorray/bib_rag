#!/usr/bin/env python3
"""
bind_zotero.py — Bind parent_store papers to their Zotero records via the
BibTeX snapshot (merged successor of bib_to_parent_store.py +
fill_meta_key_in_parent_store.py).

One pass over My Library.bib writes BOTH Zotero-bound fields per paper:

  meta.doi — from the matched BibTeX entry (what chroma queries filter on)
  meta.key — the entry's citation key (Zotero article_key, e.g.
             'abdul-wajid_t-type_2015'), consumed by query_bib_rag's
             parent-store overlay and Zotero round-trip workflows

Matching (honesty protocol, unchanged from the originals):

  Pass 1 — triple match: normalized author-lastname + year (+/-1 tolerance)
           + title-prefix overlap. Only 'matched' bindings are written;
           'multi_match' bindings need --low-confidence (or an abstract
           tiebreak) and are written with a warning if that flag is passed.
  Pass 2 — title reverse-search for papers pass 1 couldn't bind (no_year /
           no_match): requires a sufficiently long title.
  Pass 3 — DOI bridge: papers whose chroma/parent_store meta already has a
           DOI (from meta_audit's remote layer) match the bib entry by DOI
           even when the triple match failed — fills meta.key via the
           existing DOI.

Everything else is reported, never guessed: unmatched / low_confidence /
multi_match papers appear in the report CSV with the reason.

Usage:
    <name>-rag scripts/metadata/bind_zotero.py --dry-run      # default
    <name>-rag scripts/metadata/bind_zotero.py --apply        # write (with backup)
    <name>-rag scripts/metadata/bind_zotero.py --bib "/path/My Library.bib" --apply
"""
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(SCRIPTS_DIR))

from kb_config import get_config  # noqa: E402
from library_config import get_setting  # noqa: E402

_CFG = get_config()

BIB_PATH = Path(get_setting(_CFG["data_root"], "bib_path") or "My Library.bib")
PARENT_STORE = Path(_CFG["parent_store_dir"])
DATA_DIR = Path(_CFG["data_dir"])
OUT_DIR = Path(_CFG["outputs_dir"])
BACKUP_DIR = DATA_DIR / "parent_store_backup_bind"

REPORT_FIELDS = ["json_file", "source", "status", "match_type", "doi",
                 "zotero_key", "title", "detail"]


# ── matching (reuses bib_to_parent_store's vetted matchers) ────────────────

def _doi_bridge(doi: str, bib_entries):
    """Find the bib entry whose DOI matches (normalized)."""
    from bib_utils import normalize_doi
    nd = normalize_doi(doi)
    for be in bib_entries:
        if be["doi"] and normalize_doi(be["doi"]) == nd:
            return be, "doi_bridge"
    return None, "doi_no_bib"


def bind_paper(fp: Path, bib_entries, parent_meta: dict):
    """Return (doi, key, status, match_type) for one parent_store file.

    Never raises; unmatched papers return (…, '', '', status, reason).
    """
    from bib_to_parent_store import (
        match_paper_by_title, match_paper_to_entry,
        extract_year_from_paper_content, normalize_paper_title,
    )

    stem = fp.stem                      # e.g. 10087273_md or Author_2020_Title
    source = stem if stem.endswith(".md") else stem + ".md"
    title = (parent_meta.get("title") or "").strip()
    year = str(parent_meta.get("year") or "").strip()
    stored_doi = (parent_meta.get("doi") or "").strip()

    # lastname from the filename stem (works for both naming schemes:
    # '10087273_md' → digit → no lastname; 'Abbate_2021_...' → 'Abbate')
    from bib_utils import filename_to_key
    try:
        paper_key = filename_to_key(stem)
    except Exception:
        paper_key = None

    entry, status = None, "no_match"

    # Pass 1: triple match via filename-derived key
    if paper_key:
        entry, status = match_paper_to_entry(paper_key, bib_entries)

    # Pass 2: title reverse-search (skip noisy titles — journal headers)
    if entry is None and title and not _is_noisy_title(title) and year:
        entry, status = match_paper_by_title(
            title, year, bib_entries, paper_abstract_norm="")

    # Pass 2.5: PMID bridge — PMID-stemmed files (10068468_md.json) resolve via
    # PubMed esummary: PMID → DOI/title/year/authors, then DOI-bridge to the bib
    if entry is None and stored_doi:
        entry, status = _doi_bridge(stored_doi, bib_entries)

    pmid = (parent_meta.get("pmid") or "").strip()
    if entry is None and not pmid:
        import re as _re
        m = _re.fullmatch(r"(\d{6,9})_md", fp.stem)
        if m:
            pmid = m.group(1)
    if entry is None and pmid:
        try:
            from meta_audit import PubmedClient
            rec = PubmedClient().summary(pmid)
        except Exception:
            rec = None
        if rec:
            if rec.get("doi"):
                entry, status = _doi_bridge(rec["doi"], bib_entries)
                if entry is not None:
                    status = "pmid_bridge"
            if entry is None:
                # no bib entry — PubMed record itself is the fix
                return {"doi": rec.get("doi", ""), "key": "",
                        "title": rec.get("title", ""),
                        "authors": rec.get("authors", ""),
                        "year": rec.get("year", ""),
                        "status": "matched", "match_type": "pubmed_only"}

    if entry is None:
        return {"doi": stored_doi, "key": "", "title": title, "authors": "",
                "year": "", "status": "unmatched", "match_type": status}

    doi = entry.get("doi") or stored_doi
    key = entry.get("key") or ""
    base = {"doi": doi, "key": key, "title": entry.get("title", title),
            "authors": entry.get("author", ""), "year": entry.get("year", "")}
    if status == "multi_match":
        base.update(status="low_confidence", match_type=status)
        return base
    if status in ("matched", "title_matched", "doi_bridge", "pmid_bridge"):
        base.update(status="matched", match_type=status)
        return base
    return {"doi": stored_doi, "key": "", "title": title, "authors": "",
            "year": "", "status": "unmatched", "match_type": status}


def _is_noisy_title(title: str) -> bool:
    """Journal-header signature: digits early + parenthetical volume/year."""
    if not title:
        return True
    head = title[:40]
    return any(c.isdigit() for c in title[:12]) and "(" in head


# ── apply (backup + atomic write of meta.doi / meta.key) ───────────────────

def apply_bindings(bindings: dict, dry_run: bool):
    """bindings: parent_store filename -> {doi, key}. Writes meta fields with
    backup + atomic_json_dump."""
    import chromadb
    from chunking import atomic_json_dump

    if not dry_run:
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    n_files = n_meta = 0
    for fp, bind in bindings.items():
        if dry_run:
            continue
        # backup
        backup_path = BACKUP_DIR / fp.name
        if not backup_path.exists():
            backup_path.write_text(fp.read_text(encoding="utf-8"))

        data = json.loads(fp.read_text(encoding="utf-8"))
        changed = False
        for chunk in data:
            m = chunk.get("meta") if isinstance(chunk, dict) else None
            if not isinstance(m, dict):
                continue
            for fld in ("doi", "key", "title", "authors", "year"):
                if bind.get(fld) and m.get(fld) != bind[fld]:
                    m[fld] = bind[fld]
                    changed = True
        if changed:
            atomic_json_dump(data, fp)
            n_files += 1
            n_meta += 2  # doi + key slots

    # chroma metadata: mirror doi + key onto chunks
    if not dry_run and bindings:
        col = chromadb.PersistentClient(path=_CFG["chroma_path"]).get_collection(
            _CFG["collection_name"])
        by_source = {}
        for fp, bind in bindings.items():
            stem = fp.stem
            src = stem if stem.endswith(".md") else stem + ".md"
            by_source[src] = bind
            by_source[stem] = bind
        offset, page = 0, 2000
        while True:
            res = col.get(limit=page, offset=offset, include=["metadatas"])
            ids = res["ids"]
            if not ids:
                break
            new_metas = []
            for m in res["metadatas"]:
                new = dict(m)
                src = m.get("source", "")
                stem = src[:-3] if src.endswith(".md") else src
                rec = by_source.get(src) or by_source.get(stem)
                if rec:
                    if rec.get("doi"):
                        new["doi"] = rec["doi"]
                    if rec.get("key"):
                        new["zotero_key"] = rec["key"]
                new_metas.append(new)
            col.update(ids=ids, metadatas=new_metas)
            offset += page
            if len(ids) < page:
                break
    return n_files, n_meta


# ── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Bind parent_store papers to Zotero (doi + key) via BibTeX snapshot")
    ap.add_argument("--bib", type=Path, default=BIB_PATH, help="My Library.bib path")
    ap.add_argument("--parent-dir", type=Path, default=PARENT_STORE)
    ap.add_argument("--dry-run", action="store_true", help="report only (default)")
    ap.add_argument("--apply", action="store_true", help="write meta.doi + meta.key (with backup)")
    ap.add_argument("--low-confidence", action="store_true",
                    help="also write multi_match bindings (with warning)")
    ap.add_argument("--limit", type=int, default=0, help="bind only first N files")
    ap.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    args = ap.parse_args()
    if not args.dry_run and not args.apply:
        args.dry_run = True

    if not args.bib.exists():
        sys.exit(f"ERROR: BibTeX file not found: {args.bib} — pass --bib or set "
                 "bib_path in <library>/config.json")
    from bib_utils import parse_bib_entries
    bib_entries = parse_bib_entries(args.bib)
    print(f"library : {_CFG['kb_name']} ({_CFG['data_root']})")
    print(f"mode    : {'APPLY' if args.apply else 'DRY-RUN'}")
    print(f"bib     : {args.bib} ({len(bib_entries)} entries)")

    files = sorted(args.parent_dir.glob("*.json"))
    if args.limit:
        files = files[:args.limit]
    print(f"papers  : {len(files)} parent_store files")

    report_rows, bindings = [], {}
    counts = {"matched": 0, "low_confidence": 0, "unmatched": 0}
    t0 = time.time()
    for i, fp in enumerate(files, 1):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as e:
            report_rows.append({"json_file": fp.name, "status": f"error:{e}"[:60],
                                "match_type": "error", "doi": "", "zotero_key": "",
                                "title": "", "source": fp.name})
            continue
        meta = {}
        for chunk in data:
            m = chunk.get("meta", {}) if isinstance(chunk, dict) else {}
            if m:
                meta = m
                break
        rec = bind_paper(fp, bib_entries, meta)
        status = rec["status"]
        counts[status if status in counts else "unmatched"] += 1
        source = fp.stem if fp.stem.endswith(".md") else fp.stem + ".md"
        report_rows.append({
            "json_file": fp.name, "status": status, "match_type": rec["match_type"],
            "doi": rec["doi"], "zotero_key": rec["key"],
            "title": rec["title"], "source": source,
        })
        if status == "matched" or (status == "low_confidence" and args.low_confidence):
            bindings[fp] = {k: rec[k] for k in ("doi", "key", "title", "authors", "year")
                            if rec.get(k)}
        if i % 300 == 0:
            print(f"  ...{i}/{len(files)} ({time.time()-t0:.0f}s)")

    # report CSV
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUT_DIR / f"zotero_bind_report_{args.date}.csv"
    with open(report_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=REPORT_FIELDS)
        w.writeheader()
        w.writerows(report_rows)

    print(f"\nresults: {counts['matched']} matched, {counts['low_confidence']} multi_match "
          f"(use --low-confidence to write), {counts['unmatched']} unmatched")
    print(f"report : {report_path}")
    if args.dry_run:
        print(f"would bind: {len(bindings)} papers (doi + zotero key)")
        print("(dry-run — re-run with --apply to write)")
    else:
        n_files, n_meta = apply_bindings(bindings, dry_run=False)
        print(f"bound: {n_files} parent_store files updated, "
              f"{len(bindings)} papers (meta.doi + meta.key), chroma doi/zotero_key mirrored")

    return 0


if __name__ == "__main__":
    sys.exit(main())