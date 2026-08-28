#!/usr/bin/env python3
"""
fill_meta_key_in_parent_store.py

For every parent_store JSON file, fill the 'key' field under `meta` with the
Zotero article_key from My Library.bib, matched on DOI (fallback: PDF-basename
title matching).

Output:
  - In-place update of <parent-dir>/*.json: each record's `meta` dict gets a
    `key` field (the Zotero article_key, e.g. 'abdul-wajid_t-type_2015').
  - outputs/meta_key_fill_report_<DATE>.csv  (one row per paper:
    json_file, source, doi, article_key, match_type, action)

Safety (aligned with the sibling scripts):
  - Dry-run by default; pass --apply to write.
  - Before the first write of a file, its original content is backed up to
    data/parent_store_backup_key_<DATE>/.
  - Idempotent: a file is only rewritten when at least one record's key differs.

Strategy:
  1. Build DOI -> (article_key, title) lookup from My Library.bib
  2. For each parent_store file, read meta.doi and meta.title
  3. Match by DOI; if found, write meta['key'] = article_key
  4. If DOI not in bib, fall back to PDF-basename-title matching
  5. Report: matched / unmatched (with reason) / written / unchanged
"""

import argparse
import csv
import glob
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from bib_utils import normalize_doi

from kb_config import get_config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

_CFG = get_config()

PARENT_DIR = _CFG["parent_store_dir"]
from library_config import get_setting as _lib_setting
BIB_PATH = _lib_setting(_CFG["data_root"], "bib_path", "My Library.bib")
OUT_DIR = _CFG["outputs_dir"]
DATA_DIR = _CFG["data_dir"]

# ----------------------------- helpers -----------------------------

_NORM_NONALNUM = re.compile(r"[^a-zA-Z0-9]+")
_AUTHOR_YEAR_LOOSE = re.compile(
    r"^[\s\S]{0,300}?\s*[-_:,]\s*\d{2,4}\s*[-_:]\s*",
)
_YEAR_PREFIX = re.compile(r"^\s*\d{4}\s*[-_:]\s*")


def norm(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = _NORM_NONALNUM.sub("_", s).strip("_")
    return s


def strip_pdf_title_prefix(s: str) -> str:
    s = s.strip()
    if s.endswith(".md"):
        s = s[:-3]
    for _ in range(3):
        new = _AUTHOR_YEAR_LOOSE.sub("", s, count=1)
        if new == s:
            break
        s = new.strip()
    s = _YEAR_PREFIX.sub("", s, count=1)
    return s


# ----------------------------- bib parser -----------------------------

_HEADER_RE = re.compile(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", re.IGNORECASE)


def parse_bib(bib_path):
    """
    Return:
      - doi_index: dict normalized_doi -> article_key
      - title_index: dict normalized_title -> article_key (PDF basename title index)
    """
    with open(bib_path, encoding="utf-8") as f:
        text = f.read()
    n = len(text)
    doi_index = {}
    title_index = {}
    _FILE_FIELD_RE = re.compile(r"\bfile\s*=\s*\{", re.IGNORECASE)

    for m in _HEADER_RE.finditer(text):
        etype = m.group(1).lower()
        key = m.group(2)
        start = m.end()
        depth = 1
        j = start
        while j < n and depth > 0:
            c = text[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            j += 1
        body = text[start:j - 1]

        def balanced_field(name):
            fm = re.search(r"\b" + re.escape(name) + r"\s*=\s*\{", body, re.IGNORECASE)
            if not fm:
                fm = re.search(r"\b" + re.escape(name) + r"\s*=\s*\"", body, re.IGNORECASE)
                if not fm:
                    return None
                k = fm.end()
                end = body.find('"', k)
                return body[k:end]
            k = fm.end()
            d = 1
            kk = k
            buf = []
            while kk < len(body) and d > 0:
                cc = body[kk]
                if cc == "{":
                    d += 1
                    buf.append(cc)
                elif cc == "}":
                    d -= 1
                    if d == 0:
                        break
                    buf.append(cc)
                else:
                    buf.append(cc)
                kk += 1
            return "".join(buf)

        # DOI
        d = balanced_field("doi")
        if d:
            nd = normalize_doi(d)
            if nd and nd not in doi_index:
                doi_index[nd] = key

        # PDF basename title index (only PDF: prefix)
        fm = _FILE_FIELD_RE.search(body)
        if fm:
            k = fm.end()
            d = 1
            kk = k
            buf = []
            while kk < len(body) and d > 0:
                cc = body[kk]
                if cc == "{":
                    d += 1
                    buf.append(cc)
                elif cc == "}":
                    d -= 1
                    if d == 0:
                        break
                    buf.append(cc)
                else:
                    buf.append(cc)
                kk += 1
            file_val = "".join(buf).strip()
            if file_val.startswith("{"):
                file_val = file_val[1:]
            if file_val.endswith("}"):
                file_val = file_val[:-1]
            for seg in file_val.split(";"):
                seg = seg.strip()
                if not seg.lower().startswith("pdf:"):
                    continue
                rest = seg[4:]
                if ":" in rest:
                    rest = rest.rsplit(":", 1)[0]
                if not rest.lower().endswith(".pdf"):
                    continue
                base = os.path.basename(rest)
                base = re.sub(r"\.pdf$", "", base, flags=re.IGNORECASE)
                title = strip_pdf_title_prefix(base)
                nt = norm(title)
                if nt and nt not in title_index:
                    title_index[nt] = key
    return doi_index, title_index


# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Fill meta.key in parent_store/*.json from My Library.bib (DOI / PDF-title match).",
    )
    ap.add_argument("--bib", default=BIB_PATH)
    ap.add_argument("--parent-dir", default=PARENT_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"),
                    help="date tag for report/backup dir names (default: today)")
    ap.add_argument("--backup-dir", default=None,
                    help="default: data/parent_store_backup_key_<date>")
    ap.add_argument("--dry-run", action="store_true", help="report only, don't write")
    ap.add_argument("--apply", action="store_true", help="actually write (with backup)")
    args = ap.parse_args()

    if not args.dry_run and not args.apply:
        args.dry_run = True

    if not os.path.exists(args.bib):
        print(f"[fill_meta_key] ERROR: bib not found: {args.bib}")
        return 1
    if not os.path.isdir(args.parent_dir):
        print(f"[fill_meta_key] ERROR: parent dir not found: {args.parent_dir}")
        return 1

    out_report = os.path.join(args.out_dir, f"meta_key_fill_report_{args.date}.csv")
    backup_dir = Path(args.backup_dir
                      or os.path.join(args.data_dir, f"parent_store_backup_key_{args.date}"))

    print(f"[1/3] parsing bib for DOI + PDF-title indices")
    try:
        doi_index, title_index = parse_bib(args.bib)
    except OSError as e:
        print(f"[fill_meta_key] ERROR reading bib: {e}")
        return 1
    print(f"      {len(doi_index)} DOIs, {len(title_index)} PDF-titles")

    mode = "apply" if args.apply else "dry-run"
    print(f"[2/3] walking parent_store and filling meta.key (mode={mode})")
    if args.apply:
        backup_dir.mkdir(parents=True, exist_ok=True)

    report_rows = []
    n_total = 0
    n_doi_matched = 0
    n_title_matched = 0
    n_unmatched = 0
    n_written = 0
    n_unchanged = 0
    n_errors = 0

    for fp in sorted(glob.glob(os.path.join(args.parent_dir, "*.json"))):
        try:
            with open(fp, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            n_errors += 1
            report_rows.append({
                "json_file": os.path.basename(fp), "source": "", "doi": "",
                "article_key": "", "match_type": "none", "action": f"error:{e}",
            })
            continue
        if not isinstance(data, list) or not data:
            n_errors += 1
            report_rows.append({
                "json_file": os.path.basename(fp), "source": "", "doi": "",
                "article_key": "", "match_type": "none", "action": "error:not-a-list",
            })
            continue
        rec = data[0]
        meta = rec.get("meta", {}) or {}
        raw_doi = meta.get("doi", "") or ""
        nd = normalize_doi(raw_doi)
        raw_title = meta.get("title", "") or rec.get("source", "") or ""
        if raw_title.endswith(".md"):
            raw_title = raw_title[:-3]

        article_key = None
        match_type = None
        if nd and nd in doi_index:
            article_key = doi_index[nd]
            match_type = "doi"
            n_doi_matched += 1
        else:
            # fallback: title match
            nt = norm(strip_pdf_title_prefix(raw_title))
            if nt and nt in title_index:
                article_key = title_index[nt]
                match_type = "title"
                n_title_matched += 1
        n_total += 1

        action = "none"
        if article_key:
            # write into every record's meta.key (idempotent: only when changed)
            changed = any((r.get("meta") or {}).get("key") != article_key for r in data)
            if not changed:
                action = "unchanged"
                n_unchanged += 1
            elif args.apply:
                try:
                    bp = backup_dir / os.path.basename(fp)
                    if not bp.exists():
                        with open(fp, encoding="utf-8") as f:
                            bp.write_text(f.read(), encoding="utf-8")
                    for r in data:
                        m = r.setdefault("meta", {})
                        m["key"] = article_key
                    with open(fp, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    action = "written"
                    n_written += 1
                except Exception as e:
                    n_errors += 1
                    action = f"error:{e}"
            else:
                action = "would_write"
                n_written += 1
        else:
            n_unmatched += 1

        report_rows.append({
            "json_file": os.path.basename(fp),
            "source": raw_title,
            "doi": raw_doi,
            "article_key": article_key or "",
            "match_type": match_type or "none",
            "action": action,
        })

    print(f"[3/3] writing report")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(out_report, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "json_file", "source", "doi", "article_key", "match_type", "action",
        ])
        w.writeheader()
        for r in report_rows:
            w.writerow(r)

    print()
    print(f"  total parent_store files: {n_total}")
    print(f"  matched by DOI:           {n_doi_matched}")
    print(f"  matched by title:         {n_title_matched}")
    print(f"  unmatched:                {n_unmatched}")
    print(f"  written (would-write):    {n_written}")
    print(f"  unchanged (already set):  {n_unchanged}")
    print(f"  errors:                   {n_errors}")
    print(f"  mode:                     {mode}")
    print(f"  report:                   {out_report}")
    if args.apply:
        print(f"  backup:                   {backup_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
