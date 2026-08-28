#!/usr/bin/env python3
"""
backfill_metadata.py — General-purpose metadata backfill for any RAG library.

Overwrites noisy md-extracted chunk metadata (titles often come out as journal
headers like "BIOLOGY OF REPRODUCTION 63, 797-804 (2000)") with clean values
from a metadata CSV, and optionally applies article_type + topic tags from a
classification CSV.

Input CSV requirements (flexible column names via flags):
  - a `source` column matching chroma metadata `source` values (e.g. 10087273.md
    or Author_2020_topic.md), plus any of: title / authors / journal / doi / year.
  - if `article_type` and `topics` (JSON list) columns are present, tags are
    applied too (same format classify_papers.py writes).

Also normalizes the source-key matching: a labeled row keyed "10087273" matches
source "10087273.md" and vice versa, so both PMID-style and name-style CSVs work.

Usage (with a wrapper):
    <name>-rag scripts/backfill_metadata.py --csv metadata/labeled.csv --dry-run
    <name>-rag scripts/backfill_metadata.py --csv metadata/labeled.csv \
        --tags outputs/tags.csv --apply
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

from kb_config import get_config  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

_CFG = get_config()

FIELDS = ("title", "authors", "journal", "doi", "year")


def load_clean(path, title_col, year_col):
    """source-key -> clean metadata dict. Tolerates PMID-stem keys."""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            src = (row.get("source") or "").strip()
            if not src:
                continue
            rec = {}
            for fld in FIELDS:
                col = {"year": year_col, "title": title_col}.get(fld, fld)
                v = (row.get(col) or "").strip()
                if v:
                    rec[fld] = v
            # also index the stem, so a row keyed "10087273" matches source
            # "10087273.md" and vice versa
            stem = src[:-3] if src.endswith(".md") else src
            if stem and stem != src:
                out.setdefault(stem, rec)
            out[src] = rec
    return out


def load_tags(path):
    if not path:
        return {}
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            src = row.get("source", "")
            at = row.get("article_type", "")
            if at in ("ERROR", "", "?"):
                continue
            tp = []
            try:
                parsed = json.loads(row.get("topics", "[]"))
                if isinstance(parsed, list):
                    tp = [str(x).strip().lower() for x in parsed if str(x).strip()][:5]
            except Exception:
                tp = []
            out[src] = {"article_type": at, "topics": tp}
    return out


def main():
    ap = argparse.ArgumentParser(description="Backfill clean metadata into the active library")
    ap.add_argument("--csv", required=True, help="CSV with source + clean metadata columns")
    ap.add_argument("--title-col", default="title")
    ap.add_argument("--year-col", default="year")
    ap.add_argument("--tags", help="classification CSV from classify_papers.py (optional)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import chromadb
    from chunking import atomic_json_dump

    clean = load_clean(args.csv, args.title_col, args.year_col)
    tags = load_tags(args.tags)
    print(f"metadata rows: {len(clean)}, tag rows: {len(tags)}")
    print(f"library: {cfg_name} ({_CFG['data_root']})" if (cfg_name := _CFG['kb_name']) else "")

    col = chromadb.PersistentClient(path=_CFG["chroma_path"]).get_collection(_CFG["collection_name"])
    t0 = time.time()
    updated = matched_sources = 0
    offset, page = 0, 2000
    while True:
        r = col.get(limit=page, offset=offset, include=["metadatas"])
        ids = r["ids"]
        if not ids:
            break
        metas = r["metadatas"]
        new_metas = []
        for m in metas:
            src = m.get("source", "")
            new = dict(m)
            # try source, then source-without-.md, then source-with-.md
            for key in (src, src[:-3] if src.endswith(".md") else src + ".md"):
                if key in clean:
                    lm = clean[key]
                    for fld in FIELDS:
                        if lm.get(fld):
                            new[fld] = lm[fld]
                    matched_sources += 1
                    break
            if src in tags:
                new["article_type"] = tags[src]["article_type"]
                for k in list(new.keys()):
                    if k.startswith("topic_"):
                        del new[k]
                for kw in tags[src]["topics"]:
                    if kw:
                        new[f"topic_{kw}"] = 1
            new_metas.append(new)
        if not args.dry_run:
            col.update(ids=ids, metadatas=new_metas)
        updated += len(ids)
        offset += page
        if updated % 40000 == 0:
            print(f"...{updated} chunks ({time.time()-t0:.0f}s)", file=sys.stderr)
        if len(ids) < page:
            break

    print(f"done: {updated} chunks processed, {matched_sources} matched clean metadata "
          f"({time.time()-t0:.0f}s)")
    if args.dry_run:
        print("(dry-run — no writes; pass without --dry-run to apply)")


if __name__ == "__main__":
    main()