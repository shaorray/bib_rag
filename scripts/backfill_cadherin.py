#!/usr/bin/env python3
"""
Backfill Cadherin paper metadata from cadherin_labeled.csv into ChromaDB.

The md files are named by PMID (e.g. 10087273.md) and index_single_paper.py
extracts a noisy title from the PDF text (often a journal header like
"BIOLOGY OF REPRODUCTION 63, 797-804 (2000)"). cadherin_labeled.csv has clean
title/authors/journal/doi/year per PMID. This script overwrites the noisy
metadata with the clean values for all chunks whose source matches a PMID.md.

Also sets article_type + topics from the classification CSV (same format as
classify_all.py output) if provided.

Usage:
    /usr/bin/python3.10 -B backfill_cadherin.py \
        --labeled metadata/cadherin_labeled.csv \
        [--tags outputs/cadherin_tags.csv]
"""
import argparse
import csv
import json
import sys
import time

import chromadb

CHROMA_DB_PATH = "/Disk_bot/Eph/bib_rag/chroma_db_new"


def load_labeled(path):
    """pmid -> clean metadata dict."""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pmid = row.get("pmid", "").strip()
            if not pmid:
                continue
            out[pmid] = {
                "title": row.get("title", "").strip(),
                "authors": row.get("authors", "").strip(),
                "journal": row.get("journal", "").strip(),
                "doi": row.get("doi", "").strip(),
                "year": row.get("pub_year", "").strip(),
            }
    return out


def load_tags(path):
    """source -> {article_type, topics} from classify_all.py CSV."""
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", required=True)
    ap.add_argument("--tags", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    labeled = load_labeled(args.labeled)
    tags = load_tags(args.tags)
    print(f"labeled: {len(labeled)} pmids, tags: {len(tags)} sources")

    col = chromadb.PersistentClient(path=CHROMA_DB_PATH).get_collection("bib_rag_papers")
    t0 = time.time()
    updated = 0
    matched_sources = 0
    # iterate all chunks in pages
    offset = 0
    page = 2000
    while True:
        r = col.get(limit=page, offset=offset, include=["metadatas"])
        ids = r["ids"]
        if not ids:
            break
        metas = r["metadatas"]
        new_metas = []
        for m in metas:
            src = m.get("source", "")
            # source is like "10087273.md" -> pmid = "10087273"
            pmid = src.replace(".md", "") if src.endswith(".md") else ""
            new = dict(m)
            if pmid in labeled:
                lm = labeled[pmid]
                new["title"] = lm["title"]
                if lm["authors"]:
                    new["authors"] = lm["authors"]
                if lm["journal"]:
                    new["journal"] = lm["journal"]
                if lm["doi"]:
                    new["doi"] = lm["doi"]
                if lm["year"]:
                    new["year"] = lm["year"]
                matched_sources += 1
            if src in tags:
                new["article_type"] = tags[src]["article_type"]
                # remove stale topic_* keys then re-add
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

    print(f"done: {updated} chunks processed, {matched_sources} matched labeled, "
          f"{time.time()-t0:.0f}s")
    if args.dry_run:
        print("DRY RUN — no writes performed")


if __name__ == "__main__":
    main()
