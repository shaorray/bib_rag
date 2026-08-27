#!/usr/bin/env python3
"""
Apply article_type + topics tags from the classification CSV back into the
ChromaDB collection metadata. Preserves all existing metadata fields; only adds
article_type (str) and topics (JSON list) per chunk sharing the same source.

Usage:
    /usr/bin/python3.10 -B scripts/apply_tags.py outputs/tags.csv
"""
import argparse
import csv
import json
import sys
import time

import chromadb

sys.path.insert(0, "/Disk_bot/Eph/bib_rag/src")
from kb_config import get_config

_CFG = get_config()
CHROMA_DB_PATH = _CFG["chroma_path"]
COLLECTION = _CFG["collection_name"]


def load_tags(path):
    tags = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            src = row["source"]
            at = row.get("article_type", "")
            if at in ("ERROR", "", "?"):
                continue
            tp = []
            raw = row.get("topics", "")
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    tp = [str(x).strip().lower() for x in parsed if str(x).strip()][:5]
            except Exception:
                tp = []
            tags[src] = {"article_type": at, "topics": tp}
    return tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="tags CSV from classify_all.py")
    ap.add_argument("--dry-run", action="store_true", help="only report, don't write")
    args = ap.parse_args()

    tags = load_tags(args.csv)
    print(f"loaded {len(tags)} tagged sources")

    col = chromadb.PersistentClient(path=CHROMA_DB_PATH).get_collection(COLLECTION)
    t0 = time.time()
    updated, skipped, errored = 0, 0, 0
    for src, tag in tags.items():
        try:
            q = col.get(where={"source": src})
        except Exception as e:
            errored += 1
            print(f"[err] {src[:60]} : {e}", file=sys.stderr)
            continue
        ids = q["ids"]
        if not ids:
            skipped += 1
            continue
        if args.dry_run:
            updated += len(ids)
            continue
        # read original metadatas to preserve them
        m = col.get(include=["metadatas"], where={"source": src})
        metas = m["metadatas"]
        new_metas = []
        for mm in metas:
            new = dict(mm)
            new["article_type"] = tag["article_type"]
            new["topics"] = json.dumps(tag["topics"], ensure_ascii=False)
            new_metas.append(new)
        col.update(ids=ids, metadatas=new_metas)
        updated += len(ids)
        if updated % 2000 == 0:
            print(f"...{updated} chunks updated ({time.time()-t0:.0f}s)", file=sys.stderr)

    print(f"done: updated {updated} chunks, skipped {skipped} (no chunks), errored {errored}, "
          f"{time.time()-t0:.0f}s")
    if args.dry_run:
        print("DRY RUN — no writes performed")


if __name__ == "__main__":
    main()
