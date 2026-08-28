#!/usr/bin/env python3
"""
Migrate topics metadata from a JSON-string field to per-keyword boolean keys
(topic_<kw>: 1) so they can be filtered with ChromaDB's $eq operator (which does
NOT support $contains in v1.4.1). Keeps the original 'topics' JSON field too.

Keyword normalization: lowercase, spaces -> hyphens, strip non-alnum/hyphen.
Usage:
    /usr/bin/python3.10 -B scripts/migrate_topics.py [--dry-run]
"""
import argparse
import json
import re
import sys
import time

import chromadb

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ (bib_utils, zotero_access)
from kb_config import get_config

_CFG = get_config()
CHROMA_DB_PATH = _CFG["chroma_path"]
COLLECTION = _CFG["collection_name"]


def norm(kw: str) -> str:
    kw = kw.strip().lower()
    kw = re.sub(r"[^a-z0-9]+", "-", kw).strip("-")
    return kw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    col = chromadb.PersistentClient(path=CHROMA_DB_PATH).get_collection(COLLECTION)
    t0 = time.time()
    updated = 0
    skipped = 0
    # iterate in pages to avoid loading all 341k at once
    offset = 0
    page = 1000
    while True:
        r = col.get(limit=page, offset=offset, include=["metadatas"])
        ids = r["ids"]
        if not ids:
            break
        metas = r["metadatas"]
        new_metas = []
        for m in metas:
            new = dict(m)
            raw = m.get("topics", "")
            kws = []
            if raw:
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, list):
                        kws = [norm(x) for x in parsed if str(x).strip()]
                except Exception:
                    kws = []
            # remove any stale topic_* keys, then re-add
            for k in list(new.keys()):
                if k.startswith("topic_"):
                    del new[k]
            for kw in kws:
                if kw:
                    new[f"topic_{kw}"] = 1
            new_metas.append(new)
        if not args.dry_run:
            col.update(ids=ids, metadatas=new_metas)
        updated += len(ids)
        offset += page
        if updated % 20000 == 0:
            print(f"...{updated} chunks ({time.time()-t0:.0f}s)", file=sys.stderr)
        if len(ids) < page:
            break

    print(f"done: {updated} chunks processed, {skipped} skipped, {time.time()-t0:.0f}s")
    if args.dry_run:
        print("DRY RUN — no writes performed")


if __name__ == "__main__":
    main()
