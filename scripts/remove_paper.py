#!/usr/bin/env python3
"""
remove_paper.py — Delete a paper from the active RAG library.

Removes all ChromaDB child chunks for a source, its parent_store JSON, and
its checkpoint/metadata-log entries, so the paper is fully gone and can be
re-indexed cleanly later (e.g. after fixing the md).

Usage:
    # By md filename (the `source` metadata value):
    /usr/bin/python3.10 -B scripts/remove_paper.py 10087273.md

    # By filename stem (matches any source containing it):
    /usr/bin/python3.10 -B scripts/remove_paper.py --match Salvucci

    # Dry-run (default): show what would be deleted, nothing is written.
    # --apply: actually delete.

Safety:
  - Always dry-run unless --apply is passed.
  - Refuses wildcard patterns; use --match with a literal substring.
  - Works on the ACTIVE library (respects BIB_RAG_KB_NAME / wrappers).
"""
import argparse
import json
import os
import sys
from pathlib import Path

from kb_config import get_config  # noqa: E402

import chromadb  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def find_sources(col, pattern, exact):
    """Return (ids, source) grouped by source for matching chunks."""
    if exact:
        r = col.get(where={"source": pattern}, include=["metadatas"])
        return r["ids"], {pattern}
    # substring match over all sources — page through metadata
    sources = set()
    offset, page = 0, 5000
    while True:
        r = col.get(include=["metadatas"], limit=page, offset=offset)
        if not r["ids"]:
            break
        for m in r["metadatas"]:
            src = m.get("source", "")
            if pattern.lower() in src.lower():
                sources.add(src)
        offset += page
    ids = []
    for src in sources:
        r = col.get(where={"source": src})
        ids.extend(r["ids"])
    return ids, sources


def main():
    ap = argparse.ArgumentParser(description="Delete a paper from the active RAG library")
    ap.add_argument("source", help="md filename (e.g. 10087273.md) or --match pattern")
    ap.add_argument("--match", action="store_true",
                    help="treat SOURCE as a case-insensitive substring of the filename")
    ap.add_argument("--apply", action="store_true",
                    help="actually delete (default is dry-run)")
    args = ap.parse_args()

    cfg = get_config()
    print(f"library : {cfg['kb_name']} ({cfg['data_root']})")
    print(f"mode    : {'APPLY' if args.apply else 'DRY-RUN (pass --apply to delete)'}")

    client = chromadb.PersistentClient(path=cfg["chroma_path"])
    col = client.get_collection(cfg["collection_name"])

    ids, sources = find_sources(col, args.source, exact=not args.match)
    if not sources:
        print(f"no chunks found matching '{args.source}' — nothing to do")
        return

    print(f"\nmatches: {len(sources)} source(s), {len(ids)} chunks")
    for s in sorted(sources):
        print(f"  - {s}")

    if not args.apply:
        print("\ndry-run: pass --apply to delete these chunks + parent_store + logs")
        return

    # 1. chroma chunks
    col.delete(ids=ids)
    print(f"[ok] deleted {len(ids)} chroma chunks")

    # 2. parent_store JSON
    from parent_store_manager import ParentStoreManager
    pm = ParentStoreManager(store_dir=cfg["parent_store_dir"])
    for src in sources:
        # save_parent_store hashed the FULL source ('x.md' → 'x_md.json');
        # also try the stem-stripped variant for papers added the other way.
        candidates = [pm._safe_filename(src), pm._safe_filename(src[:-3] if src.endswith(".md") else src)]
        removed = False
        for cand in candidates:
            p = Path(cfg["parent_store_dir"]) / f"{cand}.json"
            if p.exists():
                p.unlink()
                print(f"[ok] removed {p.name}")
                removed = True
        if not removed:
            print(f"[--] no parent_store file for {src} (already gone)")

    # 3. checkpoint + metadata log entries
    for log_path in (cfg["checkpoint_file"], cfg["metadata_log"]):
        p = Path(log_path)
        if not p.exists():
            continue
        try:
            data = json.load(open(p, encoding="utf-8"))
        except Exception as e:
            print(f"[warn] could not parse {p}: {e} — left untouched")
            continue
        if isinstance(data, dict) and "processed" in data:      # checkpoint
            before = len(data["processed"])
            data["processed"] = [x for x in data["processed"] if x not in sources]
            if len(data["processed"]) != before:
                from chunking import atomic_json_dump
                atomic_json_dump(data, p)
                print(f"[ok] checkpoint: removed {before - len(data['processed'])} entries")
        elif isinstance(data, dict) and any(k in sources for k in data):  # metadata log
            before = len(data)
            data = {k: v for k, v in data.items() if k not in sources}
            from chunking import atomic_json_dump
            atomic_json_dump(data, p)
            print(f"[ok] metadata log: now {len(data)} entries")

    print(f"\nDone. '{args.source}' fully removed from {cfg['kb_name']}.")
    print(f"Re-index any time with: {cfg['kb_name'].replace('_rag','')}-rag src/index_single_paper.py <md>")


if __name__ == "__main__":
    main()