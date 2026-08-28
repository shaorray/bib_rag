#!/usr/bin/env python3
"""
build_fts_index.py — Build/refresh the BM25 (FTS5) index for hybrid search.

Reads every parent_store JSON for the active library and re-chunks with the
SAME chunking.py logic used at build time, then FTS5-indexes the children.
Never touches chroma_db_new/. Safe to re-run (idempotent, WAL mode).

Usage:
    python3 scripts/build_fts_index.py                # full rebuild
    python3 scripts/build_fts_index.py --source X.md  # one paper (incremental)
    python3 scripts/build_fts_index.py --status       # show index stats
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from kb_config import parse_kb_arg, get_config, print_config  # noqa: E402


def main():
    argv = parse_kb_arg()
    ap = argparse.ArgumentParser(description="Build BM25 FTS5 index for hybrid search")
    ap.add_argument("--source", help="index a single source (incremental)")
    ap.add_argument("--status", action="store_true", help="show index status only")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args(argv)

    from hybrid_search import HybridIndex

    cfg = get_config()
    if not args.quiet:
        print_config()
        print()

    idx = HybridIndex()
    if args.status:
        n = idx.indexed_sources()
        print(f"FTS index: {idx.fts_path}")
        print(f"indexed sources: {n}")
        return 0

    if args.source:
        n = idx.upsert_source(args.source)
        print(f"[fts] {args.source}: {n} children indexed")
        return 0

    print(f"[fts] rebuilding index at {idx.fts_path} ...")
    total = idx.rebuild(progress=not args.quiet)
    print(f"[fts] total children indexed: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())