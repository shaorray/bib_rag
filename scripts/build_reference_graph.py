#!/usr/bin/env python3
"""
build_reference_graph.py — Build the citation graph for snowballing tools.

Scans parent_store JSONs for in-text citations (Author, Year) and surviving
reference-list entries, writes <data_root>/data/reference_graph.json.
Enables the agent tools find_papers_citing / get_paper_references.

Usage:
    python3 scripts/build_reference_graph.py            # build + stats
    python3 scripts/build_reference_graph.py --stats    # stats only
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from kb_config import parse_kb_arg, get_config, print_config  # noqa: E402


def main():
    argv = parse_kb_arg()
    ap = argparse.ArgumentParser(description="Build citation reference graph")
    ap.add_argument("--stats", action="store_true", help="show stats only")
    args = ap.parse_args(argv)

    cfg = get_config()
    print_config()
    print()

    from reference_graph import load_graph, build_reference_graph

    if args.stats:
        g = load_graph()
        if g is None:
            print("no reference graph built yet")
            return 1
    else:
        print("[refgraph] building (this scans all parent_store JSONs)...")
        g = build_reference_graph(progress=True)

    # quick stats
    papers = g.get("papers", {})
    edges = g.get("edges", [])
    with_hints = sum(1 for e in edges if e.get("to_title_hint"))
    print(f"\npapers: {len(papers)}")
    print(f"edges:  {len(edges)}  (with title hints: {with_hints})")
    return 0


if __name__ == "__main__":
    sys.exit(main())