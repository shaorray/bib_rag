#!/usr/bin/env python3
"""
build_reference_graph.py — Build the citation graph for snowballing tools.

Scans parent_store JSONs for in-text citations (Author, Year) and surviving
reference-list entries, writes <data_root>/data/reference_graph.json.
Enables the agent tools find_papers_citing / get_paper_references.

--icite: additionally fetch PubMed/iCite-verified citation edges (network:
PMID resolution via eutils esearch + iCite records; resumable caches at
data/icite_pmid_resolution.json / data/icite_corpus_icite.json), write
data/citation_graph.json, then rebuild reference_graph.json so the
verified edges are merged in as `resolved` edges (exact lookups instead
of author-year heuristics). Libraries without resolvable PMIDs (e.g.
geo_rag) simply get an empty graph — safe no-op.

Usage:
    python3 scripts/build_reference_graph.py                 # heuristic build + stats
    python3 scripts/build_reference_graph.py --stats         # stats only
    python3 scripts/build_reference_graph.py --icite         # fetch verified edges + rebuild
    python3 scripts/build_reference_graph.py --icite --icite-resolve-limit 5   # probe run
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
    ap.add_argument("--icite", action="store_true",
                    help="fetch PubMed/iCite verified edges first (network, "
                         "resumable), then rebuild the heuristic graph and "
                         "merge them")
    ap.add_argument("--icite-resolve-limit", type=int, default=0, metavar="N",
                    help="cap Phase-A PMID resolution calls (probe runs); "
                         "0 = unlimited")
    args = ap.parse_args(argv)

    cfg = get_config()
    print_config()
    print()

    if args.stats:
        from reference_graph import load_graph
        g = load_graph()
        if g is None:
            print("no reference graph built yet")
            return 1
    elif args.icite:
        from reference_graph import build_icite_graph
        print("[refgraph] fetching PubMed/iCite verified edges "
              "(resumable caches in data/)...")
        build_icite_graph(progress=True, resolve_limit=args.icite_resolve_limit)
        # the icite pipeline already rebuilt reference_graph.json (merged);
        # fall through to stats below using the fresh graph
        from reference_graph import load_graph
        g = load_graph()
    else:
        from reference_graph import build_reference_graph
        print("[refgraph] building (this scans all parent_store JSONs)...")
        g = build_reference_graph(progress=True)

    # quick stats
    if g is None:
        print("no reference graph built (empty corpus?)")
        return 1
    papers = g.get("papers", {})
    edges = g.get("edges", [])
    with_hints = sum(1 for e in edges if e.get("to_title_hint"))
    resolved = sum(1 for e in edges if e.get("resolved"))
    print(f"\npapers: {len(papers)}")
    print(f"edges:  {len(edges)}  (resolved/iCite: {resolved}, "
          f"with title hints: {with_hints})")
    return 0


if __name__ == "__main__":
    sys.exit(main())