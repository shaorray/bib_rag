#!/usr/bin/env python3
"""DEPRECATED — merged into bib_rag_grill.py (--no-llm mode).

Forwards to: bib_rag_grill.py <topic> --auto --no-llm
Archived original: scripts/archive_project_specific/bib_rag_writer.py
"""
import subprocess
import sys
from pathlib import Path

print("⚠️  bib_rag_writer.py is deprecated — use bib_rag_grill.py --auto --no-llm "
      "(same retrieval, spec-aware scope, deterministic synthesis).", file=sys.stderr)

args = sys.argv[1:]
cleaned, skip = [], False
for a in args:
    if skip:
        skip = False
        continue
    if a == "--style":          # grill uses CONTEXT.md citation style
        skip = True
        continue
    cleaned.append(a)

grill = str(Path(__file__).resolve().parent / "bib_rag_grill.py")
subprocess.call([sys.executable, "-B", grill, *cleaned, "--auto", "--no-llm"])
