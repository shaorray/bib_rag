#!/usr/bin/env python3
"""DEPRECATED — merged into bib_rag_grill.py (--debate mode).

This shim forwards to: eph-rag|geo-rag src/bib_rag_grill.py "topic" --auto --debate
Archived original: scripts/archive_project_specific/bib_rag_writer_debate.py
"""
import subprocess, sys

print("⚠️  bib_rag_writer_debate.py is deprecated — use bib_rag_grill.py --auto --debate "
      "(spec-aware scope + relational debate synthesis in one tool).", file=sys.stderr)
args = list(sys.argv[1:])
subprocess.call([sys.executable, "-B",
                 str(Path(__file__).parent / "bib_rag_grill.py"), *args, "--auto", "--debate"])
