#!/usr/bin/env python3
"""bibrag — console-script dispatcher for the bib_rag toolkit.

Runs subcommands by file path (same scripts as the eph-rag/geo-rag wrappers,
so behavior is identical), with the chromadb protobuf fix applied and the
active library selectable via --kb / BIB_RAG_KB_NAME.

Usage:
    bibrag [--kb eph_rag|geo_rag] <command> [args...]

Commands:
    query <text> [--top N]        → src/query_bib_rag.py
    agentic <text> | --interactive→ agentic_query.py (repo root)
    add <pdfs...> [--skip-extract]→ add_papers.py
    doctor [--fix]                → scripts/doctor.py
    setup [args]                  → scripts/setup_library.py
    index <md>                    → src/index_single_paper.py
    build [args]                  → src/build_hierarchical_gpu.py
    remove [args]                 → scripts/remove_paper.py
    classify [args]               → scripts/classify_papers.py
    graph [args]                  → src/reference_graph.py
    bib export [args]             → scripts/bibtex_export... (auto-discovered)
    config                        → print active library config
    run <path> [args]             → run ANY script in the toolkit by
                                    relative path (e.g. run src/broaden.py)

Examples:
    bibrag query "EphA4 receptor expression" --top 3
    bibrag --kb geo_rag query "hypoxia markers"
    bibrag agentic --interactive
    bibrag add /path/to/pdfs/ --batch-size 5
"""

import os
import sys
import runpy
import argparse
from pathlib import Path

# ── chromadb protobuf fix (same as the eph-rag/geo-rag wrappers) ─────────
# A system-level google.protobuf can shadow the pip one and break chromadb's
# opentelemetry chain ("cannot import name 'builder'"). Prepending user
# site-packages fixes resolution order without touching anything else.
_USP = Path.home() / ".local/lib/python3.10/site-packages"
if _USP.is_dir():
    os.environ["PYTHONPATH"] = str(_USP) + (
        ":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")


def _code_root() -> Path:
    """Repo root (parent of src/) — for repo checkouts / editable installs.

    For a pip-installed package the parent of the installed package dir is
    site-packages (no repo there), so fall back to the canonical repo
    location. Same resolution precedence as kb_config._resolve_rag_home:
    BIB_RAG_CODE_ROOT env > derived parent with real layout > canonical.
    """
    env_root = os.environ.get("BIB_RAG_CODE_ROOT")
    if env_root:
        return Path(env_root)
    derived = Path(__file__).resolve().parent.parent
    if (derived / "src").is_dir() and (derived / "scripts").is_dir():
        return derived
    return Path("/Disk_bot/RAG/bib_rag")


# command → repo-relative script path
COMMANDS = {
    "query": "src/query_bib_rag.py",
    "agentic": "agentic_query.py",
    "add": "add_papers.py",
    "doctor": "scripts/doctor.py",
    "setup": "scripts/setup_library.py",
    "index": "src/index_single_paper.py",
    "build": "src/build_hierarchical_gpu.py",
    "remove": "scripts/remove_paper.py",
    "classify": "scripts/classify_papers.py",
    "graph": "src/reference_graph.py",
    "retraction": "src/retraction_watch.py",
    "zotero": "scripts/zotero_access.py",
    "eval": "scripts/eval_retrieval.py",
    "fts": "scripts/build_fts_index.py",
    "bib": "src/bibtex_export.py",
    "grill": "src/bib_rag_grill.py",
    "writer": "src/bib_rag_writer.py",
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    # --kb handling (before subcommand): set env, strip flag
    kb_name = None
    if "--kb" in argv:
        i = argv.index("--kb")
        try:
            kb_name = argv[i + 1]
        except IndexError:
            print("bibrag: --kb requires a library name", file=sys.stderr)
            return 2
        del argv[i:i + 2]
    if kb_name:
        os.environ["BIB_RAG_KB_NAME"] = kb_name

    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, args = argv[0], argv[1:]

    root = _code_root()

    if cmd == "config":
        # import kb_config AFTER --kb was applied
        sys.path.insert(0, str(root / "src"))
        from kb_config import print_config  # noqa: E402
        print_config()
        return 0

    if cmd == "run":
        # generic passthrough: any script path relative to repo root
        if not args:
            print("bibrag run: needs a script path (e.g. run src/broaden.py)",
                  file=sys.stderr)
            return 2
        target = Path(args[0])
        if not target.is_absolute():
            target = root / target
        if not target.exists():
            print(f"bibrag: script not found: {target}", file=sys.stderr)
            return 2
        script = str(target)
        rest = args[1:]
    else:
        if cmd not in COMMANDS:
            print(f"bibrag: unknown command {cmd!r}. Try --help, or "
                  f"'run <script.py>' for arbitrary scripts.", file=sys.stderr)
            return 2
        script = str(root / COMMANDS[cmd])
        if not Path(script).exists():
            print(f"bibrag: command script missing: {script}", file=sys.stderr)
            return 2
        rest = args

    # Dispatch: same execution model as eph-rag (run the file as __main__).
    # sys.argv as the target script expects it.
    sys.argv = [script] + rest
    runpy.run_path(script, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())