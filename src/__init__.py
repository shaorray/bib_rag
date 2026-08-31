"""bib_rag — academic bibliography RAG toolkit.

Package layout: this directory ships AS the `bib_rag` Python package
(setuptools package-dir remap of src/), while ALSO remaining a runnable
loose-script collection (`python3 -B src/query_bib_rag.py ...` works with
src/ on sys.path) — modules use the dual-import pattern:

    try:  # bib_rag-package-try
        from .chunking import X
    except ImportError:  # flat (loose-script mode)
        from chunking import X

Libraries (data folders: chroma_db_new/, parent_store/, md/, ...) are NOT
part of the package. They live as siblings of the repo clone under
BIB_RAG_HOME and are resolved by kb_config.py (registry + BIB_RAG_KB_NAME /
BIB_RAG_ROOT / --kb flag).

Entry points (see README):
    bibrag            — console-script dispatcher for all subcommands
    agentic_query.py  — LangGraph agentic query CLI (repo root)
    add_papers.py     — PDF ingest CLI (repo root)
    eph-rag / geo-rag — per-library wrapper commands (~/.local/bin)
"""

from .kb_config import get_kb_name, get_config, print_config

__version__ = "0.9.0"
__all__ = ["get_kb_name", "get_config", "print_config"]