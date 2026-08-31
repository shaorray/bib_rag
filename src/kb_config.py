#!/usr/bin/env python3
"""
kb_config.py — Shared configuration for the RAG toolkit (code) + knowledge-base
stores (data). Architecture (2026-08-27):

    <RAG home>/bib_rag/        CODE  — src/, scripts/, docs/ (this repo)
    <RAG home>/<name>_rag/     DATA  — one folder per domain library

The RAG home defaults to the parent directory of this repo (so sibling
library folders are found wherever the toolkit is cloned) and can be moved
with BIB_RAG_HOME.

A "library" = a data directory that is self-describing (its own chroma_db_new/,
parent_store/, data/, outputs/, CONTEXT.md, LIBRARY.md). The tool code is
shared; adding a new library never touches code beyond one registry line.

Resolution (highest wins):
  1. BIB_RAG_ROOT       — explicit data-root override
  2. BIB_RAG_KB_NAME    — registry lookup (also accepts --kb flag)
  3. legacy: "bib_rag" name still resolves to eph_rag (deprecated alias)
  4. default: eph_rag

Env vars:
  BIB_RAG_KB_NAME     eph_rag (default) | geo_rag | ...
  BIB_RAG_ROOT        explicit library/data root (overrides registry)
  BIB_RAG_COLLECTION  collection name override (default comes from registry)
  BIB_RAG_CODE_ROOT   override the tool-code root (default: parent of src/)
"""

import os
from pathlib import Path

# ─── Fixed tool-code root ───────────────────────────────────────────────────
# Default code root = parent of this src/ directory (portable: wherever the
# repo is cloned, the toolkit follows). Override with BIB_RAG_CODE_ROOT.
_CODE_ROOT = os.environ.get(
    "BIB_RAG_CODE_ROOT",
    str(Path(__file__).resolve().parent.parent))

# ─── Named library registry ────────────────────────────────────────────────
# Each entry: data root + canonical collection name. Adding a library = adding
# one entry here (or setting BIB_RAG_ROOT/BIB_RAG_COLLECTION directly, or
# running scripts/setup_library.py which patches this dict).
#
# Root defaults derive from BIB_RAG_HOME (default: the directory containing
# this repo) so a cloned toolkit finds its sibling libraries anywhere:
#   <BIB_RAG_HOME>/<name>/  e.g. <BIB_RAG_HOME>/eph_rag/
# INSTALLED-package fallback: when the toolkit is pip-installed (not a repo
# checkout / editable install), parent-of-package lands inside
# site-packages — no libraries live there. Detect that (neither eph_rag/
# nor geo_rag/ exists) and fall back to the canonical install location.
# BIB_RAG_HOME always wins over both.
_DERIVED_RAG_HOME = Path(__file__).resolve().parent.parent.parent

def _resolve_rag_home() -> Path:
    env_home = os.environ.get("BIB_RAG_HOME")
    if env_home:
        return Path(env_home)
    # repo-checkout / editable-install layout: sibling library folders exist
    if (_DERIVED_RAG_HOME / "eph_rag").is_dir() or \
       (_DERIVED_RAG_HOME / "geo_rag").is_dir():
        return _DERIVED_RAG_HOME
    # pip-installed package (site-packages) — no sibling libraries; use the
    # canonical data location next to the original install root.
    return Path("/Disk_bot/RAG")

_RAG_HOME = _resolve_rag_home()

_KB_REGISTRY = {
    "eph_rag": {
        "root": str(_RAG_HOME / "eph_rag"),
        "collection": "bib_rag_papers",   # historical name, kept for the existing 470K chunks
    },
    "geo_rag": {
        "root": str(_RAG_HOME / "geo_rag"),
        "collection": "geo_rag_papers"
    }
}

# Legacy aliases (deprecated, print a warning once)
_LEGACY_ALIASES = {
    "bib_rag": "eph_rag",
}
_WARNED = set()


def get_kb_name() -> str:
    """Active library name; 'bib_rag' is accepted as a deprecated alias for eph_rag."""
    name = os.environ.get("BIB_RAG_KB_NAME", "eph_rag")
    if name in _LEGACY_ALIASES and name not in _WARNED:
        _WARNED.add(name)
        import sys
        print(f"[kb_config] WARNING: '{name}' is deprecated, use '{_LEGACY_ALIASES[name]}'",
              file=sys.stderr)
        name = _LEGACY_ALIASES[name]
    return name


def get_code_root() -> str:
    """Tool-code root (src/, scripts/). Independent of which library is active."""
    return _CODE_ROOT


def get_data_root() -> str:
    """Resolve the active library's data directory.
    Priority: BIB_RAG_ROOT env > registry[name] > default eph_rag."""
    root = os.environ.get("BIB_RAG_ROOT")
    if root:
        return root
    name = get_kb_name()
    if name in _KB_REGISTRY:
        return _KB_REGISTRY[name]["root"]
    return _KB_REGISTRY["eph_rag"]["root"]


def get_collection_name() -> str:
    """Collection for the active library.
    Priority: BIB_RAG_COLLECTION env > registry > 'bib_rag_papers' (eph default)."""
    env = os.environ.get("BIB_RAG_COLLECTION")
    if env:
        return env
    name = get_kb_name()
    if name in _KB_REGISTRY:
        return _KB_REGISTRY[name].get("collection", "bib_rag_papers")
    return "bib_rag_papers"


def get_config() -> dict:
    """All paths/urls for the active library. Embedding/LLM endpoints are shared
    (bge-m3 and Qwen/Ollama are domain-general services on fixed ports)."""
    root = get_data_root()
    return {
        "kb_name": get_kb_name(),
        "code_root": get_code_root(),
        "kb_root": root,               # deprecated alias for data_root
        "data_root": root,
        "chroma_path": os.path.join(root, "chroma_db_new"),
        "chroma_sqlite": os.path.join(root, "chroma_db_new", "chroma.sqlite3"),
        "parent_store_dir": os.path.join(root, "parent_store"),
        "parent_store_disabled_dir": os.path.join(root, "parent_store_disabled"),
        "data_dir": os.path.join(root, "data"),
        "outputs_dir": os.path.join(root, "outputs"),
        "metadata_log": os.path.join(root, "data", "incremental_metadata.json"),
        "checkpoint_file": os.path.join(root, "data", "build_hierarchical_checkpoint.json"),
        "context_md": os.path.join(root, "CONTEXT.md"),   # per-library domain glossary
        "fts_index_path": os.path.join(root, "data", "fts_index.db"),  # BM25 (hybrid_search)
        "reference_graph_path": os.path.join(root, "data", "reference_graph.json"),  # snowballing
        "embed_url": "http://localhost:8081/v1/embeddings",
        "embed_url_raw": "http://localhost:8081/embedding",
        "llm_url": "http://localhost:5015/v1",
        "collection_name": get_collection_name(),
    }


# Backwards-compat aliases (older code imported these names)
def get_kb_root() -> str:
    """Deprecated: use get_data_root()."""
    return get_data_root()


# ─── Convenience: CLI --kb flag support ────────────────────────────────────
def parse_kb_arg(argv=None) -> list:
    """
    Scan argv for --kb <name> or --kb=<name> and set env accordingly.
    Returns the remaining argv (with --kb stripped).
    """
    import sys
    if argv is None:
        argv = list(sys.argv[1:])

    remaining = []
    i = 0
    while i < len(argv):
        if argv[i] == "--kb" and i + 1 < len(argv):
            os.environ["BIB_RAG_KB_NAME"] = argv[i + 1]
            i += 2
        elif argv[i].startswith("--kb="):
            os.environ["BIB_RAG_KB_NAME"] = argv[i].split("=", 1)[1]
            i += 1
        else:
            remaining.append(argv[i])
            i += 1

    # Auto-resolve root+collection from the registry if not explicitly set
    name = os.environ.get("BIB_RAG_KB_NAME", "eph_rag")
    if name in _LEGACY_ALIASES:
        name = _LEGACY_ALIASES[name]
        os.environ["BIB_RAG_KB_NAME"] = name
    if "BIB_RAG_ROOT" not in os.environ and name in _KB_REGISTRY:
        os.environ["BIB_RAG_ROOT"] = _KB_REGISTRY[name]["root"]
    if "BIB_RAG_COLLECTION" not in os.environ and name in _KB_REGISTRY:
        os.environ["BIB_RAG_COLLECTION"] = _KB_REGISTRY[name]["collection"]

    return remaining


def print_config():
    """Print active config (for debugging)."""
    cfg = get_config()
    print(f"  Library:       {cfg['kb_name']}")
    print(f"  Data root:     {cfg['data_root']}")
    print(f"  Code root:     {cfg['code_root']}")
    print(f"  ChromaDB:      {cfg['chroma_path']}")
    print(f"  Parent store:  {cfg['parent_store_dir']}")
    print(f"  Outputs:       {cfg['outputs_dir']}")
    print(f"  Embed URL:     {cfg['embed_url']}")
    print(f"  Collection:    {cfg['collection_name']}")