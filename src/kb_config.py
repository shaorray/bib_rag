#!/usr/bin/env python3
"""
kb_config.py — Shared configuration for the RAG toolkit (code) + knowledge-base
stores (data). Generic, library-agnostic.

    <RAG home>/bib_rag/        CODE  — src/, scripts/, docs/ (this repo)
    <RAG home>/<name>_rag/     DATA  — one folder per domain library

A "library" = a data directory that is self-describing (its own
chroma_db_new/, parent_store/, data/, outputs/, CONTEXT.md, LIBRARY.md,
config.json). The tool code is SHARED and holds NO library-specific binding
(no registry of name → root/collection, no default-library name, no
machine path). Every library resolves entirely from:

  - environment (its ~/.local/bin/<stem>-rag wrapper, or explicit exports), and
  - its own <root>/config.json ("settings" block).

Resolution (highest wins):

  data root:
    1. BIB_RAG_ROOT env           — explicit data-root override
    2. <BIB_RAG_HOME>/<BIB_RAG_KB_NAME>   — convention (default layout)
    if neither is set: RuntimeError (fail loudly; never guess a default library)

  collection name:
    1. BIB_RAG_COLLECTION env     — explicit override
    2. <root>/config.json settings.collection  — per-library value (authoritative
       for e.g. eph_rag whose Chroma collection predates the "_papers" convention)
    3. <BIB_RAG_KB_NAME minus _rag>_papers     — convention fallback

Env vars:
  BIB_RAG_KB_NAME     library stem, e.g. eph_rag | geo_rag | prompt_rag
  BIB_RAG_HOME        dir that holds the *_rag/ library folders
  BIB_RAG_ROOT        explicit library/data root (wins over the convention)
  BIB_RAG_COLLECTION  collection name override
  BIB_RAG_CODE_ROOT   override the tool-code root (default: parent of src/)

To add a library: create its folder (see scripts/setup_library.py), set the
collection (if non-conventional) in its config.json, and emit a wrapper that
exports BIB_RAG_KB_NAME/BIB_RAG_ROOT/BIB_RAG_COLLECTION. No code edit.
"""

import json
import os
from pathlib import Path

# ─── Fixed tool-code root ───────────────────────────────────────────────────
# Default code root = parent of this src/ directory (portable: wherever the
# repo is cloned, the toolkit follows). Override with BIB_RAG_CODE_ROOT.
_CODE_ROOT = os.environ.get(
    "BIB_RAG_CODE_ROOT",
    str(Path(__file__).resolve().parent.parent))

# ─── RAG home ──────────────────────────────────────────────────────────────
# Dir that contains the *_rag/ library folders. Defaults to the parent of this
# repo (so a cloned toolkit finds its sibling libraries anywhere). Override
# with BIB_RAG_HOME. No machine-specific fallback lives in source.
_DERIVED_RAG_HOME = Path(__file__).resolve().parent.parent.parent
_RAG_HOME = os.environ.get("BIB_RAG_HOME") or str(_DERIVED_RAG_HOME)


def get_kb_name() -> str:
    """Active library stem (e.g. 'eph_rag'). Required unless BIB_RAG_ROOT is set."""
    return os.environ.get("BIB_RAG_KB_NAME") or ""


def get_code_root() -> str:
    """Tool-code root (src/, scripts/). Independent of which library is active."""
    return _CODE_ROOT


def _read_collection_from_library_config(root: str) -> str:
    """collection from <root>/config.json settings block; '' if absent."""
    try:
        data = json.loads(Path(root, "config.json").read_text(encoding="utf-8"))
        s = data.get("settings", {})
        return s.get("collection", "") if isinstance(s, dict) else ""
    except Exception:
        return ""


def get_data_root() -> str:
    """Resolve the active library's data directory.
    Priority: BIB_RAG_ROOT env > <BIB_RAG_HOME>/<BIB_RAG_KB_NAME>.
    Fails loudly (no silent default) if neither resolves.
    """
    root = os.environ.get("BIB_RAG_ROOT")
    if root:
        return root
    name = get_kb_name()
    if name and Path(_RAG_HOME, name).is_dir():
        return str(Path(_RAG_HOME, name))
    if name:
        raise RuntimeError(
            "kb_config: no data root for library %r. Expected it at %r "
            "(does it exist?) or set BIB_RAG_ROOT explicitly."
            % (name, str(Path(_RAG_HOME, name))))
    raise RuntimeError(
        "kb_config: no library selected. Export BIB_RAG_KB_NAME=<name> (or "
        "BIB_RAG_ROOT=<root>), or run through the <name>-rag wrapper.")


def get_collection_name() -> str:
    """Collection for the active library.
    Priority: BIB_RAG_COLLECTION env > <root>/config.json settings.collection
    > <BIB_RAG_KB_NAME minus _rag>_papers.
    """
    env = os.environ.get("BIB_RAG_COLLECTION")
    if env:
        return env
    root = get_data_root()
    from_cfg = _read_collection_from_library_config(root)
    if from_cfg:
        return from_cfg
    name = get_kb_name()
    stem = name[:-4] if name.endswith("_rag") else name
    return f"{stem}_papers"


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
    Scan argv for --kb <name> or --kb=<name> and set BIB_RAG_KB_NAME accordingly.
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
