#!/usr/bin/env python3
"""
kb_config.py — Shared knowledge-base configuration for bib_rag tools.

Supports multiple named knowledge bases via BIB_RAG_ROOT env var.
Defaults to the original bib_rag (Eph-ephrin) database.

Usage:
  # Default (bib_rag / Eph-ephrin):
  python3 -B query_bib_rag.py "Eph receptor signaling"

  # Switch to geo_rag:
  BIB_RAG_ROOT=/Disk_bot/Eph/geo_rag python3 -B query_bib_rag.py "subduction zone"

  # Or use --kb flag (if script supports it):
  python3 -B query_bib_rag.py --kb geo_rag "subduction zone"

Named KB registry:
  Set BIB_RAG_KB_NAME to "bib_rag" (default) or "geo_rag", and BIB_RAG_ROOT
  to the KB directory. If only BIB_RAG_KB_NAME is set, the root is auto-resolved.
"""

import os
from pathlib import Path

# ─── Named KB registry ─────────────────────────────────────────────────────
_KB_REGISTRY = {
    "bib_rag": "/Disk_bot/Eph/bib_rag",
    "geo_rag": "/Disk_bot/Eph/geo_rag",
}

def get_kb_name() -> str:
    """Get the active KB name from env, default 'bib_rag'."""
    return os.environ.get("BIB_RAG_KB_NAME", "bib_rag")

def get_kb_root() -> str:
    """
    Resolve the active KB root directory.
    Priority: BIB_RAG_ROOT env > BIB_RAG_KB_NAME registry > default bib_rag.
    """
    # Explicit root override
    root = os.environ.get("BIB_RAG_ROOT")
    if root:
        return root

    # Named KB lookup
    name = get_kb_name()
    if name in _KB_REGISTRY:
        return _KB_REGISTRY[name]

    # Fallback: default
    return "/Disk_bot/Eph/bib_rag"

def get_config() -> dict:
    """
    Return all paths/urls for the active knowledge base.
    Embedding endpoint is shared (bge-m3 is domain-general).
    """
    root = get_kb_root()
    return {
        "kb_root": root,
        "kb_name": get_kb_name(),
        "chroma_path": os.path.join(root, "chroma_db_new"),
        "chroma_sqlite": os.path.join(root, "chroma_db_new", "chroma.sqlite3"),
        "parent_store_dir": os.path.join(root, "parent_store"),
        "parent_store_disabled_dir": os.path.join(root, "parent_store_disabled"),
        "metadata_log": os.path.join(root, "data", "incremental_metadata.json"),
        "checkpoint_file": os.path.join(root, "data", "build_hierarchical_checkpoint.json"),
        "embed_url": "http://localhost:8081/v1/embeddings",
        "embed_url_raw": "http://localhost:8081/embedding",
        "collection_name": os.environ.get("BIB_RAG_COLLECTION", "bib_rag_papers"),
    }

# ─── Convenience: CLI --kb flag support ────────────────────────────────────
def parse_kb_arg(argv: list = None) -> str:
    """
    Scan argv for --kb <name> or --kb=<name> and set env accordingly.
    Returns the remaining argv (with --kb stripped).
    Call at the top of any script that wants --kb support.
    """
    import sys
    if argv is None:
        argv = sys.argv[1:]

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

    # Auto-resolve root from name if not explicitly set
    name = os.environ.get("BIB_RAG_KB_NAME", "bib_rag")
    if "BIB_RAG_ROOT" not in os.environ and name in _KB_REGISTRY:
        os.environ["BIB_RAG_ROOT"] = _KB_REGISTRY[name]

    return remaining

def print_config():
    """Print active KB config (for debugging)."""
    cfg = get_config()
    print(f"  KB name:       {cfg['kb_name']}")
    print(f"  KB root:       {cfg['kb_root']}")
    print(f"  ChromaDB:      {cfg['chroma_path']}")
    print(f"  Parent store:  {cfg['parent_store_dir']}")
    print(f"  Embed URL:     {cfg['embed_url']}")
    print(f"  Collection:    {cfg['collection_name']}")