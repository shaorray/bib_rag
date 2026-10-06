#!/usr/bin/env python3
"""
embedding.py — single bge-m3 query embedding path for the whole toolkit.

Before this module, five files each carried their own `embed_query` (and their
own copy of the llama-server POST). That is five places for the model name,
port, timeout, or normalization to drift — and a drifted embedder silently
returns incomparable vectors (retrieval quietly degrades, nothing errors).

Everything that needs a query vector now calls here, so there is ONE contract:
POST the active library's embed_url, "bge-m3", 30s, L2-normalized on success.
"""

from __future__ import annotations

import os
from typing import List, Optional

import requests


def _embed_url() -> str:
    """Active library's embedding endpoint, via kb_config when available."""
    try:
        try:
            from .kb_config import get_config
        except ImportError:  # flat (loose-script mode)
            from kb_config import get_config
        return get_config()["embed_url"]
    except Exception:
        return os.environ.get("BIB_RAG_EMBED_URL", "http://localhost:8081/v1/embeddings")


def embed_query(text: str, timeout: int = 30) -> Optional[List[float]]:
    """Embed query text via llama-server bge-m3, L2-normalized.

    Returns the vector on success, None on any failure (the toolkit's callers
    already treat None as "retrieval unavailable" and degrade loudly).
    """
    try:
        resp = requests.post(
            _embed_url(),
            headers={"Content-Type": "application/json"},
            json={"input": text, "model": "bge-m3"},
            timeout=timeout,
        )
        emb = resp.json()["data"][0]["embedding"]
        norm = sum(x * x for x in emb) ** 0.5
        return [x / norm for x in emb] if norm > 0 else emb
    except Exception as e:
        print(f"❌ Embedding failed: {e}")
        return None
