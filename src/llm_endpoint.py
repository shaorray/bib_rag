#!/usr/bin/env python3
"""
llm_endpoint.py — shared resolution of the local chat-LLM endpoint.

Before this module, subquery.py and distill.py each carried their own
`_llm_endpoint` / `_discover_model` / `_env_int` / `_env_float`. That is the
same drift trap src/embedding.py was created to close for the embedder: two
copies of the model-name resolution, two chances for one to 404 silently.

ONE contract, used by every LLM-calling helper (subquery, distill):

    llm_endpoint() -> (base_url, model)
      - base_url: LLM_URL env, else the active library's llm_url
      - model:    LLM_MODEL env, else the model the server reports at
                  /v1/models (cached per-process). A guessed placeholder id
                  (e.g. "local-model") 404s and silently degrades the caller,
                  so discovery beats guessing.
      - never raises; on total failure returns ("", "") and the caller
        degrades safely.
"""
from __future__ import annotations

import os

import requests

try:  # bib_rag-package-try
    from .kb_config import get_config
except ImportError:  # flat (loose-script mode)
    try:
        from kb_config import get_config
    except ImportError:  # src/ on sys.path directly
        from src.kb_config import get_config


def env_int(name: str, default: int) -> int:
    """Env int with silent fallback — a malformed override must not crash
    retrieval (matches broaden.py / reranker.py convention)."""
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


_MODEL_CACHE: dict = {}


def _discover_model(url: str) -> str:
    """First concrete (non-alias) model id served at `url`, cached. '' on
    failure. Prefers a local model over cloud/alias placeholders."""
    if url in _MODEL_CACHE:
        return _MODEL_CACHE[url]
    model = ""
    try:
        resp = requests.get(f"{url}/models", timeout=10)
        resp.raise_for_status()
        ids = [d.get("id") for d in (resp.json().get("data") or [])
               if isinstance(d, dict) and d.get("id")]
        ids.sort(key=lambda i: ("cloud" in i.lower(), "abiterated" in i.lower()))
        if ids:
            model = ids[0]
    except Exception:
        pass
    _MODEL_CACHE[url] = model
    return model


def llm_endpoint() -> tuple[str, str]:
    """(base_url, model) for the local chat LLM. Never raises."""
    url = os.environ.get("LLM_URL")
    if not url:
        try:
            url = get_config().get("llm_url", "http://localhost:5015/v1")
        except Exception:
            url = "http://localhost:5015/v1"
    url = url.rstrip("/")
    model = os.environ.get("LLM_MODEL") or _discover_model(url)
    return url, model
