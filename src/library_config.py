#!/usr/bin/env python3
"""
library_config.py — Per-library configuration layer.

A library folder (<name>_rag/) can carry its own `config.json` holding
machine/model settings that belong to THAT library (classify model path,
BibTeX file, temp dir, domain topic seeds). Resolution order for any setting:

    1. environment variable (BIB_RAG_* / CLASSIFY_MODEL / ...)  — session override
    2. <library>/config.json "settings" block                   — per-library default
    3. toolkit default                                          — generic fallback

The config file travels with the library (it sits inside <name>_rag/), so
moving or copying a library keeps its machine bindings. Secrets never go here
— API keys stay in the environment.

config.json shape (all keys optional):
{
  "library": "eph_rag",            // informational
  "settings": {
    "classify_model":  "/path/to/model.gguf",
    "llm_url":         "http://localhost:11434/v1",
    "llm_model":       "glm-5.2:cloud",
    "bib_path":        "/path/to/My Library.bib",
    "tmpdir":          "/path/to/scratch",
    "domain_topics":   ["topic-a", "topic-b"],
    "zotero_url":      "http://localhost:23119"
  }
}

setup_library.py writes a starter config.json with the settings block left
empty; tools read it via library_config.get_setting().
"""
import json
import os
from pathlib import Path

# env var that each settings key falls back to (session-level override)
_ENV_OVERRIDES = {
    "classify_model": "CLASSIFY_MODEL",
    "llm_url": "LLM_URL",
    "llm_model": "LLM_MODEL",
    "bib_path": "BIB_RAG_BIB_PATH",
    "tmpdir": "BIB_RAG_TMPDIR",
    "zotero_url": "BIB_RAG_ZOTERO_URL",
    "embed_url": "BIB_RAG_EMBED_URL",
}


def config_path(data_root: str) -> Path:
    return Path(data_root) / "config.json"


def load_library_settings(data_root: str) -> dict:
    """Read <library>/config.json settings block; {} if absent/invalid."""
    p = config_path(data_root)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        s = data.get("settings", {})
        return s if isinstance(s, dict) else {}
    except Exception as e:
        # a broken config must not brick the toolkit — warn and continue
        import sys
        print(f"[library_config] WARNING: could not parse {p}: {e}", file=sys.stderr)
        return {}


def get_setting(data_root: str, key: str, default=None):
    """Resolve one setting: env override > library config.json > default."""
    env = _ENV_OVERRIDES.get(key)
    if env and os.environ.get(env):
        return os.environ[env]
    return load_library_settings(data_root).get(key, default)


def write_starter(data_root: str, library: str, domain: str = "") -> Path:
    """Write a starter config.json (setup_library.py calls this). Never overwrites."""
    p = config_path(data_root)
    if p.exists():
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    starter = {
        "library": library,
        "domain": domain,
        "_comment": ("Per-library machine/model settings. Resolution: env var > "
                     "this file > toolkit default. Never put secrets here."),
        "settings": {
            # "classify_model": "/path/to/local-model.gguf",
            # "llm_url": "http://localhost:11434/v1",
            # "llm_model": "glm-5.2:cloud",
            # "bib_path": "/path/to/My Library.bib",
            # "tmpdir": "/path/to/scratch",
            # "domain_topics": ["topic-a", "topic-b"]
        },
    }
    p.write_text(json.dumps(starter, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return p