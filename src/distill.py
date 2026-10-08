#!/usr/bin/env python3
"""
distill.py — Per-document "analyst extraction" stage for bib_rag retrieval.

Borrowed mechanism (Khoj `extract_relevant_info` / `extract_relevant_summary`,
see github.com/khoj-ai/khoj src/khoj/routers/helpers.py + processor/
conversation/prompts.py): after retrieval, instead of shoving raw chunks
straight into the answering LLM, a "professional analyst" model reads each
retrieved passage WITH the target query and extracts only the information
that answers it — and is instructed to QUOTE VERBATIM, so every extracted
sentence is traceable to the source text.

Why this fits bib_rag: the toolkit's whole contract is "every claim is
backed by the source" (citation_guard, evidence_gate). Distillation makes
that contract cheaper to keep — the extract step is constrained (quote the
document, no external info) and it compresses multi-document context before
synthesis. It is NOT a summarizer: it returns only answer-relevant, verbatim
spans.

Relationship to existing stages:
  search → fuse → rerank  (which chunks)   ← hybrid_search.py
  → DISTILL              (what in them)      ← this module
  → synthesize           (the answer)       ← agent / writer

Domain adaptation vs Khoj: Khoj's extractor returns prose summaries. For an
academic corpus the answer-bearing unit is often a claim sentence or a
result statement, so the prompt asks for verbatim claim spans (with the
sentence kept intact), not a paraphrase.

Contract:
    distill_hits(query, hits, ...) -> List[dict]
      - hits: list of entry dicts with at least {"text", ...}; returns the
        SAME dicts, each with an added "distilled" key (verbatim extract).
      - On ANY failure (LLM down, timeout, empty output) each hit's
        "distilled" is left unset and the ORIGINAL text is used downstream —
        a missing distill never loses a hit and never fabricates.
      - Never raises.

Env:
    DISTILL=0            disable the stage (kill-switch)
    DISTILL_MAX_CHARS    source passage cap fed to the extractor (default 1500)
    DISTILL_TIMEOUT      per-call timeout (default 45)
    DISTILL_MAX_TOKENS   generation cap (default 400)
    DISTILL_MIN_SIM      skip hits below this similarity (default 0 = no skip)
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import requests

try:  # bib_rag-package-try
    from .llm_endpoint import llm_endpoint as _llm_endpoint_shared
except ImportError:  # flat (loose-script mode)
    try:
        from llm_endpoint import llm_endpoint as _llm_endpoint_shared
    except ImportError:  # src/ on sys.path directly (no package context)
        import os as _os, sys as _sys
        _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
        from llm_endpoint import llm_endpoint as _llm_endpoint_shared


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _llm_endpoint() -> tuple[str, str]:
    """Delegates to the shared resolver (src/llm_endpoint.py)."""
    return _llm_endpoint_shared()





# ─── prompt (adapted from Khoj system_prompt_extract_relevant_information) ──
SYSTEM_PROMPT = """You are a meticulous research analyst extracting evidence from a scientific paper.

You will be given a <target_query> and a <document> passage taken directly from one paper. \
Extract ONLY the information in the passage that helps answer the target query.

Rules:
1. Quote the passage VERBATIM. Reproduce the exact wording of claim-bearing sentences and any \
numbers, gene/receptor names, organisms, or statistics. Do not paraphrase, reinterpret, or translate.
2. Rely strictly on the passage. Never add outside knowledge, never infer results the passage does \
not state, never guess a citation.
3. If the passage contains nothing relevant to the target query, reply exactly: NOTHING_RELEVANT
4. Prefer the claim/result sentences that bear on the query over background or methods boilerplate.
5. Output the verbatim extract only — no preamble, no commentary, no markdown fences."""


def _build_message(query: str, passage: str) -> str:
    return (f"<target_query>\n{query}\n</target_query>\n\n"
            f"<document>\n{passage}\n</document>\n\n"
            f"Extract the verbatim evidence from the document that answers the target query.")


NOTHING = "NOTHING_RELEVANT"


def _call_llm(query: str, passage: str, timeout: int, max_tokens: int) -> Optional[str]:
    """One analyst call. Returns the verbatim extract, None on any failure."""
    url, model = _llm_endpoint()
    try:
        resp = requests.post(
            f"{url}/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _build_message(query, passage)},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.0,
                # Local reasoning models need this or content comes back empty.
                "reasoning_effort": "none",
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        content = (resp.json()["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        return None
    if not content or content.strip() == NOTHING:
        return None
    return content


def is_available() -> bool:
    """True unless the stage is explicitly disabled. Cheap; no network I/O."""
    return os.environ.get("DISTILL", "1") != "0"


def distill_hits(query: str, hits: List[dict],
                 timeout: Optional[int] = None,
                 max_tokens: Optional[int] = None) -> List[dict]:
    """Attach a verbatim "distilled" extract to each hit, in place.

    Only hits whose extractor call succeeds AND returns relevant text get the
    key. Every other stage (unreachable LLM, timeout, NOTHING_RELEVANT, error)
    leaves the hit untouched, so downstream code that prefers hit.get(
    "distilled") or hit["text"] always has real source text to fall back on.
    """
    if not hits or not query:
        return hits
    if os.environ.get("DISTILL", "1") == "0":
        return hits

    timeout = timeout if timeout is not None else _env_int("DISTILL_TIMEOUT", 45)
    max_tokens = max_tokens if max_tokens is not None else _env_int("DISTILL_MAX_TOKENS", 400)
    max_chars = _env_int("DISTILL_MAX_CHARS", 1500)
    min_sim = _env_float("DISTILL_MIN_SIM", 0.0)

    for h in hits:
        sim = h.get("similarity")
        if min_sim and sim is not None and sim < min_sim:
            continue
        passage = (h.get("text") or "").strip()
        if not passage:
            continue
        out = _call_llm(query, passage[:max_chars], timeout, max_tokens)
        if out:
            h["distilled"] = out
    return hits


def distilled_text(hit: dict) -> str:
    """Preferred text for a hit: the verbatim extract when present, else raw."""
    return hit.get("distilled") or hit.get("text") or ""


# ─── CLI (manual inspection / debugging) ─────────────────────────────────────
def _cli() -> int:
    import sys
    if len(sys.argv) < 3:
        print("Usage: python3 distill.py \"query\" \"passage text\"")
        return 1
    out = _call_llm(sys.argv[1], sys.argv[2],
                    _env_int("DISTILL_TIMEOUT", 45),
                    _env_int("DISTILL_MAX_TOKENS", 400))
    print(out if out is not None else f"(no extract / {NOTHING} / LLM unreachable)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
