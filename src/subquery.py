#!/usr/bin/env python3
"""
subquery.py — Lightweight LLM sub-query rewriting for bib_rag retrieval.

Borrowed mechanism (Khoj `extract_questions`, see github.com/khoj-ai/khoj
src/khoj/processor/conversation/prompts.py): ONE fast-model call decomposes
the user's question into ≤N retrieval queries, each carrying ONE concept,
explicitly forbidding boolean operators (they degrade dense retrieval).

Why this exists here: bib_rag already rewrites queries via the agentic
graph (agent_nodes.rewrite_query → QueryAnalysis), but that path is
graph-coupled, requires the full LangGraph stack, and is unavailable to
the plain CLI (`query_bib_rag.search`) and to any non-agentic caller.
This module is the dependency-light seam: text in → queries out, no
langgraph / no langchain.

Domain adaptation (NOT a literal Khoj port):
  - Khoj asks for who/what/where/when/why/how perspectives; that suits
    personal notes. For an academic corpus the useful decomposition axes
    are mechanism / entity / method / context. The prompt below uses
    those.
  - Khoj targets a *personal* KB and forbids boolean operators; we keep
    that rule (BM25 channel ANDs terms, dense dilutes on long strings).
  - Gene/receptor aliases (see CONTEXT.md): the prompt asks the model to
    emit an alias-bearing query as its OWN query rather than gluing
    aliases into every query (one concept per query).

Contract:
    expand_query(query, n=3, timeout=...) -> List[str]
      - returns the ORIGINAL query first, always
      - returns [] when the LLM is unreachable / output is unparseable
        (callers treat [] as "no expansion, run the original") — never
        raises, never fabricates.
      - deterministic de-dup, preserves order, hard cap n.

Env:
    SUBQUERY_EXPAND=0   disable expansion entirely (kill-switch)
    SUBQUERY_N          max extra queries (default 3)
    SUBQUERY_TIMEOUT    per-call timeout seconds (default 30)
    SUBQUERY_MAX_TOKENS generation cap (default 300)
"""
from __future__ import annotations

import json
import os
import re
from typing import List, Optional

import requests

try:  # bib_rag-package-try
    from .llm_endpoint import llm_endpoint as _llm_endpoint, env_int as _env_int  # noqa: F401
except ImportError:  # flat (loose-script mode)
    try:
        from llm_endpoint import llm_endpoint as _llm_endpoint, env_int as _env_int  # noqa: F401
    except ImportError:  # src/ on sys.path directly (no package context)
        import os as _os, sys as _sys
        _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
        from llm_endpoint import llm_endpoint as _llm_endpoint, env_int as _env_int  # noqa: F401


# ─── prompt ────────────────────────────────────────────────────────────────
# Ported in spirit from Khoj's extract_questions_system_prompt, re-aimed at
# an academic literature corpus. Few-shot examples are kept short on purpose:
# a 3B-27B local model follows 3 concrete examples better than 6 verbose ones.
SYSTEM_PROMPT = """You are a query analyst for a scientific literature search engine over papers on \
Eph receptors, ephrins, axon guidance, cell repulsion, and related developmental/cancer biology.

Decompose the user's question into up to {n} search queries that retrieve the supporting literature.

Rules:
1. One concept per query. Do NOT combine terms with AND/OR — boolean operators degrade both
   semantic and lexical retrieval. Emit separate queries instead.
2. Cover complementary angles, e.g.: the core mechanism; the specific molecules/receptors/ligands;
   the experimental method or assay; the biological context (organism, tissue, disease).
3. For a question about a specific molecule, emit ONE query that spells out its known aliases
   (e.g. "EFNB2 Efnb2 ephrin-B2 EPLG5 LERK5") as its own query — aliases are strong lexical signals.
4. Keep every query self-contained and semantically faithful to the user's question. Do not invent
   facts, results, or paper titles.
5. If the question is already a single focused lookup, return just one query.

Examples:
Q: Does ephrin-B1 signalling through EphB2 require phosphorylation?
A: {{"queries": ["ephrin-B1 EphB2 reverse signalling phosphorylation", "EphB2 tyrosine phosphorylation kinase domain", "ephrin-B reverse signaling mechanism"]}}

Q: How does EphA4 regulate somite boundary formation?
A: {{"queries": ["EphA4 somite boundary formation", "ephrin EphA4 segmentation clock", "EphA4 ephrin-B2 somite epithelialization"]}}

Q: What are the main methods used to study Eph receptor forward signalling?
A: {{"queries": ["Eph receptor forward signalling assay", "EphA activation phosphorylation methods", "Eph receptor clustering experimental approaches"]}}

Current date: {today}

Return ONLY a JSON object: {{"queries": ["...", "..."]}}. No prose, no markdown fences."""


def _build_user_message(query: str) -> str:
    return f"Question:\n{query}"


# ─── parsing ────────────────────────────────────────────────────────────────
_JSON_OBJ = re.compile(r"\{.*\}", re.S)


def _parse_queries(text: str) -> List[str]:
    """Pull the queries list out of a model response.

    Tolerant of: bare JSON, fenced ```json blocks, prose wrapped around the
    object, and the model returning a bare JSON list. Returns [] when nothing
    parseable is found (caller degrades to the original query).
    """
    if not text:
        return []
    raw = text.strip()
    # strip markdown fences
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    candidates = [raw]
    m = _JSON_OBJ.search(raw)
    if m:
        candidates.append(m.group(0))
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        if isinstance(obj, dict):
            qs = obj.get("queries")
        elif isinstance(obj, list):
            qs = obj
        else:
            qs = None
        if isinstance(qs, list):
            out = [str(q).strip() for q in qs if isinstance(q, (str, int, float))]
            out = [q for q in out if q]
            if out:
                return out
    return []


# ─── public API ─────────────────────────────────────────────────────────────
def expand_query(query: str, n: Optional[int] = None,
                 timeout: Optional[int] = None) -> List[str]:
    """Decompose `query` into ≤n retrieval queries via one local LLM call.

    Returns the original query FIRST, followed by the model's distinct extra
    queries (order preserved, hard-capped at n). Returns just [query] when:
      - expansion is disabled (SUBQUERY_EXPAND=0)
      - the query is empty
      - the LLM is unreachable, errors, or returns unparseable output

    Never raises. Never fabricates: on any failure the caller simply runs the
    original query, exactly as before this module existed.
    """
    query = (query or "").strip()
    if not query:
        return []
    if os.environ.get("SUBQUERY_EXPAND", "1") == "0":
        return [query]

    n = n if n is not None else _env_int("SUBQUERY_N", 3)
    if n <= 0:
        return [query]
    timeout = timeout if timeout is not None else _env_int("SUBQUERY_TIMEOUT", 30)
    max_tokens = _env_int("SUBQUERY_MAX_TOKENS", 300)

    url, model = _llm_endpoint()
    prompt = SYSTEM_PROMPT.format(
        n=n + 1,  # the model's first query is usually the original restated
        today=__import__("datetime").date.today().isoformat(),
    )
    try:
        resp = requests.post(
            f"{url}/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": _build_user_message(query)},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.2,
                # Local Qwen/GLM reasoning models burn the token budget on
                # chain-of-thought and return empty content without this
                # (see TOOLS.md FreeToken note). Harmless for models that
                # ignore it.
                "reasoning_effort": "none",
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
    except Exception:
        return [query]  # LLM down / slow / malformed → no expansion

    extra = _parse_queries(content)

    # de-dup (case-insensitive exact) preserving order; original always first.
    seen = {query.strip().lower()}
    out = [query]
    for q in extra:
        key = q.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(q.strip())
        if len(out) >= n + 1:
            break
    return out


def is_available() -> bool:
    """True unless expansion is explicitly disabled. Cheap; no network I/O."""
    return os.environ.get("SUBQUERY_EXPAND", "1") != "0"


# ─── CLI (manual inspection / debugging) ─────────────────────────────────────
def _cli() -> int:
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 subquery.py \"your question\" [n]")
        return 1
    q = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else None
    queries = expand_query(q, n=n)
    print(json.dumps(queries, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
