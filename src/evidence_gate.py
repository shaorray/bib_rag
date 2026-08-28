#!/usr/bin/env python3
"""
evidence_gate.py — Evidence sufficiency check + explicit gap reporting.

Borrowed mechanism (ragent 证据门槛, see /Disk_bot/notes/Agentic_RAG/ragent_技术笔记.md):
before the agent settles for an answer (especially in fallback_response when
iteration limits were hit), a deterministic gate checks whether each retrieval
query actually returned evidence. The gate produces:
  1. a coverage report the fallback prompt MUST honour ("X has no evidence
     in the KB" becomes part of the output contract, not an afterthought);
  2. a coverage block appended to the answer listing unanswerable aspects.

Zero LLM. The "did this query return anything" signal comes from the tool
messages already in the transcript: search_child_chunks replies with either
formatted results or NO_RELEVANT_CHUNKS / RETRIEVAL_ERROR.

Usage (in agent_nodes.fallback_response):
    from .evidence_gate import evidence_coverage
    gate = evidence_coverage(state["messages"], state.get("retrieval_keys", set()))
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Set

# Tool output markers that mean "this retrieval produced nothing usable".
_EMPTY_MARKERS = (
    "NO_RELEVANT_CHUNKS",
    "NO_PARENT_DOCUMENT",
    "NO_PARENT_DOCUMENTS",
    "EMBEDDING_ERROR",
    "RETRIEVAL_ERROR",
    "PARENT_RETRIEVAL_ERROR",
)

# Sentinel the tools emit when results exist but were clipped empty.
_MIN_RESULTS = 1

# All queries from retrieval_keys 'search::<query>' entries (the dedup ledger
# tracks every query the agent ran) — this is the complete retrieval history,
# surviving even compress_context message wipes.
_SEARCH_KEY_RE = re.compile(r"^search::(.+)$")

# ToolMessage content prefix emitted by search_child_chunks on success
_RESULT_PREFIX = "--- RESULT"

# Queries shorter than this are too vague to judge coverage on.
_MIN_QUERY_LEN = 8


def _queries_from_keys(retrieval_keys: Set[str]) -> List[str]:
    out = []
    for k in retrieval_keys or set():
        m = _SEARCH_KEY_RE.match(k)
        if m and len(m.group(1).strip()) >= _MIN_QUERY_LEN:
            out.append(m.group(1).strip())
    return out


def _tool_messages(messages) -> List[tuple]:
    """(tool_name, content) pairs for retrieval tool messages."""
    pairs = []
    for m in messages or []:
        name = getattr(m, "name", "") or ""
        content = getattr(m, "content", "")
        if content and isinstance(content, str):
            pairs.append((name, content))
    return pairs


def evidence_coverage(messages, retrieval_keys: Set[str]) -> Dict:
    """Deterministic evidence audit.

    Returns {
      'queries':            all distinct queries the agent ran,
      'empty_queries':      queries whose tool output was an empty marker,
      'productive_queries': queries that returned evidence,
      'no_evidence':        True when NO query produced evidence,
      'n_results_total':    approximate count of retrieved result blocks,
    }
    """
    queries = _queries_from_keys(retrieval_keys)
    tool_pairs = _tool_messages(messages)

    # Map each query → best tool output status. A query is "productive" if
    # any tool message it plausibly produced contains results. Because
    # compress_context may have wiped the messages, fall back to:
    # if messages were wiped we cannot link outputs → treat queries as
    # productive unless the CURRENT transcript shows empty markers.
    empty_markers = 0
    result_blocks = 0
    for _name, content in tool_pairs:
        if any(mark in content for mark in _EMPTY_MARKERS):
            empty_markers += 1
        elif "--- RESULT" in content or "--- PARENT" in content:
            result_blocks += content.count("--- RESULT") + content.count("--- PARENT")

    empty_queries = []
    productive = list(queries)
    if empty_markers > 0 and not result_blocks and queries:
        # every visible retrieval was empty
        empty_queries = queries
        productive = []
    elif empty_markers > 0 and result_blocks and queries:
        # mixed — cannot attribute per-query after compression; report both
        empty_queries = []
    else:
        empty_queries = []

    return {
        "queries": queries,
        "empty_queries": empty_queries,
        "productive_queries": productive,
        "no_evidence": bool(queries) and empty_markers > 0 and result_blocks == 0,
        "n_results_total": result_blocks,
    }


def coverage_block(coverage: Dict) -> str:
    """The mandatory transparency block appended to fallback answers."""
    lines: List[str] = []
    if coverage.get("no_evidence"):
        lines.append(
            "**Evidence coverage:** No retrieval in this session returned "
            "usable results — the answer below reflects only weak/partial "
            "signals; treat it as ungrounded."
        )
    elif coverage.get("queries") and coverage.get("n_results_total", 0) == 0:
        lines.append(
            "**Evidence coverage:** Retrieval ran but no results are visible "
            "in the current context (compressed); grounding could not be "
            "re-verified."
        )
    return "\n\n".join(lines)


def gap_instruction(coverage: Dict) -> str:
    """Extra instruction injected INTO the fallback prompt (not the answer)
    forcing explicit reporting of unanswerable aspects."""
    if coverage.get("no_evidence"):
        return (
            "EVIDENCE GATE: NO retrieval in this session returned usable "
            "results. You must state this explicitly and list what kinds of "
            "sources would be needed, instead of presenting an answer."
        )
    if coverage.get("empty_queries"):
        return (
            f"EVIDENCE GATE: {len(coverage['empty_queries'])} of "
            f"{len(coverage['queries'])} retrievals returned nothing. In the "
            "answer, explicitly flag which aspects of the question have NO "
            "supporting evidence in the knowledge base."
        )
    return ""


def search_child_chunks_tool_note() -> str:
    """One-line note for the orchestrator prompt about hybrid channels."""
    return (
        "search_child_chunks fuses dense + BM25 channels (RRF); results "
        "carry a `channels:` tag — results found by BOTH channels are "
        "typically the strongest matches."
    )