#!/usr/bin/env python3
"""
broaden.py — Deterministic broadening of weak search results.

Borrowed mechanism (zotero-redisearch-rag should_broaden_retrieval /
retrieve_with_broadening, see /Disk_bot/notes/zotero_RAG/09): zero-LLM
signals decide whether a retrieval was TOO WEAK to trust, and a fixed
escalation ladder relaxes the search once. bib_rag's agent already
re-queries on its own, but a weak-result search (2 chunks of 30 chars with
a mediocre best score) looks "successful" to the evidence gate while being
garbage — this module catches exactly that gap.

Four weakness signals (any one triggers):
  S1 few chunks      → n_results < BROADEN_MIN_RESULTS
  S2 few characters  → total content < BROADEN_MIN_CHARS
  S3 weak best score → best similarity < BROADEN_MIN_SIM (bge-m3 cosine-ish
                       similarity is 1/(1+dist); calibrated: strong hits
                       land >0.45, noise drifts toward 0.2)
  S4 narrative-starved → narrative (prose-sentence) share of the retrieved
                       text < BROADEN_MIN_NARRATIVE (tables/reference dregs
                       match queries lexically but carry no claims)

Escalation ladder (one step per call, signature-guarded so the same plan
never runs twice):
  1. re-search WITHOUT the where filter, wider (limit ×3)
  2. re-search without filter, OR-split query terms, limit ×3

Output format mirrors the search tool's, with a [BROADENED] header line so
the agent knows the first pass was weak and why. Env kill-switch:
BROADEN_RETRY=0. All zero-LLM.
"""
from __future__ import annotations

import os
import re
import hashlib
from typing import Dict, List, Optional, Tuple


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


MIN_RESULTS = _env_int("BROADEN_MIN_RESULTS", 2)
MIN_CHARS = _env_int("BROADEN_MIN_CHARS", 300)
MIN_SIM = _env_float("BROADEN_MIN_SIM", 0.40)
MIN_NARRATIVE = _env_float("BROADEN_MIN_NARRATIVE", 0.30)

# Matches prose sentences (>=3 words, mostly letters); tables/lists/refs
# produce token debris that fails this test.
_SENTENCE_RE = re.compile(r"[^.!?\n]{40,}[.!?]")


def retrieval_metrics(results: List[dict]) -> Dict:
    """Deterministic quality metrics over a list of search-result dicts
    (each needs 'text'; 'similarity' optional)."""
    texts = [r.get("text", "") or "" for r in results]
    total_chars = sum(len(t) for t in texts)
    narrative_chars = sum(len(m.group(0)) for t in texts
                          for m in _SENTENCE_RE.finditer(t))
    sims = [r.get("similarity") for r in results
            if r.get("similarity") is not None]
    return {
        "n_results": len(results),
        "total_chars": total_chars,
        "narrative_share": (narrative_chars / total_chars) if total_chars else 0.0,
        "best_similarity": max(sims) if sims else None,
    }


def should_broaden(metrics: Dict) -> Tuple[bool, List[str]]:
    """The four-signal rule. Returns (verdict, triggered_signal_names)."""
    reasons: List[str] = []
    if metrics["n_results"] < MIN_RESULTS:
        reasons.append(f"few-results({metrics['n_results']}<{MIN_RESULTS})")
    if metrics["total_chars"] < MIN_CHARS:
        reasons.append(f"few-chars({metrics['total_chars']}<{MIN_CHARS})")
    best = metrics.get("best_similarity")
    if best is not None and best < MIN_SIM:
        reasons.append(f"weak-best-score({best:.2f}<{MIN_SIM})")
    if metrics["n_results"] and metrics["narrative_share"] < MIN_NARRATIVE:
        reasons.append(
            f"narrative-starved({metrics['narrative_share']:.2f}<{MIN_NARRATIVE})")
    return bool(reasons), reasons


def plan_broadening(query: str, had_where: bool, attempt: int) -> Optional[Dict]:
    """Next broadening step, or None when the ladder is exhausted.

    attempt 0 → drop where-filter, widen limit
    attempt 1 → additionally OR-split the query (lexical recall boost)
    Signature-guarded by the CALLER (it knows what already ran).
    """
    if attempt >= 2:
        return None
    plan = {"drop_where": True, "or_split": attempt >= 1, "attempt": attempt}
    return plan


def broaden_signature(query: str, plan: Dict) -> str:
    """Stable signature of (query, plan) — the caller dedups on this so the
    same broadened search never repeats within one agent run."""
    h = hashlib.sha1(
        f"{query}|{plan.get('drop_where')}|{plan.get('or_split')}".encode()
    ).hexdigest()[:12]
    return f"broaden::{h}"


def or_split_query(query: str) -> str:
    """Split a multi-term query into OR-terms for the vector store.

    ChromaDB `where`-only filtering can't express OR over query terms; the
    OR-boost happens by re-querying with the RAREST terms (longest tokens —
    gene symbols/method names) instead of the full phrase, which dense
    models often dilute. Returns '' for single-term queries (nothing to split).
    """
    terms = [t for t in re.findall(r"[\w][\w\-]*", query or "")
             if len(t) >= 2]
    if len(terms) <= 1:
        return ""
    # rarest = longest tokens first (Ephb2, knockdown > of, the)
    terms.sort(key=len, reverse=True)
    return " ".join(terms[:3])