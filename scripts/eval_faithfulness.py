#!/usr/bin/env python3
"""
eval_faithfulness.py — end-to-end answer faithfulness eval (Feature 3).

Runs each gold_faithfulness.jsonl row through the FULL agentic graph (fresh
uuid4 thread_id per row + invoke_with_interrupt_handling — cached graph with a
fixed thread_id is a stale-answer bug), then scores zero-LLM:

  hallucination_rate      share of ledger claims with zero verified/external
                          evidence (in-process validate_claims, as agent_nodes
                          does; retrieval_keys from the graph result)
  citation_accuracy       share of cited Sources lines whose folded key is in
                          the expected_sources' folded keys (skipped for
                          refusal rows)
  refusal_correctness     refusal_expected AND a refusal marker in the answer
                          body (without Sources), or not refusal_expected and
                          no marker
  low_confidence_exposure share of weak-confidence claims surfaced in the
                          answer body

Usage:
    python3 scripts/eval_faithfulness.py [--kb eph_rag] [--gold <jsonl>]
        [--limit N] [--json out.json]

Exit codes: 0 ok · 2 LLM endpoint down · 3 gold file / KB missing.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import uuid

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
for _p in (_SRC, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from kb_config import get_config, parse_kb_arg  # noqa: E402

#: Answer phrases that signal a legitimate refusal. Feature 1's wording is
#: 'no literature supporting'.
REFUSAL_MARKERS = [
    "no literature supporting",
    "no supporting evidence",
    "not answerable",
    "cannot be answered",
]


# ── Key folding (identical to eval_retrieval.py / hybrid_search) ───────────

def _source_key(s: str) -> str:
    """Fold Chroma `source` values and parent_store filenames into one domain."""
    return re.sub(r"[^\w\-]", "_", s or "")[:100]


def _folded_expected(expected_sources):
    return {_source_key(e) for e in (expected_sources or [])}


# ── Refusal detection ──────────────────────────────────────────────────────

def strip_sources_section(answer: str) -> str:
    """Answer with its trailing Sources section removed (answer body only)."""
    from citation_guard import split_answer_sources
    try:
        body, _, _ = split_answer_sources(answer or "")
        return body
    except Exception:
        return answer or ""


def has_refusal_marker(answer_body: str) -> bool:
    low = (answer_body or "").lower()
    return any(m in low for m in REFUSAL_MARKERS)


def score_refusal(answer: str, refusal_expected: bool):
    """refusal_correctness = (expected AND marker) OR (unexpected AND none)."""
    body = strip_sources_section(answer)
    marker_present = has_refusal_marker(body)
    if refusal_expected:
        return marker_present, marker_present
    return not marker_present, marker_present


# ── Claim ledger scoring (in-process, mirrors agent_nodes.collect_answer) ──

def score_claims(claims, retrieval_keys, tool_events):
    """Validate claims like agent_nodes does and derive faithfulness shares.

    Returns (hallucination_rate, low_confidence_exposure, n_claims).
    hallucination_rate = share of claims with zero verified/external evidence.
    low_confidence_exposure = share of weak-confidence claims (0.0 when no
    claims were built).
    """
    if not claims:
        return 0.0, 0.0, 0
    from claim_ledger import validate_claims
    res = validate_claims(claims, retrieval_keys=retrieval_keys or set(),
                          tool_events=tool_events or [], strict=False)
    kept = res.claims
    if not kept:
        # every claim failed structural validation → all hallucinated
        return 1.0, 0.0, len(claims)
    n_hall = 0
    for c in kept:
        sc = c.get("status_counts", {})
        if not sc.get("verified") and not sc.get("external"):
            n_hall += 1
    n_weak = sum(1 for c in kept if c.get("confidence") == "weak")
    return n_hall / len(kept), n_weak / len(kept), len(kept)


def score_citations(source_lines, expected_sources, refusal_expected):
    """citation_accuracy = share of cited lines folding into expected_sources.

    Skipped (None) for refusal rows — there are no expected sources to match.
    Matching is containment on folded keys: answer source lines carry
    '#section#hash' suffixes ('22144690.md#results#zz99') that a bare-filename
    gold entry must still match (fold(expected) in fold(line)).
    """
    if refusal_expected:
        return None, len(source_lines)
    expected = _folded_expected(expected_sources)
    if not source_lines:
        return 0.0, 0
    hit = 0
    for line in source_lines:
        fl = _source_key(line)
        if any(exp and exp in fl for exp in expected):
            hit += 1
    return hit / len(source_lines), len(source_lines)


# ── LLM reachability probe (exit 2 path) ───────────────────────────────────

def probe_llm(base_url, model, api_key):
    """True when the OpenAI-compatible endpoint answers a 1-token request.

    Connection-refused / timeout → False (caller prints a clear message and
    exits 2). Any HTTP response (even 401) counts as 'up'.
    """
    try:
        import urllib.request
        req = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions",
            data=json.dumps({"model": model, "messages": [{"role": "user",
                             "content": "ping"}], "max_tokens": 1}).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {api_key}"},
            method="POST")
        urllib.request.urlopen(req, timeout=10)
        return True
    except Exception as e:
        # 401/404/etc still mean the server is alive
        return hasattr(e, "code") and isinstance(getattr(e, "code", None), int)


# ── Graph invocation (same pattern as agentic_query.run_query) ─────────────

def _answer_text(msg) -> str:
    c = getattr(msg, "content", "")
    if isinstance(c, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b)
                       for b in c)
    return c if isinstance(c, str) else str(c)


def run_row(graph, query: str):
    """One row through the graph. Returns (answer, retrieval_keys, messages)."""
    from langchain_core.messages import HumanMessage
    from agentic_query import invoke_with_interrupt_handling
    config = {"configurable": {"thread_id": f"eval-{uuid.uuid4()}"}}
    result = invoke_with_interrupt_handling(
        graph,
        {"messages": [HumanMessage(content=query)]},
        config,
        follow_up_provider=None,
    )
    retrieval_keys = set()
    for aa in result.get("agent_answers", []) or []:
        retrieval_keys.update(aa.get("retrieval_keys") or [])
    if not retrieval_keys:
        try:
            snap = graph.get_state(config)
            retrieval_keys.update(snap.values.get("retrieval_keys") or [])
        except Exception:
            pass
    messages = list(result.get("messages") or [])
    answer = _answer_text(messages[-1]) if messages else ""
    return answer, retrieval_keys, messages


def main():
    argv = parse_kb_arg()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold", default=None)
    ap.add_argument("--limit", type=int, default=0, help="0 = all rows")
    ap.add_argument("--json", default=None, help="dump per-query results")
    args = ap.parse_args(argv)

    try:
        cfg = get_config()
    except RuntimeError as e:
        print(f"kb_config: {e}", file=sys.stderr)
        return 3
    gold_path = args.gold or os.path.join(cfg["data_dir"],
                                          "gold_faithfulness.jsonl")
    if not os.path.isfile(gold_path):
        print(f"gold file missing: {gold_path}", file=sys.stderr)
        return 3
    if not os.path.isdir(cfg["parent_store_dir"]):
        print(f"parent store missing: {cfg['parent_store_dir']}",
              file=sys.stderr)
        return 3
    rows = []
    with open(gold_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if args.limit:
        rows = rows[:args.limit]

    # env probe FIRST (clear message, exit 2) — ChatOpenAI defers the
    # connection, so a dead endpoint would otherwise surface as per-row errors.
    llm_url = os.environ.get("LLM_URL", "http://localhost:11434/v1")
    llm_model = os.environ.get("LLM_MODEL", "glm-5.2:cloud")
    llm_api_key = os.environ.get("LLM_API_KEY", "not-required")
    if not probe_llm(llm_url, llm_model, llm_api_key):
        print(f"LLM endpoint unreachable: {llm_url} (model {llm_model}). "
              f"Start the gateway (default :11434) or vLLM (:5015) first.",
              file=sys.stderr)
        return 2

    # kb_config.parse_kb_arg() ran above → get_config() at import time is safe
    from agentic_query import create_llm, _get_cached_graph
    from src.agent_tools import create_tools
    from citation_guard import split_answer_sources
    from claim_ledger import build_claims_from_sources, messages_to_tool_events

    llm = create_llm()
    graph = _get_cached_graph(llm, create_tools())
    expected_by_fold = {r: r for r in []}  # placeholder, unused

    per_q, t0 = [], time.time()
    for r in rows:
        refusal_expected = bool(r.get("refusal_expected"))
        expected = r.get("expected_sources") or []
        try:
            answer, retrieval_keys, messages = run_row(graph, r["query"])
        except Exception as e:
            per_q.append({"id": r.get("id"), "query": r["query"],
                          "error": f"{type(e).__name__}: {e}"})
            print(f"  [{r.get('id')}] ERROR {type(e).__name__}: {e}",
                  file=sys.stderr)
            continue

        body, source_lines, _ = split_answer_sources(answer)
        claims = build_claims_from_sources(answer)
        tool_events = messages_to_tool_events(messages)
        hall_rate, weak_exposure, n_claims = score_claims(
            claims, retrieval_keys, tool_events)
        cit_acc, n_cited = score_citations(source_lines, expected,
                                           refusal_expected)
        ref_ok, marker_present = score_refusal(answer, refusal_expected)

        per_q.append({
            "id": r.get("id"), "query": r["query"],
            "refusal_expected": refusal_expected,
            "n_claims": n_claims, "n_cited": n_cited,
            "hallucination_rate": round(hall_rate, 3),
            "citation_accuracy": (None if cit_acc is None else round(cit_acc, 3)),
            "refusal_correct": ref_ok,
            "refusal_marker_present": marker_present,
            "low_confidence_exposure": round(weak_exposure, 3),
            "answer_chars": len(answer),
        })
        print(f"  [{r.get('id')}] hall={hall_rate:.2f} "
              f"cit={('—' if cit_acc is None else f'{cit_acc:.2f}')} "
              f"ref={'ok' if ref_ok else 'X'} weak={weak_exposure:.2f} "
              f"{r['query'][:48]}")

    # ── aggregates ────────────────────────────────────────────────────────
    answered = [q for q in per_q if "error" not in q]
    nonref = [q for q in answered if not q["refusal_expected"]]
    ref = [q for q in answered if q["refusal_expected"]]
    n = len(answered)
    agg = {
        "n_rows": len(per_q), "n_answered": n, "elapsed_s": round(time.time() - t0, 1),
        "hallucination_rate": (round(sum(q["hallucination_rate"] for q in answered) / n, 3)
                               if n else None),
        "citation_accuracy": (round(sum(q["citation_accuracy"] for q in nonref)
                                    / len(nonref), 3) if nonref else None),
        "refusal_correctness": (round(sum(q["refusal_correct"] for q in answered) / n, 3)
                                if n else None),
        "low_confidence_exposure": (round(sum(q["low_confidence_exposure"] for q in answered) / n, 3)
                                    if n else None),
    }
    print(f"\n{'=' * 70}")
    print(f"hallucination_rate      : {agg['hallucination_rate']}")
    print(f"citation_accuracy       : {agg['citation_accuracy']} "
          f"(non-refusal rows: {len(nonref)})")
    print(f"refusal_correctness     : {agg['refusal_correctness']} "
          f"(refusal rows: {len(ref)})")
    print(f"low_confidence_exposure : {agg['low_confidence_exposure']}")
    print(f"rows: {n}/{len(per_q)} answered   elapsed: {agg['elapsed_s']}s")

    if args.json:
        json.dump({"aggregate": agg, "per_query": per_q},
                  open(args.json, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"per-query results -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())