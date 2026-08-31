#!/usr/bin/env python3
"""test_guard_redirect.py — E2E pin for the citation-guard redirect loop.

Reproduces the F1 stale-flag bug class: budget-exhausted collect_answer
returns must CLEAR guard_redirect, or route_after_collect sees the stale
True and routes back to the orchestrator forever (GraphRecursionError).

Strategy: patch orchestrator_call to emit an answer whose Sources lines
reference papers never retrieved — the deterministic citation guard drops
every line on every round. With GUARD_REDIRECT_MAX=1:
  round 1 → guard fails → redirect (guard_retries 0→1)
  round 2 → guard fails again → budget exhausted → terminal return
The pre-fix code forgot guard_redirect=False on the exhausted return →
the router read stale True → infinite orchestrator loop. This test fails
(red) on that code and passes (green) on the fix.

Run: /usr/bin/python3.10 -B tests/test_guard_redirect.py
"""

import os
import sys
import importlib.util
import inspect
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("CITATION_GUARD", "1")
os.environ["GUARD_REDIRECT_MAX"] = "1"     # smallest budget → fastest loop
os.environ.setdefault("HYDE", "0")          # keep the graph minimal

CHECKS = []
def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond), detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f" — {detail}" if detail and not cond else ""))

# ── module loading: load agent_nodes/edges as the bib_rag package ─────────
# (they use pure relative imports — package context is required; loose
# spec_from_file_location fails with "no known parent package")
import bib_rag  # noqa: F401  (ensures the editable install resolves)

import bib_rag.agent_schemas as schemas
import bib_rag.agent_nodes as nodes
import bib_rag.agent_edges as edges

try:
    # ── unit pin: the exhausted branch clears guard_redirect ──────────────
    src = inspect.getsource(nodes.collect_answer)
    check("collect_answer source: exhausted branch clears guard_redirect",
          '"guard_redirect": False' in src,
          "budget-exhausted return must overwrite the stale flag")

    # ── E2E: subgraph run with always-unverifiable citations ─────────────
    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.graph import START, END, StateGraph
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.prebuilt import ToolNode

    print("E2E: subgraph with always-unverifiable citations "
          "(GUARD_REDIRECT_MAX=1)")
    # Fake LLM injected exactly like agentic_graph.py does (partial): every
    # turn returns a finished answer (no tool_calls) whose Sources section
    # cites only papers absent from retrieval_keys → guard drops every
    # Sources line on every round.
    class AlwaysBadCiteLLM:
        def bind_tools(self, tools, **kw):
            return self
        def invoke(self, messages, config=None, **kw):
            return AIMessage(
                content=("The ephrin signaling cascade regulates boundary "
                         "formation.\n\nSources:\n- Fake et al. 2031 "
                         "(10.9999/fake.never.indexed)"),
            )

    from functools import partial
    bad_llm = AlwaysBadCiteLLM()

    def build_subgraph():
        b = StateGraph(schemas.AgentState)
        b.add_node("orchestrator",
                   partial(nodes.orchestrator, llm_with_tools=bad_llm))
        b.add_node("tools", ToolNode([]))
        b.add_node("fallback_response",
                   partial(nodes.fallback_response, llm=bad_llm))
        b.add_node("collect_answer", nodes.collect_answer)
        b.add_edge(START, "orchestrator")
        b.add_conditional_edges(
            "orchestrator", edges.route_after_orchestrator_call,
            ["tools", "fallback_response", "collect_answer"])
        b.add_edge("tools", "orchestrator")
        b.add_edge("fallback_response", "collect_answer")
        b.add_conditional_edges(
            "collect_answer", edges.route_after_collect,
            ["orchestrator", END])
        return b.compile(checkpointer=InMemorySaver())

    g = build_subgraph()
    init = {"question": "What regulates boundaries?",
            "question_index": 0,
            "messages": [HumanMessage(content="What regulates boundaries?")],
            "retrieval_keys": set()}
    result = g.invoke(init, config={"configurable": {"thread_id": "t1"}},
                      recursion_limit=25)
    check("subgraph terminated (no GraphRecursionError)", True)
    check("final_answer shipped non-empty",
          bool(result.get("final_answer", "").strip()))
    check("guard_redirect cleared at exit",
          result.get("guard_redirect") is False,
          f"got {result.get('guard_redirect')!r}")
    note = result.get("guard_note", "") or ""
    check("guard_note records the redirect audit trail",
          "redirected" in note.lower())
finally:
    pass  # partial-injected fakes leave no module state to restore

print()
FAILED = [c for c in CHECKS if not c[1]]
if FAILED:
    print(f"✗ {len(FAILED)} check(s) FAILED")
    sys.exit(1)
print("✓ all checks passed")