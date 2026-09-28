#!/usr/bin/env python3
"""test_empty_refusal.py — Feature 1: hard empty-retrieval refusal.

Pin: when every retrieval in the session returned an empty marker
(NO_RELEVANT_CHUNKS / RETRIEVAL_ERROR) and none produced evidence, the
agent subgraph must refuse INSIDE the subgraph, BEFORE the orchestrator's
first LLM call, and ship a refusal containing the exact phrase
"no literature supporting" (Feature 3's scorer matches on it).

Covers:
  (a) unit: empty_retrieval_refusal routing decision + env kill-switch;
  (b) E2E: full compiled subgraph — refusal path skips the orchestrator
      LLM entirely (call counter stays 0), final answer carries the phrase;
  (c) E2E: productive retrieval → orchestrator IS called;
  (d) E2E: BIB_RAG_EMPTY_REFUSAL=0 → orchestrator called even on empty
      transcript (kill-switch restores legacy behavior);
  (e) phrase contract: coverage_block / gap_instruction emit it.

Run with pytest: uv run python -m pytest tests/test_empty_refusal.py -q
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("HYDE", "0")
os.environ.setdefault("CITATION_GUARD", "1")
os.environ.setdefault("CLAIM_LEDGER", "0")   # keep collect_answer minimal
os.environ.setdefault("EVIDENCE_GATE", "1")

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode

import bib_rag.agent_schemas as schemas
import bib_rag.agent_nodes as nodes
import bib_rag.agent_edges as edges
from src.evidence_gate import evidence_coverage, coverage_block, gap_instruction


class CallCountingLLM:
    """Fake chat model: counts every invoke, returns a static answer.

    Injected exactly like agentic_graph.py does (partial) — same pattern as
    test_guard_redirect.py's AlwaysBadCiteLLM. `bind_tools` returns a
    separate wrapper so ORCHESTRATOR invokes (the only ones Feature 1
    must eliminate) are counted apart from fallback/compress calls.
    """

    def __init__(self, content="The retrieved evidence does not cover this."):
        self.content = content
        self.calls = 0            # bare-llm invokes (fallback_response etc.)
        self.orchestrator_calls = 0

    def bind_tools(self, tools, **kw):
        outer = self

        class _Bound:
            def invoke(self, messages, config=None, **kw2):
                outer.orchestrator_calls += 1
                return AIMessage(content=outer.content)

        return _Bound()

    def with_config(self, *a, **kw):
        return self

    def with_structured_output(self, *a, **kw):
        return self

    def invoke(self, messages, config=None, **kw):
        self.calls += 1
        return AIMessage(content=self.content)


# ---------------------------------------------------------------------------
# Fixtures: the compiled agent subgraph, mirroring agentic_graph.py wiring
# ---------------------------------------------------------------------------

def build_subgraph(llm):
    """Compile the agent subgraph with the Feature 1 passthrough wired in.

    Mirrors create_agent_graph's inner subgraph exactly, but with ToolNode
    replaced by a stub tool node (no server / no KB access in tests).
    """
    from functools import partial

    class _StubToolNode:
        """Records tool_calls on the state (as ToolNode would) without
        executing anything — the transcript manipulations under test are
        pre-seeded, so the tools themselves are never invoked here."""

        def __init__(self):
            self.invocations = 0

        def __call__(self, state, config=None, **kw):
            self.invocations += 1
            # Emulate ToolNode's state merge for any pending tool calls.
            return {"messages": []}

    llm_with_tools = llm.bind_tools([])
    b = StateGraph(schemas.AgentState)
    b.add_node("orchestrator", partial(nodes.orchestrator, llm_with_tools=llm_with_tools))
    b.add_node("tools", _StubToolNode())
    b.add_node("compress_context", partial(nodes.compress_context, llm=llm))
    b.add_node("fallback_response", partial(nodes.fallback_response, llm=llm))
    b.add_node(nodes.should_compress_context)
    b.add_node(nodes.collect_answer)
    b.add_node(nodes.empty_retrieval_refusal)
    # agentic_graph.py wiring (Command-routing constraint): should_compress_context
    # has NO registered edges (its Command gotos are retargeted via
    # ORCH_INJECT_NODE); the passthrough emits Command only — no edges out.
    nodes.ORCH_INJECT_NODE = "empty_retrieval_refusal"
    b.add_edge(START, "empty_retrieval_refusal")
    b.add_edge("compress_context", "empty_retrieval_refusal")
    b.add_conditional_edges(
        "orchestrator", edges.route_after_orchestrator_call,
        {"tools": "tools", "fallback_response": "fallback_response",
         "collect_answer": "collect_answer"})
    b.add_edge("fallback_response", "collect_answer")
    b.add_conditional_edges(
        "collect_answer", edges.route_after_collect,
        {"orchestrator": "orchestrator", "__end__": END})
    return b.compile(checkpointer=InMemorySaver())


def _empty_transcript():
    """Session state after one retrieval that found nothing: the ledger has
    the query, the only tool message is an empty marker."""
    return {
        "question": "What is the role of Eph receptors in boundary formation?",
        "question_index": 0,
        "messages": [
            HumanMessage(content="What is the role of Eph receptors?"),
            AIMessage(content="", tool_calls=[{
                "name": "search_child_chunks",
                "args": {"query": "Eph receptor boundary formation"},
                "id": "call_1",
            }]),
            ToolMessage(content="NO_RELEVANT_CHUNKS", name="search_child_chunks",
                        tool_call_id="call_1"),
        ],
        "retrieval_keys": {"search::Eph receptor boundary formation"},
        "tool_call_count": 1,
        "iteration_count": 1,
        "force_search_done": True,
    }


def _productive_transcript():
    """Session state after one retrieval that DID return evidence."""
    return {
        "question": "What is the role of Eph receptors in boundary formation?",
        "question_index": 0,
        "messages": [
            HumanMessage(content="What is the role of Eph receptors?"),
            AIMessage(content="", tool_calls=[{
                "name": "search_child_chunks",
                "args": {"query": "Eph receptor boundary formation"},
                "id": "call_1",
            }]),
            ToolMessage(
                content=("--- RESULT 1 (similarity: 0.82) ---\nSource: eph_papers/pasquale2002.pdf\n"
                         "Content: Eph receptors regulate cell boundary formation "
                         "through ephrin-B1 repulsion signaling."),
                name="search_child_chunks", tool_call_id="call_1"),
        ],
        "retrieval_keys": {"search::Eph receptor boundary formation"},
        "tool_call_count": 1,
        "iteration_count": 1,
        "force_search_done": True,
    }


def run_subgraph(llm, init):
    return build_subgraph(llm).invoke(
        init, config={"configurable": {"thread_id": "t-empty-refusal"}},
        recursion_limit=25)


# ---------------------------------------------------------------------------
# (a) Unit: the routing function itself
# ---------------------------------------------------------------------------

def test_unit_refusal_routes_to_fallback(monkeypatch):
    llm = CallCountingLLM()
    assert llm.calls == 0
    decision = nodes.empty_retrieval_refusal(_empty_transcript())
    assert decision.goto == "fallback_response"
    assert llm.calls == 0  # zero-LLM gate


def test_unit_productive_routes_to_orchestrator():
    decision = nodes.empty_retrieval_refusal(_productive_transcript())
    assert decision.goto == "orchestrator"


def test_unit_no_retrieval_attempted_routes_to_orchestrator():
    state = _empty_transcript()
    state["retrieval_keys"] = set()   # never searched → cannot refuse yet
    state["messages"] = [HumanMessage(content="q")]
    decision = nodes.empty_retrieval_refusal(state)
    assert decision.goto == "orchestrator"


def test_unit_kill_switch(monkeypatch):
    monkeypatch.setenv("BIB_RAG_EMPTY_REFUSAL", "0")
    decision = nodes.empty_retrieval_refusal(_empty_transcript())
    assert decision.goto == "orchestrator"


def test_unit_gate_failure_falls_back_to_orchestrator(monkeypatch):
    monkeypatch.setenv("BIB_RAG_EMPTY_REFUSAL", "1")
    # Corrupt the state so evidence_coverage raises → gate must never wedge
    # the graph; failure degrades to the normal orchestrator path.
    state = _empty_transcript()
    state["retrieval_keys"] = {"search::ok"}
    state["messages"] = [object()]  # non-message object breaks getattr chain
    decision = nodes.empty_retrieval_refusal(state)
    assert decision.goto == "orchestrator"


# ---------------------------------------------------------------------------
# (a) E2E: empty transcript → refusal, orchestrator LLM never called
# ---------------------------------------------------------------------------

def test_e2e_empty_transcript_refuses_without_orchestrator():
    llm = CallCountingLLM(
        content="No supporting evidence was found for this question.")
    result = run_subgraph(llm, _empty_transcript())
    # The refusal path must bypass the orchestrator ENTIRELY (fallback's own
    # single no-tools synthesis call is allowed and expected).
    assert llm.orchestrator_calls == 0, (
        f"orchestrator called {llm.orchestrator_calls}x on an empty-evidence transcript")
    assert llm.calls == 1  # exactly the fallback synthesis call
    answer = result["final_answer"]
    assert "no literature supporting" in answer
    # The evidence coverage block must ride along.
    assert "Evidence coverage" in answer


def test_e2e_refusal_phrase_comes_from_coverage_block():
    llm = CallCountingLLM(content="placeholder body text")
    result = run_subgraph(llm, _empty_transcript())
    answer = result["final_answer"]
    assert "no literature supporting" in answer
    assert "Evidence coverage" in answer


# ---------------------------------------------------------------------------
# (b) E2E: one real chunk result → orchestrator IS called
# ---------------------------------------------------------------------------

def test_e2e_productive_retrieval_calls_orchestrator():
    llm = CallCountingLLM(
        content=("Eph receptors regulate boundary formation.\n\n---\n**Sources:**\n"
                 "- pasquale2002.pdf"))
    result = run_subgraph(llm, _productive_transcript())
    assert llm.orchestrator_calls >= 1, "productive retrieval must reach the orchestrator LLM"
    assert result["final_answer"]


# ---------------------------------------------------------------------------
# (c) E2E: BIB_RAG_EMPTY_REFUSAL=0 → orchestrator called even when empty
# ---------------------------------------------------------------------------

def test_e2e_kill_switch_calls_orchestrator(monkeypatch):
    monkeypatch.setenv("BIB_RAG_EMPTY_REFUSAL", "0")
    llm = CallCountingLLM(content="best-effort answer from compressed context")
    result = run_subgraph(llm, _empty_transcript())
    assert llm.orchestrator_calls >= 1, "kill-switch must restore the legacy orchestrator path"
    assert result["final_answer"]


# ---------------------------------------------------------------------------
# (e) Phrase contract (Feature 3's scorer depends on the exact phrase)
# ---------------------------------------------------------------------------

def test_phrase_contract_coverage_block_and_gap_instruction():
    msgs = [ToolMessage(content="NO_RELEVANT_CHUNKS",
                        name="search_child_chunks", tool_call_id="c")]
    cov = evidence_coverage(msgs, {"search::ephb1 boundary"})
    assert cov["no_evidence"] is True
    assert "no literature supporting" in coverage_block(cov)
    assert "no literature supporting" in gap_instruction(cov)