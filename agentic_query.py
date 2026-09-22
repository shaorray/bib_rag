#!/usr/bin/env python3
"""
Agentic Query CLI for bib_rag

Phase 5: Full conversational RAG with hierarchical retrieval.

Usage:
    python3 -B agentic_query.py "What is the role of Eph receptors in neural development?"
    python3 -B agentic_query.py --interactive
    python3 -B agentic_query.py "What is Eph? What is ephrin?" --verbose
"""

import os
import sys
import argparse

# Ensure src/ is on the path
_src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, _src_dir)

# ─── Multi-KB config ─────────────────────────────────────────────────────
# Supports --kb flag and BIB_RAG_KB_NAME env var for switching knowledge bases.
# Must run BEFORE importing agent_tools (which calls get_config() at import time).
from kb_config import parse_kb_arg
_argv = parse_kb_arg()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.errors import GraphInterrupt
from typing import Any, Dict, Tuple
import uuid

from src.agentic_graph import create_agent_graph
from src.agent_tools import create_tools


def create_llm():
    """Create LLM client for agentic RAG.

    Default: fast cloud glm-5.2 (supports function calling + JSON mode, all
    tasks incl. final generation). Override via env vars:
        LLM_URL   (default http://localhost:11434/v1 — OpenAI-compatible gateway)
        LLM_MODEL (default glm-5.2:cloud)
        LLM_API_KEY (default "not-required" — the gateway doesn't validate it)
    NOTE: kimi-k3:cloud is a pure reasoning model — its function calling and
    structured output fail through the OpenAI-compat gateway endpoint. Use
    glm-5.2 or deepseek-v4-flash instead.
    To fall back to local Qwen3.8-27B, set LLM_URL=http://localhost:5015/v1
    and LLM_MODEL=/path/to/model.gguf
    """
    llm_url = os.environ.get("LLM_URL", "http://localhost:11434/v1")
    model_name = os.environ.get("LLM_MODEL", "glm-5.2:cloud")
    api_key = os.environ.get("LLM_API_KEY", "not-required")

    # Bound decode length for slow local models: a 27B GGUF at ~33 t/s takes
    # ~4s per 100 tokens, so a runaway 8192-token answer would be ~5 min.
    # Override via LLM_MAX_TOKENS.
    try:
        max_tokens = int(os.environ.get("LLM_MAX_TOKENS", ""))
    except ValueError:
        max_tokens = 2048 if model_name.endswith(".gguf") else 8192

    # vLLM-served Qwen3.8 AWQ defaults to thinking mode, but this abliterated
    # merge emits its thinking as bare tokens (no </think> tags), so the qwen3
    # reasoning parser matches nothing: reasoning_content AND content both come
    # back empty while the model burns the whole max_tokens budget in invisible
    # deliberation → finish_reason=length → empty answer → agent fallback.
    # Disabling thinking at the chat-template level makes every call
    # answer-first (measured: 8192 wasted tokens → 1918, finish=stop).
    # Env kill-switch: LLM_ENABLE_THINKING=1 restores thinking mode.
    extra = {}
    if (os.environ.get("LLM_ENABLE_THINKING", "0") != "1"
            and "5015" in llm_url and not model_name.endswith(".gguf")):
        extra = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}

    return ChatOpenAI(
        base_url=llm_url,
        model=model_name,
        api_key=api_key,
        temperature=0.1,
        max_tokens=max_tokens,
        model_kwargs=extra,
    )


def _last_assistant_text(state) -> str:
    """Pull the most recent assistant text out of a graph state, if any.

    Used after a GraphInterrupt to surface the already-generated
    clarification question to the user.
    """
    messages = (state or {}).get("messages", []) if isinstance(state, dict) else []
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            content = msg.content
            return content if isinstance(content, str) else str(content)
    return ""


def invoke_with_interrupt_handling(
    graph,
    initial_input: dict,
    config: dict,
    follow_up_provider=None,
    max_clarifications: int = 3,
):
    """Invoke the graph and gracefully handle GraphInterrupt.

    LangGraph raises GraphInterrupt when the graph reaches a node listed in
    `interrupt_before`. For bib_rag that's `request_clarification`, fired
    when the LLM judges the user query too ambiguous to answer. The graph
    has already produced an AIMessage containing the clarification question;
    we just need to surface it, get a follow-up from the user, update the
    state, and re-invoke.

    Args:
        graph: Compiled LangGraph (with InMemorySaver checkpointer).
        initial_input: Initial state dict (e.g. {"messages": [HumanMessage(...)]}).
        config: LangGraph config (must include thread_id for checkpointing).
        follow_up_provider: Callable returning the user's follow-up string,
            or None to signal "no follow-up possible" (e.g. non-interactive
            CLI will pass a default).
        max_clarifications: Safety cap to prevent infinite clarification loops.

    Returns:
        Final graph state dict from the last successful invoke().

    Raises:
        RuntimeError: If the graph interrupts but no follow-up can be obtained.
    """
    state = initial_input
    clarifications = 0

    while True:
        try:
            return graph.invoke(state, config=config)
        except GraphInterrupt as interrupt_err:
            clarifications += 1
            if clarifications > max_clarifications:
                raise RuntimeError(
                    f"Query required more than {max_clarifications} clarifications; aborting."
                ) from interrupt_err

            # Pull the pending state — LangGraph exposes it on the interrupt
            # payload in recent versions. Fall back to get_state() otherwise.
            pending_state = None
            try:
                payload = getattr(interrupt_err, "args", [None])[0]
                if isinstance(payload, list) and payload:
                    first = payload[0]
                    pending_state = getattr(first, "value", None) or getattr(first, "state", None)
            except Exception:
                pending_state = None

            if pending_state is None:
                try:
                    pending_state = graph.get_state(config)
                except Exception:
                    pending_state = None

            clarification_msg = _last_assistant_text(pending_state)
            if not clarification_msg:
                raise RuntimeError(
                    "Graph interrupted but no clarification message was produced. "
                    "This is a graph-design bug — please report."
                ) from interrupt_err

            if follow_up_provider is None:
                raise RuntimeError(
                    "Query needs clarification. Re-run with --interactive to provide one.\n"
                    f"Clarification requested: {clarification_msg}"
                )

            follow_up = follow_up_provider(clarification_msg)
            if follow_up is None:
                raise RuntimeError(
                    f"Query needs clarification but no follow-up was given.\n"
                    f"Clarification requested: {clarification_msg}"
                )

            # Resume the graph with the user's follow-up appended to messages.
            existing_messages = list(state.get("messages", [])) if isinstance(state, dict) else []
            state = {
                **state,
                "messages": existing_messages + [HumanMessage(content=follow_up)],
            }


_GRAPH_CACHE: Dict[Tuple[str, str], Any] = {}


def _get_cached_graph(llm, tools):
    """Return a cached compiled graph for this (base_url, model) backend.

    Compiling the two-level LangGraph (main graph + agent subgraph) costs
    real time per query; a benchmark loop calling run_query() N times used
    to recompile N times for the same backend. The cache key is the LLM
    endpoint tuple — changing LLM_URL/LLM_MODEL between calls lands in a
    different slot instead of silently reusing a stale client.

    SAFETY: with a cached graph the InMemorySaver persists across calls, so
    every run_query() call MUST use a fresh uuid4 thread_id (see config
    below) — otherwise the agent_answers accumulate channel carries prior
    answers into aggregate_answers' len==1 fast path (stale-answer bug,
    verified in langgraph 1.1.3).
    """
    key = (getattr(llm, "openai_api_base", "") or "", getattr(llm, "model_name", "") or "")
    graph = _GRAPH_CACHE.get(key)
    if graph is None:
        graph = create_agent_graph(llm, tools)
        _GRAPH_CACHE[key] = graph
    return graph


def run_query(query: str, verbose: bool = False):
    """Run a single query through the agentic graph."""
    llm = create_llm()
    tools = create_tools()
    graph = _get_cached_graph(llm, tools)

    if verbose:
        print(f"🔍 Query: {query}")
        print("=" * 70)

    # Cached graph ⇒ persistent InMemorySaver ⇒ fresh thread_id per call is
    # MANDATORY (see _get_cached_graph docstring: fixed id + cached graph =
    # stale-answer bug in the agent_answers accumulate channel).
    config = {"configurable": {"thread_id": f"cli-{uuid.uuid4()}"}}

    try:
        result = invoke_with_interrupt_handling(
            graph,
            {"messages": [HumanMessage(content=query)]},
            config,
            follow_up_provider=None,  # non-interactive: surface error
        )
    except RuntimeError as e:
        # Surface clarification requests as a clean error rather than a traceback.
        print(f"\n⚠️  {e}", file=sys.stderr)
        return f"[Clarification needed] {e}"

    answer = result["messages"][-1].content
    if isinstance(answer, list):
        answer = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in answer
        )

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"✅ Answer ({len(answer)} chars)")

    return answer


def interactive_mode():
    """Run interactive chat loop."""
    print("🤖 bib_rag Agentic RAG — Interactive Mode")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 70)

    llm = create_llm()
    tools = create_tools()
    graph = create_agent_graph(llm, tools)

    thread_id = "interactive-thread"

    def follow_up_provider(clarification_msg: str):
        """Prompt the user for clarification, in-place."""
        print(f"\n🤖 {clarification_msg}")
        try:
            return input("📝 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            return None

    while True:
        try:
            query = input("\n📝 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            print("👋 Goodbye!")
            break

        print("🤖 Thinking...")
        config = {"configurable": {"thread_id": thread_id}}
        try:
            result = invoke_with_interrupt_handling(
                graph,
                {"messages": [HumanMessage(content=query)]},
                config,
                follow_up_provider=follow_up_provider,
            )
        except RuntimeError as e:
            print(f"\n⚠️  {e}")
            continue

        answer = result["messages"][-1].content
        if isinstance(answer, list):
            answer = "".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in answer
            )
        print(f"\n🤖 {answer}")


def main():
    parser = argparse.ArgumentParser(description="Agentic RAG Query for bib_rag")
    parser.add_argument("query", nargs="?", help="Query string")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
    elif args.query:
        answer = run_query(args.query, verbose=args.verbose)
        print(answer)
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
