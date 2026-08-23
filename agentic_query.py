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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.errors import GraphInterrupt

from src.agentic_graph import create_agent_graph
from src.agent_tools import create_tools


def create_llm():
    """Create LLM client for agentic RAG.

    Default: Ollama cloud glm-5.2 (fast, supports function calling + JSON mode,
    all tasks incl. final generation). Override via env vars:
        LLM_URL   (default http://localhost:11434/v1)
        LLM_MODEL (default glm-5.2:cloud)
        LLM_API_KEY (default "ollama")
    NOTE: kimi-k3:cloud is a pure reasoning model — its function calling and
    structured output fail through Ollama's OpenAI-compat endpoint. Use glm-5.2
    or deepseek-v4-flash instead.
    To fall back to local Qwen3.8-27B, set LLM_URL=http://localhost:5015/v1
    and LLM_MODEL=/Disk_bot/models/huihui_Qwen3.8-27B-abliterated-GGUF/Huihui-Qwen3.8-27B-abliterated-Q5_K_L.gguf
    """
    llm_url = os.environ.get("LLM_URL", "http://localhost:11434/v1")
    model_name = os.environ.get("LLM_MODEL", "glm-5.2:cloud")
    api_key = os.environ.get("LLM_API_KEY", "ollama")

    # Bound decode length for slow local models: a 27B GGUF at ~33 t/s takes
    # ~4s per 100 tokens, so a runaway 8192-token answer would be ~5 min.
    # Override via LLM_MAX_TOKENS.
    try:
        max_tokens = int(os.environ.get("LLM_MAX_TOKENS", ""))
    except ValueError:
        max_tokens = 2048 if model_name.endswith(".gguf") else 8192

    return ChatOpenAI(
        base_url=llm_url,
        model=model_name,
        api_key=api_key,
        temperature=0.1,
        max_tokens=max_tokens,
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


def run_query(query: str, verbose: bool = False):
    """Run a single query through the agentic graph."""
    llm = create_llm()
    tools = create_tools()
    graph = create_agent_graph(llm, tools)

    if verbose:
        print(f"🔍 Query: {query}")
        print("=" * 70)

    config = {"configurable": {"thread_id": "cli-thread-1"}}
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
