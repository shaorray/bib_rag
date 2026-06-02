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
from langchain_core.messages import HumanMessage

from src.agentic_graph import create_agent_graph
from src.agent_tools import create_tools


def create_llm():
    """Create LLM client for Qwen3.6-35B via llama-server."""
    llm_url = os.environ.get("LLM_URL", "http://localhost:5015/v1")
    model_name = os.environ.get("LLM_MODEL", "qwen3.6-35b")
    
    return ChatOpenAI(
        base_url=llm_url,
        model=model_name,
        api_key="not-needed",
        temperature=0.1,
        max_tokens=8192,
    )


def run_query(query: str, verbose: bool = False):
    """Run a single query through the agentic graph."""
    llm = create_llm()
    tools = create_tools()
    graph = create_agent_graph(llm, tools)
    
    if verbose:
        print(f"🔍 Query: {query}")
        print("=" * 70)
    
    result = graph.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"configurable": {"thread_id": "cli-thread-1"}}
    )
    
    answer = result["messages"][-1].content
    
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
        result = graph.invoke(
            {"messages": [HumanMessage(content=query)]},
            config={"configurable": {"thread_id": thread_id}}
        )
        
        answer = result["messages"][-1].content
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
