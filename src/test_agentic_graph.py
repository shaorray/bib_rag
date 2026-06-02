#!/usr/bin/env python3
"""
Test script for the bib_rag agentic graph.

Creates the graph, runs a test query about Eph receptors in neural development,
and prints the final answer.

Usage:
    cd /Disk_bot/Eph/bib_rag
    python src/test_agentic_graph.py
"""

import sys
import os

# Ensure we can import from src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage

    # --- LLM setup: Qwen3.6-35B via llama-server on port 5015 ---
    llm_url = os.environ.get("LLM_URL", "http://localhost:5015/v1")
    model_name = os.environ.get("LLM_MODEL", "qwen3.6-35b")
    print(f"🔗 LLM URL: {llm_url}")
    print(f"🤖 Model: {model_name}")

    try:
        llm = ChatOpenAI(
            base_url=llm_url,
            model=model_name,
            api_key="not-needed",  # llama-server doesn't require auth
            temperature=0.1,
            max_tokens=8192,
        )
        # Quick health check
        print("⏳ Testing LLM connection...")
        health = llm.invoke("Say 'hello' in 3 words.", max_tokens=20)
        print(f"✅ LLM OK: {health.content}")
    except Exception as e:
        print(f"❌ LLM connection failed: {e}")
        print("   Make sure llama-server is running on port 5015 with Qwen3.6-35B loaded.")
        print("   Exiting.")
        return 1

    # --- Tools setup ---
    from src.agent_tools import create_tools

    print("\n🔧 Loading tools...")
    tools = create_tools()
    print(f"✅ Loaded {len(tools)} tools:")
    for t in tools:
        print(f"   - {t.name}: {t.description[:80]}...")

    # --- Build graph ---
    from src.agentic_graph import create_agent_graph

    print("\n🏗️  Building agentic graph...")
    graph = create_agent_graph(llm, tools)
    print("✅ Graph built.\n")

    # --- Run test query ---
    test_query = "What is the role of Eph receptors in neural development?"
    print(f"📝 Test query: \"{test_query}\"\n")
    print("=" * 70)

    try:
        result = graph.invoke(
            {
                "messages": [HumanMessage(content=test_query)],
            },
            config={"configurable": {"thread_id": "test-thread-1"}}
        )
        final_answer = result["messages"][-1].content
        print("\n" + final_answer)
        print("=" * 70)
        print(f"\n✅ Test complete. Answer length: {len(final_answer)} chars")
        return 0

    except Exception as e:
        print(f"\n❌ Graph execution failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
