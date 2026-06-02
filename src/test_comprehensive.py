#!/usr/bin/env python3
"""
Comprehensive test suite for bib_rag agentic graph.

Runs 5 test queries covering different capabilities:
1. Simple factual query
2. Multi-part query (map-reduce)
3. Follow-up query (conversation memory)
4. Complex comparison
5. Domain-specific technical query
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from src.agentic_graph import create_agent_graph
from src.agent_tools import create_tools


def create_llm():
    return ChatOpenAI(
        base_url="http://localhost:5015/v1",
        model="qwen3.6-35b",
        api_key="not-needed",
        temperature=0.1,
        max_tokens=8192,
    )


def run_test(graph, query, test_name, thread_id):
    print(f"\n{'='*70}")
    print(f"📝 TEST: {test_name}")
    print(f"Query: {query}")
    print(f"{'='*70}")
    
    start = time.time()
    try:
        result = graph.invoke(
            {"messages": [HumanMessage(content=query)]},
            config={"configurable": {"thread_id": thread_id}}
        )
        elapsed = time.time() - start
        answer = result["messages"][-1].content
        
        print(f"\n✅ SUCCESS ({elapsed:.1f}s)")
        print(f"Answer length: {len(answer)} chars")
        print(f"\n--- ANSWER ---\n{answer[:800]}...")
        print(f"\n[truncated, full answer: {len(answer)} chars]")
        return True, answer
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n❌ FAILED ({elapsed:.1f}s): {e}")
        return False, str(e)


def main():
    print("🧪 bib_rag Agentic RAG — Comprehensive Test Suite")
    print(f"{'='*70}")
    
    # Setup
    print("\n🔧 Setting up...")
    llm = create_llm()
    tools = create_tools()
    graph = create_agent_graph(llm, tools)
    print("✅ Ready\n")
    
    results = []
    
    # Test 1: Simple factual query
    results.append(run_test(
        graph, 
        "What are ephrins?",
        "1. Simple Factual Query",
        "test-1"
    ))
    
    # Test 2: Multi-part query (map-reduce)
    results.append(run_test(
        graph,
        "What is forward signaling? What is reverse signaling? How do they differ?",
        "2. Multi-Part Query (Map-Reduce)",
        "test-2"
    ))
    
    # Test 3: Follow-up query (conversation memory)
    # First establish context
    print(f"\n{'='*70}")
    print("📝 TEST 3a: Establishing context for follow-up...")
    print(f"{'='*70}")
    graph.invoke(
        {"messages": [HumanMessage(content="What is the role of EphA4 in spinal cord development?")]},
        config={"configurable": {"thread_id": "test-3"}}
    )
    
    results.append(run_test(
        graph,
        "How does it interact with ephrin-B2?",
        "3. Follow-Up Query (Conversation Memory)",
        "test-3"
    ))
    
    # Test 4: Complex comparison
    results.append(run_test(
        graph,
        "Compare the roles of Eph receptors in cancer versus development. What are the key differences?",
        "4. Complex Comparison",
        "test-4"
    ))
    
    # Test 5: Domain-specific technical
    results.append(run_test(
        graph,
        "Explain the mechanism of growth cone collapse via EphA-Ephexin-RhoA signaling pathway",
        "5. Domain-Specific Technical Query",
        "test-5"
    ))
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 TEST SUMMARY")
    print(f"{'='*70}")
    
    passed = sum(1 for ok, _ in results if ok)
    total = len(results)
    
    for i, (ok, _) in enumerate(results, 1):
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  Test {i}: {status}")
    
    print(f"\nTotal: {passed}/{total} passed ({passed*100//total}%)")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
