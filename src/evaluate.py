#!/usr/bin/env python3
"""
Evaluation script for bib_rag agentic vs non-agentic baseline.

Compares:
1. Agentic RAG (full LangGraph pipeline)
2. Direct ChromaDB query (retrieve top-K, format answer with LLM)

Metrics:
- Time to answer
- Source diversity (unique papers cited)
- Answer length / completeness
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from src.agentic_graph import create_agent_graph
from src.agent_tools import create_tools
import chromadb


def agentic_answer(query, graph):
    """Full agentic pipeline."""
    start = time.time()
    result = graph.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"configurable": {"thread_id": f"eval-{hash(query) % 10000}"}}
    )
    elapsed = time.time() - start
    answer = result["messages"][-1].content
    return answer, elapsed


def baseline_answer(query, llm):
    """Direct ChromaDB query → top results → LLM synthesis."""
    start = time.time()
    
    # Direct ChromaDB query
    client = chromadb.PersistentClient(path="chroma_db_new")
    coll = client.get_collection("bib_rag_papers")
    
    # Get embedding from llama-server
    import requests
    resp = requests.post(
        "http://localhost:8081/v1/embeddings",
        json={"input": query, "model": "bge-m3"},
        timeout=30
    )
    emb = resp.json()["data"][0]["embedding"]
    
    # Search ChromaDB
    results = coll.query(query_embeddings=[emb], n_results=10)
    
    # Format context
    context_parts = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        source = meta.get("source", "unknown")
        context_parts.append(f"Source: {source}\n{doc[:500]}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    # LLM synthesis
    prompt = f"""Based on the following research excerpts, answer the question concisely.

Question: {query}

Research excerpts:
{context}

Answer:"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    elapsed = time.time() - start
    
    return response.content, elapsed


def count_sources(answer):
    """Count unique paper sources in answer."""
    import re
    # Find citations like "- Author et al. - YEAR - Title.pdf"
    sources = re.findall(r'- ([^\n]+\.md|[^\n]+\.pdf)', answer)
    return len(set(sources))


def main():
    queries = [
        "What is the role of Eph receptors in neural development?",
        "How does ephrin signaling regulate cell migration?",
        "Compare forward and reverse signaling in Eph-ephrin system",
    ]
    
    llm = ChatOpenAI(
        base_url="http://localhost:5015/v1",
        model="qwen3.6-35b",
        api_key="not-needed",
        temperature=0.1,
        max_tokens=4096,
    )
    
    tools = create_tools()
    graph = create_agent_graph(llm, tools)
    
    print("📊 bib_rag Evaluation: Agentic vs Baseline")
    print("=" * 70)
    
    for query in queries:
        print(f"\n📝 Query: {query}")
        
        # Agentic
        ans_a, time_a = agentic_answer(query, graph)
        sources_a = count_sources(ans_a)
        
        # Baseline
        ans_b, time_b = baseline_answer(query, llm)
        sources_b = count_sources(ans_b)
        
        print(f"  Agentic:  {time_a:.1f}s | {len(ans_a)} chars | {sources_a} sources")
        print(f"  Baseline: {time_b:.1f}s | {len(ans_b)} chars | {sources_b} sources")
        print(f"  Speedup:  {time_a/time_b:.1f}x")
        quality_len = f"{len(ans_a)/len(ans_b):.1f}x" if len(ans_b) > 0 else "N/A"
        quality_src = f"{sources_a/sources_b:.1f}x" if sources_b > 0 else f"{sources_a} vs 0"
        print(f"  Quality:  length {quality_len} | sources {quality_src}")
    
    print("\n" + "=" * 70)
    print("✅ Evaluation complete")


if __name__ == "__main__":
    sys.exit(main())
