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

from src.kb_config import get_config

_CFG = get_config()


def agentic_answer(query, graph):
    """Full agentic pipeline. Also returns the retrieval_keys ledger so
    citation_faithfulness can score the answer's sources. Keys are carried
    through agent_answers by collect_answer (subgraph AgentState fields do
    not persist to the main-graph checkpoint)."""
    start = time.time()
    result = graph.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"configurable": {"thread_id": f"eval-{hash(query) % 10000}"}}
    )
    elapsed = time.time() - start
    answer = result["messages"][-1].content
    retrieval_keys = set()
    for a in result.get("agent_answers") or []:
        if isinstance(a, dict) and a.get("retrieval_keys"):
            retrieval_keys.update(a["retrieval_keys"])
    if not retrieval_keys:
        # fallback: try the main-graph checkpoint state
        try:
            snap = graph.get_state({"configurable": {"thread_id": f"eval-{hash(query) % 10000}"}})
            retrieval_keys = set(snap.values.get("retrieval_keys") or set())
        except Exception:
            pass
    return answer, elapsed, retrieval_keys


def baseline_answer(query, llm):
    """Direct ChromaDB query → top results → LLM synthesis."""
    start = time.time()
    
    # Direct ChromaDB query
    client = chromadb.PersistentClient(path=_CFG["chroma_path"])
    coll = client.get_collection(_CFG["collection_name"])
    
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


# ---------------------------------------------------------------------------
# Citation-faithfulness metrics (borrowed: production-rag-assistant citation
# enforcement + rag-eval-harness citation-faithfulness; zero LLM).
# Reuses citation_guard — the SAME code path that guards live answers, so
# eval scores reflect what production actually enforces.
# ---------------------------------------------------------------------------

def citation_faithfulness(answer: str, retrieval_keys: set) -> dict:
    """Score an answer's Sources section against the retrieved parent set.

    Returns {
      'n_source_lines':   sources listed in the answer,
      'n_whitelisted':    sources resolvable to a retrieved parent,
      'whitelist_rate':   fraction of source lines grounded in retrieval,
      'n_dropped':        lines the guard would remove (hallucination risk),
      'n_annotated':      lines with low lexical support,
      'lexical_scores':   per-kept-line best lexical support score,
    }
    """
    from src.citation_guard import (
        parent_ids_from_keys, load_parent_meta_map,
        split_answer_sources, resolve_source_lines, load_parent_text,
        claim_supported_lexically,
    )
    known = parent_ids_from_keys(retrieval_keys)
    parent_meta = load_parent_meta_map(known)
    body, source_lines, header = split_answer_sources(answer)
    if not header or not source_lines:
        return {"n_source_lines": 0, "n_whitelisted": 0, "whitelist_rate": None,
                "n_dropped": 0, "n_annotated": 0, "lexical_scores": []}

    resolved = resolve_source_lines(source_lines, known, parent_meta)
    n_ok = sum(1 for _l, pid in resolved if pid is not None)
    lexical_scores = []
    n_annotated = 0
    body_sentences = [s for s in __import__("re").split(r"(?<=[.!?])\s+", body) if len(s) > 40]
    for line, pid in resolved:
        if pid is None:
            continue
        ptext = load_parent_text(pid)
        if not ptext or not body_sentences:
            continue
        best = 0.0
        for sent in body_sentences[:12]:
            ok, score = claim_supported_lexically(sent, ptext)
            best = max(best, score)
            if ok:
                break
        lexical_scores.append(round(best, 3))
        if best < 0.05:
            n_annotated += 1
    return {
        "n_source_lines": len(source_lines),
        "n_whitelisted": n_ok,
        "whitelist_rate": round(n_ok / len(source_lines), 3) if source_lines else None,
        "n_dropped": len(source_lines) - n_ok,
        "n_annotated": n_annotated,
        "lexical_scores": lexical_scores,
    }


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
        ans_a, time_a, keys_a = agentic_answer(query, graph)
        sources_a = count_sources(ans_a)
        faith = citation_faithfulness(ans_a, keys_a)
        faith_str = ""
        if faith["n_source_lines"]:
            faith_str = (f" | faithfulness: {faith['whitelist_rate']} "
                         f"({faith['n_whitelisted']}/{faith['n_source_lines']} grounded,"
                         f" {faith['n_dropped']} dropped, {faith['n_annotated']} low-support)")

        # Baseline (baseline_answer returns (answer, elapsed))
        ans_b, time_b = baseline_answer(query, llm)
        sources_b = count_sources(ans_b)

        print(f"  Agentic:  {time_a:.1f}s | {len(ans_a)} chars | {sources_a} sources{faith_str}")
        print(f"  Baseline: {time_b:.1f}s | {len(ans_b)} chars | {sources_b} sources")
        print(f"  Speedup:  {time_a/time_b:.1f}x")
        quality_len = f"{len(ans_a)/len(ans_b):.1f}x" if len(ans_b) > 0 else "N/A"
        quality_src = f"{sources_a/sources_b:.1f}x" if sources_b > 0 else f"{sources_a} vs 0"
        print(f"  Quality:  length {quality_len} | sources {quality_src}")
    
    print("\n" + "=" * 70)
    print("✅ Evaluation complete")


if __name__ == "__main__":
    sys.exit(main())
