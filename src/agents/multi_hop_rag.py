#!/usr/bin/env python3
"""
Multi-Hop RAG - 简化版 (无需 LangChain)

多跳推理：将复杂查询分解为子问题，分别检索后整合
"""

import os
import json
import requests
from typing import Dict, List, Any, TypedDict


# ==================== 数据结构 ====================

class SubQuery(TypedDict):
    """子查询"""
    id: int
    query: str
    rationale: str


class HopResult(TypedDict):
    """单跳结果"""
    sub_query: SubQuery
    documents: List[Dict]
    answer_snippet: str
    confidence: float


class MultiHopResult(TypedDict):
    """多跳结果"""
    original_query: str
    sub_queries: List[SubQuery]
    hop_results: List[HopResult]
    final_answer: str
    confidence: float
    reasoning_chain: str
    is_consistent: bool


# ==================== LLM 调用 ====================

def call_ollama(prompt: str, model: str = "qwen3.5:397b-cloud", temperature: float = 0.0) -> str:
    """调用 Ollama API (使用 generate 端点)"""
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature}
    }
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["response"].strip()
    except Exception as e:
        print(f"  ⚠️  Ollama 调用失败：{e}")
        return ""


# ==================== Multi-Hop RAG ====================

class MultiHopRAG:
    """多跳推理 RAG"""
    
    def __init__(self, retriever_fn, model: str = "qwen3.5:397b-cloud", max_hops: int = 3):
        self.retriever = retriever_fn
        self.model = model  # 使用 cloud 模型
        self.max_hops = max_hops
    
    def needs_multi_hop(self, query: str) -> bool:
        """判断是否需要多跳推理"""
        triggers = [
            "对比", "差异", "vs", "versus", "compare",
            "和...都", "both", "multiple",
            "如何影响", "how does", "mechanism",
            "为什么", "why", "reason"
        ]
        
        query_lower = query.lower()
        return any(t in query_lower for t in triggers)
    
    def run(self, query: str) -> MultiHopResult:
        """运行 Multi-Hop RAG"""
        print(f"\n🔗 [Multi-Hop] 处理查询：{query}")
        
        # Step 1: 查询分解
        sub_queries = self._decompose_query(query)
        print(f"  分解为 {len(sub_queries)} 个子问题")
        
        # Step 2: 并行检索
        hop_results = self._parallel_hops(sub_queries)
        print(f"  完成 {len(hop_results)} 跳检索")
        
        # Step 3: 答案整合
        final_answer, reasoning_chain, confidence = self._synthesize(query, hop_results)
        print(f"  整合答案，置信度：{confidence:.2f}")
        
        # Step 4: 一致性验证
        is_consistent = self._verify(query, final_answer)
        print(f"  一致性验证：{'✓' if is_consistent else '✗'}")
        
        return MultiHopResult(
            original_query=query,
            sub_queries=sub_queries,
            hop_results=hop_results,
            final_answer=final_answer,
            confidence=confidence,
            reasoning_chain=reasoning_chain,
            is_consistent=is_consistent
        )
    
    def _decompose_query(self, query: str) -> List[SubQuery]:
        """查询分解"""
        prompt = f"""请将以下查询分解为 2-3 个子问题:

**Original Query**: {query}

**输出格式 (JSON)**:
{{
    "sub_queries": ["子问题 1", "子问题 2"],
    "rationale": ["理由 1", "理由 2"]
}}
"""
        response = call_ollama(prompt, self.model)
        
        try:
            # 提取 JSON
            if "{" in response:
                response = response[response.find("{"):response.rfind("}")+1]
            data = json.loads(response)
            
            return [
                SubQuery(id=i+1, query=sq, rationale=data.get("rationale", [""]*len(data["sub_queries"]))[i])
                for i, sq in enumerate(data["sub_queries"][:self.max_hops])
            ]
        except:
            return [SubQuery(id=1, query=query, rationale="回退策略")]
    
    def _parallel_hops(self, sub_queries: List[SubQuery]) -> List[HopResult]:
        """并行检索"""
        hop_results = []
        
        for sq in sub_queries:
            print(f"    [Hop {sq['id']}] {sq['query']}")
            documents = self.retriever(sq['query'], k=5)
            answer = self._generate_hop_answer(sq['query'], documents)
            
            hop_results.append(HopResult(
                sub_query=sq,
                documents=documents,
                answer_snippet=answer,
                confidence=self._calc_hop_confidence(documents)
            ))
        
        return hop_results
    
    def _generate_hop_answer(self, query: str, documents: List[Dict]) -> str:
        """生成单跳答案"""
        if not documents:
            return "未找到相关文档。"
        
        docs_text = "\n\n".join([f"[Doc {i+1}]: {d.get('text', '')[:500]}" for i, d in enumerate(documents)])
        
        prompt = f"""**Query**: {query}

**Documents**:
{docs_text}

**Answer**: """
        
        return call_ollama(prompt, self.model)
    
    def _synthesize(self, query: str, hop_results: List[HopResult]) -> tuple:
        """答案整合"""
        hop_answers = "\n\n".join([
            f"**Sub-Query {hr['sub_query']['id']}: {hr['sub_query']['query']}**\nAnswer: {hr['answer_snippet']}"
            for hr in hop_results
        ])
        
        prompt = f"""**Original Query**: {query}

**Sub-Queries and Answers**:
{hop_answers}

**任务**: 整合所有子答案，形成连贯的最终答案。

**输出格式 (JSON)**:
{{
    "final_answer": "整合后的完整答案",
    "reasoning_chain": "推理过程",
    "confidence": 0.85
}}
"""
        response = call_ollama(prompt, self.model)
        
        try:
            if "{" in response:
                response = response[response.find("{"):response.rfind("}")+1]
            data = json.loads(response)
            return data.get("final_answer", ""), data.get("reasoning_chain", ""), data.get("confidence", 0.5)
        except:
            return "答案整合失败。", "解析错误", 0.3
    
    def _verify(self, query: str, answer: str) -> bool:
        """一致性验证"""
        contradiction = ["however", "contradicts", "矛盾", "但是"]
        return not any(c in answer.lower() for c in contradiction)
    
    def _calc_hop_confidence(self, documents: List[Dict]) -> float:
        """计算单跳置信度"""
        if not documents:
            return 0.1
        doc_count = min(len(documents), 5) / 5.0
        avg_sim = sum(d.get('similarity', 0) for d in documents) / len(documents)
        return min(doc_count * 0.6 + avg_sim * 0.4, 0.9)


# ==================== 使用示例 ====================

if __name__ == "__main__":
    from rag_core import SimpleEmbedding, DocumentStore
    
    doc_store = DocumentStore('ephrin_papers', '/Disk_2/claw_working_dir/ephrin_agentic_rag/chroma_db')
    embedder = SimpleEmbedding()
    
    def retriever(query, k=5):
        emb = embedder.embed(query)
        return doc_store.query(emb, n_results=k)
    
    multi_hop = MultiHopRAG(retriever)
    
    query = "EphA2 与 EphB4 在癌症中的功能差异？"
    print(f"\nQuery: {query}")
    print(f"Needs Multi-Hop: {multi_hop.needs_multi_hop(query)}")
    
    if multi_hop.needs_multi_hop(query):
        result = multi_hop.run(query)
        print(f"\n最终答案：{result['final_answer'][:300]}...")
        print(f"置信度：{result['confidence']:.2f}")
