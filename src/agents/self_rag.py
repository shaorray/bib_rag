#!/usr/bin/env python3
"""
Self-RAG Implementation - 简化版 (无需 LangChain)

基于论文: "Self-RAG: Learning to Reflect and Self-Improve"

核心功能:
1. Retrieve: 检索候选文档
2. Critique: LLM 评估检索相关性
3. Generate: 基于相关文档生成答案
4. Reflect: 评估答案支持度
"""

import os
import json
import requests
from typing import Dict, List, Any, Literal, TypedDict
from dataclasses import dataclass


# ==================== 数据结构 ====================

class SelfRAGResult(TypedDict):
    """Self-RAG 最终结果"""
    query: str
    answer: str
    support_level: str  # fully_supported / partially_supported / no_support / rejected
    confidence: float
    retrieval_used: bool
    documents_used: int
    reflections: Dict[str, Any]


@dataclass
class SelfRAGStats:
    """Self-RAG 统计信息"""
    total_queries: int = 0
    retrieval_used: int = 0
    relevant_docs: int = 0
    irrelevant_docs: int = 0
    fully_supported: int = 0
    partially_supported: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "total_queries": self.total_queries,
            "retrieval_rate": self.retrieval_used / self.total_queries if self.total_queries > 0 else 0,
            "relevance_rate": self.relevant_docs / (self.relevant_docs + self.irrelevant_docs) if (self.relevant_docs + self.irrelevant_docs) > 0 else 0
        }


# ==================== LLM 调用 (Ollama API) ====================

def call_ollama(prompt: str, model: str = "qwen3.5:397b-cloud", temperature: float = 0.0) -> str:
    """调用 Ollama API (使用 generate 端点)"""
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["response"].strip()
    except Exception as e:
        print(f"  ⚠️  Ollama 调用失败：{e}")
        return ""


# ==================== Self-RAG 核心实现 ====================

class SelfRAGEvaluator:
    """Self-RAG 评估器"""
    
    def __init__(self, model: str = "qwen3.5:397b-cloud"):
        self.model = model  # 使用 cloud 模型
    
    def critique_retrieval(self, query: str, document: str) -> str:
        """评估检索文档相关性"""
        prompt = f"""请评估以下文档是否与查询相关:

**Query**: {query}

**Document**: {document[:1000]}

**评估标准**:
- **Relevant**: 文档直接包含回答问题所需的关键信息
- **Partially Relevant**: 文档包含部分相关信息
- **Irrelevant**: 文档与查询无关

请只输出一个词: Relevant / Partially Relevant / Irrelevant
"""
        response = call_ollama(prompt, self.model)
        
        if "Relevant" in response and "Partially" not in response:
            return "Relevant"
        elif "Partially Relevant" in response:
            return "Partially Relevant"
        else:
            return "Irrelevant"
    
    def reflect_generation(self, query: str, documents: List[str], answer: str) -> str:
        """评估答案支持度"""
        docs_text = "\n\n".join([f"[Doc {i+1}]: {doc[:500]}" for i, doc in enumerate(documents)])
        
        prompt = f"""请评估以下答案是否基于提供的文档:

**Query**: {query}

**Documents**:
{docs_text}

**Generated Answer**:
{answer}

**评估标准**:
- **fully_supported**: 答案的所有关键信息都能在文档中找到支持
- **partially_supported**: 答案的部分信息有文档支持
- **no_support**: 答案的信息在文档中找不到支持
- **rejected**: 答案与文档信息矛盾

请只输出一个词: fully_supported / partially_supported / no_support / rejected
"""
        response = call_ollama(prompt, self.model)
        
        support_levels = ["fully_supported", "partially_supported", "no_support", "rejected"]
        for level in support_levels:
            if level in response.lower():
                return level
        
        return "partially_supported"
    
    def decide_retrieval(self, query: str) -> bool:
        """判断是否需要检索"""
        prompt = f"""请判断以下查询是否需要检索外部信息:

**Query**: {query}

**判断标准**:
- **No Retrieval**: 查询是通用知识、主观问题、或不需要具体事实
- **Retrieval**: 查询需要具体事实、数据、或专业知识

请只输出一个词: No Retrieval / Retrieval
"""
        response = call_ollama(prompt, self.model)
        return "Retrieval" in response


class SelfRAGWorkflow:
    """Self-RAG 完整工作流 (2026 工业级黄金参数)"""
    
    def __init__(self, 
                 retriever_fn, 
                 evaluator: SelfRAGEvaluator = None,
                 similarity_threshold: float = 0.75,  # 黄金参数：0.75
                 reflection_threshold: float = 0.8,  # 黄金参数：0.8
                 max_retries: int = 3,  # 黄金参数：3 轮
                 top_k: int = 10):  # 黄金参数：10
        """
        初始化 Self-RAG 工作流
        
        Args:
            retriever_fn: 检索函数
            evaluator: Self-RAG 评估器
            similarity_threshold: 检索相似度阈值 (0.75 为最佳)
            reflection_threshold: 反思阈值 (0.8 为最佳)
            max_retries: 最大重试次数 (3 为最佳)
            top_k: 检索文档数量 (10 为最佳)
        """
        self.retriever = retriever_fn
        self.evaluator = evaluator or SelfRAGEvaluator()
        self.similarity_threshold = similarity_threshold
        self.reflection_threshold = reflection_threshold
        self.max_retries = max_retries
        self.top_k = top_k
        self.stats = SelfRAGStats()
    
    def run(self, query: str) -> SelfRAGResult:
        """运行 Self-RAG 工作流 (使用黄金参数)"""
        self.stats.total_queries += 1
        
        # Step 1: 判断是否需要检索
        needs_retrieval = self.evaluator.decide_retrieval(query)
        
        if not needs_retrieval:
            return self._direct_answer(query)
        
        # Step 2-6: Self-RAG 循环 (最多 max_retries 轮)
        for retry in range(self.max_retries):
            result = self._self_rag_loop(query, retry)
            
            # 早停机制：支持度达到阈值就停止
            if result["confidence"] >= self.reflection_threshold:
                print(f"  ✓ 早停：置信度 {result['confidence']:.2f} ≥ {self.reflection_threshold}")
                return result
            
            # 支持度足够也可以停止
            if result["support_level"] in ["fully_supported", "partially_supported"]:
                return result
        
        # 达到最大重试次数，返回最后结果
        print(f"  ⚠️  达到最大重试次数：{self.max_retries}")
        return result
    
    def _self_rag_loop(self, query: str, retry: int = 0) -> SelfRAGResult:
        """单次 Self-RAG 循环 (使用黄金参数)"""
        
        # Step 2: 检索文档 (使用 top_k 参数)
        documents = self.retriever(query, k=self.top_k)
        self.stats.retrieval_used += 1
        
        if not documents:
            return SelfRAGResult(
                query=query,
                answer="未找到相关文档。",
                support_level="no_support",
                confidence=0.1,
                retrieval_used=True,
                documents_used=0,
                reflections={"error": "no_documents"}
            )
        
        # Step 3: 评估每个文档的相关性 (Critique) + 相似度阈值过滤
        relevant_docs = []
        
        for doc in documents:
            doc_text = doc.get('text', '')[:1000]
            similarity = doc.get('similarity', 0)
            
            # 黄金参数：相似度阈值 0.75
            if similarity < self.similarity_threshold:
                self.stats.irrelevant_docs += 1
                continue  # 过滤低质量文档
            
            grade = self.evaluator.critique_retrieval(query, doc_text)
            
            if grade == "Relevant":
                relevant_docs.append(doc)
                self.stats.relevant_docs += 1
            elif grade == "Partially Relevant" and len(relevant_docs) < self.top_k:
                relevant_docs.append(doc)
            else:
                self.stats.irrelevant_docs += 1
        
        if not relevant_docs:
            # 如果没有文档达到阈值，尝试降低阈值重试
            if retry < self.max_retries - 1:
                print(f"  ⚠️  无文档达到阈值 {self.similarity_threshold}，重试 {retry + 1}/{self.max_retries}")
            return SelfRAGResult(
                query=query,
                answer="检索到的文档与查询不相关。",
                support_level="no_support",
                confidence=0.2,
                retrieval_used=True,
                documents_used=len(documents),
                reflections={"relevant_count": 0}
            )
        
        # Step 4: 基于相关文档生成答案
        answer = self._generate_answer(query, relevant_docs)
        
        # Step 5: 评估答案支持度 (Reflect)
        support_level = self.evaluator.reflect_generation(
            query,
            [d.get('text', '') for d in relevant_docs],
            answer
        )
        
        # 更新统计
        if support_level == "fully_supported":
            self.stats.fully_supported += 1
        elif support_level == "partially_supported":
            self.stats.partially_supported += 1
        
        # 计算置信度
        confidence = self._calculate_confidence(support_level, len(relevant_docs), len(documents))
        
        return SelfRAGResult(
            query=query,
            answer=answer,
            support_level=support_level,
            confidence=confidence,
            retrieval_used=True,
            documents_used=len(relevant_docs),
            reflections={
                "relevant_count": len(relevant_docs),
                "total_retrieved": len(documents)
            }
        )
    
    def _direct_answer(self, query: str) -> SelfRAGResult:
        """直接回答"""
        prompt = f"请简洁回答以下问题:\n\n{query}"
        answer = call_ollama(prompt, self.evaluator.model)
        
        return SelfRAGResult(
            query=query,
            answer=answer,
            support_level="fully_supported",
            confidence=0.8,
            retrieval_used=False,
            documents_used=0,
            reflections={"direct_answer": True}
        )
    
    def _generate_answer(self, query: str, documents: List[Dict]) -> str:
        """基于文档生成答案"""
        docs_text = "\n\n".join([
            f"[Document {i+1}]:\n{d.get('text', '')[:800]}"
            for i, d in enumerate(documents)
        ])
        
        prompt = f"""基于以下文档回答用户问题。如果文档信息不足，请说明。

**Query**: {query}

**Documents**:
{docs_text}

**Answer**: """
        
        return call_ollama(prompt, self.evaluator.model)
    
    def _calculate_confidence(self, support_level: str, relevant_count: int, total_count: int) -> float:
        """计算置信度"""
        base_confidence = {
            "fully_supported": 0.9,
            "partially_supported": 0.6,
            "no_support": 0.3,
            "rejected": 0.1
        }.get(support_level, 0.5)
        
        doc_ratio = relevant_count / total_count if total_count > 0 else 0
        doc_bonus = min(doc_ratio * 0.1, 0.1)
        
        return min(base_confidence + doc_bonus, 0.95)
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return self.stats.to_dict()


# ==================== 使用示例 ====================

if __name__ == "__main__":
    from rag_core import SimpleEmbedding, DocumentStore
    
    # 加载知识库
    doc_store = DocumentStore('ephrin_papers', '/Disk_2/claw_working_dir/ephrin_agentic_rag/chroma_db')
    embedder = SimpleEmbedding()
    
    def retriever(query, k=8):
        emb = embedder.embed(query)
        return doc_store.query(emb, n_results=k)
    
    # 创建 Self-RAG 工作流
    self_rag = SelfRAGWorkflow(retriever)
    
    # 测试查询
    query = "What is cis-interaction in Eph receptors?"
    print(f"Query: {query}")
    print("=" * 60)
    
    result = self_rag.run(query)
    
    print(f"\nAnswer: {result['answer'][:300]}...")
    print(f"Support Level: {result['support_level']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Documents Used: {result['documents_used']}")
    print(f"\nStats: {self_rag.get_stats()}")
