#!/usr/bin/env python3
"""
Agentic RAG Workflow - Self-RAG + CRAG Implementation
基于 LangGraph 的 Agentic RAG 实现
"""

import os
from typing import List, Dict, Any, TypedDict, Literal, Annotated
from dataclasses import dataclass
import operator
from enum import Enum

# 尝试导入 LangGraph
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("⚠️  LangGraph 未安装，使用简化实现")


class RAGState(TypedDict):
    """RAG 状态定义"""
    query: str
    rewritten_query: str
    documents: List[Dict]
    document_grades: Dict[str, Any]
    answer: str
    confidence: float
    retries: Annotated[int, operator.add]
    needs_retrieval: bool
    routing_decision: str
    errors: List[str]


class GradeResult(Enum):
    """文档相关性评级"""
    HIGH = "high"
    MEDIUM = "medium" 
    LOW = "low"


class AgenticRAGWorkflow:
    """
    Agentic RAG 工作流
    
    特点：
    1. Self-RAG: 自我评估检索质量
    2. CRAG: 检索失败时自我纠正
    3. 自适应路由: 根据问题复杂度选择策略
    4. 多跳推理: 支持复杂查询
    """
    
    def __init__(self, retriever_fn, llm_client=None, use_langgraph: bool = True):
        self.retriever = retriever_fn
        self.llm = llm_client
        self.use_langgraph = use_langgraph and LANGGRAPH_AVAILABLE
        
        # 初始化工作流
        if self.use_langgraph:
            self.app = self._build_langgraph_workflow()
        else:
            self.app = None
            print("   使用简化工作流实现")
    
    def _build_langgraph_workflow(self):
        """使用 LangGraph 构建工作流"""
        
        # ===== 节点定义 =====
        
        def analyze_query(state: RAGState) -> RAGState:
            """分析查询 - 决定是否需要检索"""
            query = state["query"]
            
            # 启发式判断
            # 简单事实性问题可能不需要检索
            simple_patterns = ["what is", "define", "who is"]
            needs_rag = not any(p in query.lower() for p in simple_patterns)
            
            # 更新状态
            return {
                **state,
                "needs_retrieval": needs_rag,
                "routing_decision": "retrieve" if needs_rag else "direct",
                "retries": 0,
                "errors": []
            }
        
        def retrieve_documents(state: RAGState) -> RAGState:
            """检索文档"""
            query = state.get("rewritten_query", state["query"])
            
            try:
                documents = self.retriever(query, k=8)
                return {**state, "documents": documents}
            except Exception as e:
                return {
                    **state,
                    "documents": [],
                    "errors": state.get("errors", []) + [f"Retrieval error: {str(e)}"]
                }
        
        def grade_documents(state: RAGState) -> RAGState:
            """评估文档相关性 (Self-RAG)"""
            documents = state["documents"]
            query = state["query"]
            
            if not documents:
                return {
                    **state,
                    "document_grades": {"overall": GradeResult.LOW.value, "reason": "No documents retrieved"}
                }
            
            # 计算平均相似度
            similarities = [d.get('similarity', 0) for d in documents]
            avg_sim = sum(similarities) / len(similarities) if similarities else 0
            max_sim = max(similarities) if similarities else 0
            
            # 评级逻辑
            if max_sim >= 0.7 and avg_sim >= 0.5:
                grade = GradeResult.HIGH.value
                reason = f"High relevance (max_sim={max_sim:.3f}, avg_sim={avg_sim:.3f})"
            elif max_sim >= 0.5 or avg_sim >= 0.4:
                grade = GradeResult.MEDIUM.value
                reason = f"Medium relevance (max_sim={max_sim:.3f}, avg_sim={avg_sim:.3f})"
            else:
                grade = GradeResult.LOW.value
                reason = f"Low relevance (max_sim={max_sim:.3f}, avg_sim={avg_sim:.3f})"
            
            return {
                **state,
                "document_grades": {
                    "overall": grade,
                    "reason": reason,
                    "max_similarity": max_sim,
                    "avg_similarity": avg_sim
                }
            }
        
        def rewrite_query(state: RAGState) -> RAGState:
            """重写查询 (CRAG 纠正)"""
            original_query = state["query"]
            
            # 简单的查询扩展策略
            # 实际项目中应使用 LLM 进行查询重写
            
            expanded_keywords = {
                "Eph": ["Eph receptor", "Eph family", "Eph signaling"],
                "ephrin": ["ephrin ligand", "ephrin signaling", "Eph ligand"],
                "cis": ["cis interaction", "cis inhibition", "cis signaling"],
                "trans": ["trans interaction", "trans activation", "trans signaling"],
                "signaling": ["signal transduction", "downstream", "pathway"],
                "axon": ["axon guidance", "growth cone", "neuronal migration"],
                "cancer": ["tumor", "oncogenic", "carcinoma", "metastasis"]
            }
            
            rewritten = original_query
            
            # 添加同义词
            for keyword, synonyms in expanded_keywords.items():
                if keyword.lower() in original_query.lower():
                    for syn in synonyms:
                        if syn.lower() not in rewritten.lower():
                            rewritten += f" OR {syn}"
                            break
                    break  # 只扩展一个关键词
            
            return {
                **state,
                "rewritten_query": rewritten,
                "retries": state["retries"] + 1
            }
        
        def generate_answer(state: RAGState) -> RAGState:
            """生成答案 (RAG)"""
            query = state["query"]
            documents = state["documents"]
            
            if not documents:
                answer = "No relevant documents found to answer this question."
                confidence = 0.1
            else:
                # 构建答案
                answer_parts = []
                sources = []
                
                for i, doc in enumerate(documents[:5], 1):
                    text = doc['text'][:500]
                    source = doc['metadata'].get('paper_title', f'Doc {i}')
                    similarity = doc.get('similarity', 0)
                    
                    answer_parts.append(f"[{i}] {text}...")
                    sources.append(f"[{i}] {source} (relevance: {similarity:.3f})")
                
                answer = f"Based on the retrieved documents:\n\n"
                answer += "\n\n".join(answer_parts)
                answer += f"\n\nSources:\n" + "\n".join(sources)
                
                # 计算置信度
                avg_sim = sum(d.get('similarity', 0) for d in documents) / len(documents)
                confidence = min(avg_sim + 0.2, 0.95)  # 稍微提高一点
            
            # Store documents in answer for consistency
            return {
                **state,
                "answer": answer,
                "confidence": confidence,
                "documents": documents
            }
        
        def direct_answer(state: RAGState) -> RAGState:
            """直接生成答案（无需检索）"""
            query = state["query"]
            
            # 对于简单问题返回通用答案
            if "what is" in query.lower():
                answer = """Eph receptors are a family of receptor tyrosine kinases that, together with their ligands (ephrins), play crucial roles in various biological processes including:
                
- Axon guidance and neuronal development
- Cell migration and tissue boundary formation
- Angiogenesis and vascular development
- Cancer progression and metastasis

There are two classes of Eph receptors (A and B) that bind to two classes of ephrin ligands (A and B)."""
            else:
                answer = f"This is a straightforward question that doesn't require document retrieval.\n\nQuery: {query}"
            
            return {
                **state,
                "answer": answer,
                "confidence": 0.5,
                "documents": []
            }
        
        # ===== 条件路由 =====
        
        def route_after_analysis(state: RAGState) -> Literal["retrieve", "direct"]:
            """分析后路由"""
            return "direct" if state["routing_decision"] == "direct" else "retrieve"
        
        def route_after_grade(state: RAGState) -> Literal["generate", "rewrite"]:
            """评估后路由 - 支持更多迭代次数"""
            grade = state["document_grades"].get("overall", GradeResult.LOW.value)
            retries = state["retries"]
            
            # 如果相关性高，直接生成答案
            if grade == GradeResult.HIGH.value:
                return "generate"
            
            # 如果重试次数不足，尝试重写查询 (max 5 retries for complex queries)
            if grade in [GradeResult.MEDIUM.value, GradeResult.LOW.value] and retries < 5:
                return "rewrite"
            
            # 否则直接生成（用现有文档）
            return "generate"
        
        # ===== 构建工作流图 =====
        
        workflow = StateGraph(RAGState)
        
        # 添加节点
        workflow.add_node("analyze", analyze_query)
        workflow.add_node("retrieve", retrieve_documents)
        workflow.add_node("grade", grade_documents)
        workflow.add_node("rewrite", rewrite_query)
        workflow.add_node("generate", generate_answer)
        workflow.add_node("direct", direct_answer)
        
        # 设置入口
        workflow.set_entry_point("analyze")
        
        # 分析 -> [检索|直接回答]
        workflow.add_conditional_edges(
            "analyze",
            route_after_analysis,
            {"retrieve": "retrieve", "direct": "direct"}
        )
        
        # 检索 -> 评估
        workflow.add_edge("retrieve", "grade")
        
        # 评估 -> [生成|重写]
        workflow.add_conditional_edges(
            "grade",
            route_after_grade,
            {"generate": "generate", "rewrite": "rewrite"}
        )
        
        # 重写 -> 重新检索 (CRAG 循环)
        workflow.add_edge("rewrite", "retrieve")
        
        # 生成 -> 结束
        workflow.add_edge("generate", END)
        workflow.add_edge("direct", END)
        
        return workflow.compile()
    
    def _run_simplified(self, query: str) -> Dict[str, Any]:
        """简化版工作流（无 LangGraph）"""
        
        state = {
            "query": query,
            "rewritten_query": "",
            "documents": [],
            "document_grades": {},
            "answer": "",
            "confidence": 0,
            "retries": 0,
            "needs_retrieval": True,
            "routing_decision": "retrieve",
            "errors": []
        }
        
        # 分析查询
        simple_patterns = ["what is", "define", "who is"]
        if any(p in query.lower() for p in simple_patterns):
            state["needs_retrieval"] = False
            state["routing_decision"] = "direct"
        
        # 如果需要检索
        if state["needs_retrieval"]:
            # 最多重试 5 次 (支持复杂查询的多轮优化)
            while state["retries"] < 5:
                # 检索
                search_query = state["rewritten_query"] if state["rewritten_query"] else query
                state["documents"] = self.retriever(search_query, k=8)
                
                if not state["documents"]:
                    # 尝试重写
                    state["rewritten_query"] = f"{query} Eph ephrin signaling"
                    state["retries"] += 1
                    continue
                
                # 评估质量
                similarities = [d.get('similarity', 0) for d in state["documents"]]
                avg_sim = sum(similarities) / len(similarities)
                max_sim = max(similarities)
                
                if max_sim >= 0.15:  # 降低阈值以适应简单嵌入
                    # 质量足够
                    break
                elif state["retries"] < 1:
                    # 尝试重写
                    state["rewritten_query"] = f"Eph ephrin {query}"
                    state["retries"] += 1
                else:
                    break
        
        # 生成答案
        if state["documents"]:
            docs = state["documents"][:5]
            answer_parts = []
            for i, doc in enumerate(docs, 1):
                text = doc['text'][:500]
                source = doc['metadata'].get('paper_title', f'Doc {i}')
                answer_parts.append(f"[{i}] {text}...\n   Source: {source}")
            
            state["answer"] = "\n\n".join(answer_parts)
            state["confidence"] = avg_sim
        else:
            state["answer"] = "No relevant documents found."
            state["confidence"] = 0.1
        
        return {
            "answer": state["answer"],
            "documents": state["documents"],
            "confidence": state["confidence"],
            "retries": state["retries"],
            "rewritten": state["rewritten_query"] != ""
        }
    
    def run(self, query: str) -> Dict[str, Any]:
        """运行 Agentic RAG 工作流"""
        
        if self.use_langgraph and self.app:
            # 使用 LangGraph
            initial_state = {
                "query": query,
                "rewritten_query": "",
                "documents": [],
                "document_grades": {},
                "answer": "",
                "confidence": 0,
                "retries": 0,
                "needs_retrieval": True,
                "routing_decision": "retrieve",
                "errors": []
            }
            
            result = self.app.invoke(initial_state)
            
            return {
                "answer": result["answer"],
                "documents": result["documents"],
                "confidence": result["confidence"],
                "retries": result["retries"],
                "rewritten": result["rewritten_query"] != "",
                "grades": result["document_grades"],
                "errors": result["errors"]
            }
        else:
            # 使用简化版
            return self._run_simplified(query)


class MultiHopRAG(AgenticRAGWorkflow):
    """
    多跳 RAG 实现
    支持需要多步推理的复杂查询
    """
    
    def __init__(self, retriever_fn, llm_client=None):
        super().__init__(retriever_fn, llm_client)
    
    def decompose_query(self, query: str) -> List[str]:
        """将复杂查询分解为子查询"""
        # 启发式分解
        sub_queries = []
        
        # 检查是否需要分解
        if " and " in query.lower() or " compare " in query.lower():
            parts = query.lower().split(" and ")
            for part in parts:
                sub_queries.append(part.strip())
        
        if " vs " in query.lower() or " versus " in query.lower():
            parts = query.lower().split(" vs ")
            if len(parts) == 2:
                sub_queries.extend([p.strip() for p in parts])
        
        # 如果没有分解成功，使用原查询
        if not sub_queries:
            sub_queries = [query]
        
        return sub_queries
    
    def run_multi_hop(self, query: str) -> Dict[str, Any]:
        """运行多跳 RAG"""
        
        # 分解查询
        sub_queries = self.decompose_query(query)
        
        all_results = []
        for sq in sub_queries:
            result = self.run(sq)
            all_results.append({
                "sub_query": sq,
                "result": result
            })
        
        # 合并结果
        combined_docs = []
        seen_ids = set()
        
        for r in all_results:
            for doc in r["result"]["documents"]:
                if doc.get('id') not in seen_ids:
                    combined_docs.append(doc)
                    seen_ids.add(doc.get('id'))
        
        # 生成综合答案
        combined_answer = f"Multi-hop analysis for: {query}\n\n"
        for i, r in enumerate(all_results, 1):
            combined_answer += f"\n## Sub-query {i}: {r['sub_query']}\n"
            combined_answer += r["result"]["answer"][:500] + "...\n"
        
        # 计算综合置信度
        avg_conf = sum(r["result"]["confidence"] for r in all_results) / len(all_results)
        
        return {
            "answer": combined_answer,
            "documents": combined_docs,
            "confidence": avg_conf,
            "sub_queries": sub_queries,
            "sub_results": all_results
        }


# 测试代码
if __name__ == "__main__":
    print("Testing Agentic RAG Workflow...")
    
    # 模拟检索函数
    def mock_retriever(query: str, k: int = 5) -> List[Dict]:
        return [
            {
                "id": "doc1",
                "text": f"Mock document about {query}",
                "metadata": {"paper_title": "Test Paper"},
                "similarity": 0.8
            }
        ]
    
    # 创建工作流
    workflow = AgenticRAGWorkflow(mock_retriever)
    
    # 测试查询
    result = workflow.run("What is the role of cis-interaction in Eph signaling?")
    
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nConfidence: {result['confidence']:.2f}")
    print(f"Retries: {result.get('retries', 0)}")
