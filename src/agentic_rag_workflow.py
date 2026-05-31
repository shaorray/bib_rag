#!/usr/bin/env python3
"""
Agentic RAG 完整工作流 - 生产级部署 (2026 工业级标准)

架构: Planner → Retriever → Generator → Reflector → Output
                                      ↓
                              └── 重查 (if <0.8) ──┘

黄金参数:
- similarity_threshold: 0.75
- reflection_threshold: 0.8
- max_iterations: 3
- top_k: 10
- max_sub_queries: 5
"""

import sys
import json
sys.path.insert(0, '/Disk_2/claw_working_dir/ephrin_agentic_rag')

from planner_agent import PlannerAgent
from reflector_agent import ReflectorAgent
from self_rag import SelfRAGWorkflow, SelfRAGEvaluator
from rag_core import SimpleEmbedding, DocumentStore
from datetime import datetime


# ==================== 配置 ====================

GOLDEN_PARAMS = {
    "similarity_threshold": 0.75,
    "reflection_threshold": 0.8,
    "max_iterations": 3,
    "top_k": 10,
    "max_sub_queries": 5,
    "temperature": 0.1,
    "timeout": 120,
    "model": "qwen3.5:397b-cloud"
}


# ==================== Agentic RAG 工作流 ====================

class AgenticRAGWorkflow:
    """
    完整 Agentic RAG 工作流 (生产级)
    
    流程:
    1. Planner: 拆解问题
    2. Retriever: 检索每个子问题
    3. Generator: 整合答案
    4. Reflector: 校验答案
    5. 如果分数<0.8，触发重查
    """
    
    def __init__(self, 
                 retriever_fn,
                 model: str = "qwen3.5:397b-cloud"):
        """
        初始化 Agentic RAG 工作流
        
        Args:
            retriever_fn: 检索函数 (query, k) -> List[Dict]
            model: LLM 模型
        """
        self.model = model
        
        # 初始化 Agent (黄金参数)
        self.planner = PlannerAgent(
            model=model,
            max_sub_queries=GOLDEN_PARAMS["max_sub_queries"],
            max_iterations=GOLDEN_PARAMS["max_iterations"]
        )
        
        self.reflector = ReflectorAgent(
            model=model,
            reflection_threshold=GOLDEN_PARAMS["reflection_threshold"]
        )
        
        self.rag = SelfRAGWorkflow(
            retriever_fn,
            evaluator=SelfRAGEvaluator(model=model),
            similarity_threshold=GOLDEN_PARAMS["similarity_threshold"],
            reflection_threshold=GOLDEN_PARAMS["reflection_threshold"],
            max_retries=GOLDEN_PARAMS["max_iterations"],
            top_k=GOLDEN_PARAMS["top_k"]
        )
        
        print("✅ Agentic RAG 工作流已初始化")
        print(f"   模型：{model}")
        print(f"   similarity_threshold: {GOLDEN_PARAMS['similarity_threshold']}")
        print(f"   reflection_threshold: {GOLDEN_PARAMS['reflection_threshold']}")
        print(f"   max_iterations: {GOLDEN_PARAMS['max_iterations']}")
        print(f"   top_k: {GOLDEN_PARAMS['top_k']}")
    
    def run(self, query: str, verbose: bool = True) -> dict:
        """
        运行完整工作流
        
        Args:
            query: 用户查询
            verbose: 是否输出详细日志
            
        Returns:
            完整结果字典
        """
        start_time = datetime.now()
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Agentic RAG 处理：{query}")
            print(f"{'='*70}")
        
        # Step 1: Planner 拆解
        if verbose:
            print(f"\n[1/4] Planner 拆解问题...")
        
        plan = self.planner.plan(query)
        
        if verbose:
            print(f"   复杂度：{'复杂' if plan['is_complex'] else '简单'}")
            print(f"   子问题数：{len(plan['sub_queries'])}")
            print(f"   检索策略：{plan['search_strategy']}")
            
            if plan['is_complex'] and plan['sub_queries']:
                print(f"   子问题:")
                for i, sq in enumerate(plan['sub_queries'][:3], 1):
                    print(f"     {i}. {sq['query']}")
                if len(plan['sub_queries']) > 3:
                    print(f"     ... 还有 {len(plan['sub_queries'])-3} 个")
        
        # Step 2: 检索 + 生成
        if verbose:
            print(f"\n[2/4] 检索 + 生成答案...")
        
        # 选择检索查询
        if plan['is_complex'] and len(plan['sub_queries']) > 1:
            # 复杂查询：整合所有子问题
            search_query = " ".join([sq['query'] for sq in plan['sub_queries'][:3]])
        else:
            search_query = query
        
        if verbose:
            print(f"   检索查询：{search_query[:100]}...")
        
        rag_result = self.rag.run(search_query)
        answer = rag_result['answer']
        documents = rag_result.get('documents', [])
        
        if verbose:
            print(f"   答案长度：{len(answer)} 字符")
            print(f"   使用文档：{rag_result['documents_used']}")
        
        # Step 3: Reflector 校验
        if verbose:
            print(f"\n[3/4] Reflector 校验答案...")
        
        reflect_result = self.reflector.reflect(
            query, 
            answer, 
            documents[:5] if documents else []
        )
        
        if verbose:
            print(f"   分数：{reflect_result['score']:.2f}")
            print(f"   是否充分：{'✓' if reflect_result['is_sufficient'] else '✗'}")
            print(f"   需要重查：{'是' if reflect_result['needs_reretrieval'] else '否'}")
            
            if reflect_result['issues']:
                print(f"   问题：{reflect_result['issues'][:2]}")
            
            if reflect_result['suggestions']:
                print(f"   建议：{reflect_result['suggestions'][:2]}")
        
        # Step 4: 决策
        if verbose:
            print(f"\n[4/4] 决策...")
        
        if reflect_result['is_sufficient']:
            status = "success"
            if verbose:
                print(f"   ✅ 答案质量合格 (≥{GOLDEN_PARAMS['reflection_threshold']})，输出")
        else:
            status = "needs_reretrieval"
            if verbose:
                print(f"   ⚠️  答案质量不足 (<{GOLDEN_PARAMS['reflection_threshold']})，建议重查")
                print(f"   建议：{reflect_result['suggestions'][:2]}")
        
        # 计算耗时
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        if verbose:
            print(f"\n⏱️  总耗时：{duration:.2f}秒")
        
        # 返回结果
        return {
            "query": query,
            "plan": plan,
            "answer": answer,
            "confidence": rag_result['confidence'],
            "support_level": rag_result['support_level'],
            "reflect_score": reflect_result['score'],
            "is_sufficient": reflect_result['is_sufficient'],
            "status": status,
            "issues": reflect_result['issues'],
            "suggestions": reflect_result['suggestions'],
            "documents_used": rag_result['documents_used'],
            "duration_seconds": duration,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "planner": self.planner.get_stats(),
            "reflector": self.reflector.get_stats(),
            "rag": self.rag.stats.to_dict()
        }


# ==================== 主函数 ====================

def main():
    """主函数 - 演示使用"""
    print("="*70)
    print("Agentic RAG 工作流 - 生产级部署")
    print("="*70)
    
    # 加载知识库
    print("\n📂 加载知识库...")
    doc_store = DocumentStore(
        'ephrin_papers', 
        '/Disk_2/claw_working_dir/ephrin_agentic_rag/chroma_db'
    )
    embedder = SimpleEmbedding()
    
    def retriever(query, k=10):
        emb = embedder.embed(query)
        return doc_store.query(emb, n_results=k)
    
    print(f"✓ 已加载 {doc_store.count()} 个文档")
    
    # 创建工作流
    print("\n🔧 初始化 Agentic RAG 工作流...")
    workflow = AgenticRAGWorkflow(
        retriever,
        model=GOLDEN_PARAMS["model"]
    )
    
    # 测试查询
    test_queries = [
        "Eph 受体和 ephrin 配体如何分类？",
        "EphA2 与 EphB4 在癌症中的功能差异？"
    ]
    
    print("\n" + "="*70)
    print("开始测试")
    print("="*70)
    
    results = []
    for query in test_queries:
        result = workflow.run(query, verbose=True)
        results.append(result)
    
    # 汇总
    print("\n" + "="*70)
    print("测试汇总")
    print("="*70)
    
    print(f"\n总查询数：{len(results)}")
    print(f"成功：{sum(1 for r in results if r['status'] == 'success')}")
    print(f"需要重查：{sum(1 for r in results if r['status'] == 'needs_reretrieval')}")
    
    print(f"\n统计信息:")
    stats = workflow.get_stats()
    print(f"  Planner: {stats['planner']}")
    print(f"  Reflector: {stats['reflector']}")
    print(f"  RAG: {stats['rag']}")
    
    print(f"\n✅ 部署完成！")
    
    return workflow


if __name__ == "__main__":
    workflow = main()
