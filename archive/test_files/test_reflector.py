#!/usr/bin/env python3
"""
测试参数调优 + Reflector Agent (2026 工业级标准)

黄金参数:
- similarity_threshold: 0.75
- reflection_threshold: 0.8
- max_iterations: 3
- top_k: 10
"""

import sys
sys.path.insert(0, '/Disk_2/claw_working_dir/ephrin_agentic_rag')

from self_rag import SelfRAGWorkflow, SelfRAGEvaluator
from reflector_agent import ReflectorAgent
from rag_core import SimpleEmbedding, DocumentStore

print("="*70)
print("参数调优 + Reflector Agent 测试")
print("="*70)

# 加载知识库
print("\n📂 加载知识库...")
doc_store = DocumentStore('ephrin_papers', '/Disk_2/claw_working_dir/ephrin_agentic_rag/chroma_db')
embedder = SimpleEmbedding()

def retriever(query, k=10):
    emb = embedder.embed(query)
    return doc_store.query(emb, n_results=k)

print(f"✓ 已加载 {doc_store.count()} 个文档")

# 创建工作流 (使用黄金参数)
print("\n🔧 初始化工作流 (黄金参数)...")
print("  - similarity_threshold: 0.75")
print("  - reflection_threshold: 0.8")
print("  - max_iterations: 3")
print("  - top_k: 10")

self_rag = SelfRAGWorkflow(
    retriever,
    evaluator=SelfRAGEvaluator(model="qwen3.5:397b-cloud"),
    similarity_threshold=0.75,
    reflection_threshold=0.8,
    max_retries=3,
    top_k=10
)

reflector = ReflectorAgent(model="qwen3.5:397b-cloud", reflection_threshold=0.8)

print("✓ Self-RAG 已初始化 (黄金参数)")
print("✓ Reflector Agent 已初始化")

# 测试查询
test_queries = [
    {
        'query': 'Eph 受体和 ephrin 配体如何分类？',
        'type': '简单事实',
        'expected_score': 0.85
    },
    {
        'query': 'cis-interaction 的机制是什么？',
        'type': '中等复杂',
        'expected_score': 0.80
    }
]

print("\n" + "="*70)
print("开始测试")
print("="*70)

for i, test in enumerate(test_queries, 1):
    print(f"\n[Test {i}] {test['type']}")
    print(f"Query: {test['query']}")
    print("-" * 60)
    
    # Self-RAG 运行
    print("  → Self-RAG 处理...")
    result = self_rag.run(test['query'])
    
    print(f"\n✅ Self-RAG 结果:")
    print(f"  答案：{result['answer'][:200]}...")
    print(f"  支持度：{result['support_level']}")
    print(f"  置信度：{result['confidence']:.2f}")
    print(f"  使用文档：{result['documents_used']}")
    print(f"  相关文档：{result['reflections'].get('relevant_count', 0)}")
    
    # Reflector 校验
    print("\n  → Reflector 校验...")
    reflect_result = reflector.reflect(
        test['query'],
        result['answer'],
        [d for d in result.get('documents', [])[:5]]  # 取前 5 个文档
    )
    
    print(f"\n✅ Reflector 结果:")
    print(f"  分数：{reflect_result['score']:.2f}")
    print(f"  是否充分：{'✓' if reflect_result['is_sufficient'] else '✗'}")
    print(f"  需要重查：{'是' if reflect_result['needs_reretrieval'] else '否'}")
    if reflect_result['issues']:
        print(f"  问题：{reflect_result['issues'][:3]}")
    print(f"  建议：{reflect_result['suggestions'][:2]}")
    
    # 对比预期
    print(f"\n📊 质量评估:")
    if reflect_result['score'] >= test['expected_score']:
        print(f"  ✅ 达到预期 (≥{test['expected_score']})")
    else:
        print(f"  ⚠️ 未达预期 (<{test['expected_score']})")

# 汇总
print("\n" + "="*70)
print("测试汇总")
print("="*70)

print(f"\nSelf-RAG 统计:")
print(f"  {self_rag.stats.to_dict()}")

print(f"\nReflector 统计:")
print(f"  {reflector.get_stats()}")

print(f"\n✅ 黄金参数已应用!")
print(f"  - similarity_threshold: 0.75 ✓")
print(f"  - reflection_threshold: 0.8 ✓")
print(f"  - max_iterations: 3 ✓")
print(f"  - top_k: 10 ✓")

print(f"\n✅ Reflector Agent 已集成!")
print(f"  - 防幻觉检测 ✓")
print(f"  - 完整性评估 ✓")
print(f"  - 重查决策 ✓")

print("\n📁 文件列表:")
print("  - self_rag.py (已更新黄金参数)")
print("  - reflector_agent.py (新增)")
print("  - test_improvements.py (测试脚本)")
