#!/usr/bin/env python3
"""
测试生产级 Agentic RAG 工作流 - 真实知识库 + 重试机制验证
"""

import sys
sys.path.insert(0, '/Disk_2/claw_working_dir/ephrin_agentic_rag')

from production_workflow import ProductionAgenticRAG
from rag_core import SimpleEmbedding, DocumentStore

print("="*70)
print("生产级 Agentic RAG 测试 - 真实知识库")
print("="*70)

# 加载真实知识库
print("\n📂 加载知识库...")
doc_store = DocumentStore('ephrin_papers', '/Disk_2/claw_working_dir/ephrin_agentic_rag/chroma_db')
embedder = SimpleEmbedding()

print(f"✓ 已加载 {doc_store.count()} 个文档块")

# 创建检索器
def retriever(query, k=8):
    emb = embedder.embed(query)
    return doc_store.query(emb, n_results=k)

# 创建工作流
print("\n🔧 初始化生产级工作流...")
workflow = ProductionAgenticRAG(
    retriever,
    use_cache=True,
    use_metrics=True
)

# 测试查询
test_queries = [
    {
        'query': 'What is cis-interaction in Eph receptors?',
        'type': '中等复杂',
        'expected_retries': 0
    },
    {
        'query': 'Compare cis and trans signaling mechanisms EphA2 ephrinB2',
        'type': '复杂 (多跳)',
        'expected_retries': 2
    },
    {
        'query': 'Eph receptor ephrin ligand classification types',
        'type': '简单事实',
        'expected_retries': 0
    }
]

print("\n" + "="*70)
print("开始测试")
print("="*70)

results = []
for i, test in enumerate(test_queries, 1):
    print(f"\n[Test {i}] {test['type']}")
    print(f"Query: {test['query']}")
    print("-" * 60)
    
    result = workflow.run(test['query'])
    
    # 显示摘要
    answer_preview = result['answer'][:300].replace('\n', ' ')
    print(f"答案预览：{answer_preview}...")
    print(f"置信度：{result['confidence']:.3f}")
    print(f"重试次数：{result['retries']} (预期：{test['expected_retries']})")
    print(f"文档数：{len(result['documents'])}")
    
    # 显示前 3 个文档
    if result['documents']:
        print("\n检索到的文档:")
        for j, doc in enumerate(result['documents'][:3], 1):
            print(f"  [{j}] {doc['metadata'].get('paper_title', 'Unknown')} "
                  f"(sim: {doc.get('similarity', 0):.3f})")
    
    results.append({
        'query': test['query'],
        'type': test['type'],
        'confidence': result['confidence'],
        'retries': result['retries'],
        'num_docs': len(result['documents'])
    })

# 测试缓存
print("\n" + "="*70)
print("缓存测试")
print("="*70)

print("\n重复查询第一次的测试...")
cached_result = workflow.run(test_queries[0]['query'])
print(f"Cache Hit: {cached_result['cache_hit']} ✓")

# 指标摘要
print("\n" + "="*70)
print("指标摘要")
print("="*70)

import json
metrics = workflow.get_metrics_summary()
print(json.dumps(metrics, indent=2))

# 缓存统计
print(f"\n缓存统计:")
cache_stats = workflow.get_cache_stats()
print(f"  缓存大小：{cache_stats['memory_cache_size']}")
print(f"  使用 Redis: {cache_stats['use_redis']}")

# 成本摘要
print(f"\n成本摘要:")
cost = workflow.get_cost_summary()
print(f"  今日使用：${cost['used_today']:.2f}")
print(f"  预算剩余：${cost['remaining']:.2f}")

# 测试结果汇总
print("\n" + "="*70)
print("测试结果汇总")
print("="*70)

print(f"\n总查询数：{len(results)}")
print(f"平均置信度：{sum(r['confidence'] for r in results) / len(results):.3f}")
print(f"总重试次数：{sum(r['retries'] for r in results)}")
print(f"平均文档数：{sum(r['num_docs'] for r in results) / len(results):.1f}")

print("\n✅ 测试完成!")
