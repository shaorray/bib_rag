#!/usr/bin/env python3
"""
测试 Self-RAG + Multi-Hop RAG + RAGAS 评估
"""

import sys
sys.path.insert(0, '/Disk_2/claw_working_dir/ephrin_agentic_rag')

from self_rag import SelfRAGWorkflow
from multi_hop_rag import MultiHopRAG
from ragas_evaluator import SimpleRAGASEvaluator, EvaluationSample
from rag_core import SimpleEmbedding, DocumentStore

# 加载知识库
print("📂 加载知识库...")
doc_store = DocumentStore('ephrin_papers', '/Disk_2/claw_working_dir/ephrin_agentic_rag/chroma_db')
embedder = SimpleEmbedding()

def retriever(query, k=8):
    emb = embedder.embed(query)
    return doc_store.query(emb, n_results=k)

print(f"✓ 已加载 {doc_store.count()} 个文档")

# 创建工作流
print("\n🔧 初始化工作流...")
self_rag = SelfRAGWorkflow(retriever)
multi_hop = MultiHopRAG(retriever)
ragas_eval = SimpleRAGASEvaluator()

# 测试查询
test_queries = [
    {
        'query': 'What is cis-interaction in Eph receptors?',
        'type': '简单事实',
        'use_multi_hop': False
    },
    {
        'query': 'EphA2 与 EphB4 在癌症中的功能差异？',
        'type': '对比类 (多跳)',
        'use_multi_hop': True
    },
    {
        'query': 'cis-interaction 如何影响 trans-signaling 机制？',
        'type': '因果类 (多跳)',
        'use_multi_hop': True
    }
]

print("\n" + "="*70)
print("开始测试 Self-RAG + Multi-Hop")
print("="*70)

results = []

for i, test in enumerate(test_queries, 1):
    print(f"\n[Test {i}] {test['type']}")
    print(f"Query: {test['query']}")
    print(f"Use Multi-Hop: {test['use_multi_hop']}")
    print("-" * 60)
    
    # 选择工作流
    if test['use_multi_hop'] and multi_hop.needs_multi_hop(test['query']):
        print("  → 使用 Multi-Hop RAG")
        result = multi_hop.run(test['query'])
        answer = result['final_answer']
        confidence = result['confidence']
        support_level = 'N/A (Multi-Hop)'
    else:
        print("  → 使用 Self-RAG")
        result = self_rag.run(test['query'])
        answer = result['answer']
        confidence = result['confidence']
        support_level = result['support_level']
    
    # 显示结果
    print(f"\n答案预览：{answer[:300]}...")
    print(f"置信度：{confidence:.3f}")
    print(f"支持度：{support_level}")
    
    # RAGAS 评估
    print("\n📊 RAGAS 评估...")
    sample = EvaluationSample(
        question=test['query'],
        answer=answer,
        contexts=[f"Doc {i}" for i in range(5)]  # 简化
    )
    
    eval_result = ragas_eval.evaluate_sample(sample)
    print(f"  Faithfulness: {eval_result.faithfulness:.3f}")
    print(f"  Relevance: {eval_result.answer_relevance:.3f}")
    print(f"  Precision: {eval_result.context_precision:.3f}")
    
    results.append({
        'query': test['query'],
        'type': test['type'],
        'answer': answer,
        'confidence': confidence,
        'support_level': support_level,
        'ragas': eval_result
    })

# 汇总
print("\n" + "="*70)
print("测试结果汇总")
print("="*70)

print(f"\n总查询数：{len(results)}")
print(f"平均置信度：{sum(r['confidence'] for r in results) / len(results):.3f}")

# RAGAS 平均
avg_faithfulness = sum(r['ragas'].faithfulness for r in results) / len(results)
avg_relevance = sum(r['ragas'].answer_relevance for r in results) / len(results)
avg_precision = sum(r['ragas'].context_precision for r in results) / len(results)

print(f"\nRAGAS 平均:")
print(f"  Faithfulness: {avg_faithfulness:.3f}")
print(f"  Answer Relevance: {avg_relevance:.3f}")
print(f"  Context Precision: {avg_precision:.3f}")

# Self-RAG 统计
print(f"\nSelf-RAG 统计:")
print(f"  {self_rag.stats.to_dict()}")

print("\n✅ 测试完成!")
