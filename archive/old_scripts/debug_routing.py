#!/usr/bin/env python3
"""调试路由逻辑"""

import sys
sys.path.insert(0, '/Disk_2/claw_working_dir/ephrin_agentic_rag')

from production_workflow import ProductionAgenticRAG, GradeResult
from rag_core import SimpleEmbedding, DocumentStore

# 加载知识库
doc_store = DocumentStore('ephrin_papers', '/Disk_2/claw_working_dir/ephrin_agentic_rag/chroma_db')
embedder = SimpleEmbedding()

def retriever(query, k=8):
    emb = embedder.embed(query)
    return doc_store.query(emb, n_results=k)

# 创建工作流
workflow = ProductionAgenticRAG(retriever, use_cache=False, use_metrics=False)

# 测试查询
query = "What is cis-interaction in Eph receptors?"
print(f"Query: {query}")
print("=" * 60)

# 手动测试评级逻辑
docs = retriever(query, k=8)
similarities = [d.get('similarity', 0) for d in docs]
max_sim = max(similarities)
avg_sim = sum(similarities) / len(similarities)

print(f"\n检索结果:")
print(f"  max_sim: {max_sim:.3f}")
print(f"  avg_sim: {avg_sim:.3f}")

# 评级
if max_sim >= 0.15:
    grade = GradeResult.HIGH.value
else:
    grade = GradeResult.LOW.value

print(f"\n评级：{grade}")
print(f"预期路由：generate (因为 grade=HIGH)")

# 运行工作流
print("\n" + "=" * 60)
print("运行工作流...")
result = workflow.run(query)

print(f"\n结果:")
print(f"  retries: {result['retries']}")
print(f"  confidence: {result['confidence']:.3f}")
print(f"  documents: {len(result['documents'])}")

# 检查路由
complexity = workflow._analyze_complexity(query)
print(f"\n复杂度分析：{complexity}")
print(f"routing_decision: {complexity.value}")
