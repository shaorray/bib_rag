#!/usr/bin/env python3
"""调试重试问题"""

import sys
sys.path.insert(0, '/Disk_2/claw_working_dir/ephrin_agentic_rag')

from rag_core import SimpleEmbedding, DocumentStore

# 加载知识库
doc_store = DocumentStore('ephrin_papers', '/Disk_2/claw_working_dir/ephrin_agentic_rag/chroma_db')
embedder = SimpleEmbedding()

def retriever(query, k=8):
    emb = embedder.embed(query)
    return doc_store.query(emb, n_results=k)

# 测试检索
query = "What is cis-interaction in Eph receptors?"
print(f"Query: {query}")
print("-" * 50)

docs = retriever(query, k=8)
print(f"检索到 {len(docs)} 个文档")

similarities = [d.get('similarity', 0) for d in docs]
print(f"\n相似度列表：{similarities}")
print(f"max_sim: {max(similarities):.3f}")
print(f"avg_sim: {sum(similarities)/len(similarities):.3f}")

# 评级测试
max_sim = max(similarities)
avg_sim = sum(similarities)/len(similarities)

print("\n评级测试:")
if max_sim >= 0.15:
    print(f"  → HIGH (max_sim {max_sim:.3f} >= 0.15)")
else:
    print(f"  → LOW (max_sim {max_sim:.3f} < 0.15)")
