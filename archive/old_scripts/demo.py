#!/usr/bin/env python3
"""
演示脚本 - 展示 Agentic RAG 的所有功能
"""

import sys
sys.path.insert(0, '/Disk_2/claw_working_dir/ephrin_agentic_rag')

from rag_core import SimpleEmbedding, DocumentStore
from agentic_workflow import AgenticRAGWorkflow

def demo():
    print("="*70)
    print("🧠 Eph/Ephrin Agentic RAG Demo")
    print("="*70)
    
    # 初始化
    print("\n[1/5] 初始化组件...")
    doc_store = DocumentStore(
        'ephrin_papers',
        '/Disk_2/claw_working_dir/ephrin_agentic_rag/chroma_db'
    )
    embedder = SimpleEmbedding()
    
    print(f"  ✓ 文档库: {doc_store.count()} 个块")
    
    # 创建检索器
    def retriever(query, k=8):
        emb = embedder.embed(query)
        return doc_store.query(emb, n_results=k)
    
    # 创建工作流
    print("  ✓ 创建 Agentic RAG 工作流")
    workflow = AgenticRAGWorkflow(retriever)
    
    # 测试不同场景
    test_cases = [
        ("简单查询", "Eph receptor"),
        ("专业术语", "cis interaction"),
        ("机制问题", "reverse signaling"),
        ("结构问题", "tetramerization"),
    ]
    
    print("\n[2/5] 执行测试查询...")
    for i, (desc, query) in enumerate(test_cases, 1):
        print(f"\n  [{i}] {desc}: \"{query}\"")
        result = workflow.run(query)
        print(f"      → {len(result['documents'])} docs, confidence: {result['confidence']:.2f}, retries: {result['retries']}")
    
    print("\n[3/5] 展示答案生成...")
    result = workflow.run("reverse signaling")
    print(f"\n  Query: reverse signaling")
    print(f"  Answer preview:")
    lines = result['answer'].split('\n')[:5]
    for line in lines:
        if line.strip():
            print(f"    {line[:100]}...")
    
    print("\n[4/5] 来源引用...")
    print("  Top sources:")
    for i, doc in enumerate(result['documents'][:3], 1):
        title = doc['metadata'].get('paper_title', 'Unknown')
        year = doc['metadata'].get('year', 'N/A')
        sim = doc.get('similarity', 0)
        print(f"    [{i}] {title[:50]}... ({year}) - sim: {sim:.3f}")
    
    print("\n[5/5] 系统特性...")
    print("  ✓ Self-RAG: 自动评估检索质量")
    print("  ✓ CRAG: 低质量时自动重写查询")
    print("  ✓ Multi-hop: 支持复杂查询分解")
    print("  ✓ Adaptive: 根据问题选择策略")
    
    print("\n" + "="*70)
    print("✅ Demo 完成！")
    print("="*70)
    print("\n使用方式:")
    print("  python3 quick_query.py 'your question'")
    print("  python3 query_interface.py")
    print("  python3 batch_query.py -q 'q1' 'q2' 'q3'")

if __name__ == "__main__":
    demo()
