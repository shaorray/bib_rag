#!/usr/bin/env python3
"""
简化查询接口 - 无需交互式界面
快速执行单个查询并输出结果
"""

import sys
from rag_core import SimpleEmbedding, DocumentStore
from agentic_workflow import AgenticRAGWorkflow


def quick_query(question: str, top_k: int = 5):
    """快速执行查询"""
    
    # 初始化
    doc_store = DocumentStore(
        'ephrin_papers',
        '/Disk_2/claw_working_dir/ephrin_agentic_rag/chroma_db'
    )
    embedder = SimpleEmbedding()
    
    # 创建检索器
    def retriever(query, k=8):
        emb = embedder.embed(query)
        return doc_store.query(emb, n_results=k)
    
    # 创建工作流
    workflow = AgenticRAGWorkflow(retriever)
    
    # 执行查询
    result = workflow.run(question)
    
    # 格式化输出
    print(f"\n{'='*70}")
    print(f"❓ Query: {question}")
    print(f"{'='*70}\n")
    
    print(f"📋 Answer:\n{result['answer']}\n")
    
    print(f"{'='*70}")
    print(f"📊 Statistics:")
    print(f"  • Confidence: {result['confidence']:.2f}")
    print(f"  • Documents retrieved: {len(result['documents'])}")
    print(f"  • Query retries: {result.get('retries', 0)}")
    print(f"  • Query rewritten: {result.get('rewritten', False)}")
    print(f"{'='*70}\n")
    
    # 显示来源
    if result['documents']:
        print("📚 Top Sources:")
        for i, doc in enumerate(result['documents'][:top_k], 1):
            title = doc['metadata'].get('paper_title', 'Unknown')
            year = doc['metadata'].get('year', 'N/A')
            sim = doc.get('similarity', 0)
            print(f"  [{i}] {title} ({year}) - relevance: {sim:.3f}")
    
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 quick_query.py 'your question here'")
        print("\nExamples:")
        print("  python3 quick_query.py 'cis interaction mechanism'")
        print("  python3 quick_query.py 'reverse signaling ephrinB'")
        sys.exit(1)
    
    question = ' '.join(sys.argv[1:])
    quick_query(question)
