#!/usr/bin/env python3
"""
批量查询脚本 - 用于系统测试和批量问答
"""

import json
import sys
from datetime import datetime
from rag_core import SimpleEmbedding, DocumentStore
from agentic_workflow import AgenticRAGWorkflow


def run_batch_queries(queries_file=None, output_file=None, queries=None):
    """运行批量查询"""
    
    print("=" * 70)
    print("🧠 Eph/Ephrin Agentic RAG - 批量查询")
    print("=" * 70)
    
    # 初始化
    print("\n🔧 初始化...")
    doc_store = DocumentStore(
        'ephrin_papers', 
        '/Disk_2/claw_working_dir/ephrin_agentic_rag/chroma_db'
    )
    embedder = SimpleEmbedding()
    
    print(f"✓ 加载了 {doc_store.count()} 个文档块")
    
    # 创建检索器
    def retriever(query, k=8):
        emb = embedder.embed(query)
        return doc_store.query(emb, n_results=k)
    
    # 创建工作流
    workflow = AgenticRAGWorkflow(retriever)
    
    # 获取查询列表
    if queries is None:
        if queries_file:
            with open(queries_file, 'r') as f:
                queries = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        else:
            # 默认查询
            queries = [
                "cis interaction Eph ephrin",
                "reverse signaling mechanism",
                "tetramerization structure",
                "axon guidance EphB ephrinB",
                "cancer metastasis EphA2",
                "ADAM protease shedding",
                "cell segregation boundary",
            ]
    
    print(f"\n🚀 运行 {len(queries)} 个查询...\n")
    
    results = []
    for i, query in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] {query}")
        
        result = workflow.run(query)
        
        # 收集结果
        result_data = {
            'query': query,
            'answer': result['answer'][:500] + '...' if len(result['answer']) > 500 else result['answer'],
            'confidence': result['confidence'],
            'documents_retrieved': len(result['documents']),
            'retries': result.get('retries', 0),
            'rewritten': result.get('rewritten', False),
        }
        results.append(result_data)
        
        print(f"    ✓ Confidence: {result['confidence']:.2f}, Docs: {len(result['documents'])}, Retries: {result.get('retries', 0)}")
    
    # 统计
    avg_confidence = sum(r['confidence'] for r in results) / len(results)
    total_retries = sum(r['retries'] for r in results)
    rewritten_count = sum(1 for r in results if r['rewritten'])
    
    print(f"\n{'='*70}")
    print("📊 批量查询统计")
    print(f"{'='*70}")
    print(f"  • 总查询数: {len(results)}")
    print(f"  • 平均置信度: {avg_confidence:.2f}")
    print(f"  • 总重试次数: {total_retries}")
    print(f"  • 重写查询数: {rewritten_count}")
    
    # 保存结果
    if output_file:
        output = {
            'timestamp': datetime.now().isoformat(),
            'total_queries': len(results),
            'statistics': {
                'avg_confidence': avg_confidence,
                'total_retries': total_retries,
                'rewritten_count': rewritten_count,
            },
            'results': results
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n💾 结果已保存: {output_file}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='批量查询 Agentic RAG')
    parser.add_argument('-i', '--input', help='查询文件 (每行一个查询)')
    parser.add_argument('-o', '--output', help='输出 JSON 文件')
    parser.add_argument('-q', '--queries', nargs='+', help='直接指定查询')
    
    args = parser.parse_args()
    
    queries = args.queries if args.queries else None
    run_batch_queries(args.input, args.output, queries)
