#!/usr/bin/env python3
"""查询 v2 知识库"""

import sys
sys.path.insert(0, '/Disk_2/claw_working_dir/ephrin_agentic_rag')
from process_v2_papers import PaperProcessor

def main():
    processor = PaperProcessor()
    
    print(f"📚 知识库总计: {processor.doc_store.count()} 个文档块")
    print()
    
    # 示例查询
    queries = [
        "cis interaction mechanism",
        "Eph receptor clustering",
        "axon guidance signaling",
        "tumor suppression",
        "forward reverse signaling",
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"🔍 查询: {query}")
        print('='*60)
        
        results = processor.query(query, n_results=3)
        for i, r in enumerate(results, 1):
            meta = r['metadata']
            print(f"\n[{i}] 相关度: {r['score']:.3f}")
            print(f"   PMID: {meta.get('pmid', 'N/A')}")
            print(f"   年份: {meta.get('year', 'N/A')}")
            print(f"   期刊: {meta.get('journal', 'N/A')}")
            print(f"   IF: {meta.get('if', 'N/A')}")
            print(f"   章节: {meta.get('section', 'N/A')}")
            print(f"   文本: {r['text'][:150]}...")

if __name__ == '__main__':
    main()
