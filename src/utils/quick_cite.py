#!/usr/bin/env python3
"""
快速引用工具 - 一键为文本添加PMID引用

使用方法:
    python3 quick_cite.py "你的学术论述"
    
示例:
    python3 quick_cite.py "Eph receptors play a crucial role in axon guidance"
"""

import sys
sys.path.insert(0, '/Disk_2/claw_working_dir/ephrin_agentic_rag')
from citation_manager import CitationManager


def quick_cite(text: str, n_citations: int = 2):
    """
    快速为文本添加引用
    
    Args:
        text: 学术论述
        n_citations: 引用数量（默认2个）
    
    Returns:
        带 [PMID:xxxx] 引用的文本
    """
    manager = CitationManager()
    
    # 查找引用
    citations = manager.search(text, n_results=n_citations, min_relevance=0.3)
    
    if not citations:
        print(f"⚠️ 未找到相关文献")
        return text
    
    # 生成引用标记
    pmid_tags = [f"[PMID:{c.pmid}]" for c in citations]
    cited_text = f"{text} {', '.join(pmid_tags)}"
    
    # 输出结果
    print("="*60)
    print("✅ 引用已生成")
    print("="*60)
    print(f"\n{cited_text}\n")
    
    print("📚 参考文献:")
    for i, c in enumerate(citations, 1):
        print(f"  [{i}] PMID:{c.pmid} ({c.year}) - {c.journal}")
        print(f"      相关度: {c.relevance:.3f}")
    
    return cited_text


def batch_cite(sentences: list):
    """
    批量为多个句子添加引用
    
    Args:
        sentences: 句子列表
    
    Returns:
        带引用的段落
    """
    manager = CitationManager()
    
    print("="*60)
    print("📝 批量引用生成")
    print("="*60)
    
    results = []
    all_citations = []
    
    for i, sentence in enumerate(sentences, 1):
        citations = manager.search(sentence, n_results=1, min_relevance=0.3)
        
        if citations:
            c = citations[0]
            cited = f"{sentence} [PMID:{c.pmid}]"
            results.append(cited)
            all_citations.append(c)
            print(f"\n[{i}] ✅ {cited}")
        else:
            results.append(sentence)
            print(f"\n[{i}] ⚠️ {sentence}")
    
    # 输出完整段落
    print("\n" + "="*60)
    print("📄 完整段落:")
    print("="*60)
    paragraph = ' '.join(results)
    print(paragraph)
    
    # 输出参考文献列表
    if all_citations:
        print("\n📚 参考文献:")
        for i, c in enumerate(all_citations, 1):
            print(f"  [{i}] PMID:{c.pmid} ({c.year}) - {c.journal}")
    
    return paragraph


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='快速引用工具')
    parser.add_argument('text', nargs='?', help='需要引用的文本')
    parser.add_argument('--n', type=int, default=2, help='引用数量')
    parser.add_argument('--batch', action='store_true', help='批量模式')
    
    args = parser.parse_args()
    
    if args.batch:
        # 批量模式示例
        sentences = [
            "Eph receptors are receptor tyrosine kinases",
            "They interact with ephrin ligands",
            "Eph-ephrin signaling regulates axon guidance",
        ]
        batch_cite(sentences)
    elif args.text:
        quick_cite(args.text, args.n)
    else:
        # 交互模式
        print("📝 快速引用工具")
        print("输入学术论述，自动添加 PMID 引用")
        print("输入 'quit' 退出\n")
        
        while True:
            text = input("请输入: ").strip()
            if text.lower() in ['quit', 'exit', 'q']:
                break
            if text:
                quick_cite(text, args.n)
                print()
