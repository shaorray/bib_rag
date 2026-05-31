#!/usr/bin/env python3
"""
知识库分析工具 - 统计、可视化、质量检查
"""

import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path


def analyze_knowledge_base(kb_path="/Disk_2/claw_working_dir/ephrin_agentic_rag"):
    """分析知识库统计信息"""
    
    print("=" * 70)
    print("📊 Eph/Ephrin Knowledge Base Analysis")
    print("=" * 70)
    
    # 加载元数据
    metadata_file = f"{kb_path}/paper_metadata.json"
    if not os.path.exists(metadata_file):
        print(f"❌ 未找到元数据文件: {metadata_file}")
        return
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f"\n📚 论文统计")
    print(f"  总数: {len(metadata)} 篇")
    
    # 年份分布
    years = []
    for paper in metadata.values():
        year_str = paper.get('year', '')
        if year_str and year_str.isdigit():
            years.append(int(year_str))
    
    if years:
        print(f"\n📅 年份分布")
        print(f"  范围: {min(years)} - {max(years)}")
        
        # 按年代分组
        decades = Counter(y // 10 * 10 for y in years)
        for decade in sorted(decades.keys()):
            count = decades[decade]
            bar = "█" * (count // 2)
            print(f"  {decade}s: {bar} ({count})")
    
    # 作者统计
    authors_list = []
    for paper in metadata.values():
        authors = paper.get('authors', '')
        if authors:
            # 提取第一作者
            first_author = authors.split('et al.')[0].strip()
            if first_author:
                authors_list.append(first_author)
    
    if authors_list:
        print(f"\n👥 高频第一作者")
        author_counts = Counter(authors_list)
        for author, count in author_counts.most_common(10):
            print(f"  • {author}: {count} 篇")
    
    # 关键词提取（从标题）
    print(f"\n🔬 高频研究主题（从标题提取）")
    
    all_titles = [p.get('title', '').lower() for p in metadata.values()]
    all_text = ' '.join(all_titles)
    
    keywords = {
        'signaling': '信号传导',
        'receptor': '受体',
        'cis': '顺式作用',
        'trans': '反式作用',
        'axon': '轴突',
        'guidance': '导向',
        'cancer': '癌症',
        'tumor': '肿瘤',
        'development': '发育',
        'embryonic': '胚胎',
        'cell migration': '细胞迁移',
        'boundary': '边界',
        'segregation': '分离',
        'kinase': '激酶',
        'reverse': '反向',
        'forward': '正向',
    }
    
    for keyword, cn_name in keywords.items():
        count = sum(1 for t in all_titles if keyword in t)
        if count > 0:
            bar = "█" * (count // 3)
            print(f"  {cn_name:12s} {bar} ({count})")
    
    # 文档大小
    chroma_file = f"{kb_path}/chroma_db/ephrin_papers.pkl"
    if os.path.exists(chroma_file):
        size_mb = os.path.getsize(chroma_file) / (1024 * 1024)
        print(f"\n💾 存储信息")
        print(f"  向量数据库: {size_mb:.2f} MB")
        print(f"  元数据文件: {os.path.getsize(metadata_file) / 1024:.1f} KB")
    
    print("\n" + "=" * 70)
    print("✅ 分析完成")
    print("=" * 70)


def export_citations(kb_path="/Disk_2/claw_working_dir/ephrin_agentic_rag", 
                     output_file=None):
    """导出引用列表"""
    
    metadata_file = f"{kb_path}/paper_metadata.json"
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    citations = []
    for filename, paper in metadata.items():
        citation = f"{paper.get('authors', 'Unknown')}. ({paper.get('year', 'N/A')}). {paper.get('title', 'Unknown')}."
        citations.append(citation)
    
    # 按年份排序
    citations.sort(key=lambda x: x.split('(')[1][:4] if '(' in x else '0', reverse=True)
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Eph/Ephrin Paper Citations\n\n")
            for i, citation in enumerate(citations, 1):
                f.write(f"{i}. {citation}\n")
        print(f"✅ 导出完成: {output_file}")
    else:
        print("\n".join(citations[:20]))
        print(f"\n... 共 {len(citations)} 篇")


def search_by_author(author_name, kb_path="/Disk_2/claw_working_dir/ephrin_agentic_rag"):
    """按作者搜索论文"""
    
    metadata_file = f"{kb_path}/paper_metadata.json"
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    results = []
    for filename, paper in metadata.items():
        if author_name.lower() in paper.get('authors', '').lower():
            results.append(paper)
    
    print(f"\n🔍 找到 {len(results)} 篇 '{author_name}' 的论文:")
    for paper in results:
        print(f"  • {paper.get('year', 'N/A')}: {paper.get('title', 'Unknown')[:60]}...")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='知识库分析工具')
    parser.add_argument('--export', action='store_true', help='导出引用列表')
    parser.add_argument('--author', type=str, help='按作者搜索')
    parser.add_argument('-o', '--output', type=str, help='输出文件')
    
    args = parser.parse_args()
    
    if args.export:
        export_citations(output_file=args.output or "citations.md")
    elif args.author:
        search_by_author(args.author)
    else:
        analyze_knowledge_base()
