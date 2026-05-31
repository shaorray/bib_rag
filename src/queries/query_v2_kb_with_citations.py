#!/usr/bin/env python3
"""
V2 知识库查询 - 带PMID引用格式
用于学术写作时的参考文献引用
"""

import sys
import pickle
from pathlib import Path
sys.path.insert(0, '/Disk_2/claw_working_dir/ephrin_agentic_rag')
from process_v2_papers import PaperProcessor


def format_citation(pmid, year, journal, score):
    """格式化引用为学术引用格式"""
    return f"[PMID:{pmid}]"


def query_with_citations(processor, query_text, n_results=5, min_score=0.05):
    """
    查询并返回带PMID引用的结果
    
    Args:
        processor: PaperProcessor实例
        query_text: 查询文本
        n_results: 返回结果数
        min_score: 最低相似度阈值
    
    Returns:
        带引用的结果列表
    """
    results = processor.query(query_text, n_results=n_results)
    
    cited_results = []
    for r in results:
        if r['score'] < min_score:
            continue
            
        meta = r['metadata']
        pmid = meta.get('pmid', '')
        year = meta.get('year', '')
        journal = meta.get('journal', '')
        
        cited_results.append({
            'text': r['text'],
            'pmid': pmid,
            'year': year,
            'journal': journal,
            'score': r['score'],
            'citation': f"[PMID:{pmid}]" if pmid else "",
            'full_citation': f"({pmid}, {year}, {journal})" if all([pmid, year, journal]) else ""
        })
    
    return cited_results


def generate_paragraph_with_citations(query_text, n_results=3):
    """
    生成带引用的段落（用于学术写作）
    
    示例输出:
    "Eph受体与ephrin配体的cis相互作用在神经发育中起关键作用[PMID:27820703][PMID:38649412]。
     研究表明，这种相互作用可以调节受体信号传导的强度和持续时间[PMID:11053419]。"
    """
    processor = PaperProcessor()
    results = query_with_citations(processor, query_text, n_results=n_results)
    
    if not results:
        return "未找到相关文献。"
    
    # 构建带引用的文本
    paragraphs = []
    pmids = []
    
    for i, r in enumerate(results):
        # 提取前100字作为引用内容
        text = r['text'][:100].strip()
        pmid = r['pmid']
        
        if pmid:
            pmids.append(pmid)
            paragraphs.append(f"{text}[PMID:{pmid}]")
        else:
            paragraphs.append(text)
    
    return {
        'paragraph': ' '.join(paragraphs),
        'pmids': pmids,
        'references': [f"PMID:{pmid}" for pmid in pmids]
    }


def get_reference_list(pmids):
    """
    根据PMID列表生成参考文献列表
    
    示例:
    参考文献:
    [1] PMID:27820703 - Year:2019, Journal:Journal of Neuroscience, IF:5.99
    [2] PMID:38649412 - Year:2024, Journal:Nature Microbiology, IF:13.94
    """
    processor = PaperProcessor()
    
    references = []
    for i, pmid in enumerate(pmids, 1):
        # 查询该PMID的详细信息
        results = processor.query(f"PMID:{pmid}", n_results=1)
        if results:
            meta = results[0]['metadata']
            ref_text = f"[{i}] PMID:{pmid}"
            if meta.get('year'):
                ref_text += f", Year:{meta['year']}"
            if meta.get('journal'):
                ref_text += f", Journal:{meta['journal']}"
            if meta.get('if'):
                ref_text += f", IF:{meta['if']}"
            references.append(ref_text)
    
    return references


def main():
    """命令行查询示例"""
    # 修改处理器以正确加载V2知识库
    processor = PaperProcessor()
    processor.doc_store.db_path = Path('/Disk_2/claw_working_dir/ephrin_agentic_rag/chroma_db_v2')
    processor.doc_store.name = 'ephrin_papers_v2'
    processor.doc_store._load()
    
    print(f"📚 V2知识库: {processor.doc_store.count()} 个文档块")
    print()
    
    if processor.doc_store.count() == 0:
        print("⚠️  V2知识库为空，尝试重新加载...")
        # 手动加载V2（键名是'metadatas'不是'metadata'）
        with open('/Disk_2/claw_working_dir/ephrin_agentic_rag/chroma_db_v2/ephrin_papers_v2.pkl', 'rb') as f:
            data = pickle.load(f)
        processor.doc_store.documents = data['documents']
        processor.doc_store.embeddings = data['embeddings']
        # V2使用'metadatas'（复数）
        processor.doc_store.metadata = data['metadatas']
        print(f"✓ 手动加载: {len(processor.doc_store.documents)} 个文档块")
    
    # 示例查询
    queries = [
        "cis interaction mechanism",
        "Eph receptor signaling",
        "ephrin ligand binding",
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"🔍 查询: {query}")
        print(f"{'='*60}")
        
        results = query_with_citations(processor, query, n_results=3)
        
        print(f"\n📄 带引用的结果:")
        for i, r in enumerate(results, 1):
            print(f"\n[{i}] {r['citation']}")
            print(f"    Score: {r['score']:.4f}")
            print(f"    Year: {r['year']} | Journal: {r['journal']}")
            print(f"    Text: {r['text'][:120]}...")
        
        # 生成带引用的段落
        print(f"\n📝 学术写作格式:")
        try:
            paragraph_data = generate_paragraph_with_citations(query, n_results=2)
            if isinstance(paragraph_data, dict):
                print(f"    {paragraph_data.get('paragraph', '生成失败')}")
                print(f"\n    参考文献: {', '.join(paragraph_data.get('references', []))}")
            else:
                print(f"    {paragraph_data}")
        except Exception as e:
            print(f"    生成段落时出错: {e}")


if __name__ == '__main__':
    main()
