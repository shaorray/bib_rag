#!/usr/bin/env python3
"""
V3 知识库查询脚本 - 支持元数据过滤
"""

import sys
import pickle
import numpy as np
from pathlib import Path

sys.path.insert(0, "/Disk_2/claw_working_dir/ephrin_agentic_rag")
sys.path.insert(0, "/Disk_2/claw_working_dir/ephrin_agentic_rag/src")
from process_v3_papers import MPNetEmbedding


class V3KnowledgeBase:
    """V3 知识库查询类"""
    
    def __init__(self, kb_path: str = None):
        if kb_path is None:
            kb_path = '/Disk_2/claw_working_dir/ephrin_agentic_rag/chroma_db_v3/ephrin_papers_v3.pkl'
        
        print("📚 加载 V3 知识库...")
        with open(kb_path, 'rb') as f:
            data = pickle.load(f)
        
        self.documents = data['documents']
        self.embeddings = np.array(data['embeddings'])
        self.metadata = data['metadata']
        
        print(f"✓ 已加载 {len(self.documents)} 个文档块 (768维)")
        
        # 加载嵌入模型
        self.embedder = MPNetEmbedding()
    
    def query_with_citations(self,
                                query_text: str,
                                n_results: int = 5,
                                **filters) -> list:
        """
        查询并返回带PMID引用的结果
        
        返回格式:
        {
            'text': '文献内容...',
            'pmid': '27820703',
            'citation': '[PMID:27820703]',
            'year': '2019',
            'journal': 'Journal of Neuroscience',
            'score': 0.8437,
            'full_reference': 'PMID:27820703, Year:2019, Journal:Journal of Neuroscience, IF:5.99'
        }
        """
        results = self.query(query_text, n_results=n_results, **filters)
        
        cited_results = []
        for r in results:
            pmid = r.get('pmid', '')
            year = r.get('year', '')
            journal = r.get('journal', '')
            
            # 生成引用格式
            citation = f"[PMID:{pmid}]" if pmid else ""
            
            # 生成完整引用信息
            ref_parts = []
            if pmid:
                ref_parts.append(f"PMID:{pmid}")
            if year:
                ref_parts.append(f"Year:{year}")
            if journal:
                ref_parts.append(f"Journal:{journal}")
            if r.get('if') and r['if'] != 'N/A':
                ref_parts.append(f"IF:{r['if']}")
            
            full_reference = ", ".join(ref_parts) if ref_parts else ""
            
            cited_results.append({
                **r,
                'citation': citation,
                'full_reference': full_reference,
            })
        
        return cited_results
    
    def generate_paragraph(self, query_text: str, n_results: int = 3, **filters) -> dict:
        """
        生成带PMID引用的学术段落
        
        示例输出:
        {
            'paragraph': 'Eph受体与ephrin的相互作用在神经发育中起关键作用[PMID:27820703][PMID:38649412]。',
            'pmids': ['27820703', '38649412'],
            'references': [
                'PMID:27820703, Year:2019, Journal:Journal of Neuroscience',
                'PMID:38649412, Year:2024, Journal:Nature Microbiology'
            ]
        }
        """
        results = self.query_with_citations(query_text, n_results=n_results, **filters)
        
        if not results:
            return {
                'paragraph': '未找到相关文献。',
                'pmids': [],
                'references': []
            }
        
        # 构建带引用的段落
        sentences = []
        pmids = []
        references = []
        
        for i, r in enumerate(results):
            text = r['text'][:120].strip()
            pmid = r.get('pmid', '')
            
            if pmid and pmid not in pmids:  # 避免重复PMID
                pmids.append(pmid)
                sentences.append(f"{text}[PMID:{pmid}]")
                if r.get('full_reference'):
                    references.append(r['full_reference'])
            else:
                sentences.append(text)
        
        return {
            'paragraph': ' '.join(sentences),
            'pmids': pmids,
            'references': references
        }
    
    def print_cited_results(self, results: list):
        """打印带引用的结果"""
        print(f"\n🔍 找到 {len(results)} 个结果:\n")
        
        for i, r in enumerate(results, 1):
            print(f"[{i}] {r['citation']}")
            print(f"    Score: {r['score']:.4f}")
            print(f"    PMID: {r['pmid']} | Year: {r['year']} | IF: {r['if']}")
            print(f"    Journal: {r['journal']}")
            print(f"    Section: {r['section']}")
            if r.get('full_reference'):
                print(f"    Reference: {r['full_reference']}")
            print(f"    Text: {r['text'][:150]}...")
            print()
    
    def print_paragraph(self, paragraph_data: dict):
        """打印学术段落格式"""
        print("\n" + "="*60)
        print("📝 学术写作格式（带PMID引用）")
        print("="*60)
        print(f"\n{paragraph_data['paragraph']}")
        
        if paragraph_data['references']:
            print(f"\n📚 参考文献:")
            for i, ref in enumerate(paragraph_data['references'], 1):
                print(f"   [{i}] {ref}")
        
        if paragraph_data['pmids']:
            print(f"\n🔗 PMID列表: {', '.join(paragraph_data['pmids'])}")

    def query(self, 
              query_text: str, 
              n_results: int = 10,
              year_min: int = None,
              year_max: int = None,
              journal: str = None,
              min_if: float = None,
              section: str = None,
              tier: str = None) -> list:
        """
        查询知识库，支持元数据过滤
        
        Args:
            query_text: 查询文本
            n_results: 返回结果数
            year_min: 最小年份
            year_max: 最大年份
            journal: 期刊名（部分匹配）
            min_if: 最小影响因子
            section: 章节类型 (abstract/introduction/results/discussion/methods/conclusion)
            tier: 期刊等级
        """
        # 生成查询向量
        query_embedding = self.embedder.embed(query_text)
        
        # 计算相似度
        similarities = np.dot(self.embeddings, query_embedding)
        
        # 获取所有候选结果（先取前50个）
        candidate_indices = np.argsort(similarities)[-50:][::-1]
        
        # 应用元数据过滤
        filtered_results = []
        for idx in candidate_indices:
            meta = self.metadata[idx]
            
            # 年份过滤
            if year_min is not None:
                year = meta.get('year', '')
                if not year or int(year) < year_min:
                    continue
            
            if year_max is not None:
                year = meta.get('year', '')
                if not year or int(year) > year_max:
                    continue
            
            # 期刊过滤
            if journal is not None:
                doc_journal = meta.get('journal', '').lower()
                if journal.lower() not in doc_journal:
                    continue
            
            # IF过滤
            if min_if is not None:
                if_value = meta.get('if', '')
                if not if_value:
                    continue
                try:
                    if float(if_value) < min_if:
                        continue
                except:
                    continue
            
            # 章节过滤
            if section is not None:
                doc_section = meta.get('section', '')
                if section.lower() not in doc_section.lower():
                    continue
            
            # 等级过滤
            if tier is not None:
                doc_tier = meta.get('tier', '')
                if tier.lower() not in doc_tier.lower():
                    continue
            
            # 通过过滤，添加到结果
            text = self.documents[idx]
            # 分离元数据前缀和正文
            content = text.split('\n', 1)[1] if '\n' in text else text
            
            filtered_results.append({
                'text': content,
                'pmid': meta.get('pmid', 'N/A'),
                'year': meta.get('year', 'N/A'),
                'journal': meta.get('journal', 'N/A'),
                'if': meta.get('if', 'N/A'),
                'citations': meta.get('citations', 'N/A'),
                'tier': meta.get('tier', 'N/A'),
                'section': meta.get('section', 'N/A'),
                'score': float(similarities[idx]),
            })
            
            if len(filtered_results) >= n_results:
                break
        
        return filtered_results
    
    def print_results(self, results: list, show_text: bool = True):
        """打印查询结果"""
        print(f"\n🔍 找到 {len(results)} 个结果:\n")
        
        for i, r in enumerate(results, 1):
            print(f"[{i}] Score: {r['score']:.4f}")
            print(f"    PMID: {r['pmid']} | Year: {r['year']} | IF: {r['if']}")
            print(f"    Journal: {r['journal']}")
            print(f"    Section: {r['section']} | Citations: {r['citations']}")
            if show_text:
                print(f"    Text: {r['text'][:200]}...")
            print()


def main():
    """命令行查询示例 - 带PMID引用"""
    import argparse
    
    parser = argparse.ArgumentParser(description='V3 知识库查询 - 带PMID引用')
    parser.add_argument('query', help='查询文本')
    parser.add_argument('-n', '--num', type=int, default=5, help='返回结果数')
    parser.add_argument('--year-min', type=int, help='最小年份')
    parser.add_argument('--year-max', type=int, help='最大年份')
    parser.add_argument('--journal', help='期刊名过滤')
    parser.add_argument('--min-if', type=float, help='最小影响因子')
    parser.add_argument('--section', help='章节类型')
    parser.add_argument('--tier', help='期刊等级')
    parser.add_argument('--paragraph', action='store_true', help='生成学术段落')
    
    args = parser.parse_args()
    
    # 加载知识库
    kb = V3KnowledgeBase()
    
    if args.paragraph:
        # 生成带引用的学术段落
        paragraph_data = kb.generate_paragraph(
            args.query,
            n_results=args.num,
            year_min=args.year_min,
            year_max=args.year_max,
            journal=args.journal,
            min_if=args.min_if,
            section=args.section,
            tier=args.tier
        )
        kb.print_paragraph(paragraph_data)
    else:
        # 查询并显示带引用的结果
        results = kb.query_with_citations(
            args.query,
            n_results=args.num,
            year_min=args.year_min,
            year_max=args.year_max,
            journal=args.journal,
            min_if=args.min_if,
            section=args.section,
            tier=args.tier
        )
        kb.print_cited_results(results)


if __name__ == '__main__':
    main()
