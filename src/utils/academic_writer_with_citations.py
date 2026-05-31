#!/usr/bin/env python3
"""
学术写作辅助工具 - 带引用生成
自动检索文献并在文本中插入 [PMID:xxxx] 引用标记
"""

import sys
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.insert(0, '/Disk_2/claw_working_dir/ephrin_agentic_rag')
from process_v3_papers import SimpleEmbedding


class AcademicWritingWithCitations:
    """带引用的学术写作辅助"""
    
    def __init__(self):
        self.embedder = SimpleEmbedding()
        self.data = self._load_kb()
        print(f"✓ 知识库加载完成: {len(self.data['documents'])} 块")
    
    def _load_kb(self):
        """加载知识库"""
        data_file = Path('/Disk_2/claw_working_dir/ephrin_agentic_rag/chroma_db_v3/ephrin_papers_v3.pkl')
        with open(data_file, 'rb') as f:
            return pickle.load(f)
    
    def find_evidence(self, claim: str, n_results: int = 5) -> List[Dict]:
        """
        为论断查找文献证据
        
        Args:
            claim: 需要支持的论断
            n_results: 返回结果数量
        
        Returns:
            文献证据列表，包含PMID和相关文本
        """
        query_embedding = self.embedder.embed(claim)
        
        docs = self.data['documents']
        embeddings = np.array(self.data['embeddings'])
        meta_list = self.data['metadata']
        
        # 计算相似度
        similarities = np.dot(embeddings, query_embedding)
        
        # 获取Top K
        indices = np.argsort(similarities)[::-1][:n_results]
        
        results = []
        seen_pmids = set()
        
        for idx in indices:
            meta = meta_list[idx]
            pmid = meta.get('pmid', '')
            
            # 去重：同一PMID只取一次
            if pmid in seen_pmids:
                continue
            seen_pmids.add(pmid)
            
            # 提取文本（去掉元数据前缀）
            text = docs[idx]
            if '\n' in text:
                text = text.split('\n', 1)[1]
            
            results.append({
                'pmid': pmid,
                'year': meta.get('year', ''),
                'journal': meta.get('journal', ''),
                'if': meta.get('if', ''),
                'section': meta.get('section', ''),
                'text': text[:500],  # 前500字符
                'score': float(similarities[idx]),
            })
        
        return results
    
    def generate_with_citations(self, topic: str, key_points: List[str]) -> str:
        """
        生成带引用的学术文本
        
        Args:
            topic: 写作主题
            key_points: 关键论点列表
        
        Returns:
            带 [PMID:xxxx] 引用的Markdown文本
        """
        output = []
        output.append(f"# {topic}\n")
        
        for i, point in enumerate(key_points, 1):
            output.append(f"\n## {i}. {point}\n")
            
            # 查找证据
            evidence = self.find_evidence(point, n_results=3)
            
            if evidence:
                output.append(f"\n**支持文献：**\n")
                for j, ev in enumerate(evidence, 1):
                    output.append(f"\n[{j}] PMID:{ev['pmid']} ({ev['year']}) - {ev['journal']} (IF:{ev['if']})\n")
                    output.append(f"> {ev['text'][:300]}...\n")
                
                # 生成带引用的段落
                pmid_list = [f"[PMID:{ev['pmid']}]" for ev in evidence[:2]]  # 取前2个
                citation_str = ", ".join(pmid_list)
                
                output.append(f"\n**写作建议：**\n")
                output.append(f"将以下引用插入到论述中：{citation_str}\n")
                output.append(f"\n示例句式：\n")
                output.append(f"> {point} {citation_str}\n")
            else:
                output.append(f"\n⚠️ 未找到直接支持的文献，建议进一步检索。\n")
        
        # 生成参考文献列表
        output.append(f"\n---\n")
        output.append(f"## 参考文献列表\n")
        
        all_pmids = set()
        for point in key_points:
            evidence = self.find_evidence(point, n_results=2)
            for ev in evidence:
                all_pmids.add((ev['pmid'], ev['year'], ev['journal']))
        
        for i, (pmid, year, journal) in enumerate(sorted(all_pmids), 1):
            output.append(f"{i}. PMID:{pmid} ({year}) - {journal}\n")
        
        return '\n'.join(output)
    
    def cite_statement(self, statement: str) -> Tuple[str, List[Dict]]:
        """
        为单个陈述句提供引用
        
        Args:
            statement: 学术陈述
        
        Returns:
            (带引用的文本, 引用文献列表)
        """
        evidence = self.find_evidence(statement, n_results=3)
        
        if not evidence:
            return statement, []
        
        # 选择最相关的1-2个引用
        citations = []
        for ev in evidence[:2]:
            if ev['score'] > 0.3:  # 相似度阈值
                citations.append(ev)
        
        if not citations:
            return statement, []
        
        # 生成带引用的文本
        pmid_tags = [f"[PMID:{c['pmid']}]" for c in citations]
        cited_text = f"{statement} {', '.join(pmid_tags)}"
        
        return cited_text, citations
    
    def check_citations(self, text: str) -> Dict:
        """
        检查文本中的引用完整性
        
        Args:
            text: 学术文本
        
        Returns:
            引用分析报告
        """
        import re
        
        # 查找所有 [PMID:xxxx] 标记
        pmid_pattern = r'\[PMID:(\d+)\]'
        found_pmids = re.findall(pmid_pattern, text)
        
        # 查找所有可能的论断（句号结尾的句子）
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # 统计
        total_sentences = len([s for s in sentences if len(s.strip()) > 20])
        cited_sentences = 0
        
        for sentence in sentences:
            if re.search(pmid_pattern, sentence):
                cited_sentences += 1
        
        return {
            'total_sentences': total_sentences,
            'cited_sentences': cited_sentences,
            'unique_pmids': len(set(found_pmids)),
            'pmid_list': list(set(found_pmids)),
            'citation_rate': cited_sentences / total_sentences if total_sentences > 0 else 0,
        }


def demo():
    """演示"""
    writer = AcademicWritingWithCitations()
    
    print("="*60)
    print("📝 学术写作辅助工具 - 演示")
    print("="*60)
    
    # 示例1：为单个陈述提供引用
    print("\n【示例1】为陈述提供引用")
    print("-"*60)
    
    statements = [
        "Eph receptors play a crucial role in axon guidance",
        "Ephrin-B2 forward signaling promotes angiogenesis",
        "Cis-interaction between Eph and ephrin regulates cell migration",
    ]
    
    for stmt in statements:
        print(f"\n陈述: {stmt}")
        cited_text, citations = writer.cite_statement(stmt)
        print(f"引用后: {cited_text}")
        if citations:
            print("文献:")
            for c in citations:
                print(f"  - PMID:{c['pmid']} ({c['year']}) - {c['journal']}")
    
    # 示例2：检查引用完整性
    print("\n\n【示例2】检查引用完整性")
    print("-"*60)
    
    sample_text = """
    Eph receptors are receptor tyrosine kinases that play important roles in nervous system development [PMID:10644995]. 
    They interact with ephrin ligands to regulate axon guidance and cell migration [PMID:10655584].
    Recent studies have shown that Eph-ephrin signaling is also involved in cancer progression.
    """
    
    report = writer.check_citations(sample_text)
    print(f"总句数: {report['total_sentences']}")
    print(f"有引用的句子: {report['cited_sentences']}")
    print(f"引用率: {report['citation_rate']*100:.1f}%")
    print(f"唯一PMID: {report['unique_pmids']}")
    print(f"PMID列表: {report['pmid_list']}")
    
    # 示例3：生成带引用的段落
    print("\n\n【示例3】生成带引用的写作建议")
    print("-"*60)
    
    result = writer.generate_with_citations(
        topic="Eph-Ephrin Signaling in Neural Development",
        key_points=[
            "Eph receptors guide axon pathfinding through repulsive signaling",
            "Ephrin reverse signaling regulates cell adhesion",
            "Cis-interaction modulates trans-signaling sensitivity",
        ]
    )
    
    print(result[:1500])
    print("...")


if __name__ == '__main__':
    demo()
