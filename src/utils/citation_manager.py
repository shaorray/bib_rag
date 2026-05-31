#!/usr/bin/env python3
"""
引用管理器 - 核心类
"""

import sys
import pickle
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional

sys.path.insert(0, '/Disk_2/claw_working_dir/ephrin_agentic_rag')
from process_v3_papers import SimpleEmbedding


@dataclass
class Citation:
    """引用对象"""
    pmid: str
    year: str
    journal: str
    if_score: str
    text: str
    relevance: float
    
    def __str__(self):
        return f"PMID:{self.pmid} ({self.year})"


class CitationManager:
    """引用管理器"""
    
    def __init__(self, kb_path: str = None):
        self.embedder = SimpleEmbedding()
        self.data = self._load_kb(kb_path)
        self._cache = {}
    
    def _load_kb(self, kb_path: str = None):
        if kb_path is None:
            kb_path = '/Disk_2/claw_working_dir/ephrin_agentic_rag/chroma_db_v3/ephrin_papers_v3.pkl'
        
        with open(kb_path, 'rb') as f:
            return pickle.load(f)
    
    def search(self, query: str, n_results: int = 5, min_relevance: float = 0.3) -> List[Citation]:
        """搜索文献"""
        # 缓存检查
        cache_key = query.lower().strip()
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        query_embedding = self.embedder.embed(query)
        embeddings = np.array(self.data['embeddings'])
        similarities = np.dot(embeddings, query_embedding)
        indices = np.argsort(similarities)[::-1]
        
        results = []
        seen_pmids = set()
        
        for idx in indices:
            if len(results) >= n_results:
                break
            
            meta = self.data['metadata'][idx]
            score = float(similarities[idx])
            
            if score < min_relevance:
                continue
            
            pmid = meta.get('pmid', '')
            if not pmid or pmid in seen_pmids:
                continue
            
            seen_pmids.add(pmid)
            
            # 提取文本（去掉元数据前缀）
            text = self.data['documents'][idx]
            if '\n' in text:
                text = text.split('\n', 1)[1]
            
            results.append(Citation(
                pmid=pmid,
                year=meta.get('year', ''),
                journal=meta.get('journal', ''),
                if_score=meta.get('if', ''),
                text=text[:400],
                relevance=score
            ))
        
        self._cache[cache_key] = results
        return results
    
    def insert_citations(self, text: str, style: str = "bracket", min_relevance: float = 0.35) -> str:
        """
        自动插入引用
        
        Args:
            text: 学术文本
            style: 引用样式
                - "bracket": [PMID:12345]
                - "number": [1], [2]
                - "author": (Author et al., 2020)
            min_relevance: 最小相关度
        
        Returns:
            带引用的文本
        """
        import re
        
        # 分割句子
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        cited_sentences = []
        citation_list = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 30:
                cited_sentences.append(sentence)
                continue
            
            # 查找引用
            citations = self.search(sentence, n_results=2, min_relevance=min_relevance)
            
            if citations:
                if style == "bracket":
                    tags = [f"[PMID:{c.pmid}]" for c in citations[:2]]
                    cited = f"{sentence} {', '.join(tags)}"
                
                elif style == "number":
                    for c in citations[:2]:
                        citation_list.append(c)
                    nums = [f"[{len(citation_list) - len(citations[:2]) + i + 1}]" 
                            for i in range(len(citations[:2]))]
                    cited = f"{sentence} {''.join(nums)}"
                
                elif style == "author":
                    tags = [f"({c.year})" for c in citations[:2]]
                    cited = f"{sentence} {', '.join(tags)}"
                
                else:
                    cited = sentence
                
                cited_sentences.append(cited)
            else:
                cited_sentences.append(sentence)
        
        result = ' '.join(cited_sentences)
        
        # 添加参考文献列表
        if citation_list and style == "number":
            result += "\n\n参考文献:\n"
            for i, c in enumerate(citation_list, 1):
                result += f"[{i}] PMID:{c.pmid} ({c.year}) - {c.journal}\n"
        
        return result
    
    def verify_claim(self, claim: str) -> Dict:
        """
        验证学术论断
        
        Args:
            claim: 需要验证的论断
        
        Returns:
            验证报告
        """
        citations = self.search(claim, n_results=5, min_relevance=0.3)
        
        if not citations:
            return {
                'status': 'no_evidence',
                'message': '未找到支持文献',
                'citations': [],
                'analysis': {}
            }
        
        # 分析证据质量
        high_if = [c for c in citations if c.if_score and float(c.if_score or 0) > 10]
        recent = [c for c in citations if c.year and int(c.year) >= 2020]
        
        return {
            'status': 'supported',
            'message': f'找到 {len(citations)} 篇支持文献',
            'citations': [
                {
                    'pmid': c.pmid,
                    'year': c.year,
                    'journal': c.journal,
                    'if': c.if_score,
                    'relevance': f"{c.relevance:.3f}"
                }
                for c in citations
            ],
            'analysis': {
                'total': len(citations),
                'high_impact': len(high_if),
                'recent': len(recent),
                'recommendation': '✅ 有充分文献支持' if len(citations) >= 3 else '⚠️ 文献支持有限'
            }
        }
    
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
    
    def generate_paragraph(self, topic: str, sentences: List[str]) -> str:
        """
        生成带引用的段落
        
        Args:
            topic: 主题
            sentences: 句子列表（每句一个论点）
        
        Returns:
            完整段落
        """
        output = [f"## {topic}\n"]
        all_citations = []
        
        for i, sentence in enumerate(sentences, 1):
            # 查找引用
            citations = self.search(sentence, n_results=1, min_relevance=0.3)
            
            if citations:
                # 选择最相关的
                best = citations[0]
                pmid_tag = f"[PMID:{best.pmid}]"
                
                # 构建句子
                cited_sentence = f"{sentence} {pmid_tag}"
                output.append(f"{i}. {cited_sentence}\n")
                
                # 记录引用
                all_citations.append({
                    'index': i,
                    'pmid': best.pmid,
                    'year': best.year,
                    'journal': best.journal,
                    'if': best.if_score,
                    'text': best.text[:200]
                })
            else:
                output.append(f"{i}. {sentence}\n")
        
        # 添加引用详情
        if all_citations:
            output.append("\n### 引用详情\n")
            for cite in all_citations:
                output.append(f"[{cite['index']}] **PMID:{cite['pmid']}** ({cite['year']}) ")
                output.append(f"*{cite['journal']}* (IF:{cite['if']})\n")
                output.append(f"> {cite['text']}\n\n")
        
        return '\n'.join(output)


if __name__ == '__main__':
    # 测试
    print("📝 引用管理器测试")
    print("="*60)
    
    manager = CitationManager()
    
    # 测试搜索
    print("\n【测试1】搜索文献")
    citations = manager.search("Eph receptors in axon guidance", n_results=3)
    for c in citations:
        print(f"  - {c}")
    
    # 测试插入引用
    print("\n【测试2】自动插入引用")
    text = "Eph receptors are receptor tyrosine kinases. They regulate axon guidance."
    result = manager.insert_citations(text, style="bracket")
    print(f"原文: {text}")
    print(f"引用后: {result}")
