#!/usr/bin/env python3
"""
混合搜索实现 - 语义搜索 + BM25关键词搜索
"""

import sys
import pickle
import numpy as np
from pathlib import Path
from collections import Counter
import re

sys.path.insert(0, '/Disk_2/claw_working_dir/ephrin_agentic_rag')
from process_v3_papers import MPNetEmbedding


class BM25Index:
    """简单的BM25索引实现"""
    
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.documents = []
        self.term_freqs = []
        self.doc_freqs = Counter()
        self.doc_lengths = []
        self.avg_doc_length = 0
        self.N = 0
    
    def tokenize(self, text: str) -> list:
        """简单分词"""
        # 转为小写，移除标点，分词
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return text.split()
    
    def add_document(self, text: str):
        """添加文档到索引"""
        tokens = self.tokenize(text)
        self.documents.append(text)
        self.doc_lengths.append(len(tokens))
        
        # 计算词频
        tf = Counter(tokens)
        self.term_freqs.append(tf)
        
        # 更新文档频率
        for term in tf:
            self.doc_freqs[term] += 1
        
        self.N += 1
    
    def build(self):
        """构建索引"""
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
    
    def search(self, query: str, top_k: int = 50) -> list:
        """BM25搜索"""
        query_tokens = self.tokenize(query)
        scores = np.zeros(self.N)
        
        for token in query_tokens:
            if token not in self.doc_freqs:
                continue
            
            df = self.doc_freqs[token]
            idf = np.log((self.N - df + 0.5) / (df + 0.5) + 1)
            
            for i in range(self.N):
                tf = self.term_freqs[i].get(token, 0)
                doc_len = self.doc_lengths[i]
                
                # BM25公式
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
                scores[i] += idf * numerator / denominator
        
        # 获取top_k
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]


class HybridSearch:
    """混合搜索：语义搜索 + BM25"""
    
    def __init__(self, kb_path: str = None):
        if kb_path is None:
            kb_path = '/Disk_2/claw_working_dir/ephrin_agentic_rag/chroma_db_v3/ephrin_papers_v3.pkl'
        
        print("📚 加载 V3 知识库...")
        with open(kb_path, 'rb') as f:
            data = pickle.load(f)
        
        self.documents = data['documents']
        self.embeddings = np.array(data['embeddings'])
        self.metadata = data['metadata']
        
        print(f"✓ 已加载 {len(self.documents)} 个文档块")
        
        # 加载嵌入模型
        print("🔢 加载语义模型...")
        self.embedder = MPNetEmbedding()
        
        # 构建BM25索引
        print("📖 构建BM25索引...")
        self.bm25 = BM25Index()
        for doc in self.documents:
            # 只索引正文（去掉元数据前缀）
            content = doc.split('\n', 1)[1] if '\n' in doc else doc
            self.bm25.add_document(content)
        self.bm25.build()
        print(f"✓ BM25索引完成 ({self.bm25.N} 文档)")
    
    def search(self, 
               query: str,
               n_results: int = 10,
               semantic_weight: float = 0.6,
               bm25_weight: float = 0.4,
               year_min: int = None,
               min_if: float = None) -> list:
        """
        混合搜索
        
        Args:
            query: 查询文本
            n_results: 返回结果数
            semantic_weight: 语义搜索权重 (0-1)
            bm25_weight: BM25权重 (0-1)
            year_min: 最小年份
            min_if: 最小影响因子
        """
        # 1. 语义搜索
        query_embedding = self.embedder.embed(query)
        semantic_scores = np.dot(self.embeddings, query_embedding)
        
        # 归一化语义分数到0-1
        sem_min, sem_max = semantic_scores.min(), semantic_scores.max()
        if sem_max > sem_min:
            semantic_scores = (semantic_scores - sem_min) / (sem_max - sem_min)
        
        # 2. BM25搜索
        bm25_results = self.bm25.search(query, top_k=100)
        bm25_scores = np.zeros(len(self.documents))
        for idx, score in bm25_results:
            bm25_scores[idx] = score
        
        # 归一化BM25分数到0-1
        bm25_max = bm25_scores.max()
        if bm25_max > 0:
            bm25_scores = bm25_scores / bm25_max
        
        # 3. 混合分数
        combined_scores = semantic_weight * semantic_scores + bm25_weight * bm25_scores
        
        # 4. 应用过滤并获取结果
        candidate_indices = np.argsort(combined_scores)[-100:][::-1]
        
        results = []
        for idx in candidate_indices:
            meta = self.metadata[idx]
            
            # 年份过滤
            if year_min is not None:
                year = meta.get('year', '')
                if not year or int(year) < year_min:
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
            
            # 添加结果
            text = self.documents[idx]
            content = text.split('\n', 1)[1] if '\n' in text else text
            
            results.append({
                'text': content,
                'pmid': meta.get('pmid', 'N/A'),
                'year': meta.get('year', 'N/A'),
                'journal': meta.get('journal', 'N/A'),
                'if': meta.get('if', 'N/A'),
                'citations': meta.get('citations', 'N/A'),
                'tier': meta.get('tier', 'N/A'),
                'section': meta.get('section', 'N/A'),
                'semantic_score': float(semantic_scores[idx]),
                'bm25_score': float(bm25_scores[idx]),
                'combined_score': float(combined_scores[idx]),
            })
            
            if len(results) >= n_results:
                break
        
        return results
    
    def print_results(self, results: list):
        """打印结果"""
        print(f"\n🔍 找到 {len(results)} 个结果 (语义+BM25混合):\n")
        
        for i, r in enumerate(results, 1):
            print(f"[{i}] Combined: {r['combined_score']:.4f} | "
                  f"Semantic: {r['semantic_score']:.4f} | "
                  f"BM25: {r['bm25_score']:.4f}")
            print(f"    PMID: {r['pmid']} | Year: {r['year']} | IF: {r['if']}")
            print(f"    Journal: {r['journal']} | Section: {r['section']}")
            print(f"    Text: {r['text'][:150]}...")
            print()


def main():
    """测试混合搜索"""
    import argparse
    
    parser = argparse.ArgumentParser(description='混合搜索测试')
    parser.add_argument('query', help='查询文本')
    parser.add_argument('-n', '--num', type=int, default=5, help='返回结果数')
    parser.add_argument('--semantic-weight', type=float, default=0.6, help='语义权重')
    parser.add_argument('--bm25-weight', type=float, default=0.4, help='BM25权重')
    parser.add_argument('--year-min', type=int, help='最小年份')
    parser.add_argument('--min-if', type=float, help='最小影响因子')
    
    args = parser.parse_args()
    
    # 加载混合搜索
    search = HybridSearch()
    
    # 搜索
    results = search.search(
        args.query,
        n_results=args.num,
        semantic_weight=args.semantic_weight,
        bm25_weight=args.bm25_weight,
        year_min=args.year_min,
        min_if=args.min_if
    )
    
    # 打印
    search.print_results(results)


if __name__ == '__main__':
    main()
