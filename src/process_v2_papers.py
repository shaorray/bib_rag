#!/usr/bin/env python3
"""
处理 top500_md_v2 文献并添加到 Eph/Ephrin RAG 知识库
v2 版本包含元数据（PMID, rank, Priority, IF, Citations）
"""

import sys
import os
import re
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import hashlib

sys.path.insert(0, '/Disk_2/claw_working_dir/ephrin_agentic_rag')


class PaperCleaner:
    """文献文本清理器"""
    
    @staticmethod
    def clean_pmc_text(text: str) -> str:
        """清理 PMC 文献文本，保留结构"""
        lines = text.split('\n')
        cleaned = []
        skip_next_empty = False
        
        for line in lines:
            stripped = line.strip()
            
            # 跳过 PMC 免责声明（只跳过这两行）
            if 'As a library, NLM provides access to scientific literature' in line:
                skip_next_empty = True
                continue
            if skip_next_empty and stripped == '':
                skip_next_empty = False
                continue
            if stripped == 'Learn more: PMC Disclaimer | PMC Copyright Notice':
                continue
            
            # 跳过图片占位符
            if '==> picture' in line or '==> table' in line:
                continue
            
            # 跳过版权声明（但保留期刊信息）
            if stripped.startswith('© ') and 'All rights reserved' in stripped:
                continue
            
            cleaned.append(line)
        
        return '\n'.join(cleaned)
    
    @staticmethod
    def extract_metadata(text: str) -> Dict[str, str]:
        """提取文献元数据（v2 格式）"""
        meta = {
            'pmid': '',
            'pmcid': '',
            'doi': '',
            'title': '',
            'authors': '',
            'year': '',
            'journal': '',
            'rank': '',
            'priority': '',
            'impact_factor': '',
            'citations': '',
            'tier': '',
            'area': '',
            'topic': '',
        }
        
        # v2 格式元数据（在 YAML frontmatter 中）
        rank_match = re.search(r'^rank:\s*(\d+)', text, re.MULTILINE)
        if rank_match:
            meta['rank'] = rank_match.group(1)
        
        pmid_match = re.search(r'^PMID:\s*(\d+)', text, re.MULTILINE)
        if pmid_match:
            meta['pmid'] = pmid_match.group(1)
        
        priority_match = re.search(r'^Priority:\s*([\d.]+)', text, re.MULTILINE)
        if priority_match:
            meta['priority'] = priority_match.group(1)
        
        if_match = re.search(r'^Impact Factor:\s*([\d.]+)', text, re.MULTILINE)
        if if_match:
            meta['impact_factor'] = if_match.group(1)
        
        cit_match = re.search(r'^Citations:\s*(\d+)', text, re.MULTILINE)
        if cit_match:
            meta['citations'] = cit_match.group(1)
        
        year_match = re.search(r'^Year:\s*(\d{4})', text, re.MULTILINE)
        if year_match:
            meta['year'] = year_match.group(1)
        
        journal_match = re.search(r'^Journal:\s*(.+)', text, re.MULTILINE)
        if journal_match:
            meta['journal'] = journal_match.group(1).strip()
        
        tier_match = re.search(r'^Tier:\s*(.+)', text, re.MULTILINE)
        if tier_match:
            meta['tier'] = tier_match.group(1).strip()
        
        area_match = re.search(r'^Area:\s*(.+)', text, re.MULTILINE)
        if area_match:
            meta['area'] = area_match.group(1).strip()
        
        topic_match = re.search(r'^Topic:\s*(.+)', text, re.MULTILINE)
        if topic_match:
            meta['topic'] = topic_match.group(1).strip()
        
        # 也尝试提取正文中的信息
        pmcid_match = re.search(r'PMCID:\s*(PMC\d+)', text)
        if pmcid_match:
            meta['pmcid'] = pmcid_match.group(1)
        
        doi_match = re.search(r'doi:\s*(10\.\S+)', text, re.IGNORECASE)
        if doi_match:
            meta['doi'] = doi_match.group(1)
        
        # 提取标题（通常在元数据后的 ## 标题）
        title_match = re.search(r'^##\s*\*\*(.+?)\*\*\*', text, re.MULTILINE)
        if title_match:
            meta['title'] = title_match.group(1).strip()
        else:
            # 尝试其他标题格式
            alt_title = re.search(r'^##\s*(.+)', text, re.MULTILINE)
            if alt_title:
                meta['title'] = alt_title.group(1).strip()
        
        return meta
    
    @staticmethod
    def extract_sections(text: str) -> Dict[str, str]:
        """提取文献各个章节"""
        sections = {}
        
        section_patterns = [
            (r'##\s*ABSTRACT\s*\n(.*?)(?=##\s|\Z)', 'abstract'),
            (r'##\s*SUMMARY\s*\n(.*?)(?=##\s|\Z)', 'abstract'),
            (r'##\s*INTRODUCTION\s*\n(.*?)(?=##\s|\Z)', 'introduction'),
            (r'##\s*BACKGROUND\s*\n(.*?)(?=##\s|\Z)', 'background'),
            (r'##\s*RESULTS?\s*\n(.*?)(?=##\s|\Z)', 'results'),
            (r'##\s*DISCUSSION\s*\n(.*?)(?=##\s|\Z)', 'discussion'),
            (r'##\s*METHODS?\s*\n(.*?)(?=##\s|\Z)', 'methods'),
            (r'##\s*MATERIALS?\s+AND\s+METHODS?\s*\n(.*?)(?=##\s|\Z)', 'methods'),
            (r'##\s*CONCLUSIONS?\s*\n(.*?)(?=##\s|\Z)', 'conclusion'),
        ]
        
        for pattern, name in section_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                sections[name] = match.group(1).strip()
        
        return sections


class SimpleEmbedding:
    """简单嵌入模型（本地，无需网络）"""
    
    def __init__(self):
        self.dim = 384  # 使用简单的统计特征
        print("✓ 使用简单嵌入模型 (384维)")
    
    def embed(self, text: str) -> np.ndarray:
        """生成文本嵌入"""
        # 简单的词频特征 + 哈希特征
        words = text.lower().split()
        
        # 1. 词频特征 (前128维)
        word_counts = {}
        for w in words:
            word_counts[w] = word_counts.get(w, 0) + 1
        
        # 2. 字符n-gram特征 (256维)
        char_features = np.zeros(256)
        text_clean = re.sub(r'\s+', '', text.lower())
        for i in range(len(text_clean) - 2):
            ngram = text_clean[i:i+3]
            idx = hash(ngram) % 256
            char_features[idx] += 1
        
        # 合并特征
        features = np.concatenate([
            char_features,
            np.zeros(128)  # 占位
        ])
        
        # 归一化
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features.astype(np.float32)


class DocumentStore:
    """文档存储（简化版，不依赖 ChromaDB）"""
    
    def __init__(self, name: str, db_path: str):
        self.name = name
        self.db_path = Path(db_path)
        self.documents = []
        self.embeddings = []
        self.metadata = []
        
        # 尝试加载已有数据
        self._load()
    
    def _load(self):
        """加载已有数据"""
        data_file = self.db_path / f"{self.name}_v2.pkl"
        if data_file.exists():
            try:
                with open(data_file, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data.get('documents', [])
                    self.embeddings = data.get('embeddings', [])
                    self.metadata = data.get('metadata', [])
                print(f"✓ 已加载 {len(self.documents)} 个文档块")
            except Exception as e:
                print(f"⚠️  加载失败: {e}")
    
    def _save(self):
        """保存数据"""
        self.db_path.mkdir(parents=True, exist_ok=True)
        data_file = self.db_path / f"{self.name}_v2.pkl"
        with open(data_file, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'embeddings': self.embeddings,
                'metadata': self.metadata,
            }, f)
    
    def add(self, text: str, embedding: np.ndarray, metadata: Dict):
        """添加文档"""
        self.documents.append(text)
        self.embeddings.append(embedding)
        self.metadata.append(metadata)
    
    def query(self, query_embedding: np.ndarray, n_results: int = 10) -> List[Dict]:
        """查询相似文档"""
        if not self.embeddings:
            return []
        
        # 计算余弦相似度
        embeddings_array = np.array(self.embeddings)
        similarities = np.dot(embeddings_array, query_embedding)
        
        # 获取 top-k
        top_k = min(n_results, len(similarities))
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'text': self.documents[idx],
                'metadata': self.metadata[idx],
                'score': float(similarities[idx]),
            })
        
        return results
    
    def count(self) -> int:
        return len(self.documents)


class PaperProcessor:
    """文献处理器"""
    
    def __init__(self, kb_path: str = "/Disk_2/claw_working_dir/ephrin_agentic_rag"):
        self.kb_path = kb_path
        self.cleaner = PaperCleaner()
        self.embedder = SimpleEmbedding()
        self.doc_store = DocumentStore('ephrin_papers_v2', f'{kb_path}/chroma_db_v2')
        print(f"📚 当前知识库: {self.doc_store.count()} 个文档块")
    
    def process_file(self, file_path: Path) -> Optional[Dict]:
        """处理单篇文献"""
        try:
            text = file_path.read_text(encoding='utf-8', errors='ignore')
            
            # 清理文本
            cleaned_text = self.cleaner.clean_pmc_text(text)
            
            # 如果文本太短，跳过
            if len(cleaned_text) < 500:
                return None
            
            # 提取元数据
            meta = self.cleaner.extract_metadata(cleaned_text)
            
            # 提取章节
            sections = self.cleaner.extract_sections(cleaned_text)
            
            # 构建文档块
            chunks = self._create_chunks(file_path.name, cleaned_text, sections, meta)
            
            return {
                'filename': file_path.name,
                'meta': meta,
                'sections': list(sections.keys()),
                'chunks': chunks,
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'filename': file_path.name,
                'status': 'error',
                'error': str(e)
            }
    
    def _create_chunks(self, filename: str, text: str, sections: Dict, meta: Dict) -> List[Dict]:
        """创建文档块"""
        chunks = []
        
        # 元数据前缀
        meta_prefix = f"PMID:{meta.get('pmid', 'N/A')} | Year:{meta.get('year', 'N/A')} | Journal:{meta.get('journal', 'N/A')} | IF:{meta.get('impact_factor', 'N/A')} | Citations:{meta.get('citations', 'N/A')}\n"
        
        # 定义分段策略：每个章节单独处理，每 800 词分一块
        chunk_size = 800
        overlap = 100
        
        # 处理所有重要章节
        section_priority = [
            ('abstract', 'abstract'),
            ('introduction', 'introduction'),
            ('background', 'background'),
            ('results', 'results'),
            ('discussion', 'discussion'),
            ('methods', 'methods'),
            ('conclusion', 'conclusion'),
            ('conclusions', 'conclusion'),
        ]
        
        for section_key, section_name in section_priority:
            if section_key not in sections:
                continue
            
            section_text = sections[section_key]
            section_words = section_text.split()
            section_len = len(section_words)
            
            # 短章节直接作为一个块
            if section_len <= chunk_size:
                chunks.append({
                    'text': meta_prefix + section_text,
                    'source': filename,
                    'section': section_name,
                    'pmid': meta.get('pmid', ''),
                    'year': meta.get('year', ''),
                    'journal': meta.get('journal', ''),
                    'if': meta.get('impact_factor', ''),
                    'citations': meta.get('citations', ''),
                    'tier': meta.get('tier', ''),
                })
            else:
                # 长章节分段，带重叠
                step = chunk_size - overlap
                for i in range(0, section_len, step):
                    chunk_words = section_words[i:i + chunk_size]
                    chunk_text = ' '.join(chunk_words)
                    chunks.append({
                        'text': meta_prefix + chunk_text,
                        'source': filename,
                        'section': f"{section_name}_part{i//step + 1}",
                        'pmid': meta.get('pmid', ''),
                        'year': meta.get('year', ''),
                        'journal': meta.get('journal', ''),
                        'if': meta.get('impact_factor', ''),
                        'citations': meta.get('citations', ''),
                        'tier': meta.get('tier', ''),
                    })
        
        # 如果没有章节，整个文本分块
        if not sections:
            words = text.split()
            for i in range(0, len(words), chunk_size - overlap):
                chunk_text = ' '.join(words[i:i + chunk_size])
                chunks.append({
                    'text': meta_prefix + chunk_text,
                    'source': filename,
                    'section': 'full_text',
                    'pmid': meta.get('pmid', ''),
                    'year': meta.get('year', ''),
                    'journal': meta.get('journal', ''),
                    'if': meta.get('impact_factor', ''),
                    'citations': meta.get('citations', ''),
                    'tier': meta.get('tier', ''),
                })
        
        return chunks
    
    def add_to_knowledge_base(self, papers_dir: str, batch_size: int = 50, max_papers: int = None):
        """批量添加文献到知识库"""
        papers_path = Path(papers_dir)
        files = sorted(papers_path.glob('*.md'))
        
        if max_papers:
            files = files[:max_papers]
        
        print(f"\n🔄 处理 {len(files)} 篇文献...")
        print(f"📦 批次大小: {batch_size}\n")
        
        success = 0
        failed = 0
        skipped = 0
        total_chunks = 0
        
        for i in range(0, len(files), batch_size):
            batch = files[i:i+batch_size]
            batch_num = i//batch_size + 1
            total_batches = (len(files) - 1)//batch_size + 1
            print(f"\n📦 批次 {batch_num}/{total_batches} ({len(batch)} 篇)")
            
            for j, file_path in enumerate(batch):
                result = self.process_file(file_path)
                
                if result is None:
                    skipped += 1
                    continue
                
                if result['status'] == 'error':
                    failed += 1
                    print(f"  [{j+1}] ❌ {file_path.name}: {result['error']}")
                    continue
                
                # 添加到知识库
                try:
                    for chunk in result['chunks']:
                        embedding = self.embedder.embed(chunk['text'])
                        self.doc_store.add(chunk['text'], embedding, {
                            'source': chunk['source'],
                            'section': chunk['section'],
                            'pmid': chunk.get('pmid', ''),
                            'year': chunk.get('year', ''),
                            'journal': chunk.get('journal', ''),
                            'if': chunk.get('if', ''),
                            'citations': chunk.get('citations', ''),
                            'tier': chunk.get('tier', ''),
                        })
                        total_chunks += 1
                    
                    success += 1
                    if (j + 1) % 10 == 0:
                        print(f"  [{j+1}] ✅ {file_path.name} | "
                              f"PMID:{result['meta'].get('pmid', 'N/A')} | "
                              f"章节: {', '.join(result['sections'])}")
                    
                except Exception as e:
                    failed += 1
                    print(f"  [{j+1}] ❌ 添加失败 {file_path.name}: {e}")
            
            # 每批次保存
            self.doc_store._save()
            print(f"\n📊 批次完成: 成功 {success}, 失败 {failed}, 跳过 {skipped}, 文档块 {total_chunks}")
        
        # 最终保存
        self.doc_store._save()
        
        print(f"\n{'='*50}")
        print(f"✅ 处理完成!")
        print(f"   总计: {len(files)}")
        print(f"   成功: {success}")
        print(f"   失败: {failed}")
        print(f"   跳过: {skipped}")
        print(f"   文档块: {total_chunks}")
        print(f"   知识库总计: {self.doc_store.count()}")
        print(f"{'='*50}")
    
    def query(self, query_text: str, n_results: int = 10) -> List[Dict]:
        """查询知识库"""
        query_embedding = self.embedder.embed(query_text)
        return self.doc_store.query(query_embedding, n_results)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='处理 top500_md_v2 文献并加入知识库')
    parser.add_argument('--dir', 
                        default='/Disk_2/claw_working_dir/Ephrin_papers/new_pub/Eph-ephrin/top500_md_v2',
                        help='文献目录')
    parser.add_argument('--batch-size', type=int, default=50, help='批次大小')
    parser.add_argument('--max-papers', type=int, default=None, help='最大处理数量')
    parser.add_argument('--query', type=str, default=None, help='查询测试')
    
    args = parser.parse_args()
    
    print("="*60)
    print("📚 Eph/Ephrin 文献知识库扩展工具 v2")
    print("="*60)
    
    processor = PaperProcessor()
    
    if args.query:
        print(f"\n🔍 查询: {args.query}")
        results = processor.query(args.query, n_results=5)
        for i, r in enumerate(results, 1):
            print(f"\n[{i}] 相关度: {r['score']:.3f}")
            print(f"   来源: {r['metadata'].get('source', 'N/A')}")
            print(f"   章节: {r['metadata'].get('section', 'N/A')}")
            print(f"   文本: {r['text'][:200]}...")
    else:
        processor.add_to_knowledge_base(args.dir, args.batch_size, args.max_papers)


if __name__ == '__main__':
    main()
