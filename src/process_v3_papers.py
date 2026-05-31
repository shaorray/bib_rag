#!/usr/bin/env python3
"""
改进版文献处理脚本 v3
修复问题：
1. 合并子章节到父章节
2. 过滤空章节（<50词）
3. 过滤非学术内容
4. 元数据与内容分离存储
5. 去重
"""

import sys
import os
import re
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict, Counter

sys.path.insert(0, '/Disk_2/claw_working_dir/ephrin_agentic_rag')


class ImprovedPaperCleaner:
    """改进的文献清理器"""
    
    # 非学术章节列表
    SKIP_SECTIONS = {
        'references', 'acknowledgments', 'acknowledgements',
        'acknowledgment', 'acknowledgement',
        'figure legends', 'tables', 'table', 
        'supplementary material', 'supplementary materials',
        'supplementary information', 'supplementary data',
        'competing interests', 'conflict of interest',
        'consent for publication', 'peer review', 'footnotes',
        'figure 1.', 'figure 2.', 'figure 3.', 'figure 4.',
        'figure 5.', 'figure 6.', 'figure 7.', 'figure 8.',
        'table 1.', 'table 2.', 'abbreviations', 'keywords',
        'graphical abstract', 'author contributions',
        'author information', 'funding', 'ethics approval',
        'data availability', 'permissions',
        'publisher\'s note', 'article notes',
        'copyright and license information',
        'natureportfolio', 'springernature',
        'supplementary information 1',
        'supplementary tables',
        'additional files',
        'peer review information',
        'author\'s accepted manuscript',
        # 新增
        'open in a new tab',
        'associated data',
        'author manuscript',
        'contributor information',
        'scholar ]',
        '[pubmed] [google scholar ]',
    }
    
    @staticmethod
    def clean_pmc_text(text: str) -> str:
        """清理文本，只移除PMC声明"""
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
            
            cleaned.append(line)
        
        return '\n'.join(cleaned)
    
    # 标准章节映射（大小写不敏感）
    SECTION_MAP = {
        # Abstract
        'abstract': 'abstract',
        'summary': 'abstract',
        'executive summary': 'abstract',
        
        # Introduction  
        'introduction': 'introduction',
        'background': 'introduction',
        'overview': 'introduction',
        'aims': 'introduction',
        'purpose': 'introduction',
        'objective': 'introduction',
        'objectives': 'introduction',
        'related work': 'introduction',
        'literature review': 'introduction',
        
        # Results
        'results': 'results',
        'findings': 'results',
        'observations': 'results',
        'data': 'results',
        'outcomes': 'results',
        
        # Discussion
        'discussion': 'discussion',
        'interpretation': 'discussion',
        'implications': 'discussion',
        'perspectives': 'discussion',
        'commentary': 'discussion',
        
        # Methods
        'methods': 'methods',
        'methodology': 'methods',
        'materials and methods': 'methods',
        'experimental procedures': 'methods',
        'procedures': 'methods',
        'protocol': 'methods',
        'protocols': 'methods',
        'materials': 'methods',
        'experimental setup': 'methods',
        
        # Conclusion
        'conclusion': 'conclusion',
        'conclusions': 'conclusion',
        'final remarks': 'conclusion',
        'closing remarks': 'conclusion',
        'synthesis': 'conclusion',
        'future directions': 'conclusion',
    }
    
    # 方法关键词（用于检测子章节）
    METHOD_KEYWORDS = [
        'antibodies', 'plasmids', 'cell culture', 'transfection', 
        'western blot', 'immunoprecipitation', 'pcr', 'rt-pcr',
        'sequencing', 'cloning', 'assay', 'assays', 'kinetics',
        'materials', 'reagents', 'buffers', 'solutions',
        'protocol', 'protocols', 'experimental', 'procedures',
        'purification', 'isolation', 'preparation', 'analysis',
        'statistics', 'software', 'equipment', 'instrumentation',
        'elisa', 'facs', 'microscopy', 'imaging',
        'knockdown', 'knockout', 'sirna', 'shrna', 'crispr',
        'sds-page', 'gel electrophoresis', 'chromatography',
        'mass spectrometry', 'nmr', 'crystallography',
        'mutagenesis', 'site-directed', 'deletion',
        'constructs', 'expression', 'transduction',
        'viability', 'proliferation', 'migration', 'invasion',
        'boyden', 'transwell', 'wound healing', 'scratch',
        'immunofluorescence', 'immunohistochemistry', 'ihc',
        'immunoblotting', 'co-immunoprecipitation', 'co-ip',
        'gst pulldown', 'yeast two-hybrid', 'two-hybrid',
        'luciferase', 'reporter assay', 'chip', 'chip-seq',
        'rna-seq', 'rt-qpcr', 'qpcr', 'quantitative pcr'
    ]
    
    @staticmethod
    def extract_hierarchical_sections(text: str) -> Dict[str, str]:
        """提取章节，合并子章节到父章节"""
        lines = text.split('\n')
        
        # 章节栈
        section_stack = []
        section_content = defaultdict(list)
        current_path = []
        
        for line in lines:
            stripped = line.strip()
            
            # 检测章节标题
            if stripped.startswith('## ') and not stripped.startswith('###'):
                # 一级章节
                level = 2
                title = stripped[2:].strip()
                current_path = [title]
                section_stack = [(level, title)]
                
            elif stripped.startswith('### '):
                # 二级章节
                level = 3
                title = stripped[3:].strip()
                
                # 弹出更深层
                while section_stack and section_stack[-1][0] >= level:
                    section_stack.pop()
                
                section_stack.append((level, title))
                current_path = [s[1] for s in section_stack]
                
            elif stripped.startswith('#### '):
                # 三级章节
                level = 4
                title = stripped[4:].strip()
                
                while section_stack and section_stack[-1][0] >= level:
                    section_stack.pop()
                
                section_stack.append((level, title))
                current_path = [s[1] for s in section_stack]
            
            else:
                # 内容行，添加到当前路径
                if current_path:
                    key = ' > '.join(current_path[:2])  # 最多两级
                    section_content[key].append(line)
        
        # 合并内容
        sections = {}
        for key, content_lines in section_content.items():
            content = '\n'.join(content_lines).strip()
            if content:
                sections[key] = content
        
        return sections
    
    @staticmethod
    def extract_metadata(text: str) -> Dict[str, str]:
        """提取元数据"""
        meta = {}
        
        # YAML frontmatter
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
        
        return meta


class MPNetEmbedding:
    """all-mpnet-base-v2 嵌入模型 (768维)"""
    
    def __init__(self):
        import os
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        
        from sentence_transformers import SentenceTransformer
        print("正在加载 all-mpnet-base-v2 (768维)...")
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self.dim = self.model.get_embedding_dimension()
        self.max_tokens = self.model.max_seq_length
        print(f"✓ 模型加载成功: {self.dim}维, 最大{self.max_tokens} tokens")
    
    def embed(self, text: str) -> np.ndarray:
        """生成文本嵌入"""
        embedding = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return embedding.astype(np.float32)


class ImprovedDocumentStore:
    """改进的文档存储 (支持768维)"""
    
    def __init__(self, name: str, db_path: str):
        self.name = name
        self.db_path = Path(db_path)
        self.documents = []
        self.embeddings = []
        self.metadata = []
        self.text_hashes = set()
        
        self._load()
    
    def _load(self):
        data_file = self.db_path / f"{self.name}.pkl"
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
        self.db_path.mkdir(parents=True, exist_ok=True)
        data_file = self.db_path / f"{self.name}.pkl"
        with open(data_file, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'embeddings': self.embeddings,
                'metadata': self.metadata,
            }, f)
    
    def add(self, text: str, embedding: np.ndarray, metadata: Dict) -> bool:
        """添加文档，返回是否成功（去重）"""
        # 去重检查（基于文本前300字符）
        text_hash = hash(text[:300])
        if text_hash in self.text_hashes:
            return False
        
        self.text_hashes.add(text_hash)
        self.documents.append(text)
        self.embeddings.append(embedding)
        self.metadata.append(metadata)
        return True
    
    def query(self, query_embedding: np.ndarray, n_results: int = 10) -> List[Dict]:
        """查询相似文档"""
        if not self.embeddings:
            return []
        
        embeddings_array = np.array(self.embeddings)
        
        # 确保维度匹配
        if embeddings_array.shape[1] != query_embedding.shape[0]:
            print(f"⚠️  维度不匹配: 知识库{embeddings_array.shape[1]} vs 查询{query_embedding.shape[0]}")
            return []
        
        similarities = np.dot(embeddings_array, query_embedding)
        
        top_k = min(n_results, len(similarities))
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            meta = self.metadata[idx]
            results.append({
                'text': self.documents[idx],
                'pmid': meta.get('pmid', 'N/A'),
                'year': meta.get('year', 'N/A'),
                'journal': meta.get('journal', 'N/A'),
                'if': meta.get('if', 'N/A'),
                'citations': meta.get('citations', 'N/A'),
                'tier': meta.get('tier', 'N/A'),
                'section': meta.get('section', 'N/A'),
                'score': float(similarities[idx]),
            })
        
        return results
    
    def count(self) -> int:
        return len(self.documents)


class ImprovedPaperProcessor:
    """改进的文献处理器"""
    
    def __init__(self, kb_path: str = "/Disk_2/claw_working_dir/ephrin_agentic_rag"):
        self.kb_path = kb_path
        self.cleaner = ImprovedPaperCleaner()
        self.embedder = MPNetEmbedding()
        self.doc_store = ImprovedDocumentStore('ephrin_papers_v3', f'{kb_path}/chroma_db_v3')
        print(f"📚 当前知识库: {self.doc_store.count()} 个文档块")
    
    def process_file(self, file_path: Path) -> Optional[Dict]:
        """处理单篇文献"""
        try:
            text = file_path.read_text(encoding='utf-8', errors='ignore')
            
            # 清理文本
            cleaned = self.cleaner.clean_pmc_text(text)
            
            # 提取元数据
            meta = self.cleaner.extract_metadata(cleaned)
            
            # 提取章节（层次化）
            sections = self.cleaner.extract_hierarchical_sections(cleaned)
            
            # 构建文档块
            chunks = self._create_chunks(file_path.name, cleaned, sections, meta)
            
            if not chunks:
                return None
            
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
        """创建文档块（改进版）"""
        chunks = []
        
        chunk_size = 800
        overlap = 200
        min_chunk_size = 50
        
        # 元数据
        meta_dict = {
            'pmid': meta.get('pmid', ''),
            'year': meta.get('year', ''),
            'journal': meta.get('journal', ''),
            'if': meta.get('impact_factor', ''),
            'citations': meta.get('citations', ''),
            'tier': meta.get('tier', ''),
        }
        
        # 处理每个章节
        for section_key, section_text in sections.items():
            # 获取基础章节名（使用完整路径的最后两级）
            path_parts = section_key.split(' > ')
            base_name = path_parts[0].lower() if len(path_parts) == 1 else path_parts[-2].lower()
            
            # 检查是否需要跳过
            if any(skip in base_name for skip in self.cleaner.SKIP_SECTIONS):
                continue
            
            # 映射到标准名（更广泛的匹配，包含关键词检测）
            section_name = ImprovedPaperCleaner.SECTION_MAP.get(base_name, 'other')
            
            # 如果标题没匹配到，尝试关键词检测
            if section_name == 'other':
                # 检查是否包含方法关键词
                if any(base_name.startswith(kw) for kw in ImprovedPaperCleaner.METHOD_KEYWORDS):
                    section_name = 'methods'
                # 检查是否是结果关键词
                elif any(base_name.startswith(kw) for kw in ['figure', 'table', 'supplementary']):
                    section_name = 'results'
            
            # 过滤空内容
            words = section_text.split()
            if len(words) < min_chunk_size:
                continue
            
            # 分段
            step = chunk_size - overlap
            for i in range(0, len(words), step):
                chunk_words = words[i:i + chunk_size]
                
                # 确保最后一块不会太小
                if len(chunk_words) < min_chunk_size and i > 0:
                    continue
                
                chunk_text = ' '.join(chunk_words)
                
                chunks.append({
                    'text': chunk_text,
                    'meta': meta_dict,
                    'section': section_name if i == 0 else f"{section_name}_cont",
                })
        
        return chunks
    
    def add_to_knowledge_base(self, papers_dir: str, batch_size: int = 50, max_papers: int = None):
        """批量添加文献"""
        papers_path = Path(papers_dir)
        files = sorted(papers_path.glob('*.md'))
        
        if max_papers:
            files = files[:max_papers]
        
        print(f"\n🔄 处理 {len(files)} 篇文献...")
        
        success = 0
        failed = 0
        skipped = 0
        total_chunks = 0
        unique_chunks = 0
        
        for i in range(0, len(files), batch_size):
            batch = files[i:i+batch_size]
            batch_num = i//batch_size + 1
            total_batches = (len(files) - 1)//batch_size + 1
            
            for j, file_path in enumerate(batch):
                result = self.process_file(file_path)
                
                if result is None:
                    skipped += 1
                    continue
                
                if result['status'] == 'error':
                    failed += 1
                    continue
                
                # 添加到知识库
                added_in_file = 0
                for chunk in result['chunks']:
                    try:
                        # 添加元数据前缀到文本
                        meta_prefix = f"PMID:{chunk['meta'].get('pmid', 'N/A')} | "
                        meta_prefix += f"Year:{chunk['meta'].get('year', 'N/A')} | "
                        meta_prefix += f"Journal:{chunk['meta'].get('journal', 'N/A')} | "
                        meta_prefix += f"IF:{chunk['meta'].get('if', 'N/A')} | "
                        meta_prefix += f"Citations:{chunk['meta'].get('citations', 'N/A')}\n"
                        
                        full_text = meta_prefix + chunk['text']
                        
                        # 元数据用于检索显示
                        display_meta = {
                            'pmid': chunk['meta'].get('pmid', ''),
                            'year': chunk['meta'].get('year', ''),
                            'journal': chunk['meta'].get('journal', ''),
                            'if': chunk['meta'].get('if', ''),
                            'citations': chunk['meta'].get('citations', ''),
                            'tier': chunk['meta'].get('tier', ''),
                            'section': chunk['section'],
                        }
                        
                        embedding = self.embedder.embed(full_text)
                        if self.doc_store.add(full_text, embedding, display_meta):
                            unique_chunks += 1
                            added_in_file += 1
                        total_chunks += 1
                    except Exception as e:
                        print(f"  ❌ 嵌入失败: {e}")
                
                success += 1
                
                if (j + 1) % 10 == 0 or added_in_file == 0:
                    print(f"  [{j+1}] ✅ {file_path.name} | "
                          f"PMID:{result['meta'].get('pmid', 'N/A')} | "
                          f"块:{added_in_file}")
            
            # 每批次保存
            self.doc_store._save()
            print(f"\n📦 批次 {batch_num}/{total_batches}: "
                  f"成功 {success}, 失败 {failed}, 跳过 {skipped}, "
                  f"总块 {total_chunks}, 唯一块 {unique_chunks}")
        
        # 最终保存
        self.doc_store._save()
        
        print(f"\n{'='*60}")
        print(f"✅ 处理完成!")
        print(f"   总计: {len(files)}")
        print(f"   成功: {success}")
        print(f"   失败: {failed}")
        print(f"   跳过: {skipped}")
        print(f"   生成块: {total_chunks}")
        print(f"   唯一块: {unique_chunks}")
        print(f"   去重: {total_chunks - unique_chunks}")
        print(f"   知识库总计: {self.doc_store.count()}")
        print(f"{'='*60}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='改进版文献处理')
    parser.add_argument('--dir', 
                        default='/Disk_2/claw_working_dir/Ephrin_papers/new_pub/Eph-ephrin/top500_md_v2',
                        help='文献目录')
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--max-papers', type=int, default=None)
    
    args = parser.parse_args()
    
    print("="*60)
    print("📚 Eph/Ephrin 文献知识库 - 改进版 v3")
    print("="*60)
    
    processor = ImprovedPaperProcessor()
    processor.add_to_knowledge_base(args.dir, args.batch_size, args.max_papers)


if __name__ == '__main__':
    main()
