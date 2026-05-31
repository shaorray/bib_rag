#!/usr/bin/env python3
"""
增量添加论文到 Eph/Ephrin RAG 知识库
基于旧版 add_new_papers.py 重构，实际可用版本

功能:
- 段落级分块 (skip reference sections)
- bge-m3 嵌入 (本地 server port 11435)
- 自动去重 (按文件名 + 内容 hash)
- 批量追加 (默认 100 篇/批)
- 元数据从文件内容提取 (跳过 Zotero bib)
"""

import os
import sys
import re
import json
import hashlib
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

# LangChain
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# ======================== 配置 ========================

KB_ROOT = "/Disk_bot/Eph/ephrin_agentic_rag"
CHROMA_DB_PATH = f"{KB_ROOT}/chroma_db_v2"  # 使用 v2 或新建
METADATA_LOG = f"{KB_ROOT}/data/incremental_metadata.json"

# bge-m3 本地嵌入服务
EMBEDDING_CONFIG = {
    "model": "bge-m3",
    "base_url": "http://127.0.0.1:11435",  # 本地 Flask wrapper
}

# 分块配置
CHUNK_CONFIG = {
    "chunk_size": 800,      # 每块目标字数
    "chunk_overlap": 100,   # 段落间重叠字数
    "max_chunk_size": 1200, # 单块上限
}

# 批次配置
BATCH_CONFIG = {
    "batch_size": 100,
    "persist_interval": 1,   # 每处理完 1 批就 persist
}

# ======================== 数据结构 ========================

@dataclass
class PaperMetadata:
    filename: str
    title: str = ""
    authors: str = ""
    year: str = ""
    journal: str = ""
    doi: str = ""
    pmid: str = ""
    pmcid: str = ""
    word_count: int = 0
    chunk_count: int = 0
    added_at: str = ""
    content_hash: str = ""  # 用于检测内容变更
    sections_found: List[str] = None
    
    def __post_init__(self):
        if self.sections_found is None:
            self.sections_found = []


# ======================== 文本处理 ========================

class TextProcessor:
    """文本清理、分块、章节提取"""
    
    REFERENCE_HEADERS = [
        r'^#*\s*REFERENCES?\s*$',
        r'^#*\s*BIBLIOGRAPHY\s*$',
        r'^#*\s*ACKNOWLEDGMENTS?\s*$',
        r'^#*\s*SUPPLEMENTARY\s*',
        r'^#*\s*APPENDIX\s*$',
        r'^#*\s*DATA AVAILABILITY\s*',
        r'^#*\s*CONFLICT OF INTEREST\s*',
        r'^#*\s*AUTHOR CONTRIBUTIONS?\s*$',
    ]
    
    SECTION_HEADERS = [
        (r'^#*\s*ABSTRACT\s*$', 'abstract'),
        (r'^#*\s*SUMMARY\s*$', 'abstract'),
        (r'^#*\s*INTRODUCTION\s*$', 'introduction'),
        (r'^#*\s*BACKGROUND\s*$', 'introduction'),
        (r'^#*\s*METHODS?\s*$', 'methods'),
        (r'^#*\s*MATERIALS?\s*', 'methods'),
        (r'^#*\s*EXPERIMENTAL\s*', 'methods'),
        (r'^#*\s*RESULTS?\s*$', 'results'),
        (r'^#*\s*FINDINGS?\s*$', 'results'),
        (r'^#*\s*DISCUSSION\s*$', 'discussion'),
        (r'^#*\s*CONCLUSIONS?\s*$', 'conclusion'),
    ]
    
    @classmethod
    def clean_text(cls, text: str) -> str:
        """清理 PDF 提取产生的噪声"""
        lines = text.split('\n')
        cleaned = []
        
        for line in lines:
            # 跳过图片占位符
            if '==> image' in line.lower() or '==> picture' in line.lower():
                continue
            # 跳过版权声明行
            if line.strip().startswith('©') and 'rights reserved' in line:
                continue
            # 跳过 PMC 免责声明
            if 'As a library, NLM provides access' in line:
                continue
            # 保留其他内容（包括空行用于段落结构）
            cleaned.append(line)
        
        return '\n'.join(cleaned)
    
    @classmethod
    def truncate_at_references(cls, text: str) -> str:
        """在 References 章节处截断"""
        lines = text.split('\n')
        cutoff_idx = len(lines)
        
        for i, line in enumerate(lines):
            for pattern in cls.REFERENCE_HEADERS:
                if re.match(pattern, line, re.IGNORECASE):
                    cutoff_idx = i
                    break
            if cutoff_idx < len(lines):
                break
        
        return '\n'.join(lines[:cutoff_idx])
    
    @classmethod
    def extract_sections(cls, text: str) -> Dict[str, str]:
        """按章节提取内容"""
        sections = {}
        lines = text.split('\n')
        
        current_section = None
        current_content = []
        
        for line in lines:
            matched = False
            for pattern, sec_name in cls.SECTION_HEADERS:
                if re.match(pattern, line, re.IGNORECASE):
                    # 保存上一章节
                    if current_section and current_content:
                        sections[current_section] = '\n'.join(current_content).strip()
                    current_section = sec_name
                    current_content = []
                    matched = True
                    break
            
            if not matched and current_section:
                current_content.append(line)
        
        # 保存最后一个章节
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    @classmethod
    def smart_chunk(cls, text: str, source: str, section: str = "full_text",
                    chunk_size: int = 800, overlap: int = 100,
                    max_size: int = 1200) -> List[Dict]:
        """
        智能分块:
        - 按段落分割
        - 保留语义完整性（不拆分句子）
        - 重叠窗口确保连贯性
        """
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        chunk_idx = 0
        
        for para in paragraphs:
            para_words = len(para.split())
            
            # 如果当前块加上新段落超过上限，保存当前块
            if current_chunk and (current_word_count + para_words > max_size):
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'source': source,
                    'section': section,
                    'chunk_index': chunk_idx,
                    'word_count': len(chunk_text.split())
                })
                
                # 重叠：保留最后 1-2 个段落
                overlap_paras = current_chunk[-2:] if len(current_chunk) >= 2 else [current_chunk[-1]]
                current_chunk = overlap_paras.copy()
                current_word_count = sum(len(p.split()) for p in current_chunk)
                chunk_idx += 1
            
            current_chunk.append(para)
            current_word_count += para_words
            
            # 如果刚好达到目标大小附近，也保存（避免无限膨胀）
            if current_word_count >= chunk_size and current_word_count >= max_size * 0.8:
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'source': source,
                    'section': section,
                    'chunk_index': chunk_idx,
                    'word_count': len(chunk_text.split())
                })
                current_chunk = []
                current_word_count = 0
                chunk_idx += 1
        
        # 保存最后一个块
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'source': source,
                'section': section,
                'chunk_index': chunk_idx,
                'word_count': len(chunk_text.split())
            })
        
        return chunks
    
    @classmethod
    def extract_metadata_from_text(cls, text: str, filename: str) -> Dict[str, str]:
        """从文本内容提取元数据（简单规则）"""
        meta = {
            'title': '',
            'authors': '',
            'year': '',
            'journal': '',
            'doi': '',
            'pmid': '',
            'pmcid': ''
        }
        
        # PMID
        m = re.search(r'PMID:\s*(\d+)', text)
        if m:
            meta['pmid'] = m.group(1)
        
        # PMCID
        m = re.search(r'PMCID:\s*(PMC\d+)', text)
        if m:
            meta['pmcid'] = m.group(1)
        
        # DOI
        m = re.search(r'(?:doi:|DOI:|https?://doi\.org/)(10\.\S+)', text)
        if m:
            meta['doi'] = m.group(1)
        
        # Year
        m = re.search(r'\b(19\d{2}|20\d{2})\b', text)
        if m:
            meta['year'] = m.group(1)
        
        # Title: 尝试从文件名或文本开头提取
        # 简单策略：取文本第一个非空行作为标题候选
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if lines:
            first_line = lines[0]
            if len(first_line) > 20 and len(first_line) < 300:
                meta['title'] = first_line
        
        # 如果文件名包含更清晰的标题信息
        name = Path(filename).stem
        # 去掉常见前缀如 +++ 或 ^
        name = re.sub(r'^[\+\^\s]+', '', name)
        if len(name) > 10 and (not meta['title'] or len(name) > len(meta['title'])):
            meta['title'] = name
        
        # Journal (简单匹配: 大写期刊名 + 年份)
        m = re.search(r'^([A-Z][A-Za-z\s\&\.]+)\s*\.?\s*\d{4}', text, re.MULTILINE)
        if m:
            meta['journal'] = m.group(1).strip()
        
        return meta


# ======================== 核心添加器 ========================

class IncrementalPaperAdder:
    """增量添加论文到 ChromaDB"""
    
    def __init__(self, chroma_path: str = CHROMA_DB_PATH,
                 embedding_config: Dict = None,
                 metadata_log: str = METADATA_LOG):
        self.chroma_path = chroma_path
        self.metadata_log = metadata_log
        self.embedding_config = embedding_config or EMBEDDING_CONFIG
        
        # 初始化嵌入
        print(f"🔌 初始化嵌入模型: {self.embedding_config['model']}")
        self.embeddings = OllamaEmbeddings(
            model=self.embedding_config['model'],
            base_url=self.embedding_config['base_url']
        )
        # 测试连接
        try:
            test_vec = self.embeddings.embed_query("test")
            print(f"   ✅ 嵌入维度: {len(test_vec)}")
        except Exception as e:
            print(f"   ⚠️  嵌入服务测试失败: {e}")
            print(f"   请确认 bge-m3 server 在 {self.embedding_config['base_url']} 运行")
        
        # 连接或创建 ChromaDB
        print(f"💾 连接向量库: {chroma_path}")
        self.vectordb = Chroma(
            persist_directory=chroma_path,
            embedding_function=self.embeddings,
            collection_name="ephrin_papers"
        )
        
        # 加载已有元数据记录
        self.existing_metadata = self._load_metadata_log()
        print(f"   已有记录: {len(self.existing_metadata)} 篇论文")
        
        self.processor = TextProcessor()
    
    def _load_metadata_log(self) -> Dict[str, Dict]:
        """加载已添加论文的元数据日志"""
        if os.path.exists(self.metadata_log):
            try:
                with open(self.metadata_log, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"   加载元数据日志失败: {e}，创建新日志")
        return {}
    
    def _save_metadata_log(self):
        """保存元数据日志"""
        os.makedirs(os.path.dirname(self.metadata_log), exist_ok=True)
        with open(self.metadata_log, 'w', encoding='utf-8') as f:
            json.dump(self.existing_metadata, f, ensure_ascii=False, indent=2)
    
    def _compute_content_hash(self, text: str) -> str:
        """计算内容 hash 用于检测变更"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()[:16]
    
    def _check_duplicate(self, filename: str, content_hash: str) -> Tuple[bool, Optional[str]]:
        """
        检查是否已存在
        返回: (is_duplicate, existing_id)
        """
        if filename in self.existing_metadata:
            existing = self.existing_metadata[filename]
            if existing.get('content_hash') == content_hash:
                return True, existing.get('doc_id')
            else:
                # 内容变了，需要更新
                return False, existing.get('doc_id')
        return False, None
    
    def process_single_paper(self, file_path: Path) -> Optional[Dict]:
        """
        处理单篇论文，返回可用于 add_documents 的 Document 列表
        返回 None 表示跳过（太短、重复、或失败）
        """
        filename = file_path.name
        
        try:
            # 读取
            text = file_path.read_text(encoding='utf-8', errors='ignore')
            if len(text.strip()) < 500:
                return {'status': 'skip', 'reason': 'too_short', 'filename': filename}
            
            # 清理 + 截断参考文献
            cleaned = self.processor.clean_text(text)
            cleaned = self.processor.truncate_at_references(cleaned)
            
            if len(cleaned.strip()) < 500:
                return {'status': 'skip', 'reason': 'too_short_after_clean', 'filename': filename}
            
            # 内容 hash
            content_hash = self._compute_content_hash(cleaned)
            
            # 查重
            is_dup, existing_id = self._check_duplicate(filename, content_hash)
            if is_dup:
                return {'status': 'skip', 'reason': 'duplicate', 'filename': filename}
            
            # 提取元数据
            meta = self.processor.extract_metadata_from_text(cleaned, filename)
            
            # 提取章节
            sections = self.processor.extract_sections(cleaned)
            sections_found = list(sections.keys())
            
            # 分块
            all_chunks = []
            
            if sections:
                # 按章节分块
                for sec_name, sec_text in sections.items():
                    if sec_text.strip():
                        sec_chunks = self.processor.smart_chunk(
                            sec_text, filename, sec_name,
                            CHUNK_CONFIG['chunk_size'],
                            CHUNK_CONFIG['chunk_overlap'],
                            CHUNK_CONFIG['max_chunk_size']
                        )
                        all_chunks.extend(sec_chunks)
            else:
                # 无章节，全文分块
                all_chunks = self.processor.smart_chunk(
                    cleaned, filename, "full_text",
                    CHUNK_CONFIG['chunk_size'],
                    CHUNK_CONFIG['chunk_overlap'],
                    CHUNK_CONFIG['max_chunk_size']
                )
            
            if not all_chunks:
                return {'status': 'skip', 'reason': 'no_chunks', 'filename': filename}
            
            # 构建 LangChain Documents
            documents = []
            for chunk in all_chunks:
                doc = Document(
                    page_content=chunk['text'],
                    metadata={
                        'source': filename,
                        'section': chunk['section'],
                        'chunk_index': chunk['chunk_index'],
                        'word_count': chunk['word_count'],
                        'title': meta.get('title', filename),
                        'authors': meta.get('authors', ''),
                        'year': meta.get('year', ''),
                        'journal': meta.get('journal', ''),
                        'doi': meta.get('doi', ''),
                        'pmid': meta.get('pmid', ''),
                        'pmcid': meta.get('pmcid', ''),
                        'content_hash': content_hash,
                        'added_at': datetime.now().isoformat(),
                    }
                )
                documents.append(doc)
            
            return {
                'status': 'success',
                'filename': filename,
                'documents': documents,
                'meta': meta,
                'sections_found': sections_found,
                'chunk_count': len(documents),
                'content_hash': content_hash,
                'word_count': len(cleaned.split()),
                'existing_id': existing_id,  # 如果有旧版本需要覆盖
            }
            
        except Exception as e:
            return {'status': 'error', 'filename': filename, 'error': str(e)}
    
    def add_papers(self, papers_dir: str, batch_size: int = 100,
                   dry_run: bool = False) -> Dict:
        """
        批量添加论文
        
        Args:
            papers_dir: md 文件目录
            batch_size: 每批处理数量
            dry_run: True=只统计不实际添加
        """
        papers_path = Path(papers_dir)
        if not papers_path.exists():
            print(f"❌ 目录不存在: {papers_dir}")
            return {'success': 0, 'failed': 0, 'skipped': 0}
        
        md_files = sorted(papers_path.glob('*.md'))
        if not md_files:
            # 递归查找
            md_files = sorted(papers_path.rglob('*.md'))
        
        total = len(md_files)
        print(f"\n{'='*60}")
        print(f"📚 增量添加论文")
        print(f"   目录: {papers_dir}")
        print(f"   发现: {total} 篇 markdown")
        print(f"   批次: {batch_size} 篇/批")
        print(f"   模式: {'试运行' if dry_run else '实际写入'}")
        print(f"{'='*60}\n")
        
        stats = {'success': 0, 'failed': 0, 'skipped': 0, 'total_chunks': 0}
        all_documents = []  # 缓冲当前批次的文档
        current_batch_meta = {}  # 缓冲当前批次的元数据
        
        for i, file_path in enumerate(md_files, 1):
            # 处理单篇
            result = self.process_single_paper(file_path)
            
            if result['status'] == 'skip':
                stats['skipped'] += 1
                reason = result.get('reason', 'unknown')
                print(f"  [{i}/{total}] ⏭️  跳过 {file_path.name} ({reason})")
                continue
            
            if result['status'] == 'error':
                stats['failed'] += 1
                print(f"  [{i}/{total}] ❌ 失败 {file_path.name}: {result['error']}")
                continue
            
            # 成功
            docs = result['documents']
            meta = result['meta']
            filename = result['filename']
            
            if dry_run:
                stats['success'] += 1
                stats['total_chunks'] += len(docs)
                print(f"  [{i}/{total}] ✅ [DRY] {filename} | "
                      f"chunks:{len(docs)} sections:{','.join(result['sections_found'][:3])}")
                continue
            
            # 实际添加：缓冲到批次
            all_documents.extend(docs)
            
            # 记录元数据
            paper_meta = PaperMetadata(
                filename=filename,
                title=meta.get('title', filename),
                authors=meta.get('authors', ''),
                year=meta.get('year', ''),
                journal=meta.get('journal', ''),
                doi=meta.get('doi', ''),
                pmid=meta.get('pmid', ''),
                pmcid=meta.get('pmcid', ''),
                word_count=result['word_count'],
                chunk_count=len(docs),
                added_at=datetime.now().isoformat(),
                content_hash=result['content_hash'],
                sections_found=result['sections_found']
            )
            current_batch_meta[filename] = asdict(paper_meta)
            
            # 如果达到批次大小，执行写入
            if len(all_documents) >= batch_size or i == total:
                self._flush_batch(all_documents, current_batch_meta, stats)
                all_documents = []
                current_batch_meta = {}
                
                # 批次报告
                batch_num = (i - 1) // batch_size + 1
                total_batches = (total - 1) // batch_size + 1
                print(f"\n📦 批次 {batch_num}/{total_batches} 已持久化")
                print(f"   累计: 成功{stats['success']} 跳过{stats['skipped']} 失败{stats['failed']}")
                print(f"   总 chunks: {stats['total_chunks']}\n")
        
        # 最终统计
        print(f"\n{'='*60}")
        print(f"✅ 处理完成!")
        print(f"   总计: {total}")
        print(f"   成功: {stats['success']} ({stats['total_chunks']} chunks)")
        print(f"   跳过: {stats['skipped']}")
        print(f"   失败: {stats['failed']}")
        print(f"{'='*60}")
        
        return stats
    
    def _flush_batch(self, documents: List[Document],
                     batch_meta: Dict, stats: Dict):
        """将缓冲的批次写入 ChromaDB"""
        if not documents:
            return
        
        try:
            # 如果有旧版本，先删除 (基于 metadata 中的 filename)
            # 注意：Chroma 的 delete 需要 ids，但我们这里用 upsert 逻辑
            # 简化处理：直接 add，如果重复由 content_hash 在外层拦截
            
            self.vectordb.add_documents(documents)
            self.vectordb.persist()
            
            # 更新元数据日志
            self.existing_metadata.update(batch_meta)
            self._save_metadata_log()
            
            stats['success'] += len(batch_meta)
            stats['total_chunks'] += len(documents)
            
        except Exception as e:
            print(f"\n❌ 批次写入失败: {e}")
            # 这批算失败
            stats['failed'] += len(batch_meta)
    
    def get_stats(self) -> Dict:
        """获取知识库统计"""
        try:
            count = self.vectordb._collection.count()
        except:
            count = 0
        return {
            'total_vectors': count,
            'total_papers': len(self.existing_metadata),
            'chroma_path': self.chroma_path,
        }


# ======================== CLI ========================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='增量添加论文到 Eph/Ephrin RAG 知识库',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 试运行（不实际写入）
  python3 incremental_add_papers.py /Disk_bot/paper_lib/md --dry-run
  
  # 实际添加，默认 100 篇/批
  python3 incremental_add_papers.py /Disk_bot/paper_lib/md/patterning
  
  # 小批次测试
  python3 incremental_add_papers.py /Disk_bot/paper_lib/md/signaling -b 10
        """
    )
    parser.add_argument('papers_dir', help='论文 markdown 目录')
    parser.add_argument('-b', '--batch-size', type=int, default=100,
                        help='批次大小 (默认: 100)')
    parser.add_argument('--dry-run', action='store_true',
                        help='试运行，只统计不写入')
    parser.add_argument('--db-path', default=CHROMA_DB_PATH,
                        help=f'ChromaDB 路径 (默认: {CHROMA_DB_PATH})')
    
    args = parser.parse_args()
    
    print("="*60)
    print("📚 Eph/Ephrin RAG - 增量添加工具")
    print("="*60)
    
    # 初始化
    adder = IncrementalPaperAdder(chroma_path=args.db_path)
    
    # 显示当前统计
    stats = adder.get_stats()
    print(f"\n📊 当前知识库:")
    print(f"   论文数: {stats['total_papers']}")
    print(f"   向量数: {stats['total_vectors']}")
    print(f"   存储: {stats['chroma_path']}")
    
    # 执行
    result = adder.add_papers(
        papers_dir=args.papers_dir,
        batch_size=args.batch_size,
        dry_run=args.dry_run
    )
    
    # 最终统计
    final_stats = adder.get_stats()
    print(f"\n📊 最终知识库:")
    print(f"   论文数: {final_stats['total_papers']} (+{result['success']})")
    print(f"   向量数: {final_stats['total_vectors']}")


if __name__ == '__main__':
    main()
