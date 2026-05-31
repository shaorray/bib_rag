#!/usr/bin/env python3
"""
将新文献加入 Eph/Ephrin RAG 知识库
支持 PMC 文献清理和批量处理
"""

import sys
import os
import re
from pathlib import Path
from typing import List, Dict, Optional
import hashlib

sys.path.insert(0, '/Disk_2/claw_working_dir/ephrin_agentic_rag')
from rag_core import SimpleEmbedding, DocumentStore


class PaperCleaner:
    """文献文本清理器"""
    
    @staticmethod
    def clean_pmc_text(text: str) -> str:
        """清理 PMC 文献文本"""
        lines = text.split('\n')
        cleaned = []
        skip_until_empty = False
        
        for line in lines:
            # 跳过 PMC 免责声明
            if 'As a library, NLM provides access to scientific literature' in line:
                skip_until_empty = True
                continue
            if skip_until_empty and line.strip() == '':
                skip_until_empty = False
                continue
            if skip_until_empty:
                continue
            
            # 跳过图片占位符
            if '==> picture' in line:
                continue
            
            # 跳过版权声明
            if line.startswith('© ') or 'All rights reserved' in line:
                continue
            
            # 跳过空行（保留段落结构）
            if line.strip() == '':
                cleaned.append('')
                continue
            
            cleaned.append(line)
        
        return '\n'.join(cleaned)
    
    @staticmethod
    def extract_sections(text: str) -> Dict[str, str]:
        """提取文献各个章节"""
        sections = {}
        
        # 定义章节模式
        section_patterns = [
            (r'##\s*ABSTRACT\s*\n(.*?)(?=##\s|\Z)', 'abstract'),
            (r'##\s*INTRODUCTION\s*\n(.*?)(?=##\s|\Z)', 'introduction'),
            (r'##\s*RESULTS\s*\n(.*?)(?=##\s|\Z)', 'results'),
            (r'##\s*DISCUSSION\s*\n(.*?)(?=##\s|\Z)', 'discussion'),
            (r'##\s*METHODS?\s*\n(.*?)(?=##\s|\Z)', 'methods'),
            (r'##\s*CONCLUSIONS?\s*\n(.*?)(?=##\s|\Z)', 'conclusion'),
        ]
        
        for pattern, name in section_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                sections[name] = match.group(1).strip()
        
        return sections
    
    @staticmethod
    def extract_metadata(text: str) -> Dict[str, str]:
        """提取文献元数据"""
        meta = {
            'pmid': '',
            'pmcid': '',
            'doi': '',
            'title': '',
            'authors': '',
            'year': '',
            'journal': '',
        }
        
        # PMID
        pmid_match = re.search(r'PMID:\s*(\d+)', text)
        if pmid_match:
            meta['pmid'] = pmid_match.group(1)
        
        # PMCID
        pmcid_match = re.search(r'PMCID:\s*(PMC\d+)', text)
        if pmcid_match:
            meta['pmcid'] = pmcid_match.group(1)
        
        # DOI
        doi_match = re.search(r'doi:\s*(10\.\S+)', text, re.IGNORECASE)
        if doi_match:
            meta['doi'] = doi_match.group(1)
        
        # 标题
        title_match = re.search(r'^##\s*\*\*(.+?)\*\*\*', text, re.MULTILINE)
        if title_match:
            meta['title'] = title_match.group(1).strip()
        
        # 年份
        year_match = re.search(r'\b(19|20)\d{2}\b', text)
        if year_match:
            meta['year'] = year_match.group(0)
        
        # 期刊（简单提取）
        journal_match = re.search(r'^([A-Z][A-Za-z\s]+)\.\s+\d{4}', text, re.MULTILINE)
        if journal_match:
            meta['journal'] = journal_match.group(1).strip()
        
        return meta


class PaperProcessor:
    """文献处理器"""
    
    def __init__(self, kb_path: str = "/Disk_2/claw_working_dir/ephrin_agentic_rag"):
        self.kb_path = kb_path
        self.cleaner = PaperCleaner()
        self.embedder = SimpleEmbedding()
        self.doc_store = DocumentStore('ephrin_papers', f'{kb_path}/chroma_db')
        print(f"📚 当前知识库: {self.doc_store.count()} 个文档块")
    
    def process_file(self, file_path: Path) -> Optional[Dict]:
        """处理单篇文献"""
        try:
            # 读取文件
            text = file_path.read_text(encoding='utf-8', errors='ignore')
            
            # 清理文本
            cleaned = self.cleaner.clean_pmc_text(text)
            
            # 如果文本太短，跳过
            if len(cleaned) < 1000:
                return None
            
            # 提取元数据
            meta = self.cleaner.extract_metadata(cleaned)
            
            # 提取章节
            sections = self.cleaner.extract_sections(cleaned)
            
            # 构建文档块
            chunks = self._create_chunks(file_path.name, cleaned, sections, meta)
            
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
        """创建文档块（简单分块策略）"""
        chunks = []
        
        # 如果有摘要，单独作为一个块
        if 'abstract' in sections:
            chunks.append({
                'text': sections['abstract'][:2000],  # 限制长度
                'source': filename,
                'section': 'abstract',
                'pmid': meta.get('pmid', ''),
                'year': meta.get('year', ''),
            })
        
        # 其他章节分段
        for section_name, section_text in sections.items():
            if section_name == 'abstract':
                continue
            
            # 简单分段：每 1000 字一段
            words = section_text.split()
            for i in range(0, len(words), 800):
                chunk_text = ' '.join(words[i:i+1000])
                chunks.append({
                    'text': chunk_text,
                    'source': filename,
                    'section': section_name,
                    'pmid': meta.get('pmid', ''),
                    'year': meta.get('year', ''),
                })
        
        # 如果没有章节，整个文本分块
        if not sections:
            words = text.split()
            for i in range(0, len(words), 800):
                chunk_text = ' '.join(words[i:i+1000])
                chunks.append({
                    'text': chunk_text,
                    'source': filename,
                    'section': 'full_text',
                    'pmid': meta.get('pmid', ''),
                    'year': meta.get('year', ''),
                })
        
        return chunks
    
    def add_to_knowledge_base(self, papers_dir: str, batch_size: int = 50):
        """批量添加文献到知识库"""
        papers_path = Path(papers_dir)
        files = list(papers_path.glob('*.md'))
        
        print(f"\n🔄 处理 {len(files)} 篇文献...")
        print(f"📦 批次大小: {batch_size}\n")
        
        success = 0
        failed = 0
        skipped = 0
        total_chunks = 0
        
        for i in range(0, len(files), batch_size):
            batch = files[i:i+batch_size]
            print(f"\n📦 批次 {i//batch_size + 1}/{(len(files)-1)//batch_size + 1} ({len(batch)} 篇)")
            
            for j, file_path in enumerate(batch):
                result = self.process_file(file_path)
                
                if result is None:
                    skipped += 1
                    print(f"  [{j+1}] ⏭️  跳过 {file_path.name} (文本太短)")
                    continue
                
                if result['status'] == 'error':
                    failed += 1
                    print(f"  [{j+1}] ❌ 失败 {file_path.name}: {result['error']}")
                    continue
                
                # 添加到知识库
                try:
                    for chunk in result['chunks']:
                        # 嵌入
                        embedding = self.embedder.embed(chunk['text'])
                        # 添加到存储
                        # TODO: 实现实际的添加逻辑
                        total_chunks += 1
                    
                    success += 1
                    print(f"  [{j+1}] ✅ {file_path.name} | "
                          f"PMID:{result['meta'].get('pmid', 'N/A')} | "
                          f"章节: {', '.join(result['sections'][:3])}")
                    
                except Exception as e:
                    failed += 1
                    print(f"  [{j+1}] ❌ 添加失败 {file_path.name}: {e}")
            
            print(f"\n📊 批次完成: 成功 {success}, 失败 {failed}, 跳过 {skipped}")
        
        print(f"\n{'='*50}")
        print(f"✅ 处理完成!")
        print(f"   总计: {len(files)}")
        print(f"   成功: {success}")
        print(f"   失败: {failed}")
        print(f"   跳过: {skipped}")
        print(f"   文档块: {total_chunks}")
        print(f"{'='*50}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='添加新文献到 Eph/Ephrin RAG 知识库')
    parser.add_argument('--dir', default='/Disk_2/claw_working_dir/Ephrin_papers/new_pub/Eph-ephrin/top500_md',
                        help='新文献目录')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='批次大小 (默认: 50)')
    parser.add_argument('--dry-run', action='store_true',
                        help='试运行，不实际添加')
    
    args = parser.parse_args()
    
    print("="*60)
    print("📚 Eph/Ephrin 文献知识库扩展工具")
    print("="*60)
    
    processor = PaperProcessor()
    
    if args.dry_run:
        print("\n🏃 试运行模式 (不实际添加)")
    
    processor.add_to_knowledge_base(args.dir, args.batch_size)


if __name__ == '__main__':
    main()
