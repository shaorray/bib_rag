#!/usr/bin/env python3
"""
Eph/Ephrin Agentic RAG Knowledge Base Builder
使用 Self-RAG + CRAG 技术构建智能知识库
"""

import os
import sys
import glob
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import pickle

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import RAG components
from rag_core import DocumentStore, SimpleEmbedding, RAGPipeline
from agentic_workflow import AgenticRAGWorkflow

# Configuration
MARKDOWN_DIR = "/Disk_2/claw_working_dir/Ephrin_papers/review_output/markdown_round2"
KB_DIR = "/Disk_2/claw_working_dir/ephrin_agentic_rag"
CHROMA_DB_PATH = f"{KB_DIR}/chroma_db"
METADATA_PATH = f"{KB_DIR}/paper_metadata.json"

@dataclass
class PaperMetadata:
    """论文元数据结构"""
    title: str
    authors: str = ""
    year: str = ""
    journal: str = ""
    sections: List[str] = field(default_factory=list)
    chunk_count: int = 0
    word_count: int = 0

class EphEphrinKnowledgeBase:
    """Eph/Ephrin 知识库管理器"""
    
    def __init__(self):
        self.kb_dir = KB_DIR
        self.chroma_path = CHROMA_DB_PATH
        self.metadata_path = METADATA_PATH
        self.markdown_dir = MARKDOWN_DIR
        
        # 确保目录存在
        os.makedirs(self.kb_dir, exist_ok=True)
        os.makedirs(self.chroma_path, exist_ok=True)
        
        # 初始化组件
        print("🔧 初始化向量嵌入模型...")
        self.embedder = SimpleEmbedding(model_name="all-MiniLM-L6-v2")
        
        print("📚 初始化文档存储...")
        self.doc_store = DocumentStore(
            collection_name="ephrin_papers",
            persist_directory=self.chroma_path
        )
        
        # 元数据缓存
        self.metadata: Dict[str, PaperMetadata] = {}
        
    def _parse_filename(self, filename: str) -> Dict[str, str]:
        """从文件名解析论文信息"""
        # 移除 .md 扩展名
        name = filename.replace('.md', '')
        
        # 尝试解析: "Author et al. - Year - Title"
        parts = name.split(' - ')
        
        result = {
            'filename': filename,
            'authors': '',
            'year': '',
            'title': name
        }
        
        if len(parts) >= 3:
            result['authors'] = parts[0].strip()
            result['year'] = parts[1].strip()
            result['title'] = ' - '.join(parts[2:]).strip()
        
        return result
    
    def _smart_chunk(self, text: str, chunk_size: int = 800, overlap: int = 150) -> List[Dict]:
        """
        智能分块策略
        - 按段落分割
        - 保留语义完整性
        - 重叠窗口确保连贯性
        """
        chunks = []
        
        # 按段落分割
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        current_chunk = []
        current_size = 0
        chunk_index = 0
        
        for para in paragraphs:
            para_len = len(para)
            
            # 如果当前块加上新段落超过大小，保存当前块
            if current_size + para_len > chunk_size and current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'index': chunk_index,
                    'word_count': len(chunk_text.split())
                })
                
                # 保留重叠部分
                overlap_text = '\n\n'.join(current_chunk[-2:]) if len(current_chunk) >= 2 else current_chunk[-1]
                current_chunk = [overlap_text] if overlap else []
                current_size = len(overlap_text)
                chunk_index += 1
            
            current_chunk.append(para)
            current_size += para_len
        
        # 保存最后一个块
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'index': chunk_index,
                'word_count': len(chunk_text.split())
            })
        
        return chunks
    
    def build_knowledge_base(self, batch_size: int = 50) -> bool:
        """构建完整知识库"""
        print(f"\n{'='*60}")
        print("📚 构建 Eph/Ephrin 知识库")
        print(f"{'='*60}\n")
        
        # 检查是否已有数据
        existing_count = self.doc_store.count()
        if existing_count > 0:
            print(f"⚠️  知识库已存在 {existing_count} 个文档块")
            response = input("是否重新构建？(y/n): ").lower().strip()
            if response != 'y':
                print("使用现有知识库")
                self._load_metadata()
                return True
            print("\n🗑️  清除现有数据...")
            self.doc_store.clear()
        
        # 获取所有 markdown 文件
        md_files = sorted(glob.glob(os.path.join(self.markdown_dir, "*.md")))
        total_files = len(md_files)
        
        if total_files == 0:
            print(f"❌ 在 {self.markdown_dir} 未找到 markdown 文件")
            return False
        
        print(f"📂 找到 {total_files} 个论文文件\n")
        
        all_chunks = []
        total_words = 0
        
        for i, file_path in enumerate(md_files, 1):
            filename = os.path.basename(file_path)
            
            if i % 10 == 0 or i == 1:
                print(f"   处理中... {i}/{total_files} ({filename[:50]}...)")
            
            try:
                # 解析文件名
                parsed = self._parse_filename(filename)
                
                # 读取内容
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 智能分块
                chunks = self._smart_chunk(content)
                
                # 创建元数据
                meta = PaperMetadata(
                    title=parsed['title'],
                    authors=parsed['authors'],
                    year=parsed['year'],
                    chunk_count=len(chunks),
                    word_count=len(content.split())
                )
                self.metadata[filename] = meta
                
                total_words += meta.word_count
                
                # 准备文档块
                for chunk in chunks:
                    chunk_meta = {
                        'source_file': filename,
                        'paper_title': parsed['title'],
                        'authors': parsed['authors'],
                        'year': parsed['year'],
                        'chunk_index': chunk['index'],
                        'word_count': chunk['word_count']
                    }
                    
                    all_chunks.append({
                        'text': chunk['text'],
                        'metadata': chunk_meta
                    })
                
                # 批量处理
                if len(all_chunks) >= batch_size:
                    self._add_documents_batch(all_chunks)
                    all_chunks = []
                    
            except Exception as e:
                print(f"   ⚠️  跳过文件 {filename}: {e}")
                continue
        
        # 处理剩余文档
        if all_chunks:
            self._add_documents_batch(all_chunks)
        
        # 保存元数据
        self._save_metadata()
        
        # 持久化文档存储
        self.doc_store.persist()
        print(f"💾 文档存储已持久化")
        
        # 显示统计
        final_count = self.doc_store.count()
        print(f"\n{'='*60}")
        print("✅ 知识库构建完成！")
        print(f"{'='*60}")
        print(f"📊 统计信息:")
        print(f"   • 论文数量: {len(self.metadata)}")
        print(f"   • 文档块数量: {final_count}")
        print(f"   • 总词数: {total_words:,}")
        print(f"   • 平均每篇论文: {final_count // len(self.metadata) if self.metadata else 0} 个块")
        print(f"\n💾 数据保存在: {self.chroma_path}")
        
        return True
    
    def _add_documents_batch(self, chunks: List[Dict]):
        """批量添加文档"""
        texts = [c['text'] for c in chunks]
        metadatas = [c['metadata'] for c in chunks]
        
        # 生成嵌入
        embeddings = self.embedder.embed_batch(texts)
        
        # 添加到存储
        ids = self.doc_store.add_documents(texts, embeddings, metadatas)
        
        print(f"      已添加 {len(ids)} 个文档块")
    
    def _save_metadata(self):
        """保存元数据"""
        data = {k: {
            'title': v.title,
            'authors': v.authors,
            'year': v.year,
            'journal': v.journal,
            'chunk_count': v.chunk_count,
            'word_count': v.word_count
        } for k, v in self.metadata.items()}
        
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 元数据已保存: {self.metadata_path}")
    
    def _load_metadata(self):
        """加载元数据"""
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.metadata = {k: PaperMetadata(**v) for k, v in data.items()}
            print(f"📂 已加载 {len(self.metadata)} 篇论文的元数据")
    
    def get_statistics(self) -> Dict:
        """获取知识库统计信息"""
        total_papers = len(self.metadata)
        total_chunks = self.doc_store.count()
        
        years = [m.year for m in self.metadata.values() if m.year.isdigit()]
        year_range = f"{min(years)}-{max(years)}" if years else "Unknown"
        
        total_words = sum(m.word_count for m in self.metadata.values())
        
        return {
            'total_papers': total_papers,
            'total_chunks': total_chunks,
            'year_range': year_range,
            'total_words': total_words,
            'avg_chunks_per_paper': total_chunks // total_papers if total_papers else 0
        }
    
    def query(self, question: str, top_k: int = 5) -> List[Dict]:
        """简单查询接口"""
        # 生成查询向量
        query_embedding = self.embedder.embed(question)
        
        # 检索
        results = self.doc_store.query(query_embedding, n_results=top_k)
        
        return results
    
    def get_retriever(self):
        """获取检索器（用于 Agentic RAG）"""
        def retriever_fn(query: str, k: int = 8) -> List[Dict]:
            query_embedding = self.embedder.embed(query)
            return self.doc_store.query(query_embedding, n_results=k)
        
        return retriever_fn


def main():
    """主函数"""
    print("\n" + "="*60)
    print("🧠 Eph/Ephrin Agentic RAG Knowledge Base")
    print("="*60 + "\n")
    
    # 初始化知识库
    kb = EphEphrinKnowledgeBase()
    
    # 构建或加载知识库
    if not kb.build_knowledge_base(batch_size=50):
        print("❌ 知识库构建失败")
        return
    
    # 显示统计
    stats = kb.get_statistics()
    print(f"\n📊 知识库统计:")
    for key, value in stats.items():
        print(f"   • {key}: {value}")
    
    # 初始化 Agentic RAG 工作流
    print(f"\n{'='*60}")
    print("🤖 初始化 Agentic RAG 工作流")
    print(f"{'='*60}\n")
    
    workflow = AgenticRAGWorkflow(kb.get_retriever())
    
    # 测试查询
    test_questions = [
        "What is the role of cis-interaction in Eph-ephrin signaling?",
        "How does ephrin-B1 regulate axon guidance through reverse signaling?",
        "What are the controversies about Eph receptor clustering and tetramerization?",
        "Explain the role of ADAM proteases in Eph-ephrin signaling",
    ]
    
    print("\n🧪 运行测试查询...")
    print("-" * 60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n📌 Query {i}: {question}")
        print("-" * 60)
        
        result = workflow.run(question)
        
        print(f"\n📋 Answer:\n{result['answer'][:500]}...")
        print(f"\n📊 Confidence: {result['confidence']:.2f}")
        docs_count = len(result.get('documents', []))
        print(f"🔍 Retrieved: {docs_count} docs")
        print(f"🔄 Retries: {result.get('retries', 0)}")
        print(f"📝 Query rewritten: {result.get('rewritten', False)}")
        print("\n" + "="*60)
    
    print("\n✅ 所有测试完成！")
    print(f"\n知识库已保存到: {CHROMA_DB_PATH}")
    print("可以使用 query_knowledge_base.py 进行交互式查询")


if __name__ == "__main__":
    main()
