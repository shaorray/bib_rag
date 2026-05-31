#!/usr/bin/env python3
"""
RAG Core Components - 向量存储与检索核心
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import pickle
import hashlib

# 尝试导入 sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    print("⚠️  sentence-transformers 未安装，使用简单词嵌入")


class SimpleEmbedding:
    """简单的句子嵌入器（基于 sentence-transformers）"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        
        if ST_AVAILABLE:
            try:
                print(f"   加载嵌入模型: {model_name}")
                self.model = SentenceTransformer(model_name, cache_folder="/tmp/sentence_transformers")
                self.dimension = self.model.get_sentence_embedding_dimension()
                print(f"   ✓ 模型加载成功，维度: {self.dimension}")
            except Exception as e:
                print(f"   ⚠️ 模型加载失败: {e}，使用简单嵌入")
                self.dimension = 768  # 默认维度
        else:
            self.dimension = 768
    
    def embed(self, text: str) -> np.ndarray:
        """嵌入单个文本"""
        if self.model:
            return self.model.encode(text, convert_to_numpy=True)
        else:
            # 简单词袋嵌入（备选）
            return self._simple_embedding(text)
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """批量嵌入"""
        if self.model:
            return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        else:
            return np.array([self._simple_embedding(t) for t in texts])
    
    def _simple_embedding(self, text: str) -> np.ndarray:
        """简单词袋嵌入（备选方案）"""
        # 基于词哈希的简单嵌入
        words = text.lower().split()
        embedding = np.zeros(self.dimension)
        
        for word in words:
            # 使用哈希生成伪随机向量
            hash_val = int(hashlib.md5(word.encode()).hexdigest(), 16)
            np.random.seed(hash_val % 2**32)
            word_vec = np.random.randn(self.dimension)
            embedding += word_vec
        
        # 归一化
        if len(words) > 0:
            embedding /= len(words)
        
        return embedding / (np.linalg.norm(embedding) + 1e-8)


class DocumentStore:
    """简单的文档存储（基于文件系统）"""
    
    def __init__(self, collection_name: str = "default", persist_directory: str = "./chroma_db"):
        self.collection_name = collection_name
        # 确保使用绝对路径
        self.persist_directory = os.path.abspath(os.path.expanduser(persist_directory))
        self.data_file = os.path.join(self.persist_directory, f"{collection_name}.pkl")
        
        # 数据存储
        self.documents: List[str] = []
        self.embeddings: List[np.ndarray] = []
        self.metadatas: List[Dict] = []
        self.ids: List[str] = []
        
        # 尝试加载已有数据
        self._load()
    
    def _load(self):
        """加载已有数据"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'rb') as f:
                    data = pickle.load(f)
                self.documents = data['documents']
                self.embeddings = data['embeddings']
                self.metadatas = data['metadatas']
                self.ids = data['ids']
                print(f"   ✓ 已加载 {len(self.documents)} 个文档")
            except Exception as e:
                print(f"   ⚠️ 加载失败: {e}")
    
    def _save(self):
        """保存数据"""
        os.makedirs(self.persist_directory, exist_ok=True)
        data = {
            'documents': self.documents,
            'embeddings': self.embeddings,
            'metadatas': self.metadatas,
            'ids': self.ids
        }
        with open(self.data_file, 'wb') as f:
            pickle.dump(data, f)
    
    def add_documents(self, texts: List[str], embeddings: np.ndarray, 
                     metadatas: List[Dict]) -> List[str]:
        """添加文档"""
        # 生成 ID
        new_ids = []
        for i, text in enumerate(texts):
            doc_id = hashlib.md5(f"{text[:100]}_{len(self.documents) + i}".encode()).hexdigest()[:16]
            new_ids.append(doc_id)
        
        self.documents.extend(texts)
        self.embeddings.extend(embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings)
        self.metadatas.extend(metadatas)
        self.ids.extend(new_ids)
        
        return new_ids
    
    def query(self, query_embedding: np.ndarray, n_results: int = 5) -> List[Dict]:
        """查询相似文档"""
        if not self.embeddings:
            return []
        
        # 转换为 numpy
        embeddings_array = np.array(self.embeddings)
        
        # 计算余弦相似度
        similarities = np.dot(embeddings_array, query_embedding) / (
            np.linalg.norm(embeddings_array, axis=1) * np.linalg.norm(query_embedding) + 1e-8
        )
        
        # 获取前 K 个
        top_k_idx = np.argsort(similarities)[::-1][:n_results]
        
        results = []
        for idx in top_k_idx:
            results.append({
                'id': self.ids[idx],
                'text': self.documents[idx],
                'metadata': self.metadatas[idx],
                'similarity': float(similarities[idx])
            })
        
        return results
    
    def count(self) -> int:
        """获取文档数量"""
        return len(self.documents)
    
    def clear(self):
        """清空数据"""
        self.documents = []
        self.embeddings = []
        self.metadatas = []
        self.ids = []
        if os.path.exists(self.data_file):
            os.remove(self.data_file)
    
    def persist(self):
        """持久化数据"""
        self._save()


class RAGPipeline:
    """基础 RAG 管道"""
    
    def __init__(self, retriever_fn, llm_client=None):
        self.retriever = retriever_fn
        self.llm = llm_client
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """检索文档"""
        return self.retriever(query, k=k)
    
    def generate(self, query: str, documents: List[Dict], 
                 system_prompt: str = None) -> str:
        """生成答案"""
        # 构建上下文
        context = "\n\n---\n\n".join([
            f"[Source: {d['metadata'].get('paper_title', 'Unknown')}]\n{d['text'][:800]}"
            for d in documents[:5]
        ])
        
        # 构建提示
        if system_prompt is None:
            system_prompt = """You are an expert in Eph/Ephrin biology research.
Answer the question based on the provided research papers.
Always cite specific papers when making claims.
If the documents don't contain relevant information, say so."""
        
        prompt = f"""{system_prompt}

Papers:
{context}

Question: {query}

Provide a comprehensive answer with citations:"""
        
        if self.llm:
            return self.llm.generate(prompt)
        else:
            # 简单的关键词匹配答案（无 LLM 时）
            return self._simple_answer(query, documents)
    
    def _simple_answer(self, query: str, documents: List[Dict]) -> str:
        """简单答案生成（无 LLM 时）"""
        if not documents:
            return "No relevant documents found."
        
        # 提取关键词
        query_words = set(query.lower().split())
        
        # 找到最相关的段落
        relevant_sentences = []
        for doc in documents:
            sentences = doc['text'].split('. ')
            for sent in sentences:
                sent_words = set(sent.lower().split())
                overlap = len(query_words & sent_words)
                if overlap >= 2:  # 至少 2 个词重叠
                    relevant_sentences.append({
                        'text': sent,
                        'score': overlap,
                        'source': doc['metadata'].get('paper_title', 'Unknown')
                    })
        
        # 排序并组合
        relevant_sentences.sort(key=lambda x: x['score'], reverse=True)
        
        if relevant_sentences:
            top_sentences = relevant_sentences[:5]
            answer_parts = []
            for sent in top_sentences:
                answer_parts.append(f"- {sent['text'][:200]}... (Source: {sent['source']})")
            
            return "Based on the retrieved documents:\n\n" + "\n".join(answer_parts)
        else:
            return "Retrieved documents don't contain specific information about this query."
    
    def run(self, query: str, k: int = 5) -> Dict[str, Any]:
        """运行完整 RAG 流程"""
        # 检索
        documents = self.retrieve(query, k=k)
        
        # 生成
        answer = self.generate(query, documents)
        
        return {
            'answer': answer,
            'documents': documents,
            'num_docs': len(documents)
        }


# 简单的 LLM 客户端接口
class SimpleLLMClient:
    """简单的 LLM 客户端（实际项目中应替换为 OpenAI/Claude 等）"""
    
    def __init__(self, model: str = "default"):
        self.model = model
        print(f"🤖 初始化 LLM: {model}")
    
    def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        """生成文本（占位符）"""
        # 这里应该调用实际的 LLM API
        # 例如: OpenAI, Claude, DeepSeek 等
        
        # 现在返回简单的提示信息
        return f"[LLM Placeholder Response]\n\nPrompt length: {len(prompt)} chars\n\nTo get real responses, configure an actual LLM API."


if __name__ == "__main__":
    # 测试代码
    print("Testing RAG Core Components...")
    
    # 测试嵌入
    embedder = SimpleEmbedding()
    text = "Eph receptors and ephrins mediate cell signaling"
    embedding = embedder.embed(text)
    print(f"Embedding shape: {embedding.shape}")
    
    # 测试文档存储
    store = DocumentStore("test", "/tmp/test_chroma")
    texts = ["Doc 1 about Eph receptors", "Doc 2 about ephrins"]
    embeddings = embedder.embed_batch(texts)
    metadatas = [{"source": f"doc_{i}"} for i in range(len(texts))]
    
    store.add_documents(texts, embeddings, metadatas)
    store.persist()
    
    # 测试查询
    query = "What are Eph receptors?"
    query_emb = embedder.embed(query)
    results = store.query(query_emb, n_results=2)
    
    print(f"\nQuery results: {len(results)} documents")
    for r in results:
        print(f"  - {r['text'][:50]}... (sim: {r['similarity']:.3f})")
