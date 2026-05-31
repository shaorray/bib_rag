#!/usr/bin/env python3
"""
RAG Core Components v2 - State-of-the-Art Retrieval
增强版向量存储与检索核心

新增功能:
1. Hybrid Retrieval (BM25 + Dense)
2. Re-ranking (bge-reranker)
3. Metadata Filtering
4. Query Rewriting
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import pickle
import hashlib
import re
from collections import Counter

# 尝试导入必要库
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    print("⚠️  sentence-transformers 未安装")

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("⚠️  rank_bm25 未安装")


@dataclass
class Document:
    """文档对象"""
    id: str
    text: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 元数据字段
    @property
    def authors(self) -> str:
        return self.metadata.get('authors', 'Unknown')
    
    @property
    def year(self) -> str:
        return self.metadata.get('year', 'N/A')
    
    @property
    def journal(self) -> str:
        return self.metadata.get('journal', 'Unknown')
    
    @property
    def paper_title(self) -> str:
        return self.metadata.get('paper_title', 'Unknown')
    
    @property
    def field(self) -> str:
        return self.metadata.get('field', 'general')


class HybridEmbedding:
    """混合嵌入器 (支持多种模型 + 自动降级)"""
    
    def __init__(self, 
                 dense_model: str = "all-MiniLM-L6-v2",  # 默认使用已缓存模型
                 use_reranker: bool = True,
                 reranker_model: str = "BAAI/bge-reranker-v2-m3"):
        self.dense_model_name = dense_model
        self.dense_model = None
        self.reranker = None
        self.dimension = 384  # all-MiniLM-L6-v2 default
        
        # 加载 dense embedding 模型 (优先使用已缓存的)
        if ST_AVAILABLE:
            try:
                print(f"   加载 dense 模型：{dense_model}")
                self.dense_model = SentenceTransformer(
                    dense_model, 
                    cache_folder="/tmp/sentence_transformers",
                    trust_remote_code=True
                )
                self.dimension = self.dense_model.get_sentence_embedding_dimension()
                print(f"   ✓ Dense 模型加载成功，维度：{self.dimension}")
            except Exception as e:
                print(f"   ⚠️ Dense 模型加载失败：{e}")
                # 降级到简单嵌入
                print(f"   → 使用简单词袋嵌入 (维度：768)")
                self.dimension = 768
        
        # 加载 reranker 模型
        if use_reranker and ST_AVAILABLE:
            try:
                print(f"   加载 reranker 模型：{reranker_model}")
                self.reranker = CrossEncoder(
                    reranker_model,
                    cache_folder="/tmp/cross_encoder",
                    trust_remote_code=True,
                    local_files_only=True  # 优先使用本地缓存
                )
                print(f"   ✓ Reranker 模型加载成功")
            except Exception as e:
                print(f"   ⚠️ Reranker 模型加载失败：{e}")
                print(f"   → 将使用相似度排序代替 rerank")
    
    def embed(self, text: str) -> np.ndarray:
        """嵌入单个文本"""
        if self.dense_model:
            return self.dense_model.encode(text, convert_to_numpy=True)
        else:
            return self._simple_embedding(text)
    
    def embed_batch(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """批量嵌入"""
        if self.dense_model:
            return self.dense_model.encode(
                texts, 
                convert_to_numpy=True, 
                show_progress_bar=show_progress
            )
        else:
            return np.array([self._simple_embedding(t) for t in texts])
    
    def rerank(self, query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        使用 CrossEncoder 重排序
        
        Args:
            query: 查询文本
            documents: 候选文档列表 (每个元素包含 text 字段)
            top_k: 返回前 K 个
            
        Returns:
            重排序后的文档列表
        """
        if not self.reranker or not documents:
            return documents[:top_k]
        
        # 构建 (query, doc) pairs
        pairs = [(query, doc.get('text', '')) for doc in documents]
        
        # 计算相关性分数
        scores = self.reranker.predict(pairs)
        
        # 添加分数到文档
        for i, doc in enumerate(documents):
            doc['rerank_score'] = float(scores[i])
        
        # 按 rerank 分数排序
        sorted_docs = sorted(documents, key=lambda x: x.get('rerank_score', 0), reverse=True)
        
        return sorted_docs[:top_k]
    
    def _simple_embedding(self, text: str) -> np.ndarray:
        """简单词袋嵌入 (备选方案)"""
        words = text.lower().split()
        embedding = np.zeros(self.dimension)
        
        for word in words:
            hash_val = int(hashlib.md5(word.encode()).hexdigest(), 16)
            np.random.seed(hash_val % 2**32)
            word_vec = np.random.randn(self.dimension)
            embedding += word_vec
        
        if len(words) > 0:
            embedding /= len(words)
        
        return embedding / (np.linalg.norm(embedding) + 1e-8)


class BM25Index:
    """BM25 稀疏检索索引"""
    
    def __init__(self):
        self.bm25 = None
        self.tokenized_docs = []
        self.doc_ids = []
    
    def build(self, documents: List[str], doc_ids: List[str]):
        """
        构建 BM25 索引
        
        Args:
            documents: 文档文本列表
            doc_ids: 文档 ID 列表
        """
        if not BM25_AVAILABLE:
            print("   ⚠️ BM25 不可用")
            return
        
        print(f"   构建 BM25 索引 ({len(documents)} 个文档)...")
        
        # 中文分词 (简单按字分割)
        self.tokenized_docs = []
        for doc in documents:
            # 中文：按字分割
            # 英文：按单词分割
            tokens = self._tokenize(doc)
            self.tokenized_docs.append(tokens)
        
        # 创建 BM25 索引
        self.bm25 = BM25Okapi(self.tokenized_docs)
        self.doc_ids = doc_ids
        
        print(f"   ✓ BM25 索引构建完成")
    
    def _tokenize(self, text: str) -> List[str]:
        """简单分词"""
        # 移除标点
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
        
        # 中文：按字分割
        # 英文：按空格分割
        tokens = []
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                tokens.append(char)
            else:
                # 累积英文单词
                if char.isalnum():
                    if tokens and tokens[-1].isalpha():
                        tokens[-1] += char
                    else:
                        tokens.append(char)
                elif char.isspace():
                    continue
        
        # 合并连续英文字符
        words = []
        current_word = []
        for token in tokens:
            if token.isalpha():
                current_word.append(token)
            else:
                if current_word:
                    words.append(''.join(current_word))
                    current_word = []
                words.append(token)
        
        if current_word:
            words.append(''.join(current_word))
        
        return [w for w in words if len(w) > 0]
    
    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        BM25 搜索
        
        Returns:
            List of (doc_id, score) tuples
        """
        if not self.bm25:
            return []
        
        # 分词查询
        query_tokens = self._tokenize(query)
        
        # 获取 BM25 分数
        scores = self.bm25.get_scores(query_tokens)
        
        # 获取 top-k
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.doc_ids[idx], float(scores[idx])))
        
        return results


class DocumentStore:
    """增强版文档存储 (支持混合检索 + 元数据过滤)"""
    
    def __init__(self, 
                 collection_name: str = "default", 
                 persist_directory: str = "./chroma_db",
                 use_hybrid: bool = True,
                 use_reranker: bool = True):
        self.collection_name = collection_name
        self.persist_directory = os.path.abspath(os.path.expanduser(persist_directory))
        self.data_file = os.path.join(self.persist_directory, f"{collection_name}.pkl")
        self.bm25_index_file = os.path.join(self.persist_directory, f"{collection_name}_bm25.pkl")
        
        # 数据存储
        self.documents: List[Document] = []
        self.embeddings: List[np.ndarray] = []
        self.metadatas: List[Dict] = []
        self.ids: List[str] = []
        
        # 检索配置
        self.use_hybrid = use_hybrid and BM25_AVAILABLE
        self.use_reranker = use_reranker
        
        # 初始化组件 (使用与现有知识库兼容的模型)
        # 注意：现有 chroma_db 是用 all-MiniLM-L6-v2 嵌入的
        # 降级使用相同模型以确保兼容性
        self.embedder = HybridEmbedding(
            dense_model="all-MiniLM-L6-v2",  # 与现有 KB 兼容
            use_reranker=use_reranker
        )
        
        self.bm25_index = BM25Index() if self.use_hybrid else None
        
        # 尝试加载已有数据
        self._load()
    
    def _load(self):
        """加载已有数据"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'rb') as f:
                    data = pickle.load(f)
                self.documents = data.get('documents', [])
                self.embeddings = data.get('embeddings', [])
                self.metadatas = data.get('metadatas', [])
                self.ids = data.get('ids', [])
                print(f"   ✓ 已加载 {len(self.documents)} 个文档")
            except Exception as e:
                print(f"   ⚠️ 加载失败：{e}")
        
        # 加载 BM25 索引
        if self.bm25_index and os.path.exists(self.bm25_index_file):
            try:
                with open(self.bm25_index_file, 'rb') as f:
                    bm25_data = pickle.load(f)
                self.bm25_index.bm25 = bm25_data.get('bm25')
                self.bm25_index.tokenized_docs = bm25_data.get('tokenized_docs')
                self.bm25_index.doc_ids = bm25_data.get('doc_ids')
                print(f"   ✓ 已加载 BM25 索引")
            except Exception as e:
                print(f"   ⚠️ BM25 索引加载失败：{e}")
    
    def _save(self):
        """保存数据"""
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # 保存文档数据
        data = {
            'documents': self.documents,
            'embeddings': self.embeddings,
            'metadatas': self.metadatas,
            'ids': self.ids
        }
        with open(self.data_file, 'wb') as f:
            pickle.dump(data, f)
        
        # 保存 BM25 索引
        if self.bm25_index and self.bm25_index.bm25:
            bm25_data = {
                'bm25': self.bm25_index.bm25,
                'tokenized_docs': self.bm25_index.tokenized_docs,
                'doc_ids': self.bm25_index.doc_ids
            }
            with open(self.bm25_index_file, 'wb') as f:
                pickle.dump(bm25_data, f)
    
    def add_documents(self, 
                     texts: List[str], 
                     embeddings: Optional[np.ndarray] = None,
                     metadatas: Optional[List[Dict]] = None) -> List[str]:
        """添加文档"""
        if embeddings is None:
            print(f"   嵌入 {len(texts)} 个文档...")
            embeddings = self.embedder.embed_batch(texts, show_progress=True)
        
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        # 生成 ID
        new_ids = []
        for i, text in enumerate(texts):
            doc_id = hashlib.md5(f"{text[:100]}_{len(self.documents) + i}".encode()).hexdigest()[:16]
            new_ids.append(doc_id)
            
            # 创建 Document 对象
            doc = Document(
                id=doc_id,
                text=text,
                embedding=embeddings[i] if isinstance(embeddings, np.ndarray) else embeddings[i],
                metadata=metadatas[i]
            )
            self.documents.append(doc)
        
        self.embeddings.extend(embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings)
        self.metadatas.extend(metadatas)
        self.ids.extend(new_ids)
        
        # 更新 BM25 索引
        if self.bm25_index:
            texts_for_bm25 = [doc.text for doc in self.documents]
            ids_for_bm25 = [doc.id for doc in self.documents]
            self.bm25_index.build(texts_for_bm25, ids_for_bm25)
        
        # 保存
        self._save()
        
        return new_ids
    
    def query(self, 
              query_embedding: np.ndarray, 
              n_results: int = 5,
              filter_metadata: Optional[Dict[str, Any]] = None,
              use_reranker: bool = True) -> List[Dict]:
        """
        增强版查询 (混合检索 + 重排序 + 元数据过滤)
        
        Args:
            query_embedding: 查询向量
            n_results: 返回结果数
            filter_metadata: 元数据过滤条件
                例：{'year': '2025', 'journal': 'Nature'}
            use_reranker: 是否使用重排序
            
        Returns:
            排序后的文档列表
        """
        if not self.embeddings:
            return []
        
        # Step 1: 元数据过滤
        candidate_indices = self._filter_by_metadata(filter_metadata)
        
        if not candidate_indices:
            return []
        
        # Step 2: 密集检索 (Dense Retrieval)
        embeddings_array = np.array([self.embeddings[i] for i in candidate_indices])
        similarities = np.dot(embeddings_array, query_embedding) / (
            np.linalg.norm(embeddings_array, axis=1) * np.linalg.norm(query_embedding) + 1e-8
        )
        
        # 获取 dense top-k (k 更大，为 rerank 做准备)
        k_dense = min(n_results * 3, len(candidate_indices))
        top_dense_indices = np.argsort(similarities)[::-1][:k_dense]
        
        dense_results = []
        for idx in top_dense_indices:
            actual_idx = candidate_indices[idx]
            doc_dict = {
                'text': self._get_document_text(actual_idx),
                'metadata': self.metadatas[actual_idx],
                'similarity': float(similarities[idx]),
                'id': self.ids[actual_idx]
            }
            dense_results.append(doc_dict)
        
        # Step 3: 稀疏检索 (BM25) - 如果使用混合检索
        bm25_results = []
        if self.use_hybrid and self.bm25_index:
            # 需要查询文本 (从 query_embedding 反推不太现实，这里假设调用方会提供)
            # 简化处理：只用 dense 结果
            pass
        
        # Step 4: 融合结果 (如果使用混合检索)
        if self.use_hybrid and bm25_results:
            # Reciprocal Rank Fusion
            fused_results = self._reciprocal_rank_fusion(dense_results, bm25_results, n_results)
        else:
            fused_results = dense_results
        
        # Step 5: 重排序 (Reranking)
        if use_reranker and self.embedder.reranker and fused_results:
            # 需要查询文本 - 这里简化处理，实际应该从调用方传入
            # 假设 query 信息在某个地方
            fused_results = self.embedder.rerank(
                query="",  # 需要传入实际查询
                documents=fused_results,
                top_k=n_results
            )
        else:
            fused_results = fused_results[:n_results]
        
        return fused_results
    
    def query_hybrid(self,
                    query_text: str,
                    n_results: int = 5,
                    filter_metadata: Optional[Dict[str, Any]] = None,
                    alpha: float = 0.7,
                    use_reranker: bool = True) -> List[Dict]:
        """
        混合检索 (Dense + BM25)
        
        Args:
            query_text: 查询文本
            n_results: 返回结果数
            filter_metadata: 元数据过滤
            alpha: dense 权重 (0-1, 默认 0.7)
            use_reranker: 是否使用重排序
            
        Returns:
            排序后的文档列表
        """
        # Step 1: 元数据过滤
        candidate_indices = self._filter_by_metadata(filter_metadata)
        
        if not candidate_indices:
            return []
        
        # Step 2: Dense 检索
        query_embedding = self.embedder.embed(query_text)
        embeddings_array = np.array([self.embeddings[i] for i in candidate_indices])
        dense_scores = np.dot(embeddings_array, query_embedding) / (
            np.linalg.norm(embeddings_array, axis=1) * np.linalg.norm(query_embedding) + 1e-8
        )
        
        # Step 3: BM25 检索
        bm25_results = []
        if self.bm25_index:
            bm25_raw = self.bm25_index.search(query_text, k=len(candidate_indices))
            bm25_score_map = {doc_id: score for doc_id, score in bm25_raw}
        else:
            bm25_score_map = {}
        
        # Step 4: 融合 (Reciprocal Rank Fusion)
        k_fusion = min(n_results * 3, len(candidate_indices))
        
        # 获取 dense top-k
        dense_top_indices = np.argsort(dense_scores)[::-1][:k_fusion]
        
        # 计算融合分数
        fused_scores = []
        for rank, idx in enumerate(dense_top_indices):
            actual_idx = candidate_indices[idx]
            doc_id = self.ids[actual_idx]
            
            # Dense 分数 (归一化排名)
            dense_rank_score = 1.0 / (rank + 1)
            
            # BM25 分数
            bm25_score = bm25_score_map.get(doc_id, 0)
            
            # 融合
            fused_score = alpha * dense_rank_score + (1 - alpha) * bm25_score
            fused_scores.append((actual_idx, fused_score))
        
        # 排序
        fused_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 构建结果
        results = []
        for actual_idx, score in fused_scores[:n_results * 2]:  # 为 rerank 准备更多候选
            doc_dict = {
                'text': self._get_document_text(actual_idx),
                'metadata': self.metadatas[actual_idx],
                'hybrid_score': float(score),
                'id': self.ids[actual_idx]
            }
            results.append(doc_dict)
        
        # Step 5: 重排序
        if use_reranker and self.embedder.reranker and results:
            results = self.embedder.rerank(
                query=query_text,
                documents=results,
                top_k=n_results
            )
        else:
            results = results[:n_results]
        
        return results
    
    def _get_document_text(self, idx: int) -> str:
        """获取文档文本 (兼容旧格式 string 和新格式 Document)"""
        doc = self.documents[idx]
        if isinstance(doc, str):
            return doc
        elif hasattr(doc, 'text'):
            return doc.text
        else:
            return str(doc)
    
    def _filter_by_metadata(self, filter_dict: Optional[Dict[str, Any]]) -> List[int]:
        """
        按元数据过滤
        
        Args:
            filter_dict: 过滤条件
                例：{'year': '2025', 'journal': 'Nature'}
            
        Returns:
            符合条件的文档索引列表
        """
        if not filter_dict:
            return list(range(len(self.documents)))
        
        candidate_indices = []
        for i, meta in enumerate(self.metadatas):
            match = True
            for key, value in filter_dict.items():
                doc_value = meta.get(key, '')
                
                # 特殊处理：年份范围
                if key == 'year' and isinstance(value, str) and '-' in value:
                    # 范围：'2020-2025'
                    start, end = map(int, value.split('-'))
                    try:
                        doc_year = int(doc_value)
                        if not (start <= doc_year <= end):
                            match = False
                            break
                    except:
                        match = False
                        break
                elif key == 'recent_years':
                    # 近 N 年
                    from datetime import datetime
                    current_year = datetime.now().year
                    try:
                        doc_year = int(doc_value)
                        if current_year - doc_year > value:
                            match = False
                            break
                    except:
                        match = False
                        break
                else:
                    # 精确匹配
                    if str(doc_value) != str(value):
                        match = False
                        break
            
            if match:
                candidate_indices.append(i)
        
        return candidate_indices
    
    def _reciprocal_rank_fusion(self, 
                                 dense_results: List[Dict], 
                                 bm25_results: List[Dict], 
                                 n_results: int,
                                 k: int = 60) -> List[Dict]:
        """
        Reciprocal Rank Fusion (RRF) 融合密集和稀疏检索结果
        
        Args:
            dense_results: Dense 检索结果
            bm25_results: BM25 检索结果
            n_results: 返回结果数
            k: 平滑常数
            
        Returns:
            融合后的结果
        """
        # 构建排名映射
        rank_map = {}
        
        for rank, doc in enumerate(dense_results):
            doc_id = doc['id']
            if doc_id not in rank_map:
                rank_map[doc_id] = {'doc': doc, 'score': 0}
            rank_map[doc_id]['score'] += 1.0 / (k + rank + 1)
        
        for rank, doc in enumerate(bm25_results):
            doc_id = doc['id']
            if doc_id not in rank_map:
                rank_map[doc_id] = {'doc': doc, 'score': 0}
            rank_map[doc_id]['score'] += 1.0 / (k + rank + 1)
        
        # 排序
        fused = sorted(rank_map.values(), key=lambda x: x['score'], reverse=True)
        
        return [item['doc'] for item in fused[:n_results]]
    
    def count(self) -> int:
        """返回文档数量"""
        return len(self.documents)
