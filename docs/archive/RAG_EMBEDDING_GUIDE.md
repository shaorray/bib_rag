# RAG Embedding 技术指南

## 核心概念

创建高质量的RAG嵌入包含两个关键步骤：**Chunking（分块）** 和 **Embedding（嵌入）**

---

## 一、Chunking 策略（从简单到高级）

### 1. Fixed-Size Chunking（固定大小分块）
- **方法**：按固定字符数或token数分割（如512 tokens）
- **优点**：快速、可预测
- **缺点**：可能切断句子或语义，导致信息丢失
- **工具**：LangChain `RecursiveCharacterTextSplitter`

### 2. Semantic Chunking（语义分块）
- **方法**：基于语义相似度而非固定长度
- **实现**：逐句创建embedding，将语义相似的句子组合在一起
- **优点**：保持自然语义流，检索效果更好

### 3. Sliding Window（滑动窗口）
- **方法**：创建重叠的chunks
- **示例**：chunk大小500词，下一个chunk从300词处开始（200词重叠）
- **优点**：保留跨chunk的上下文

### 📌 专家建议
- **起始配置**：512 tokens，10-15%重叠
- **复杂文档**：可用1024 tokens
- **简单问答**：64-128 tokens更优
- **关键**：需要根据实际数据测试调整

---

## 二、Embedding Model 选择

### 关键考量因素

| 因素 | 说明 | 建议 |
|------|------|------|
| Context Window | 模型一次能处理的文本量 | chunk size 不能超过此限制 |
| Embedding Dimensions | 向量长度（384/768/1536等） | 768-1024是较好的平衡点 |
| Language Support | 多语言支持 | 多语言选multilingual-e5或BGE-M3 |
| Performance | MTEB排行榜分数 | 参考但需在真实数据上测试 |

### 推荐开源模型

| 模型 | 维度 | 适用场景 | 代码 |
|------|------|----------|------|
| **all-MiniLM-L6-v2** | 384 | **速度优先**，生产环境低延迟 | `SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')` |
| **all-mpnet-base-v2** | 768 | **高质量**，英语通用 | `SentenceTransformer('sentence-transformers/all-mpnet-base-v2')` |
| **BAAI/bge-m3** | 1024 | **多语言+长上下文**，支持100+语言，8k token | `SentenceTransformer('BAAI/bge-m3')` |
| **intfloat/multilingual-e5-base** | 768 | **多语言**，语义搜索 | `SentenceTransformer('intfloat/multilingual-e5-base')` |

---

## 三、实施流程

### Step 1: 加载模型
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
```

### Step 2: Chunk文本
```python
from haystack import Document
docs = [Document(content=chunk) for chunk in chunked_text_list]
```

### Step 3: 生成嵌入
```python
from haystack.components.embedders import SentenceTransformersDocumentEmbedder

doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
docs_with_embeddings = doc_embedder.run(docs)
```

### Step 4: 存储到向量数据库
```python
from haystack.document_stores.in_memory import InMemoryDocumentStore

document_store = InMemoryDocumentStore()
document_store.write_documents(docs_with_embeddings["documents"])
```

**可选数据库**：FAISS, Milvus, Weaviate, Pinecone

---

## 四、持续优化策略

### 1. 迭代优化Chunking
- 测试不同的chunk size和策略（固定vs语义）
- 根据实际检索效果调整

### 2. 使用元数据过滤
- 为chunks附加元数据：来源文档、页码、日期、作者
- 支持高级过滤："仅搜索2025年发布的文档"

### 3. 混合搜索（Hybrid Search）
- **组合**：向量嵌入的语义搜索 + 传统关键词匹配（BM25）
- **重排序**：使用reranker模型对结果重新排序
- **优势**：兼顾语义理解和精确关键词匹配

---

## 五、关键要点总结

1. **Chunking是质量基础**：策略选择直接影响检索效果
2. **模型选择需权衡**：速度vs质量vs多语言支持
3. **测试是必须的**：MTEB分数仅供参考，真实数据测试才是关键
4. **元数据增强检索**：来源、时间等信息大幅提升实用性
5. **混合搜索是进阶**：语义+关键词+重排序=最佳效果
