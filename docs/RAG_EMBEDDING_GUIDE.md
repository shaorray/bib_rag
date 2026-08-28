> **Status note (2026-08-27):** chunking/embedding concepts still apply. The sentence_transformers/haystack examples are educational — this toolkit actually embeds via the bge-m3 llama-server on port 8081 (see src/index_single_paper.py).

# RAG Embedding Guide

## Core concepts

High-quality RAG embeddings come down to two steps: **chunking** and **embedding**.

---

## 1. Chunking strategies (simple → advanced)

### 1. Fixed-size chunking
- **Method**: split on a fixed character/token count (e.g. 512 tokens)
- **Pros**: fast, predictable
- **Cons**: can cut sentences or semantics mid-thought, losing information
- **Tools**: LangChain `RecursiveCharacterTextSplitter`

### 2. Semantic chunking
- **Method**: split on semantic similarity instead of fixed length
- **Implementation**: embed sentence by sentence, group sentences that are semantically close
- **Pros**: preserves natural semantic flow, better retrieval quality

### 3. Sliding window
- **Method**: create overlapping chunks
- **Example**: chunk size 500 words, next chunk starts at word 300 (200-word overlap)
- **Pros**: keeps cross-chunk context

### 📌 Practical advice
- **Starting point**: 512 tokens with 10–15% overlap
- **Complex documents**: up to 1024 tokens
- **Simple Q&A**: 64–128 tokens often works better
- **Key**: tune against your actual data — always test

---

## 2. Embedding model selection

### Key factors

| Factor | What it means | Advice |
|------|------|------|
| Context window | how much text the model processes at once | chunk size must not exceed it |
| Embedding dimensions | vector length (384/768/1536...) | 768–1024 is a good balance |
| Language support | multilingual coverage | use multilingual-e5 or BGE-M3 for multilingual corpora |
| Performance | MTEB leaderboard scores | use as reference; validate on real data |

### Recommended open models

| Model | Dims | Best for | Code |
|------|------|----------|------|
| **all-MiniLM-L6-v2** | 384 | **speed first**, low-latency production | `SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')` |
| **all-mpnet-base-v2** | 768 | **high quality**, general English | `SentenceTransformer('sentence-transformers/all-mpnet-base-v2')` |
| **BAAI/bge-m3** | 1024 | **multilingual + long context**, 100+ languages, 8k tokens | `SentenceTransformer('BAAI/bge-m3')` |
| **intfloat/multilingual-e5-base** | 768 | **multilingual**, semantic search | `SentenceTransformer('intfloat/multilingual-e5-base')` |

---

## 3. Implementation walkthrough

### Step 1: load the model
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
```

### Step 2: chunk the text
```python
from haystack import Document
docs = [Document(content=chunk) for chunk in chunked_text_list]
```

### Step 3: generate embeddings
```python
from haystack.components.embedders import SentenceTransformersDocumentEmbedder

doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
docs_with_embeddings = doc_embedder.run(docs)
```

### Step 4: store in a vector database
```python
from haystack.document_stores.in_memory import InMemoryDocumentStore

document_store = InMemoryDocumentStore()
document_store.write_documents(docs_with_embeddings["documents"])
```

**Alternative stores**: FAISS, Milvus, Weaviate, Pinecone

---

## 4. Ongoing optimization

### 1. Iterate on chunking
- Test different chunk sizes and strategies (fixed vs semantic)
- Adjust based on actual retrieval quality

### 2. Metadata filtering
- Attach metadata to chunks: source document, page, date, author
- Enables advanced filters like "only documents published in 2025"

### 3. Hybrid search
- **Combine**: semantic search over vector embeddings + classic keyword matching (BM25)
- **Rerank**: reorder results with a reranker model
- **Benefit**: semantic understanding plus exact keyword precision

---

## 5. Takeaways

1. **Chunking is the quality foundation** — the strategy directly determines retrieval quality
2. **Model choice is a trade-off** — speed vs quality vs multilingual support
3. **Testing is mandatory** — MTEB scores are reference only; real-data validation is what counts
4. **Metadata boosts retrieval** — provenance, time, and similar fields add a lot of practical value
5. **Hybrid search is the advanced play** — semantic + keyword + reranking = best results