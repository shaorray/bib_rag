# V3 Knowledge Base Improvement Report

## 1. Section Classification Optimization ✅

### Problem
- Original: "other" accounted for 81.4%, with core sections at only 18.6%
- Cause: Inconsistent section title formats and missing mapping rules

### Improvements
1. **Expanded SECTION_MAP**: Added 40+ section mapping rules
2. **Added keyword detection**:
   - Methods keywords: antibodies, plasmids, cell culture, western blot, etc.
   - Results keywords: figure, table, supplementary, etc.
3. **Expanded SKIP_SECTIONS**: Filtered non-academic content such as "Open in a new tab" and "Author Manuscript"

### Results
```
Section distribution comparison:

Before improvement (initial V3):
  other:         81.4%
  Core sections: 18.6%

After improvement (final V3):
  other:         73.4%  ↓8%
  methods:        8.2%  ↑7.5%
  discussion:     6.8%  ↑3.5%
  abstract:       4.1%  ↑0.9%
  introduction:   4.0%  ↑1.4%
  results:        1.8%  ↑0.9%
  Core sections: 26.6%  ↑8%
```

## 2. Metadata-Filtered Queries ✅

### Implemented Features
```python
# Filter by year
results = kb.query("cis interaction", year_min=2020)

# Filter by impact factor
results = kb.query("signaling", min_if=10.0)

# Filter by section
results = kb.query("binding", section="results")

# Combined filters
results = kb.query("EphA4", year_min=2015, min_if=5.0, journal="Nature")
```

### Supported Fields
- year_min/year_max: Year range
- journal: Journal name (partial match)
- min_if: Minimum impact factor
- section: Section type
- tier: Journal tier

## 3. Hybrid Search ✅

### How It Works
```
Hybrid search = semantic search × 0.6 + BM25 keyword search × 0.4
```

### Implementation
```python
class HybridSearch:
    def search(self, query, semantic_weight=0.6, bm25_weight=0.4):
        # 1. Semantic search (768-dim vector similarity)
        semantic_scores = np.dot(embeddings, query_embedding)
        
        # 2. BM25 search (keyword matching)
        bm25_scores = bm25_index.search(query)
        
        # 3. Combined score
        combined = semantic_weight * semantic_scores + bm25_weight * bm25_scores
        
        return top_k_results
```

### Results Comparison
| Query | Semantic only | Hybrid search | Gain |
|------|--------|----------|------|
| "Eph receptor signaling" | 0.97 | 0.98 | +0.01 |
| "cis interaction" | 0.40 | 0.45 | +0.05 |
| "ephrin binding" | 0.73 | 0.78 | +0.05 |

## 4. New Files

| File | Purpose |
|------|------|
| `process_v3_papers.py` | V3 processing script (all-mpnet-base-v2) |
| `query_v3_kb.py` | V3 query script (supports metadata filters) |
| `hybrid_search.py` | Hybrid search implementation (semantic + BM25) |
| `chroma_db_v3/ephrin_papers_v3.pkl` | V3 knowledge base data |

## 5. Tech Stack

- **Embedding model**: all-mpnet-base-v2 (768 dims)
- **Chunking strategy**: 800 words per chunk, 200-word overlap
- **Deduplication**: Hash-based dedup on the first 300 characters of text
- **Indexing**: Vector inner product + BM25
- **Filters**: Year / journal / IF / section / tier

## 6. Remaining Optimization Items

1. **Section classification**: Further reduce the "other" proportion
2. **Reranker**: Add cross-encoder reranking
3. **Incremental updates**: Support adding new papers without rebuilding
4. **API wrapper**: Provide a RESTful API interface