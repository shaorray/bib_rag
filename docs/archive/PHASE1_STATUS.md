# Phase 1: Parent/Child Hierarchical Indexing — Implementation Status

## ✅ Completed Files

### 1. `build_hierarchical.py` — CPU Version
- Section-based parent chunking (ABSTRACT, INTRODUCTION, RESULTS, etc.)
- Small child chunks (500 chars, 100 overlap) for precise search
- JSON parent store in `parent_store/`
- ChromaDB child chunks with `parent_id` metadata
- Checkpoint/resume support
- ~5-10 papers/minute (CPU bge-m3)

### 2. `build_hierarchical_gpu.py` — GPU Version  
- Same functionality as CPU version
- GPU batch embedding via llama-server (port 8081)
- 64 chunks per batch
- ~50-100 papers/minute (estimated)

### 3. `src/parent_store_manager.py` — Parent Store Manager
- `load_content(parent_id)` — load single parent
- `load_content_many(parent_ids)` — batch load
- `load_by_source(source)` — all parents for a paper
- `get_stats()` — store statistics

### 4. `src/agent_tools.py` — Agent Tools
- `search_child_chunks(query, limit)` — semantic search in ChromaDB
- `retrieve_parent_chunks(parent_id)` — load full parent from JSON
- `retrieve_many_parents(parent_ids)` — batch parent retrieval
- `embed_query(text)` — query embedding via llama-server

## 🏗️ Architecture

```
User Query
    ↓
search_child_chunks(query, 5) → Top-K child chunks (with parent_id)
    ↓
retrieve_parent_chunks(parent_id) → Full parent context
    ↓
LLM Synthesis
```

| Component | Size | Storage | Purpose |
|-----------|------|---------|---------|
| Parent chunks | Section-level (200-4000 chars) | JSON in `parent_store/` | Rich context |
| Child chunks | 500 chars, 100 overlap | ChromaDB with `parent_id` | Precise search |

## 📊 Test Results

- **Parent store**: 78 JSON files created (from interrupted CPU run)
- **ChromaDB**: 32,741 total chunks (14,502 with parent_id from new build + 18,239 from old build)
- **Sample parent**: Adams et al. 2018, 6161 words, full_text section

## ⚠️ Known Issues

1. **Mixed ChromaDB**: Old chunks (from `build_stable.py`) don't have `parent_id`. Need `--rebuild` for clean hierarchical index.
2. **CPU Speed**: `build_hierarchical.py` is very slow (~20 min for 3 papers). Use `build_hierarchical_gpu.py` instead.
3. **Batch Size**: llama-server needs `--ubatch-size 8192` to avoid HTTP 500.

## 🔄 Next Steps (Phase 2)

1. Run `build_hierarchical_gpu.py --rebuild` for clean index
2. Create `src/agentic_graph.py` — LangGraph workflow
3. Create `src/agent_nodes.py` — all node functions
4. Create `src/agent_edges.py` — routing logic

## 📝 Usage

```bash
# Build hierarchical index (GPU recommended)
cd /Disk_bot/Eph/bib_rag
python3 -B build_hierarchical_gpu.py --rebuild

# Test agent tools
python3 -B src/agent_tools.py

# Check parent store stats
python3 -c "from src.parent_store_manager import ParentStoreManager; print(ParentStoreManager().get_stats())"
```
