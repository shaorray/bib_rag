# Folder Cleanup Report

## State Before Cleanup

Over 60 files were originally scattered in the root directory, including:
- Multiple script versions (v1, v2, v3)
- A large number of test files (test_*.py)
- Debug scripts (debug_*.py)
- Outdated docs (PHASE1*, OPTIMIZATION*)
- Experimental code (LangGraph-related)
- Python cache files (__pycache__)
- Log files (metrics.log)

## Structure After Cleanup

```
ephrin_agentic_rag/
├── chroma_db/          # V1 knowledge base (legacy)
├── chroma_db_v2/       # V2 knowledge base
├── chroma_db_v3/       # V3 knowledge base (recommended)
├── src/                # Source code
│   ├── agents/         # Agentic RAG components
│   ├── queries/        # Query scripts
│   ├── utils/          # Utility scripts
│   └── *.py            # Core processing scripts
├── docs/               # Documentation
├── data/               # Configuration files
└── archive/            # Archived files
    ├── old_scripts/    # Old scripts
    ├── test_files/     # Test files
    ├── documents/      # Old docs
    └── experimental/   # Experimental code
```

## Core Files Kept

### Processing scripts (src/)
- `src/process_v3_papers.py` - V3 processing (recommended)
- `src/process_v2_papers.py` - V2 processing
- `src/build_knowledge_base.py` - Build the knowledge base
- `src/add_new_papers.py` - Add new papers

### Query scripts (src/queries/)
- `src/queries/query_v3_kb.py` - V3 queries (recommended)
- `src/queries/query_v2_kb.py` - V2 queries
- `src/queries/hybrid_search.py` - Hybrid search
- `src/queries/quick_query.py` - Quick queries

### Agent components (src/agents/)
- `src/agents/agentic_workflow.py` - Main workflow
- `src/agents/rag_core.py` - Core logic
- `src/agents/self_rag.py` - Self-RAG
- `src/agents/multi_hop_rag.py` - Multi-hop RAG

### Academic writing (src/utils/)
- `src/utils/academic_writer.py` - Writing assistant
- `src/utils/citation_manager.py` - Citation management

### Docs (docs/)
- `docs/README.md` - Main doc
- `docs/USAGE.md` - Usage guide
- `docs/QUICK_START.md` - Quick start
- `docs/PMID_CITATION_GUIDE.md` - PMID citation guide
- `docs/RAG_EMBEDDING_GUIDE.md` - Embedding guide
- `docs/V3_IMPROVEMENT_REPORT.md` - V3 improvement report

## Archived Files

### Old scripts (archive/old_scripts/)
- demo.py, debug_retry.py, debug_routing.py
- analyze_kb.py, langfuse_monitor.py
- redis_cache.py

### Test files (archive/test_files/)
- test_*.py (10 test scripts)
- verify_improvements.py

### Old docs (archive/documents/)
- DEPLOYMENT_REPORT.md
- OPTIMIZATION_*.md (3 files)
- PERFORMANCE_EVALUATION.md
- PHASE1_*.md (3 files)
- STATUS.md, TEST_REPORT.md
- VERSION_COMPARISON.md
- OPENDATALOADER_*.md (2 files)

### Experimental code (archive/experimental/)
- langgraph_agentic_rag.py
- langgraph_phase1.py

## Deleted Files

- `metrics.log` - Log file
- `query_history.json` - Query history
- `__pycache__/` - Python cache

## Issues Fixed

1. **Import paths**: Updated sys.path for all moved files
2. **Clear structure**: Organized by function
3. **Version management**: Clearly separated V1/V2/V3
4. **Doc housekeeping**: Kept the latest docs, archived old ones

## Usage

### Recommended commands (V3)
```bash
# Query
python3 src/queries/query_v3_kb.py "cis interaction" -n 5

# With cited paragraphs
python3 src/queries/query_v3_kb.py "Eph signaling" -n 3 --paragraph

# Hybrid search
python3 src/queries/hybrid_search.py "axon guidance"
```

### If you need to restore old files
```bash
# Restore from archive
cp archive/old_scripts/demo.py .
cp archive/documents/STATUS.md docs/
```

## File Count Comparison

| Item | Before | After |
|------|--------|-------|
| Root-directory files | 60+ | 1 (README.md) |
| Source code | Scattered | Organized under src/ |
| Docs | Scattered | Unified under docs/ |
| Archived files | 0 | 4 subdirectories under archive/ |