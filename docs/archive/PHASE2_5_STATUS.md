> **Status note (2026-08-27):** architecture table and module list below reflect the current toolkit (ChromaDB `chroma_db_new/`, collection `bib_rag_papers`, parent_store, build_hierarchical_gpu). Some chunk/paper counts are historical snapshots — query the store for current numbers.

# Phase 1-5: Complete Agentic RAG Implementation Status

## ✅ Phase 1: Foundation (Parent/Child Hierarchical Indexing) — COMPLETE

### Files
- `build_hierarchical_gpu.py` — GPU-accelerated hierarchical build (batch_size=10, embed_batch=16)
- `build_hierarchical.py` — CPU fallback version
- `src/parent_store_manager.py` — Parent store manager (load, batch_load, stats)
- `data/build_hierarchical_checkpoint.json` — Resume checkpoint (1,643 papers)

### Results
- **1,643 papers** fully processed
- **3,248 parent chunks** in `parent_store/` (section-level, 200-4,000 chars)
- **312,173 child embeddings** in ChromaDB (500 chars, 100 overlap)
- All children have `parent_id` metadata for parent retrieval

---

## ✅ Phase 2: LangGraph Core — COMPLETE

### Files
- `src/agentic_graph.py` — Main LangGraph workflow (two-level StateGraph)
- `src/agent_nodes.py` — 9 node functions (summarize, rewrite, orchestrate, compress, etc.)
- `src/agent_edges.py` — Routing logic (route_after_rewrite, route_after_orchestrator_call)
- `src/agent_prompts.py` — 6 system prompts for academic domain
- `src/agent_schemas.py` — TypedDict state definitions + QueryAnalysis Pydantic model

### Architecture
```
Main Graph:
  START → summarize_history → rewrite_query → [clarification?] → agent subgraphs → aggregate_answers → END

Agent Subgraph (per sub-query):
  START → orchestrator → tools → [compress_context? → orchestrator loop] → collect_answer → END
```

### Key Components
- **InMemorySaver** checkpointer for state persistence
- **Interrupt** before `request_clarification` for user feedback
- **MAX_ITERATIONS** = 5, **MAX_TOOL_CALLS** = 7

---

## ✅ Phase 3: Tools + Compression — COMPLETE

### Files
- `src/agent_tools.py` — LangChain @tool-decorated functions
  - `search_child_chunks(query, limit)` — Semantic search via bge-m3 (port 8081)
  - `retrieve_parent_chunks(parent_id)` — Load full parent from JSON store
  - `retrieve_many_parents(parent_ids)` — Batch parent retrieval

### Context Compression
- `should_compress_context()` — Token threshold check (BASE_TOKEN_THRESHOLD + growth factor)
- `compress_context()` — LLM summarizes conversation into structured research context
- Prevents duplicate searches/parent retrievals via `retrieval_keys` tracking

---

## ✅ Phase 4: Map-Reduce + Memory — COMPLETE

### Map-Reduce
- `rewrite_query()` — Splits multi-part queries into separate sub-queries (max 3)
- `route_after_rewrite()` — Spawns parallel agent subgraphs via `langgraph.types.Send`
- `aggregate_answers()` — Merges multiple agent responses with LLM synthesis

### Conversation Memory
- `summarize_history()` — Compresses last 6 messages into 1-2 sentence summary
- `rewrite_query()` — Uses summary for pronoun resolution ("How does it work?" → "How does Eph signaling work?")
- State persists across turns via `InMemorySaver`

---

## ✅ Phase 5: Integration + CLI — COMPLETE

### Files
- `agentic_query.py` — CLI entry point

### Usage
```bash
# Single query
python3 -B agentic_query.py "What is the role of Eph receptors in neural development?"

# Verbose mode
python3 -B agentic_query.py "Compare EphA and EphB functions" --verbose

# Interactive chat
python3 -B agentic_query.py --interactive
```

---

## 🏗️ System Architecture

| Component | Technology | Details |
|-----------|-----------|---------|
| **LLM** | Qwen3.6-35B | llama-server port 5015, OpenAI-compatible API |
| **Embeddings** | bge-m3 (1024-dim) | llama-server port 8081, GPU-accelerated |
| **Vector DB** | ChromaDB | `chroma_db_new/`, collection `bib_rag_papers` |
| **Parent Store** | JSON files | `parent_store/`, section-level chunks |
| **Graph** | LangGraph | Two-level StateGraph with checkpointer |
| **Domain** | Academic | Eph receptors, ephrins, development, cancer |

---

## 📊 Test Results

### Query 1: "What is the role of Eph receptors in neural development?"
- **Answer**: 4,722 chars covering axon guidance, neural precursor differentiation, neurotransmitter regulation
- **Sources**: 6 papers cited with proper attribution
- **Pipeline**: summarize → rewrite → search → retrieve → compress → answer

### Query 2: "Compare EphA and EphB receptor functions in axon guidance and neural development"
- **Answer**: 5,354 chars with structured comparison table
- **Sources**: 8 papers cited
- **Pipeline**: multi-aspect search → parent retrieval → synthesis

### Query 3: "What is Eph? What is ephrin? How do they interact?"
- **Answer**: 6,939 chars with detailed molecular structure and signaling mechanisms
- **Pipeline**: Map-reduce (3 parallel agents) → aggregate answers

---

## 🔧 Infrastructure Fixes Applied

| Issue | Fix |
|-------|-----|
| Embedding server 500 errors | `--ubatch-size 512` (was 8192) |
| Memory leak (338 GB) | Same fix — ubatch-size too large |
| ChromaDB corruption | Fresh `--rebuild` with batch_size=10 |
| Log file bloat (15 GB) | `--log-file` removed, output → `/dev/null` |
| Build timeout/stuck | Reduced paper batch from 50 → 10 |
| LLM auth error | `api_key="not-needed"` for llama-server |

---

## 🚀 Next Steps (Optional)

1. **Deploy as API**: Wrap graph in FastAPI server
2. **Persistent checkpointer**: Replace InMemorySaver with SQLite/Postgres
3. **Streaming responses**: Use `graph.stream()` for real-time output
4. **Evaluation**: Benchmark against non-agentic baseline
5. **Fine-tuning**: Tune prompts for specific Eph/ephrin subtopics

---

*Status: All 5 phases complete. Agentic RAG pipeline is fully operational.*
*Last updated: 2026-06-02*
