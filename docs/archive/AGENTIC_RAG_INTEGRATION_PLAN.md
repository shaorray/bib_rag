> **Status note (2026-08-27):** this plan WAS implemented — agentic_query.py, src/agent_tools.py and the LangGraph pipeline exist as described. Read as design rationale, not a todo list.

# Agentic RAG Integration Plan for bib_rag

Based on: `/Disk_bot/github_repos/agentic-rag-for-dummies`

## 4 Features to Add

### 1. LangGraph Workflow + Context Compression

**What it does:**
- LangGraph state machine orchestrates the full RAG pipeline
- Summarize conversation history → Rewrite query → [Clarification?] → Agent research → Aggregate answers
- Agent subgraph: Orchestrator → Search tools → [Compress context?] → Orchestrator (loop) → Collect answer
- Context compression: LLM summarizes retrieved data when token count exceeds threshold

**Integration into bib_rag:**
- Create `src/agentic_graph.py` — main LangGraph workflow
- Create `src/agent_nodes.py` — node functions (summarize, rewrite, orchestrate, compress)
- Create `src/agent_edges.py` — routing logic (clarification, parallel agents)
- Use existing Qwen3.6-35B LLM (port 5015) for all LLM calls
- Reuse existing ChromaDB (1,643 papers) as vector store

### 2. Parent/Child Hierarchical Indexing

**What it does:**
- **Parent chunks**: Large sections based on Markdown headers (H1, H2, H3) → rich context
- **Child chunks**: Small, fixed-size pieces (500 chars, 100 overlap) → precise search
- Search child chunks → retrieve parent chunks for full context

**Integration into bib_rag:**
- Modify `build_stable.py` to create **two-level chunks**:
  1. Split by headers → parent chunks (merge small, split large)
  2. Split parents into child chunks
- Store children in ChromaDB with `parent_id` metadata
- Store parents as JSON in `parent_store/` directory
- Add `search_child_chunks` and `retrieve_parent_chunks` tools

### 3. Multi-Agent Map-Reduce

**What it does:**
- Rewrite multi-part queries into separate sub-queries
- Spawn parallel agent subgraphs (one per sub-query)
- Each agent independently searches, compresses, answers
- Aggregate all responses into single coherent answer

**Integration into bib_rag:**
- `rewrite_query` node splits queries like "What is Eph? What is ephrin?" → ["What is Eph receptor?", "What is ephrin ligand?"]
- `route_after_rewrite` uses `langgraph.types.Send` to spawn parallel agents
- `aggregate_answers` node merges responses with LLM

### 4. Conversation Memory

**What it does:**
- Summarize chat history to 1-2 sentences
- Rewrite follow-up queries using conversation context
- Maintain state across turns with `InMemorySaver` checkpointer

**Integration into bib_rag:**
- `summarize_history` node: compress last 6 messages
- `rewrite_query` node: use summary for pronoun resolution ("How does it work?" → "How does Eph signaling work?")
- `InMemorySaver` as checkpointer for state persistence

---

## Implementation Phases

### Phase 1: Foundation (Parent/Child Indexing)
- [ ] Update `build_stable.py` to create hierarchical chunks
- [ ] Create `parent_store_manager.py` for JSON parent storage
- [ ] Rebuild index with parent_id metadata

### Phase 2: LangGraph Core
- [ ] Install langgraph: `pip install langgraph`
- [ ] Create `src/agentic_graph.py` with main graph
- [ ] Create `src/agent_nodes.py` with all node functions
- [ ] Create `src/agent_edges.py` with routing logic
- [ ] Create `src/agent_prompts.py` with all system prompts

### Phase 3: Tools + Compression
- [ ] Create `search_child_chunks` tool (hybrid search in ChromaDB)
- [ ] Create `retrieve_parent_chunks` tool (load from JSON)
- [ ] Implement `compress_context` node with token threshold

### Phase 4: Map-Reduce + Memory
- [ ] Implement `rewrite_query` with query splitting
- [ ] Implement `route_after_rewrite` with `Send()` parallel dispatch
- [ ] Implement `aggregate_answers` with LLM merging
- [ ] Implement `summarize_history` with conversation compression

### Phase 5: Integration + CLI
- [ ] Create `agentic_query.py` — CLI entry point
- [ ] Update README with new workflow
- [ ] Test end-to-end with sample queries

---

## File Structure

```
bib_rag/
├── src/
│   ├── build_stable.py          # Updated: hierarchical indexing
│   ├── agentic_graph.py          # NEW: LangGraph workflow
│   ├── agent_nodes.py            # NEW: all node functions
│   ├── agent_edges.py            # NEW: routing logic
│   ├── agent_prompts.py          # NEW: system prompts
│   ├── agent_tools.py            # NEW: search/retrieve tools
│   └── parent_store_manager.py   # NEW: parent chunk JSON storage
├── parent_store/                 # NEW: parent chunk JSON files
├── agentic_query.py            # NEW: CLI entry point
└── docs/
    └── AGENTIC_RAG_INTEGRATION_PLAN.md  # This file
```

## Key Differences from Reference Repo

| Aspect | Reference (agentic-rag-for-dummies) | bib_rag Integration |
|--------|-------------------------------------|---------------------|
| Vector DB | Qdrant (hybrid dense+sparse) | ChromaDB (dense only, bge-m3) |
| Embeddings | all-mpnet-base-v2 + BM25 | bge-m3 (1024-dim) |
| LLM | qwen3:4b via Ollama | Qwen3.6-35B via llama-server |
| Documents | General PDFs → Markdown | Academic papers (Eph/ephrin) |
| Chunking | Markdown headers | Paragraph-level + headers |
| Scale | Small demo | 1,643 papers, 18K chunks |

## Dependencies to Add

```
langgraph>=0.2.0
langchain-core>=0.3.0
```

## Next Steps

1. Start with **Phase 1** (Parent/Child indexing) — requires rebuilding the index
2. Then **Phase 2** (LangGraph core) — can test with existing index
3. Then **Phase 3** (Tools + Compression) — enables agent loop
4. Finally **Phase 4** (Map-Reduce + Memory) — full conversational RAG
