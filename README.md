# bib_rag

Academic bibliography RAG system for evidence-based writing.

## What's New: Agentic RAG (v2.0)

Full LangGraph-powered agentic pipeline with hierarchical retrieval, context compression, and conversation memory.

### Quick Start — Agentic Query

```bash
cd bib_rag

# Single query
python3 -B agentic_query.py "What is the role of Eph receptors in neural development?"

# Interactive chat mode
python3 -B agentic_query.py --interactive

# Verbose mode (shows pipeline progress)
python3 -B agentic_query.py "Compare EphA and EphB functions" --verbose
```

**Prerequisites**: Both servers must be running
- LLM: Qwen3.6-35B on port 5015
- Embeddings: bge-m3 on port 8081

### Architecture

```
User Query
  ↓
summarize_history → rewrite_query → [clarification?]
  ↓
Agent Subgraphs (parallel for multi-part queries)
  orchestrator → search_child_chunks → retrieve_parent_chunks → [compress?]
  ↓
aggregate_answers → Final Answer with Sources
```

**Key features:**
- **Hierarchical retrieval**: Search small child chunks → retrieve full parent context
- **Context compression**: LLM summarizes retrieved data when token threshold exceeded
- **Map-Reduce**: Multi-part queries split into parallel agents
- **Conversation memory**: Follow-up queries resolved from prior context
- **Source attribution**: Every answer cites specific papers

---

## Classic RAG Tools (Still Available)

### Quick Search
```bash
python3 -B src/query_bib_rag.py "cis interaction mechanism"
```

### Find Citations
```bash
python3 -B src/query_bib_rag.py --cite "Eph receptors promote tumor suppression" --top 3
```

### Write Paragraph with Citations
```bash
python3 -B src/bib_rag_writer.py "Eph receptor signaling regulates cell segregation" \
  --top 5 --style APA --output /path/to/output.odt
```

---

## Adding New Papers

Put new PDFs in your paper library and run:

```bash
cd bib_rag
python3 -B add_papers.py /path/to/new/pdfs/
```

This will:
1. Extract PDFs to Markdown (pymupdf4llm)
2. Copy PDFs to the library
3. Run hierarchical build to index new papers
4. Resume from checkpoint automatically

**Options:**
```bash
python3 -B add_papers.py /path/to/pdfs/ --skip-extract   # if already markdown
python3 -B add_papers.py /path/to/pdfs/ --batch-size 5   # smaller batches
```

**Prerequisite:** Embedding server must be running on port 8081.

---

## Zotero Access (MCP)

bib_rag reaches your Zotero library through the **Zotero MCP server**
([54yyyu/zotero-mcp](https://github.com/54yyyu/zotero-mcp),
`zotero-mcp-server` on PyPI):

```bash
uv tool install zotero-mcp-server        # provides `zotero-mcp` + `zotero-cli`
```

Zotero desktop must be running with the local API enabled
(Settings → Advanced → *Allow other applications on this computer to
communicate with Zotero*). No API key is needed for local read access.

The access layer (`scripts/zotero_access.py`) tries, in order:

1. **Zotero MCP server** — `zotero-mcp serve --transport stdio`, queried over
   MCP (`zotero_search_items`, `zotero_get_item_metadata` with `format=json`)
2. **Zotero local HTTP API** (`http://localhost:23119`) — dependency-free fallback
3. Graceful `None`/`[]` when Zotero is unreachable — consumers never crash

Used by:

- `scripts/meta_audit.py` — Zotero as one of the corroboration sources for
  metadata proof-reading
- `src/bib_rag_writer.py` / `src/bib_rag_writer_debate.py` — real author /
  volume / issue / pages / DOI for APA & Vancouver citations

Env knobs: `BIB_RAG_ZOTERO_MCP=0` forces the HTTP path;
`BIB_RAG_ZOTERO_URL=...` overrides the local HTTP API base.

---

## File Structure

```
bib_rag/
├── add_papers.py              ← Add new PDFs to index
├── agentic_query.py           ← Query the agentic RAG system
│
├── src/
│   ├── build_hierarchical_gpu.py  ← GPU build script
│   ├── build_hierarchical.py     ← CPU build fallback
│   ├── query_bib_rag.py           ← Quick semantic search
│   ├── bib_rag_writer.py          ← Paragraph synthesis with citations
│   ├── bib_rag_writer_debate.py   ← LLM debate synthesis
│   ├── agentic_graph.py           ← Main LangGraph workflow
│   ├── agent_nodes.py             ← 9 agent node functions
│   ├── agent_edges.py             ← Routing logic
│   ├── agent_prompts.py           ← System prompts
│   ├── agent_schemas.py           ← State definitions
│   ├── agent_tools.py             ← Hierarchical search/retrieve tools
│   ├── parent_store_manager.py    ← JSON parent chunk loader
│   ├── evaluate.py                ← Agentic vs baseline evaluation
│   └── test_comprehensive.py      ← 5-test suite
│
├── chroma_db_new/               ← Vector database (ChromaDB)
├── parent_store/                ← Parent chunks (JSON)
└── data/
    ├── build_hierarchical_checkpoint.json
    └── incremental_metadata.json
```

---

## Server Setup

```bash
# Start embedding server (bge-m3)
bash start_llama_bge_m3.sh

# Start LLM server
bash llama-server_run.sh

# Both should respond:
curl http://localhost:8081/health   # embeddings
curl http://localhost:5015/health   # LLM
```

---

## Build Index (First Time)

```bash
cd bib_rag

# GPU build (recommended)
python3 -B src/build_hierarchical_gpu.py --rebuild --batch-size 10

# Resume interrupted build
python3 -B src/build_hierarchical_gpu.py --batch-size 10

# CPU fallback (slower)
python3 -B src/build_hierarchical.py
```

---

## Testing

```bash
# Comprehensive test suite (5 queries, ~5 minutes)
cd bib_rag
python3 -B src/test_comprehensive.py

# Single test query
python3 -B src/test_agentic_graph.py

# Evaluation: agentic vs baseline
python3 -B src/evaluate.py
```

---

## License

MIT
