# bib_rag

Academic bibliography RAG system for evidence-based writing.

## What's New: Agentic RAG (v2.0)

Full LangGraph-powered agentic pipeline with hierarchical retrieval, context compression, and conversation memory.

### Quick Start — Agentic Query

```bash
cd bib_rag

# Single query (default: fast cloud model)
/usr/bin/python3.10 -B agentic_query.py "What is the role of Eph receptors in neural development?"

# Interactive chat mode
/usr/bin/python3.10 -B agentic_query.py --interactive

# Verbose mode (shows pipeline progress)
/usr/bin/python3.10 -B agentic_query.py "Compare EphA and EphB functions" --verbose
```

> **Note**: use `/usr/bin/python3.10` (the interpreter with langchain deps
> installed), not bare `python3`.

### Model Backend — Local vs Cloud

Two backends, chosen via `LLM_URL` / `LLM_MODEL`:

- **Cloud (default)**: `glm-5.2:cloud` through a local OpenAI-compatible
  gateway (port 11434). Fast, but needs network and may incur cost.
- **Local**: a GGUF file served by llama-server (port 5015), e.g. Qwen3.8-27B.
  Fully offline and private, but slower — so it gets a smaller per-query
  budget (3 iterations / 4 tool calls vs 10/8).

The agent auto-detects the backend and adjusts the budget; override with
`AGENT_MAX_ITERATIONS` / `AGENT_MAX_TOOL_CALLS`.

```bash
# Cloud (default)
/usr/bin/python3.10 -B agentic_query.py "your question"

# Local model (offline)
LLM_URL=http://localhost:5015/v1 \
LLM_MODEL=/path/to/model.gguf \
/usr/bin/python3.10 -B agentic_query.py "your question"
```

| Env | Default | Meaning |
|-----|---------|---------|
| `LLM_URL` | `http://localhost:11434/v1` | OpenAI-compatible gateway (cloud) |
| `LLM_MODEL` | `glm-5.2:cloud` | Cloud model name, or local `.gguf` path |
| `LLM_API_KEY` | `not-required` | placeholder — the gateway doesn't validate it |

> Note: pure reasoning models (e.g. `kimi-k3:cloud`) fail at function calling
> through the gateway — use `glm-5.2:cloud` or `deepseek-v4-flash:cloud`.

**Prerequisites**: Embedding server must be running.
- Embeddings: bge-m3 on port 8081
- LLM: the cloud gateway (port 11434) **or** a local GGUF (port 5015)

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
/usr/bin/python3.10 -B src/query_bib_rag.py "your keywords"
```

### Find Citations
```bash
/usr/bin/python3.10 -B src/query_bib_rag.py --cite "a claim to find supporting papers for" --top 3
```

### Write Paragraph with Citations
```bash
/usr/bin/python3.10 -B src/bib_rag_writer.py "your topic sentence" \
  --top 5 --style APA --output /path/to/output.odt
```

---

## Adding New Papers

Put new PDFs in your paper library and run:

```bash
cd bib_rag
/usr/bin/python3.10 -B add_papers.py /path/to/new/pdfs/
```

This will:
1. Extract PDFs to Markdown (pymupdf4llm)
2. Copy PDFs to the library
3. Run hierarchical build to index new papers
4. Resume from checkpoint automatically

**Options:**
```bash
/usr/bin/python3.10 -B add_papers.py /path/to/pdfs/ --skip-extract   # if already markdown
/usr/bin/python3.10 -B add_papers.py /path/to/pdfs/ --batch-size 5   # smaller batches
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
├── add_papers.py              ← Add new PDFs to the index (entry point)
├── agentic_query.py           ← Query the agentic RAG system (entry point)
├── CONTEXT.md                 ← Domain glossary (exact terminology)
│
├── scripts/                   ← Utilities & pipelines
│   ├── meta_audit.py          ← Metadata proof-reader + genuine-info fetcher
│   ├── bib_to_parent_store.py · fill_meta_key_in_parent_store.py ← fill meta from My Library.bib
│   ├── bib_utils.py           ← Shared normalization / .bib / filename helpers
│   ├── zotero_access.py       ← Zotero access layer (MCP first, HTTP fallback)
│   ├── test_utilities.py      ← Regression tests (no network)
│   ├── classify_all.py · classify_demo.py · apply_tags.py · migrate_topics.py ← LLM classification pipeline
│   └── classify_cadherin*.py · backfill_cadherin.py · batch_index_cadherin.py · retry_index_cadherin.py ← Cadherin corpus ingestion
│
├── src/                       ← RAG library & query/writer tools
│   ├── agentic_graph.py · agent_nodes.py · agent_edges.py ← LangGraph pipeline
│   ├── agent_prompts.py · agent_schemas.py · agent_tools.py ← prompts / state / retrieval tools
│   ├── build_hierarchical_gpu.py · build_hierarchical.py · index_single_paper.py · chunking.py ← index build
│   ├── query_bib_rag.py       ← Quick semantic search & citations
│   ├── bib_rag_writer.py · bib_rag_writer_debate.py · bib_rag_grill.py ← writers with citations
│   ├── parent_store_manager.py ← Parent chunk loader
│   └── evaluate.py · test_comprehensive.py · test_agentic_graph.py ← eval & tests
│
├── docs/                      ← Guides (QUICK_START, USAGE, ZOTERO_MCP_USAGE, …)
├── data/                      ← Checkpoints & audit reports (gitignored)
├── outputs/                   ← Generated reports (gitignored)
├── chroma_db_new/             ← Vector database (gitignored)
├── parent_store/              ← Parent chunks, JSON (gitignored)
└── archive/                   ← Obsolete experiments (gitignored)
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
/usr/bin/python3.10 -B src/build_hierarchical_gpu.py --rebuild --batch-size 10

# Resume interrupted build
/usr/bin/python3.10 -B src/build_hierarchical_gpu.py --batch-size 10

# CPU fallback (slower)
/usr/bin/python3.10 -B src/build_hierarchical.py
```

---

## Testing

```bash
# Comprehensive test suite (5 queries, ~5 minutes)
cd bib_rag
/usr/bin/python3.10 -B src/test_comprehensive.py

# Single test query
/usr/bin/python3.10 -B src/test_agentic_graph.py

# Evaluation: agentic vs baseline
/usr/bin/python3.10 -B src/evaluate.py
```

---

## License

MIT
