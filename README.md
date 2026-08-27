# bib_rag

Academic bibliography RAG toolkit for evidence-based writing. One toolkit,
many libraries:

```
/Disk_bot/RAG/
├── bib_rag/   ← HERE: the toolkit (code only, this repo)
├── eph_rag/   ← Eph-ephrin library (default; collection bib_rag_papers, ~470K chunks)
├── geo_rag/   ← geology/renewables library
└── <name>_rag/ ← your libraries (create with scripts/setup_library.py)
```

> **New here?** Read [`GUIDE.md`](GUIDE.md). The maintained how-to procedures
> live in the Hermes skills `bib-rag-ingest` and `bib-rag-query`. Historical
> docs: `docs/archive/`.

---

## Quick Start

```bash
cd /Disk_bot/RAG/bib_rag

# 0. Prerequisites — embedding server (REQUIRED for index + query):
LD_LIBRARY_PATH=/Disk_2/llama.cpp/build/bin /Disk_2/llama.cpp/build/bin/llama-server \
  -m /Disk_bot/models/embeddings/bge-m3-Q4_K_M.gguf \
  --port 8081 -c 8192 -ngl 999 --embedding -ub 2048 &
curl http://localhost:8081/health          # → {"status":"ok"}

# 1. Create a library (one command: folders + manifest + registry + wrapper)
/usr/bin/python3.10 -B scripts/setup_library.py --name mytopic_rag \
    --domain "your domain" --wrapper yes --no-interactive

# 2. Index a paper
mytopic-rag src/index_single_paper.py /path/to/paper.md

# 3. Query it
mytopic-rag src/query_bib_rag.py "distinctive phrase from the paper"

# 4. Batch-add PDFs (extracts PDF→md, then indexes incrementally)
mytopic-rag add_papers.py /path/to/pdf/dir/

# 5. Delete a paper (dry-run by default)
mytopic-rag scripts/remove_paper.py paper_stem.md --apply

# 6. Agentic Q&A over the library
mytopic-rag agentic_query.py "What do my papers say about X?"
```

Existing libraries: use `eph-rag ...` or `geo-rag ...` instead of `mytopic-rag`.
(`bib-rag` is a deprecated alias for `eph-rag`.)

> **Python**: always `/usr/bin/python3.10 -B` — the wrappers set this plus the
> `PYTHONPATH` fix chromadb needs. Bare `python3` is a pyenv shim without deps.

## Building a RAG library

```bash
# Interactive — prompts for name/domain with defaults:
/usr/bin/python3.10 -B scripts/setup_library.py

# Scripted:
/usr/bin/python3.10 -B scripts/setup_library.py \
    --name neuro_rag \
    --domain "neuroscience / axon guidance" \
    --wrapper yes --no-interactive
```

This creates `/Disk_bot/RAG/neuro_rag/` (`chroma_db_new/`, `parent_store/`,
`data/`, `outputs/`, `md/`), writes `LIBRARY.md` + `CONTEXT.md`, registers it
in `src/kb_config.py`, and emits the `neuro-rag` command. Library names are
`snake_case` ending in `_rag`; the Chroma collection defaults to `<stem>_papers`.

Switch libraries any time — each wrapper pins its own library, or use
`BIB_RAG_KB_NAME=geo_rag <command>` per-call (never `export BIB_RAG_ROOT` — it
leaks across calls and silently writes into the wrong library).

## Indexing papers

```bash
# Single paper (chunks parent+child, embeds via 8081, stores into the active library)
eph-rag src/index_single_paper.py /path/to/paper.md

# First-time bulk build / rebuild (GPU build — incremental via checkpoint)
eph-rag src/build_hierarchical_gpu.py --papers-dir /path/to/md/ --batch-size 50

# Batch verification after indexing: check coverage
eph-rag src/query_bib_rag.py "<distinctive phrase>" --top 3
```

## Adding papers

```bash
# PDFs → markdown → index (incremental; skips already-indexed papers)
eph-rag add_papers.py /path/to/new/pdfs/
eph-rag add_papers.py /path/to/md/ --skip-extract   # already markdown
eph-rag add_papers.py /path/to/pdfs/ --batch-size 5 # smaller batches

# Verify a specific paper landed:
eph-rag src/query_bib_rag.py "<phrase unique to the new paper>" --top 3
```

Downloads land in the library's `md/` by default; keep PDFs durably (never /tmp).

## Deleting papers

```bash
# Dry-run (default) — shows exactly what matches, deletes nothing:
eph-rag scripts/remove_paper.py 10087273.md

# By filename substring:
eph-rag scripts/remove_paper.py --match Salvucci

# Actually delete (chroma chunks + parent_store JSON + checkpoint/metadata entries):
eph-rag scripts/remove_paper.py 10087273.md --apply
```

Deletion removes the paper completely — re-index the same md any time to bring
it back cleanly. To wipe a whole library, just delete its folder (and its
registry line in `src/kb_config.py`).

---

## Agentic RAG (v2.0)

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
bib_rag/                      ← toolkit (code only; DATA lives in sibling eph_rag/, geo_rag/)
├── GUIDE.md                    ← NEW-USER START: library setup + usage walkthrough
├── add_papers.py               ← Add new PDFs to the active library's index (entry point)
├── agentic_query.py            ← Query the agentic RAG system (entry point)
├── scripts/setup_library.py    ← Scaffold + register a new library (one command)
│
├── scripts/                    ← Utilities & pipelines
│   ├── meta_audit.py           ← Metadata proof-reader + genuine-info fetcher
│   ├── bib_to_parent_store.py · fill_meta_key_in_parent_store.py ← fill meta from My Library.bib
│   ├── bib_utils.py            ← Shared normalization / .bib / filename helpers
│   ├── zotero_access.py        ← Zotero access layer (MCP first, HTTP fallback)
│   ├── test_utilities.py       ← Regression tests (no network)
│   ├── classify_all.py · classify_demo.py · apply_tags.py · migrate_topics.py ← LLM classification pipeline
│   └── (classify_cadherin*.py · backfill_cadherin.py · batch_index_cadherin.py ← eph corpus copies live in eph_rag/scripts/)
│
├── src/                        ← RAG library & query/writer tools
│   ├── kb_config.py            ← SINGLE PATH-RESOLUTION POINT: registry + BIB_RAG_* env
│   ├── agentic_graph.py · agent_nodes.py · agent_edges.py ← LangGraph pipeline
│   ├── agent_prompts.py · agent_schemas.py · agent_tools.py ← prompts / state / retrieval tools
│   ├── build_hierarchical_gpu.py · build_hierarchical.py · index_single_paper.py · chunking.py ← index build
│   ├── query_bib_rag.py        ← Quick semantic search & citations
│   ├── bib_rag_writer.py · bib_rag_writer_debate.py · bib_rag_grill.py ← writers with citations
│   ├── parent_store_manager.py ← Parent chunk loader
│   └── evaluate.py · test_comprehensive.py · test_agentic_graph.py ← eval & tests
│
├── docs/                       ← README.md (index) + archive/ (historical 2026-03→06, obsolete systems)
│
└── .gitignore                  ← libraries (eph_rag/, data dirs) never enter this repo
```

**Library folder layout** (created by `scripts/setup_library.py`, e.g. `eph_rag/`):
`chroma_db_new/` (vectors) · `parent_store/` + `parent_store_disabled/` (full-text
JSON) · `data/` (checkpoints) · `outputs/` (tag CSVs) · `CONTEXT.md` (domain
glossary) · `LIBRARY.md` (manifest) · optional `scripts/` (domain-owned batch tools).

---

## Server Setup

```bash
# Start embedding server (bge-m3) — -ub 2048 is REQUIRED (see Troubleshooting)
LD_LIBRARY_PATH=/Disk_2/llama.cpp/build/bin /Disk_2/llama.cpp/build/bin/llama-server \
  -m /Disk_bot/models/embeddings/bge-m3-Q4_K_M.gguf \
  --port 8081 -c 8192 -ngl 999 --embedding -ub 2048 &

# LLM: cloud gateway (Ollama, port 11434) is usually already running;
# for a local GGUF instead:
# llama-server -m /path/to/model.gguf --port 5015 -c 524288 -ngl 999 \
#   --flash-attn on -np 2 --cache-type-k q8_0 --cache-type-v q8_0

# Both should respond:
curl http://localhost:8081/health   # embeddings
curl http://localhost:5015/health   # LLM (only if local)
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
