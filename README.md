# bib_rag

Academic bibliography RAG toolkit for evidence-based writing. One toolkit,
many libraries:

- **this repo** — the toolkit (code only: `src/`, `scripts/`, `docs/`)
- **libraries** — one self-describing data folder per domain, each with its own
  ChromaDB (`chroma_db_new/`), full-text store (`parent_store/`), checkpoints
  (`data/`), tags (`outputs/`), and a `LIBRARY.md` manifest. Create yours with
  `scripts/setup_library.py`; the toolkit serves any of them via
  `BIB_RAG_KB_NAME` (or the per-library command it generates).

> **New here?** Read [`GUIDE.md`](GUIDE.md). The maintained how-to procedures
> live in the Hermes skills `bib-rag-ingest` and `bib-rag-query`. Historical
> docs: `docs/archive/`.

---

## Quick Start

```bash
# 0. Get the toolkit (if you haven't cloned it yet)
git clone https://github.com/shaorray/bib_rag.git && cd bib_rag

# 0b. Optional (recommended): install as an editable package
#     → gives you the `bibrag` console command from any directory.
/usr/bin/python3.10 -m pip install -e . --user
bibrag config                      # verify: prints active library
bibrag query "some phrase" --top 3 # works without any wrapper
#     The repo still works un-installed (loose-script mode) — installing
#     changes nothing for existing eph-rag/geo-rag workflows.

# 0c. Prerequisites
#    a) Python dependencies (Python 3.10; see requirements.txt for details):
pip install -r requirements.txt
#    b) an OpenAI-compatible embedding server (bge-m3 recommended) — REQUIRED
#       for indexing and querying; this toolkit expects it on port 8081
#       (see "Server Setup" below).
curl http://localhost:8081/health          # → {"status":"ok"}

# 1. Create a library (one command: folders + manifest + registry + wrapper)
/usr/bin/python3.10 -B scripts/setup_library.py --name <name>_rag \
    --domain "your domain" --wrapper yes --no-interactive

# 2. Index a paper
<name>-rag src/index_single_paper.py /path/to/paper.md

# 3. Query it
<name>-rag src/query_bib_rag.py "distinctive phrase from the paper"

# 4. Batch-add PDFs (extracts PDF→md, then indexes incrementally)
<name>-rag add_papers.py /path/to/pdf/dir/

# 5. Delete a paper (dry-run by default)
<name>-rag scripts/remove_paper.py paper_stem.md --apply

# 6. Agentic Q&A over the library
<name>-rag agentic_query.py "What do my papers say about X?"
```

`<name>-rag` is the per-library command the setup script generates (it pins
the library and sets up the Python environment).

> **Python**: always `/usr/bin/python3.10 -B` — the wrappers set this plus the
> `PYTHONPATH` fix chromadb needs. Bare `python3` is a pyenv shim without deps.

## Building a RAG library

```bash
# Interactive — prompts for name/domain with defaults:
/usr/bin/python3.10 -B scripts/setup_library.py

# Scripted:
/usr/bin/python3.10 -B scripts/setup_library.py \
    --name <name>_rag \
    --domain "your domain" \
    --wrapper yes --no-interactive
```

This creates the library folder (`chroma_db_new/`, `parent_store/`, `data/`,
`outputs/`, `md/`), writes `LIBRARY.md` + `CONTEXT.md`, registers it in
`src/kb_config.py`, and emits the `<name>-rag` command. Library names are
`snake_case` ending in `_rag`; the Chroma collection defaults to `<stem>_papers`.

Switch libraries any time — each wrapper pins its own library, or use
`BIB_RAG_KB_NAME=<other>_rag <command>` per-call (never `export BIB_RAG_ROOT` — it
leaks across calls and silently writes into the wrong library).

## Indexing papers

```bash
# Single paper (chunks parent+child, embeds via 8081, stores into the active library)
<name>-rag src/index_single_paper.py /path/to/paper.md

# First-time bulk build / rebuild (GPU build — incremental via checkpoint)
<name>-rag src/build_hierarchical_gpu.py --papers-dir /path/to/md/ --batch-size 50

# Batch verification after indexing: check coverage
<name>-rag src/query_bib_rag.py "<distinctive phrase>" --top 3
```

## Adding papers

```bash
# PDFs → markdown → index (incremental; skips already-indexed papers)
<name>-rag add_papers.py /path/to/new/pdfs/
<name>-rag add_papers.py /path/to/md/ --skip-extract   # already markdown
<name>-rag add_papers.py /path/to/pdfs/ --batch-size 5 # smaller batches

# Verify a specific paper landed:
<name>-rag src/query_bib_rag.py "<phrase unique to the new paper>" --top 3
```

Downloads land in the library's `md/` by default; keep PDFs durably (never /tmp).

## Deleting papers

```bash
# Dry-run (default) — shows exactly what matches, deletes nothing:
<name>-rag scripts/remove_paper.py <paper>.md

# By filename substring:
<name>-rag scripts/remove_paper.py --match <author>

# Actually delete (chroma chunks + parent_store JSON + checkpoint/metadata entries):
<name>-rag scripts/remove_paper.py <paper>.md --apply
```

Deletion removes the paper completely — re-index the same md any time to bring
it back cleanly. To wipe a whole library, just delete its folder (and its
registry line in `src/kb_config.py`).

---

## Agentic RAG

Full LangGraph-powered agentic pipeline with hierarchical retrieval, context compression, and conversation memory.

### Quick Start — Agentic Query

```bash
cd bib_rag

# Single query (default: fast cloud model)
/usr/bin/python3.10 -B agentic_query.py "What is the role of retinoic acid in neural development?"

# Interactive chat mode
/usr/bin/python3.10 -B agentic_query.py --interactive

# Verbose mode (shows pipeline progress)
/usr/bin/python3.10 -B agentic_query.py "Compare E-cadherin and N-cadherin functions" --verbose
```

> **Note**: use `/usr/bin/python3.10` (the interpreter with langchain deps
> installed), not bare `python3`.

**Model choice**: agentic mode runs on a cloud model by default, but you can
point it at a local model instead — set `LLM_URL` / `LLM_MODEL` to your
llama-server endpoint and a `.gguf` path for a fully offline, private run.
See `agentic_query.py --help` for the backend options.

**Prerequisites**: Embedding server must be running.
- Embeddings: bge-m3 on port 8081
- LLM: a cloud gateway (port 11434) **or** a local model (port 5015)

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
bib_rag/                      ← toolkit (code only; DATA lives in sibling library folders)
├── GUIDE.md                    ← NEW-USER START: library setup + usage walkthrough
├── add_papers.py               ← Add new PDFs to the active library's index (entry point)
├── agentic_query.py            ← Query the agentic RAG system (entry point)
├── scripts/setup_library.py    ← Scaffold + register a new library (one command)
│
├── scripts/                    ← Utilities & pipelines
│   ├── metadata/               ← METADATA FIXATION PIPELINE (see metadata/README.md):
│   │     bind_zotero (BibTeX→doi+key) · backfill_metadata · meta_audit (Crossref/PubMed/OpenAlex)
│   │     apply_tags · migrate_topics ← classification tags; sources: BibTeX snapshot > live Zotero > registries
│   ├── classify_papers.py      ← LLM tagging (article_type + open-vocab topics, 3 input modes)
│   ├── bib_utils.py            ← Shared normalization / .bib / filename helpers
│   ├── zotero_access.py        ← Zotero access layer (MCP first, HTTP fallback)
│   ├── remove_paper.py · setup_library.py ← library lifecycle
│   └── test_utilities.py       ← Regression tests (no network)
│   └── archive_project_specific/ ← superseded one-off corpus scripts (provenance only)
│
├── src/                        ← RAG library & query/writer tools
│   ├── kb_config.py            ← SINGLE PATH-RESOLUTION POINT: registry + BIB_RAG_* env
│   ├── agentic_graph.py · agent_nodes.py · agent_edges.py ← LangGraph pipeline
│   ├── agent_prompts.py · agent_schemas.py · agent_tools.py ← prompts / state / retrieval tools
│   ├── build_hierarchical_gpu.py · index_single_paper.py · chunking.py ← index build
│   ├── query_bib_rag.py        ← Quick semantic search & citations
│   ├── bib_rag_writer.py       ← PARAGRAPH COMPOSER: grill→retrieve→synthesize (backends:
│   │     default LLM · --no-llm template · --debate relational; markdown + citations;
│   │     2026-08-31 merged from bib_rag_grill.py + writer/writer_debate shims;
│   │     dependency policy: retrieval-metadata citations, no odfpy/zotero_access)
│   ├── parent_store_manager.py ← Parent chunk loader
│   └── evaluate.py · test_comprehensive.py · test_agentic_graph.py ← eval & tests
│
├── docs/                       ← README.md (index) + archive/ (historical 2026-03→06, obsolete systems)
│
└── .gitignore                  ← library folders / data dirs never enter this repo
```

**Library folder layout** (created by `scripts/setup_library.py`):
`chroma_db_new/` (vectors) · `parent_store/` + `parent_store_disabled/` (full-text
JSON) · `data/` (checkpoints) · `outputs/` (tag CSVs) · `CONTEXT.md` (domain
glossary) · `LIBRARY.md` (manifest) · optional `scripts/` (domain-owned batch tools).

---

## Server Setup

The toolkit talks to two local services over OpenAI-compatible HTTP APIs.
Start them with your own `llama-server` build (or any compatible server);
the exact binary path, model path and GPU flags depend on your machine.

**Embedding server (required)** — an embedding model such as `bge-m3`,
with embedding mode enabled. The `-ub 2048` physical-batch flag matters:
the default of 512 rejects any chunk over 512 tokens (see Troubleshooting).

```bash
llama-server -m /path/to/bge-m3.gguf \
  --port 8081 -c 8192 -ngl 999 --embedding -ub 2048 &
curl http://localhost:8081/health   # → {"status":"ok"}
```

**LLM (required only for agentic Q&A and classification)** — either a
cloud gateway (e.g. Ollama's OpenAI-compatible endpoint) or a local GGUF:

```bash
# local model:
llama-server -m /path/to/model.gguf --port 5015 -c 524288 -ngl 999 \
  --flash-attn on -np 2 --cache-type-k q8_0 --cache-type-v q8_0 &
curl http://localhost:5015/health
```

Point the toolkit at whichever backend you chose with `LLM_URL` / `LLM_MODEL`
(see the Agentic RAG section). Default ports the toolkit expects:
embeddings **8081**, LLM **11434** (cloud) or **5015** (local).

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
