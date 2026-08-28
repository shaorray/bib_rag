# GUIDE — Setting up a RAG library with the bib_rag toolkit

This repo is the **toolkit** (code only). Your papers live in **libraries** —
self-describing data folders under a common RAG root. One toolkit, many libraries.

```
<RAG root>/
├── bib_rag/        ← the toolkit (CODE: src/, scripts/, docs/) — this repo
├── <name>_rag/     ← one data folder per domain (created in 1 command, below)
└── ...
```

## Quick start (2 minutes)

```bash
cd <this repo>
/usr/bin/python3.10 -B scripts/setup_library.py
```

Interactive prompts (name, domain, optional wrapper) — everything has sensible
defaults. Fully scripted alternative:

```bash
/usr/bin/python3.10 -B scripts/setup_library.py \
    --name <name>_rag \
    --domain "neuroscience / axon guidance" \
    --wrapper yes --no-interactive
```

One command does all of:
1. Creates `<RAG root>/<name>_rag/` with `chroma_db_new/`, `parent_store/`,
   `parent_store_disabled/`, `data/`, `outputs/`, `md/`
2. Writes `LIBRARY.md` (manifest) + `CONTEXT.md` (domain glossary starter)
   + `config.json` (per-library machine/model settings — local classify model,
   BibTeX path, tmpdir, domain topic seeds. Resolution: env var > config.json
   > toolkit default. Never put secrets in it.)
3. Registers the library in `src/kb_config.py` (`_KB_REGISTRY`: root + collection,
   brace-counted patch with post-write sanity checks — cannot clobber the file)
4. Emits a `neuro-rag` wrapper in `~/.local/bin/` (name = library name minus `_rag`)

Naming rule: library names are `snake_case` ending in `_rag`.
The Chroma collection defaults to `<stem>_papers` (neuro_papers).

Safety: the script refuses to touch a directory that holds real data without a
LIBRARY.md, and aborts rather than corrupting kb_config.py if the patch would
drop any function. Re-running a failed scaffold resumes cleanly.

## Prerequisites (shared services, one-time)

| Service | Port | Purpose | Notes |
|---|---|---|---|
| bge-m3 embeddings | 8081 | indexing + semantic query | see start command below |
| Qwen classifier | 5015 | batch article_type/topic tagging (optional) | only for classify step |
| Ollama | 11434 | agentic Q&A, cloud classify (optional) | `ollama serve` |

```bash
# start the embedding server (the -ub 2048 flag is REQUIRED — see Troubleshooting)
LD_LIBRARY_PATH=/Disk_2/llama.cpp/build/bin /Disk_2/llama.cpp/build/bin/llama-server \
  -m /Disk_bot/models/embeddings/bge-m3-Q4_K_M.gguf \
  --port 8081 -c 8192 -ngl 999 --embedding -ub 2048 &
curl http://localhost:8081/health   # → {"status":"ok"}
```

**Python**: always `/usr/bin/python3.10 -B`. Bare `python3` is a pyenv shim
without deps. The wrappers also set PYTHONPATH to dodge a system-protobuf
shadowing issue that breaks chromadb imports.

## First paper in, first answer out

```bash
# 1. markdown paper somewhere durable (convert PDFs with pymupdf4llm; never /tmp)
# 2. index:
neuro-rag src/index_single_paper.py /path/to/paper.md
#    → chunks, embeds via 8081, stores into <name>_rag/chroma_db_new

# 3. query it back:
neuro-rag src/query_bib_rag.py "distinctive phrase from that paper" --top 3

# 4. batch PDFs later:
neuro-rag add_papers.py /path/to/pdf/dir/
```

## Growing the library properly

1. **Convert** PDFs → md (pymupdf4llm), keep md + PDF in durable storage.
2. **Index** (batch, incremental): `neuro-rag add_papers.py <dir>`
3. **Classify** article_type + topics: copy a `classify_*.py` from `scripts/`,
   adapt the topic taxonomy to your domain, run in the background (~0.5 papers/s).
   Feed title+abstract, not title alone — titles miss method papers.
4. **Apply tags**: `neuro-rag scripts/apply_tags.py outputs/<name>_tags.csv`
   (writes `article_type` + `topic_<kw>:1` keys into chunk metadata).
5. **Fill in** `<name>_rag/CONTEXT.md` with your domain vocabulary — the
   writer/grill/agentic tools read it to constrain generation.

Full procedures + pitfalls: Hermes skills `bib-rag-ingest` (write side) and
`bib-rag-query` (search side).

## Moving / backing up a library

Libraries are plain folders — `mv` / `tar` / `rsync` as a unit. After moving,
update one registry line in `src/kb_config.py` (root path) and the wrapper's
`BIB_RAG_ROOT` default. Nothing else references the old location.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ImportError: cannot import name 'builder' from 'google.protobuf.internal'` | system protobuf shadows user-site | use the wrappers, or prefix `PYTHONPATH=/home/rui/.local/lib/python3.10/site-packages` |
| `Collection [X] does not exist` | BIB_RAG_ROOT leaked from an earlier `export` | `unset BIB_RAG_ROOT BIB_RAG_COLLECTION`; use per-command env or wrappers |
| embedding HTTP 500 `input too large... increase physical batch size` | 8081 running with default ubatch 512 | restart 8081 with `-ub 2048` (command above) |
| `ModuleNotFoundError: chromadb` | bare `python3` (pyenv shim) | use `/usr/bin/python3.10 -B` |