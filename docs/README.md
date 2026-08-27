# docs/ — technical reference for bib_rag

New users: start with [`../GUIDE.md`](../GUIDE.md) (set up a library in one command), then [`../README.md`](../README.md) (toolkit layout). The maintained,
always-current procedures live in the Hermes skills `bib-rag-ingest` and
`bib-rag-query`.

## Technical references in this folder

| File | What you'll learn |
|---|---|
| `PHASE2_5_STATUS.md` | The hierarchical parent/child indexing design: why child chunks are searched and full parents retrieved, the ChromaDB/parent-store split, module responsibilities, and the store's scale characteristics. |
| `AGENTIC_RAG_BEST_PRACTICES.md` | Agentic RAG design principles behind this toolkit: the four core agents, golden parameters (iterations/tool-call budgets, compression thresholds), planner/reflector prompt patterns for hallucination control. |
| `AGENTIC_RAG_INTEGRATION_PLAN.md` | Design rationale for the LangGraph pipeline: state-machine workflow, context compression, map-reduce over multi-part queries — implemented as described in `src/agentic_graph.py` + `agentic_query.py`. |
| `IMPROVEMENT_ANALYSIS.md` | Agentic-vs-classical RAG concepts, retrieval-as-tool pattern, and the evaluation thinking (RAGAS metrics) behind the toolkit's design choices. |
| `RAG_EMBEDDING_GUIDE.md` | Chunking and embedding fundamentals: chunk-size/overlap tradeoffs, section-aware splitting, why embeddings quality dominates retrieval quality. |
| `ZOTERO_MCP_USAGE.md` | Zotero local API (port 23119) endpoint reference: item listing/search, `item["data"]` nesting, collections — used by `scripts/zotero_access.py` and the metadata/writer tools. |

Each file carries a status banner marking what is current vs historical —
check it before copying code examples.

## archive/ — obsolete systems (do not follow)

Files describing the OLD `ephrin_agentic_rag` system (`rag_core.py`,
`SimpleEmbedding`, `academic_writer.py`, `query_v2/v3_kb`) that lived at
`/Disk_2/claw_working_dir/`, plus completed one-off cleanup/inventory reports.
None of those modules exist in this toolkit. Historical provenance only.