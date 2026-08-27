# docs/ — documentation index

## For new users (read in this order)
1. **../GUIDE.md** (repo root) — set up a new RAG library in one command, services,
   first paper in/out, troubleshooting. START HERE.
2. **../README.md** — toolkit overview and layout (code vs libraries).
3. **../CONTEXT.md** — what the domain-glossary convention is (each library has one).

## Canonical how-to references (live in the Hermes skills, not here)
The complete, maintained procedures live in the Hermes agent skills — they are
updated with every architecture change:
- `bib-rag-ingest` — indexing, classification, tagging, metadata backfill, Zotero verification
- `bib-rag-query` — semantic search, ToolFactory API, parent_store scanning, agentic Q&A,
  DOI citation audit, Tavily supplementation

## Zotero
- `archive/ZOTERO_MCP_USAGE.md` — local Zotero API (port 23119) quick reference.
  Superseded by the `research-paper-management` Hermes skill but still accurate
  for endpoint shapes (userID 0, /items/top meta-only, item["data"] nesting).

## archive/ — historical (2026-03 → 2026-06), kept for provenance only
Everything in `archive/` describes **obsolete systems or completed one-off work** and
does NOT reflect the current architecture. Notable:
- `README.md`, `USAGE.md`, `QUICK_START.md`, `SCHOLAR_API.md`, `ACADEMIC_USAGE.md`,
  `PMID_CITATION_GUIDE.md` — the OLD ephrin_agentic_rag system (rag_core/SimpleEmbedding,
  agentic_workflow.py, query_v2/v3_kb) which lived at /Disk_2/claw_working_dir/.
  Replaced by this toolkit (ChromaDB + bge-m3 + kb_config multi-library).
- `IMPROVEMENT_*`, `V3_IMPROVEMENT_REPORT`, `PHASE*_STATUS`, `AGENTIC_RAG_*`,
  `CLEANUP_REPORT`, `FILE_INVENTORY`, `RAG_EMBEDDING_GUIDE` — design/learning reports
  from the March–June 2026 build-out. Historical interest only.

Do not follow code examples in archive/ — module names and paths no longer exist.
