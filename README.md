# bib_rag — Stop AI from inventing your citations

**An evidence-gated RAG toolkit for academic writing.** Every sentence your AI
drafts must trace back to a paper that is actually *in your library* —
retrieved by you, chunk-verified, and citation-checked before it reaches the
page.

Built and battle-tested in daily empirical-research writing (molecular
biology + genomics): multi-omics papers, review sections, grant-style claims.
Not a demo RAG — the honesty machinery (claim ledger, citation guard,
permutation-grade evidence gates) exists because LLM-written bibliographies
fail quietly unless you force them not to.

<!-- TODO(shot): 15s terminal GIF — `bibrag query --cite "claim" → sources + writer output` -->

## The problem it solves

Generic LLM + RAG pipelines fail three ways on academic text:

1. **Hallucinated citations** — plausible authors/years/DOIs that don't exist
2. **Right paper, wrong claim** — the source is real but says something weaker
   than the sentence citing it
3. **Context amnesia** — after compaction, the model "forgets" which evidence
   it already pulled and re-derives from memory instead

bib_rag attacks all three at the pipeline level, not the prompt level:

| Layer | Mechanism | LLM cost |
|---|---|---|
| Retrieval | Hybrid dense + FTS5-BM25 with RRF fusion, reranker service, hierarchical child→parent chunk expansion | zero |
| Generation | Locked "no answer → say so" source-integrity rules; structured output schemas; claim-level confidence (`claim_ledger`) | — |
| Output | **`citation_guard`** — every cited source must be in the retrieval whitelist *and* lexically support the claim; violations are dropped or downgraded | zero |
| Verification | `evidence_gate` — insufficient-evidence audits + explicit gap reporting; refuse-before-LLM short-circuit when retrieval returns nothing | zero |
| Metadata | `meta_audit` — author/volume/DOI cross-checked against Zotero + Crossref/PubMed/OpenAlex before a citation is emitted | zero |
| Regression | `evaluate.py` — retrieval + citation-faithfulness harness over gold claims, so config changes get graded, not vibes-reviewed | — |

## Quick start

```bash
git clone https://github.com/shaorray/bib_rag.git && cd bib_rag
python3.10 -m pip install -e . --user        # provides the `bibrag` command
```

You need two [llama-server](https://github.com/ggml-org/llama.cpp) style
services (any OpenAI-compatible endpoint works):

```bash
# embeddings — REQUIRED (bge-m3 recommended; -ub 2048 matters, see docs)
llama-server -m bge-m3.gguf --port 8081 -c 8192 -ngl 999 --embedding -ub 2048
# LLM — required only for agentic Q&A / classification / writer-synthesis
llama-server -m your-model.gguf --port 5015 -c 524288 -ngl 999
```

Create a library and index your papers:

```bash
python3.10 -B scripts/setup_library.py --name mybib_rag \
    --domain "your domain" --wrapper yes --no-interactive

python3.10 -B add_papers.py /path/to/pdfs/   # PDF → md → chunked → embedded

bibrag config                                # sanity check
```

## Everyday usage

**Find support for a claim (the core "cite" loop):**

```bash
bibrag query --cite "the specific claim you are about to write" --top 3
```

**Write a paragraph with live citations:**

```bash
bibrag src/bib_rag_writer.py "topic sentence" --top 5 --style APA
```
Writer pipeline: scope-grill → retrieve → synthesize → citation-guard →
metadata-proofread → markdown with verified APA/Vancouver citations.

**Agentic Q&A over the whole library** (LangGraph; follow-ups keep context):

```bash
bibrag agentic "What do my papers say about X?"
```

**Metadata repair for a messy library** (Zotero snapshot → live Zotero →
Crossref/PubMed/OpenAlex fallback chain, per-paper audit ledger):

```bash
bibrag scripts/metadata/backfill_all.py --bib "My Library.bib" --apply
```

**Delete/rebuild are safe by default** — deletion dry-runs first; re-indexing
the same markdown restores it cleanly.

## Design notes

- **Toolkit vs library**: code lives here; your papers, vectors, and stores
  live in self-describing library folders (`LIBRARY.md` manifest, one ChromaDB
  per domain). One command scaffolds a new library; switching is an env var —
  never a fragile global root.
- **Chunking respects scientific tables/figures**: captions become atomic
  parent chunks with `chunk_type` tags, so citation checks can quote the
  exact panel a claim rests on.
- **Snowballing built in**: `reference_graph` supports forward/backward
  citation walking — useful when the seed paper's method section is the
  actual question.
- **PageIndex / vectorless mode**: tree-index retrieval is vendored as an
  alternative to embeddings for small, structured corpora.
- **Fully local option**: bge-m3 + any llama.cpp GGUF = zero cloud, zero
  cost, private library. Cloud gateway works identically.

## Requirements

Python ≥ 3.10 · ChromaDB 1.4.1 · LangGraph/LangChain ≥ 0.3 · pymupdf4llm
(embedding service ≥ 8081; LLM endpoint for agentic/writer paths). Full list:
`requirements.txt`. GPU recommended for bulk indexing; querying runs fine CPU-only with the reranker optional.

## Docs

- [`GUIDE.md`](GUIDE.md) — new-user walkthrough (library setup → first query)
- [`docs/hallucination_governance_checklist.md`](docs/hallucination_governance_checklist.md)
  — the three-layer honesty audit (input/generation/output) with current
  per-item status — kept honest, including the ⚠️/❌ rows
- [`docs/GUARD_HYBRID.md`](docs/GUARD_HYBRID.md) — citation-guard & hybrid
  retrieval internals, borrowed-mechanism attribution
- [`scripts/metadata/README.md`](scripts/metadata/README.md) — metadata
  fixation pipeline

## Status

v0.9.0 — single-maintainer, used daily in real manuscript preparation.
APIs may still move; the honest bits (citation guard, claim ledger) are the
stable core. Issues and PRs welcome — especially reproducible failure cases
where the guard let something hallucinated through.

## License

MIT