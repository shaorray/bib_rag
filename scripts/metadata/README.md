# metadata/ — metadata fixation pipeline

Everything that writes or corrects paper metadata (bibliographic fields +
retrieval tags) after the noisy index-time extraction. Run via wrappers:
`<name>-rag scripts/metadata/<tool>.py ...`

## Sources of truth (in priority order)

1. **Zotero BibTeX snapshot** — `My Library.bib` (BIB_RAG_BIB_PATH / --bib).
   Offline export of the Zotero library; freshest right after a re-export.
2. **Live Zotero** — `scripts/zotero_access.py` (MCP-first, HTTP fallback
   port 23119). Targeted lookups without re-exporting.
3. **Remote registries** — Crossref / PubMed / OpenAlex (inside meta_audit).
   For papers not in Zotero + cross-validation of 1 and 2.

## Pipeline order (per batch of papers)

| Step | Tool | Writes |
|---|---|---|
| 1. Tag generation (LLM) | `classify_papers.py` (in `scripts/`) | `outputs/<tags>.csv` (source, year, title, article_type, topics) |
| 2. Clean-metadata overwrite + tags | `backfill_metadata.py` | chroma metadata: title/authors/journal/doi/year + article_type + topic_* |
| 3. DOI backfill from BibTeX | `bib_to_parent_store.py` | parent_store meta.doi (triple-match: lastname+year+title-prefix; unmatched never written) |
| 4. Zotero key fill | `fill_meta_key_in_parent_store.py` | parent_store meta.key (Zotero article_key, DOI-matched) |
| 5. Audit + remote fetch | `meta_audit.py` | verifies/fetches title, year, journal, authors, doi via Crossref/PubMed/OpenAlex; confidence-gated, backs up before write; reports in `data/` |
| 6. Tag migration (legacy) | `migrate_topics.py` | converts older topic representations → `topic_<kw>:1` boolean keys |

`apply_tags.py` = tags-only variant of step 2 (use when there is no clean
metadata CSV). `scripts/remove_paper.py` is the reset button for a paper whose
metadata is wrong beyond repair: delete, fix the source, re-index.

## Invariants

- Every writer tool defaults to dry-run or confidence-gating; nothing writes
  on a guess. Unmatched/low-confidence bindings are reported, never guessed.
- parent_store JSONs and chroma metadata are updated atomically
  (chunking.atomic_json_dump).
- Classification CSVs are decoupled from DB writes — inspect before applying.
