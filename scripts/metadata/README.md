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

## Orchestrator (recommended entry point)

`backfill_all.py` runs the layers in fallback order per paper, stopping at the
first layer that resolves it:

```
0. BibTeX snapshot  — if provided (--bib or config.json bib_path):
                      bind_zotero.py — binds meta.doi + meta.key (Zotero
                      citation key) per paper; triple-match + title-search +
                      PMID-bridge (PubMed esummary) + DOI-bridge
1. Live Zotero      — if reachable (MCP or HTTP): zotero_search/zotero_item
2. Remote registries — Crossref DOI verify → Crossref/OpenAlex/PubMed title search
3. Final proof-read  — meta_audit.py round (confidence-gated, backs up)
```

Writes a per-paper ledger (`data/backfill_status.csv`: which layer fixed what)
so re-runs skip fixed papers and each layer's contribution is auditable.
Noisy titles (journal headers) are detected and skipped for Zotero search,
and cleaned heuristically before registry search — Crossref recovers real
titles from "Developmental Biology **207, available online..." style noise.

```bash
<name>-rag scripts/metadata/backfill_all.py --dry-run            # plan + probe
<name>-rag scripts/metadata/backfill_all.py --bib "/path/My Library.bib" --apply
```

## Manual pipeline order (equivalent, step-by-step)

| Step | Tool | Writes |
|---|---|---|
| 1. Tag generation (LLM) | `classify_papers.py` (in `scripts/`) | `outputs/<tags>.csv` (source, year, title, article_type, topics) |
| 2. Clean-metadata overwrite + tags | `backfill_metadata.py` | chroma metadata: title/authors/journal/doi/year + article_type + topic_* |
| 3. Zotero binding (BibTeX snapshot) | `bind_zotero.py` | parent_store meta.doi + meta.key (Zotero citation key) in one pass; matchers: triple-match → title reverse-search → PMID-bridge (PubMed esummary) → DOI-bridge; unmatched never written |
| 5. Audit + remote fetch | `meta_audit.py` | verifies/fetches title, year, journal, authors, doi via Crossref/PubMed/OpenAlex; confidence-gated, backs up before write; reports in `data/` |
| 6. Tag migration (legacy) | `migrate_topics.py` | converts older topic representations → `topic_<kw>:1` boolean keys |

`apply_tags.py` = tags-only variant of step 2 (use when there is no clean
metadata CSV). `bib_to_parent_store.py` still holds the matchers that
`bind_zotero.py` imports (the old `fill_meta_key_in_parent_store.py` is
archived; its logic merged into bind_zotero). `scripts/remove_paper.py` is the reset button for a paper whose
metadata is wrong beyond repair: delete, fix the source, re-index.

## Invariants

- Every writer tool defaults to dry-run or confidence-gating; nothing writes
  on a guess. Unmatched/low-confidence bindings are reported, never guessed.
- parent_store JSONs and chroma metadata are updated atomically
  (chunking.atomic_json_dump).
- Classification CSVs are decoupled from DB writes — inspect before applying.
