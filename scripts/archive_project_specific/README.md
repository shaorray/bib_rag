# archive_project_specific/ — superseded one-off corpus scripts

These were written for specific ingestion batches (Cadherin_papers,
cadherin_code review corpus). Their functionality has been distilled into the
general-purpose scripts kept in `scripts/`:

| archived | purpose it had | superseded by |
|---|---|---|
| `classify_cadherin.py` | classify papers from a clean-metadata CSV (PMID labeled.csv) | `scripts/classify_papers.py --titles-csv` |
| `classify_cadherin_code.py` | classify papers from md files (title + pseudo-abstract) | `scripts/classify_papers.py --md-dir` |
| `classify_all.py` | classify every source already in ChromaDB | `scripts/classify_papers.py --from-chroma` |
| `classify_demo.py` | early cloud-GLM classification demo | `scripts/classify_papers.py --backend cloud` |
| `backfill_cadherin.py` | overwrite noisy metadata from labeled.csv + apply tags | `scripts/backfill_metadata.py` |
| `batch_index_cadherin.py` | loop index_single_paper over a md dir | `add_papers.py` (or inline loop — 3 lines) |
| `retry_index_cadherin.py` | re-index papers missing from chroma | `add_papers.py` (incremental) |

Note: working copies of the cadherin batch tools live in
`/Disk_bot/RAG/eph_rag/scripts/` (library-owned, pinned to eph_rag) — those
remain the canonical ones for that corpus.

These archived copies reference `/Disk_bot/Eph/Cadherin_papers/` and other
corpus-specific paths; they are kept for provenance only.

| `fill_meta_key_in_parent_store.py` | fill Zotero article_key into parent_store (DOI-bridged) | merged into `scripts/metadata/bind_zotero.py` (writes meta.doi + meta.key in one pass) |
