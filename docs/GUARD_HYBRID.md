# GUARD/HYBRID — Citation verification, hybrid retrieval & snowballing

Six-gap hardening (2026-08-28), mechanisms borrowed from the 23-repo survey
(`/Disk_bot/notes/{citation_rag,Agentic_RAG,zotero_RAG}/`, comparison in
`/Disk_bot/notes/bib_rag_对比与借鉴.md`).

## What was added

| Module | Mechanism (source repo) | Cost |
|---|---|---|
| `src/citation_guard.py` | Sources whitelist vs `retrieval_keys` + lexical support check (paper-qa / LumiCite / citelocal-agent) | zero LLM |
| `src/zotero_match.py` | Title-similarity + DOI verification for Zotero hits (paper-qa) | zero LLM |
| `src/hybrid_search.py` | FTS5 BM25 channel + RRF k=60 fusion (DocsGPT / seerai) | zero LLM |
| `src/evidence_gate.py` | Evidence-sufficiency audit + gap reporting in fallback (ragent 证据门槛) | zero LLM |
| `src/reference_graph.py` | Citation graph + forward/backward snowballing (Corvus) | zero LLM |
| `src/chunking.py` (extended) | Figure/table captions become atomic parents with `chunk_type` tag (LumiCite) | zero LLM |
| `src/evaluate.py` (extended) | `citation_faithfulness()` metric reusing the guard | zero LLM |

## Data files (per library, under `<data_root>/data/`)

- `fts_index.db` — BM25 index over child chunks (482,971 children for eph_rag).
  Rebuild: `python3 scripts/build_fts_index.py`; single paper:
  `--source <name>_md`; status: `--status`.
- `reference_graph.json` — 2,514 papers / 22,158 edges for eph_rag.
  Rebuild: `python3 scripts/build_reference_graph.py`.

Both never touch `chroma_db_new/`. Both are optional at query time: if the
file is missing, hybrid search silently degrades to dense-only and the
snowballing tools return `NO_REFERENCE_GRAPH` with instructions.

## Env switches (all default ON)

- `CITATION_GUARD=0` — disable answer Sources whitelisting (collect_answer)
- `HYBRID_SEARCH=0` — disable BM25 channel (search_child_chunks)
- `EVIDENCE_GATE=0` — disable fallback gap reporting
- `CITATION_LEXICAL_THRESHOLD` (0.15), `CITATION_RARE_TOKENS` (3),
  `ZOTERO_MATCH_MIN_SIM` (0.55), `ZOTERO_MATCH_MIN_DOI_PREFIX` (8)

## How the citation guard works (collect_answer tail)

1. Parse the answer's final `**Sources:**` section into lines.
2. Resolve each line to a parent_id: literal id → filename match against the
   `source` part of `retrieval_keys` → `Parent ID:` hint → fuzzy title
   (Jaccard ≥ 0.6).
3. Lines that resolve to NOTHING are DROPPED (paper-qa: unanchored citation
   = hallucination risk; deterministic, not LLM-discipline).
4. Kept lines are lexically spot-checked against the parent's full text;
   near-zero overlap gets a ⚠️ annotation (not dropped — paraphrase-heavy
   but legitimate citations survive).
5. An HTML comment records how many lines were removed.

## Agent-facing changes

- `search_child_chunks` results now carry a `channels:` tag (`vec`, `bm25`,
  or `bm25+vec`); dual-channel hits are the strongest matches. Gene symbols
  (Ephb1, ephrin-B1, Mab21l2) match exactly through the BM25 channel.
- Two new tools: `find_papers_citing(source)` (forward snowball) and
  `get_paper_references(source)` (backward, with in-library resolution).
- `fallback_response` appends an **Evidence coverage** block and receives an
  EVIDENCE GATE instruction when the session's retrievals returned nothing.
- Orchestrator/fallback prompts updated accordingly.

## Metrics (evaluate.py)

`citation_faithfulness(answer, retrieval_keys)` →
`{n_source_lines, n_whitelisted, whitelist_rate, n_dropped, n_annotated,
lexical_scores}` — same code path as the live guard, so eval scores reflect
what production enforces.

## Chunking changes (next ingest)

`create_parent_chunks` now also emits caption parents
(`chunk_type='figure_caption'|'table_caption'`, section tag mirrors it,
bypasses MIN_PARENT_SIZE). Filter with:
`where={"chunk_type": "figure_caption"}` or
`{"$and": [{"article_type": "review"}, {"chunk_type": "figure_caption"}]}`.
Existing parents are unchanged (`chunk_type='section'` only appears on
newly built ones); re-index a paper with `scripts/index_single_paper.py`.

## Tests

`python3 src/test_guard_modules.py` → 21/21 (offline: no LLM/network/ChromaDB
writes; FTS roundtrip uses a tmp library via BIB_RAG_ROOT). Both runners work:
`pytest src/test_guard_modules.py` and `python3 src/test_guard_modules.py`.

---

# Round 2 — CJK bigrams, identifier normalization, broadened retry (2026-08-28)

Three first-tier mechanisms from the Zotero-repo v2 survey
(`/Disk_bot/notes/zotero_RAG/`), all zero-LLM:

## B2 — CJK character-bigram FTS channel (`src/hybrid_search.py` v2)

FTS5's unicode61 tokenizer indexes a whole CJK run as ONE token, so Chinese
text was unsearchable by sub-phrase. Schema v2 adds a `cjk_text` column
holding a bigram-segmented copy of each chunk's CJK runs
(`轴突导向` → `轴突 突导 导向`); queries are split the same way and both
channels run. v1 indexes migrate in place on first write (rename → rebuild →
copy); latin queries are unaffected. Note: the current eph_rag corpus is
pure English — the CJK channel is defense for future Chinese-language
ingest (notes, Chinese review articles).

## B5 — Identifier normalization (`src/identifiers.py`, new)

Canonical forms for DOI / arXiv / PMID: strips `https://doi.org/`, `doi:`,
case, and version suffixes (`...002v2` → `...002`); arXiv old+new style;
PMID digits. Wired into:
- `zotero_match.doi_match` — two-tier: exact canonical equality, then
  truncated-metadata prefix (short side being a strict prefix of the long
  side, ≥8 chars). **Fixed a real bug**: the old char-prefix comparison
  counted same-JOURNAL DOIs (`10.1016/j.ydbio.2021.01.002` vs
  `...2019.05.007`) as the same paper — the shared `10.1016/j.ydbio.` prefix
  is 18 chars, far above any fixed threshold.
- `reference_graph._resolve_source` — snowball tools now accept a bare or
  URL-wrapped DOI and resolve it to the library paper via canonical match
  against parent_store metadata.

## B1 — Deterministic broadened retry (`src/broaden.py`, new)

Weak-result detection + one automatic re-search, borrowed from
zotero-redisearch-rag's `should_broaden_retrieval`. Four zero-LLM signals on
the first-pass results: too few chunks (<2), too few chars (<300), weak best
similarity (<0.40), narrative-starved (<30% prose sentences). Trigger → one
re-search with the where-filter dropped and limit ×3; the better pass wins;
output is prefixed with `[BROADENED]` + the triggering signals so the agent
knows the first pass was thin. Escalation ladder and `broaden_signature`
dedup prevent loops. Env switches:

- `BROADEN_RETRY=0` — disable
- `BROADEN_MIN_RESULTS` (2), `BROADEN_MIN_CHARS` (300),
  `BROADEN_MIN_SIM` (0.40), `BROADEN_MIN_NARRATIVE` (0.30)

Also fixed while wiring: relative imports (`from .broaden import …`) raised
ImportError when `src/` is on sys.path directly (CLI/test entry), silently
killing the whole broaden branch — all intra-`src` imports now fall back to
top-level imports (`agent_tools.py` ×3, matches the existing
`hybrid_search` pattern).