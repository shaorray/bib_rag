#!/usr/bin/env python3
"""
hybrid_search.py — BM25 (FTS5) + dense-vector hybrid retrieval with RRF.

Borrowed mechanism: DocsGPT hybrid_rag (RRF fusion, k=60) + zotero-rag-cli
(BM25 FTS5 over local chunks) + seerai (RRF consensus k=60). See
/Disk_bot/notes/citation_rag/ and /Disk_bot/notes/zotero_RAG/.

Why: bib_rag's retrieval was dense-only (bge-m3). Gene symbols / receptor
names (Ephb1, ephrin-B1, Mab21l2) are strong LEXICAL signals that embedding
models routinely miss. BM25 catches them exactly; RRF (k=60) fuses both
rankings without score calibration.

Design:
  - FTS5 index lives at <data_root>/data/fts_index.db — SEPARATE from
    chroma.sqlite3 (never touches the vector DB).
  - Index is rebuilt incrementally from the parent_store JSONs (same source
    of truth the parents come from) — children are re-chunked with the exact
    same chunking.py logic so chunk texts match what the agent retrieves.
  - search_hybrid(query, limit, where_keys) → RRF-ranked list with the SAME
    output format as ToolFactory.search_child_chunks, so the agent needs no
    prompt changes.

Usage:
    from hybrid_search import HybridIndex
    idx = HybridIndex()                    # opens/creates FTS5 db
    idx.rebuild(progress=True)             # full rebuild from parent_store
    idx.upsert_source(source_name)         # incremental: one paper
    hits = idx.bm25_search("Ephb1 ephrin-B1 repulsion", limit=10)
"""
from __future__ import annotations

import os
import re
import sys
import json
import math
import sqlite3
import hashlib
from typing import Dict, List, Optional, Tuple

try:
    from .kb_config import get_config
    from . import chunking
except ImportError:  # direct script execution (scripts/build_fts_index.py)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:  # bib_rag-package-try
        from .kb_config import get_config
    except ImportError:  # flat (loose-script mode)
        from kb_config import get_config
    try:  # bib_rag-package-try
        from .chunking import chunking
    except ImportError:  # flat (loose-script mode)
        import chunking

# RRF constant (k=60 is the literature standard, used by DocsGPT/seerai).
# Env-tunable for A/B ablation (eval harness sets env before import).
RRF_K = int(os.environ.get("RRF_K", "60"))

# FTS5 query sanitization: strip operators/quotes that would error
_FTS_BAD = re.compile(r'["\^{}()\[\]*:]')

# CJK ranges: unified ideographs + kana + compatibility ideographs.
# Chinese/Japanese text has no spaces, so FTS5's unicode61 tokenizer indexes a
# whole CJK run as ONE token — "轴突导向" never matches "轴突导向因子".
# Fix (borrowed from RAG-Assistant-for-Zotero's CJK char-level BM25, see
# /Disk_bot/notes/zotero_RAG/03): index a bigram-segmented copy of the text in
# a separate `cjk_text` column; queries are likewise cut into bigrams.
_CJK_RE = re.compile(r'[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+')

# Schema version: v2 adds the cjk_text column. A v1 index keeps working for
# latin queries; the first WRITE (upsert/rebuild) migrates it in place.
# v3 adds the meta_text column (title/authors/journal/year) — the metadata
# channel. Entity-flavored queries ("Koolpe 2002 ephrin mimetic", "the 2018
# Nature maternal-fetal paper") match NOTHING in body text; this channel
# catches them lexically. A v1/v2 index keeps working; first WRITE migrates.
_SCHEMA_VERSION = 3


def _cjk_prepare(text: str) -> str:
    """CJK runs → space-separated character bigrams (unigram for len-1 runs).
    Latin/other text is dropped (the main `text` column covers it)."""
    out = []
    for run in _CJK_RE.findall(text or ""):
        if len(run) == 1:
            out.append(run)
        else:
            out.extend(run[i:i + 2] for i in range(len(run) - 1))
    return " ".join(out)


# -- meta_text construction (metadata channel) --------------------------------
#
# meta_text is a compact lexical fingerprint of a paper's metadata, indexed
# in EVERY row of that source (see _SCHEMA_VERSION=3). It exists so FTS can
# match entity-flavored queries ("Koolpe 2002", "the 2018 Nature
# maternal-fetal paper") that never appear in body text.

# How many authors to index. First author carries most entity queries
# ("Koolpe 2002"); beyond ~5 the string bloats every row with rare surnames
# that can false-match OTHER papers' queries.
_META_N_AUTHORS = 5

# Authors field separators: Zotero-style "Paulson,Alicia F.; Fang,Xiang" →
# tokens. Splitting on ; gives one entry per author.
_AUTHOR_SPLIT = re.compile(r"[;,]")


def _author_tokens(authors: str) -> str:
    """'Paulson,Alicia F.; Fang,Xiang; ...' → 'paulson alicia fang xiang' ...

    Keeps only the first _META_N_AUTHORS authors. Surnames and given names
    are all indexed (unicode61 lowercases): queries hit on either form
    ("Koolpe" surname / "Michael" given name both match).
    """
    entries = [a.strip() for a in _AUTHOR_SPLIT.split(authors or "")
               if a.strip()][:_META_N_AUTHORS]
    tokens = []
    for e in entries:
        # "Paulson,Alicia F." — after the ; split the comma inside the entry
        # is surname/given separator; both halves are worth indexing.
        tokens.extend(t for t in re.split(r"[,\s]+", e) if t)
    return " ".join(tokens)


def _join_meta_text(meta: Dict[str, str]) -> str:
    """title + author tokens + journal + year → one indexed string.

    Order is irrelevant to BM25; what matters is compactness (the string is
    duplicated into every chunk row of the source) and tokenization that
    unicode61 can split (no glued compound like 'Koolpe;Michael').
    """
    parts: List[str] = []
    if meta.get("title"):
        parts.append(meta["title"])
    if meta.get("authors"):
        tok = _author_tokens(meta["authors"])
        if tok:
            parts.append(tok)
    if meta.get("journal"):
        parts.append(meta["journal"])
    if meta.get("year"):
        parts.append(str(meta["year"]))
    return " ".join(parts)[:2000]


_FM_FIELD = {
    "title": re.compile(r"^Title:\s*(.+)$", re.M),
    "authors": re.compile(r"^Authors?:\s*(.+)$", re.M),
    "year": re.compile(r"^Year:\s*(\d{4})", re.M),
    "journal": re.compile(r"^Journal:\s*(.+)$", re.M),
}


def _meta_from_frontmatter(text: str) -> Dict[str, str]:
    """Scrape Title:/Authors:/Year:/Journal: labels from a parent's content
    head (fallback when the JSON meta dict lacks fields). The chunking
    pipeline's front-matter block (if present) sits at the top of parent 0."""
    out: Dict[str, str] = {}
    for k, pat in _FM_FIELD.items():
        m = pat.search(text or "")
        if m:
            out[k] = m.group(1).strip()
    return out


# Lastname_Year_Journal_Title filename pattern (e.g.
# "Koolpe_2002_The Journal of Biological Chemistry_1589 - 1619").
_FNAME_YEAR = re.compile(r"^(?:[\w\-.]+?)[_\s]+((?:19|20)\d{2})[_\s]+(.*)$")


def _meta_from_filename(source: str) -> Dict[str, str]:
    """Last-ditch fallback: derive a year (and title tail) from the source
    filename. Returns only what it can infer — never fabricates fields."""
    out: Dict[str, str] = {}
    m = _FNAME_YEAR.match(source or "")
    if m:
        out["year"] = m.group(1)
        rest = m.group(2).strip()
        # "Journal_Name_Tail" → journal is the segment before the last one
        segs = [s.strip() for s in rest.split("_") if s.strip()]
        if len(segs) >= 2:
            out["journal"] = segs[0]
            out["title"] = " ".join(segs[1:])
        elif segs:
            out["title"] = segs[0]
    return out


def _split_latin_cjk(query: str) -> Tuple[str, List[str]]:
    """Split a query into (escaped-latin-terms, cjk-runs)."""
    runs = _CJK_RE.findall(query or "")
    latin = _CJK_RE.sub(" ", query or "")
    return _fts_escape(latin), runs


def _fts_escape(query: str) -> str:
    """Make a user/LLM query safe for MATCH. Terms are AND-ed implicitly by
    joining with spaces; empty result falls back to OR in bm25_search.

    CRITICAL: in FTS5 a bare hyphen inside a token is parsed as the NOT
    operator ('ephrin-B1' → ephrin AND NOT B1 → 'no such column B1' error),
    so any hyphenated token is wrapped in double quotes (exact token match).
    """
    q = _FTS_BAD.sub(" ", query)
    parts = []
    for tok in q.split():
        if "-" in tok:
            # quote hyphenated tokens: gene names like ephrin-B1 / IL-6 / Sfprd
            parts.append(f'"{tok}"')
        elif tok:
            parts.append(tok)
    q = " ".join(parts)
    return re.sub(r"\s+", " ", q).strip()


def fts_tokenize(text: str) -> str:
    """Reuse citation_guard's tokenizer philosophy (word chars + hyphens).
    FTS5 unicode61 tokenizer handles the rest; we only strip noise."""
    return text


class HybridIndex:
    """FTS5 index over child chunks, mirroring the ChromaDB child population."""

    def __init__(self, fts_path: Optional[str] = None):
        cfg = get_config()
        self.fts_path = fts_path or cfg.get("fts_index_path") or os.path.join(
            cfg["data_dir"], "fts_index.db")
        if self.fts_path != ":memory:":
            os.makedirs(os.path.dirname(self.fts_path), exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None

    # -- connection / schema -------------------------------------------------

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            conn = sqlite3.connect(self.fts_path)
            conn.execute("PRAGMA journal_mode=WAL")
            self._ensure_schema(conn)
            conn.commit()
            self._conn = conn
        return self._conn

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        """Create (or migrate v1→v2→v3) the FTS schema.

        FTS5 tables can't ALTER: migration = rename old table → rebuild with
        the new schema → copy rows (recomputing derived columns on the fly).
        """
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS child_fts USING fts5(
                text,
                cjk_text,
                meta_text,
                parent_id UNINDEXED,
                source UNINDEXED,
                section UNINDEXED,
                chunk_idx UNINDEXED,
                tokenize = 'unicode61 remove_diacritics 2'
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fts_meta (
                source TEXT PRIMARY KEY,
                n_children INTEGER,
                built_at TEXT DEFAULT (datetime('now')),
                rep_parent_id TEXT,
                rep_section TEXT,
                rep_text TEXT
            )
        """)
        # fts_meta is a regular table (not FTS5), so missing v3 columns can be
        # added in place — no table rebuild needed.
        _meta_cols = [r[1] for r in conn.execute("PRAGMA table_info(fts_meta)")]
        for _col in ("rep_parent_id", "rep_section", "rep_text"):
            if _col not in _meta_cols:
                conn.execute(f"ALTER TABLE fts_meta ADD COLUMN {_col} TEXT")
        # Detect old schema (missing cjk_text/meta_text) and migrate in place.
        try:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(child_fts)")]
        except sqlite3.Error:
            cols = []
        if cols and ("cjk_text" not in cols or "meta_text" not in cols):
            conn.executescript("""
                DROP TABLE IF EXISTS child_fts_v1_migrate;
                ALTER TABLE child_fts RENAME TO child_fts_v1_migrate;
            """)
            conn.execute("""
                CREATE VIRTUAL TABLE child_fts USING fts5(
                    text,
                    cjk_text,
                    meta_text,
                    parent_id UNINDEXED,
                    source UNINDEXED,
                    section UNINDEXED,
                    chunk_idx UNINDEXED,
                    tokenize = 'unicode61 remove_diacritics 2'
                )
            """)
            has_cjk = "cjk_text" in cols
            has_meta = "meta_text" in cols
            if has_cjk and has_meta:
                rows = conn.execute(
                    "SELECT text, cjk_text, meta_text, parent_id, source, "
                    "section, chunk_idx FROM child_fts_v1_migrate").fetchall()
                conn.executemany(
                    "INSERT INTO child_fts(text, cjk_text, meta_text, "
                    "parent_id, source, section, chunk_idx) "
                    "VALUES (?,?,?,?,?,?,?)", rows)
            else:
                # v1/v2 -> v3: recompute every derived column from text.
                # meta_text can't be recomputed here (it needs parent_store
                # metadata) — filled by the next upsert/rebuild of each
                # source; until then it's empty and the meta channel simply
                # misses for that source.
                rows = conn.execute(
                    "SELECT text, parent_id, source, section, chunk_idx "
                    "FROM child_fts_v1_migrate").fetchall()
                conn.executemany(
                    "INSERT INTO child_fts(text, cjk_text, meta_text, "
                    "parent_id, source, section, chunk_idx) "
                    "VALUES (?,?,?,?,?,?,?)",
                    [(t, _cjk_prepare(t), "", p, s, sec, idx)
                     for (t, p, s, sec, idx) in rows])
            conn.execute("DROP TABLE child_fts_v1_migrate")
            conn.commit()

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # -- population ----------------------------------------------------------

    def _children_for_source(self, source: str) -> List[dict]:
        """Re-chunk one source's parents into children (same logic as build)."""
        cfg = get_config()
        store = cfg["parent_store_dir"]
        # 0. Exact stem: rebuild() passes the parent filename stem verbatim.
        #    Stems legally contain unicode (‐ U+2010, curly quotes, é, 等 …)
        #    that sanitize() below would rewrite to '_', so try the literal
        #    file first — this is the only correct lookup for stem callers.
        path = os.path.join(store, f"{source}.json")
        safe = None
        if not os.path.exists(path):
            safe = re.sub(r"[^\w\-]", "_", source)[:100]
            path = os.path.join(store, f"{safe}.json")
        if not os.path.exists(path):
            # Long-titled papers: sanitize()[:100] truncates the stem, so the
            # exact filename doesn't exist. Fall back to prefix match — the
            # truncated safe name is still a unique prefix of the real file.
            assert safe is not None  # only reachable when sanitize path ran
            cands = [f for f in os.listdir(store)
                     if f.startswith(safe[:80]) and f.endswith(".json")]
            if len(cands) == 1:
                path = os.path.join(store, cands[0])
            elif len(cands) > 1:
                # prefer the shortest (closest to the truncated name)
                path = os.path.join(store, min(cands, key=len))
            else:
                return []
        with open(path, encoding="utf-8") as f:
            parents = json.load(f)
        children: List[dict] = []
        for p in parents:
            kids = chunking.create_child_chunks(
                p["content"], p["parent_id"], p["source"], p["section"])
            children.extend(kids)
        return children

    def upsert_source(self, source: str, children: Optional[List[dict]] = None) -> int:
        """(Re)index one paper's children. Returns child count."""
        if children is None:
            children = self._children_for_source(source)
        meta_text = self._meta_text_for_source(source, children)
        c = self.conn
        c.execute("DELETE FROM child_fts WHERE source = ?", (source,))
        c.execute("DELETE FROM fts_meta WHERE source = ?", (source,))
        rows = [
            # NOTE: rows are keyed by the `source` ARGUMENT (parent_store
            # stem = chroma's source domain), NOT ch["source"] — parent
            # JSONs can carry an older source spelling ("12351647.md") that
            # mismatches the stem ("12351647_md"), silently breaking the
            # fts_meta JOIN and where_source filters (found 2026-08-31:
            # 3002/3002 sources orphaned this way).
            (ch["text"], _cjk_prepare(ch["text"]), meta_text, ch["parent_id"],
             source, ch["section"], ch.get("idx", i))
            for i, ch in enumerate(children)
        ]
        c.executemany(
            "INSERT INTO child_fts(text, cjk_text, meta_text, parent_id, source, section, chunk_idx) "
            "VALUES (?,?,?,?,?,?,?)", rows)
        rep = children[0] if children else {}
        rep_text = (rep.get("text") or "")[:2000]
        if meta_text and rep_text:
            # Entity-enriched carrier: the cross-encoder reranker scores
            # query vs this string. A bare body-text chunk loses entity
            # queries ("Koolpe Pasquale 2002" appears in NO chunk text);
            # prepending the meta fingerprint lets the reranker see
            # title/authors/journal/year. rep_text is never displayed —
            # the agent fetches the real parent by parent_id — so this
            # enrichment only affects rerank scoring.
            rep_text = f"{meta_text}. {rep_text}"[:2500]
        c.execute(
            "INSERT INTO fts_meta(source, n_children, rep_parent_id, rep_section, rep_text) "
            "VALUES (?,?,?,?,?)",
            (source, len(rows), rep.get("parent_id", ""), rep.get("section", ""),
             rep_text))
        c.commit()
        return len(rows)

    # -- metadata channel -----------------------------------------------------

    @staticmethod
    def _meta_text_for_source(source: str, children: Optional[List[dict]] = None) -> str:
        """Build the metadata string indexed in the meta_text column.

        Source of truth: the parent_store JSON's first-parent `meta` dict
        (title/authors/year/journal), written by the metadata pipeline
        (bind_zotero / backfill_all / repair_meta). The string is compact
        and tokenized (surnames split out) so FTS unicode61 can lexically
        match entity-flavored queries like "Koolpe 2002 ephrin mimetic
        peptide" or "the 2018 Nature maternal-fetal paper" that body text
        never contains.

        Fallbacks (rare — a 2026-08-31 audit found 0/3002 eph sources with
        an empty meta dict, but the index must survive ANY store state):
        front-matter scrape from the first child's content, then the
        Lastname_Year_Journal filename pattern.
        """
        meta: Dict[str, str] = {}
        # Path 1: parent_store JSON meta dict (authoritative).
        try:
            cfg = get_config()
            path = os.path.join(cfg["parent_store_dir"], f"{source}.json")
            if not os.path.exists(path):
                safe = re.sub(r"[^\w\-]", "_", source)[:100]
                path = os.path.join(cfg["parent_store_dir"], f"{safe}.json")
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    parents = json.load(f)
                if parents:
                    m = parents[0].get("meta") or {}
                    for k in ("title", "authors", "year", "journal"):
                        v = (m.get(k) or "").strip()
                        if v and k not in meta:
                            meta[k] = v
        except (OSError, json.JSONDecodeError):
            pass
        # Path 2: front-matter scrape from children content (fallback when
        # the meta dict is missing fields, e.g. legacy pre-pipeline papers).
        if len(meta) < 4 and children:
            fm = _meta_from_frontmatter(children[0]["text"][:2000])
            for k in ("title", "authors", "year", "journal"):
                if k not in meta and fm.get(k):
                    meta[k] = fm[k]
        # Path 3: the source filename itself (Lastname_Year_Journal_Title
        # pattern) — the last-ditch lexical hint when both above are empty.
        if "title" not in meta and "authors" not in meta:
            fm = _meta_from_filename(source)
            for k in ("title", "authors", "year", "journal"):
                if k not in meta and fm.get(k):
                    meta[k] = fm[k]
        return _join_meta_text(meta)

    def _meta_hits(self, match_q: str, limit: int,
                   where_source: Optional[str] = None) -> List[dict]:
        """Run a MATCH against meta_text and return REPRESENTATIVE chunks
        (one per source) instead of raw rows — a meta hit has no specific
        chunk semantics; the source's first child is the carrier.

        A meta_text MATCH hits EVERY row of the source (they all carry the
        same meta_text), so raw rows would flood the results with ONE
        source. GROUP BY source collapses to one hit per source; the
        carrier chunk (parent 0 = title/front-matter region) is read from
        the fts_meta side table — O(1) per hit, no parent_store re-chunk.
        """
        c = self.conn
        sql = ("SELECT f.source, m.rep_parent_id, m.rep_section, m.rep_text "
               "FROM child_fts f JOIN fts_meta m ON f.source = m.source "
               "WHERE meta_text MATCH ?")
        params: list = [match_q]
        if where_source:
            sql += " AND f.source = ?"
            params.append(where_source)
        # MIN(bm25(...)) is illegal (FTS5 aux functions can't nest inside
        # aggregates); MIN(rank) does the same job — `rank` IS the bm25
        # ordering (fts5 rank = -bm25 by default; more negative = better).
        sql += " GROUP BY f.source ORDER BY MIN(rank) LIMIT ?"
        # All rows of one source share one meta_text, so their bm25 ties —
        # MIN(bm25) is the GROUP BY tiebreak, not a per-chunk ranking.
        params.append(int(limit))
        rows = c.execute(sql, params).fetchall()
        out: List[dict] = []
        for src, pid, sec, text in rows:
            if not text:
                # Pre-rep-column index (v2 fts_meta): fall back to the source's
                # first FTS row as carrier so the hit isn't lost.
                raw = c.execute(
                    "SELECT parent_id, section, text FROM child_fts "
                    "WHERE source = ? ORDER BY chunk_idx LIMIT 1", (src,)).fetchone()
                if raw:
                    pid, sec, text = raw[0], raw[1], raw[2]
                else:
                    continue
            out.append({"parent_id": pid or src, "source": src,
                        "section": sec or "", "text": text or "",
                        "bm25": 0.0, "meta_hit": True})
        return out

    def rebuild(self, progress: bool = False) -> int:
        """Full rebuild from parent_store. Returns total children indexed."""
        cfg = get_config()
        store = cfg["parent_store_dir"]
        total = 0
        names = sorted(f for f in os.listdir(store) if f.endswith(".json"))
        c = self.conn
        c.execute("DELETE FROM child_fts")
        c.execute("DELETE FROM fts_meta")
        c.commit()
        for i, fname in enumerate(names):
            source = fname[:-5]  # strip .json
            try:
                n = self.upsert_source(source)
            except (OSError, json.JSONDecodeError) as e:
                if progress:
                    print(f"  [fts] skip {fname}: {e}")
                continue
            total += n
            if progress and (i + 1) % 200 == 0:
                print(f"  [fts] {i+1}/{len(names)} papers, {total} children")
        if progress:
            print(f"  [fts] done: {len(names)} papers, {total} children")
        return total

    def indexed_sources(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM fts_meta").fetchone()
        return row[0] if row else 0

    # -- retrieval -----------------------------------------------------------

    def bm25_search(self, query: str, limit: int = 10,
                    where_source: Optional[str] = None) -> List[dict]:
        """BM25 ranking over the FTS5 index. Falls back to OR-terms when the
        implicit AND returns nothing (common with long LLM queries).

        CJK handling: the query is split into latin terms (matched against
        `text`) and CJK runs (matched as character bigrams against
        `cjk_text`); both sub-queries run and results are inter-leaved.

        Metadata channel (v3): the same latin terms are additionally
        matched against the `meta_text` column (title/authors/journal/year
        fingerprint). Meta hits return one REPRESENTATIVE chunk per source
        (see _meta_hits); they join the result list after the body-text
        hits, marked `meta_hit=True`, and are fused by RRF downstream like
        any other channel entry.
        """
        q = _fts_escape(query)
        latin_q, cjk_runs = _split_latin_cjk(query)
        if not q and not cjk_runs:
            return []
        c = self.conn
        # Bigram-escape CJK runs: each run becomes a quoted bigram sequence.
        cjk_match_parts = []
        for run in cjk_runs:
            grams = [run[i:i + 2] for i in range(len(run) - 1)] or [run]
            cjk_match_parts.append(
                " ".join(f'"{g}"' for g in grams))

        def _run(match_q: str, column: str = "child_fts") -> List[dict]:
            if not match_q:
                return []
            sql = ("SELECT parent_id, source, section, text, "
                   f"bm25({column}) AS score FROM {column} "
                   f"WHERE {column} MATCH ?")
            params: list = [match_q]
            if where_source:
                sql += " AND source = ?"
                params.append(where_source)
            sql += " ORDER BY score LIMIT ?"
            params.append(int(limit))
            rows = c.execute(sql, params).fetchall()
            return [{"parent_id": r[0], "source": r[1], "section": r[2],
                     "text": r[3], "bm25": -r[4]} for r in rows]

        results: List[dict] = []
        try:
            # Cross-sub-run double-count fix (the ONLY dedup BM25 needs):
            # the latin and CJK sub-queries can return the SAME chunk twice
            # (one hit per sub-run). That is one piece of evidence counted
            # twice — drop the repeat, keep the first occurrence. NOTE: the
            # same PARENT appearing with DIFFERENT children is NOT a
            # double-count — chunk frequency is a relevance signal (a paper
            # with 4 matching sections is probably the right paper; goldset
            # ablation 2026-08-31: collapsing it cost recall 0.742→0.700 and
            # MRR 0.638→0.528), so parent-level dedup deliberately does NOT
            # happen here (see rrf_fuse RRF_DEDUP=off default).
            seen_chunks: set = set()
            def _extend(res: List[dict]):
                for r in res:
                    key = (r.get("parent_id"), r.get("text", "")[:120])
                    if key in seen_chunks:
                        continue
                    seen_chunks.add(key)
                    results.append(r)
            if latin_q:
                res = _run(latin_q)
                if not res and " " in latin_q:
                    res = _run(" OR ".join(latin_q.split()))
                _extend(res)
            for run_q in cjk_match_parts:
                res = _run(run_q)
                if not res and " " in run_q:
                    res = _run(" OR ".join(
                        p.strip('"') for p in run_q.split()))
                _extend(res)
        except sqlite3.OperationalError:
            # match-syntax issue on channel shouldn't kill the other —
            # but a partial failure here means results may be partially
            # populated; returning them is still better than nothing.
            pass
        return results

    # -- RRF fusion ----------------------------------------------------------

    def rrf_fuse(self, vector_results: List[dict], bm25_results: List[dict],
                 top_k: int = 6,
                 meta_results: Optional[List[dict]] = None) -> List[dict]:
        """Reciprocal Rank Fusion of the ranked lists.

        vector_results / bm25_results entries need: parent_id, source,
        section, text (+ similarity / bm25 for display).
        RRF score = Σ 1/(k + rank) over lists where the chunk appears
        (k=60).

        Metadata channel (third list, optional): entries are the
        SOURCE-LEVEL representative chunks from _meta_hits, carrying an
        entity-enriched carrier text (meta fingerprint prepended). When
        the same parent_id already fused from vec/bm25, the meta_hit
        entry REPLACES the carrier — body chunks never contain author/year
        ("Koolpe Pasquale 2002" appears in no body text), so the enriched
        carrier is the reranker's only view of the entity signal. The
        parent_id stays the same, so the RRF score still accumulates
        across channels as usual.

        RRF semantics per channel (env RRF_DEDUP, ablated on the gold set):
          - "bm25" (default): the dense channel KEEPS chunk-level ranks —
            several children of one parent in the dense top-N is a
            relevance-frequency signal (a paper with 4 matching sections is
            probably the right source), NOT a bug. Only the bm25 channel is
            collapsed to parent level, because its latin+CJK sub-runs can
            return the SAME chunk twice — a true double-count of one piece
            of evidence.
          - "full": both channels parent-deduped (strict RRF item semantics).
          - "off": pre-2026-08-31 behavior (no dedup at all).
        The fused key is parent_id regardless: chroma's historical chunk
        texts and the FTS index drift within the same parent (generational
        drift), so any text-based key silently loses dual-channel signal.
        Parent-level is the correct evidence-unit anyway: two children of
        the same parent are two views of one document.
        """
        mode = os.environ.get("RRF_DEDUP", "off")

        def _channel_dedup(results, dedup: bool):
            """parent_id → its best-ranked entry (first occurrence wins)."""
            if not dedup:
                return results
            seen, out = set(), []
            for r in results:
                k = r.get("parent_id", "")
                if k in seen:
                    continue
                seen.add(k)
                out.append(r)
            return out

        channels = [("vec", vector_results, mode == "full"),
                     ("bm25", bm25_results, mode != "off")]
        if meta_results:
            # Meta hits are one-per-source by construction — dedup is a
            # no-op semantically, but passing dedup=False keeps them
            # rank-weighted as their own channel.
            channels.append(("meta", meta_results, False))

        fused: dict = {}
        for channel, results, dedup in channels:
            for rank, r in enumerate(_channel_dedup(results, dedup), 1):
                k = r.get("parent_id", "")
                d = fused.setdefault(k, {"entry": r, "rrf": 0.0, "hits": []})
                if r.get("meta_hit") and not d["entry"].get("meta_hit"):
                    # meta carrier wins the slot: it's the entity-enriched
                    # view the reranker needs (see docstring).
                    d["entry"] = r
                d["rrf"] += 1.0 / (RRF_K + rank)
                d["hits"].append(channel)
        ranked = sorted(fused.values(), key=lambda d: -d["rrf"])[:top_k]
        out = []
        for d in ranked:
            e = dict(d["entry"])
            e["rrf"] = round(d["rrf"], 5)
            e["channels"] = "+".join(sorted(set(d["hits"])))
            out.append(e)
        return out

    def search(self, query: str, vector_results: List[dict], top_k: int = 6,
               where_source: Optional[str] = None) -> List[dict]:
        """One-call hybrid: caller supplies dense results, we add BM25 + meta
        + fuse. If FTS index is empty, returns vector_results unchanged
        (graceful degradation — the agent never breaks because BM25 is
        missing).

        Pool symmetry: BOTH body channels contribute top_k*3 candidates so
        RRF ranks are earned, not biased by an asymmetric pool depth (a
        6-deep dense list vs an 18-deep bm25 list hands the fusion to
        bm25's tail). Callers already pass limit-sized dense lists, so the
        widening to top_k*3 happens in the RETRIEVAL wrapper, not here —
        anything past top_k*3 entries is dropped to keep both channels
        equal-footed.

        Metadata channel: the latin query terms are ALSO matched against
        meta_text (title/authors/journal/year). Hits are source-level
        representative chunks; they join RRF as a THIRD channel, so an
        entity query ("Koolpe Pasquale 2002 …") that both body channels
        miss can still surface the right paper.
        """
        if self.indexed_sources() == 0:
            return vector_results[:top_k]
        bm = self.bm25_search(query, limit=top_k * 3, where_source=where_source)
        vec = vector_results[:top_k * 3]
        meta: Optional[List[dict]] = None
        try:
            latin_q, _runs = _split_latin_cjk(query)
            if latin_q:
                meta = self._meta_hits(latin_q, limit=top_k,
                                       where_source=where_source)
                if not meta:
                    meta = None
        except sqlite3.Error:
            meta = None
        return self.rrf_fuse(vec, bm, top_k=top_k, meta_results=meta)


def is_available() -> bool:
    """True when the FTS index exists and has content."""
    try:
        return HybridIndex().indexed_sources() > 0
    except sqlite3.Error:
        return False