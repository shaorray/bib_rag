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
    from kb_config import get_config
    import chunking

# RRF constant (k=60 is the literature standard, used by DocsGPT/seerai)
RRF_K = 60

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
_SCHEMA_VERSION = 2


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
        """Create (or migrate v1→v2) the FTS schema.

        FTS5 tables can't ALTER: migration = rename old table → rebuild with
        the new schema → copy rows (recomputing cjk_text on the fly)."""
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS child_fts USING fts5(
                text,
                cjk_text,
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
                built_at TEXT DEFAULT (datetime('now'))
            )
        """)
        # Detect v1 (no cjk_text column) and migrate in place.
        try:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(child_fts)")]
        except sqlite3.Error:
            cols = []
        if cols and "cjk_text" not in cols:
            conn.executescript("""
                DROP TABLE IF EXISTS child_fts_v1_migrate;
                ALTER TABLE child_fts RENAME TO child_fts_v1_migrate;
            """)
            conn.execute("""
                CREATE VIRTUAL TABLE child_fts USING fts5(
                    text,
                    cjk_text,
                    parent_id UNINDEXED,
                    source UNINDEXED,
                    section UNINDEXED,
                    chunk_idx UNINDEXED,
                    tokenize = 'unicode61 remove_diacritics 2'
                )
            """)
            rows = conn.execute(
                "SELECT text, parent_id, source, section, chunk_idx "
                "FROM child_fts_v1_migrate").fetchall()
            conn.executemany(
                "INSERT INTO child_fts(text, cjk_text, parent_id, source, section, chunk_idx) "
                "VALUES (?,?,?,?,?,?)",
                [(t, _cjk_prepare(t), p, s, sec, idx)
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
        c = self.conn
        c.execute("DELETE FROM child_fts WHERE source = ?", (source,))
        c.execute("DELETE FROM fts_meta WHERE source = ?", (source,))
        rows = [
            (ch["text"], _cjk_prepare(ch["text"]), ch["parent_id"], ch["source"],
             ch["section"], ch.get("idx", i))
            for i, ch in enumerate(children)
        ]
        c.executemany(
            "INSERT INTO child_fts(text, cjk_text, parent_id, source, section, chunk_idx) "
            "VALUES (?,?,?,?,?,?)", rows)
        c.execute("INSERT INTO fts_meta(source, n_children) VALUES (?,?)",
                  (source, len(rows)))
        c.commit()
        return len(rows)

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
            # match-syntax issue on one channel shouldn't kill the other —
            # but a partial failure here means results may be partially
            # populated; returning them is still better than nothing.
            pass
        return results

    # -- RRF fusion ----------------------------------------------------------

    def rrf_fuse(self, vector_results: List[dict], bm25_results: List[dict],
                 top_k: int = 6) -> List[dict]:
        """Reciprocal Rank Fusion of the two ranked lists.

        vector_results / bm25_results entries need: parent_id, source,
        section, text (+ similarity / bm25 for display).
        RRF score = Σ 1/(k + rank) over lists where the chunk appears
        (k=60).

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

        fused: dict = {}
        for channel, results, dedup in (
                ("vec", vector_results, mode == "full"),
                ("bm25", bm25_results, mode != "off")):
            for rank, r in enumerate(_channel_dedup(results, dedup), 1):
                k = r.get("parent_id", "")
                d = fused.setdefault(k, {"entry": r, "rrf": 0.0, "hits": []})
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
        """One-call hybrid: caller supplies dense results, we add BM25 + fuse.
        If FTS index is empty, returns vector_results unchanged (graceful
        degradation — the agent never breaks because BM25 is missing).

        Pool symmetry: BOTH channels contribute top_k*3 candidates so RRF
        ranks are earned, not biased by an asymmetric pool depth (a 6-deep
        dense list vs an 18-deep bm25 list hands the fusion to bm25's tail).
        Callers already pass limit-sized dense lists, so the widening to
        top_k*3 happens in the RETRIEVAL wrapper, not here — anything past
        top_k*3 entries is dropped to keep both channels equal-footed."""
        if self.indexed_sources() == 0:
            return vector_results[:top_k]
        bm = self.bm25_search(query, limit=top_k * 3, where_source=where_source)
        vec = vector_results[:top_k * 3]
        return self.rrf_fuse(vec, bm, top_k=top_k)


def is_available() -> bool:
    """True when the FTS index exists and has content."""
    try:
        return HybridIndex().indexed_sources() > 0
    except sqlite3.Error:
        return False