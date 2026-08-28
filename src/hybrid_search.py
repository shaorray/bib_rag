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
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS child_fts USING fts5(
                    text,
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
            conn.commit()
            self._conn = conn
        return self._conn

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # -- population ----------------------------------------------------------

    def _children_for_source(self, source: str) -> List[dict]:
        """Re-chunk one source's parents into children (same logic as build)."""
        cfg = get_config()
        safe = re.sub(r"[^\w\-]", "_", source)[:100]
        path = os.path.join(cfg["parent_store_dir"], f"{safe}.json")
        if not os.path.exists(path):
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
            (ch["text"], ch["parent_id"], ch["source"], ch["section"], ch.get("idx", i))
            for i, ch in enumerate(children)
        ]
        c.executemany(
            "INSERT INTO child_fts(text, parent_id, source, section, chunk_idx) "
            "VALUES (?,?,?,?,?)", rows)
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
        implicit AND returns nothing (common with long LLM queries)."""
        q = _fts_escape(query)
        if not q:
            return []
        c = self.conn
        def _run(match_q: str) -> List[dict]:
            sql = ("SELECT parent_id, source, section, text, "
                   "bm25(child_fts) AS score FROM child_fts "
                   "WHERE child_fts MATCH ?")
            params: list = [match_q]
            if where_source:
                sql += " AND source = ?"
                params.append(where_source)
            sql += " ORDER BY score LIMIT ?"
            params.append(int(limit))
            rows = c.execute(sql, params).fetchall()
            return [{"parent_id": r[0], "source": r[1], "section": r[2],
                     "text": r[3], "bm25": -r[4]} for r in rows]
        try:
            res = _run(q)
        except sqlite3.OperationalError:
            return []
        if not res and " " in q:
            res = _run(" OR ".join(q.split()))
        return res

    # -- RRF fusion ----------------------------------------------------------

    def rrf_fuse(self, vector_results: List[dict], bm25_results: List[dict],
                 top_k: int = 6) -> List[dict]:
        """Reciprocal Rank Fusion of the two ranked lists.

        vector_results / bm25_results entries need: parent_id, source,
        section, text (+ similarity / bm25 for display).
        RRF score = Σ 1/(k + rank) over lists where the chunk appears
        (k=60). Dedup key: (parent_id, first 80 chars of text) — the same
        chunk from both channels must not double-count."""
        def _key(r):
            h = hashlib.md5((r.get("parent_id", "") + "|" + r.get("text", "")[:80]).encode())
            return h.hexdigest()

        fused: dict = {}
        for rank, r in enumerate(vector_results, 1):
            k = _key(r)
            d = fused.setdefault(k, {"entry": r, "rrf": 0.0, "hits": []})
            d["rrf"] += 1.0 / (RRF_K + rank)
            d["hits"].append("vec")
        for rank, r in enumerate(bm25_results, 1):
            k = _key(r)
            d = fused.setdefault(k, {"entry": r, "rrf": 0.0, "hits": []})
            d["rrf"] += 1.0 / (RRF_K + rank)
            d["hits"].append("bm25")
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
        degradation — the agent never breaks because BM25 is missing)."""
        if self.indexed_sources() == 0:
            return vector_results[:top_k]
        bm = self.bm25_search(query, limit=top_k * 3, where_source=where_source)
        return self.rrf_fuse(vector_results, bm, top_k=top_k)


def is_available() -> bool:
    """True when the FTS index exists and has content."""
    try:
        return HybridIndex().indexed_sources() > 0
    except sqlite3.Error:
        return False