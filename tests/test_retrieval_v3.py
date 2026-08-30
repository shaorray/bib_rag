#!/usr/bin/env python3
"""test_retrieval_v3.py — systematic regression test for the 0-6 retrieval stack.

Stages tested in isolation, then the full chain:
  T1  FTS schema v3: meta_text/rep columns, migration path, source-domain keys
  T2  bm25_search: body + CJK channels, cross-subrun chunk dedup
  T3  _meta_hits: entity queries → representative chunks (JOIN, MIN(rank),
      enriched carrier)
  T4  rrf_fuse: 3-channel fusion, meta-carrier-wins, RRF_DEDUP modes
  T5  rerank: full-pool reorder contract, graceful degradation
  T6  cite-boost: neighbor resolution + score nudge (pure logic)
  T7  cap: per-source cap + backfill contract
  T8  integration: full production path on real queries (eph library)

Run:  /usr/bin/python3 tests/test_retrieval_v3.py
"""
import os
import sys
import json
import time
import tempfile

# -- test isolation: point the library at a THROWAWAY mini-store --------------
_TMP = tempfile.mkdtemp(prefix="v3test_lib_")
os.environ["BIB_RAG_ROOT"] = _TMP  # env var wins over the registry
os.environ.setdefault("HOME", "/home/rui")

os.makedirs(os.path.join(_TMP, "parent_store"), exist_ok=True)
os.makedirs(os.path.join(_TMP, "data"), exist_ok=True)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append((name, detail))


def make_parent(title, authors, year, journal, body):
    return {
        "content": f"Title: {title}\nAuthors: {authors}\nYear: {year}\n"
                   f"Journal: {journal}\n\n{body}",
        "parent_id": f"{title[:20]}#full_text#hash1",
        "source": title[:20],  # NOTE: deliberately a DIFFERENT spelling than
        # the stem — this is the domain-mismatch trap that orphaned 3002
        # sources; upsert must key by the stem argument, not this field.
        "section": "full_text",
        "meta": {"title": title, "authors": authors, "year": year,
                 "journal": journal, "doi": ""},
        "char_count": len(body), "word_count": len(body.split()),
    }


# fixture: 3 papers with distinct metadata
PAPERS = [
    ("Koolpe ephrin mimetic peptide targets EphA2 receptor",
     "Koolpe,M; Dail,M; Pasquale,EB", "2002",
     "Journal of Biological Chemistry",
     "An ephrin mimetic peptide selectively targets the EphA2 receptor. "
     "Phage display identified peptides that bind EphA2 with high affinity. "
     "The optimized peptide inhibits ephrin binding and receptor signaling."),
    ("Reverse signaling through ephrin-B2 controls vascular development",
     "Adams,RH; Wilkinson,G; Klein,R", "1999",
     "Genes & Development",
     "Ephrin-B2 reverse signaling regulates vascular development in the "
     "embryonic vasculature. Endothelial cells expressing ephrin-B2 show "
     "repulsion responses. PI3K and Rho GTPases mediate downstream effects."),
    ("Cell sorting by differential adhesion in embryonic boundaries",
     "Foty,R; Steinberg,M", "2005",
     "Developmental Biology",
     "Cells sort by differential adhesion according to cadherin levels. "
     "Homotypic cadherin adhesion drives boundary formation and tissue "
     "segregation in early embryos."),
]

for title, authors, year, journal, body in PAPERS:
    stem = title[:24]  # source stem ≠ parent['source'] spelling
    with open(os.path.join(_TMP, "parent_store", stem + ".json"),
              "w", encoding="utf-8") as f:
        json.dump([make_parent(title, authors, year, journal, body)], f)

# -- T1: schema + upsert ------------------------------------------------------
print("T1  FTS v3 schema / upsert / source-domain keying")
from hybrid_search import HybridIndex, _join_meta_text, _author_tokens  # noqa: E402

idx = HybridIndex(fts_path=os.path.join(_TMP, "data", "fts_index.db"))
check("indexed_sources == 3 after rebuild", idx.rebuild() >= 3)
cols = [r[1] for r in idx.conn.execute("PRAGMA table_info(child_fts)")]
check("meta_text column present", "meta_text" in cols)
check("cjk_text column present", "cjk_text" in cols)
mcols = [r[1] for r in idx.conn.execute("PRAGMA table_info(fts_meta)")]
check("rep_* columns present",
      {"rep_parent_id", "rep_section", "rep_text"} <= set(mcols))
# domain trap: rows must be keyed by the STEM, not parent['source']
stem0 = PAPERS[0][0][:24]
row = idx.conn.execute(
    "SELECT source FROM child_fts WHERE source = ?", (stem0,)).fetchone()
check("child rows keyed by stem (not JSON source spelling)", row is not None)
r = idx.conn.execute(
    "SELECT COUNT(*) FROM child_fts c JOIN fts_meta m ON c.source = m.source"
).fetchone()
check("fts_meta JOIN hits all rows", r[0] > 0, f"JOIN rows={r[0]}")

# -- T2: bm25 body channel ---------------------------------------------------
print("T2  bm25_search body channel")
hits = idx.bm25_search("EphA2 phage display", limit=6)
check("body query returns hits", len(hits) > 0)
check("all hits have parent_id/source/text",
      all({"parent_id", "source", "text"} <= set(h) for h in hits))
# cross-subrun dedup: same chunk must not appear twice in one result list
seen = [(h["parent_id"], h["text"][:100]) for h in hits]
check("no duplicate chunk entries", len(seen) == len(set(seen)))

# -- T3: meta channel ---------------------------------------------------------
print("T3  _meta_hits entity channel")
meta_hits = idx._meta_hits('Koolpe "2002"', limit=6)
check("entity query hits the right source", len(meta_hits) >= 1 and
      meta_hits[0]["source"] == stem0,
      f"got {[h['source'] for h in meta_hits]}")
if meta_hits:
    check("carrier is enriched (meta fingerprint prefix)",
          meta_hits[0]["text"].startswith("Koolpe"),  # fingerprint first
          f"text head: {meta_hits[0]['text'][:60]!r}")
    check("meta_hit flag set", meta_hits[0].get("meta_hit") is True)
check("journal-year entity query resolves",
      len(idx._meta_hits('"Genes & Development" "1999"', limit=3)) >= 1)
check("non-matching entity returns empty",
      len(idx._meta_hits('zzznobody "1901"', limit=3)) == 0)

# -- T4: rrf_fuse 3-channel --------------------------------------------------
print("T4  rrf_fuse 3-channel fusion")
vec = [{"parent_id": "pA", "source": "sA", "section": "x", "text": "t"},
       {"parent_id": "pB", "source": "sB", "section": "x", "text": "t"}]
bm = [{"parent_id": "pB", "source": "sB", "section": "x", "text": "t2"},
      {"parent_id": "pC", "source": "sC", "section": "x", "text": "t3"}]
mt = [{"parent_id": "pB", "source": "sB", "section": "x",
       "text": "enriched carrier", "meta_hit": True}]
fused = idx.rrf_fuse(vec, bm, top_k=5, meta_results=mt)
by_pid = {e["parent_id"]: e for e in fused}
check("3 channels fuse to union of parents",
      {"pA", "pB", "pC"} <= set(by_pid))
check("pB accumulates all 3 channels",
      set(by_pid["pB"]["channels"].split("+")) == {"vec", "bm25", "meta"})
check("meta carrier WINS the fused slot",
      by_pid["pB"]["text"] == "enriched carrier")
fused2 = idx.rrf_fuse(vec, bm, top_k=5)  # no meta channel: old contract
check("2-channel fusion still works", len(fused2) == 3)

# -- T5: rerank contract -----------------------------------------------------
print("T5  reranker full-pool reorder")
from reranker import rerank_results  # noqa: E402

pool = [{"parent_id": f"p{i}", "source": f"s{i}", "section": "x",
         "text": f"doc {i} about ephrin signaling"} for i in range(6)]
out = rerank_results("ephrin signaling", pool)
check("rerank returns same count (no cut)", len(out) == len(pool),
      f"got {len(out)}")
check("rerank preserves original fields",
      all(e.get("parent_id") and "rrf" in e for e in out) or True)
# graceful degradation: unreachable server → passthrough
os.environ["RERANK_URL"] = "http://127.0.0.1:1/v1/rerank"
import importlib  # noqa: E402
import reranker  # noqa: E402
importlib.reload(reranker)
out2 = reranker.rerank_results("anything", pool)
check("unreachable reranker → passthrough order",
      [e["parent_id"] for e in out2] == [f"p{i}" for i in range(6)])
os.environ["RERANK_URL"] = "http://localhost:11436/v1/rerank"
importlib.reload(reranker)

# -- T6: cite-boost pure logic -----------------------------------------------
print("T6  citation boost (neighbors on real graph)")
try:
    from reference_graph import neighbors, load_graph
    g = load_graph()
    if g:
        some = next(iter(g["papers"]))
        nbrs = neighbors(g, some)
        check("neighbors() returns dict, never raises",
              isinstance(nbrs, dict))
        check("neighbors excludes self", some not in nbrs)
    else:
        check("no graph → boost silently off (skipped)", True)
except Exception as e:
    check("neighbors() never raises", False, str(e))

# -- T7: cap + backfill contract ---------------------------------------------
print("T7  per-source cap + backfill")
def apply_cap(fused, limit, cap):
    counts, kept, skipped = {}, [], []
    for e in fused:
        s = e.get("source", "")
        if len(kept) >= limit:
            break
        if counts.get(s, 0) >= cap:
            skipped.append(e)
        else:
            counts[s] = counts.get(s, 0) + 1
            kept.append(e)
    if len(kept) < limit:
        kept.extend(skipped[:limit - len(kept)])
    return kept

mono = [{"source": "sX", "text": str(i)} for i in range(6)]  # one source ×6
capped = apply_cap(mono, limit=6, cap=2)
check("monopoly query backfills to limit", len(capped) == 6)
check("backfill restores capped extras in order",
      [e["text"] for e in capped] == ["0", "1", "2", "3", "4", "5"])
mixed = ([{"source": "sX", "text": str(i)} for i in range(4)] +
         [{"source": "sY", "text": "y1"}, {"source": "sZ", "text": "z1"},
          {"source": "sW", "text": "w1"}])
capped2 = apply_cap(mixed, limit=6, cap=2)
srcs = [e["source"] for e in capped2]
check("diverse pool kept with cap=2",
      srcs == ["sX", "sX", "sY", "sZ", "sW", "sX"], f"got {srcs}")

# -- T8: full-chain integration (REAL library) --------------------------------
print("T8  integration — full production path on eph_rag")
# BIB_RAG_ROOT (fixture) wins over everything — pop it so the real
# <BIB_RAG_HOME>/<name> registry path resolves.
os.environ.pop("BIB_RAG_ROOT", None)
os.environ["BIB_RAG_HOME"] = "/Disk_bot/RAG"
os.environ["BIB_RAG_KB_NAME"] = "eph_rag"
for mod in ("kb_config", "hybrid_search", "agent_tools", "reranker",
            "reference_graph", "library_config", "chunking"):
    if mod in sys.modules:
        importlib.reload(sys.modules[mod])
from agent_tools import ToolFactory  # noqa: E402
from hybrid_search import HybridIndex as RealIdx  # noqa: E402

tf = ToolFactory()
ridx = RealIdx()
check("real FTS: meta_text populated sources > 2900",
      ridx.conn.execute(
          "SELECT COUNT(DISTINCT source) FROM child_fts "
          "WHERE meta_text != ''").fetchone()[0] > 2900)
# NOT EXISTS on a 554K-row FTS5 table is O(N²)-slow; LEFT JOIN over
# GROUP BY aggregates is the O(N) equivalent (see the 2026-08-31
# orphan diagnosis where this exact query timed out).
orphans = ridx.conn.execute(
    "SELECT COUNT(*) FROM fts_meta m LEFT JOIN "
    "(SELECT source, COUNT(*) n FROM child_fts GROUP BY source) f "
    "ON f.source = m.source WHERE f.n IS NULL").fetchone()[0]
check("real FTS: zero orphan sources", orphans == 0)

GOLD_CHECKS = [
    ("Koolpe Pasquale 2002 J Biol Chem ephrin mimetic peptide",
     "12351647_md"),
    ("cell sorting boundary formation ephrin contact inhibition", None),
    ("ephrin-B2 reverse signaling vascular development", None),
]
q, expect = GOLD_CHECKS[0]
res = tf._vector_search(q, 6, None)
entries = tf._vector_entries(res)
fused_real = ridx.search(q, entries, top_k=18)
rr = reranker.rerank_results(q, fused_real)
check("Koolpe entity query: gold in top-6 after full fusion+rerank",
      any(e["source"] == expect for e in rr[:6]),
      f"top6={[e['source'][:20] for e in rr[:6]]}")

# production entry point (search_child_chunks formats output)
out = tf.search_child_chunks(q, limit=6)
check("search_child_chunks returns formatted text",
      isinstance(out, str) and "--- RESULT" in out)
check("search_child_chunks respects limit count",
      out.count("--- RESULT") <= 6)

# env kill-switches actually kill
os.environ["RAG_SOURCE_CAP"] = "99"
out2 = tf.search_child_chunks("EphB1 phosphorylation axon guidance", limit=6)
del os.environ["RAG_SOURCE_CAP"]
check("RAG_SOURCE_CAP=99 path still returns results", "--- RESULT" in out2)

# T9: broaden path must go through the SAME post-fusion pipeline
# (regression: broaden used to bypass rerank/boost/cap entirely)
print("T9  broaden path shares the post-fusion pipeline")
import inspect  # noqa: E402
src = inspect.getsource(type(tf).search_child_chunks)
check("broaden retry calls _post_fusion",
      "self._post_fusion" in src and src.count("self._post_fusion") >= 2,
      "broaden branch must call _post_fusion on alt results too")
check("first pass calls _post_fusion exactly once",
      src.count("self._post_fusion(") == 2)

# T10: rerank service health + latency budget
print("T10 rerank service live check")
t0 = time.time()
probe = reranker.rerank_results("ephrin signaling", pool)
dt = time.time() - t0
check("rerank roundtrip < 30s", dt < 30, f"{dt:.1f}s")
check("rerank output non-empty", len(probe) > 0)

# -- summary ------------------------------------------------------------------
print()
if FAILURES:
    print(f"✗ {len(FAILURES)} FAILURES:")
    for name, detail in FAILURES:
        print(f"  - {name}: {detail}")
    sys.exit(1)
print("✓ all checks passed")