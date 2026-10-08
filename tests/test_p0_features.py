#!/usr/bin/env python3
"""test_p0_features.py — regression tests for the P0 mechanisms borrowed
from Khoj (2026-10-08):

  P0-1 sub-query rewriting   (src/subquery.py, wired into query_bib_rag)
  P0-2 analyst distillation  (src/distill.py, wired into query_bib_rag)

Everything here is pure: no network, no real ChromaDB, no real LLM. The LLM
calls are mocked, so the tests pin the CONTRACT (parse, de-dup, degradation,
kill-switches, merge semantics) rather than the model's wording.

Run:  /usr/bin/python3 tests/test_p0_features.py
"""
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

# -- test isolation: throwaway library so kb_config never hits the real store --
_TMP = tempfile.mkdtemp(prefix="p0test_lib_")
os.environ["BIB_RAG_ROOT"] = _TMP
os.environ.setdefault("HOME", "/home/rui")
os.makedirs(os.path.join(_TMP, "parent_store"), exist_ok=True)
os.makedirs(os.path.join(_TMP, "data"), exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import subquery as SQ  # noqa: E402
import distill as DS  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append((name, detail))


# ---------------------------------------------------------------------------
# subquery: parsing (tolerant of fences / prose / bare list)
# ---------------------------------------------------------------------------
print("P0-1a subquery._parse_queries")

check("plain object",
      SQ._parse_queries('{"queries": ["a b", "c d"]}') == ["a b", "c d"])
check("fenced json block",
      SQ._parse_queries('```json\n{"queries": ["x y"]}\n```') == ["x y"])
check("prose wrapped around object",
      SQ._parse_queries('Sure! Here you go:\n{"queries": ["p q"]}\nHope that helps.')
      == ["p q"])
check("bare json list",
      SQ._parse_queries('["one two", "three four"]') == ["one two", "three four"])
check("non-json garbage → []",
      SQ._parse_queries("I cannot help with that.") == [])
check("empty string → []", SQ._parse_queries("") == [])
check("empty queries list → []",
      SQ._parse_queries('{"queries": []}') == [])
check("drops non-string entries",
      SQ._parse_queries('{"queries": ["ok query", null, 42, ""]}') == ["ok query", "42"])


# ---------------------------------------------------------------------------
# subquery: expand_query contract (original first, dedup, cap, degradation)
# ---------------------------------------------------------------------------
print("P0-1b subquery.expand_query")

# LLM down → original only, never raises
with mock.patch.object(SQ.requests, "post", side_effect=RuntimeError("conn refused")):
    out = SQ.expand_query("ephrin-B1 reverse signalling")
    check("LLM down → returns [original]", out == ["ephrin-B1 reverse signalling"])

# kill-switch
os.environ["SUBQUERY_EXPAND"] = "0"
out = SQ.expand_query("anything")
del os.environ["SUBQUERY_EXPAND"]
check("SUBQUERY_EXPAND=0 → [original] (no call)", out == ["anything"])

# happy path with a mocked response object
class _Resp:
    def __init__(self, payload):
        self._p = payload
    def raise_for_status(self):
        pass
    def json(self):
        return {"choices": [{"message": {"content": self._p}}]}


def _mock_post(content):
    return mock.patch.object(SQ.requests, "post",
                             return_value=_Resp(content))


with _mock_post('{"queries": ["EphB2 activation", "ephrin-B reverse signal", '
                '"EphB2 phosphorylation assay", "one-too-many query"]}'):
    out = SQ.expand_query("Does ephrin-B1 signalling through EphB2 require "
                          "phosphorylation?", n=3)
    check("original is first", out[0].startswith("Does ephrin-B1"))
    check("caps at n+1 total (n=3 → 4)", len(out) == 4, f"got {len(out)}")
    check("dedups the original restated", out.count(out[0]) == 1)

# dedup: model echoes the original verbatim
with _mock_post('{"queries": ["same query", "same query", "other"]}'):
    out = SQ.expand_query("same query", n=3)
    check("exact-duplicate extra query dropped", out == ["same query", "other"])

# n=0 disables decomposition but keeps the original
with _mock_post('{"queries": ["should not appear"]}'):
    out = SQ.expand_query("q here", n=0)
    check("n=0 → [original]", out == ["q here"])

check("empty query → []", SQ.expand_query("") == [])
check("is_available true by default", SQ.is_available() is True)


# ---------------------------------------------------------------------------
# distill: contract (in-place annotate, verbatim, degradation, kill-switch)
# ---------------------------------------------------------------------------
print("P0-2a distill.distill_hits")

hits = [
    {"text": "Ephrin-B1 reverse signaling requires EphB2 tyrosine "
             "phosphorylation at three conserved tyrosines.", "similarity": 0.7},
    {"text": "Background on Eph receptors.", "similarity": 0.3},
]


class _RespD:
    def __init__(self, content):
        self._c = content
    def raise_for_status(self):
        pass
    def json(self):
        return {"choices": [{"message": {"content": self._c}}]}


with mock.patch.object(DS.requests, "post",
                       return_value=_RespD("EphB2 phosphorylation at three conserved tyrosines.")):
    out = DS.distill_hits("Does EphB2 need phosphorylation?", hits)
    check("annotates in place (same list object)", out is hits)
    check("distilled key attached", hits[0].get("distilled")
          == "EphB2 phosphorylation at three conserved tyrosines.")
    check("distilled_text prefers extract",
          DS.distilled_text(hits[0]).startswith("EphB2 phosphorylation"))
    check("raw text preserved under 'text'", "reverse signaling" in hits[0]["text"])

# NOTHING_RELEVANT → no distilled key
hits2 = [{"text": "irrelevant background", "similarity": 0.5}]
with mock.patch.object(DS.requests, "post", return_value=_RespD("NOTHING_RELEVANT")):
    DS.distill_hits("q", hits2)
    check("NOTHING_RELEVANT → no distilled key", "distilled" not in hits2[0])
    check("distilled_text falls back to raw",
          DS.distilled_text(hits2[0]) == "irrelevant background")

# LLM down → untouched, never raises
hits3 = [{"text": "keep me", "similarity": 0.5}]
with mock.patch.object(DS.requests, "post", side_effect=RuntimeError("down")):
    out = DS.distill_hits("q", hits3)
    check("LLM down → hit untouched", out[0].get("distilled") is None
          and out[0]["text"] == "keep me")

# DISTILL_MAX_CHARS clips the passage fed to the model
seen = {}


def _capture(url, headers=None, json=None, timeout=None):
    seen["passage"] = json["messages"][1]["content"]
    return _RespD("x")


with mock.patch.object(DS.requests, "post", side_effect=_capture):
    DS.distill_hits("q", [{"text": "A" * 5000, "similarity": 0.9}],
                    max_tokens=10)
    check("DISTILL_MAX_CHARS clips passage",
          len(seen["passage"]) < 5000, f"passage len {len(seen['passage'])}")

# kill-switch
os.environ["DISTILL"] = "0"
hits4 = [{"text": "x", "similarity": 0.5}]
with mock.patch.object(DS.requests, "post", side_effect=AssertionError("should not call")):
    DS.distill_hits("q", hits4)
    check("DISTILL=0 → no call, no key", "distilled" not in hits4[0])
del os.environ["DISTILL"]

check("empty hits → returned as-is", DS.distill_hits("q", []) == [])


# ---------------------------------------------------------------------------
# query_bib_rag: multi-query merge semantics (pure, no ChromaDB)
# ---------------------------------------------------------------------------
print("P0-1c query_bib_rag._merge_query_results")

import query_bib_rag as QB  # noqa: E402

base = [
    {"id": "c1", "distance": 0.4, "text": "base1", "metadata": {}},
    {"id": "c2", "distance": 0.9, "text": "base2", "metadata": {}},
]
# second query returns c2 with a BETTER distance, plus a new c3
q2 = [
    {"id": "c2", "distance": 0.2, "text": "sub2", "metadata": {}},
    {"id": "c3", "distance": 0.5, "text": "sub3", "metadata": {}},
]


def _fake_embed(text):
    return [0.0]


def _fake_native(query, emb, top_k):
    return q2 if query == "sub query" else []


with mock.patch.object(QB, "embed_query", _fake_embed), \
     mock.patch.object(QB, "native_chroma_search", _fake_native):
    merged = QB._merge_query_results("orig query", ["orig query", "sub query"],
                                     base, pool_size=10)
    ids = [m["id"] for m in merged]
    check("union keeps base + new hits", set(ids) == {"c1", "c2", "c3"})
    check("best (smallest) distance wins per id",
          next(m["distance"] for m in merged if m["id"] == "c2") == 0.2)
    check("merged pool distance-sorted",
          ids == ["c2", "c1", "c3"], f"order {ids}")

# sub-query failure is skipped silently (base preserved)
with mock.patch.object(QB, "embed_query", _fake_embed), \
     mock.patch.object(QB, "native_chroma_search", side_effect=RuntimeError("boom")):
    merged = QB._merge_query_results("orig", ["orig", "sub query"], base, pool_size=10)
    check("sub-query failure → base preserved",
          [m["id"] for m in merged] == ["c1", "c2"])


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
print()
if FAILURES:
    print(f"✗ {len(FAILURES)} FAILURES:")
    for name, detail in FAILURES:
        print(f"  - {name}: {detail}")
    sys.exit(1)
print("✓ all checks passed")
