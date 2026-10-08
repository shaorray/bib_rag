#!/usr/bin/env python3
"""test_verify.py — regression tests for src/verify.py (autonomous-research
Phase 3: verify proposals against the corpus with the toolkit's rules).

Pure: no network, no real LLM, no real Chroma. Retrieval + critic are mocked,
so the tests pin the DECISION LOGIC (the layered status rule) — the most
safety-critical part: a hypothesis with no whitelisted evidence must NEVER
come out "verified".

Run:  /usr/bin/python3 tests/test_verify.py
"""
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

_TMP = tempfile.mkdtemp(prefix="verifytest_")
os.environ["BIB_RAG_ROOT"] = _TMP
os.environ.setdefault("HOME", "/home/rui")
os.makedirs(os.path.join(_TMP, "parent_store"), exist_ok=True)
os.makedirs(os.path.join(_TMP, "data"), exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import verify as V  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append((name, detail))


HYP = {
    "id": "H-D1-abc",
    "claim": "Ephrin-B reverse signalling requires EphB2 phosphorylation.",
    "expected_evidence": ["EphB2 phospho-tyrosine", "ephrin-B reverse signalling"],
    "falsifier": "Reverse signalling proceeds in EphB2 phospho-null mutants.",
    "scores": {"interesting": 8, "feasible": 7, "testable": 9},
    "status": "proposed",
}


def _ev(pid, sim=0.6):
    return {"parent_id": pid, "source": pid + ".pdf",
            "snippet": "evidence text", "similarity": sim}


# ---------------------------------------------------------------------------
# _parse_verdict
# ---------------------------------------------------------------------------
print("_parse_verdict")
check("clean verdict", V._parse_verdict('{"verdict":"supported","reason":"x"}')
      == {"verdict": "supported", "reason": "x"})
check("fenced verdict", V._parse_verdict('```json\n{"verdict":"contradicted","reason":"y"}\n```')["verdict"] == "contradicted")
check("unknown verdict → None", V._parse_verdict('{"verdict":"maybe"}') is None)
check("garbage → None", V._parse_verdict("nope") is None)


# ---------------------------------------------------------------------------
# _decide_status — the safety core
# ---------------------------------------------------------------------------
print("_decide_status")
check("no evidence → unsupported",
      V._decide_status([], True, {"verdict": "supported"}) == "unsupported")
check("evidence but ledger failed → unsupported (KEY RULE)",
      V._decide_status([_ev("p1")], False, {"verdict": "supported"}) == "unsupported")
check("evidence + ledger ok + supported → verified",
      V._decide_status([_ev("p1")], True, {"verdict": "supported"}) == "verified")
check("evidence + ledger ok + contradicted → refuted",
      V._decide_status([_ev("p1")], True, {"verdict": "contradicted"}) == "refuted")
check("evidence + ledger ok + inconclusive → speculative",
      V._decide_status([_ev("p1")], True, {"verdict": "inconclusive"}) == "speculative")


# ---------------------------------------------------------------------------
# verify(): full path with mocked retrieval + critic
# ---------------------------------------------------------------------------
print("verify")

# CASE 1: evidence retrieved, critic supports → verified
with mock.patch.object(V, "collect_evidence",
                       return_value=([_ev("p1"), _ev("p2")], ["q1", "q2"])), \
     mock.patch.object(V, "ledger_check", return_value=(True, [])), \
     mock.patch.object(V, "_critic",
                       return_value={"verdict": "supported", "reason": "matches"}):
    r = V.verify(HYP)
    check("supporting evidence → verified", r["status"] == "verified")
    check("evidence carried", r["n_evidence"] == 2)
    check("critic recorded", r["critic"]["verdict"] == "supported")
    check("ledger_ok recorded", r["ledger_ok"] is True)

# CASE 2: NO evidence retrieved → unsupported (must NOT be verified)
with mock.patch.object(V, "collect_evidence", return_value=([], ["q1"])), \
     mock.patch.object(V, "ledger_check", return_value=(False, ["no evidence"])), \
     mock.patch.object(V, "_critic",
                       return_value={"verdict": "supported", "reason": "hallucinated"}):
    r = V.verify(HYP)
    check("no evidence → unsupported even if critic says supported",
          r["status"] == "unsupported", f"got {r['status']}")

# CASE 3: evidence exists but ledger fails → unsupported
with mock.patch.object(V, "collect_evidence", return_value=([_ev("p1")], ["q1"])), \
     mock.patch.object(V, "ledger_check", return_value=(False, ["not whitelisted"])), \
     mock.patch.object(V, "_critic", return_value={"verdict": "supported", "reason": ""}):
    r = V.verify(HYP)
    check("ledger failure → unsupported", r["status"] == "unsupported")

# CASE 4: contradicted → refuted (a first-class outcome, not hidden)
with mock.patch.object(V, "collect_evidence", return_value=([_ev("p1")], ["q1"])), \
     mock.patch.object(V, "ledger_check", return_value=(True, [])), \
     mock.patch.object(V, "_critic",
                       return_value={"verdict": "contradicted", "reason": "matches falsifier"}):
    r = V.verify(HYP)
    check("contradicted → refuted", r["status"] == "refuted")

# CASE 5: critic unavailable → inconclusive → speculative (degrade, not verify)
with mock.patch.object(V, "collect_evidence", return_value=([_ev("p1")], ["q1"])), \
     mock.patch.object(V, "ledger_check", return_value=(True, [])), \
     mock.patch.object(V, "_critic",
                       return_value={"verdict": "inconclusive", "reason": "critic unavailable"}):
    r = V.verify(HYP)
    check("critic down → speculative (never verified)", r["status"] == "speculative")

# no hypothesis → unsupported, no raise
r = V.verify({})
check("empty hypothesis → unsupported, no raise", r["status"] == "unsupported")

# verify_all keeps every hypothesis (failures are DATA, not dropped)
with mock.patch.object(V, "verify",
                       side_effect=[{"status": "verified"},
                                    {"status": "refuted"}]):
    out = V.verify_all([{"id": "H-1"}, {"id": "H-2"}])
    check("verify_all keeps all (incl. refuted)", len(out) == 2)


# ---------------------------------------------------------------------------
# collect_evidence: dedup by parent_id, best similarity kept
# ---------------------------------------------------------------------------
print("collect_evidence")
def _fake_search(q, top_k=5, expand=True, distill=True):
    if q == "EphB2 phospho-tyrosine":
        return [{"metadata": {"parent_id": "p1", "source": "a.pdf"},
                 "text": "t1", "distance": 0.5},
                {"metadata": {"parent_id": "p2", "source": "b.pdf"},
                 "text": "t2", "distance": 0.9}]
    # second phrase hits p1 again with a BETTER distance
    return [{"metadata": {"parent_id": "p1", "source": "a.pdf"},
             "text": "t1b", "distance": 0.2}]


with mock.patch.object(V, "_retrieve", side_effect=lambda q, k: (
        [{"parent_id": "p1", "source": "a.pdf", "snippet": "t1", "similarity": 1/(1+0.5)},
         {"parent_id": "p2", "source": "b.pdf", "snippet": "t2", "similarity": 1/(1+0.9)}]
        if q == "EphB2 phospho-tyrosine" else
        [{"parent_id": "p1", "source": "a.pdf", "snippet": "t1b", "similarity": 1/(1+0.2)}])):
    ev, qs = V.collect_evidence(HYP, top_k=5)
    ids = [e["parent_id"] for e in ev]
    check("dedups by parent_id", ids.count("p1") == 1)
    check("keeps best similarity for p1",
          next(e for e in ev if e["parent_id"] == "p1")["similarity"] == 1/(1+0.2))
    check("sorted by similarity desc", ids[0] == "p1")
    check("queries recorded", qs == HYP["expected_evidence"])

# no expected_evidence → falls back to the claim as probe (never empty)
ev, qs = V.collect_evidence({"id": "H-x", "claim": "some claim",
                             "expected_evidence": []}, top_k=3)
check("empty phrases → claim used as probe", qs == ["some claim"])

# L0 relevance floor: below-threshold hits are dropped (corpus silent)
with mock.patch.object(V, "_retrieve", return_value=[
        {"parent_id": "hi", "source": "a.pdf", "snippet": "s", "similarity": 0.7},
        {"parent_id": "lo", "source": "b.pdf", "snippet": "s", "similarity": 0.3}]):
    ev, _ = V.collect_evidence({"id": "H-y", "claim": "c",
                                "expected_evidence": ["q"]}, top_k=5)
    ids = [e["parent_id"] for e in ev]
    check("below-threshold hit dropped", "lo" not in ids)
    check("above-threshold hit kept", "hi" in ids)

# threshold is env-tunable; VERIFY_MIN_SIM=0 disables the floor
os.environ["VERIFY_MIN_SIM"] = "0"
with mock.patch.object(V, "_retrieve", return_value=[
        {"parent_id": "lo", "source": "b.pdf", "snippet": "s", "similarity": 0.3}]):
    ev, _ = V.collect_evidence({"id": "H-z", "claim": "c",
                                "expected_evidence": ["q"]}, top_k=5)
    check("VERIFY_MIN_SIM=0 disables the floor", [e["parent_id"] for e in ev] == ["lo"])
del os.environ["VERIFY_MIN_SIM"]

# a hit with no similarity cannot be judged → kept (don't silently drop)
with mock.patch.object(V, "_retrieve", return_value=[
        {"parent_id": "n", "source": "c.pdf", "snippet": "s", "similarity": None}]):
    ev, _ = V.collect_evidence({"id": "H-w", "claim": "c",
                                "expected_evidence": ["q"]}, top_k=5)
    check("missing similarity → kept", [e["parent_id"] for e in ev] == ["n"])


print()
if FAILURES:
    print(f"✗ {len(FAILURES)} FAILURES:")
    for name, detail in FAILURES:
        print(f"  - {name}: {detail}")
    sys.exit(1)
print("✓ all checks passed")
