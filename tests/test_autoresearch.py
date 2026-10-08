#!/usr/bin/env python3
"""test_autoresearch.py — regression tests for src/autoresearch.py (Phase 4:
CONSOLIDATE — the run directory, append-only logs, and report).

Pure: the three LLM/graph phases are mocked, so the tests pin P4's contract —
run_id determinism, the run directory layout, append-only JSONL, dedup across
rounds, and the honest report tally. No network, no real corpus.

Run:  /usr/bin/python3 tests/test_autoresearch.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

_TMP = tempfile.mkdtemp(prefix="artest_")
os.environ["BIB_RAG_ROOT"] = _TMP
os.environ.setdefault("HOME", "/home/rui")
os.makedirs(os.path.join(_TMP, "parent_store"), exist_ok=True)
os.makedirs(os.path.join(_TMP, "data"), exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import autoresearch as AR  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append((name, detail))


# ─── fixtures ────────────────────────────────────────────────────────────────
# fake metrics: two communities, one high-betweenness bridge
METRICS = {
    "a.md": {"community_id": 1, "betweenness": 0.5, "rcr": 8.0,
             "in_corpus_cited_by": 0, "citation_count": 100, "pagerank": 0.1},
    "b.md": {"community_id": 2, "betweenness": 0.1, "rcr": 1.0,
             "in_corpus_cited_by": 5, "citation_count": 10, "pagerank": 0.05},
}
GRAPH = {"nodes": {"a.md": {"title": "ephrin bridge"}, "b.md": {"title": "other"}},
         "edges": []}

DIRS = [
    {"id": "D1-a", "source": "bridge", "anchors": ["a.md"],
     "signal": {"betweenness": 0.5}, "topic": "ephrin", "note": "bridge"},
]
HYPS = [
    {"id": "H-D1-a", "from_direction": "D1-a", "from_source": "bridge",
     "anchors": ["a.md"], "claim": "Ephrin-B requires EphB2 phosphorylation.",
     "rationale": "r", "expected_evidence": ["e1"], "falsifier": "f",
     "scores": {"interesting": 8, "feasible": 7, "testable": 9},
     "status": "proposed"},
]
VERS = [
    {"hypothesis_id": "H-D1-a", "status": "verified", "support": 1,
     "n_evidence": 1, "evidence": [{"parent_id": "p1", "source": "a.pdf"}],
     "critic": {"verdict": "supported", "reason": "ok"},
     "ledger_ok": True, "ledger_errors": [], "queries": ["e1"]},
]


def _fake_phases(dirs=None, hyps=None, vers=None):
    """Patch dream/formulate/verify on the AR module's imported names."""
    dirs = DIRS if dirs is None else dirs
    hyps = HYPS if hyps is None else hyps
    vers = VERS if vers is None else vers
    return (
        mock.patch.object(AR, "dream") if False else
        mock.patch.dict(sys.modules)
    )


# We patch the actual modules AR imports lazily (dream/formulate/verify).
import dream as _d  # noqa: E402
import formulate as _f  # noqa: E402
import verify as _v  # noqa: E402


def _run(tmp_out, rounds=1, dirs=None, hyps=None, vers=None):
    dirs = DIRS if dirs is None else dirs
    hyps = HYPS if hyps is None else hyps
    vers = VERS if vers is None else vers
    with mock.patch.object(_d, "diverge", return_value=dirs), \
         mock.patch.object(_d, "graph_fingerprint", return_value="fptest"), \
         mock.patch.object(_f, "formulate_all", return_value=hyps), \
         mock.patch.object(_v, "verify_all", return_value=vers):
        return AR.run("test question", rounds=rounds, out_dir=tmp_out)


# ---------------------------------------------------------------------------
# run directory + artifacts
# ---------------------------------------------------------------------------
print("run() artifacts")
out = tempfile.mkdtemp(prefix="arout_")
st = _run(out, rounds=1)
check("run_state.json written", os.path.exists(os.path.join(out, "run_state.json")))
check("dream_candidates.jsonl written",
      os.path.exists(os.path.join(out, "dream_candidates.jsonl")))
check("hypotheses.jsonl written", os.path.exists(os.path.join(out, "hypotheses.jsonl")))
check("verification.jsonl written",
      os.path.exists(os.path.join(out, "verification.jsonl")))
check("final_report.md written", os.path.exists(os.path.join(out, "final_report.md")))
check("run_id = fingerprint-seed", st["run_id"] == "fptest-s7")
check("all phases marked done",
      all(st["phases"][p] == "done" for p in
          ("setup", "dream", "formulate", "verify", "consolidate")))
check("counts recorded", st["counts"] == {"directions": 1, "hypotheses": 1,
                                          "verifications": 1})

# run_state is valid JSON
rs = json.load(open(os.path.join(out, "run_state.json"), encoding="utf-8"))
check("run_state parses", rs["question"] == "test question")

# report contains the tally and the claim
rep = open(os.path.join(out, "final_report.md"), encoding="utf-8").read()
check("report has status tally", "**verified**: 1" in rep)
check("report includes the claim", "EphB2 phosphorylation" in rep)
check("report is honest about method", "never inflated" in rep)


# ---------------------------------------------------------------------------
# append-only + cross-round dedup
# ---------------------------------------------------------------------------
print("append-only + dedup")
out2 = tempfile.mkdtemp(prefix="arout2_")
# 2 rounds, SAME deterministic hypotheses → 2nd round adds nothing new
st2 = _run(out2, rounds=2)
hyps_lines = open(os.path.join(out2, "hypotheses.jsonl"),
                  encoding="utf-8").read().strip().splitlines()
check("dedup: identical hypothesis not duplicated across rounds",
      len(hyps_lines) == 1, f"got {len(hyps_lines)}")
check("early-stop recorded rounds_done=1", st2["rounds_done"] == 1,
      f"got {st2['rounds_done']}")


# ---------------------------------------------------------------------------
# honest tally: a refuted/unverifiable run reports them correctly
# ---------------------------------------------------------------------------
print("honest tally")
out3 = tempfile.mkdtemp(prefix="arout3_")
vers_mixed = [
    {"hypothesis_id": "H-D1-a", "status": "refuted", "support": 1,
     "n_evidence": 1, "evidence": [], "critic": {"verdict": "contradicted"},
     "ledger_ok": True, "ledger_errors": [], "queries": []},
]
st3 = _run(out3, rounds=1, vers=vers_mixed)
rep3 = open(os.path.join(out3, "final_report.md"), encoding="utf-8").read()
check("refuted tally reported", "**refuted**: 1" in rep3)
check("verified tally zero when none", "**verified**: 0" in rep3)


# ---------------------------------------------------------------------------
# CLI arg surface
# ---------------------------------------------------------------------------
print("main() CLI")
out4 = tempfile.mkdtemp(prefix="arout4_")
with mock.patch.object(AR, "run", return_value={
        "run_id": "x", "rounds_done": 1, "rounds_requested": 1,
        "counts": {"directions": 0, "hypotheses": 0, "verifications": 0},
        "run_path": out4}) as m:
    rc = AR.main(["q here", "--rounds", "2", "--n-each", "4", "--no-anchor"])
    check("main returns 0", rc == 0)
    _kw = m.call_args.kwargs
    check("CLI maps --rounds", _kw["rounds"] == 2)
    check("CLI maps --n-each", _kw["n_each"] == 4)
    check("CLI maps --no-anchor", _kw["anchor"] is False)


print()
if FAILURES:
    print(f"✗ {len(FAILURES)} FAILURES:")
    for name, detail in FAILURES:
        print(f"  - {name}: {detail}")
    sys.exit(1)
print("✓ all checks passed")
