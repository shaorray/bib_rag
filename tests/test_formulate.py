#!/usr/bin/env python3
"""test_formulate.py — regression tests for src/formulate.py (autonomous-
research Phase 2: turn graph directions into structured hypotheses).

Pure: no network, no real LLM. The LLM call is mocked so the tests pin the
CONTRACT (tolerant JSON parse, score clamping, skip-on-failure, audit link),
not the model's wording.

Run:  /usr/bin/python3 tests/test_formulate.py
"""
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

_TMP = tempfile.mkdtemp(prefix="formtest_")
os.environ["BIB_RAG_ROOT"] = _TMP
os.environ.setdefault("HOME", "/home/rui")
os.makedirs(os.path.join(_TMP, "parent_store"), exist_ok=True)
os.makedirs(os.path.join(_TMP, "data"), exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import formulate as F  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append((name, detail))


class _Resp:
    def __init__(self, content):
        self._c = content
    def raise_for_status(self):
        pass
    def json(self):
        return {"choices": [{"message": {"content": self._c}}]}


def _mock(content):
    return mock.patch.object(F.requests, "post", return_value=_Resp(content))


DIR = {"id": "D1-abc", "source": "bridge", "anchors": ["Pasquale 2008.md"],
       "signal": {"betweenness": 0.0048}, "topic": "ephrin signalling",
       "note": "structural bridge in community 0"}

GOOD = ('{"claim": "Ephrin-B reverse signalling requires EphB2 tyrosine '
        'phosphorylation.", "rationale": "Pasquale links the two subfields.", '
        '"expected_evidence": ["EphB2 phospho-tyrosine immunoblot", '
        '"ephrin-B reverse signalling assay"], '
        '"falsifier": "Reverse signalling proceeds normally in EphB2 '
        'phospho-null mutants.", '
        '"scores": {"interesting": 8, "feasible": 7, "testable": 9}}')


# ---------------------------------------------------------------------------
# parsing (tolerant)
# ---------------------------------------------------------------------------
print("_parse_hypothesis")
p = F._parse_hypothesis(GOOD)
check("parses a clean object", p and p["claim"].startswith("Ephrin-B reverse"))
check("evidence is a list", isinstance(p["expected_evidence"], list)
      and len(p["expected_evidence"]) == 2)
check("scores kept 1-10", p["scores"] == {"interesting": 8, "feasible": 7, "testable": 9})

check("fenced block parsed",
      F._parse_hypothesis('```json\n' + GOOD + '\n```') is not None)
check("prose-wrapped parsed",
      F._parse_hypothesis('Sure:\n' + GOOD + '\nDone.') is not None)

# empty claim → None (explicit "no testable claim")
check("empty claim → None",
      F._parse_hypothesis('{"claim": ""}') is None)
check("garbage → None", F._parse_hypothesis("no json here") is None)
check("empty string → None", F._parse_hypothesis("") is None)

# score clamping + missing scores
c = F._parse_hypothesis('{"claim": "X mediates Y.", "scores": {"interesting": 99, "feasible": -3}}')
check("scores clamped to [1,10]", c["scores"]["interesting"] == 10 and c["scores"]["feasible"] == 1)
check("missing score → None", c["scores"]["testable"] is None)
c2 = F._parse_hypothesis('{"claim": "X mediates Y."}')
check("no scores block → all None", all(v is None for v in c2["scores"].values()))

# expected_evidence as a bare string is wrapped
c3 = F._parse_hypothesis('{"claim": "X does Y.", "expected_evidence": "one phrase"}')
check("string evidence wrapped to list", c3["expected_evidence"] == ["one phrase"])


# ---------------------------------------------------------------------------
# formulate(): contract
# ---------------------------------------------------------------------------
print("formulate")
with _mock(GOOD):
    h = F.formulate(DIR, context="question")
    check("returns a hypothesis", h is not None)
    check("id derived from direction (audit link)", h["id"] == "H-D1-abc")
    check("from_direction recorded", h["from_direction"] == "D1-abc")
    check("from_source + anchors carried", h["from_source"] == "bridge"
          and h["anchors"] == ["Pasquale 2008.md"])
    check("status is proposed", h["status"] == "proposed")
    check("no fabricated evidence field beyond expected_evidence",
          "evidence" not in h)

# LLM down → None (skip, not fabricate)
with mock.patch.object(F.requests, "post", side_effect=RuntimeError("down")):
    check("LLM down → None (no fabrication)", F.formulate(DIR) is None)

# model says it cannot yield a claim → None
with _mock('{"claim": ""}'):
    check("empty claim → None", F.formulate(DIR) is None)

# kill-switch
os.environ["FORMULATE"] = "0"
with mock.patch.object(F.requests, "post", side_effect=AssertionError("must not call")):
    check("FORMULATE=0 → None, no call", F.formulate(DIR) is None)
del os.environ["FORMULATE"]

check("no direction → None", F.formulate({}) is None)


# ---------------------------------------------------------------------------
# formulate_all(): drops failures, keeps order
# ---------------------------------------------------------------------------
print("formulate_all")
dirs = [
    {"id": "D1-a", "source": "bridge", "anchors": ["a.md"]},
    {"id": "D2-b", "source": "uncited", "anchors": ["b.md", "c.md"]},
    {"id": "D3-c", "source": "orphan", "anchors": ["d.md"]},
]
# alternate: good, unusable, good
calls = {"n": 0}
GOODS = [GOOD, '{"claim": ""}', GOOD]
def _seq(*a, **k):
    i = calls["n"]; calls["n"] += 1
    return _Resp(GOODS[i])
with mock.patch.object(F.requests, "post", side_effect=_seq):
    out = F.formulate_all(dirs, context="q")
    check("drops the unusable direction", len(out) == 2, f"got {len(out)}")
    check("preserves order of survivors",
          [h["from_direction"] for h in out] == ["D1-a", "D3-c"])

check("empty input → empty", F.formulate_all([]) == [])


print()
if FAILURES:
    print(f"✗ {len(FAILURES)} FAILURES:")
    for name, detail in FAILURES:
        print(f"  - {name}: {detail}")
    sys.exit(1)
print("✓ all checks passed")
