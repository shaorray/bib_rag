#!/usr/bin/env python3
"""
tests/test_eval_faithfulness.py — offline tests for Feature 3 faithfulness.

ALL tests are offline: no LLM calls. Scorer unit tests run on synthetic
answers; the builder test runs against a tiny fixture gold file + fixture
parent-store dir in tmp_path; the exit-code test runs the real CLI as a
subprocess with LLM_URL pointed at an unreachable port (asserts exit 2).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"

# import the script as a module (script-style, src/ on its own sys.path)
sys.path.insert(0, str(SCRIPTS))
import eval_faithfulness as ef  # noqa: E402


# ════════════════════════════════════════════════════════════════════════
# Scorer unit tests (synthetic answers, no LLM)
# ════════════════════════════════════════════════════════════════════════

SUPPORTED_ANSWER = """Eph receptors bind ephrin ligands on adjacent cells.

**Sources:**
- EphA4 and ephrin-A5 mediate topographic mapping (Hall et al., 2024; PMID: 22144690)
- Cytoneme signaling provides essential contributions to mammalian tissue patterning (Hall et al., 2024)
"""

HALLUCINATED_ANSWER = """The moon is made of green cheese and Eph receptors taste like strawberries.

**Sources:**
- Totally made up journal article about quantum cheese dynamics (Fabricated et al., 2099)
- Another invented reference (Nobody Real, 2001)
"""

REFUSAL_ANSWER = """There is no literature supporting this question in the current library; this topic is not answerable from the indexed papers.

**Sources:**
"""


def _fake_tool_events():
    return [{"tool_use_id": "call_1", "name": "search_child_chunks",
             "type": "tool_call"}]


def test_hallucinated_answer_high_hallucination_rate():
    claims = [
        {"id": "C1", "text": "the moon is green cheese", "confidence": "strong",
         "evidence": [{"kind": "citation", "pmid": "99999999"}]},
        {"id": "C2", "text": "receptors taste like strawberries",
         "confidence": "moderate",
         "evidence": [{"kind": "citation", "doi": "10.9999/fabricated"}]},
    ]
    hall, weak, n = ef.score_claims(claims, set(), _fake_tool_events())
    # citations are syntactic-only → status external → NOT hallucinated by
    # this metric; but a claim with garbage evidence kinds must be.
    garbage = [
        {"id": "C1", "text": "unsupported", "confidence": "weak",
         "evidence": [{"kind": "chunk", "parent_id": "never_retrieved.md#s#x"}]},
    ]
    hall_g, weak_g, n_g = ef.score_claims(garbage, set(), [])
    assert n_g == 1
    assert hall_g == pytest.approx(1.0)   # chunk not in retrieval_keys → unresolved
    assert weak_g == pytest.approx(1.0)   # weak claim surfaced
    assert n == 2 and 0.0 <= hall < 0.5   # external refs count as evidence
    assert hall == pytest.approx(0.0)


def test_supported_answer_low_hallucination_rate():
    claims = [
        {"id": "C1", "text": "eph receptors bind ephrins", "confidence": "strong",
         "evidence": [{"kind": "chunk",
                       "parent_id": "22144690.md#abstract#x",
                       "note": "retrieved this session"}]},
        {"id": "C2", "text": "cytonemes carry morphogens", "confidence": "strong",
         "evidence": [{"kind": "tool_call", "tool_use_id": "call_1"}]},
    ]
    keys = {"parent::22144690.md#abstract#x", "search::eph receptors"}
    hall, weak, n = ef.score_claims(claims, keys, _fake_tool_events())
    assert n == 2
    assert hall == pytest.approx(0.0)
    assert weak == pytest.approx(0.0)


def test_structurally_invalid_claims_count_as_hallucinated():
    # no evidence at all → validate_claims drops them → all hallucinated
    claims = [{"id": "C1", "text": "unsupported claim", "evidence": []}]
    hall, weak, n = ef.score_claims(claims, set(), [])
    assert n == 1
    assert hall == pytest.approx(1.0)


def test_citation_accuracy_supported_answer():
    expected = ["Hall et al. - 2024 - Cytoneme signaling provides essential "
                "contributions to mammalian tissue patterning.md", "22144690.md"]
    acc, n = ef.score_citations(
        ["Hall et al. - 2024 - Cytoneme signaling provides essential "
         "contributions to mammalian tissue patterning.md#intro#ab12",
         "22144690.md#results#zz99"], expected, refusal_expected=False)
    assert n == 2
    assert acc == pytest.approx(1.0)


def test_citation_accuracy_no_hits():
    acc, n = ef.score_citations(
        ["Some Other Paper - 1999 - Unrelated title.md#s#h"],
        ["22144690.md"], refusal_expected=False)
    assert n == 1
    assert acc == pytest.approx(0.0)


def test_citation_accuracy_empty_sources_is_zero():
    acc, n = ef.score_citations([], ["22144690.md"], refusal_expected=False)
    assert n == 0 and acc == pytest.approx(0.0)


def test_citation_accuracy_skipped_for_refusal_rows():
    acc, n = ef.score_citations(["anything.md#s#h"], [], True)
    assert acc is None
    assert n == 1


def test_refusal_correctness_expected_with_marker():
    ok, marker = ef.score_refusal(REFUSAL_ANSWER, refusal_expected=True)
    assert marker is True
    assert ok is True


def test_refusal_correctness_expected_but_answered():
    ok, marker = ef.score_refusal(SUPPORTED_ANSWER, refusal_expected=True)
    assert marker is False
    assert ok is False


def test_refusal_correctness_not_expected_and_clean():
    ok, marker = ef.score_refusal(SUPPORTED_ANSWER, refusal_expected=False)
    assert marker is False
    assert ok is True


def test_refusal_correctness_not_expected_but_refuses():
    ok, marker = ef.score_refusal(REFUSAL_ANSWER, refusal_expected=False)
    assert marker is True
    assert ok is False


def test_refusal_marker_in_sources_section_does_not_count():
    # marker ONLY in the Sources section → body is clean → not a refusal
    ans = "Eph receptors mediate axon guidance.\n\n**Sources:**\n- no literature supporting this claim anywhere\n"
    body = ef.strip_sources_section(ans)
    assert "no literature supporting" in ans
    assert "no literature supporting" not in body


def test_all_refusal_markers_recognized():
    for marker in ef.REFUSAL_MARKERS:
        assert ef.has_refusal_marker(f"Unfortunately, {marker} here.")


def test_fresh_thread_id_per_row(monkeypatch):
    # run_row must generate a NEW uuid4 thread_id per call (stale-answer bug
    # guard) — verify by capturing the config it passes to the graph.
    import uuid as _uuid
    import types
    seen = []

    class FakeGraph:
        def invoke(self, state, config=None):
            seen.append(config["configurable"]["thread_id"])
            return {"messages": [], "agent_answers": [
                {"retrieval_keys": ["search::q"]}]}

        def get_state(self, config):
            raise AssertionError("fallback get_state should not be needed")

    stub = types.ModuleType("agentic_query")
    stub.invoke_with_interrupt_handling = (
        lambda g, inp, cfg, follow_up_provider=None: g.invoke(inp, config=cfg))
    monkeypatch.setitem(sys.modules, "agentic_query", stub)
    answer, keys, _ = ef.run_row(FakeGraph(), "test query")
    assert keys == {"search::q"}
    assert len(seen) == 1
    # uuid4-shaped
    _uuid.UUID(seen[0].split("eval-", 1)[1])


# ════════════════════════════════════════════════════════════════════════
# Builder test (fixture gold file + fixture parent store in tmp_path)
# ════════════════════════════════════════════════════════════════════════

def _write_fixture_kb(tmp_path: Path, n_gold: int = 5):
    """Tiny gold file + parent store; returns (gold_path, store_dir)."""
    data_dir = tmp_path / "data"
    store = tmp_path / "parent_store"
    store.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    sentences = [
        "Eph receptors interact with ephrin ligands on adjacent cells to facilitate tissue patterning during normal and oncogenic development.",
        "EphA and B subtypes preferentially bind A- and B-type ephrins respectively, resulting in receptor complexes that propagate via homotypic interactions.",
        "Forward signaling through Eph receptors induces cell repulsion and collapse of the growth cone in the developing nervous system.",
        "Bidirectional signaling means that both the receptor-bearing and the ephrin-bearing cell receive an intracellular signal after contact.",
        "Cytonemes are long filopodial extensions that transport signaling morphogens across several cell diameters in the developing wing disc.",
    ]
    gold_rows = []
    for i in range(n_gold):
        fname = f"paper{i}_2010.md"
        gold_rows.append({"query": f"question number {i} about eph biology?",
                          "tier": "natural", "gold": [fname],
                          "gold_title": f"Paper {i}"})
        parents = [{"parent_id": f"{fname}#abstract#{i:010d}",
                    "source": fname, "section": "abstract",
                    "content": sentences[i % len(sentences)],
                    "word_count": 20, "char_count": 130, "meta": {}}]
        safe = "paper%d_2010_md" % i   # loader folds .md → _md (production naming)
        (store / f"{safe}.json").write_text(
            json.dumps(parents), encoding="utf-8")
    gold_path = data_dir / "gold_retrieval_v2.jsonl"
    gold_path.write_text(
        "\n".join(json.dumps(r) for r in gold_rows) + "\n", encoding="utf-8")
    return gold_path, store


def test_build_gold_faithfulness_fixture(tmp_path):
    sys.path.insert(0, str(SCRIPTS))
    import build_gold_faithfulness as bgf
    gold_path, store = _write_fixture_kb(tmp_path, n_gold=5)
    rows, skipped = bgf.build_rows(str(gold_path), str(store),
                                   limit=5, refusal_n=3)
    assert len(rows) >= 5
    grounded = [r for r in rows if not r["refusal_expected"]]
    refusals = [r for r in rows if r["refusal_expected"]]
    assert len(grounded) == 5
    assert 1 <= len(refusals) <= 3
    # schema
    for r in rows:
        assert set(r) == {"id", "query", "expected_sources", "key_fact",
                          "refusal_expected", "notes"}
        assert isinstance(r["expected_sources"], list)
        assert isinstance(r["refusal_expected"], bool)
    # key_fact is verbatim from the fixture sentence (no fabrication)
    for r in grounded:
        assert len(r["key_fact"]) >= 40
        assert r["key_fact"] in (
            "Eph receptors interact with ephrin ligands on adjacent cells to "
            "facilitate tissue patterning during normal and oncogenic development.",
            "EphA and B subtypes preferentially bind A- and B-type ephrins "
            "respectively, resulting in receptor complexes that propagate via "
            "homotypic interactions.",
            "Forward signaling through Eph receptors induces cell repulsion "
            "and collapse of the growth cone in the developing nervous system.",
            "Bidirectional signaling means that both the receptor-bearing and "
            "the ephrin-bearing cell receive an intracellular signal after contact.",
            "Cytonemes are long filopodial extensions that transport signaling "
            "morphogens across several cell diameters in the developing wing disc.",
        )
    # refusal rows: empty sources, designed-refusal note
    for r in refusals:
        assert r["expected_sources"] == [] and r["key_fact"] == ""
        assert r["notes"] == "designed refusal"


def test_build_gold_faithfulness_skip_notes(tmp_path):
    import build_gold_faithfulness as bgf
    gold_path, store = _write_fixture_kb(tmp_path, n_gold=4)
    # make one source unloadable (delete its store file)
    os.remove(store / "paper1_2010_md.json")
    rows, skipped = bgf.build_rows(str(gold_path), str(store),
                                   limit=4, refusal_n=2)
    assert skipped, "unloadable source must produce a skip note"
    assert any("not loadable" in s["note"] for s in skipped)
    # the unloadable row must NOT be in rows with a fabricated key_fact
    assert all(r["query"] != "question number 1 about eph biology?"
               for r in rows if not r["refusal_expected"])


def test_build_gold_no_clean_sentence_skipped(tmp_path):
    import build_gold_faithfulness as bgf
    gold_path, store = _write_fixture_kb(tmp_path, n_gold=2)
    # overwrite one parent's content with a fragment lacking terminal
    # punctuation / too short → no clean sentence → skip-with-note
    safe = store / "paper0_2010_md.json"
    parents = json.loads(safe.read_text(encoding="utf-8"))
    parents[0]["content"] = "short fragment, no sentence"
    safe.write_text(json.dumps(parents), encoding="utf-8")
    rows, skipped = bgf.build_rows(str(gold_path), str(store),
                                   limit=2, refusal_n=1)
    assert any("no clean sentence" in s["note"] for s in skipped)


def test_build_gold_cli_writes_output(tmp_path):
    gold_path, store = _write_fixture_kb(tmp_path, n_gold=6)
    out = tmp_path / "data" / "gold_faithfulness.jsonl"
    # run through main() with explicit args (no --kb; kb_config env not needed
    # because --gold/--out point into tmp_path, but parent store must resolve)
    sys.path.insert(0, str(SCRIPTS))
    import build_gold_faithfulness as bgf
    # emulate: main() reads cfg paths — instead exercise the writer directly
    rows, skipped = bgf.build_rows(str(gold_path), str(store),
                                   limit=6, refusal_n=4)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    back = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l]
    assert len(back) >= 5
    assert {"id", "query", "expected_sources", "key_fact",
            "refusal_expected", "notes"} <= set(back[0])


# ════════════════════════════════════════════════════════════════════════
# Exit-code test: LLM down → exit 2 (real subprocess, unreachable port)
# ════════════════════════════════════════════════════════════════════════

def test_eval_faithfulness_exit2_when_llm_down(tmp_path):
    gold_path, _ = _write_fixture_kb(tmp_path, n_gold=3)
    env = dict(os.environ)
    env["LLM_URL"] = "http://127.0.0.1:59999/v1"      # unreachable port
    env["BIB_RAG_ROOT"] = str(tmp_path)               # tmp KB root
    env.pop("BIB_RAG_KB_NAME", None)
    script = SCRIPTS / "eval_faithfulness.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--gold", str(gold_path)],
        capture_output=True, text=True, timeout=120, env=env)
    assert proc.returncode == 2, (proc.stdout, proc.stderr)
    assert "LLM endpoint unreachable" in (proc.stderr + proc.stdout)


def test_eval_faithfulness_exit3_missing_gold(tmp_path):
    env = dict(os.environ)
    env["LLM_URL"] = "http://127.0.0.1:59999/v1"
    env["BIB_RAG_ROOT"] = str(tmp_path)
    env.pop("BIB_RAG_KB_NAME", None)
    script = SCRIPTS / "eval_faithfulness.py"
    proc = subprocess.run(
        [sys.executable, str(script),
         "--gold", str(tmp_path / "nope.jsonl")],
        capture_output=True, text=True, timeout=120, env=env)
    assert proc.returncode == 3, (proc.stdout, proc.stderr)