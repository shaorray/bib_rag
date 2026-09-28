"""Feature 2 tests: numeric claim re-check + unverified-claim surfacing.

All offline — no LLM, no services. Covers:
- claim_numbers extraction (thousand separators, fold forms, years)
- numeric_mismatch boundary matching + source-year support
- the surfacing block appears before **Sources:** and disappears with
  BIB_RAG_UNVERIFIED_SURFACING=0
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ.setdefault("BIB_RAG_ROOT", "/tmp/bibrag_numeric_fixture")


def _reset_root():
    root = Path(os.environ["BIB_RAG_ROOT"])
    (root / "parent_store").mkdir(parents=True, exist_ok=True)
    return root


from claim_ledger import claim_numbers, numeric_mismatch  # noqa: E402


# ── claim_numbers ────────────────────────────────────────────────────────

def test_claim_numbers_basic():
    assert claim_numbers("Treatment increased expression 2-fold") == ["2"]


def test_claim_numbers_thousands_separator():
    assert claim_numbers("1,234 cells were counted") == ["1234"]


def test_claim_numbers_decimal_and_year():
    nums = claim_numbers("By 2024 the yield reached 3.5-fold higher")
    assert "2024" in nums and "3.5" in nums


def test_claim_numbers_no_numbers():
    assert claim_numbers("No numbers in this sentence.") == []


def test_claim_numbers_percent():
    assert "30" in claim_numbers("30% of cells were Nfatc1-positive")


# ── numeric_mismatch ─────────────────────────────────────────────────────

def test_supported_number_passes():
    r = numeric_mismatch("Treatment increased expression 2-fold",
                         ["the mutant showed a 2-fold increase in marker expression"])
    assert r is None


def test_unsupported_number_flagged():
    r = numeric_mismatch("Treatment increased expression 2-fold",
                         ["the mutant showed strong expression changes"])
    assert r == "numeric-mismatch"


def test_boundary_no_inflation_false_positive():
    # '10' must NOT be satisfied by '100' or '2010'
    r = numeric_mismatch("10 genes were doubled",
                         ["about 100 genes changed around 2010"])
    assert r == "numeric-mismatch"
    r2 = numeric_mismatch("10 genes were doubled",
                          ["exactly 10 genes changed"])
    assert r2 is None


def test_source_year_counts_as_support():
    # claim mentions only the year; cited source published 2024 → supported
    r = numeric_mismatch("The 2024 study reported the phenotype",
                         ["Efnb2 mutants show remodeled endocardium"],
                         source_years=[2024])
    assert r is None


def test_empty_texts_flagged():
    r = numeric_mismatch("3-fold change observed", ["", None])
    assert r == "numeric-mismatch"


def test_no_numbers_never_flagged():
    assert numeric_mismatch("Expression changes were observed", [""]) is None


# ── surfacing block (formatting logic as wired in collect_answer) ────────

def _surface(answer, claims):
    """Mirror of the collect_answer surfacing block (kept in sync)."""
    unverified = [
        c for c in claims
        if all(e.get("status") == "unresolved"
               or e.get("reason") == "numeric-mismatch"
               for e in c.get("evidence", []) if isinstance(e, dict))
        and any(e.get("reason") == "numeric-mismatch"
                for e in c.get("evidence", []) if isinstance(e, dict))
    ]
    if (unverified and claims
            and os.environ.get("BIB_RAG_UNVERIFIED_SURFACING", "1") != "0"
            and "**Sources:**" in answer):
        block = ("\n\n⚠ Unverified claims "
                 f"({len(unverified)}):\n" + "\n".join(
                     f"- {c['text'][:120]}" for c in unverified))
        return answer.replace("**Sources:**", block + "\n\n**Sources:**", 1)
    return answer


def test_surfacing_block_inserted_before_sources():
    ans = "Some findings.\n\n---\n**Sources:**\n- x.md"
    claims = [{"id": "S1", "text": "c" * 200,
               "evidence": [{"kind": "chunk", "status": "unresolved",
                             "reason": "numeric-mismatch"}]}]
    out = _surface(ans, claims)
    assert "⚠ Unverified claims (1):" in out
    assert out.index("Unverified claims") < out.index("**Sources:**")
    # truncation to 120
    line = [l for l in out.splitlines() if l.startswith("- ")][0]
    assert len(line) <= 2 + 120


def test_surfacing_disabled_by_env(monkeypatch):
    monkeypatch.setenv("BIB_RAG_UNVERIFIED_SURFACING", "0")
    ans = "Findings.\n\n---\n**Sources:**\n- x.md"
    claims = [{"id": "S1", "text": "t",
               "evidence": [{"status": "unresolved",
                             "reason": "numeric-mismatch"}]}]
    assert _surface(ans, claims) == ans


def test_surfacing_skips_supported_claims():
    ans = "Findings.\n\n---\n**Sources:**\n- x.md"
    claims = [{"id": "S1", "text": "t",
               "evidence": [{"status": "verified", "verified": True}]}]
    assert _surface(ans, claims) == ans


def test_surfacing_needs_sources_marker():
    ans = "Findings without a Sources section."
    claims = [{"id": "S1", "text": "t",
               "evidence": [{"status": "unresolved",
                             "reason": "numeric-mismatch"}]}]
    assert _surface(ans, claims) == ans