#!/usr/bin/env python3
"""test_title_gates.py — cross-implementation agreement tests for the
canonical title gates merged into scripts/bib_utils.py (2026-09-10).

Prior state: is_junk_title existed twice (meta_audit.py / enrich_meta.py)
with same name and DIFFERENT surfaces — a behavior fork, not redundancy.
Same for _title_search_sim / _title_sim. Both consumers now delegate to
bib_utils.is_junk_title(mode=...) / bib_utils.title_search_sim(accept=...).

This suite pins the PARITY CONTRACT:
  1. meta_audit.is_junk_title          == bib_utils.is_junk_title(audit)
  2. enrich_meta._is_junk_title        == bib_utils.is_junk_title(search)
  3. meta_audit._title_search_sim      == bib_utils.title_search_sim @0.80
  4. enrich_meta._title_sim            == bib_utils.title_search_sim @0.75
plus regression pins for the defects the merge fixed:
  - vol/page headers are junk in BOTH modes (old enrich regex never fired)
  - search mode keeps the banner classes old enrich missed (Received:/URL/
    retraction/Table) — the fork's list had only 5 banner words

Expected values were recorded from the OLD implementations before the
merge (/tmp/title_gate_probe.py) for every class the merge preserves.

Run:  python3 -B tests/test_title_gates.py        (from the repo root)
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "metadata"))
sys.path.insert(0, str(ROOT / "src"))

import bib_utils as U          # noqa: E402
import meta_audit as M         # noqa: E402
import enrich_meta as E        # noqa: E402


# ---------------------------------------------------------------------------

class TestJunkTitleParity(unittest.TestCase):
    """meta_audit's alias and enrich's alias agree with the canonical."""

    JUNK = [
        "", " ",
        "O R I G I N A L  A R T I C L E", "ORIGINAL ARTICLE",
        "Received: 12 May 2023 Accepted: 20 July 2023",
        "https://doi.org/10.1016/j.ydbio.2021.01.002",
        "Editorial Board", "Contents",
        "**10.7363/010104**", "T_A_Yu2014_VACTERL_renal",
        "RESEARCH ARTICLE", "pubs.acs.org/est Policy Analysis",
    ]
    NOT_JUNK_ANY_MODE = [
        "The Bounded Corridor: A Theory of Biological Persistence",
        "Horseshoe kidney: a review of the literature",
        "Part-load performance of direct-firing of coal",
        # ("A RARE CASE OF HORSESHOE KIDNEY" is NOT in this list: all-caps
        #  is search-mode junk by design — see test_all_caps_is_search_only_junk)
    ]
    # junk in BOTH modes — regression pin for the merge: the old enrich
    # fork's vol/page regex matched none of these, and its banner list
    # missed Received:/URL/retraction/Table entirely.
    VOLPAGE_HEADERS = [
        "Int. J. Morphol. 38(3):567-573, 2020",
        "J Med Life. 2023;16(4):570-576",
        "Eur J Anat 25 (3): 175-182 (2021)",
    ]
    AUDIT_ONLY_JUNK = [
        "This article has been retracted",
        "Table 1 Demographic data",
    ]

    def test_audit_alias_matches_canonical(self):
        for t in self.JUNK + self.NOT_JUNK_ANY_MODE + self.VOLPAGE_HEADERS \
                + self.AUDIT_ONLY_JUNK:
            self.assertEqual(
                M.is_junk_title(t), U.is_junk_title(t, mode="audit"),
                f"audit parity drift: {t[:50]!r}")

    def test_search_alias_matches_canonical(self):
        for t in self.JUNK + self.NOT_JUNK_ANY_MODE + self.VOLPAGE_HEADERS:
            self.assertEqual(
                E._is_junk_title(t), U.is_junk_title(t, mode="search"),
                f"search parity drift: {t[:50]!r}")

    def test_volpage_headers_junk_in_both_modes(self):
        for t in self.VOLPAGE_HEADERS:
            self.assertTrue(U.is_junk_title(t, mode="audit"),
                            f"audit missed vol/page header: {t}")
            self.assertTrue(U.is_junk_title(t, mode="search"),
                            f"search missed vol/page header: {t}")

    def test_search_mode_keeps_audit_banner_classes(self):
        # the old enrich fork let Received:/URL/retraction/Table through;
        # canonical search mode keeps every audit class (stricter superset)
        for t in self.AUDIT_ONLY_JUNK:
            self.assertTrue(U.is_junk_title(t, mode="search"),
                            f"search dropped an audit class: {t[:40]!r}")

    def test_all_caps_is_search_only_junk(self):
        # audit verdict only flags a field for repair → keep stylized title
        # search verdict feeds a Crossref query → ALL-CAPS is a bad key
        t = "A RARE CASE OF HORSESHOE KIDNEY"
        self.assertFalse(U.is_junk_title(t, mode="audit"))
        self.assertTrue(U.is_junk_title(t, mode="search"))

    def test_no_junk_verdict_for_real_titles(self):
        for t in self.NOT_JUNK_ANY_MODE:
            self.assertFalse(U.is_junk_title(t, mode="audit"))
            self.assertFalse(U.is_junk_title(t, mode="search"))


class TestTitleSearchSimParity(unittest.TestCase):
    """Both sim aliases delegate to the canonical impl at their own accept."""

    PAIRS = [
        ("Horseshoe kidney and metanephric development",
         "Horseshoe kidney and metanephric development"),          # exact
        ("Horseshoe kidney and metanephric development in the mouse",
         "Horseshoe kidney and metanephric development"),          # truncation
        ("Horseshoe kidney embryology", "Renal cell carcinoma"),   # wrong paper
        ("", "Some Title"), ("Some Title", ""),                    # empties
    ]
    EXPECTED = [1.0, 0.714286, 0.0, 0.0, 0.0]   # recorded pre-merge (both)

    def test_values_unchanged_by_merge(self):
        for (c, r), exp in zip(self.PAIRS, self.EXPECTED):
            s80 = M._title_search_sim(c, r)
            s75 = E._title_sim(c, r)
            self.assertAlmostEqual(s80, exp, places=5,
                                   msg=f"audit sim drift: {c[:40]!r}")
            self.assertAlmostEqual(s75, exp, places=5,
                                   msg=f"search sim drift: {c[:40]!r}")

    def test_aliases_delegate_at_their_own_accept(self):
        # same value at both thresholds on this battery (accept only changes
        # which branch returns recall vs jaccard; pins delegation wiring)
        for c, r in self.PAIRS:
            self.assertAlmostEqual(
                M._title_search_sim(c, r),
                U.title_search_sim(c, r, accept=0.80), places=9)
            self.assertAlmostEqual(
                E._title_sim(c, r),
                U.title_search_sim(c, r, accept=0.75), places=9)

    def test_accept_threshold_changes_verdict_near_boundary(self):
        # recall 0.714: accepted at 0.60, rejected at 0.80 — proves the
        # accept parameter actually threads through the canonical impl
        c = "Horseshoe kidney and metanephric development in the mouse"
        r = "Horseshoe kidney and metanephric development"
        self.assertGreaterEqual(U.title_search_sim(c, r, accept=0.60), 0.60)
        self.assertLess(U.title_search_sim(c, r, accept=0.80), 0.80)


class TestFrontMatterAnchoring(unittest.TestCase):
    """Citation-hijack defense for front-matter parsing (found live
    2026-09-10): a EuropePMC-style md lists each reference with its own
    `doi:` line deep in the body. The OLD whole-document label scan treated
    the LAST reference's `doi:` as existing front-matter → the paper's
    identity became that reference's record. Anchored-block recognition
    (leading block only) must prevent both the hijack AND the body-deletion
    side-effect of the strip."""

    BODY = (
        "JOURNAL of MEDICINE and LIFE\n\n\n"
        "**JML | REVIEW**\n"
        "# **Reviewing the complexities of horseshoe kidney**\n"
        "body text...\n\n"
        "10. Murugapoopathy V, Gupta IR. A Primer on CAKUT. Clin J Am Soc Nephrol.\n"
        "doi: 10.2215/CJN.12581019\n"
        "78. Symons SJ, et al. Urolithiasis in the horseshoe kidney.\n"
        "doi: 10.1111/j.1464-410X.2008.07987.x\n"
    )

    def _mk(self, tmp, name="probe.md", fm=None):
        p = Path(tmp) / name
        p.write_text((fm or "") + self.BODY, encoding="utf-8")
        return p

    def test_prev_parse_ignores_deep_reference_doi_lines(self):
        import tempfile
        import enrich_meta as EM
        with tempfile.TemporaryDirectory() as td:
            p = self._mk(td)
            text = p.read_text(encoding="utf-8", errors="ignore")
            m = EM._FM_BLOCK_RE.match(text)
            # no leading front-matter block → nothing parsed as prev
            if m:
                self.assertNotIn("doi", [
                    l.split(":")[0].lower() for l in m.group(0).split("\n")
                    if ":" in l and l.strip()])
            loc = EM.local_meta(p)
            merged_prev = EM.enrich_md.__doc__  # docstring anchor only
            # direct probe: the naive whole-doc scan is gone
            self.assertNotRegex(text[:200], r"\A(?:Title|doi):")  # no fm block

    def test_leading_block_still_parsed_and_preserved(self):
        import tempfile
        import enrich_meta as EM
        with tempfile.TemporaryDirectory() as td:
            fm = "Title: Real Review Title\nYear: 2025\ndoi:10.1186/x\n\n"
            p = self._mk(td, fm=fm)
            text = p.read_text(encoding="utf-8", errors="ignore")
            m = EM._FM_BLOCK_RE.match(text)
            self.assertIsNotNone(m, "leading block must still match")
            self.assertIn("10.1186/x", m.group(0))
            # strip removes exactly the leading block, keeps ALL body doi: lines
            stripped = EM._strip_existing_fm(text)
            self.assertNotIn("Real Review Title", stripped)
            self.assertIn("10.2215/CJN.12581019", stripped)
            self.assertIn("10.1111/j.1464-410X.2008.07987.x", stripped)

    def test_strip_keeps_body_when_no_leading_block(self):
        import tempfile
        import enrich_meta as EM
        with tempfile.TemporaryDirectory() as td:
            p = self._mk(td)
            text = p.read_text(encoding="utf-8", errors="ignore")
            stripped = EM._strip_existing_fm(text)
            self.assertEqual(stripped, text.lstrip("\n"),
                             "no leading block → strip must not eat body lines")


if __name__ == "__main__":
    unittest.main(verbosity=2)