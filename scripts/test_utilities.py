#!/usr/bin/env python3
"""test_utilities.py — regression tests for scripts/bib_utils.py and the
scripts/ utilities' matching logic.

Run:  python3 -B scripts/test_utilities.py        (from the repo root)
The tests are pure (no network, no real .bib / parent_store), using inline
fixtures that mirror the real Zotero export and parent_store formats.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bib_utils as U
import bib_to_parent_store as B
import fill_meta_key_in_parent_store as F


# ---------------------------------------------------------------------------
# bib_utils.normalize / normalize_doi / tokens / jaccard
# ---------------------------------------------------------------------------

class TestNormalize(unittest.TestCase):
    def test_normalize_basic(self):
        self.assertEqual(U.normalize("Abdul-Wajid"), "abdulwajid")
        self.assertEqual(U.normalize("Boström"), "bostrom")
        self.assertEqual(U.normalize("  Multi  Word Title! "), "multiwordtitle")
        self.assertEqual(U.normalize(""), "")

    def test_normalize_doi(self):
        self.assertEqual(U.normalize_doi("https://doi.org/10.1186/x."), "10.1186/x")
        self.assertEqual(U.normalize_doi("doi:10.1/2,"), "10.1/2")
        self.assertEqual(U.normalize_doi("10.1002/(SICI)X"), "10.1002/(sici)x")
        self.assertEqual(U.normalize_doi(""), "")

    def test_is_doi_like(self):
        self.assertTrue(U.is_doi_like("10.1002/xyz.123"))
        self.assertFalse(U.is_doi_like("123456"))       # PMID
        self.assertFalse(U.is_doi_like("AB12CD34"))     # Zotero key
        self.assertFalse(U.is_doi_like(""))

    def test_jaccard(self):
        self.assertEqual(U.jaccard({"a"}, {"a"}), 1.0)
        self.assertEqual(U.jaccard({"a"}, {"b"}), 0.0)
        self.assertEqual(U.jaccard(set(), set()), 1.0)
        self.assertAlmostEqual(U.jaccard({"a", "b"}, {"a"}), 0.5)

    def test_is_fake_doi(self):
        self.assertTrue(U.is_fake_doi("10.1002/(SICI)1097-0177(199803)211:3<204::AID-AJA2>3.0.CO;2-L"))
        self.assertTrue(U.is_fake_doi("BibKey:placeholder"))
        self.assertTrue(U.is_fake_doi("https://doi.org/10.1/x"))
        self.assertFalse(U.is_fake_doi("10.1186/s13041-021-00829-z"))
        self.assertFalse(U.is_fake_doi(""))


# ---------------------------------------------------------------------------
# filename_to_key — the drift/typo regressions
# ---------------------------------------------------------------------------

class TestFilenameToKey(unittest.TestCase):
    def test_et_al_format1(self):
        # REGRESSION: meta_audit's old regex (`__-?-?`) failed on `et_al__-_<year>`
        # and fell through to format 2, producing a garbage lastname
        # `abdul-wajid_et_al__-`. The correct result keeps a clean lastname.
        key = U.filename_to_key(
            "Abdul-Wajid_et_al__-_2015_-_T-type_Calcium_Channel_Regulation_md"
        )
        self.assertEqual(key, ("abdul-wajid", "2015", "ttypecalciumchannelregulation"))

    def test_et_al_unicode_lastname(self):
        key = U.filename_to_key(
            "Boström_et_al__-_2017_-_Comparative_cell_cycle_transcriptomics_md"
        )
        self.assertEqual(key[0], "boström")
        self.assertEqual(key[1], "2017")
        # matching later normalizes both sides, so this must normalize cleanly
        self.assertEqual(U.normalize(key[0]), "bostrom")

    def test_et_al_multiword_lastname(self):
        key = U.filename_to_key(
            "Abou_Chakra_et_al__-_2021_-_Control_of_tissue_development_md"
        )
        self.assertEqual(key[0], "abou_chakra")
        self.assertEqual(key[1], "2021")

    def test_single_author_and(self):
        key = U.filename_to_key(
            "Alert_and_Trepat_-_2020_-_Physical_Models_of_Collective_Cell_Migration_md"
        )
        self.assertEqual(key[0], "alert")
        self.assertEqual(key[1], "2020")

    def test_single_author_plain(self):
        key = U.filename_to_key("Adelmann_2022_Impact_of_cell_size_md")
        self.assertEqual(key, ("adelmann", "2022", "impactofcellsize"))

    def test_short_format(self):
        self.assertEqual(U.filename_to_key("Adelmann_2022"), ("adelmann", "2022", ""))

    def test_year_first_filenames_return_none(self):
        # filenames starting with the year have no parseable (lastname, year) key;
        # callers must fall back to title matching — not crash.
        self.assertIsNone(U.filename_to_key("2021_pv_rnaseq_md"))
        self.assertIsNone(U.filename_to_key("2022_-_A_statistical_method_md"))

    def test_internal_md_marker_preserved(self):
        # REGRESSION: the old `.replace('_md', '')` mangled titles containing
        # '_md'; only a TRAILING `_md` marker may be stripped.
        key = U.filename_to_key("Foo_et_al__-_2020_-_The_md_model_of_X_md")
        self.assertEqual(key[2], "themdmodelofx")

    def test_no_year_returns_none(self):
        # `Agrawal_et_al__-_NiCo_...` has no 4-digit year after et_al
        self.assertIsNone(U.filename_to_key(
            "Agrawal_et_al__-_NiCo_Identifies_Extrinsic_Drivers_md"
        ))


# ---------------------------------------------------------------------------
# parse_bib_entries — tab-indented Zotero export handling
# ---------------------------------------------------------------------------

BIB_FIXTURE = """\
@article{abdul-wajid_t-type_2015,
\ttitle = {T-type calcium channel regulation of neural tube closure and EphrinAEPHA signaling},
\tvolume = {11},
\tdoi = {10.1000/example.1},
\tabstract = {The neural tube closes during development.},
\tauthor = {Abdul-Wajid, Sarah and Smith, John and Doe, Jane},
\tyear = {2015},
\tjournaltitle = {Journal of Example Biology},
}

@article{parres-gold_contextual_2025,
\ttitle = {Contextual computation by competitive protein dimers},
\tdate = {2025-03-01},
\tdoi = {10.1016/j.cell.2025.01.036},
\tauthor = {Parres-Gold, M. and Others, A.},
\tjournal = {Cell},
}

@misc{moghimianavval_light-based_nodate,
\ttitle = {Light-based approaches without a year field},
\tdoi = {10.9999/noyear.1},
\tauthor = {Moghimianavval, Y.},
}

@comment{not an entry}
"""


class TestParseBibEntries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "lib.bib"
            p.write_text(BIB_FIXTURE, encoding="utf-8")
            cls.entries = U.parse_bib_entries(p)

    def test_entry_count(self):
        # @comment must not be parsed as an entry
        self.assertEqual(len(self.entries), 3)

    def test_tab_indented_year_field(self):
        # REGRESSION: bare `^year` (MULTILINE) never matched tab-indented Zotero
        # lines; the parser silently fell back to the entry-key year.
        be = next(e for e in self.entries if e.key == "abdul-wajid_t-type_2015")
        self.assertEqual(be.year, "2015")   # from the year field, not the key

    def test_date_field_fallback(self):
        be = next(e for e in self.entries if e.key == "parres-gold_contextual_2025")
        self.assertEqual(be.year, "2025")   # from `date = {2025-03-01}`

    def test_key_fallback_when_no_year_field(self):
        be = next(e for e in self.entries if e.key == "moghimianavval_light-based_nodate")
        self.assertEqual(be.year, "")       # key has no 4-digit year

    def test_authors(self):
        be = next(e for e in self.entries if e.key == "abdul-wajid_t-type_2015")
        self.assertEqual(be.author_first_lastname, "Abdul-Wajid")
        self.assertIn("Sarah", be.author_full)
        self.assertEqual(U.normalize(be.author_first_lastname), "abdulwajid")

    def test_nested_brace_title_unwrapped(self):
        be = next(e for e in self.entries if e.key == "abdul-wajid_t-type_2015")
        self.assertNotIn("{", be.title)

    def test_journal(self):
        be = next(e for e in self.entries if e.key == "abdul-wajid_t-type_2015")
        self.assertEqual(be.journal, "Journal of Example Biology")

    def test_bibentry_dict_style_access(self):
        be = next(e for e in self.entries if e.key == "abdul-wajid_t-type_2015")
        self.assertEqual(be["doi"], "10.1000/example.1")
        self.assertEqual(be.get("doi"), "10.1000/example.1")
        self.assertIsNone(be.get("nonexistent"))
        self.assertEqual(be["year"], "2015")

    def test_doi_index_and_by_doi(self):
        idx = U.BibIndex(self.entries)
        be = idx.by_doi("https://doi.org/10.1000/example.1")
        self.assertIsNotNone(be)
        self.assertEqual(be.key, "abdul-wajid_t-type_2015")
        self.assertIsNone(idx.by_doi("10.0/missing"))


# ---------------------------------------------------------------------------
# BibIndex.by_filename_key — normalization both sides
# ---------------------------------------------------------------------------

class TestBibIndexMatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "lib.bib"
            p.write_text(BIB_FIXTURE, encoding="utf-8")
            cls.idx = U.BibIndex(U.parse_bib_entries(p))

    def test_hyphenated_lastname_matches(self):
        # REGRESSION: bib_to_parent_store compared the RAW filename lastname
        # ('abdul-wajid') against the NORMALIZED bib lastname ('abdulwajid'),
        # so hyphenated surnames never matched.
        key = U.filename_to_key(
            "Abdul-Wajid_et_al__-_2015_-_T-type_Calcium_Channel_Regulation_md"
        )
        entry, status = self.idx.by_filename_key(key)
        self.assertEqual(status, "matched")
        self.assertEqual(entry.key, "abdul-wajid_t-type_2015")

    def test_year_first_key_no_year(self):
        entry, status = self.idx.by_filename_key(("", "", ""))
        self.assertEqual(status, "no_year")
        self.assertIsNone(entry)

    def test_by_title(self):
        entry, status = self.idx.by_title(
            "T-type calcium channel regulation of neural tube closure", "2015"
        )
        self.assertIn(status, ("title_matched", "multi_match"))
        self.assertIsNotNone(entry)


# ---------------------------------------------------------------------------
# bib_to_parent_store matching logic
# ---------------------------------------------------------------------------

class TestBibToParentStoreMatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "lib.bib"
            p.write_text(BIB_FIXTURE, encoding="utf-8")
            cls.entries = U.parse_bib_entries(p)

    def test_match_paper_to_entry_hyphenated_lastname(self):
        # REGRESSION: the filename lastname ('abdul-wajid', raw) must match the
        # NORMALIZED bib lastname ('abdulwajid'). Before the fix this fell
        # through to the title fallback (or no_match for unicode names).
        key = U.filename_to_key(
            "Abdul-Wajid_et_al__-_2015_-_T-type_Calcium_Channel_Regulation_md"
        )
        entry, status = B.match_paper_to_entry(key, self.entries)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.key, "abdul-wajid_t-type_2015")
        self.assertEqual(status, "matched")

    def test_match_paper_to_entry_no_year(self):
        entry, status = B.match_paper_to_entry(("foo", "", "x"), self.entries)
        self.assertEqual(status, "no_year")
        self.assertIsNone(entry)

    def test_match_paper_to_entry_no_match(self):
        entry, status = B.match_paper_to_entry(("zzz-not-a-real-author", "1999", "xyz"), self.entries)
        self.assertIsNone(entry)

    def test_match_paper_by_title_year_tolerance(self):
        # year ±1 tolerance (Fix 3): claimed 2016 should still find 2015
        entry, status = B.match_paper_by_title(
            "T-type calcium channel regulation of neural tube closure", "2016",
            self.entries,
        )
        self.assertIsNotNone(entry)


# ---------------------------------------------------------------------------
# fill_meta_key helpers
# ---------------------------------------------------------------------------

class TestFillMetaKeyHelpers(unittest.TestCase):
    def test_norm(self):
        self.assertEqual(F.norm("The  md   model!"), "the_md_model")

    def test_strip_pdf_title_prefix(self):
        s = F.strip_pdf_title_prefix("Smith et al. - 2020 - A Great Paper")
        self.assertEqual(s, "A Great Paper")
        s2 = F.strip_pdf_title_prefix("2020 - Leading Year Title")
        self.assertEqual(s2, "Leading Year Title")

    def test_shared_normalize_doi(self):
        # fill_meta_key now uses bib_utils.normalize_doi
        self.assertEqual(F.normalize_doi("http://dx.doi.org/10.1/x."), "10.1/x")


# ---------------------------------------------------------------------------
# meta_audit helpers (module import is side-effect free)
# ---------------------------------------------------------------------------

class TestMetaAuditHelpers(unittest.TestCase):
    def test_load_audited_files_pass_only(self):
        import meta_audit as M
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "meta_audit_log_2026-01-01.json").write_text(json.dumps({
                "log_entries": [
                    {"file": "a.json", "status": "pass"},
                    {"file": "b.json", "status": "fail"},
                    {"file": "c.json", "status": "unverified"},
                ]
            }), encoding="utf-8")
            self.assertEqual(M.load_audited_files(d), {"a.json"})

    def test_load_audited_files_no_log(self):
        import meta_audit as M
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(M.load_audited_files(Path(td)), set())

    def test_apply_fixes_backup_and_confidence_gate(self):
        import meta_audit as M
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            parent = root / "parent"
            parent.mkdir()
            backup = root / "backup"
            fp = parent / "x.json"
            fp.write_text(json.dumps([{"meta": {"title": "T"}}]), encoding="utf-8")

            results = [{
                "file": "x.json", "status": "fail",
                "suggested_fix": {"doi": "10.1/x", "title": "Real Title"},
                "confidence": "high",
            }]
            stats = M.apply_fixes(results, parent, backup, min_confidence="medium")
            self.assertEqual(stats["applied"], 1)
            self.assertEqual(stats["errors"], 0)
            data = json.loads(fp.read_text(encoding="utf-8"))
            self.assertEqual(data[0]["meta"]["doi"], "10.1/x")
            self.assertTrue((backup / "x.json").exists())

            # low-confidence fix below the gate must be skipped
            results2 = [{
                "file": "x.json", "status": "fail",
                "suggested_fix": {"doi": "10.1/y"}, "confidence": "low",
            }]
            stats2 = M.apply_fixes(results2, parent, backup, min_confidence="medium")
            self.assertEqual(stats2["skipped"], 1)
            self.assertEqual(json.loads(fp.read_text(encoding="utf-8"))[0]["meta"]["doi"], "10.1/x")


if __name__ == "__main__":
    unittest.main(verbosity=2)
