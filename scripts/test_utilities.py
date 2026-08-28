#!/usr/bin/env python3
"""test_utilities.py — regression tests for scripts/bib_utils.py and the
scripts/ utilities' matching logic.

Run:  python3 -B scripts/test_utilities.py        (from the repo root)
The tests are pure (no network, no real .bib / parent_store), using inline
fixtures that mirror the real Zotero export and parent_store formats.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "metadata"))

import bib_utils as U
import bib_to_parent_store as B
import zotero_access as ZA


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


# ---------------------------------------------------------------------------
# zotero_access — MCP-first Zotero layer (network-free; HTTP mocked)
# ---------------------------------------------------------------------------

SEARCH_MD = """\
# Search Results for 'eph receptor'

## 1. Cell segregation and border sharpening by Eph receptor
**Type:** journalArticle
**Item Key:** GB78JG9S
**Date:** 07/2017
**Authors:** Taylor, Harriet B.; Khuong, Anaïs; Wu, Zhonglin
**Abstract:** Eph receptor and ephrin signalling...

## 2. The Shb scaffold binds the Nck adaptor protein
**Type:** journalArticle
**Item Key:** UUFX4CT5
**Date:** 03/2020
**Authors:** Wagner, Melany J.
"""


class TestZoteroAccess(unittest.TestCase):
    def setUp(self):
        self._saved_env = os.environ.get("BIB_RAG_ZOTERO_MCP")
        self._saved_flag = ZA._mcp_disabled
        os.environ["BIB_RAG_ZOTERO_MCP"] = "0"
        ZA._mcp_disabled = True

    def tearDown(self):
        if self._saved_env is None:
            os.environ.pop("BIB_RAG_ZOTERO_MCP", None)
        else:
            os.environ["BIB_RAG_ZOTERO_MCP"] = self._saved_env
        ZA._mcp_disabled = self._saved_flag

    def test_parse_search_markdown(self):
        items = ZA._parse_search_markdown(SEARCH_MD)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["key"], "GB78JG9S")
        self.assertEqual(items[0]["title"],
                         "Cell segregation and border sharpening by Eph receptor")
        self.assertIn("Taylor", items[0]["authors"])
        self.assertEqual(items[1]["key"], "UUFX4CT5")

    def test_parse_search_markdown_empty(self):
        self.assertEqual(ZA._parse_search_markdown(""), [])
        self.assertEqual(ZA._parse_search_markdown("## 1. no key here"), [])

    def test_mcp_search_unwraps_structured_content(self):
        # REGRESSION: zotero_search_items returns structuredContent = {"result": "<markdown>"}.
        # It must be parsed as markdown — not JSON-dumped (which yielded 0 items).
        fake = mock.Mock()
        fake.call_tool.return_value = {"result": SEARCH_MD}
        with mock.patch.object(ZA, "_get_mcp", return_value=fake):
            items = ZA._mcp_search("eph")
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["key"], "GB78JG9S")

    def test_mcp_item_unwraps_result(self):
        fake = mock.Mock()
        fake.call_tool.return_value = {"result": {
            "key": "K9",
            "data": {"title": "T", "DOI": "10.1/9", "date": "2020-05-01",
                     "publicationTitle": "J", "creators": []},
        }}
        with mock.patch.object(ZA, "_get_mcp", return_value=fake):
            item = ZA._mcp_item("K9")
        self.assertEqual(item["doi"], "10.1/9")
        self.assertEqual(item["year"], "2020")

    def test_normalize_item(self):
        d = {
            "key": "K1", "title": "A Title", "DOI": "10.1/x", "date": "2021-03-01",
            "publicationTitle": "Journal X", "volume": "5", "issue": "2",
            "pages": "10-20",
            "creators": [{"firstName": "Jane", "lastName": "Doe"},
                         {"lastName": "Smith"}],
        }
        n = ZA._normalize_item(d, "K1")
        self.assertEqual(n["doi"], "10.1/x")
        self.assertEqual(n["year"], "2021")
        self.assertEqual(n["journal"], "Journal X")
        self.assertEqual(n["authors"], "Doe,Jane; Smith")
        self.assertEqual(n["item_type"], "")

    def test_display_authors(self):
        self.assertEqual(ZA.display_authors("Doe,Jane;Smith,John"),
                         "Jane Doe and John Smith")
        self.assertEqual(ZA.display_authors("Doe,Jane;Smith,John;Wong,Ann;Li,Bo"),
                         "Jane Doe et al.")
        self.assertEqual(ZA.display_authors("Smith,John"), "John Smith")
        self.assertEqual(ZA.display_authors(""), "Unknown")

    def test_http_fallback_search(self):
        with mock.patch.object(ZA, "_http_json", return_value=[
            {"key": "K9", "data": {"title": "T", "DOI": "10.1/9", "date": "2020",
                                   "creators": []}}
        ]):
            items = ZA.zotero_search("anything")
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["doi"], "10.1/9")
        self.assertEqual(items[0]["key"], "K9")

    def test_http_fallback_item(self):
        with mock.patch.object(ZA, "_http_json", return_value={
            "key": "K1", "data": {"title": "T", "DOI": "10.1/x",
                                  "date": "2019-01-01", "creators": []}
        }):
            item = ZA.zotero_item("K1")
        self.assertIsNotNone(item)
        self.assertEqual(item["year"], "2019")

    def test_both_unavailable_graceful(self):
        with mock.patch.object(ZA, "_http_json", return_value=None):
            self.assertEqual(ZA.zotero_search("anything"), [])
            self.assertIsNone(ZA.zotero_item("K1"))
            self.assertFalse(ZA.available())


if __name__ == "__main__":
    unittest.main(verbosity=2)
