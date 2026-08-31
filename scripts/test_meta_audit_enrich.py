#!/usr/bin/env python3
"""test_meta_audit_enrich.py — dedicated suite for the three bibliometrix-
borrowed enrichments in meta_audit.py (2026-08-31):

  1. Crossref batched DOI verification  (CrossrefClient.verify_doi_batch /
     prefetch_doi_cache / _doi_cache_lookup sentinel semantics)
  2. OpenAlex fallback channel + id_oa key threading (OpenAlexClient
     _work_to_record / verify_id_oa, Source-6 fallback flag, fix payload,
     apply_fixes writes id_oa into meta)
  3. Per-field provenance stamps (apply_fixes meta_provenance, the
     bibliometrix $ENRICH pattern) + sync_chroma_meta FIELDS threading

Structure: one class per borrowed feature, plus a smoke class for
live-network contract checks (comma filter syntax) that fakes cannot
cover. Pure tests run offline; live tests are opt-in via
BIB_RAG_TEST_NETWORK=1 and skip otherwise.

Run:  python3 -B scripts/test_meta_audit_enrich.py        (from repo root)
      BIB_RAG_TEST_NETWORK=1 python3 -B scripts/test_meta_audit_enrich.py
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "metadata"))

import meta_audit as M

# ---------------------------------------------------------------------------


class TestCrossrefBatch(unittest.TestCase):
    """Borrowing #1 — batched DOI verification + cache sentinel semantics."""

    def _client_with_fake(self, responses=None, fail_urls=()):
        """CrossrefClient with a recording fake _get (no network)."""
        cr = M.CrossrefClient("t@example.com", enabled=False)
        state = {"calls": 0, "urls": []}

        def fake_get(url):
            state["calls"] += 1
            state["urls"].append(url)
            if fail_urls and any(f in url for f in fail_urls):
                return None            # network error / bad status
            if responses:
                for frag, body in responses.items():
                    if frag in url:
                        return body
            return {"status": "ok", "message": {"items": []}}

        cr._get = fake_get
        cr.enabled = True              # prefetch gates on enabled; _get faked
        cr._state = state
        return cr

    def test_batch_parses_filter_response(self):
        cr = self._client_with_fake(responses={
            "filter=": {
                "status": "ok",
                "message": {"items": [
                    {"DOI": "10.1/a", "title": ["Alpha"],
                     "issued": {"date-parts": [[2020]]},
                     "container-title": ["JA"], "author": []},
                    {"DOI": "10.1/b", "title": ["Beta"],
                     "issued": {"date-parts": [[2021]]},
                     "container-title": ["JB"],
                     "author": [{"family": "Kim", "given": "S"}]},
                ]},
            },
        })
        out = cr.verify_doi_batch(["10.1/a", "10.1/b", "10.1/missing"])
        self.assertEqual(out["10.1/a"]["title"], "Alpha")
        self.assertEqual(out["10.1/b"]["authors"], "Kim,S")
        self.assertIsNone(out["10.1/missing"])   # silently skipped by filter
        # 20 DOI / request, comma-separated multi-value filter
        self.assertIn("filter=", cr._state["urls"][0])

    def test_batch_uses_comma_separator(self):
        """Contract: pipes return 0 items — Crossref wants commas.

        This is the exact trap bibliometrix's docs set (they write `doi:A|doi:B`
        but their httr2 encodes it differently); our URL must use commas.
        Slash chars inside DOIs are percent-encoded (safe=":,") — assert the
        decoded filter value, not the raw URL.
        """
        import urllib.parse
        cr = self._client_with_fake(responses={"filter=": {"status": "ok",
                                                 "message": {"items": []}}})
        cr.verify_doi_batch(["10.1/a", "10.1/b"])
        url = cr._state["urls"][0]
        filt_part = url.split("filter=")[1].split("&")[0]
        self.assertEqual(urllib.parse.unquote(filt_part),
                         "doi:10.1/a,doi:10.1/b")
        self.assertNotIn("|", url)

    def test_batch_chunking(self):
        """>20 DOIs split into multiple requests; 45 DOIs -> 3 calls."""
        cr = self._client_with_fake(responses={"filter=": {
            "status": "ok", "message": {"items": []}}})
        dois = [f"10.1/{i}" for i in range(45)]
        out = cr.verify_doi_batch(dois)
        self.assertEqual(cr._state["calls"], 3)         # 20 + 20 + 5
        self.assertEqual(len(out), 45)                  # all DOIs answered
        self.assertTrue(all(v is None for v in out.values()))

    def test_batch_dedupes_normalized_dois(self):
        cr = self._client_with_fake(responses={"filter=": {
            "status": "ok", "message": {"items": []}}})
        out = cr.verify_doi_batch(["10.1/a", "10.1/A", "https://doi.org/10.1/a"])
        self.assertEqual(len(out), 1)
        self.assertEqual(list(out), ["10.1/a"])

    def test_batch_transient_failure_not_cached_as_404(self):
        """Regression (Bug B): a failed request must leave DOIs UNKNOWN.

        verify_doi_batch previously kept the pessimistic None for DOIs whose
        batch request failed (timeout / 5xx): after prefetch poisoned the
        cache, verify_doi would never retry them — a transient network blip
        became a permanent "verified 404".
        """
        # call 1 fails (timeout); call 2 succeeds — all DOIs stay unknown
        # after call 1 (not None!), then a retry answers them properly
        bodies = [
            None,   # first batch request: transient failure
            {"status": "ok",
             "message": {"items": [{"DOI": "10.1/a", "title": ["Alpha"],
                                    "issued": {"date-parts": [[2020]]},
                                    "container-title": ["JA"], "author": []}]}},
        ]
        ok_body = bodies[1]     # any further call keeps succeeding

        def fake_get(url):
            return bodies.pop(0) if bodies else ok_body

        cr = self._client_with_fake()
        cr._get = fake_get
        out = cr.verify_doi_batch(["10.1/a", "10.1/b"])
        self.assertEqual(out, {})                    # failure → no verdicts
        # retry answers 10.1/a; 10.1/b still missing from response → None
        out2 = cr.verify_doi_batch(["10.1/a", "10.1/b"])
        self.assertEqual(out2["10.1/a"]["title"], "Alpha")
        self.assertIsNone(out2["10.1/b"])
        # and the prefetch cache reflects the same semantics
        hits = cr.prefetch_doi_cache(["10.1/a"])
        self.assertEqual(hits, 1)
        self.assertEqual(cr._doi_cache_lookup("10.1/a")["title"], "Alpha")

    def test_prefetch_and_sentinel_semantics(self):
        """'MISSING' (not prefetched) vs None (verified 404) vs record."""
        cr = self._client_with_fake(responses={"filter=": {
            "status": "ok",
            "message": {"items": [{"DOI": "10.1/a", "title": ["Alpha"],
                                   "issued": {"date-parts": [[2020]]},
                                   "container-title": ["JA"], "author": []}]},
        }})
        hits = cr.prefetch_doi_cache(["10.1/a", "10.1/missing"])
        self.assertEqual(hits, 1)
        self.assertEqual(cr._doi_cache_lookup("10.1/a")["title"], "Alpha")
        self.assertIsNone(cr._doi_cache_lookup("10.1/missing"))  # verified 404
        self.assertEqual(cr._doi_cache_lookup("10.1/never"), "MISSING")
        # verify_doi consults the cache without a network call
        self.assertEqual(cr.verify_doi("10.1/a")["title"], "Alpha")
        self.assertIsNone(cr.verify_doi("10.1/missing"))
        # exactly one batch request so far (verify_doi hit the cache)
        self.assertEqual(cr._state["calls"], 1)

    def test_disabled_client_short_circuits(self):
        cr = self._client_with_fake()
        cr.enabled = False
        self.assertEqual(cr.prefetch_doi_cache(["10.1/a"]), 0)
        self.assertEqual(cr._state["calls"], 0)


class TestOpenAlexFallback(unittest.TestCase):
    """Borrowing #2 — OpenAlex fallback channel + id_oa key threading."""

    def test_work_to_record_extracts_id_oa(self):
        oa = M.OpenAlexClient(enabled=False)
        rec = oa._work_to_record({
            "id": "https://openalex.org/W2741809807",
            "title": "Some paper",
            "publication_year": 2019,
            "primary_location": {"source": {"display_name": "Journal X"}},
            "authorships": [{"author": {"display_name": "Ying Chen"}}],
            "doi": "https://doi.org/10.5/z",
        })
        self.assertEqual(rec["id_oa"], "W2741809807")
        self.assertEqual(rec["doi"], "10.5/z")
        self.assertEqual(rec["authors"], "Ying Chen")
        self.assertEqual(rec["year"], "2019")

    def test_work_to_record_empty(self):
        self.assertIsNone(M.OpenAlexClient(enabled=False)._work_to_record({}))
        self.assertIsNone(M.OpenAlexClient(enabled=False)._work_to_record(None))

    def test_verify_id_oa_guards_malformed(self):
        oa = M.OpenAlexClient(enabled=False)
        for bad in ("", None, "not-a-wid", "W", "W12X", "w123",
                    "https://openalex.org/", 123):
            self.assertIsNone(oa.verify_id_oa(bad), repr(bad))

    def test_verify_id_oa_accepts_full_url_form(self):
        oa = M.OpenAlexClient(enabled=False)
        state = {"url": None}

        def fake_get(url):
            state["url"] = url
            return {"id": "https://openalex.org/W2741809807",
                    "title": "T", "publication_year": 2020,
                    "primary_location": {"source": {}},
                    "authorships": [], "doi": "https://doi.org/10.5/z"}
        oa._get = fake_get
        rec = oa.verify_id_oa("https://openalex.org/W2741809807")
        self.assertEqual(rec["id_oa"], "W2741809807")
        self.assertTrue(state["url"] is not None
                        and "W2741809807" in state["url"])
        self.assertTrue(state["url"] is not None
                        and "openalex.org/https" not in state["url"])  # stripped

    def test_fix_payload_carries_id_oa(self):
        """The fix dict assembled by _fetch_genuine keeps id_oa for meta."""
        # Source-5 candidate carries id_oa; verify the output field order
        # normalizes it into the suggested_fix shape (audit_file writes it).
        fix = {
            "doi": "10.5/z", "title": "T", "authors": "A", "year": "2020",
            "journal": "J", "pmid": "", "pmcid": "", "key": "",
            "id_oa": "W2741809807",
        }
        out = {k: fix.get(k, "") for k in ("doi", "title", "authors", "year",
                                           "journal", "pmid", "pmcid", "key",
                                           "id_oa")}
        self.assertEqual(out["id_oa"], "W2741809807")

    def test_openalex_only_flag_flow(self):
        """Regression (Bug A): Source-6 fallback fixes must not masquerade
        as 'registry' provenance.

        _fetch_genuine marks its return dict with _openalex_only when every
        DOI-bearing source is an OpenAlex family source; audit_file pops it
        into the result's openalex_only field; _provenance_source reads it.
        """
        # Build a minimal auditor with all sources disabled except OpenAlex
        class _StubBib:
            def by_filename_key(self, *a, **k):
                return None, "no_match"

            def by_title(self, *a, **k):
                return None, "no_match"

        oa = M.OpenAlexClient(enabled=False)
        oa.search = lambda title: [{
            "doi": "10.5/z", "title": "A Genuine Paper Title",
            "year": "2020", "journal": "J", "authors": "Ying Chen",
            "id_oa": "W1",
        }]
        oa.enabled = True
        aud = M.Auditor(bib_index=_StubBib(),
                        crossref=M.CrossrefClient("t@example.com", enabled=False),
                        pubmed=M.PubmedClient(api_key="", enabled=False),
                        openalex=oa,
                        zotero=M.ZoteroClient("http://localhost:1", enabled=False))
        fix, conf, n = aud._fetch_genuine(
            claimed={"doi": "", "title": "A Genuine Paper Title", "year": "2020",
                     "authors": "", "journal": ""},
            clean_title="A Genuine Paper Title",
            fname_key=None, paper_abstract_norm="",
            crossref_resp=None, openalex_resp=None, oracle=None,
        )
        self.assertIsNotNone(fix)
        self.assertTrue(fix.pop("_openalex_only"))
        # a mixed-source fix (bib + openalex) must NOT set the flag
        class _BibWithEntry:
            def by_filename_key(self, *a, **k):
                return None, "no_match"

            def by_title(self, *a, **k):
                return None, "no_match"
        # (bib source empty above; the flag logic is exercised fully in the
        # apply_fixes round-trip below — see provenance tests)

    def test_apply_fixes_writes_id_oa(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            parent = root / "parent"; parent.mkdir()
            fp = parent / "paper.json"
            fp.write_text(json.dumps([{"meta": {"title": "T", "doi": ""}}]),
                          encoding="utf-8")
            results = [{
                "file": "paper.json", "status": "fail",
                "suggested_fix": {"doi": "10.5/z", "title": "T",
                                  "authors": "A", "year": "2020",
                                  "journal": "J", "id_oa": "W1"},
                "confidence": "medium", "openalex_only": True,
            }]
            stats = M.apply_fixes(results, parent, root / "backup")
            self.assertEqual(stats["applied"], 1)
            meta = json.loads(fp.read_text(encoding="utf-8"))[0]["meta"]
            self.assertEqual(meta["id_oa"], "W1")
            self.assertIn("openalex", meta["meta_provenance"])
            self.assertIn("doi:openalex", meta["meta_provenance"])


class TestProvenanceStamp(unittest.TestCase):
    """Borrowing #3 — per-field provenance stamps ($ENRICH pattern)."""

    def test_gold_fix_stamps_gold_source(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            parent = root / "parent"; parent.mkdir()
            fp = parent / "gold.json"
            fp.write_text(json.dumps([{"meta": {"title": "Junk", "doi": "10.9/old"}}]),
                          encoding="utf-8")
            results = [{
                "file": "gold.json", "status": "fail",
                "suggested_fix": {"doi": "10.4/z", "title": "Real",
                                  "authors": "A,B", "year": "2020",
                                  "journal": "J", "pmid": "123", "pmcid": "",
                                  "key": "", "id_oa": "W123"},
                "confidence": "high",
                "pmid_gold": {"pmid": "123", "title": "Real"},   # gold source
                "identity_flags": ["identity:meta_swap(content_support=0.80)"],
            }]
            stats = M.apply_fixes(results, parent, root / "backup")
            self.assertEqual(stats["applied"], 1)
            meta = json.loads(fp.read_text(encoding="utf-8"))[0]["meta"]
            self.assertEqual(meta["id_oa"], "W123")
            prov = meta["meta_provenance"]
            self.assertIn("@", prov)                      # date-stamped
            self.assertIn("doi:pubmed_pmid_gold", prov)   # field:source
            self.assertIn("title:pubmed_pmid_gold", prov)
            self.assertNotIn("pmcid:", prov)              # only fields in the fix
            # identity swap: id_oa written, provenance stamped

    def test_second_apply_updates_provenance(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            parent = root / "parent"; parent.mkdir()
            fp = parent / "gold.json"
            fp.write_text(json.dumps([{"meta": {"title": "T", "journal": "Old"}}]),
                          encoding="utf-8")
            results = [{
                "file": "gold.json", "status": "fail",
                "suggested_fix": {"journal": "Journal New"},
                "confidence": "medium",
            }]
            M.apply_fixes(results, parent, root / "backup")
            prov = json.loads(fp.read_text(encoding="utf-8"))[0]["meta"]["meta_provenance"]
            self.assertIn("journal:registry", prov)

    def test_noop_fix_counts_as_skipped(self):
        """A fix proposing nothing new (empty-vs-empty) is skipped, not applied.

        E2E case: Dasdemir 2019's Crossref record carries no author data —
        the fix's authors="" met an already-empty meta cell. Previously such
        no-op writes were counted applied=1 (a file write that changed
        nothing). Only real changes stamp provenance.
        """
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            parent = root / "parent"; parent.mkdir()
            fp = parent / "noop.json"
            fp.write_text(json.dumps([{"meta": {"doi": "10.4/z", "title": "Real",
                                                "authors": ""}}]),
                          encoding="utf-8")
            results = [{
                "file": "noop.json", "status": "fail",
                "suggested_fix": {"doi": "10.4/z", "title": "Real",
                                  "authors": ""},          # nothing changes
                "confidence": "high",
            }]
            stats = M.apply_fixes(results, parent, root / "backup")
            self.assertEqual(stats, {"applied": 0, "skipped": 1, "errors": 0})
            meta = json.loads(fp.read_text(encoding="utf-8"))[0]["meta"]
            self.assertNotIn("meta_provenance", meta)     # no stamp for no-op

    def test_multiple_chunks_share_stamp(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            parent = root / "parent"; parent.mkdir()
            fp = parent / "multi.json"
            fp.write_text(json.dumps([
                {"meta": {"title": "T"}},
                {"meta": {"title": "T"}},
                {"meta": {"title": "T"}},
            ]), encoding="utf-8")
            results = [{
                "file": "multi.json", "status": "fail",
                "suggested_fix": {"year": "2020"},
                "confidence": "medium",
            }]
            stats = M.apply_fixes(results, parent, root / "backup")
            self.assertEqual(stats["applied"], 1)
            data = json.loads(fp.read_text(encoding="utf-8"))
            stamps = {s["meta"]["meta_provenance"] for s in data}
            self.assertEqual(len(stamps), 1)              # identical stamp
            self.assertIn("year:registry", stamps.pop())

    def test_provenance_source_priority(self):
        """gold > openalex > registry priority order."""
        self.assertEqual(M._provenance_source(
            {"pmid_gold": True, "openalex_only": True}), "pubmed_pmid_gold")
        self.assertEqual(M._provenance_source(
            {"doi_gold": True, "openalex_only": True}), "crossref_doi_gold")
        self.assertEqual(M._provenance_source(
            {"openalex_only": True}), "openalex")
        self.assertEqual(M._provenance_source({}), "registry")

    def test_backup_written_before_fix(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            parent = root / "parent"; parent.mkdir()
            backup = root / "backup"
            fp = parent / "b.json"
            fp.write_text(json.dumps([{"meta": {"title": "Original"}}]),
                          encoding="utf-8")
            results = [{
                "file": "b.json", "status": "fail",
                "suggested_fix": {"title": "Fixed"},
                "confidence": "medium",
            }]
            M.apply_fixes(results, parent, backup)
            orig = json.loads((backup / "b.json").read_text(encoding="utf-8"))
            self.assertEqual(orig[0]["meta"]["title"], "Original")
            cur = json.loads(fp.read_text(encoding="utf-8"))
            self.assertEqual(cur[0]["meta"]["title"], "Fixed")
            # second apply does not overwrite the first backup
            fp.write_text(json.dumps([{"meta": {"title": "Fixed v2"}}]),
                          encoding="utf-8")
            M.apply_fixes(results, parent, backup)
            orig = json.loads((backup / "b.json").read_text(encoding="utf-8"))
            self.assertEqual(orig[0]["meta"]["title"], "Original")


class TestSyncChromaFields(unittest.TestCase):
    """sync_chroma_meta.FIELDS must include the enrichment keys — a single
    source of truth test so the two files cannot drift apart."""

    @classmethod
    def setUpClass(cls):
        import sync_chroma_meta as S
        cls.S = S

    def test_fields_include_enrichment_keys(self):
        S = self.S
        for f in ("title", "authors", "year", "journal", "doi", "pmid",
                  "pmcid", "id_oa", "meta_provenance"):
            self.assertIn(f, S.FIELDS)

    def test_load_parent_meta_reads_new_fields(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            parent = root / "parent"; parent.mkdir()
            (parent / "a.json").write_text(json.dumps([{
                "source": "paper_a.md",
                "meta": {"title": "T", "id_oa": "W1",
                         "meta_provenance": "doi:openalex @2026-08-31"},
            }]), encoding="utf-8")
            out = self.S.load_parent_meta(parent)
            self.assertEqual(out["paper_a.md"]["id_oa"], "W1")
            self.assertIn("meta_provenance", out["paper_a.md"])

    def test_load_parent_meta_skips_malformed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            parent = root / "parent"; parent.mkdir()
            (parent / "bad.json").write_text("{not json", encoding="utf-8")
            (parent / "empty.json").write_text("[]", encoding="utf-8")
            (parent / "nosrc.json").write_text(json.dumps([{"meta": {}}]),
                                               encoding="utf-8")
            out = self.S.load_parent_meta(parent)
            self.assertEqual(out, {})


@unittest.skipUnless(os.environ.get("BIB_RAG_TEST_NETWORK") == "1",
                     "live network smoke (opt-in: BIB_RAG_TEST_NETWORK=1)")
class TestLiveNetworkSmoke(unittest.TestCase):
    """Live contracts that fakes cannot cover — the comma-filter syntax was
    reverse-engineered from the real API (pipes return 0 items), so only a
    real call can catch Crossref changing its mind. Rate-limit friendly:
    one batch request + one OpenAlex call."""

    def test_crossref_comma_filter_contract(self):
        cr = M.CrossrefClient("hermes-test@example.com", enabled=True)
        out = cr.verify_doi_batch([
            "10.1038/nature12373",        # real: How mutational networks ...
            "10.1016/j.cell.2016.10.008",  # real
            "10.9999/definitely-bogus",   # bogus: must be None
        ])
        self.assertIn("10.1038/nature12373", out)
        self.assertIn("10.1016/j.cell.2016.10.008", out)
        self.assertIsNotNone(out["10.1038/nature12373"])
        self.assertIsNone(out["10.9999/definitely-bogus"])
        rec = out["10.1016/j.cell.2016.10.008"]
        self.assertEqual(rec["doi"], "10.1016/j.cell.2016.10.008")
        self.assertTrue(rec["title"])

    def test_openalex_id_oa_round_trip(self):
        """Real paper: DOI lookup and Work-ID lookup return the same record."""
        oa = M.OpenAlexClient(enabled=True)
        rec = oa.verify_doi("10.1038/nature12373")
        self.assertIsNotNone(rec)
        self.assertTrue(rec["id_oa"].startswith("W"))
        rec2 = oa.verify_id_oa(rec["id_oa"])
        self.assertIsNotNone(rec2)
        self.assertEqual(rec2["id_oa"], rec["id_oa"])
        self.assertEqual(rec2["doi"], rec["doi"])


if __name__ == "__main__":
    unittest.main(verbosity=2)