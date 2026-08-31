#!/usr/bin/env python3
"""test_doi_normalizers.py — cross-implementation DOI canonicalization pin.

F3 regression: scripts/bib_utils.normalize_doi and src/identifiers.normalize_doi
must agree on every input class (URL-wrapped, doi:-prefixed, versioned,
Oxford-style). They serve different call sites (metadata scripts do
equality compares against canonical forms written by src/) — any drift is
a silent match miss.

Contract note: identifiers.normalize_doi returns None for junk (strict);
bib_utils.normalize_doi returns '' for empty input (lenient) — agreement is
asserted only where both produce a DOI.

Run: /usr/bin/python3.10 -B tests/test_doi_normalizers.py
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from identifiers import normalize_doi as canon  # noqa: E402
from bib_utils import normalize_doi as utils  # noqa: E402

CASES = [
    # (input, expected canonical form) — inputs BOTH normalizers accept
    ("https://doi.org/10.1186/x.", "10.1186/x"),
    ("10.1002/(SICI)X", "10.1002/(sici)x"),
    ("https://doi.org/10.1016/J.YDBIO.2021.01.002", "10.1016/j.ydbio.2021.01.002"),
    ("doi:10.1016/j.ydbio.2021.01.002v2", "10.1016/j.ydbio.2021.01.002"),  # v-strip
    ("10.1016/j.ydbio.2021.01.002.v2", "10.1016/j.ydbio.2021.01.002"),     # .vN
    ("10.1093/nar/gkv370", "10.1093/nar/gkv370"),   # Oxford: v-digits NOT a version
    ("10.1093/cvr/cvr154", "10.1093/cvr/cvr154"),   # Oxford: keep
]

UTILS_ONLY = [
    # lenient-contract fixtures accepted by bib_utils but rejected by the
    # strict canonical regex (10.\d{4,} — 10.1/2 is malformed as a DOI).
    # bib_utils must still normalize them sanely (prefix/punctuation strip).
    ("doi:10.1/2,", "10.1/2"),
]


class TestDoiNormalizerAgreement(unittest.TestCase):
    def test_utils_matches_canonical_on_all_classes(self):
        for raw, expected in CASES:
            with self.subTest(raw=raw):
                self.assertEqual(canon(raw), expected,
                                 f"canonical normalizer changed: {raw}")
                self.assertEqual(utils(raw), expected,
                                 f"bib_utils drifted from canonical: {raw}")

    def test_utils_versioned_doi_equality_compare(self):
        # the F3 failure mode: bind_zotero compares normalize_doi(a) ==
        # normalize_doi(b) — a versioned DOI must equal its canonical form
        self.assertEqual(utils("10.1016/j.ydbio.2021.01.002v2"),
                         utils("10.1016/j.ydbio.2021.01.002"))
        self.assertNotEqual(utils("10.1093/nar/gkv370"),
                            utils("10.1093/nar/gkv371"))

    def test_utils_lenient_fixtures(self):
        for raw, expected in UTILS_ONLY:
            with self.subTest(raw=raw):
                self.assertEqual(utils(raw), expected)

    def test_empty_input_contracts(self):
        self.assertEqual(utils(""), "")
        self.assertIsNone(canon("not a doi"))


if __name__ == "__main__":
    unittest.main(verbosity=2)