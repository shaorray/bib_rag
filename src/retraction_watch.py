#!/usr/bin/env python3
"""
retraction_watch.py — Retracted-DOI lookup against the Retraction Watch database.

Borrowed mechanism (paper-qa RetractionDataPostProcessor — see the
citation-management cross-survey notes): Crossref Labs mirrors the Retraction
Watch database as a CSV
(https://api.labs.crossref.org/data/retractionwatch). We keep a snapshot in
the library's data dir, extract the retracted-DOI set (both the retraction
notice's DOI and the original paper's DOI), and expose:

  - is_retracted(doi)          — set membership, zero network after load
  - load_retractions()         — download if missing/stale, then parse
  - retraction_details(doi)    — nature/type of retraction for reporting

Design: the CSV snapshot (~50-100MB) is downloaded ONCE per library and
cached with a configurable TTL (default 30 days). The DOI set itself is
reduced to a plain set of normalized DOIs at load time — matching uses
identifiers.normalize_doi on both sides so case/URL wrappers never break
the join. Zero LLM; all failures degrade to "not retracted" + a warning
flag so callers never hard-fail on network problems.

CLI:
    python3 retraction_watch.py --update              # (re)download snapshot
    python3 retraction_watch.py --check DOI [DOI...]  # check specific DOIs
    python3 retraction_watch.py --stats               # snapshot summary
"""
from __future__ import annotations

import os
import csv
import io
import json
import time
import argparse
import urllib.request
from typing import Dict, Optional, Set

try:
    from .identifiers import normalize_doi
except ImportError:  # src/ on sys.path directly (CLI/tests)
    from identifiers import normalize_doi

try:
    from .kb_config import get_config
except ImportError:
    from kb_config import get_config

RETRACTIONWATCH_URL = "https://api.labs.crossref.org/data/retractionwatch"
DEFAULT_CACHE_DAYS = 30
DOWNLOAD_TIMEOUT = 600  # the CSV is large

# env kill-switch: RETRACTION_CHECK=0 disables every check (no download, no lookup)


def snapshot_path() -> str:
    """Where the CSV snapshot lives (per active library)."""
    try:
        cfg = get_config()
        return os.path.join(cfg.get("data_dir", ""), "retraction_watch.csv")
    except Exception:
        return os.path.join(os.path.expanduser("~"), ".bib_rag_retraction_watch.csv")


def _cache_expired(path: str, days: int) -> bool:
    if days < 0:
        return False
    try:
        age = time.time() - os.path.getmtime(path)
        return age > days * 86400
    except OSError:
        return True


def download_snapshot(dest_path: str, mailto: str = "") -> bool:
    """Download the Retraction Watch CSV from Crossref Labs. Returns success."""
    url = RETRACTIONWATCH_URL + (f"?mailto={mailto}" if mailto else "")
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    tmp = dest_path + ".tmp"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "bib_rag/1.0"})
        with urllib.request.urlopen(req, timeout=300) as r:
            with open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
        os.replace(tmp, dest_path)  # atomic swap
        return True
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        return False


_retraction_cache = None  # module memo: parsed retracted-DOI set (66MB CSV)


def load_retractions(cache_days: int = DEFAULT_CACHE_DAYS,
                     snapshot: Optional[str] = None,
                     ) -> Set[str]:
    """Load the retracted-DOI set from the local snapshot.
    Missing + expired snapshot → empty set (offline-safe; caller may
    trigger --update separately). DOIs are normalized at load time.

    The parsed set is memoized module-level: the snapshot is a 66MB CSV
    (~1.5s to parse) and `is_retracted()` calls it once per DOI — without
    the memo a 1270-DOI batch scan re-parses the CSV 1270 times (~33 min
    measured). Pass `snapshot` explicitly to bypass the cache.
    """
    global _retraction_cache
    path = snapshot or snapshot_path()
    if snapshot is None and _retraction_cache is not None:
        return _retraction_cache
    if not os.path.exists(path) or _cache_expired(path, cache_days):
        return set()
    dois: Set[str] = set()
    try:
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            # Retraction Watch columns: RetractionDOI, OriginalPaperDOI, RetractionNature, ...
            for row in reader:
                for col in ("RetractionDOI", "OriginalPaperDOI"):
                    nd = normalize_doi(row.get(col, "") or "")
                    if nd:
                        dois.add(nd)
    except (OSError, csv.Error, UnicodeDecodeError):
        return set()
    if snapshot is None:
        _retraction_cache = dois
    return dois


def retraction_details(doi: str, snapshot: Optional[str] = None) -> Optional[Dict[str, str]]:
    """Nature/type + retraction date for one retracted DOI (for reports)."""
    path = snapshot or snapshot_path()
    if not os.path.exists(path):
        return None
    nd = normalize_doi(doi)
    if not nd:
        return None
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                for col in ("RetractionDOI", "OriginalPaperDOI"):
                    if normalize_doi(row.get(col, "") or "") == nd:
                        return {
                            "nature": row.get("RetractionNature", "") or "",
                            "reason": row.get("Reason", "") or "",
                            "date": row.get("RetractionDate", "") or "",
                            "original_doi": row.get("OriginalPaperDOI", "") or "",
                            "retraction_doi": row.get("RetractionDOI", "") or "",
                        }
    except (OSError, csv.Error, UnicodeDecodeError):
        return None
    return None


def is_retracted(doi: str, retractions: Optional[Set[str]] = None) -> bool:
    """Zero-network membership check (normalize both sides first)."""
    nd = normalize_doi(doi)
    if not nd:
        return False
    if retractions is None:
        retractions = load_retractions()
    return nd in retractions


def check_sources(store_dir: Optional[str] = None,
                  retractions: Optional[Set[str]] = None) -> Dict[str, dict]:
    """Check the parent_store's papers for retracted DOIs.
    Returns {source: {'doi':..., 'retracted': bool}}.

    `retractions=None` (default) loads the snapshot once (memoized) — this
    used to be a silent no-op (`retractions is not None and ...` never
    fired), so the default call reported nothing instead of scanning."""
    if os.environ.get("RETRACTION_CHECK", "1") == "0":
        return {}
    if store_dir is None:
        try:
            cfg = get_config()
            store_dir = cfg["parent_store_dir"]
        except Exception:
            return {}
    if not store_dir or not os.path.isdir(store_dir):
        return {}
    if retractions is None:
        retractions = load_retractions()
    out: Dict[str, dict] = {}
    for fn in sorted(os.listdir(store_dir)):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(store_dir, fn), encoding="utf-8") as f:
                parents = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        meta = (parents[0].get("meta", {}) or {}) if parents else {}
        doi = meta.get("doi", "") or ""
        if not doi:
            continue
        nd = normalize_doi(doi)
        if not nd:
            continue
        if nd in retractions:
            out[meta.get("title") or fn] = {"doi": nd, "retracted": True}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Retraction Watch DOI checks (bib_rag)")
    ap.add_argument("--update", action="store_true",
                    help="download/refresh the CSV snapshot from Crossref Labs")
    ap.add_argument("--check", nargs="*", help="DOIs to check")
    ap.add_argument("--scan", action="store_true",
                    help="scan the active library's parent_store for retracted papers")
    ap.add_argument("--mailto", default=os.environ.get("CROSSREF_MAILTO", ""))
    ap.add_argument("--cache-days", type=int, default=DEFAULT_CACHE_DAYS)
    args = ap.parse_args()

    if args.update:
        dest = snapshot_path()
        print(f"downloading {RETRACTIONWATCH_URL} → {dest} (large file, be patient)")
        ok = download_snapshot(dest, mailto=args.mailto)
        print("OK" if ok else "DOWNLOAD FAILED")
        return 0 if ok else 1

    if args.check is not None:
        retr = load_retractions(cache_days=args.cache_days)
        if not retr:
            print("no local snapshot (run --update); reporting all as not-retracted")
        for doi in args.check:
            status = "RETRACTED" if is_retracted(doi, retr) else "ok"
            print(f"{doi}: {status}")
        return 0

    if args.scan:
        retr = load_retractions(cache_days=args.cache_days)
        if not retr:
            print("no local snapshot (run --update first)")
            return 1
        hits = check_sources(retractions=retr)
        print(f"{len(hits)} retracted paper(s) in the active library")
        for title, info in hits.items():
            print(f"  {info['doi']}  {title[:70]}")
        return 0 if not hits else 2

    ap.error("pass --update, --check, or --scan")


if __name__ == "__main__":
    raise SystemExit(main())