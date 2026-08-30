#!/usr/bin/env python3
"""
repair_meta.py — Crossref-backed metadata repair for RAG parent stores.

Companion to `doctor --repair-meta` (deterministic local fixes). This tool
handles everything that needs verification against an authority:

  * missing authors      ← Crossref author list (fill-only, never overwrite)
  * missing/junk journal ← Crossref container-title (fill-only)
  * filename-form titles ← Crossref real title (only when current title is
                           empty or looks like the filename: 'RE123 - ...',
                           'PMID123456 - ...', bare DOI)
  * wrong year           ← Crossref issued year, replacing when the stored
                           year is implausible OR differs by more than
                           YEAR_TOLERANCE from the verified record
  * missing DOI          ← extracted from first chunks' content (folded
                           full-width forms included)

Repair sources, in order: DOI verify → bibliographic title search (guarded
by token overlap ≥ 0.6). Papers with nothing to fix are untouched. Papers
that can't be verified are skipped (reported, not guessed).

Usage:
    python3 scripts/metadata/repair_meta.py [--dry-run] [--limit N]
        [--kb <name>]        # library selector, same as doctor

After running: sync_chroma_meta.py pushes changes into chroma chunks.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent   # bib_rag/
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # metadata/ (meta_audit)

from kb_config import parse_kb_arg, get_config  # noqa: E402
from identifiers import extract_doi, normalize_doi, has_fullwidth  # noqa: E402

FILENAME_TITLE_RE = re.compile(r"^(RE\d+ - |\d{6,} - |10\.\d{4,5}/)")
YEAR_TOLERANCE = 1          # |stored - verified| > 1 → replace
TOKEN_OVERLAP_MIN = 0.6     # title-search guard against wrong-paper matches
META_FIELDS = ("title", "authors", "year", "journal", "doi", "pmid", "pmcid")
_META_CLEAN_DOI = re.compile(r"^10\.\d{4,5}/\S+$")

_JUNK_JOURNAL_WORDS = {"as of", "the", "article", "research", "journal", "in"}


def _junk_journal(j: str) -> bool:
    """Journal field holds PDF-text debris rather than a venue name."""
    j = (j or "").strip()
    if not j:
        return True
    if j.lower() in _JUNK_JOURNAL_WORDS or j.isdigit():
        return True
    # venue names are word-dense ("Journal of Land Use Science"); scraped
    # fragments are short or newline-riddled ("May Jun\\n\\nJul Aug")
    if "\n" in j or "\\n" in j:
        return True
    return len(j) < 4


def _year_ok(y: str) -> bool:
    import datetime
    if not re.match(r"^(19|20)\d{2}$", y or ""):
        return False
    return 1900 <= int(y) <= datetime.date.today().year + 1


def _token_overlap(a: str, b: str) -> float:
    ta = {w for w in re.findall(r"[a-z]+", (a or "").lower()) if len(w) > 3}
    tb = {w for w in re.findall(r"[a-z]+", (b or "").lower()) if len(w) > 3}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def _rewrite_parent(path: Path, chunks: list, updates: dict) -> None:
    for ch in chunks:
        m = ch.get("meta")
        if isinstance(m, dict):
            m.update(updates)
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=1)
    os.replace(tmp, str(path))


def needs_repair(meta: dict) -> bool:
    """True when any field this tool can fix is missing or damaged."""
    title = (meta.get("title") or "").strip()
    authors = (meta.get("authors") or "").strip()
    year = str(meta.get("year") or "").strip()
    journal = (meta.get("journal") or "").strip()
    doi = (meta.get("doi") or "").strip()
    return (not authors
            or not title
            or bool(FILENAME_TITLE_RE.match(title))
            or not _year_ok(year)
            or not journal
            or _junk_journal(journal)
            or (bool(doi) and not _META_CLEAN_DOI.match(
                unicodedata.normalize("NFKC", doi))))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="repair at most N papers (0 = all)")
    argv = parse_kb_arg() + (argv or [])
    args = ap.parse_args(argv)

    from meta_audit import CrossrefClient, DEFAULT_CROSSREF_MAILTO
    cx = CrossrefClient(DEFAULT_CROSSREF_MAILTO, enabled=True)

    cfg = get_config()
    store = Path(cfg["parent_store_dir"])
    print(f"library : {cfg['kb_name']} ({cfg['data_root']})")
    print(f"mode    : {'DRY-RUN' if args.dry_run else 'APPLY'}")

    targets = []
    for p in sorted(store.glob("*.json")):
        try:
            chunks = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not (chunks and isinstance(chunks[0], dict)):
            continue
        meta = dict(chunks[0].get("meta") or {})
        if needs_repair(meta):
            targets.append((p, chunks, meta))
    print(f"targets : {len(targets)} paper(s) need repair")
    if args.limit:
        targets = targets[:args.limit]

    repaired, unfixed = [], []
    t0 = time.time()
    for i, (p, chunks, meta) in enumerate(targets, 1):
        updates: dict = {}

        title = (meta.get("title") or "").strip()
        year = str(meta.get("year") or "").strip()
        journal = (meta.get("journal") or "").strip()
        doi_raw = (meta.get("doi") or "").strip()
        doi = doi_raw
        if doi:
            # existing DOI may be mangled (markdown tails, truncations,
            # full-width) — extract the clean form before verifying
            doi = extract_doi(doi) or doi
        else:
            head = "\n".join((ch.get("content") or "") for ch in chunks[:2])[:2000]
            doi = extract_doi(head)
        clean_doi = doi  # post-extraction form (differs from doi_raw if mangled)

        rec = cx.verify_doi(normalize_doi(doi) or doi) if doi else None
        via = "doi"
        # title-search only when the title itself looks like a real title —
        # PDF debris ("Received: ... Accepted: ...") produces garbage matches
        title_searchable = bool(title) and not re.match(
            r"^(received|accepted|available|article|open access|"
            r"editorial|volume|issue|www|http)", title, re.I)
        if not rec and title_searchable:
            hits = cx.search(title)
            for h in hits:
                jt = (h.get("journal") or "").lower()
                # skip recommendations/reviews platforms that mirror paper
                # titles but aren't the original venue (Faculty Opinions /
                # F1000, Europe PMC comments, Frontiers peer review …)
                if any(bad in jt for bad in (
                        "faculty opinions", "f1000", "peer review of",
                        "post-publication", "recommendation")):
                    continue
                if _token_overlap(h.get("title") or "", title) >= TOKEN_OVERLAP_MIN:
                    rec, via = h, "title-search"
                    break

        if not rec:
            # no Crossref record — but if the stored DOI was mangled, the
            # local extraction is still a strict improvement; write it back
            if clean_doi and normalize_doi(clean_doi) != normalize_doi(doi_raw):
                updates["doi"] = clean_doi
                if not args.dry_run:
                    _rewrite_parent(p, chunks, updates)
                repaired.append({"file": p.stem, "via": "local-extract",
                                 "changes": dict(updates)})
            unfixed.append((p.stem, "no crossref record"))
            continue

        rt = (rec.get("title") or "").strip()
        ra = (rec.get("authors") or "").strip()
        ry = str(rec.get("year") or "").strip()
        rj = (rec.get("journal") or "").strip()
        rd = (rec.get("doi") or "").strip()

        if (not title or FILENAME_TITLE_RE.match(title)) and rt:
            updates["title"] = rt
        if not (meta.get("authors") or "").strip() and ra:
            updates["authors"] = ra
        if ry and _year_ok(ry):
            if not _year_ok(year) or abs(int(ry) - int(year or 0)) > YEAR_TOLERANCE:
                updates["year"] = ry
        if (not journal or _junk_journal(journal)) and rj:
            updates["journal"] = rj
        if rd:
            # always write the verified canonical DOI — meta.doi may still be
            # the mangled form even when the cleaned query form matched it
            updates["doi"] = rd

        if not updates:
            unfixed.append((p.stem, "crossref agrees — nothing to change"))
            continue

        if not args.dry_run:
            _rewrite_parent(p, chunks, updates)
        repaired.append({"file": p.stem, "via": via, "changes": updates})
        if i % 50 == 0:
            print(f"  ...{i}/{len(targets)} ({time.time()-t0:.0f}s, "
                  f"{len(repaired)} repaired)", flush=True)

    n_agree = sum(1 for _, why in unfixed if "agrees" in why)
    n_unver = len(unfixed) - n_agree
    print(f"\nrepaired: {len(repaired)}, crossref-agrees: {n_agree}, "
          f"unverifiable: {n_unver}")
    if args.dry_run:
        for r in repaired[:6]:
            print(f"  DRY {r['file'][:58]} [{r['via']}]")
            for k, v in r["changes"].items():
                print(f"      {k} -> {str(v)[:70]}")
        return 0

    ledger = Path(cfg["data_dir"]) / "repair_meta_ledger.json"
    ledger.parent.mkdir(exist_ok=True)
    json.dump({"repaired": repaired,
               "unfixed": [{"file": f, "why": w} for f, w in unfixed]},
              open(ledger, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"ledger  : {ledger}")
    print("next    : python3 scripts/metadata/sync_chroma_meta.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())