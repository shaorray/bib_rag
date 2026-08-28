#!/usr/bin/env python3
"""bib_to_parent_store.py — pull DOIs from My Library.bib back into parent_store.

Never touches paper content (that is literature cited *inside* papers — not ours
to change). Only rewrites the meta.doi field (the one bib_rag actually queries).

Follows Coding Principles #3 Surgical: dry-run by default; only an explicit
--apply writes to disk.

Follows the SOUL.md Honesty Protocol:
- never fake a match: entries with no counterpart are marked unmatched, nothing written
- a matched entry must pass triple matching (lastname + year + title prefix) to be trusted
- entries failing the triple check are marked low_confidence: written with a warning, or skipped
"""
import json
import re
import argparse
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ (bib_utils, zotero_access)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import os

from kb_config import get_config
from bib_utils import (
    extract_year_from_content, filename_to_key, normalize,
    normalize_doi, parse_bib_entries, strip_author_year_prefix,
)

_CFG = get_config()

from library_config import get_setting as _lib_setting
BIB_PATH = Path(_lib_setting(_CFG["data_root"], "bib_path", "My Library.bib"))
if not BIB_PATH.exists():
    print(f"[warn] BibTeX file not found: {BIB_PATH} — set BIB_RAG_BIB_PATH env "
          f"or pass --bib <path> (needed only for metadata-fill workflows)")
PARENT_STORE = Path(_CFG["parent_store_dir"])
BACKUP_DIR = Path(_CFG["data_dir"]) / "parent_store_backup_doi"
MATCH_LOG = Path(_CFG["data_dir"]) / "bib_to_parent_store_log.json"


def match_paper_to_entry(paper_key, bib_entries, paper_abstract_norm=''):
    """Triple match: (lastname, year) + title prefix overlap.
    Optional fourth dimension: paper abstract vs bib abstract overlap (disambiguates multi_match).

    Both lastnames are normalized before comparison — so `Boström`, `Abdul-Wajid`,
    `Abou_Chakra` in filenames match `bostrom` / `abdulwajid` / `abouchakra` in the .bib.
    """
    paper_lastname, paper_year, paper_title_norm = paper_key
    if not paper_year: return None, 'no_year'
    paper_lastname_norm = normalize(paper_lastname)

    candidates = []
    for be in bib_entries:
        if not be['doi']: continue
        if be['author_norm'] != paper_lastname_norm: continue
        if be['year'] != paper_year: continue
        # title overlap (paper_title_norm is the bib title's first 50 chars, normalized)
        bib_title_norm = be['title_norm']
        if not bib_title_norm or not paper_title_norm: continue
        # Jaccard-like overlap: the bib title's first 30 chars must appear in the paper title
        if bib_title_norm[:25] in paper_title_norm or paper_title_norm[:25] in bib_title_norm:
            candidates.append(be)

    if not candidates: return None, 'no_match'
    if len(candidates) == 1:
        return candidates[0], 'matched'
    # multi_match: disambiguate with weighted (title overlap + abstract overlap) scoring
    def score(be):
        title_score = len(set(be['title_norm'][:25]) & set(paper_title_norm[:25]))
        # abstract overlap (when available)
        abstract_score = 0
        if paper_abstract_norm and be.get('abstract_norm'):
            bib_abs_norm = be['abstract_norm']
            if bib_abs_norm[:30] in paper_abstract_norm or paper_abstract_norm[:30] in bib_abs_norm:
                abstract_score = 10  # abstract overlap carries high weight (strong disambiguation signal)
        return title_score + abstract_score
    best = max(candidates, key=score)
    return best, 'multi_match'


def match_paper_by_title(paper_title, paper_year, bib_entries, year_tolerance=1, paper_abstract_norm=''):
    """Step 3: reverse-search the .bib by title (no_match fallback).

    Fix 3: year_tolerance=1 (paper year ±1, covers Adelmann 2022 → adelmann_impact_2023)
    Fix 6: title prefix widened to [:20] (from [:30]) to tolerate truncation at different offsets
    Fix 7: paper_abstract_norm weighting (multi_match disambiguation)
    Matches paper.meta.title (full title) against the .bib entry title.
    Confidence: normalized title[:20] must appear in the normalized bib title (both directions).
    Returns (entry, status): status in (title_matched, multi_match, no_match)
    """
    if not paper_title or len(paper_title) < 15:
        return None, 'title_too_short'

    paper_title_norm = normalize(paper_title[:80])
    candidates = []
    for be in bib_entries:
        if not be['doi']: continue
        # year matching with tolerance (Fix 3)
        if paper_year and be['year']:
            try:
                year_diff = abs(int(paper_year) - int(be['year']))
                if year_diff > year_tolerance:
                    continue
            except ValueError:
                if paper_year != be['year']:
                    continue
        bib_title_norm = be['title_norm']
        if not bib_title_norm or not paper_title_norm: continue
        # Fix 6: bidirectional title-prefix overlap ([:20], more tolerant)
        if bib_title_norm[:20] in paper_title_norm or paper_title_norm[:20] in bib_title_norm:
            candidates.append(be)

    if not candidates: return None, 'no_match'
    if len(candidates) == 1:
        return candidates[0], 'title_matched'
    # Fix 7: multiple candidates — title similarity + abstract weighting
    def score(be):
        title_score = len(set(be['title_norm'][:20]) & set(paper_title_norm[:20]))
        # abstract overlap
        abstract_score = 0
        if paper_abstract_norm and be.get('abstract_norm'):
            bib_abs_norm = be['abstract_norm']
            if bib_abs_norm[:30] in paper_abstract_norm or paper_abstract_norm[:30] in bib_abs_norm:
                abstract_score = 10
        return title_score + abstract_score
    best = max(candidates, key=score)
    return best, 'multi_match'


def normalize_paper_title(title):
    """Strip common meta.title prefixes (`Lastname et al. - YEAR - <title>` or
    `https://doi.org/...` anomalies).

    Logic lives in bib_utils.strip_author_year_prefix (also handles `YYYY - `
    prefixes and filename-as-title cases).
    """
    return strip_author_year_prefix(title)


def extract_year_from_paper_content(fp):
    """Fix 5: pull the year from paper content (fallback when the filename lacks one).

    Logic lives in bib_utils.extract_year_from_content; this only reads JSON +
    type-checks.
    """
    try:
        data = json.loads(fp.read_text())
    except Exception:
        return ''
    if not isinstance(data, list):
        return ''
    return extract_year_from_content(data)


def get_paper_meta(fp):
    """Extract meta.title, meta.year and abstract from a parent_store .json (for disambiguation).

    The title passes through normalize_paper_title to strip the
    `Lastname et al. - YEAR -` prefix (Fix 1).
    """
    try:
        data = json.loads(fp.read_text())
    except:
        return '', '', ''
    title = ''
    year = ''
    abstract = ''
    if isinstance(data, list):
        for sec in data:
            meta = sec.get('meta', {})
            if meta.get('title') and not title:
                title = normalize_paper_title(meta['title'])
            if meta.get('year') and not year:
                year = meta['year']
            # abstract: look for a ## Abstract / ## Summary / ## Research briefing section in content
            if not abstract:
                c = sec.get('content', '')
                # prefer ## Abstract
                m = re.search(r'(?si)##\s*\*?\*?(?:Abstract|Summary|Background|Introduction|Research\s+briefing)[:\s]*\n*(.*?)(?=\n##\s|\Z)', c)
                if m:
                    candidate = m.group(1).strip()
                    if len(candidate) > 100:  # need at least 100 chars to count as abstract
                        abstract = candidate
    return title, year, abstract


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bib', type=Path, default=BIB_PATH, help='My Library.bib path')
    ap.add_argument('--parent-dir', type=Path, default=PARENT_STORE, help='parent_store dir')
    ap.add_argument('--dry-run', action='store_true', help='inspect only; never modify parent_store')
    ap.add_argument('--apply', action='store_true', help='modify parent_store (backs up originals to BACKUP_DIR)')
    ap.add_argument('--low-confidence', action='store_true', help='also write entries failing the triple match (skipped by default)')
    args = ap.parse_args()

    if not args.dry_run and not args.apply:
        args.dry_run = True

    if not args.bib.exists():
        print(f'[bib_to_parent_store] ERROR: bib not found: {args.bib}')
        return 1
    if not args.parent_dir.exists():
        print(f'[bib_to_parent_store] ERROR: parent dir not found: {args.parent_dir}')
        return 1

    # backup dir
    if args.apply:
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    print(f'[bib_to_parent_store] mode={"apply" if args.apply else "dry-run"}')
    print(f'[bib_to_parent_store] reading {args.bib}')
    bib_entries = parse_bib_entries(args.bib)
    with_doi = [be for be in bib_entries if be['doi']]
    print(f'[bib_to_parent_store] {len(bib_entries)} entries, {len(with_doi)} with DOI')

    # scan parent_store
    print(f'[bib_to_parent_store] scanning {args.parent_dir}')
    matched_count = 0
    written_count = 0
    skipped_count = 0
    no_match_count = 0
    log_entries = []

    for fp in sorted(args.parent_dir.glob('*.json')):
        paper_key = filename_to_key(fp.stem)
        # Fix 5: when the filename lacks a year, pull one from paper content
        if paper_key and not paper_key[1]:
            content_year = extract_year_from_paper_content(fp)
            if content_year:
                paper_key = (paper_key[0], content_year, paper_key[2])
        entry = None
        confidence = None
        status = None

        # extract paper metadata (abstract included, for multi_match disambiguation)
        paper_title, paper_year, paper_abstract = get_paper_meta(fp)
        paper_abstract_norm = normalize(paper_abstract[:200]) if paper_abstract else ''

        if paper_key:
            entry, status = match_paper_to_entry(paper_key, bib_entries, paper_abstract_norm=paper_abstract_norm)
            if entry:
                confidence = 'low' if status == 'multi_match' else 'high'

        # Step 3: fallback to title reverse search (no_match only)
        if not entry:
            # use the year from the filename key (if available) or paper meta.year
            year_hint = paper_key[1] if paper_key else paper_year
            entry, status = match_paper_by_title(paper_title, year_hint, bib_entries, paper_abstract_norm=paper_abstract_norm)
            if entry:
                confidence = 'low' if status == 'multi_match' else 'medium'  # title-only confidence is lower than the triple match
                status = f'title_{status}'  # mark the title-only provenance

        if not entry:
            no_match_count += 1
            continue

        log_entry = {
            'file': fp.name,
            'bib_key': entry['key'],
            'paper_lastname': paper_key[0] if paper_key else '',
            'paper_year': paper_key[1] if paper_key else '',
            'doi_from_bib': entry['doi'],
            'author_from_bib': entry.get('author_full', ''),
            'confidence': confidence,
            'status': status,
            'bib_title': entry['title'][:80],
            'bib_author': entry['author_first_lastname'],
            'bib_year': entry['year'],
        }
        log_entries.append(log_entry)
        matched_count += 1

        # Write (apply)
        written = False
        if args.apply and (confidence in ('high', 'medium') or args.low_confidence):
            backup_path = BACKUP_DIR / fp.name
            if not backup_path.exists():
                backup_path.write_text(fp.read_text())

            try:
                data = json.loads(fp.read_text())
                if isinstance(data, list):
                    for sec in data:
                        meta = sec.get('meta', {})
                        old_doi = meta.get('doi', '')
                        # normalize the DOI before writing (strip URL prefix / trailing punctuation, consistent with meta_audit)
                        meta['doi'] = normalize_doi(entry['doi'])
                        # upgrade: also write meta.authors (when the bib entry has authors)
                        if entry.get('author_full'):
                            meta['authors'] = entry['author_full']
                        if old_doi and normalize_doi(old_doi) != normalize_doi(entry['doi']):
                            log_entry['old_doi'] = old_doi
                    fp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
                    written_count += 1
                    written = True
            except Exception as e:
                log_entry['error'] = str(e)
        if not written:
            # matched but not written (dry-run / insufficient confidence / write failure)
            skipped_count += 1

    # write log
    log = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'mode': 'apply' if args.apply else 'dry-run',
        'bib_entries_total': len(bib_entries),
        'bib_entries_with_doi': len(with_doi),
        'parent_store_total': sum(1 for _ in args.parent_dir.glob('*.json')),
        'matched_count': matched_count,
        'written_count': written_count,
        'skipped_count': skipped_count,
        'no_match_count': no_match_count,
        'high_confidence_count': sum(1 for e in log_entries if e['confidence'] == 'high'),
        'low_confidence_count': sum(1 for e in log_entries if e['confidence'] == 'low'),
        'log_entries': log_entries,
    }
    MATCH_LOG.write_text(json.dumps(log, ensure_ascii=False, indent=2))

    print(f'\n[bib_to_parent_store] DONE')
    print(f'  parent_store files: {log["parent_store_total"]}')
    print(f'  matched (any): {matched_count}')
    print(f'  high confidence: {log["high_confidence_count"]}')
    print(f'  low confidence (multi_match): {log["low_confidence_count"]}')
    print(f'  no match: {no_match_count}')
    print(f'  written: {written_count}')
    print(f'  skipped (matched, not written): {skipped_count}')
    print(f'  → {MATCH_LOG}')


if __name__ == '__main__':
    main()
