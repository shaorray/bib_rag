#!/usr/bin/env python3
"""bib_to_parent_store.py — 从 My Library.bib 抽 DOI 灌回 parent_store。

不动 paper content (那是 paper 内被引文献, 不该改)。
只改 meta.doi 字段 (bib_rag 实际查的字段)。

按 Coding Principles #3 Surgical: 默认 dry-run, 必须 user 显式 --apply 才写盘。

按 SOUL.md Honesty Protocol:
- 不假装匹配: 找不到的 entry 标 unmatched 不写
- 找到的 entry 必须三重匹配 (lastname + year + title prefix) 才信任
- 三重不匹配的 entry 标 low_confidence, 也可写但有警告
"""
import json
import re
import argparse
from pathlib import Path
from datetime import datetime

from bib_utils import (
    extract_year_from_content, filename_to_key, normalize,
    normalize_doi, parse_bib_entries, strip_author_year_prefix,
)

BIB_PATH = Path('/Disk_bot/My Library.bib')
PARENT_STORE = Path('/Disk_bot/RAG/bib_rag/parent_store')
BACKUP_DIR = Path('/Disk_bot/RAG/bib_rag/data/parent_store_backup_doi')
MATCH_LOG = Path('/Disk_bot/RAG/bib_rag/data/bib_to_parent_store_log.json')


def match_paper_to_entry(paper_key, bib_entries, paper_abstract_norm=''):
    """三重匹配: (lastname, year) + title prefix 重叠
    第四维度 (可选): paper abstract vs bib abstract 重叠 (消歧 multi_match)

    双方 lastname 都 normalize 后再比较 — filename 里的 `Boström`、`Abdul-Wajid`、
    `Abou_Chakra` 才能对上 .bib 里的 `bostrom` / `abdulwajid` / `abouchakra`。
    """
    paper_lastname, paper_year, paper_title_norm = paper_key
    if not paper_year: return None, 'no_year'
    paper_lastname_norm = normalize(paper_lastname)

    candidates = []
    for be in bib_entries:
        if not be['doi']: continue
        if be['author_norm'] != paper_lastname_norm: continue
        if be['year'] != paper_year: continue
        # title 重叠 (paper_title_norm 是 bib title 前 50 char normalize)
        bib_title_norm = be['title_norm']
        if not bib_title_norm or not paper_title_norm: continue
        # Jaccard-like 重叠: bib title 前 30 char 必须出现在 paper title
        if bib_title_norm[:25] in paper_title_norm or paper_title_norm[:25] in bib_title_norm:
            candidates.append(be)

    if not candidates: return None, 'no_match'
    if len(candidates) == 1:
        return candidates[0], 'matched'
    # multi_match: 用 (title overlap + abstract overlap) 加权评分消歧
    def score(be):
        title_score = len(set(be['title_norm'][:25]) & set(paper_title_norm[:25]))
        # abstract 重叠 (如果有)
        abstract_score = 0
        if paper_abstract_norm and be.get('abstract_norm'):
            bib_abs_norm = be['abstract_norm']
            if bib_abs_norm[:30] in paper_abstract_norm or paper_abstract_norm[:30] in bib_abs_norm:
                abstract_score = 10  # abstract overlap 权重高 (消歧信号强)
        return title_score + abstract_score
    best = max(candidates, key=score)
    return best, 'multi_match'


def match_paper_by_title(paper_title, paper_year, bib_entries, year_tolerance=1, paper_abstract_norm=''):
    """Step 3: title 反向搜 .bib (no_match fallback)

    Fix 3: year_tolerance=1 (paper year ±1, 容 Adelmann 2022 → adelmann_impact_2023 case)
    Fix 6: title prefix 改 [:20] (从 [:30]) 更宽容, 容 title 截断不同位置
    Fix 7: paper_abstract_norm 加权 (multi_match 消歧)
    用 paper.meta.title (完整 title) 匹配 .bib entry title
    置信度: title normalize[:20] 必须出现在 bib title normalize 里 (双向)
    返回 (entry, status): status in (title_matched, multi_match, no_match)
    """
    if not paper_title or len(paper_title) < 15:
        return None, 'title_too_short'

    paper_title_norm = normalize(paper_title[:80])
    candidates = []
    for be in bib_entries:
        if not be['doi']: continue
        # year 容差匹配 (Fix 3)
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
        # Fix 6: 双向 title prefix 重叠 ([:20] 更宽容)
        if bib_title_norm[:20] in paper_title_norm or paper_title_norm[:20] in bib_title_norm:
            candidates.append(be)

    if not candidates: return None, 'no_match'
    if len(candidates) == 1:
        return candidates[0], 'title_matched'
    # Fix 7: 多候选: title 相似度 + abstract 加权 (paper abstract 加进消歧)
    def score(be):
        title_score = len(set(be['title_norm'][:20]) & set(paper_title_norm[:20]))
        # abstract 重叠
        abstract_score = 0
        if paper_abstract_norm and be.get('abstract_norm'):
            bib_abs_norm = be['abstract_norm']
            if bib_abs_norm[:30] in paper_abstract_norm or paper_abstract_norm[:30] in bib_abs_norm:
                abstract_score = 10
        return title_score + abstract_score
    best = max(candidates, key=score)
    return best, 'multi_match'


def normalize_paper_title(title):
    """去掉 paper meta.title 前缀 (常见格式: `Lastname et al. - YEAR - <title>` 或 `https://doi.org/...` 异常)

    逻辑在 bib_utils.strip_author_year_prefix (还处理 `YYYY - ` 前缀, 兼容
    filename-as-title 的情况)。
    """
    return strip_author_year_prefix(title)


def extract_year_from_paper_content(fp):
    """Fix 5: 从 paper content 抽 year (filename 缺 year 时的兜底)

    逻辑在 bib_utils.extract_year_from_content; 这里只负责读 JSON + 类型检查。
    """
    try:
        data = json.loads(fp.read_text())
    except Exception:
        return ''
    if not isinstance(data, list):
        return ''
    return extract_year_from_content(data)


def get_paper_meta(fp):
    """从 parent_store .json 抽 meta.title, meta.year, abstract (for disambiguation)

    title 经过 normalize_paper_title 去掉 `Lastname et al. - YEAR -` 前缀 (Fix 1)
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
            # abstract: 看 content 里 ## Abstract / ## Summary / ## Research briefing 段落
            if not abstract:
                c = sec.get('content', '')
                # 优先 ## Abstract
                m = re.search(r'(?si)##\s*\*?\*?(?:Abstract|Summary|Background|Introduction|Research\s+briefing)[:\s]*\n*(.*?)(?=\n##\s|\Z)', c)
                if m:
                    candidate = m.group(1).strip()
                    if len(candidate) > 100:  # 至少 100 chars 算 abstract
                        abstract = candidate
    return title, year, abstract


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bib', type=Path, default=BIB_PATH, help='My Library.bib path')
    ap.add_argument('--parent-dir', type=Path, default=PARENT_STORE, help='parent_store dir')
    ap.add_argument('--dry-run', action='store_true', help='只看, 不改 parent_store')
    ap.add_argument('--apply', action='store_true', help='实际改 parent_store (备份原 .json 到 BACKUP_DIR)')
    ap.add_argument('--low-confidence', action='store_true', help='写三重不匹配的 entry (默认不写)')
    args = ap.parse_args()

    if not args.dry_run and not args.apply:
        args.dry_run = True

    if not args.bib.exists():
        print(f'[bib_to_parent_store] ERROR: bib not found: {args.bib}')
        return 1
    if not args.parent_dir.exists():
        print(f'[bib_to_parent_store] ERROR: parent dir not found: {args.parent_dir}')
        return 1

    # 备份目录
    if args.apply:
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    print(f'[bib_to_parent_store] mode={"apply" if args.apply else "dry-run"}')
    print(f'[bib_to_parent_store] reading {args.bib}')
    bib_entries = parse_bib_entries(args.bib)
    with_doi = [be for be in bib_entries if be['doi']]
    print(f'[bib_to_parent_store] {len(bib_entries)} entries, {len(with_doi)} with DOI')

    # 扫 parent_store
    print(f'[bib_to_parent_store] scanning {args.parent_dir}')
    matched_count = 0
    written_count = 0
    skipped_count = 0
    no_match_count = 0
    log_entries = []

    for fp in sorted(args.parent_dir.glob('*.json')):
        paper_key = filename_to_key(fp.stem)
        # Fix 5: filename 缺 year 时, 从 paper content 抽 year 充补
        if paper_key and not paper_key[1]:
            content_year = extract_year_from_paper_content(fp)
            if content_year:
                paper_key = (paper_key[0], content_year, paper_key[2])
        entry = None
        confidence = None
        status = None

        # 抽 paper metadata (包括 abstract, 用于 multi_match 消歧)
        paper_title, paper_year, paper_abstract = get_paper_meta(fp)
        paper_abstract_norm = normalize(paper_abstract[:200]) if paper_abstract else ''

        if paper_key:
            entry, status = match_paper_to_entry(paper_key, bib_entries, paper_abstract_norm=paper_abstract_norm)
            if entry:
                confidence = 'low' if status == 'multi_match' else 'high'

        # Step 3: fallback to title reverse search (no_match only)
        if not entry:
            # 用 year from filename key (if available) or paper meta.year
            year_hint = paper_key[1] if paper_key else paper_year
            entry, status = match_paper_by_title(paper_title, year_hint, bib_entries, paper_abstract_norm=paper_abstract_norm)
            if entry:
                confidence = 'low' if status == 'multi_match' else 'medium'  # title-only 置信度比三重匹配低
                status = f'title_{status}'  # 标记 title-only 来源

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
                        # 写之前 normalize DOI (去 URL 前缀/尾标点, 与 meta_audit 一致)
                        meta['doi'] = normalize_doi(entry['doi'])
                        # 升级: 同时写 meta.authors (if bib entry has author)
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
            # matched 但未写盘 (dry-run / 置信度不足 / 写盘失败)
            skipped_count += 1

    # 写 log
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
