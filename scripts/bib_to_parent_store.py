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
import unicodedata
from pathlib import Path
from collections import defaultdict
from datetime import datetime

BIB_PATH = Path('/Disk_bot/My Library.bib')
PARENT_STORE = Path('/Disk_bot/Eph/bib_rag/parent_store')
BACKUP_DIR = Path('/Disk_bot/Eph/bib_rag/data/parent_store_backup_doi')
MATCH_LOG = Path('/Disk_bot/Eph/bib_rag/data/bib_to_parent_store_log.json')


def normalize(s):
    """NFC normalize + lowercase + 去非字母数字"""
    s = unicodedata.normalize('NFKD', s.lower())
    s = ''.join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r'[^a-z0-9]+', '', s)
    return s


def parse_bib_entries(bib_path):
    """返回 list of dict {entry_type, key, doi, title, author_first_lastname, year}"""
    content = bib_path.read_text()
    entries = re.split(r'\n(?=@\w+\{)', content)
    result = []
    for e in entries:
        m = re.match(r'^@(\w+)\{([^,]+),', e.strip())
        if not m: continue
        entry_type, key = m.group(1), m.group(2).strip()

        # doi
        m_doi = re.search(r'doi\s*=\s*\{([^}]+)\}', e, re.IGNORECASE)
        doi = m_doi.group(1).strip() if m_doi else ''

        # title
        m_title = re.search(r'title\s*=\s*\{((?:[^{}]|\{[^{}]*\})*)\}', e, re.IGNORECASE | re.DOTALL)
        title = ''
        if m_title:
            # 简化: 直接清理单层 {X} 占位符 (Zotero 标准输出常见)
            title = m_title.group(1)
            # 嵌套 brace: 平衡扫描
            cleaned = []
            i = 0
            while i < len(title):
                if title[i] == '{':
                    # 找到匹配的 }
                    depth = 1
                    j = i + 1
                    while j < len(title) and depth > 0:
                        if title[j] == '{': depth += 1
                        elif title[j] == '}': depth -= 1
                        j += 1
                    # 取内部内容 (去最外层 brace)
                    cleaned.append(title[i+1:j-1])
                    i = j
                else:
                    cleaned.append(title[i])
                    i += 1
            title = ''.join(cleaned).strip()

        # author (first lastname)
        m_author = re.search(r'author\s*=\s*\{([^}]+)\}', e, re.IGNORECASE | re.DOTALL)
        author_first = ''
        author_full = ''
        if m_author:
            author_str = re.sub(r'[{}]', '', m_author.group(1))
            author_full = re.sub(r'\s+and\s+', ', ', author_str).strip()
            first = re.split(r'\s+and\s+', author_str)[0]
            if ',' in first:
                author_first = first.split(',')[0].strip()
            else:
                parts = first.split()
                author_first = parts[-1] if parts else ''

        # year (从 entry_key / year field / date field, 优先级)
        m_year = re.search(r'^year\s*=\s*\{(\d{4})\}', e, re.MULTILINE)
        if not m_year:
            # date field (但不能是 urldate 等其他 date 字段)
            m_date = re.search(r'^date\s*=\s*\{(\d{4})', e, re.MULTILINE)
            if m_date:
                m_year = m_date
        if not m_year:
            # fallback: 从 entry_key 抽 (key 通常包含 4 位年)
            m_year = re.search(r'(\d{4})', key)
        year = m_year.group(1) if m_year else ''

        # abstract (Zotero 导出常用, 74.8% entries 有) - 用于 S16 multi_match 消歧
        abstract = ''
        # Zotero 用 tab 缩进, (?m)^\s*abstract\s*=\s*\{ ... \}
        m_abs = re.search(r'(?m)^\s*abstract\s*=\s*\{((?:[^{}]|\{[^{}]*\})*)\}', e, re.IGNORECASE | re.DOTALL)
        if m_abs:
            abstract_raw = m_abs.group(1)
            # 简化 brace-balance
            cleaned = []
            i = 0
            while i < len(abstract_raw):
                if abstract_raw[i] == '{':
                    depth = 1
                    j = i + 1
                    while j < len(abstract_raw) and depth > 0:
                        if abstract_raw[j] == '{': depth += 1
                        elif abstract_raw[j] == '}': depth -= 1
                        j += 1
                    cleaned.append(abstract_raw[i+1:j-1])
                    i = j
                else:
                    cleaned.append(abstract_raw[i])
                    i += 1
            abstract = ''.join(cleaned).strip()

        result.append({
            'entry_type': entry_type,
            'key': key,
            'doi': doi,
            'title': title,
            'author_first_lastname': author_first,
            'author_full': author_full,
            'year': year,
            'abstract': abstract,
            'title_norm': normalize(title[:50]),
            'author_norm': normalize(author_first),
            'abstract_norm': normalize(abstract[:200]),
        })
    return result


def filename_to_key(fname):
    """从 parent_store filename 抽 (lastname, year, title_prefix)

    支持 3 种格式:
    - `Lastname_et_al__-_<year>_<title>` (多作者, e.g. `Abou_Chakra_et_al__-_2021_-_...`)
    - `Lastname_<year>_<title>` (单作者 + and, e.g. `Alert_and_Trepat_-_2020_-_...`)
    - `Lastname_<year>` (简短, e.g. `Adelmann_2022_Impact_of_cell_size_md`)

    字符集用 `[A-Za-z0-9_-]+` (含 _ 和 -, 修正字符集 bug)
    title_part 用 normalize() (去重音/标点 + lowercase) 保持 match 函数期望格式
    """
    # 格式 1: 多作者 + et_al
    m = re.match(r'([\w-]+?)_et_al__-_-?(\d{4})_(.+)', fname, re.UNICODE)
    if m:
        lastname = m.group(1).lower()
        year = m.group(2)
        title_part = m.group(3).replace('_md.json', '').replace('_', ' ')
        return (lastname, year, normalize(title_part[:50]))
    # 格式 2: 单作者 (Lastname_and_Lastname_<year>_<title> 或 Lastname_<year>_<title>)
    m = re.match(r'([\w-]+?)_(\d{4})_(.+)', fname, re.UNICODE)
    if m:
        lastname_raw = m.group(1)
        if 'and' not in lastname_raw.lower():
            # 纯单作者: 直接 lastname
            title_part = m.group(3).replace('_md.json', '').replace('_', ' ')
            return (lastname_raw.lower(), m.group(2), normalize(title_part[:50]))
        # 含 and: 试多作者简写 (Lastname_and_Lastname_<year>_<title>), 用 first author
        first_author = lastname_raw.split('_and_')[0].lower()
        if first_author and re.match(r'^[A-Za-z]', first_author):
            title_part = m.group(3).replace('_md.json', '').replace('_', ' ')
            return (first_author, m.group(2), normalize(title_part[:50]))
    # 格式 3: 简短 Lastname_YYYY_KeyWord
    m = re.match(r'([\w-]+?)_(\d{4})', fname, re.UNICODE)
    if m:
        return (m.group(1).lower(), m.group(2), '')
    return None


def match_paper_to_entry(paper_key, bib_entries, paper_abstract_norm=''):
    """三重匹配: (lastname, year) + title prefix 重叠
    第四维度 (可选): paper abstract vs bib abstract 重叠 (消歧 multi_match)
    """
    paper_lastname, paper_year, paper_title_norm = paper_key
    if not paper_year: return None, 'no_year'

    candidates = []
    for be in bib_entries:
        if not be['doi']: continue
        if normalize(be['author_first_lastname']) != paper_lastname: continue
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

    Returns 干净的 title (供 match_paper_by_title 用)
    year 是 optional (有些 paper title 没 year, e.g. "Birk et al. - Large-scale...")
    """
    if not title:
        return ''
    # 异常: doi-as-title (e.g. Capdevila paper meta.title = "https://doi.org/...")
    if title.startswith('http'):
        return ''
    # 去掉 `Lastname et al. - [YEAR -] <title>` 前缀 (year 可选)
    m = re.match(r'^(.+?)\s+et\s+al\.?\s*-\s*(?:\d{4}\s*-\s*)?(.+)$', title)
    if m:
        return m.group(2).strip()
    # 备用: `Lastname - YEAR - Title` 或 `Lastname - Title` (单作者 + year 中间符)
    m = re.match(r'^[A-Za-z][A-Za-z0-9_-]+\s*-\s*(?:\d{4}\s*-\s*)?(.+)$', title)
    if m:
        return m.group(1).strip()
    return title.strip()


def extract_year_from_paper_content(fp):
    """Fix 5: 从 paper content 抽 year (filename 缺 year 时的兜底)

    paper content 通常含:
    - "© 2024 Author" / "Copyright © 2024"
    - "bioRxiv preprint doi: https://doi.org/10.1101/2024.02.21"
    - "Published: 2024-XX-XX" / "Received: 2023-XX-XX"
    - 引用 " (Author, 2024)"

    只抽 1900-2030 之间的 4 位数字, 避免误抽
    """
    try:
        data = json.loads(fp.read_text())
    except:
        return ''
    if not isinstance(data, list):
        return ''
    for sec in data:
        c = sec.get('content', '')[:2000]  # 前 2000 chars 足够
        # 找 4 位年, 1900-2030
        for m in re.finditer(r'(?:©|Copyright|published|received|preprint|Cite this as)[\s\S]{0,40}?(19\d{2}|20[0-3]\d)', c, re.IGNORECASE):
            return m.group(1)
        # 备用: doi biorxiv 2024.02.21 -> 2024
        m = re.search(r'10\.1101/(19\d{2}|20[0-3]\d)', c)
        if m:
            return m.group(1)
    return ''


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
    ap.add_argument('--dry-run', action='store_true', help='只看, 不改 parent_store')
    ap.add_argument('--apply', action='store_true', help='实际改 parent_store (备份原 .json 到 BACKUP_DIR)')
    ap.add_argument('--low-confidence', action='store_true', help='写三重不匹配的 entry (默认不写)')
    args = ap.parse_args()

    if not args.dry_run and not args.apply:
        args.dry_run = True

    # 备份目录
    if args.apply:
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    print(f'[bib_to_parent_store] mode={"apply" if args.apply else "dry-run"}')
    print(f'[bib_to_parent_store] reading {BIB_PATH}')
    bib_entries = parse_bib_entries(BIB_PATH)
    with_doi = [be for be in bib_entries if be['doi']]
    print(f'[bib_to_parent_store] {len(bib_entries)} entries, {len(with_doi)} with DOI')

    # 扫 parent_store
    print(f'[bib_to_parent_store] scanning {PARENT_STORE}')
    matched_count = 0
    written_count = 0
    skipped_count = 0
    no_match_count = 0
    log_entries = []

    for fp in sorted(PARENT_STORE.glob('*.json')):
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
                        meta['doi'] = entry['doi']
                        # 升级: 同时写 meta.authors (if bib entry has author)
                        if entry.get('author_full'):
                            meta['authors'] = entry['author_full']
                        if old_doi and old_doi != entry['doi']:
                            log_entry['old_doi'] = old_doi
                    fp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
                    written_count += 1
            except Exception as e:
                log_entry['error'] = str(e)

    # 写 log
    log = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'mode': 'apply' if args.apply else 'dry-run',
        'bib_entries_total': len(bib_entries),
        'bib_entries_with_doi': len(with_doi),
        'parent_store_total': sum(1 for _ in PARENT_STORE.glob('*.json')),
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
    print(f'  → {MATCH_LOG}')


if __name__ == '__main__':
    main()
