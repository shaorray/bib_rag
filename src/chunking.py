#!/usr/bin/env python3
"""
Shared text-processing + hierarchical chunking logic for bib_rag.

Single source of truth for clean_text / truncate_at_references / extract_meta /
extract_sections / split_into_paragraphs / create_child_chunks /
create_parent_chunks / save_parent_store. Imported by both
build_hierarchical_gpu.py and index_single_paper.py so chunking stays consistent
across all indexing entry points.

Chunking config lives here so every caller uses the same sizes.
"""
import os
import re
import json
from typing import List


def atomic_json_dump(obj, path):
    """Write JSON atomically: temp file + os.replace, so a crash mid-write
    never leaves a truncated file (truncated checkpoints brick resume)."""
    path = str(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


import hashlib
from pathlib import Path

# ============== Parent/Child Configuration ==============
CHILD_CHUNK_SIZE = 500
CHILD_CHUNK_OVERLAP = 100
MIN_PARENT_SIZE = 200
MAX_PARENT_SIZE = 4000

# Figure/table caption atomic parents (LumiCite mechanism, see
# /Disk_bot/notes/citation_rag/06_LumiCite.md): captions carry key claims in
# biology papers but were previously merged into the surrounding section and
# hard to retrieve precisely. Caption paragraphs become their OWN parents
# with section='figure_caption' / 'table_caption' and a chunk_type tag so
# they can be filtered: where={"chunk_type": "figure_caption"}.
CAPTION_RE = re.compile(
    r"^(?:\*{0,2})(?:(?:Figure|Fig\.?|Table|Supplementary\s+(?:Figure|Table|S)\w*|Scheme)\s*(\d+[\w\.\-]*)"
    r"|((?:S|SF|ST)\d+))\s*[:.]\s*(.+)", re.I)
CAPTION_MAX_CHARS = 1500

# ============== Text Processing ==============

def clean_text(text):
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        if '==> image' in line.lower() or '==> picture' in line.lower(): continue
        if line.strip().startswith('©') and 'rights reserved' in line: continue
        if 'As a library, NLM provides access' in line: continue
        cleaned.append(line)
    return '\n'.join(cleaned)

def truncate_at_references(text):
    lines = text.split('\n'); cutoff = len(lines)
    for i, line in enumerate(lines):
        if re.match(r'^#*\s*(REFERENCES?|BIBLIOGRAPHY|ACKNOWLEDGMENTS?|SUPPLEMENTARY|APPENDIX|DATA AVAILABILITY|CONFLICT OF INTEREST|AUTHOR CONTRIBUTIONS?)\s*$', line, re.I):
            cutoff = i; break
    return '\n'.join(lines[:cutoff])

def extract_meta(text, filename):
    meta = {'title': '', 'authors': '', 'year': '', 'journal': '', 'doi': '', 'pmid': '', 'pmcid': ''}
    m = re.search(r'PMID:\s*(\d+)', text)
    if m: meta['pmid'] = m.group(1)
    m = re.search(r'PMCID:\s*(PMC\d+)', text)
    if m: meta['pmcid'] = m.group(1)
    m = re.search(r'(?:doi:|DOI:|https?://doi\.org/)(10\.\S+)', text)
    if m: meta['doi'] = m.group(1)
    m = re.search(r'\b(19\d{2}|20\d{2})\b', text)
    if m: meta['year'] = m.group(1)
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    # Prefer the first non-empty line as title (matches what front-matter injectors write),
    # but only if it's a plausible title (20-300 chars, doesn't look like a metadata field).
    if lines and 20 < len(lines[0]) < 300 and not lines[0].startswith(('Title:', 'Authors:', 'Year:', 'Journal:', 'doi:', 'PMID:', 'PMCID:', 'BibKey:', '#')):
        meta['title'] = lines[0]
    # Fall back to filename stem, but only if it's a reasonable title length
    # (avoid long filenames like 'Lastname_2025_Very long title that exceeds the 300 char cutoff.pdf').
    name = re.sub(r'^[\+\^\s]+', '', Path(filename).stem)
    name = re.sub(r'\.pdf$', '', name)  # strip .pdf from stem
    if not meta['title'] and 10 < len(name) <= 200:
        meta['title'] = name
    m = re.search(r'^([A-Z][A-Za-z\s\&\.]+)\s*\.?\s*\d{4}', text, re.M)
    if m: meta['journal'] = m.group(1).strip()
    return meta

# ============== Hierarchical Chunking ==============

def extract_sections(text):
    """Split text into sections by headers (ABSTRACT, INTRODUCTION, RESULTS, etc.)"""
    sections = {}; lines = text.split('\n')
    current = None; content = []
    patterns = {
        'abstract': r'^#*\s*(ABSTRACT|SUMMARY)\s*$',
        'introduction': r'^#*\s*(INTRODUCTION|BACKGROUND)\s*$',
        'methods': r'^#*\s*(METHODS?|MATERIALS?|EXPERIMENTAL)\s*',
        'results': r'^#*\s*(RESULTS?|FINDINGS?)\s*$',
        'discussion': r'^#*\s*DISCUSSION\s*$',
        'conclusion': r'^#*\s*CONCLUSIONS?\s*$',
    }
    for line in lines:
        matched = False
        for sec_name, pat in patterns.items():
            if re.match(pat, line, re.I):
                if current and content:
                    sections[current] = '\n'.join(content).strip()
                current = sec_name; content = []; matched = True; break
        if not matched and current:
            content.append(line)
    if current and content:
        sections[current] = '\n'.join(content).strip()
    return sections

def extract_captions(text: str) -> List[dict]:
    """Find figure/table caption paragraphs and return them as atomic
    pseudo-sections: [{'section': 'figure_caption'|'table_caption',
                       'label': 'Figure 3', 'text': caption_text}].

    A caption paragraph STARTS with the CAPTION_RE pattern; continuation
    paragraphs (indented/continuation lines until the next blank line or
    caption) are absorbed up to CAPTION_MAX_CHARS.
    """
    captions: List[dict] = []
    paragraphs = split_into_paragraphs(text)
    for para in paragraphs:
        first_line = para.split("\n", 1)[0].strip()
        m = CAPTION_RE.match(first_line)
        if not m:
            continue
        label = (m.group(1) or m.group(2) or "").strip()
        is_table = first_line.lower().startswith(("table", "supplementary table"))
        body = para
        if len(body) > CAPTION_MAX_CHARS:
            body = body[:CAPTION_MAX_CHARS]
        captions.append({
            "section": "table_caption" if is_table else "figure_caption",
            "label": label,
            "text": body,
        })
    return captions


def split_into_paragraphs(text):
    """Split text into paragraphs, preserving sentence boundaries."""
    return [p.strip() for p in text.split('\n\n') if p.strip()]

def create_child_chunks(parent_text, parent_id, source, section, chunk_size=CHILD_CHUNK_SIZE, overlap=CHILD_CHUNK_OVERLAP):
    """Split parent text into small overlapping child chunks."""
    paragraphs = split_into_paragraphs(parent_text)
    if not paragraphs:
        return []

    chunks = []
    current_text = ""

    for para in paragraphs:
        para_len = len(para)

        # If single paragraph exceeds chunk size, split by sentences
        if para_len > chunk_size * 1.5:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sent in sentences:
                if len(current_text) + len(sent) > chunk_size and current_text:
                    chunks.append({
                        'text': current_text.strip(),
                        'parent_id': parent_id,
                        'source': source,
                        'section': section,
                        'idx': len(chunks)
                    })
                    # Overlap: keep last ~overlap chars
                    words = current_text.split()
                    overlap_text = ' '.join(words[-overlap//5:]) if len(words) > overlap//5 else current_text[-overlap:]
                    current_text = overlap_text + ' ' + sent
                else:
                    current_text += ' ' + sent if current_text else sent
        else:
            if len(current_text) + para_len > chunk_size and current_text:
                chunks.append({
                    'text': current_text.strip(),
                    'parent_id': parent_id,
                    'source': source,
                    'section': section,
                    'idx': len(chunks)
                })
                # Overlap
                words = current_text.split()
                overlap_words = max(1, overlap // 5)
                overlap_text = ' '.join(words[-overlap_words:]) if len(words) > overlap_words else ""
                current_text = overlap_text + ('\n\n' if overlap_text else '') + para
            else:
                current_text += ('\n\n' if current_text else '') + para

    # Add remaining text
    if current_text.strip():
        chunks.append({
            'text': current_text.strip(),
            'parent_id': parent_id,
            'source': source,
            'section': section,
            'idx': len(chunks)
        })

    return chunks

def create_parent_chunks(text, source, meta):
    """
    Create parent chunks from text sections.
    Returns list of parent chunks with metadata.

    Figure/table captions (extract_captions) become atomic parents with
    section='figure_caption'/'table_caption' and chunk_type tag — overridable
    by the real section they live in is NOT attempted: captions are separate
    retrieval targets (LumiCite mechanism).
    """
    sections = extract_sections(text)

    if not sections:
        # No sections found — treat entire text as one parent
        sections = {'full_text': text}

    parents = []
    for sec_name, sec_text in sections.items():
        if not sec_text.strip():
            continue

        # Generate unique parent_id
        content_hash = hashlib.md5(f"{source}:{sec_name}:{sec_text[:200]}".encode()).hexdigest()[:16]
        parent_id = f"{source}#{sec_name}#{content_hash}"

        # Skip very small sections — they'll be merged into previous or next
        text_len = len(sec_text)
        if text_len < MIN_PARENT_SIZE:
            continue

        parent = {
            'parent_id': parent_id,
            'source': source,
            'section': sec_name,
            'chunk_type': 'section',
            'content': sec_text,
            'word_count': len(sec_text.split()),
            'char_count': text_len,
            'meta': {
                'title': meta.get('title', ''),
                'authors': meta.get('authors', ''),
                'year': meta.get('year', ''),
                'journal': meta.get('journal', ''),
                'doi': meta.get('doi', ''),
                'pmid': meta.get('pmid', ''),
                'pmcid': meta.get('pmcid', '')
            }
        }
        parents.append(parent)

    # Caption atomic parents — bypass MIN_PARENT_SIZE (captions are short but
    # high-value retrieval targets; LumiCite mechanism).
    for cap in extract_captions(text):
        sec_name = cap['section']
        content_hash = hashlib.md5(f"{source}:{sec_name}:{cap['label']}:{cap['text'][:200]}".encode()).hexdigest()[:16]
        parent_id = f"{source}#{sec_name}#{content_hash}"
        parents.append({
            'parent_id': parent_id,
            'source': source,
            'section': sec_name,
            'chunk_type': sec_name,  # 'figure_caption' | 'table_caption'
            'label': cap['label'],
            'content': cap['text'],
            'word_count': len(cap['text'].split()),
            'char_count': len(cap['text']),
            'meta': {
                'title': meta.get('title', ''),
                'authors': meta.get('authors', ''),
                'year': meta.get('year', ''),
                'journal': meta.get('journal', ''),
                'doi': meta.get('doi', ''),
                'pmid': meta.get('pmid', ''),
                'pmcid': meta.get('pmcid', '')
            }
        })

    return parents

def save_parent_store(parents, source, parent_store_dir):
    """Save parent chunks as JSON file in parent_store/."""
    os.makedirs(parent_store_dir, exist_ok=True)

    # One JSON file per source paper
    safe_name = re.sub(r'[^\w\-]', '_', source)[:100]
    filepath = os.path.join(parent_store_dir, f"{safe_name}.json")

    atomic_json_dump(parents, filepath)

    return filepath
