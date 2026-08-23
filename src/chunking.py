#!/usr/bin/env python3
"""
Shared text-processing + hierarchical chunking logic for bib_rag.

Single source of truth for clean_text / truncate_at_references / extract_meta /
extract_sections / split_into_paragraphs / create_child_chunks /
create_parent_chunks / save_parent_store. Imported by both
build_hierarchical.py and index_single_paper.py so chunking stays consistent
across all indexing entry points.

Chunking config lives here so every caller uses the same sizes.
"""
import os
import re
import json
import hashlib
from pathlib import Path

# ============== Parent/Child Configuration ==============
CHILD_CHUNK_SIZE = 500
CHILD_CHUNK_OVERLAP = 100
MIN_PARENT_SIZE = 200
MAX_PARENT_SIZE = 4000

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
    if lines and 20 < len(lines[0]) < 300: meta['title'] = lines[0]
    name = re.sub(r'^[\+\^\s]+', '', Path(filename).stem)
    if len(name) > 10 and (not meta['title'] or len(name) > len(meta['title'])): meta['title'] = name
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

    return parents

def save_parent_store(parents, source, parent_store_dir):
    """Save parent chunks as JSON file in parent_store/."""
    os.makedirs(parent_store_dir, exist_ok=True)

    # One JSON file per source paper
    safe_name = re.sub(r'[^\w\-]', '_', source)[:100]
    filepath = os.path.join(parent_store_dir, f"{safe_name}.json")

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(parents, f, ensure_ascii=False, indent=2)

    return filepath
