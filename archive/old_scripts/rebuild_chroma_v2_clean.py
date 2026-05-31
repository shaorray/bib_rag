#!/usr/bin/env python3
"""Clean single-pass rebuild of chroma_db_v2 with proper semantic embeddings."""

import sys, os, re, pickle, hashlib
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from sentence_transformers import SentenceTransformer

# ── Config ──────────────────────────────────────────────────────────────────
PAPERS_DIR  = "/Disk_2/claw_working_dir/Ephrin_papers/new_pub/Eph-ephrin/top500_md_v2"
DB_PATH     = "/Disk_2/claw_working_dir/ephrin_agentic_rag/chroma_db_v2"
MODEL_NAME  = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_BS    = 128
CHUNK_SIZE  = 800
OVERLAP     = 200
MIN_CHUNK   = 50

SKIP_SECTIONS = {
    'references', 'acknowledgments', 'acknowledgements', 'acknowledgment',
    'acknowledgement', 'figure legends', 'tables', 'table',
    'supplementary material', 'supplementary materials',
    'supplementary information', 'supplementary data',
    'competing interests', 'conflict of interest',
    'consent for publication', 'peer review', 'footnotes',
    'abbreviations', 'keywords', 'graphical abstract',
    'author contributions', 'author information', 'funding',
    'ethics approval', 'data availability', 'permissions',
    "publisher's note", 'article notes',
    'copyright and license information',
    'natureportfolio', 'springernature',
    'supplementary information 1',
    'supplementary tables', 'additional files',
    'peer review information', "author's accepted manuscript",
}

# ── Embedder ─────────────────────────────────────────────────────────────────
class Embedder:
    def __init__(self, model_name=MODEL_NAME):
        print(f"Loading: {model_name}")
        os.environ['HF_HUB_OFFLINE'] = '1'
        self.model = SentenceTransformer(model_name, cache_folder='/tmp/st_cache',
                                          local_files_only=True, trust_remote_code=True)
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"  -> dim={self.dim}, dev={self.model.device}")

    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                                   show_progress_bar=False, batch_size=EMBED_BS).astype(np.float32)


# ── Parser ───────────────────────────────────────────────────────────────────
def extract_metadata(text):
    meta = {}
    def grab(pat, cast=str):
        m = re.search(pat, text, re.MULTILINE)
        return cast(m.group(1)) if m else None
    meta['rank']          = grab(r'^rank:\s*(\d+)', int)
    meta['pmid']          = grab(r'^PMID:\s*(\d+)')
    meta['priority']      = grab(r'^Priority:\s*([\d.]+)', float)
    meta['impact_factor'] = grab(r'^Impact Factor:\s*([\d.]+)', float)
    meta['citations']     = grab(r'^Citations:\s*(\d+)', int)
    meta['year']          = grab(r'^Year:\s*(\d{4})', int)
    j = grab(r'^Journal:\s*(.+?)\s*$')
    meta['journal'] = j.strip() if j else None
    t = grab(r'^Tier:\s*(\S+)')
    meta['tier'] = t.strip() if t else None
    a = grab(r'^Area:\s*(.+?)\s*$')
    meta['area'] = a.strip() if a else None
    return meta


def split_sections(text):
    text = re.sub(r'^---\n.*?\n---\n', '', text, flags=re.DOTALL, count=1)
    lines = text.split('\n')
    sections = defaultdict(list)
    current = 'body'
    for line in lines:
        s = line.strip()
        if '==> picture' in line or '==> table' in line:
            continue
        if 'As a library, NLM provides access to scientific literature' in s:
            continue
        if s == 'Learn more: PMC Disclaimer | PMC Copyright Notice':
            continue
        m = re.match(r'^(#{2,4})\s+(.+)$', s)
        if m:
            title = m.group(2).strip()
            if re.match(r'^(Figure|Table)\s+\d', title, re.I):
                current = f'figtable_{title.lower()}'
                continue
            current = title.lower()
            sections[current]
            continue
        sections[current].append(line)
    return {k: '\n'.join(v).strip() for k, v in sections.items() if v}


def should_skip(name):
    return any(skip in name.lower() for skip in SKIP_SECTIONS)


def make_chunks(filename, sections, meta):
    chunks = []
    base = {
        'pmid': meta.get('pmid', ''),
        'year': meta.get('year', ''),
        'journal': meta.get('journal', ''),
        'if': meta.get('impact_factor', ''),
        'citations': meta.get('citations', ''),
        'tier': meta.get('tier', ''),
        'rank': meta.get('rank', ''),
        'area': meta.get('area', ''),
        'filename': filename,
    }
    mapping = {
        'abstract': 'abstract', 'summary': 'abstract',
        'introduction': 'introduction', 'background': 'introduction',
        'results': 'results', 'discussion': 'discussion',
        'methods': 'methods', 'materials and methods': 'methods',
        'experimental procedures': 'methods',
        'conclusion': 'conclusion', 'conclusions': 'conclusion',
    }

    for sec_name, sec_text in sections.items():
        if should_skip(sec_name):
            continue
        words = sec_text.split()
        if len(words) < MIN_CHUNK:
            continue
        base_sec = mapping.get(sec_name, 'other')
        step = CHUNK_SIZE - OVERLAP
        idx_id = 0
        for i in range(0, len(words), step):
            sub = words[i:i + CHUNK_SIZE]
            if len(sub) < MIN_CHUNK and i > 0:
                continue
            body = ' '.join(sub)
            prefix = (f"PMID:{base['pmid']} | Year:{base['year']} | "
                      f"Journal:{base['journal']} | IF:{base['if']} | Citations:{base['citations']}\n")
            full = prefix + body
            cm = dict(base)
            cm['section'] = base_sec if i == 0 else f"{base_sec}_cont"
            # unique ID per chunk: hash of filename+section+content first 80 chars
            cid = hashlib.md5(f"{filename}_{sec_name}_{idx_id}_{body[:80]}".encode()).hexdigest()[:16]
            chunks.append({'text': full, 'meta': cm, 'id': cid})
            idx_id += 1
    return chunks


# ── Main build (single-pass) ─────────────────────────────────────────────────
def main():
    import shutil
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
    os.makedirs(DB_PATH, exist_ok=True)

    files = sorted(Path(PAPERS_DIR).glob('*.md'))
    print(f"Found {len(files)} markdown files")

    embedder = Embedder()

    all_texts  = []
    all_metas  = []
    all_ids    = []
    success = failed = 0

    for fp in tqdm(files, desc="Parsing"):
        try:
            text = fp.read_text(encoding='utf-8', errors='ignore')
            meta = extract_metadata(text)
            sections = split_sections(text)
            chunks = make_chunks(fp.name, sections, meta)
            for c in chunks:
                all_texts.append(c['text'])
                all_metas.append(c['meta'])
                all_ids.append(c['id'])
            success += 1
        except Exception as e:
            print(f"  Error on {fp.name}: {e}")
            failed += 1

    print(f"\nParsed: {success} files | Chunks: {len(all_texts)}")
    print(f"Embedding {len(all_texts)} chunks ...")

    # Single batch embed (with sub-batch handling internally)
    embeddings = embedder.embed(all_texts)

    # Ensure IDs are unique (dedupe by content hash if needed)
    seen = set()
    unique_texts, unique_metas, unique_ids, unique_embs = [], [], [], []
    for t, m, i, e in zip(all_texts, all_metas, all_ids, embeddings):
        if i in seen:
            continue
        seen.add(i)
        unique_texts.append(t)
        unique_metas.append(m)
        unique_ids.append(i)
        unique_embs.append(e)

    print(f"Unique chunks after dedup: {len(unique_texts)}")

    data_file = os.path.join(DB_PATH, 'ephrin_papers_v2.pkl')
    with open(data_file, 'wb') as f:
        pickle.dump({
            'documents': unique_texts,
            'metadatas': unique_metas,
            'ids': unique_ids,
            'embeddings': [e.tolist() for e in unique_embs],  # store as list of lists
        }, f)

    print(f"\n{'='*60}")
    print(f"✅ Rebuild complete: {data_file}")
    print(f"   Files: {len(files)} | Success: {success} | Failed: {failed}")
    print(f"   Chunks: {len(unique_texts)}")
    print(f"   Dim: {np.array(unique_embs[0]).shape}")
    print(f"   Size: {os.path.getsize(data_file)/1024/1024:.1f} MB")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
