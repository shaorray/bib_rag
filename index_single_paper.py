#!/usr/bin/env python3
"""
Index a single markdown paper into bib_rag ChromaDB using llama-server embedding endpoint.
Bypasses SentenceTransformers import issue.
"""
import os, sys, re, json, hashlib, time, requests
from pathlib import Path
from langchain_community.vectorstores import Chroma

KB_ROOT = "/Disk_bot/Eph/bib_rag"
CHROMA_DB_PATH = f"{KB_ROOT}/chroma_db_new"
PARENT_STORE_DIR = f"{KB_ROOT}/parent_store"
METADATA_LOG = f"{KB_ROOT}/data/incremental_metadata.json"
CHECKPOINT_FILE = f"{KB_ROOT}/data/build_hierarchical_checkpoint.json"
EMBED_URL = "http://localhost:8081/embedding"

CHILD_CHUNK_SIZE = 500
CHILD_CHUNK_OVERLAP = 100
MIN_PARENT_SIZE = 200

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
    return meta

def extract_sections(text):
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
    return [p.strip() for p in text.split('\n\n') if p.strip()]

def create_child_chunks(parent_text, parent_id, source, section, chunk_size=CHILD_CHUNK_SIZE, overlap=CHILD_CHUNK_OVERLAP):
    paragraphs = split_into_paragraphs(parent_text)
    if not paragraphs:
        return []
    chunks = []
    current_text = ""
    for para in paragraphs:
        para_len = len(para)
        if para_len > chunk_size * 1.5:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sent in sentences:
                if len(current_text) + len(sent) > chunk_size and current_text:
                    chunks.append({'text': current_text.strip(), 'parent_id': parent_id, 'source': source, 'section': section, 'idx': len(chunks)})
                    words = current_text.split()
                    overlap_text = ' '.join(words[-overlap//5:]) if len(words) > overlap//5 else current_text[-overlap:]
                    current_text = overlap_text + ' ' + sent
                else:
                    current_text += ' ' + sent if current_text else sent
        else:
            if len(current_text) + para_len > chunk_size and current_text:
                chunks.append({'text': current_text.strip(), 'parent_id': parent_id, 'source': source, 'section': section, 'idx': len(chunks)})
                words = current_text.split()
                overlap_words = max(1, overlap // 5)
                overlap_text = ' '.join(words[-overlap_words:]) if len(words) > overlap_words else ""
                current_text = overlap_text + ('\n\n' if overlap_text else '') + para
            else:
                current_text += ('\n\n' if current_text else '') + para
    if current_text.strip():
        chunks.append({'text': current_text.strip(), 'parent_id': parent_id, 'source': source, 'section': section, 'idx': len(chunks)})
    return chunks

def create_parent_chunks(text, source, meta):
    sections = extract_sections(text)
    if not sections:
        sections = {'full_text': text}
    parents = []
    for sec_name, sec_text in sections.items():
        if not sec_text.strip():
            continue
        content_hash = hashlib.md5(f"{source}:{sec_name}:{sec_text[:200]}".encode()).hexdigest()[:16]
        parent_id = f"{source}#{sec_name}#{content_hash}"
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

def save_parent_store(parents, source):
    os.makedirs(PARENT_STORE_DIR, exist_ok=True)
    safe_name = re.sub(r'[^\w\-]', '_', source)[:100]
    filepath = os.path.join(PARENT_STORE_DIR, f"{safe_name}.json")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(parents, f, ensure_ascii=False, indent=2)
    return filepath

def embed_texts(texts, batch_size=16):
    """Embed texts using llama-server endpoint."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_embeddings = []
        for text in batch:
            resp = requests.post(EMBED_URL, json={"content": text}, timeout=120)
            data = resp.json()
            # llama-server returns [{"index": 0, "embedding": [[...]]}]
            emb = data[0]["embedding"][0] if isinstance(data[0]["embedding"][0], list) else data[0]["embedding"]
            batch_embeddings.append(emb)
        all_embeddings.extend(batch_embeddings)
        print(f"   Embedded {min(i+batch_size, len(texts))}/{len(texts)} chunks")
    return all_embeddings

def index_paper(md_path):
    md_path = Path(md_path)
    print(f"\n{'='*60}")
    print(f"Indexing: {md_path.name}")
    print(f"{'='*60}")

    text = md_path.read_text(encoding='utf-8', errors='ignore')
    if len(text.strip()) < 500:
        print("❌ Too short, skipping")
        return False

    cleaned = clean_text(text)
    cleaned = truncate_at_references(cleaned)
    if len(cleaned.strip()) < 500:
        print("❌ Too short after cleaning, skipping")
        return False

    meta = extract_meta(cleaned, md_path.name)
    print(f"  Title: {meta.get('title', 'N/A')[:80]}")
    print(f"  Year:  {meta.get('year', 'N/A')}")
    print(f"  DOI:   {meta.get('doi', 'N/A')}")

    # Create parent chunks
    parents = create_parent_chunks(cleaned, md_path.name, meta)
    if not parents:
        content_hash = hashlib.md5(f"{md_path.name}:full".encode()).hexdigest()[:16]
        parent_id = f"{md_path.name}#full#{content_hash}"
        parents = [{
            'parent_id': parent_id,
            'source': md_path.name,
            'section': 'full_text',
            'content': cleaned,
            'word_count': len(cleaned.split()),
            'char_count': len(cleaned),
            'meta': meta
        }]

    save_parent_store(parents, md_path.name)
    print(f"  Parents: {len(parents)}")

    # Create child chunks
    all_children = []
    for parent in parents:
        children = create_child_chunks(parent['content'], parent['parent_id'], md_path.name, parent['section'])
        all_children.extend(children)
    print(f"  Children: {len(all_children)}")

    # Embed
    texts_to_embed = [c['text'] for c in all_children]
    print(f"  Embedding {len(texts_to_embed)} chunks via llama-server (port 8081)...")
    embeddings = embed_texts(texts_to_embed)

    # Store in ChromaDB
    class Dummy:
        def embed_documents(self, texts): return [[0.0]*1024 for _ in texts]
        def embed_query(self, text): return [0.0]*1024

    db = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=Dummy(),
        collection_name="bib_rag_papers"
    )

    for child, emb in zip(all_children, embeddings):
        db._collection.add(
            embeddings=[emb],
            documents=[child['text']],
            metadatas=[{
                'source': child['source'],
                'section': child['section'],
                'parent_id': child['parent_id'],
                'idx': child['idx'],
                'wc': len(child['text'].split()),
                'title': meta.get('title', md_path.name),
                'authors': meta.get('authors', ''),
                'year': meta.get('year', ''),
                'journal': meta.get('journal', ''),
                'doi': meta.get('doi', ''),
                'pmid': meta.get('pmid', ''),
                'pmcid': meta.get('pmcid', ''),
                'hash': hashlib.md5(cleaned.encode()).hexdigest()[:16]
            }],
            ids=[f"{child['source']}:{child['section']}:idx{child['idx']}:child"]
        )

    # Update checkpoint
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            cp = json.load(f)
    else:
        cp = {'processed': [], 'last_batch': 0}
    if md_path.name not in cp['processed']:
        cp['processed'].append(md_path.name)
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(cp, f, indent=2)

    # Update metadata
    if os.path.exists(METADATA_LOG):
        with open(METADATA_LOG, 'r') as f:
            metadata_log = json.load(f)
    else:
        metadata_log = {}
    metadata_log[md_path.name] = {**meta, 'hash': hashlib.md5(cleaned.encode()).hexdigest()[:16],
                                   'parents': len(parents), 'children': len(all_children)}
    with open(METADATA_LOG, 'w') as f:
        json.dump(metadata_log, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Done! {len(parents)} parents, {len(all_children)} children indexed.")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 -B index_single_paper.py <markdown_file>")
        sys.exit(1)
    index_paper(sys.argv[1])