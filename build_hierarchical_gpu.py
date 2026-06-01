#!/usr/bin/env python3
"""
Phase 1: Parent/Child Hierarchical Indexing — GPU-Accelerated Build

Uses llama-server bge-m3 (port 8081) for fast GPU batch embedding.

Usage:
    python3 -B build_hierarchical_gpu.py [--rebuild]
    
With --rebuild: wipe existing index and rebuild from scratch
Without: resume from checkpoint

Requirements:
    - llama-server running on localhost:8081 with bge-m3 model
    - Must use --ubatch-size 8192 to avoid HTTP 500 errors
"""

import os, sys, re, json, hashlib, time
from pathlib import Path
from datetime import datetime

import requests
from langchain_community.vectorstores import Chroma

KB_ROOT = "/Disk_bot/Eph/bib_rag"
CHROMA_DB_PATH = f"{KB_ROOT}/chroma_db_new"
PARENT_STORE_DIR = f"{KB_ROOT}/parent_store"
METADATA_LOG = f"{KB_ROOT}/data/incremental_metadata.json"
CHECKPOINT_FILE = f"{KB_ROOT}/data/build_hierarchical_checkpoint.json"
BIB_RAG_EMBED_URL = "http://localhost:8081/v1/embeddings"

# ============== Parent/Child Configuration ==============

CHILD_CHUNK_SIZE = 500
CHILD_CHUNK_OVERLAP = 100
MIN_PARENT_SIZE = 200
MAX_PARENT_SIZE = 4000
BATCH_EMBED_SIZE = 64  # Number of texts per embedding batch (GPU)

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
    """Split text into sections by headers."""
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
    """Split text into paragraphs."""
    return [p.strip() for p in text.split('\n\n') if p.strip()]

def create_child_chunks(parent_text, parent_id, source, section, 
                        chunk_size=CHILD_CHUNK_SIZE, overlap=CHILD_CHUNK_OVERLAP):
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
                    words = current_text.split()
                    overlap_words = max(1, overlap // 5)
                    overlap_text = ' '.join(words[-overlap_words:]) if len(words) > overlap_words else ""
                    current_text = overlap_text + ' ' + sent if overlap_text else sent
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
                words = current_text.split()
                overlap_words = max(1, overlap // 5)
                overlap_text = ' '.join(words[-overlap_words:]) if len(words) > overlap_words else ""
                current_text = overlap_text + ('\n\n' if overlap_text else '') + para
            else:
                current_text += ('\n\n' if current_text else '') + para
    
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
    """Create parent chunks from text sections."""
    sections = extract_sections(text)
    
    if not sections:
        sections = {'full_text': text}
    
    parents = []
    for sec_name, sec_text in sections.items():
        if not sec_text.strip():
            continue
        
        if len(sec_text) < MIN_PARENT_SIZE:
            continue
        
        content_hash = hashlib.md5(f"{source}:{sec_name}:{sec_text[:200]}".encode()).hexdigest()[:16]
        parent_id = f"{source}#{sec_name}#{content_hash}"
        
        parent = {
            'parent_id': parent_id,
            'source': source,
            'section': sec_name,
            'content': sec_text,
            'word_count': len(sec_text.split()),
            'char_count': len(sec_text),
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
    """Save parent chunks as JSON file."""
    os.makedirs(PARENT_STORE_DIR, exist_ok=True)
    safe_name = re.sub(r'[^\w\-]', '_', source)[:100]
    filepath = os.path.join(PARENT_STORE_DIR, f"{safe_name}.json")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(parents, f, ensure_ascii=False, indent=2)
    return filepath

# ============== GPU Embedding ==============

def embed_batch(texts: list) -> list:
    """Embed batch of texts using llama-server GPU."""
    if not texts:
        return []
    
    try:
        resp = requests.post(
            BIB_RAG_EMBED_URL,
            headers={"Content-Type": "application/json"},
            json={"input": texts, "model": "bge-m3"},
            timeout=120
        )
        resp.raise_for_status()
        data = resp.json()
        
        embeddings = []
        for item in data["data"]:
            emb = item["embedding"]
            norm = sum(x*x for x in emb) ** 0.5
            if norm > 0:
                emb = [x/norm for x in emb]
            embeddings.append(emb)
        
        return embeddings
    except Exception as e:
        print(f"   ❌ Batch embedding failed: {str(e)[:80]}")
        return []

# ============== Main Build Process ==============

def build_hierarchical_gpu(papers_dir, batch_size=50, rebuild=False):
    md_files = sorted(Path(papers_dir).rglob('*.md'))
    total = len(md_files)
    
    print(f"\n{'='*70}")
    print(f"📚 bib_rag Hierarchical Build — GPU-Accelerated (Phase 1)")
    print(f"   Papers: {total}")
    print(f"   Batch size: {batch_size} papers/batch")
    print(f"   Embedding batch: {BATCH_EMBED_SIZE} chunks/request")
    print(f"   Parent store: {PARENT_STORE_DIR}")
    print(f"   Child chunks: {CHILD_CHUNK_SIZE} chars, {CHILD_CHUNK_OVERLAP} overlap")
    print(f"{'='*70}\n")
    
    # Handle rebuild
    if rebuild:
        print("⚠️  REBUILD mode — wiping existing index...")
        import shutil
        if os.path.exists(CHROMA_DB_PATH):
            shutil.rmtree(CHROMA_DB_PATH)
        if os.path.exists(PARENT_STORE_DIR):
            shutil.rmtree(PARENT_STORE_DIR)
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
        if os.path.exists(METADATA_LOG):
            os.remove(METADATA_LOG)
        print("   ✅ Wiped existing data\n")
    
    # Load checkpoint
    checkpoint = {'processed': set()}
    if os.path.exists(CHECKPOINT_FILE) and not rebuild:
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                cp = json.load(f)
                checkpoint['processed'] = set(cp.get('processed', []))
            print(f"📋 Checkpoint: {len(checkpoint['processed'])} already processed")
        except Exception as e:
            print(f"⚠️  Checkpoint load failed: {e}")
    
    # Setup directories
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    os.makedirs(PARENT_STORE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(METADATA_LOG), exist_ok=True)
    
    # Connect ChromaDB
    class Dummy:
        def embed_documents(self, texts): return [[0.0]*1024 for _ in texts]
        def embed_query(self, text): return [0.0]*1024
    
    db = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=Dummy(),
        collection_name="bib_rag_papers"
    )
    
    # Load metadata
    metadata = {}
    if os.path.exists(METADATA_LOG):
        try:
            with open(METADATA_LOG, 'r') as f:
                metadata = json.load(f)
        except: pass
    
    stats = {
        'success': 0, 'failed': 0, 'skipped': 0,
        'parents': 0, 'children': 0, 'papers_with_parents': 0
    }
    start_time = time.time()
    
    for batch_start in range(0, total, batch_size):
        files = md_files[batch_start:batch_start + batch_size]
        bn = batch_start // batch_size + 1
        tbn = (total - 1) // batch_size + 1
        
        all_children = []  # All child chunks for this batch
        batch_metas = {}
        
        for i, fp in enumerate(files, batch_start + 1):
            if fp.name in checkpoint['processed'] and not rebuild:
                stats['skipped'] += 1
                continue
            
            try:
                text = fp.read_text(encoding='utf-8', errors='ignore')
                if len(text.strip()) < 500:
                    checkpoint['processed'].add(fp.name)
                    stats['skipped'] += 1
                    continue
                
                cleaned = clean_text(text)
                cleaned = truncate_at_references(cleaned)
                if len(cleaned.strip()) < 500:
                    checkpoint['processed'].add(fp.name)
                    stats['skipped'] += 1
                    continue
                
                ch = hashlib.md5(cleaned.encode()).hexdigest()[:16]
                if fp.name in metadata and metadata[fp.name].get('hash') == ch and not rebuild:
                    checkpoint['processed'].add(fp.name)
                    stats['skipped'] += 1
                    continue
                
                meta = extract_meta(cleaned, fp.name)
                
                # === HIERARCHICAL CHUNKING ===
                parents = create_parent_chunks(cleaned, fp.name, meta)
                
                if not parents:
                    content_hash = hashlib.md5(f"{fp.name}:full".encode()).hexdigest()[:16]
                    parent_id = f"{fp.name}#full#{content_hash}"
                    parents = [{
                        'parent_id': parent_id,
                        'source': fp.name,
                        'section': 'full_text',
                        'content': cleaned,
                        'word_count': len(cleaned.split()),
                        'char_count': len(cleaned),
                        'meta': meta
                    }]
                
                save_parent_store(parents, fp.name)
                stats['parents'] += len(parents)
                stats['papers_with_parents'] += 1
                
                # Create child chunks
                for parent in parents:
                    children = create_child_chunks(
                        parent['content'],
                        parent['parent_id'],
                        fp.name,
                        parent['section']
                    )
                    all_children.extend(children)
                
                batch_metas[fp.name] = {
                    **meta, 'hash': ch,
                    'parents': len(parents),
                    'children': sum(len(create_child_chunks(p['content'], p['parent_id'], fp.name, p['section'])) for p in parents)
                }
                
                checkpoint['processed'].add(fp.name)
                stats['success'] += 1
                
            except Exception as e:
                print(f"   ❌ Error: {fp.name} — {str(e)[:60]}")
                stats['failed'] += 1
                continue
        
        # Embed and store children in ChromaDB
        if all_children:
            print(f"\n[{bn}/{tbn}] Embedding {len(all_children)} child chunks via GPU...")
            
            # Batch embed
            embeddings = []
            texts_to_embed = [c['text'] for c in all_children]
            
            for i in range(0, len(texts_to_embed), BATCH_EMBED_SIZE):
                batch_texts = texts_to_embed[i:i+BATCH_EMBED_SIZE]
                batch_embs = embed_batch(batch_texts)
                embeddings.extend(batch_embs)
                
                if not batch_embs:
                    print(f"   ⚠️  Failed to embed batch {i//BATCH_EMBED_SIZE + 1}")
            
            if len(embeddings) != len(all_children):
                print(f"   ❌ Embedding mismatch: {len(embeddings)}/{len(all_children)}")
                stats['failed'] += len(all_children) - len(embeddings)
                # Only store successfully embedded
                all_children = all_children[:len(embeddings)]
            
            # Store in ChromaDB
            try:
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
                            'title': meta.get('title', child['source']),
                            'authors': meta.get('authors', ''),
                            'year': meta.get('year', ''),
                            'journal': meta.get('journal', ''),
                            'doi': meta.get('doi', ''),
                            'pmid': meta.get('pmid', ''),
                            'pmcid': meta.get('pmcid', ''),
                            'hash': ch
                        }],
                        ids=[f"{child['source']}:{child['section']}:idx{child['idx']}:child:{int(time.time()*1000)}"]
                    )
                
                stats['children'] += len(all_children)
                print(f"   ✅ Stored {len(all_children)} children in ChromaDB")
                
            except Exception as e:
                print(f"   ❌ ChromaDB store failed: {str(e)[:80]}")
                stats['failed'] += len(all_children)
        
        # Save checkpoint
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump({'processed': list(checkpoint['processed']), 'last_batch': bn}, f)
        
        metadata.update(batch_metas)
        with open(METADATA_LOG, 'w') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # Progress
        elapsed = time.time() - start_time
        rate = stats['success'] / elapsed if elapsed > 0 else 0
        print(f"\n📊 Progress: {stats['success']}/{total} ({stats['success']*100//total}%) | "
              f"Parents: {stats['parents']} | Children: {stats['children']} | "
              f"Rate: {rate:.1f} papers/sec")
    
    # Final summary
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"✅ BUILD COMPLETE")
    print(f"   Papers: {stats['success']}/{total}")
    print(f"   Parents: {stats['parents']}")
    print(f"   Children: {stats['children']}")
    print(f"   Skipped: {stats['skipped']} | Failed: {stats['failed']}")
    print(f"   Time: {elapsed//60:.0f}m {elapsed%60:.0f}s")
    print(f"{'='*70}\n")
    
    return stats

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Wipe and rebuild")
    parser.add_argument("--papers-dir", default="/Disk_bot/paper_lib/My Library/md")
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()
    
    build_hierarchical_gpu(args.papers_dir, args.batch_size, args.rebuild)
