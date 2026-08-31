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

# ─── Multi-KB config ─────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:  # bib_rag-package-try
    from .kb_config import get_config
except ImportError:  # flat (loose-script mode)
    from kb_config import get_config
_CFG = get_config()

# Optional temp-dir override (e.g. a larger scratch disk) via BIB_RAG_TMPDIR;
# otherwise the system default applies.
import tempfile
try:  # bib_rag-package-try
    from .library_config import get_setting as _lib_setting
except ImportError:  # flat (loose-script mode)
    from library_config import get_setting as _lib_setting
_tmp_override = _lib_setting(_CFG["data_root"], "tmpdir")
if _tmp_override and os.path.isdir(_tmp_override):
    tempfile.tempdir = _tmp_override
    os.environ["TMPDIR"] = _tmp_override
KB_ROOT = _CFG["kb_root"]
CHROMA_DB_PATH = _CFG["chroma_path"]
PARENT_STORE_DIR = _CFG["parent_store_dir"]
METADATA_LOG = _CFG["metadata_log"]
CHECKPOINT_FILE = _CFG["checkpoint_file"]
BIB_RAG_EMBED_URL = _CFG["embed_url"]

# ============== Text Processing ==============
# Chunking/cleaning logic lives in chunking.py (shared with build_hierarchical
# incremental index + index_single_paper) — single source of truth, so a full
# rebuild and incremental adds always produce identical chunk structure.

try:  # bib_rag-package-try
    from .chunking import clean_text, truncate_at_references, extract_meta, extract_sections, split_into_paragraphs, create_child_chunks, create_parent_chunks, save_parent_store, CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP, MIN_PARENT_SIZE
except ImportError:  # flat (loose-script mode)
    from chunking import clean_text, truncate_at_references, extract_meta, extract_sections, split_into_paragraphs, create_child_chunks, create_parent_chunks, save_parent_store, CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP, MIN_PARENT_SIZE

# ============== Parent/Child Configuration ==============

MAX_PARENT_SIZE = 4000
BATCH_EMBED_SIZE = 16  # Reduced from 32 to avoid llama-server timeout errors
MAX_CHUNK_TOKENS=1000  # Maximum tokens per chunk before truncation

# ============== GPU Embedding ==============

def embed_batch(texts: list) -> list:
    """Embed batch of texts using llama-server GPU with retry."""
    if not texts:
        return []
    
    # Truncate very long chunks to avoid llama-server timeout
    MAX_LEN = 2000  # chars
    texts = [t[:MAX_LEN] for t in texts]
    
    for attempt in range(2):
        try:
            resp = requests.post(
                BIB_RAG_EMBED_URL,
                headers={"Content-Type": "application/json"},
                json={"input": texts, "model": "bge-m3"},
                timeout=60 if attempt == 0 else 180
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
            print(f"   ⚠️  Batch embed attempt {attempt+1} failed: {str(e)[:60]}")
            if attempt == 0:
                time.sleep(2)
    
    print(f"   ❌ Batch embedding failed after 2 attempts")
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
        collection_name=_CFG["collection_name"]
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
        children_so_far = 0   # Per-paper metadata for incremental_metadata.json
        child_metas = []   # Per-child metadata for ChromaDB batch insert
        
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
                
                save_parent_store(parents, fp.name, PARENT_STORE_DIR)
                stats['parents'] += len(parents)
                stats['papers_with_parents'] += 1
                
                # Create child chunks with per-child metadata for batch insert
                for parent in parents:
                    children = create_child_chunks(
                        parent['content'],
                        parent['parent_id'],
                        fp.name,
                        parent['section']
                    )
                    for child in children:
                        child_metas.append({
                            'title': meta.get('title', ''),
                            'authors': meta.get('authors', ''),
                            'year': meta.get('year', ''),
                            'journal': meta.get('journal', ''),
                            'doi': meta.get('doi', ''),
                            'pmid': meta.get('pmid', ''),
                            'pmcid': meta.get('pmcid', ''),
                            '__hash': ch,
                        })
                    all_children.extend(children)
                
                # Track children count per paper for metadata
                paper_children = len(all_children) - children_so_far
                batch_metas[fp.name] = {
                    **meta, 'hash': ch,
                    'parents': len(parents),
                    'children': paper_children
                }
                children_so_far = len(all_children)
                
                checkpoint['processed'].add(fp.name)
                stats['success'] += 1
                
            except Exception as e:
                print(f"   ❌ Error: {fp.name} — {str(e)[:60]}")
                stats['failed'] += 1
                continue
        
        # Embed and store children in ChromaDB
        if all_children:
            print(f"\n[{bn}/{tbn}] Embedding {len(all_children)} child chunks via GPU...")
            
            # Batch embed — pair each batch with its children so a failed batch
            # drops ITS OWN children (not the tail); storing a child under another
            # chunk's embedding permanently corrupts retrieval for both.
            texts_to_embed = [c['text'] for c in all_children]
            aligned = []  # (child, embedding) pairs that embedded successfully
            
            for i in range(0, len(texts_to_embed), BATCH_EMBED_SIZE):
                batch_slice = slice(i, i + BATCH_EMBED_SIZE)
                batch_texts = texts_to_embed[i:i+BATCH_EMBED_SIZE]
                batch_embs = embed_batch(batch_texts)
                
                if not batch_embs or len(batch_embs) != len(batch_texts):
                    print(f"   ⚠️  Failed to embed batch {i//BATCH_EMBED_SIZE + 1} "
                          f"({len(batch_texts)} children) — skipped")
                    stats['failed'] += len(batch_texts)
                    continue
                aligned.extend(zip(all_children[i:i+BATCH_EMBED_SIZE], batch_embs))
            
            if len(aligned) != len(all_children):
                print(f"   ❌ Embedding mismatch: {len(aligned)}/{len(all_children)} stored")
            
            all_children = [c for c, _ in aligned]
            embeddings = [e for _, e in aligned]
            
            # Store in ChromaDB — batch insert to reduce RAM (ChromaDB's internal buffering)
            try:
                # Extract arrays for batch add
                batch_ids = []
                batch_docs = []
                batch_meta_dicts = []
                batch_embs = []
                for j, (child, emb) in enumerate(zip(all_children, embeddings)):
                    child_meta = child_metas[j] if j < len(child_metas) else {}
                    batch_ids.append(f"{child['source']}:{child['section']}:idx{child['idx']}:child:{int(time.time()*1000)}")
                    batch_docs.append(child['text'])
                    batch_meta_dicts.append({
                        'source': child['source'],
                        'section': child['section'],
                        'parent_id': child['parent_id'],
                        'idx': child['idx'],
                        'wc': len(child['text'].split()),
                        'title': child_meta.get('title', child['source']),
                        'authors': child_meta.get('authors', ''),
                        'year': child_meta.get('year', ''),
                        'journal': child_meta.get('journal', ''),
                        'doi': child_meta.get('doi', ''),
                        'pmid': child_meta.get('pmid', ''),
                        'pmcid': child_meta.get('pmcid', ''),
                        'hash': child_meta.get('__hash', ''),
                    })
                    batch_embs.append(emb)
                
                db._collection.add(
                    ids=batch_ids,
                    embeddings=batch_embs,
                    documents=batch_docs,
                    metadatas=batch_meta_dicts
                )
                
                stats['children'] += len(all_children)
                print(f"   ✅ Stored {len(all_children)} children in ChromaDB (batch)")
                
            except Exception as e:
                print(f"   ❌ ChromaDB store failed: {str(e)[:80]}")
                stats['failed'] += len(all_children)
        
        # Save checkpoint (atomic — truncated checkpoint bricks resume)
        try:  # bib_rag-package-try
            from .chunking import atomic_json_dump
        except ImportError:  # flat (loose-script mode)
            from chunking import atomic_json_dump
        atomic_json_dump({'processed': list(checkpoint['processed']), 'last_batch': bn},
                         CHECKPOINT_FILE)
        
        metadata.update(batch_metas)
        atomic_json_dump(metadata, METADATA_LOG)
        
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
    parser.add_argument("--papers-dir", default=None, help="Markdown papers directory (default: <library>/md)")
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()
    
    if not args.papers_dir:
        args.papers_dir = str(Path(_CFG["data_root"]) / "md")
        print(f"--papers-dir not set; using library default: {args.papers_dir}")
    build_hierarchical_gpu(args.papers_dir, args.batch_size, args.rebuild)
