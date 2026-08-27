#!/usr/bin/env python3
"""
Phase 1: Parent/Child Hierarchical Indexing for bib_rag

Builds two-level index:
- Parent chunks: section-level (ABSTRACT, INTRODUCTION, RESULTS, etc.)
  → Rich context, stored as JSON in parent_store/
- Child chunks: small pieces (500 chars, 100 overlap)
  → Precise search, stored in ChromaDB with parent_id

Usage:
    python3 -B build_hierarchical.py [--rebuild]

With --rebuild: wipe existing index and rebuild from scratch
Without: resume from checkpoint (like build_stable.py)
"""

import os, sys, re, json, hashlib, time
from pathlib import Path
from datetime import datetime

from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma

# Shared chunking logic (single source of truth) — same dir as this script
from chunking import (
    clean_text, truncate_at_references, extract_meta,
    extract_sections, split_into_paragraphs,
    create_child_chunks, create_parent_chunks, save_parent_store,
    CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP, MIN_PARENT_SIZE,
)

# ─── Multi-KB config ─────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kb_config import get_config
_CFG = get_config()
KB_ROOT = _CFG["kb_root"]
CHROMA_DB_PATH = _CFG["chroma_path"]
PARENT_STORE_DIR = _CFG["parent_store_dir"]
METADATA_LOG = _CFG["metadata_log"]
CHECKPOINT_FILE = _CFG["checkpoint_file"]
BGE_M3_PATH = "/Disk_bot/models/bge-m3"  # shared model, not KB-specific

# ============== Main Build Process ==============

def build_hierarchical(papers_dir, batch_size=50, rebuild=False):
    md_files = sorted(Path(papers_dir).rglob('*.md'))
    total = len(md_files)
    
    print(f"\n{'='*70}")
    print(f"📚 bib_rag Hierarchical Build (Phase 1)")
    print(f"   Papers: {total}")
    print(f"   Batch size: {batch_size}")
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
    
    # Load embedding model
    print("🔌 Loading bge-m3 (CPU)...")
    model = SentenceTransformer(BGE_M3_PATH, trust_remote_code=True, device='cpu')
    print(f"   ✅ dim={model.get_sentence_embedding_dimension()}\n")
    
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
        
        docs_for_chroma = []  # child chunks to embed
        texts_to_embed = []
        batch_parents = []    # parent chunks to save
        batch_metas = {}
        
        for i, fp in enumerate(files, batch_start + 1):
            # Skip already processed
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
                
                # Check hash for incremental
                ch = hashlib.md5(cleaned.encode()).hexdigest()[:16]
                if fp.name in metadata and metadata[fp.name].get('hash') == ch and not rebuild:
                    checkpoint['processed'].add(fp.name)
                    stats['skipped'] += 1
                    continue
                
                meta = extract_meta(cleaned, fp.name)
                
                # === HIERARCHICAL CHUNKING ===
                
                # 1. Create parent chunks (sections)
                parents = create_parent_chunks(cleaned, fp.name, meta)
                
                if not parents:
                    # No sections found — create single parent from full text
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
                
                # Save parents to JSON store
                save_parent_store(parents, fp.name, PARENT_STORE_DIR)
                stats['parents'] += len(parents)
                stats['papers_with_parents'] += 1
                
                # 2. Create child chunks from each parent
                all_children = []
                for parent in parents:
                    children = create_child_chunks(
                        parent['content'],
                        parent['parent_id'],
                        fp.name,
                        parent['section']
                    )
                    all_children.extend(children)
                
                # 3. Prepare for ChromaDB
                for child in all_children:
                    docs_for_chroma.append({
                        'text': child['text'],
                        'meta': {
                            'source': child['source'],
                            'section': child['section'],
                            'parent_id': child['parent_id'],
                            'idx': child['idx'],
                            'wc': len(child['text'].split()),
                            'title': meta.get('title', fp.name),
                            'authors': meta.get('authors', ''),
                            'year': meta.get('year', ''),
                            'journal': meta.get('journal', ''),
                            'doi': meta.get('doi', ''),
                            'pmid': meta.get('pmid', ''),
                            'pmcid': meta.get('pmcid', ''),
                            'hash': ch
                        }
                    })
                    texts_to_embed.append(child['text'])
                
                stats['children'] += len(all_children)
                batch_parents.extend(parents)
                batch_metas[fp.name] = {**meta, 'hash': ch, 'parents': len(parents), 'children': len(all_children)}
                
                checkpoint['processed'].add(fp.name)
                stats['success'] += 1
                
            except Exception as e:
                print(f"   ❌ Error: {fp.name} — {str(e)[:60]}")
                stats['failed'] += 1
                continue
        
        # Embed and store in ChromaDB
        if texts_to_embed:
            print(f"\n[{bn}/{tbn}] Embedding {len(texts_to_embed)} child chunks...")
            try:
                embeddings = model.encode(texts_to_embed, show_progress_bar=False, batch_size=16)
                
                for doc, emb in zip(docs_for_chroma, embeddings):
                    db._collection.add(
                        embeddings=[emb.tolist()],
                        documents=[doc['text']],
                        metadatas=[doc['meta']],
                        ids=[f"{doc['meta']['source']}:{doc['meta']['section']}:idx{doc['meta']['idx']}:child"]
                    )
                
                print(f"   ✅ Stored {len(texts_to_embed)} children in ChromaDB")
                
            except Exception as e:
                print(f"   ❌ Embedding failed: {str(e)[:80]}")
                stats['failed'] += len(texts_to_embed)
        
        # Save checkpoint and metadata
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump({'processed': list(checkpoint['processed']), 'last_batch': bn}, f)
        
        metadata.update(batch_metas)
        with open(METADATA_LOG, 'w') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # Progress report
        elapsed = time.time() - start_time
        rate = stats['success'] / elapsed if elapsed > 0 else 0
        print(f"\n📊 Progress: {stats['success']}/{total} papers ({stats['success']*100//total}%) | "
              f"Parents: {stats['parents']} | Children: {stats['children']} | "
              f"Rate: {rate:.1f} papers/sec")
    
    # Final summary
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"✅ BUILD COMPLETE")
    print(f"   Papers processed: {stats['success']}/{total}")
    print(f"   Parents created: {stats['parents']}")
    print(f"   Children indexed: {stats['children']}")
    print(f"   Skipped: {stats['skipped']} | Failed: {stats['failed']}")
    print(f"   Time: {elapsed//60:.0f}m {elapsed%60:.0f}s")
    print(f"   Parent store: {PARENT_STORE_DIR}")
    print(f"   ChromaDB: {CHROMA_DB_PATH}")
    print(f"{'='*70}\n")
    
    return stats

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Wipe and rebuild from scratch")
    parser.add_argument("--papers-dir", default="/Disk_bot/paper_lib/My Library/md", help="Markdown papers directory")
    parser.add_argument("--batch-size", type=int, default=50, help="Papers per batch")
    args = parser.parse_args()
    
    build_hierarchical(args.papers_dir, args.batch_size, args.rebuild)
