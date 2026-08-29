#!/usr/bin/env python3
"""
Index a single markdown paper into bib_rag ChromaDB using llama-server embedding endpoint.
Bypasses SentenceTransformers import issue.
"""
import os, sys, re, json, hashlib, time, requests
from pathlib import Path
from langchain_community.vectorstores import Chroma

# Shared chunking logic (single source of truth) — same dir as this script
from chunking import (
    atomic_json_dump,
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
EMBED_URL = _CFG["embed_url_raw"]

def embed_texts(texts, batch_size=16):
    """Embed texts using llama-server endpoint, with retry on transient failures."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_embeddings = []
        for text in batch:
            emb = _embed_one(text)
            if emb is None:
                raise RuntimeError(f"embedding failed after retries for chunk {i}")
            batch_embeddings.append(emb)
        all_embeddings.extend(batch_embeddings)
        print(f"   Embedded {min(i+batch_size, len(texts))}/{len(texts)} chunks")
    return all_embeddings


def _embed_one(text, retries=3):
    """Embed a single text with retry + backoff. Returns embedding list or None.

    bge-m3 (this llama-server build) fails on inputs beyond ~500 words / ~2500
    chars (returns an empty embedding). Truncate long inputs to a safe length so
    a single oversized chunk (e.g. a markdown table) doesn't fail the whole paper.
    """
    import time as _time
    # bge-m3 truncates internally anyway; cap well under the observed failure point
    MAX_CHARS = 2000
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]
    for attempt in range(retries):
        try:
            resp = requests.post(EMBED_URL, json={"content": text}, timeout=120)
            data = resp.json()
            # llama-server returns [{"index": 0, "embedding": [[...]]}]
            emb = data[0]["embedding"]
            if isinstance(emb, list) and emb and isinstance(emb[0], list):
                return emb[0]
            if isinstance(emb, list) and emb:
                return emb
            # empty embedding -> treat as failure, retry
        except Exception:
            pass
        if attempt < retries - 1:
            _time.sleep(2 * (attempt + 1))
    return None

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

    save_parent_store(parents, md_path.name, PARENT_STORE_DIR)
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
        collection_name=_CFG["collection_name"]
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
    atomic_json_dump(cp, CHECKPOINT_FILE)

    # Update the FTS5 (BM25) index so the new paper is visible to the hybrid
    # BM25 channel immediately. Without this, a freshly indexed paper is only
    # reachable via dense vector search until the next full FTS rebuild
    # (silent half-indexing — caught in the 2026-08-29 module audit).
    try:
        try:
            from .hybrid_search import HybridIndex
        except ImportError:  # src/ on sys.path directly (CLI)
            from hybrid_search import HybridIndex
        HybridIndex().upsert_source(md_path.name)
        print(f"  FTS (BM25) index updated for {md_path.name}")
    except Exception as e:
        # Non-fatal: the hybrid layer degrades to dense-only.
        print(f"  ⚠️ FTS upsert failed (BM25 will miss this paper): {e}")

    # Update metadata
    if os.path.exists(METADATA_LOG):
        with open(METADATA_LOG, 'r') as f:
            metadata_log = json.load(f)
    else:
        metadata_log = {}
    metadata_log[md_path.name] = {**meta, 'hash': hashlib.md5(cleaned.encode()).hexdigest()[:16],
                                   'parents': len(parents), 'children': len(all_children)}
    atomic_json_dump(metadata_log, METADATA_LOG)

    print(f"\n✅ Done! {len(parents)} parents, {len(all_children)} children indexed.")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 -B index_single_paper.py <markdown_file>")
        sys.exit(1)
    index_paper(sys.argv[1])