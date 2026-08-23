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
    clean_text, truncate_at_references, extract_meta,
    extract_sections, split_into_paragraphs,
    create_child_chunks, create_parent_chunks, save_parent_store,
    CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP, MIN_PARENT_SIZE,
)

KB_ROOT = "/Disk_bot/Eph/bib_rag"
CHROMA_DB_PATH = f"{KB_ROOT}/chroma_db_new"
PARENT_STORE_DIR = f"{KB_ROOT}/parent_store"
METADATA_LOG = f"{KB_ROOT}/data/incremental_metadata.json"
CHECKPOINT_FILE = f"{KB_ROOT}/data/build_hierarchical_checkpoint.json"
EMBED_URL = "http://localhost:8081/embedding"

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