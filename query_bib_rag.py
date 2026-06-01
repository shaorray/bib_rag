#!/usr/bin/env python3
"""
bib_rag query tool - Simple semantic search for academic writing
Uses llama-server bge-m3 (port 8081) + ChromaDB
"""

import sys, requests, sqlite3, json
from typing import List, Dict

CHROMA_PATH = "/Disk_bot/Eph/bib_rag/chroma_db_new/chroma.sqlite3"
EMBED_URL = "http://localhost:8081/v1/embeddings"


def embed_query(text: str) -> List[float]:
    """Embed query text via llama-server bge-m3."""
    resp = requests.post(
        EMBED_URL,
        headers={"Content-Type": "application/json"},
        json={"input": text, "model": "bge-m3"},
        timeout=30
    )
    resp.raise_for_status()
    emb = resp.json()["data"][0]["embedding"]
    # Normalize
    norm = sum(x*x for x in emb) ** 0.5
    return [x/norm for x in emb] if norm > 0 else emb


def search(query: str, top_k: int = 5) -> List[Dict]:
    """Semantic search in ChromaDB."""
    emb = embed_query(query)
    
    conn = sqlite3.connect(CHROMA_PATH)
    c = conn.cursor()
    
    # Get all embeddings with metadata
    # Chroma stores: embeddings(id, segment_id, embedding_id, seq_id)
    # We need to join with embedding_metadata
    c.execute("""
        SELECT e.id, emd.string_value as doc
        FROM embeddings e
        JOIN embedding_metadata emd ON e.id = emd.id AND emd.key = '#document'
        LIMIT 10000
    """)
    
    # Simple L2 distance computation in Python
    results = []
    for row in c.fetchall():
        emb_id, doc = row
        # Get the embedding vector from the embeddings table
        # Actually chroma stores embeddings in a separate binary format
        # Let's use a simpler approach - query the fulltext search + metadata
        pass
    
    conn.close()
    
    # Better: use Chroma's native query via langchain
    return native_chroma_search(query, emb, top_k)


def native_chroma_search(query: str, embedding: List[float], top_k: int = 5):
    """Use ChromaDB's native similarity search."""
    from langchain_community.vectorstores import Chroma
    
    # Dummy embed function since we already have the embedding
    class PrecomputedEmbed:
        def __init__(self, emb):
            self.emb = emb
        def embed_documents(self, texts):
            return [self.emb] * len(texts)
        def embed_query(self, text):
            return self.emb
    
    db = Chroma(
        persist_directory="/Disk_bot/Eph/bib_rag/chroma_db_new",
        embedding_function=PrecomputedEmbed(embedding),
        collection_name="bib_rag_papers"
    )
    
    # Use similarity_search_by_vector
    docs = db._collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    results = []
    for i in range(len(docs["ids"][0])):
        results.append({
            "id": docs["ids"][0][i],
            "text": docs["documents"][0][i],
            "metadata": docs["metadatas"][0][i],
            "distance": docs["distances"][0][i]
        })
    return results


def format_results(results: List[Dict], query: str):
    """Format search results for academic writing."""
    print(f"\n{'='*70}")
    print(f"🔍 Query: {query}")
    print(f"{'='*70}\n")
    
    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        dist = r["distance"]
        # Convert distance to similarity score (L2 -> similarity)
        sim = 1.0 / (1.0 + dist) if dist is not None else 0
        
        title = meta.get("title", "Unknown")[:60]
        year = meta.get("year", "N/A")
        doi = meta.get("doi", "")
        section = meta.get("section", "")
        
        print(f"[{i}] 📄 {title}")
        print(f"    Year: {year} | Section: {section} | Relevance: {sim:.3f}")
        if doi:
            print(f"    DOI: {doi}")
        
        # Show excerpt (first 300 chars)
        text = r["text"][:300].replace("\n", " ")
        print(f"    💬 {text}...")
        print()
    
    print(f"{'='*70}")
    print(f"Found {len(results)} relevant passages\n")


def cite_mode(claim: str, top_k: int = 5):
    """Find supporting evidence for a claim."""
    print(f"\n{'='*70}")
    print(f"📚 Citation Search: \"{claim[:80]}...\"")
    print(f"{'='*70}\n")
    
    results = search(claim, top_k=top_k)
    
    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        dist = r["distance"]
        sim = 1.0 / (1.0 + dist) if dist is not None else 0
        
        print(f"[{i}] Supporting Evidence (relevance: {sim:.3f})")
        print(f"    📄 {meta.get('title', 'Unknown')[:60]} ({meta.get('year', 'N/A')})")
        if meta.get('doi'):
            print(f"    🔗 DOI: {meta.get('doi')}")
        text = r["text"][:400].replace("\n", " ")
        print(f"    💬 \"{text}...\"")
        print()
    
    return results


def main():
    if len(sys.argv) < 1:
        print("""
📚 bib_rag Query Tool
Usage:
  python3 query_bib_rag.py "your search query"
  python3 query_bib_rag.py --cite "claim you want evidence for"
  python3 query_bib_rag.py --cite "Eph receptors mediate axon guidance" --top 3

Options:
  --cite    Find supporting citations for a claim
  --top N   Return top N results (default: 5)

Examples:
  python3 query_bib_rag.py "cis interaction mechanism"
  python3 query_bib_rag.py "EphA4 reverse signaling"
  python3 query_bib_rag.py --cite "Eph receptors promote tumor suppression" --top 3
        """)
        sys.exit(1)
    
    # Parse args
    args = sys.argv[1:]
    cite_mode_flag = False
    top_k = 5
    query_parts = []
    
    i = 0
    while i < len(args):
        if args[i] == "--cite":
            cite_mode_flag = True
            i += 1
        elif args[i] == "--top" and i + 1 < len(args):
            top_k = int(args[i + 1])
            i += 2
        else:
            query_parts.append(args[i])
            i += 1
    
    query = " ".join(query_parts)
    
    if cite_mode_flag:
        cite_mode(query, top_k)
    else:
        results = search(query, top_k)
        format_results(results, query)


if __name__ == "__main__":
    main()
