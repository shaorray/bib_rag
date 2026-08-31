#!/usr/bin/env python3
"""
bib_rag query tool - Simple semantic search for academic writing
Uses llama-server bge-m3 (port 8081) + ChromaDB
"""

import sys, requests, sqlite3, json, os
from typing import List, Dict

# ─── Multi-KB config ─────────────────────────────────────────────────────
# Supports BIB_RAG_ROOT env var and --kb flag for switching knowledge bases.
# Default: bib_rag (Eph-ephrin). Use --kb geo_rag for geology papers.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:  # bib_rag-package-try
    from .kb_config import get_config, parse_kb_arg, print_config
except ImportError:  # flat (loose-script mode)
    from kb_config import get_config, parse_kb_arg, print_config

# Strip --kb from argv before legacy arg parsing
_argv = parse_kb_arg()
if _argv:
    sys.argv = [sys.argv[0]] + _argv

_CFG = get_config()
CHROMA_PATH = _CFG["chroma_sqlite"]
EMBED_URL = _CFG["embed_url"]
PARENT_STORE_DIR_PRIMARY = _CFG["parent_store_dir"]
PARENT_STORE_DIR_DISABLED = _CFG["parent_store_disabled_dir"]
_CHROMA_DIR = _CFG["chroma_path"]
_COLLECTION = _CFG["collection_name"]


def _load_parent_with_fallback(parent_id: str):
    """
    Try parent_store/ first, then parent_store_disabled/.  Returns the parent
    dict (with 'meta' inside) or None.  This makes 57 disabled papers (which
    ChromaDB still references but parent_store no longer has) recoverable via
    fallback (added 2026-06-19; recovers papers disabled from the primary store).
    """
    if not parent_id:
        return None
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        try:  # bib_rag-package-try
            from .parent_store_manager import ParentStoreManager
        except ImportError:  # flat (loose-script mode)
            from parent_store_manager import ParentStoreManager
    except Exception:
        return None
    for d in (PARENT_STORE_DIR_PRIMARY, PARENT_STORE_DIR_DISABLED):
        if not os.path.isdir(d):
            continue
        pm = ParentStoreManager(store_dir=d)
        p = pm.load_content(parent_id)
        if p is not None:
            return p
    return None


def enrich_metadata_with_parent_fallback(results: List[Dict]) -> List[Dict]:
    """
    For each ChromaDB result, look up the parent JSON in primary then disabled
    store, and overlay any fresh meta fields (key, title, doi, etc.) onto the
    chunk metadata.  This is what makes the post-2026-06-19 `meta.key` fill
    visible to retrieval without rebuilding ChromaDB.
    """
    for r in results:
        meta = r.get("metadata") or {}
        pid = meta.get("parent_id") or ""
        parent = _load_parent_with_fallback(pid)
        if not parent:
            continue
        p_meta = parent.get("meta", {}) or {}
        # Overlay: fresh parent fields win over stale ChromaDB metadata.
        for k in ("key", "title", "authors", "year", "journal", "doi",
                  "pmid", "pmcid", "source"):
            v = p_meta.get(k)
            if v:
                meta[k] = v
        # Annotate whether the chunk came from a disabled paper.
        # The JSON file on disk is named via _safe_filename(source) (strips
        # non-word chars), so we must use the same rule to check existence.
        try:
            try:  # bib_rag-package-try
                from .parent_store_manager import ParentStoreManager as _PSM
            except ImportError:  # flat (loose-script mode)
                from parent_store_manager import ParentStoreManager as _PSM
            p_source = parent.get("source", "")
            if p_source:
                # strip .md if present, then apply _safe_filename, then add .json
                if p_source.endswith(".md"):
                    stem = p_source[:-3]
                else:
                    stem = p_source
                safe_name = _PSM(store_dir=PARENT_STORE_DIR_PRIMARY)._safe_filename(stem)
                primary_path = os.path.join(PARENT_STORE_DIR_PRIMARY, f"{safe_name}.json")
                if not os.path.exists(primary_path):
                    meta["_note"] = "paper_in_disabled_store"
        except Exception:
            pass
    return results


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
        persist_directory=_CHROMA_DIR,
        embedding_function=PrecomputedEmbed(embedding),
        collection_name=_COLLECTION
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
    # Overlay fresh parent_store meta (esp. meta.key) onto each chunk
    results = enrich_metadata_with_parent_fallback(results)
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
        art_key = meta.get("key", "") or ""
        note = meta.get("_note", "") or ""
        
        print(f"[{i}] 📄 {title}")
        print(f"    Year: {year} | Section: {section} | Relevance: {sim:.3f}")
        if art_key:
            print(f"    🔑 @article{{{art_key},")
        if doi:
            print(f"    🔗 DOI: {doi}")
        if note:
            print(f"    ⚠️  {note}")
        
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
        art_key = meta.get("key", "") or ""
        if art_key:
            print(f"    🔑 @article{{{art_key},")
        if meta.get('doi'):
            print(f"    🔗 DOI: {meta.get('doi')}")
        note = meta.get('_note', '') or ''
        if note:
            print(f"    ⚠️  {note}")
        text = r["text"][:400].replace("\n", " ")
        print(f"    💬 \"{text}...\"")
        print()
    
    return results


def export_bib(results: List[Dict], out_path: str, offline: bool = False) -> int:
    """Write a References .bib for the papers in a search/cite result set.

    Consumes the chunk metadata already produced by search() (parent_store
    meta, incl. meta.key); dedupes by source filename so multi-chunk papers
    yield one entry. Returns the number of entries written.
    """
    try:
        try:  # bib_rag-package-try
            from .bibtex_export import export_answers_bib
        except ImportError:  # flat (loose-script mode)
            from bibtex_export import export_answers_bib
    except ImportError:  # src/ on sys.path directly
        from src.bibtex_export import export_answers_bib
    sources: List[str] = []
    seen = set()
    for r in results:
        src = (r.get("metadata") or {}).get("source", "")
        if src and src not in seen:
            seen.add(src)
            sources.append(src)
    if not sources:
        print("⚠️  No citable sources in this result set — nothing to export.")
        return 0
    res = export_answers_bib(sources, out_path, offline=offline)
    print(f"📖 BibTeX: {res['written']} entr{'y' if res['written'] == 1 else 'ies'} "
          f"→ {res['out_path']} ({res['skipped']} skipped)")
    return res["written"]


def main():
    if len(sys.argv) < 2:
        print(f"""
📚 {_CFG['kb_name']} Query Tool
Usage:
  python3 query_bib_rag.py "your search query"
  python3 query_bib_rag.py --cite "claim you want evidence for"
  python3 query_bib_rag.py --cite "Eph receptors mediate axon guidance" --top 3
  python3 query_bib_rag.py --kb geo_rag "subduction zone"
  python3 query_bib_rag.py "axon guidance" --export-bib refs.bib
  python3 query_bib_rag.py --cite "claim" --export-bib refs.bib --offline-bib

Options:
  --cite          Find supporting citations for a claim
  --top N         Return top N results (default: 5)
  --kb NAME       Switch knowledge base (bib_rag, geo_rag)
  --export-bib P  Also write a BibTeX References file for the results
  --offline-bib   Synthesize entries from parent_store meta only (no network)

Active KB: {_CFG['kb_name']} at {_CFG['kb_root']}
        """)
        sys.exit(1)
    
    # Parse args
    args = sys.argv[1:]
    cite_mode_flag = False
    top_k = 5
    export_bib_path = ""
    offline_bib = False
    query_parts = []
    
    i = 0
    while i < len(args):
        if args[i] == "--cite":
            cite_mode_flag = True
            i += 1
        elif args[i] == "--top" and i + 1 < len(args):
            top_k = int(args[i + 1])
            i += 2
        elif args[i] == "--export-bib" and i + 1 < len(args):
            export_bib_path = args[i + 1]
            i += 2
        elif args[i] == "--offline-bib":
            offline_bib = True
            i += 1
        else:
            query_parts.append(args[i])
            i += 1
    
    query = " ".join(query_parts)
    
    if cite_mode_flag:
        results = cite_mode(query, top_k)
    else:
        results = search(query, top_k)
        format_results(results, query)
    if export_bib_path:
        export_bib(results, export_bib_path, offline=offline_bib)


if __name__ == "__main__":
    main()
