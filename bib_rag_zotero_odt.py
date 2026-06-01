#!/usr/bin/env python3
"""
Integrated workflow: bib_rag → Zotero → LibreOffice .odt citation
Usage: python3 bib_rag_zotero_odt.py "your query sentence" /path/to/file.odt
"""

import sys, requests, json
from odf.opendocument import load
from odf.text import P

BIB_RAG_EMBED_URL = "http://localhost:8081/v1/embeddings"
ZOTERO_BASE = "http://localhost:23119/api/users/0"


def search_bib_rag(query: str, top_k: int = 3):
    """Step 1: Query bib_rag for relevant papers"""
    import requests
    from langchain_community.vectorstores import Chroma
    
    # Embed query
    resp = requests.post(
        BIB_RAG_EMBED_URL,
        headers={"Content-Type": "application/json"},
        json={"input": query, "model": "bge-m3"},
        timeout=30
    )
    emb = resp.json()["data"][0]["embedding"]
    norm = sum(x*x for x in emb) ** 0.5
    emb = [x/norm for x in emb] if norm > 0 else emb
    
    class PrecomputedEmbed:
        def __init__(self, emb): self.emb = emb
        def embed_documents(self, texts): return [self.emb] * len(texts)
        def embed_query(self, text): return self.emb
    
    db = Chroma(
        persist_directory="/Disk_bot/Eph/bib_rag/chroma_db_new",
        embedding_function=PrecomputedEmbed(emb),
        collection_name="bib_rag_papers"
    )
    
    docs = db._collection.query(
        query_embeddings=[emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    results = []
    for i in range(len(docs["ids"][0])):
        meta = docs["metadatas"][0][i]
        dist = docs["distances"][0][i]
        sim = 1.0 / (1.0 + dist)
        results.append({
            "title": meta.get("title", ""),
            "year": meta.get("year", ""),
            "doi": meta.get("doi", ""),
            "relevance": sim
        })
    
    return sorted(results, key=lambda x: x["relevance"], reverse=True)


def search_zotero(title: str, doi: str = "", limit: int = 3):
    """Step 2: Find paper metadata in Zotero"""
    query = doi if doi else title
    resp = requests.get(f"{ZOTERO_BASE}/items", params={"q": query, "limit": limit})
    data = resp.json()
    
    if not data:
        return None
    
    item = data[0]
    d = item.get("data", {})
    
    # Format authors
    authors = []
    for c in d.get("creators", []):
        first = c.get("firstName", "")
        last = c.get("lastName", "")
        if first and last:
            authors.append(f"{first} {last}")
        elif last:
            authors.append(last)
    
    # Short author list
    if len(authors) > 2:
        author_str = f"{authors[0]} et al."
    elif authors:
        author_str = ", ".join(authors)
    else:
        author_str = "Unknown"
    
    # Clean year from date (handle various formats)
    date_str = d.get("date", "")
    year = ""
    if date_str:
        # Try to extract 4-digit year
        import re
        year_match = re.search(r'\b(19\d{2}|20\d{2})\b', date_str)
        if year_match:
            year = year_match.group(1)
    
    return {
        "key": item.get("key", ""),
        "title": d.get("title", ""),
        "authors": author_str,
        "year": year,
        "journal": d.get("publicationTitle", ""),
        "volume": d.get("volume", ""),
        "issue": d.get("issue", ""),
        "pages": d.get("pages", ""),
        "doi": d.get("DOI", ""),
        "url": d.get("url", "")
    }


def format_citation(zotero_meta: dict, style: str = "APA") -> str:
    """Format citation in chosen style"""
    if style == "APA":
        return f"{zotero_meta['authors']} ({zotero_meta['year']}). {zotero_meta['title']}. {zotero_meta['journal']}, {zotero_meta['volume']}({zotero_meta['issue']}), {zotero_meta['pages']}. https://doi.org/{zotero_meta['doi']}"
    elif style == "Vancouver":
        return f"{zotero_meta['authors']}. {zotero_meta['title']}. {zotero_meta['journal']}. {zotero_meta['year']};{zotero_meta['volume']}({zotero_meta['issue']}):{zotero_meta['pages']}."
    elif style == "Nature":
        return f"{zotero_meta['authors']} {zotero_meta['title']} {zotero_meta['journal']} {zotero_meta['volume']}, {zotero_meta['pages']} ({zotero_meta['year']})."
    else:
        # Simple format
        return f"{zotero_meta['authors']} ({zotero_meta['year']}) {zotero_meta['title']}. {zotero_meta['journal']} {zotero_meta['volume']}: {zotero_meta['pages']}. https://doi.org/{zotero_meta['doi']}"


def insert_into_odt(odt_path: str, citation_text: str):
    """Step 3: Insert citation into .odt file"""
    doc = load(odt_path)
    p = P(text=citation_text)
    doc.text.addElement(p)
    doc.save(odt_path)


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 bib_rag_zotero_odt.py 'query sentence' /path/to/file.odt [style]")
        print("Styles: APA (default), Vancouver, Nature, simple")
        sys.exit(1)
    
    query = sys.argv[1]
    odt_path = sys.argv[2]
    style = sys.argv[3] if len(sys.argv) > 3 else "APA"
    
    print(f"🔍 Step 1: Searching bib_rag for: '{query}'")
    bib_results = search_bib_rag(query, top_k=3)
    
    if not bib_results:
        print("❌ No results from bib_rag")
        sys.exit(1)
    
    top = bib_results[0]
    print(f"✅ Found: {top['title'][:60]}... ({top['year']})")
    
    # Clean title for Zotero search (remove "Author et al. - Year - " prefix)
    import re
    clean_title = re.sub(r'^[A-Z][a-z]+ et al\. - \d{4} - ', '', top['title'])
    
    print(f"\n📚 Step 2: Looking up in Zotero...")
    zotero_meta = search_zotero(clean_title, top["doi"])
    
    if not zotero_meta:
        print("⚠️ Not found in Zotero, using bib_rag metadata")
        zotero_meta = {
            "authors": "Unknown",
            "year": top["year"],
            "title": clean_title,
            "journal": "",
            "volume": "",
            "issue": "",
            "pages": "",
            "doi": top["doi"],
            "url": ""
        }
    else:
        print(f"✅ Found in Zotero: {zotero_meta['authors']} ({zotero_meta['year']})")
    
    citation = format_citation(zotero_meta, style)
    
    print(f"\n📝 Step 3: Inserting into {odt_path}")
    print(f"\nCitation ({style}):")
    print(citation)
    
    insert_into_odt(odt_path, citation)
    print(f"\n✅ Done! Citation inserted.")


if __name__ == "__main__":
    main()
