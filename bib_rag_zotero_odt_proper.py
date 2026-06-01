#!/usr/bin/env python3
"""
Proper citation workflow for LibreOffice .odt
Reads inline citations from document, queries Zotero + bib_rag, inserts bibliography.

Usage: python3 bib_rag_zotero_odt_proper.py /path/to/file.odt [style]
"""

import sys, re, requests
from odf.opendocument import load
from odf.text import P, H

BIB_RAG_EMBED_URL = "http://localhost:8081/v1/embeddings"
ZOTERO_BASE = "http://localhost:23119/api/users/0"


def get_document_text(odt_path):
    """Extract all text from .odt file"""
    doc = load(odt_path)
    
    def get_text(element):
        text = []
        for node in element.childNodes:
            if hasattr(node, 'data'):
                text.append(node.data)
            elif hasattr(node, 'childNodes'):
                text.extend(get_text(node))
        return text
    
    full_text = ''.join(get_text(doc.text))
    return full_text, doc


def extract_inline_citations(text):
    """Find inline citations like 'Taylor et al. (2017)'"""
    citations = []
    seen = set()
    
    # Pattern: "Author et al. (Year)"
    pattern = r'([A-Z][a-z]+\s+et\s+al\.?)(?:\s*,)?\s*\((\d{4})\)'
    matches = re.findall(pattern, text)
    
    for author, year in matches:
        key = f"{author} {year}"
        if key in seen:
            continue
        seen.add(key)
        
        # Get context around citation
        match = re.search(rf'{re.escape(author)}\s*\({year}\)', text)
        if match:
            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 50)
            context = text[start:end]
        else:
            context = ""
        citations.append({
            'author': author.strip(),
            'year': year,
            'context': context
        })
    
    return citations


def search_zotero_by_author_year(author: str, year: str):
    """Search Zotero by author and year"""
    # Extract last name from "Author et al."
    last_name = author.split()[0]
    
    resp = requests.get(
        f"{ZOTERO_BASE}/items",
        params={"q": f"{last_name} {year}", "limit": 5}
    )
    data = resp.json()
    
    if not data:
        return None
    
    # Find the best match
    for item in data:
        d = item.get("data", {})
        item_year = ""
        date_str = d.get("date", "")
        year_match = re.search(r'\b(19\d{2}|20\d{2})\b', date_str)
        if year_match:
            item_year = year_match.group(1)
        
        if item_year == year:
            return item
    
    # Return first if no year match
    return data[0] if data else None


def format_zotero_meta(item):
    """Format Zotero item into citation dict"""
    d = item.get("data", {})
    
    authors = []
    for c in d.get("creators", []):
        first = c.get("firstName", "")
        last = c.get("lastName", "")
        if first and last:
            authors.append(f"{first} {last}")
        elif last:
            authors.append(last)
    
    if len(authors) > 3:
        author_str = f"{authors[0]} et al."
    elif authors:
        author_str = ", ".join(authors[:-1]) + f" and {authors[-1]}" if len(authors) > 1 else authors[0]
    else:
        author_str = "Unknown"
    
    # Extract year
    date_str = d.get("date", "")
    year = ""
    year_match = re.search(r'\b(19\d{2}|20\d{2})\b', date_str)
    if year_match:
        year = year_match.group(1)
    
    return {
        "authors": author_str,
        "year": year,
        "title": d.get("title", ""),
        "journal": d.get("publicationTitle", ""),
        "volume": d.get("volume", ""),
        "issue": d.get("issue", ""),
        "pages": d.get("pages", ""),
        "doi": d.get("DOI", ""),
    }


def search_bib_rag_by_doi(doi: str):
    """Search bib_rag by DOI"""
    if not doi:
        return None
    
    # DOI as query (exact match)
    resp = requests.post(
        BIB_RAG_EMBED_URL,
        headers={"Content-Type": "application/json"},
        json={"input": doi, "model": "bge-m3"},
        timeout=30
    )
    emb = resp.json()["data"][0]["embedding"]
    norm = sum(x*x for x in emb) ** 0.5
    emb = [x/norm for x in emb] if norm > 0 else emb
    
    from langchain_community.vectorstores import Chroma
    
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
        n_results=1,
        include=["documents", "metadatas", "distances"]
    )
    
    if not docs["ids"] or not docs["ids"][0]:
        return None
    
    meta = docs["metadatas"][0][0]
    dist = docs["distances"][0][0]
    sim = 1.0 / (1.0 + dist)
    
    return {
        "title": meta.get("title", ""),
        "year": meta.get("year", ""),
        "doi": meta.get("doi", ""),
        "similarity": sim
    }


def format_bibliography(zotero_meta: dict, style: str = "APA"):
    """Format full bibliography entry"""
    # Handle missing fields gracefully
    authors = zotero_meta.get("authors", "Unknown")
    year = zotero_meta.get("year", "n.d.")
    title = zotero_meta.get("title", "")
    journal = zotero_meta.get("journal", "")
    volume = zotero_meta.get("volume", "")
    issue = zotero_meta.get("issue", "")
    pages = zotero_meta.get("pages", "")
    doi = zotero_meta.get("doi", "")
    
    if style == "APA":
        parts = [f"{authors} ({year})."]
        if title:
            parts.append(f"{title}.")
        if journal:
            j_part = journal
            if volume:
                j_part += f", {volume}"
                if issue:
                    j_part += f"({issue})"
            if pages:
                j_part += f", {pages}"
            parts.append(j_part + ".")
        if doi:
            parts.append(f"https://doi.org/{doi}")
        return " ".join(parts)
    
    elif style == "Vancouver":
        parts = [f"{authors}."]
        if title:
            parts.append(f"{title}.")
        if journal:
            j_part = journal
            if year:
                j_part += f". {year}"
            if volume:
                j_part += f";{volume}"
                if issue:
                    j_part += f"({issue})"
            if pages:
                j_part += f":{pages}"
            parts.append(j_part + ".")
        return " ".join(parts)
    
    else:
        return f"{authors} ({year}) {title}. {journal} {volume}: {pages}. https://doi.org/{doi}"


def insert_bibliography(odt_path, citations, style="APA"):
    """Insert bibliography section at end of document"""
    doc = load(odt_path)
    
    # Add separator
    doc.text.addElement(P(text=""))
    
    # Add bibliography heading
    bib_heading = H(outlinelevel=1, text="References")
    doc.text.addElement(bib_heading)
    
    for i, citation in enumerate(citations, 1):
        bib_entry = format_bibliography(citation, style)
        p = P(text=f"[{i}] {bib_entry}")
        doc.text.addElement(p)
    
    doc.save(odt_path)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 bib_rag_zotero_odt_proper.py /path/to/file.odt [style]")
        print("Styles: APA (default), Vancouver, Nature, simple")
        print("\nThe document should contain inline citations like:")
        print('  - "Taylor et al. (2017) showed that..."')
        print('  - "This was demonstrated (Wilkinson, 2014)."')
        sys.exit(1)
    
    odt_path = sys.argv[1]
    style = sys.argv[2] if len(sys.argv) > 2 else "APA"
    
    print(f"📖 Reading {odt_path}...")
    full_text, doc = get_document_text(odt_path)
    
    # Truncate to avoid processing existing bibliography
    text_content = full_text
    for marker in ['References', 'Bibliography', 'Taylor et al. (2017) Cell segregation']:
        if marker in text_content:
            text_content = text_content.split(marker)[0]
    
    print(f"\n🔍 Extracting inline citations...")
    citations = extract_inline_citations(text_content)
    
    if not citations:
        print("❌ No inline citations found (e.g., 'Taylor et al., 2017')")
        print("\nAdd citations like:")
        print('  - "Taylor et al. (2017) showed that..."')
        print('  - "This was demonstrated (Wilkinson, 2014)."')
        sys.exit(1)
    
    print(f"✅ Found {len(citations)} inline citation(s):")
    for c in citations:
        print(f"  - {c['author']} ({c['year']})")
    
    bibliography = []
    for i, citation in enumerate(citations):
        print(f"\n[{i+1}] Processing: {citation['author']} ({citation['year']})")
        
        # Step 1: Search Zotero by author + year
        print(f"    Searching Zotero...")
        zotero_item = search_zotero_by_author_year(
            citation['author'],
            citation['year']
        )
        
        if zotero_item:
            zotero_meta = format_zotero_meta(zotero_item)
            print(f"    ✅ Zotero: {zotero_meta['authors']} ({zotero_meta['year']})")
            
            # Step 2: Cross-reference with bib_rag by DOI
            if zotero_meta.get('doi'):
                print(f"    Cross-referencing bib_rag (DOI: {zotero_meta['doi'][:30]}...)...")
                bib_match = search_bib_rag_by_doi(zotero_meta['doi'])
                if bib_match:
                    print(f"    ✅ bib_rag match: {bib_match['similarity']:.3f} similarity")
                else:
                    print(f"    ⚠️ Not found in bib_rag")
            else:
                print(f"    ⚠️ No DOI in Zotero entry")
        else:
            print(f"    ⚠️ Not found in Zotero")
            zotero_meta = {
                "authors": citation['author'],
                "year": citation['year'],
                "title": "",
                "journal": "",
                "volume": "",
                "issue": "",
                "pages": "",
                "doi": "",
            }
        
        bibliography.append(zotero_meta)
    
    if bibliography:
        print(f"\n📝 Inserting bibliography ({style} format)...")
        insert_bibliography(odt_path, bibliography, style)
        print(f"✅ Done! {len(bibliography)} reference(s) added.")
        print(f"\nBibliography:")
        for i, bib in enumerate(bibliography, 1):
            print(f"[{i}] {format_bibliography(bib, style)[:100]}...")
    else:
        print("❌ No bibliography entries to add.")


if __name__ == "__main__":
    main()
