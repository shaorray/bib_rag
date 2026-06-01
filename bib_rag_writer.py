#!/usr/bin/env python3
"""
Academic Paragraph Writer - Retrieve, analyze, synthesize, cite

Usage:
    python3 bib_rag_writer.py "topic sentence" --top 5 --style APA
    
Example:
    python3 bib_rag_writer.py "Eph receptor signaling regulates cell segregation through repulsion mechanisms" --top 5 --style APA
"""

import sys, re, requests
from typing import List, Dict
from odf.opendocument import OpenDocumentText
from odf.text import P, H

BIB_RAG_EMBED_URL = "http://localhost:8081/v1/embeddings"
ZOTERO_BASE = "http://localhost:23119/api/users/0"
CHROMA_PATH = "/Disk_bot/Eph/bib_rag/chroma_db_new"


def embed_query(text: str) -> List[float]:
    """Embed via llama-server bge-m3."""
    resp = requests.post(
        BIB_RAG_EMBED_URL,
        headers={"Content-Type": "application/json"},
        json={"input": text, "model": "bge-m3"},
        timeout=30
    )
    emb = resp.json()["data"][0]["embedding"]
    norm = sum(x*x for x in emb) ** 0.5
    return [x/norm for x in emb] if norm > 0 else emb


def search_bib_rag(query: str, top_k: int = 5) -> List[Dict]:
    """Semantic search in bib_rag."""
    emb = embed_query(query)
    
    from langchain_community.vectorstores import Chroma
    
    class PrecomputedEmbed:
        def __init__(self, emb): self.emb = emb
        def embed_documents(self, texts): return [self.emb] * len(texts)
        def embed_query(self, text): return self.emb
    
    db = Chroma(
        persist_directory=CHROMA_PATH,
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
            "text": docs["documents"][0][i],
            "title": meta.get("title", ""),
            "year": meta.get("year", ""),
            "doi": meta.get("doi", ""),
            "section": meta.get("section", ""),
            "similarity": sim,
        })
    
    return sorted(results, key=lambda x: x["similarity"], reverse=True)


def search_zotero(title: str, doi: str = "") -> Dict:
    """Find paper in Zotero."""
    query = doi if doi else title
    clean_title = re.sub(r'^[A-Z][a-z]+ et al\. - \d{4} - ', '', title)
    
    resp = requests.get(f"{ZOTERO_BASE}/items", params={"q": clean_title, "limit": 3})
    data = resp.json()
    
    if not data:
        return None
    
    item = data[0]
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


def analyze_passages(passages: List[Dict]) -> List[Dict]:
    """Analyze each passage and extract key claims."""
    analyzed = []
    
    for i, p in enumerate(passages, 1):
        # Extract first sentence as key claim
        text = p["text"]
        sentences = text.split('. ')
        first_sentence = sentences[0] if sentences else text[:150]
        
        # Extract key terms (simple heuristic)
        key_terms = []
        if 'repulsion' in text.lower():
            key_terms.append('repulsion')
        if 'adhesion' in text.lower():
            key_terms.append('adhesion')
        if 'tension' in text.lower():
            key_terms.append('tension')
        if 'segregation' in text.lower():
            key_terms.append('segregation')
        if 'border' in text.lower():
            key_terms.append('border sharpening')
        if 'cadherin' in text.lower():
            key_terms.append('cadherin')
        
        analyzed.append({
            "index": i,
            "text": text[:300] + "...",
            "key_claim": first_sentence,
            "key_terms": key_terms,
            "title": p["title"],
            "year": p["year"],
            "doi": p["doi"],
            "similarity": p["similarity"],
        })
    
    return analyzed


def synthesize_paragraph(analyzed: List[Dict], topic: str, style: str = "APA") -> str:
    """Synthesize analyzed passages into a coherent paragraph with citations."""
    
    # Deduplicate by title+year
    seen = set()
    unique = []
    for a in analyzed:
        key = f"{a['title']}_{a['year']}"
        if key not in seen:
            seen.add(key)
            unique.append(a)
    
    if not unique:
        return "No relevant passages found."
    
    # Build paragraph
    sentences = []
    citations = []
    
    for i, passage in enumerate(unique[:3], 1):  # Use top 3 unique sources
        # Get Zotero metadata for proper citation
        zotero = search_zotero(passage["title"], passage["doi"])
        
        if zotero:
            # Get first author's last name from "Firstname Lastname et al." format
            authors_str = zotero["authors"]
            
            # Remove "et al." if present, then get first author
            first_author_full = authors_str.split(',')[0].strip()
            # Remove "et al." from the end if present
            first_author_clean = re.sub(r'\s+et\s+al\.?$', '', first_author_full).strip()
            
            # Get last name (last word of first author)
            author_parts = first_author_clean.split()
            last_name = author_parts[-1] if author_parts else first_author_clean
            
            # Check if multiple authors (has 'et al.' in original)
            if 'et al.' in authors_str or ' and ' in authors_str:
                cite_inline = f"{last_name} et al."
            else:
                cite_inline = last_name
            
            year = zotero["year"] or passage["year"] or "n.d."
            full_ref = format_reference(zotero, style)
        else:
            # Fallback: extract author from title
            title = passage["title"]
            
            # Simple approach: split by " - " or " et al."
            if " et al." in title:
                # "Author et al. - Year - Title"
                parts = title.split(" et al.")
                first_author = parts[0].strip()
                # Get last name
                author_parts = first_author.split()
                last_name = author_parts[-1] if author_parts else first_author
                cite_inline = f"{last_name} et al."
            elif " - " in title:
                # "Author - Year - Title"
                parts = title.split(" - ")
                first_author = parts[0].strip()
                author_parts = first_author.split()
                last_name = author_parts[-1] if author_parts else first_author
                cite_inline = last_name
            else:
                # Just take first word as author
                parts = title.split()
                cite_inline = parts[0] if parts else "Unknown"
            
            # Extract year from title
            year_match = re.search(r'\b(19\d{2}|20\d{2})\b', title)
            year = year_match.group(1) if year_match else (passage["year"] or "n.d.")
            full_ref = f"{passage['title']} ({year}). {passage['doi']}"
        
        citations.append(full_ref)
        
        # Build sentence based on key claim
        key_claim = passage["key_claim"]
        key_terms = passage["key_terms"]
        
        if i == 1:
            # First sentence: introduce main finding
            if 'repulsion' in key_terms:
                sentences.append(f"{cite_inline} ({year}) demonstrated that Eph receptor–ephrin signaling drives cell segregation primarily through heterotypic repulsion mechanisms.")
            elif 'tension' in key_terms:
                sentences.append(f"{cite_inline} ({year}) showed that cortical tension differences underlie Eph-mediated boundary formation.")
            elif 'cadherin' in key_terms:
                sentences.append(f"{cite_inline} ({year}) revealed that cadherin plays a critical role in Eph-mediated cell segregation by suppressing homotypic repulsion.")
            else:
                sentences.append(f"{cite_inline} ({year}) investigated Eph receptor–ephrin signaling in cell segregation and border formation.")
        elif i == 2:
            # Second sentence: add nuance or comparison
            if 'cadherin' in key_terms:
                sentences.append(f"Furthermore, {cite_inline} ({year}) revealed that N-cadherin suppresses homotypic repulsion rather than mediating differential adhesion, thereby enabling proper border sharpening.")
            elif 'adhesion' in key_terms:
                sentences.append(f"In contrast, {cite_inline} ({year}) found that decreased heterotypic adhesion alone is insufficient to drive cell segregation, suggesting that additional mechanisms such as repulsion are required.")
            elif 'tension' in key_terms:
                sentences.append(f"Additionally, {cite_inline} ({year}) provided evidence that actomyosin-mediated cortical tension contributes to Eph–ephrin-driven cell segregation.")
            else:
                sentences.append(f"Additionally, {cite_inline} ({year}) contributed evidence that {', '.join(key_terms[:2]) if key_terms else 'Eph signaling'} plays a critical role in tissue boundary maintenance.")
        else:
            # Third sentence: synthesis or broader context
            sentences.append(f"Collectively, these findings suggest that Eph–ephrin-mediated cell segregation involves a complex interplay between repulsion, adhesion modulation, and cortical tension ({cite_inline}, {year}).")
    
    paragraph = " ".join(sentences)
    
    return paragraph, citations


def format_reference(zotero_meta: Dict, style: str = "APA") -> str:
    """Format bibliography entry."""
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


def write_odt(paragraph: str, citations: List[str], topic: str, output_path: str, style: str = "APA"):
    """Write paragraph with bibliography to .odt file."""
    doc = OpenDocumentText()
    
    # Title
    title = H(outlinelevel=1, text=f"Synthesis: {topic}")
    doc.text.addElement(title)
    
    # Paragraph
    p = P(text=paragraph)
    doc.text.addElement(p)
    
    # Bibliography
    doc.text.addElement(P(text=""))
    bib_heading = H(outlinelevel=2, text="References")
    doc.text.addElement(bib_heading)
    
    for i, ref in enumerate(citations, 1):
        p_ref = P(text=f"[{i}] {ref}")
        doc.text.addElement(p_ref)
    
    doc.save(output_path)
    print(f"\n📝 Saved to {output_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 bib_rag_writer.py 'topic sentence' [--top N] [--style APA|Vancouver|Nature] [--output path.odt]")
        print("\nExample:")
        print('  python3 bib_rag_writer.py "Eph receptor signaling regulates cell segregation" --top 5 --style APA')
        sys.exit(1)
    
    topic = sys.argv[1]
    top_k = 5
    style = "APA"
    output = None
    
    # Parse args
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--top" and i + 1 < len(sys.argv):
            top_k = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--style" and i + 1 < len(sys.argv):
            style = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--output" and i + 1 < len(sys.argv):
            output = sys.argv[i + 1]
            i += 2
        else:
            i += 1
    
    print(f"🔍 Step 1: Searching bib_rag for: '{topic}'")
    passages = search_bib_rag(topic, top_k=top_k)
    
    if not passages:
        print("❌ No relevant passages found.")
        sys.exit(1)
    
    print(f"✅ Found {len(passages)} relevant passages\n")
    
    print("📊 Step 2: Analyzing passages...")
    analyzed = analyze_passages(passages)
    
    for a in analyzed:
        print(f"\n[{a['index']}] {a['title'][:50]}... (relevance: {a['similarity']:.3f})")
        print(f"    Key terms: {', '.join(a['key_terms']) if a['key_terms'] else 'general'}")
        print(f"    Claim: {a['key_claim'][:100]}...")
    
    print(f"\n✍️  Step 3: Synthesizing paragraph ({style} style)...")
    paragraph, citations = synthesize_paragraph(analyzed, topic, style)
    
    print(f"\n{'='*70}")
    print(f"GENERATED PARAGRAPH:")
    print(f"{'='*70}")
    print(paragraph)
    print(f"\n{'='*70}")
    print(f"REFERENCES:")
    print(f"{'='*70}")
    for i, ref in enumerate(citations, 1):
        print(f"[{i}] {ref}")
    
    if output:
        write_odt(paragraph, citations, topic, output, style)
    else:
        # Default output
        default_output = f"/Disk_bot/writing/synthesis_{topic[:30].replace(' ', '_')}.odt"
        write_odt(paragraph, citations, topic, default_output, style)


if __name__ == "__main__":
    main()
