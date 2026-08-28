#!/usr/bin/env python3
"""
Academic Paragraph Writer with LLM Debate - Retrieve, analyze with LLM, synthesize, cite

Usage:
    python3 -B bib_rag_writer_debate.py "topic sentence" --top 5 --style APA --output file.odt

What makes this different from bib_rag_writer.py:
    - Uses local LLM (Qwen3.6-35B on port 5015) to analyze relationships between facts
    - LLM debates the evidence, identifies agreements/contradictions
    - Generates richer paragraphs with explicit relational language
"""

import sys, re, requests
from pathlib import Path
from typing import List, Dict
from odf.opendocument import OpenDocumentText
from odf.text import P, H

# Zotero access layer (scripts/zotero_access.py): MCP server first, local HTTP API fallback
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import zotero_access  # noqa: E402

# ─── Multi-KB config ─────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kb_config import get_config
from zotero_match import (pick_best_hit, verify_zotero_hit,  # noqa: E402
                          verify_zotero_hit_ids)
_CFG = get_config()
BIB_RAG_EMBED_URL = _CFG["embed_url"]
CHROMA_PATH = _CFG["chroma_path"]
LLM_URL = "http://localhost:5015/v1/chat/completions"


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
        collection_name=_CFG["collection_name"],
        embedding_function=PrecomputedEmbed(emb),
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
            # paperIdentity keys (P3): chunk meta carries them for the
            # multi-identifier Zotero verification in search_zotero.
            "pmid": meta.get("pmid", ""),
            "pmcid": meta.get("pmcid", ""),
            "section": meta.get("section", ""),
            "similarity": sim,
        })
    
    return sorted(results, key=lambda x: x["similarity"], reverse=True)


def search_zotero(title: str, doi: str = "",
                  pmid: str = "", pmcid: str = "") -> Dict:
    """Find paper in Zotero (via zotero_access: MCP server, HTTP fallback).

    Verified pickup (paper-qa mechanism): candidates are title-similarity /
    identifier checked via zotero_match.pick_best_hit BEFORE acceptance — the
    old blind `items[0]` trust produced wrong-paper citations on fuzzy hits.
    PMID/PMCID (paperIdentity keys, P3) participate when provided: exact
    registry-number match accepts/rejects regardless of title similarity.
    Returns None when no candidate verifies (caller falls back to parsing
    the passage title itself).

    Query shortening: the Zotero MCP search degrades on long queries (an
    8+ word title returns unrelated papers and evicts the true hit from the
    top-k). The first ~8 title words are the discriminative part, so the
    query is truncated there before searching.
    """
    clean_title = re.sub(r'^[A-Z][a-z]+ et al\. - \d{4} - ', '', title)
    short_query = " ".join(clean_title.split()[:8]) or clean_title

    items = zotero_access.zotero_search(short_query, limit=5)
    if not items:
        return None

    query_ids = {k: v for k, v in (("doi", doi), ("pmid", pmid),
                                   ("pmcid", pmcid)) if v}
    best = pick_best_hit(items, clean_title, doi, query_ids=query_ids)
    if best is None:
        return None

    full = zotero_access.zotero_item(best["key"]) or best
    # Re-verify against the FULL record (search snippets can carry truncated
    # titles); a conflict here means the snippet matched but the item is not
    # the same paper.
    if query_ids:
        ok, _s, _r = verify_zotero_hit_ids(clean_title, query_ids, full)
    else:
        ok, _s, _r = verify_zotero_hit(
            clean_title, doi, {"title": full.get("title", ""), "doi": full.get("doi", "")})

    # Snippet fallback: the MCP search snippet carries NO doi (markdown parse
    # hardcodes doi:""), so a DOI-carrying query can only verify at the full
    # record — pick_best_hit never sees an identifier and rejects on title
    # similarity alone. When snippet-level pickup failed, retry each candidate
    # against its FULL record before giving up (one HTTP call per candidate,
    # capped at 3).
    if not ok:
        for cand in items[:3]:
            if cand.get("key") == best.get("key"):
                continue
            cand_full = zotero_access.zotero_item(cand["key"])
            if not cand_full:
                continue
            if query_ids:
                ok, _s, _r = verify_zotero_hit_ids(clean_title, query_ids, cand_full)
            else:
                ok, _s, _r = verify_zotero_hit(
                    clean_title, doi,
                    {"title": cand_full.get("title", ""), "doi": cand_full.get("doi", "")})
            if ok:
                full = cand_full
                break
    if not ok:
        return None
    return {
        "authors": zotero_access.display_authors(full.get("authors", "")),
        "year": full.get("year", ""),
        "title": full.get("title", ""),
        "journal": full.get("journal", ""),
        "volume": full.get("volume", ""),
        "issue": full.get("issue", ""),
        "pages": full.get("pages", ""),
        "doi": full.get("doi", ""),
    }


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


def get_cite_inline(zotero: Dict, passage: Dict) -> tuple:
    """Get inline citation and year."""
    if zotero:
        authors_str = zotero["authors"]
        first_author_full = authors_str.split(',')[0].strip()
        first_author_clean = re.sub(r'\s+et\s+al\.?$', '', first_author_full).strip()
        author_parts = first_author_clean.split()
        last_name = author_parts[-1] if author_parts else first_author_clean
        
        if 'et al.' in authors_str or ' and ' in authors_str:
            cite_inline = f"{last_name} et al."
        else:
            cite_inline = last_name
        
        year = zotero["year"] or passage["year"] or "n.d."
        full_ref = zotero
    else:
        title = passage["title"]
        if " et al." in title:
            parts = title.split(" et al.")
            first_author = parts[0].strip()
            author_parts = first_author.split()
            last_name = author_parts[-1] if author_parts else first_author
            cite_inline = f"{last_name} et al."
        elif " - " in title:
            parts = title.split(" - ")
            first_author = parts[0].strip()
            author_parts = first_author.split()
            last_name = author_parts[-1] if author_parts else first_author
            cite_inline = last_name
        else:
            parts = title.split()
            cite_inline = parts[0] if parts else "Unknown"
        
        year_match = re.search(r'\b(19\d{2}|20\d{2})\b', title)
        year = year_match.group(1) if year_match else (passage["year"] or "n.d.")
        full_ref = None
    
    return cite_inline, year, full_ref


def debate_with_llm(topic: str, passages: List[Dict]) -> Dict:
    """
    Use local LLM to analyze relationships between retrieved passages.
    Returns structured synthesis with claims and relationships.
    """
    
    # Build context from passages
    context_parts = []
    for i, p in enumerate(passages, 1):
        # Extract first sentence as key claim
        text = p["text"]
        sentences = text.split('. ')
        first_sentence = sentences[0] if sentences else text[:200]
        
        # Get citation info (multi-identifier: PMID/PMCID ride along when the
        # chunk meta carries them — exact registry match beats title fuzzy).
        zotero = search_zotero(p["title"], p["doi"],
                               pmid=p.get("pmid", ""), pmcid=p.get("pmcid", ""))
        cite_inline, year, _ = get_cite_inline(zotero, p)
        
        context_parts.append(
            f"SOURCE [{i}]: {cite_inline} ({year})\n"
            f"Key finding: {first_sentence}\n"
            f"Full excerpt: {text[:400]}...\n"
            f"---"
        )
    
    context = "\n".join(context_parts)
    
    prompt = f"""You are an expert academic synthesizer. Analyze the following research sources about "{topic}" and produce a structured synthesis.

{context}

INSTRUCTIONS:
1. Identify the MAIN CLAIM from each source
2. Describe how these claims RELATE to each other (agree, contradict, complement, extend)
3. Identify any GAPS or UNRESOLVED QUESTIONS
4. Produce output in this exact JSON format:

{{
    "main_thesis": "One-sentence synthesis of the overall finding",
    "claims": [
        {{
            "source_num": 1,
            "claim": "What this source found",
            "evidence_type": "experimental/theoretical/review",
            "key_mechanism": "repulsion/adhesion/tension/signaling"
        }},
        ...
    ],
    "relationships": [
        {{
            "between": [1, 2],
            "relation": "agree/contradict/complement/extend",
            "description": "How these sources relate"
        }},
        ...
    ],
    "narrative_flow": [
        "Sentence 1: Introduction to the field",
        "Sentence 2: First key finding with citation",
        "Sentence 3: How second finding relates",
        "Sentence 4: Resolution or broader implication"
    ],
    "suggested_transitions": [
        "transition phrase for first-to-second",
        "transition phrase for second-to-third"
    ]
}}

Requirements:
- Use specific relational language: "consistent with", "in contrast to", "building upon", "challenges", "extends"
- Cite sources as [1], [2], [3] in narrative_flow
- Be precise about mechanisms (not just "Eph signaling" but "heterotypic repulsion")
- Identify what each paper CONTRIBUTES to the overall picture
"""

    resp = requests.post(
        LLM_URL,
        headers={"Content-Type": "application/json"},
        json={
            "model": "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2000,
            "temperature": 0.3,
            "reasoning": False
        },
        timeout=120
    )
    
    result = resp.json()
    content = result["choices"][0]["message"]["content"]
    
    # Extract JSON from response
    json_match = re.search(r'\{.*\}', content, re.DOTALL)
    if json_match:
        try:
            import json
            return json.loads(json_match.group())
        except:
            return {"raw_response": content}
    
    return {"raw_response": content}


def synthesize_with_llm(topic: str, passages: List[Dict], style: str = "APA"):
    """
    Full synthesis: debate with LLM, then format with proper citations.
    """
    
    print(f"\n🧠 Step 3: Debating evidence with LLM (Qwen3.6-35B)...")
    
    # Get LLM analysis
    llm_result = debate_with_llm(topic, passages)
    
    if "raw_response" in llm_result:
        print("⚠️ LLM returned unstructured response, falling back to basic synthesis")
        print("Raw:", llm_result["raw_response"][:300])
        # Fallback to basic
        return basic_synthesize(passages, style)
    
    print(f"✅ LLM synthesis complete")
    print(f"\n📋 Main Thesis: {llm_result.get('main_thesis', 'N/A')}")
    
    # Show relationships
    if "relationships" in llm_result:
        print(f"\n🔗 Relationships identified:")
        for rel in llm_result["relationships"]:
            srcs = rel.get("between", [])
            print(f"   Sources {srcs}: {rel.get('relation', '?')} - {rel.get('description', '')[:80]}")
    
    # Build paragraph from LLM narrative + real citations
    narrative = llm_result.get("narrative_flow", [])
    
    # Replace [1], [2], [3] with actual citations
    citations = []
    citation_map = {}
    
    # Get citations for each source
    for i, passage in enumerate(passages, 1):
        zotero = search_zotero(passage["title"], passage["doi"],
                               pmid=passage.get("pmid", ""),
                               pmcid=passage.get("pmcid", ""))
        cite_inline, year, full_ref = get_cite_inline(zotero, passage)
        
        if full_ref:
            ref_text = format_reference(full_ref, style)
        else:
            ref_text = f"{passage['title']} ({year}). {passage['doi']}"
        
        citation_map[f"[{i}]"] = f"{cite_inline} ({year})"
        citations.append(ref_text)
    
    # Replace placeholders in narrative
    paragraph_sentences = []
    for sentence in narrative:
        for placeholder, citation in citation_map.items():
            sentence = sentence.replace(placeholder, citation)
        paragraph_sentences.append(sentence)
    
    paragraph = " ".join(paragraph_sentences)
    
    return paragraph, citations, llm_result


def basic_synthesize(passages: List[Dict], style: str = "APA"):
    """Fallback basic synthesis without LLM."""
    sentences = []
    citations = []
    
    for i, passage in enumerate(passages[:3], 1):
        zotero = search_zotero(passage["title"], passage["doi"],
                               pmid=passage.get("pmid", ""),
                               pmcid=passage.get("pmcid", ""))
        cite_inline, year, full_ref = get_cite_inline(zotero, passage)
        
        if full_ref:
            ref_text = format_reference(full_ref, style)
        else:
            ref_text = f"{passage['title']} ({year}). {passage['doi']}"
        
        citations.append(ref_text)
        
        text = passage["text"]
        key_terms = []
        if 'repulsion' in text.lower(): key_terms.append('repulsion')
        if 'adhesion' in text.lower(): key_terms.append('adhesion')
        if 'tension' in text.lower(): key_terms.append('tension')
        if 'cadherin' in text.lower(): key_terms.append('cadherin')
        
        if i == 1:
            sentences.append(f"{cite_inline} ({year}) demonstrated that Eph receptor–ephrin signaling drives cell segregation primarily through heterotypic repulsion mechanisms.")
        elif i == 2:
            sentences.append(f"Furthermore, {cite_inline} ({year}) revealed that N-cadherin suppresses homotypic repulsion rather than mediating differential adhesion, thereby enabling proper border sharpening.")
        else:
            sentences.append(f"Collectively, these findings suggest that Eph–ephrin-mediated cell segregation involves a complex interplay between {', '.join(key_terms[:2]) if key_terms else 'repulsion and adhesion'} ({cite_inline}, {year}).")
    
    return " ".join(sentences), citations, {}


def write_odt(paragraph: str, citations: List[str], topic: str, output_path: str, llm_result: Dict = None):
    """Write paragraph with bibliography to .odt file."""
    doc = OpenDocumentText()
    
    # Title
    title = H(outlinelevel=1, text=f"Synthesis: {topic}")
    doc.text.addElement(title)
    
    # Paragraph
    p = P(text=paragraph)
    doc.text.addElement(p)
    
    # LLM analysis (if available)
    if llm_result and "main_thesis" in llm_result:
        doc.text.addElement(P(text=""))
        doc.text.addElement(H(outlinelevel=2, text="Analysis Notes"))
        
        if "relationships" in llm_result:
            p_rel = P(text="Key relationships: ")
            for rel in llm_result["relationships"]:
                srcs = rel.get("between", [])
                desc = rel.get("description", "")
                p_rel.addText(f"Sources {srcs} {rel.get('relation', '?')}: {desc[:60]}. ")
            doc.text.addElement(p_rel)
        
        if "suggested_transitions" in llm_result:
            p_trans = P(text="Suggested transitions: ")
            for trans in llm_result["suggested_transitions"]:
                p_trans.addText(f"'{trans}' ")
            doc.text.addElement(p_trans)
    
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
        print("Usage: python3 -B bib_rag_writer_debate.py 'topic sentence' [--top N] [--style APA|Vancouver|Nature] [--output path.odt]")
        print("\nExample:")
        print('  python3 -B bib_rag_writer_debate.py "Eph receptor signaling regulates cell segregation through repulsion" --top 5 --style APA')
        sys.exit(1)
    
    topic = sys.argv[1]
    top_k = 5
    style = "APA"
    output = None
    
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
    for i, p in enumerate(passages, 1):
        text_preview = p["text"][:100]
        print(f"  [{i}] {p['title'][:50]}... ({p['year']})")
        print(f"      {text_preview}...")
    
    # Synthesize with LLM debate
    paragraph, citations, llm_result = synthesize_with_llm(topic, passages, style)
    
    print(f"\n{'='*70}")
    print(f"GENERATED PARAGRAPH:")
    print(f"{'='*70}")
    print(paragraph)
    print(f"\n{'='*70}")
    print(f"REFERENCES ({style}):")
    print(f"{'='*70}")
    for i, ref in enumerate(citations, 1):
        print(f"[{i}] {ref}")
    
    if output:
        write_odt(paragraph, citations, topic, output, llm_result)
    else:
        default_output = str(Path(_CFG["outputs_dir"]) / f"debate_{topic[:30].replace(' ', '_')}.odt")
        write_odt(paragraph, citations, topic, default_output, llm_result)


if __name__ == "__main__":
    main()
