# Agentic RAG Knowledge Base Improvement Plan

## Current Problem Analysis (2026-04-30)

### 1. Duplicate Content Problem ⚠️
**Current state**: Of 4844 document chunks, 1055 are exact duplicates
- Metadata prefix duplication (PMID/Year/Journal) is stored in every chunk
- Multiple chunks from the same paper share the same prefix
- **Waste**: roughly 20% of storage space

**Solution**: 
- Store metadata separately from text
- Concatenate metadata only at retrieval time

### 2. Empty Section Problem ⚠️⚠️
**Current state**: Many section headings have no content following them

| Section type | Empty ratio | Impact |
|----------|--------|------|
| RESULTS | 78.3% | Severe |
| Methods | 73.3% | Severe |
| Figure Legends | 92.9% | Moderate |
| Supplementary | 90.0% | Moderate |

**Root cause**: Inconsistent Markdown formatting
- Some files use `## RESULTS` as a heading but place the content at a lower level
- Some files have content directly after the heading with no blank line
- Sub-sections (e.g. `### Results part 1`) are treated as independent sections

**Solution**:
- Improve the section extraction logic
- Merge sub-sections into their parent sections
- Filter out sections with empty content

### 3. Chunking Granularity Problem
**Current state**: 
- Average 461 words/chunk (range 9-1514)
- 100-word overlap may not be enough
- Some chunks are too short (626 are <20 words)

**Solution**:
- Set a minimum chunk size (100 words)
- Set a maximum chunk size (1000 words)
- Increase overlap to 200 words

### 4. Content Quality Problem
**Current state**:
- Contains meaningless chunks like "Figure 1.", "Table 2."
- Contains non-academic content like "Acknowledgments", "References"
- Contains auxiliary content like "Supplementary Material"

**Solution**:
- Filter out non-academic sections
- Keep only core content (abstract, introduction, results, discussion, methods)

---

## Improvement Options

### Option A: Fix the Current Knowledge Base (Recommended)

1. **Re-chunk**
   - Filter out empty sections
   - Merge sub-sections
   - Set minimum/maximum chunk sizes

2. **Deduplicate**
   - Remove exactly duplicated chunks
   - Separate metadata from content

3. **Filter**
   - Remove Figure/Table chunks
   - Remove Acknowledgments/References
   - Keep only core academic content

### Option B: Incremental Optimization

1. **Add a citation network**
   - Extract citation relationships for each paper
   - Build a citation graph
   - Support citation-based retrieval

2. **Add entity tags**
   - Use NER to extract proteins, genes, cell types
   - Attach entity tags to document chunks
   - Support entity retrieval

3. **Add summary vectors**
   - Generate a summary vector for each paper
   - Retrieve at the paper level first, then at the chunk level
   - Improve retrieval efficiency

### Option C: Multi-Knowledge-Base Fusion

1. **Hierarchical retrieval**
   - Level 1: paper level (based on title/abstract)
   - Level 2: section level (based on section content)
   - Level 3: paragraph level (based on specific text)

2. **Cross-knowledge-base queries**
   - v1 (199 classic papers)
   - v2 (500 new papers)
   - Graphify knowledge graph
   - Automatically select the best source

---

## Concrete Implementation Steps

### Step 1: Fix the Chunking Logic

```python
# Improved chunking strategy
def create_chunks_v3(text, sections, meta):
    """
    Improved chunking strategy:
    1. Filter out empty sections (<50 words)
    2. Merge sub-sections into parent sections
    3. Set min chunk 100 words, max chunk 800 words
    4. Overlap of 200 words
    5. Filter out non-academic sections
    """
    
    # Sections to filter out
    skip_sections = {
        'references', 'acknowledgments', 'acknowledgements',
        'figure legends', 'tables', 'supplementary material',
        'supplementary information', 'competing interests',
        'consent for publication', 'peer review', 'footnotes',
        'figure 1.', 'figure 2.', 'figure 3.', 'figure 4.',
        'table 1.', 'table 2.', 'abbreviations', 'keywords',
        'graphical abstract', 'author contributions',
        'funding', 'ethics approval', 'data availability',
    }
    
    chunks = []
    chunk_size = 800
    overlap = 200
    min_chunk_size = 100
    
    # Metadata prefix (stored only once)
    meta_dict = {
        'pmid': meta.get('pmid', ''),
        'year': meta.get('year', ''),
        'journal': meta.get('journal', ''),
        'if': meta.get('impact_factor', ''),
        'citations': meta.get('citations', ''),
        'tier': meta.get('tier', ''),
    }
    
    # Process core sections
    core_sections = ['abstract', 'introduction', 'background', 
                     'results', 'discussion', 'methods', 'conclusion']
    
    for section_key in core_sections:
        if section_key not in sections:
            continue
        
        section_text = sections[section_key].strip()
        section_words = section_text.split()
        
        # Filter out empty sections
        if len(section_words) < min_chunk_size:
            continue
        
        # Filter out non-academic content
        if any(skip in section_key.lower() for skip in skip_sections):
            continue
        
        # Split into chunks
        step = chunk_size - overlap
        for i in range(0, len(section_words), step):
            chunk_words = section_words[i:i + chunk_size]
            
            # Ensure the last chunk is not too small
            if len(chunk_words) < min_chunk_size and i > 0:
                continue
            
            chunk_text = ' '.join(chunk_words)
            
            # Deduplication check
            chunk_hash = hash(chunk_text[:200])
            if chunk_hash in seen_hashes:
                continue
            seen_hashes.add(chunk_hash)
            
            chunks.append({
                'text': chunk_text,
                'meta': meta_dict,
                'section': section_key,
            })
    
    return chunks
```

### Step 2: Add a Citation Network

```python
# Extract citation relationships
def extract_citations(text):
    """Extract citations from the paper"""
    # Match the (Author, Year) format
    citations = re.findall(r'\(([A-Z][a-z]+\s+et\s+al\.?,?\s+\d{4}[a-z]?)\)', text)
    return citations

# Build the citation graph
citation_graph = {}
for doc in documents:
    pmid = doc['pmid']
    citations = extract_citations(doc['text'])
    citation_graph[pmid] = citations
```

### Step 3: Add Entity Tags

```python
# Extract entities with regular expressions
Eph_PATTERN = re.compile(r'\b(Eph[A-Z]\d?|ephrin-[AB]\d?)\b', re.IGNORECASE)
CELL_PATTERN = re.compile(r'\b(neuron|astrocyte|oligodendrocyte|microglia|HEK293|COS7)\b', re.IGNORECASE)

def extract_entities(text):
    entities = {
        'proteins': Eph_PATTERN.findall(text),
        'cells': CELL_PATTERN.findall(text),
    }
    return entities
```

---

## Expected Outcomes

| Metric | Current | After Improvement |
|------|------|--------|
| Total chunks | 4844 | ~3500 (dedup + filter) |
| Duplicate chunks | 1055 (22%) | <50 (1%) |
| Average chunk size | 461 words | ~600 words |
| Empty/short chunks | 626 (13%) | <100 (2%) |
| Retrieval quality | Medium | High |

---

## Long-Term Improvement Directions

1. **Multimodal retrieval**
   - Add image/table descriptions
   - Support figure/table retrieval

2. **Temporal analysis**
   - Analyze research trends by year
   - Discover emerging research directions

3. **Citation network analysis**
   - Discover high-impact papers
   - Identify research communities

4. **Automatic summarization**
   - Generate a summary for each paper
   - Support quick browsing