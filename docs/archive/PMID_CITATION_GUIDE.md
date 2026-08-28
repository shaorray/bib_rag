# PMID Citation Feature User Guide

## Overview

Both the V2 and V3 knowledge bases now support automatically adding PMID citations to query results, making it easy to cite references directly in academic writing.

---

## V3 Knowledge Base - With PMID Citations

### File
- `query_v3_kb.py` - Updated to support PMID citations

### Basic Query (with citations)

```bash
# Regular query - displays PMID citations
python3 query_v3_kb.py "cis interaction" -n 3

# Output format:
# [1] [PMID:27820703]
#     Score: 0.8437
#     PMID: 27820703 | Year: 2019 | IF: 5.99
#     Text: EphA4 expression is increased in the injured cortex...
```

### Generate Academic Paragraph (with citations)

```bash
# Use the --paragraph flag to generate a paragraph with citations
python3 query_v3_kb.py "Eph receptor signaling" -n 3 --paragraph

# Output format:
# ============================================================
# 📝 Academic writing format (with PMID citations)
# ============================================================
# 
# Eph receptors and ephrins function as classic receptors and ligands 
# in ephrin:Eph forward signaling[PMID:30819650]. However, the roles 
# of Eph and ephrin proteins can...[PMID:31406248]
#
# 📚 References:
#    [1] PMID:30819650, Year:2019, Journal:Trends in Molecular Medicine, IF:6.51
#    [2] PMID:31406248, Year:2019, Journal:Oncogene, IF:5.58
#
# 🔗 PMID list: 30819650, 31406248
```

### Python API Usage

```python
from query_v3_kb import V3KnowledgeBase

# Initialize the knowledge base
kb = V3KnowledgeBase()

# Query and get results with citations
results = kb.query_with_citations("cis interaction", n_results=5)

for r in results:
    print(f"{r['text'][:100]}...{r['citation']}")
    # Output: "EphA4 expression is increased...[PMID:27820703]"

# Generate an academic paragraph
paragraph = kb.generate_paragraph("Eph signaling", n_results=3)
print(paragraph['paragraph'])
# Output: "Eph receptors function as...[PMID:30819650]..."

print(paragraph['pmids'])
# Output: ['30819650', '31406248']

print(paragraph['references'])
# Output: ['PMID:30819650, Year:2019, Journal:Trends in Molecular Medicine, IF:6.51', ...]
```

---

## V2 Knowledge Base - With PMID Citations

### File
- `query_v2_kb_with_citations.py` - Adds PMID citation support

### Basic Query

```bash
# Run a query
python3 query_v2_kb_with_citations.py

# Every result will show [PMID:xxxx]
```

### Python API Usage

```python
from query_v2_kb_with_citations import query_with_citations, generate_paragraph_with_citations
from process_v2_papers import PaperProcessor

# Initialize
processor = PaperProcessor()
# Manually load the V2 knowledge base
processor.doc_store.db_path = Path('/Disk_2/claw_working_dir/ephrin_agentic_rag/chroma_db_v2')
processor.doc_store.name = 'ephrin_papers_v2'
processor.doc_store._load()

# Query
results = query_with_citations(processor, "cis interaction", n_results=5)
for r in results:
    print(f"{r['text'][:100]}...{r['citation']}")

# Generate a paragraph
paragraph = generate_paragraph_with_citations("Eph signaling", n_results=3)
print(paragraph['paragraph'])
```

---

## Applications in Academic Writing

### Copy Directly into a Paper

Query results can be copied directly into a paper:

```
The interaction between Eph receptors and ephrin ligands plays a key role in 
neural development[PMID:27820703][PMID:38649412].
Studies have shown that this interaction can modulate the strength and duration 
of receptor signaling[PMID:11053419].
```

### Generate a Reference List

Use the PMID list to automatically generate references:

```python
pmids = ['27820703', '38649412', '11053419']
references = get_reference_list(pmids)
for ref in references:
    print(ref)
    
# Output:
# [1] PMID:27820703, Year:2019, Journal:Journal of Neuroscience, IF:5.99
# [2] PMID:38649412, Year:2024, Journal:Nature Microbiology, IF:13.94
# [3] PMID:11053419, Year:2000, Journal:Journal of Biological Chemistry, IF:4.01
```

---

## Technical Implementation

### V3 Implementation Details

```python
def query_with_citations(self, query_text, n_results=5, **filters):
    results = self.query(query_text, n_results=n_results, **filters)
    
    cited_results = []
    for r in results:
        pmid = r.get('pmid', '')
        citation = f"[PMID:{pmid}]" if pmid else ""
        
        cited_results.append({
            **r,
            'citation': citation,
            'full_reference': f"PMID:{pmid}, Year:{year}, Journal:{journal}, IF:{if_value}"
        })
    
    return cited_results
```

### V2 Implementation Details

```python
def query_with_citations(processor, query_text, n_results=5):
    results = processor.query(query_text, n_results=n_results)
    
    cited_results = []
    for r in results:
        meta = r['metadata']
        pmid = meta.get('pmid', '')
        citation = f"[PMID:{pmid}]" if pmid else ""
        
        cited_results.append({
            'text': r['text'],
            'pmid': pmid,
            'citation': citation,
            # ...
        })
    
    return cited_results
```

---

## Notes

1. **PMID deduplication**: Paragraph generation automatically deduplicates; the same PMID is not cited twice
2. **Citation format**: The default `[PMID:xxxx]` format can be adjusted to match journal requirements
3. **Completeness**: All results include metadata such as PMID, Year, Journal, and IF
4. **Filtering support**: Filters such as year/journal/IF can be applied when generating paragraphs

---

## Example Comparison

### V3 Query Results
```
Query: "Eph receptor signaling"
Results:
  [PMID:30819650] Score:0.8202, Trends in Molecular Medicine, IF:6.51
  [PMID:31406248] Score:0.8081, Oncogene, IF:5.58
  [PMID:31406248] Score:0.7716, Oncogene, IF:5.58
```

### V2 Query Results
```
Query: "Eph receptor signaling"
Results:
  [PMID:15537545] Score:0.1821, Cell, IF:33.6
  [PMID:18448254] Score:0.1814, Pain, IF:3.77
  [PMID:31689239] Score:0.1766, JCI, IF:7.98
```

**Conclusion**: V3 has better semantic understanding and higher scores; V2 also returns results with PMIDs, but with weaker relevance.