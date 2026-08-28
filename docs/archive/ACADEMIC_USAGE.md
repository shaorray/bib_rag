# 📝 Academic Paper Writing - Agentic RAG Usage Guide

## Quick Start

```python
# Import in Python
import sys
sys.path.insert(0, '/Disk_2/claw_working_dir/ephrin_agentic_rag')
from academic_writer import AcademicWritingAssistant

# Initialize the assistant
assistant = AcademicWritingAssistant()
```

---

## Core Features

### 1. Find References

**Scenario**: While writing a paper, you need to find supporting literature for a claim

```python
# Method 1: Python code
citations = assistant.find_references(
    claim="cis interaction inhibits Eph receptor signaling",
    min_relevance=0.15
)

for cit in citations:
    print(f"{cit.authors} ({cit.year}). {cit.title}")
```

**Command line**:
```bash
python3 academic_writer.py --cite "cis interaction inhibits Eph receptor signaling"
```

**Example output**:
```
[1] Kao and Kania (2011) - rel: 0.240
    Ephrin-Mediated cis-Attenuation of Eph Receptor Signaling Is Essential...

[2] Carvalho et al. (2006) - rel: 0.193
    Silencing of EphA3 through a cis interaction with ephrinA5...
```

---

### 2. Fact Check

**Scenario**: After writing a paragraph, you want to confirm whether the statement is accurate

```python
result = assistant.fact_check(
    "Eph receptors require clustering for full activation"
)

print(f"Support level: {result['support_level']}")  # strong/moderate/weak
print(f"Suggestion: {result['suggestion']}")
```

**Command line**:
```bash
python3 academic_writer.py --check "Eph receptors require clustering"
```

**Output**:
```
✓ Support level: moderate
  Confidence: 0.23
  Suggestion: This statement has some literature support; consider consulting related literature further or adjusting the wording.
```

---

### 3. Find Related Work

**Scenario**: While writing a Related Work section, you need papers on similar research

```python
papers = assistant.find_related_work(
    topic="axon guidance",
    n_papers=10
)

for paper in papers:
    print(f"{paper['authors']} ({paper['year']}): {paper['title']}")
```

**Command line**:
```bash
python3 academic_writer.py --related "axon guidance"
```

---

### 4. Generate Paragraph Support Material

**Scenario**: You are writing a paragraph and need literature supporting the topic

```python
material = assistant.generate_paragraph_support(
    topic="cis-interaction",
    aspect="mechanism"  # Optional: mechanism, function, controversy, evidence
)
print(material)
```

**Output**:
```
### Cis-interaction - Mechanism

[1] Kao and Kania (2011) reported that Ephrin-mediated cis-attenuation 
of Eph receptor signaling is essential for cell sorting...

[2] Carvalho et al. (2006) reported that silencing of EphA3 through 
a cis interaction with ephrinA5...

[Support level: 0.25]
```

---

### 5. Check for Controversy

**Scenario**: You want to confirm whether a claim is controversial, to avoid pitfalls

```python
result = assistant.check_controversial_claim(
    "Eph-ephrin cis-interaction is inhibitory"
)

print(f"Controversy level: {result['controversy_level']}")  # high/moderate/low
print(f"Advice: {result['advice']}")
```

**Command line**:
```bash
python3 academic_writer.py --controversial "cis-interaction is inhibitory"
```

---

## Interactive Writing Mode

The most convenient way is to use interactive mode:

```bash
python3 academic_writer.py
```

Then enter commands:

```
> cite cis interaction inhibits signaling
> check EphB4 promotes tumor growth
> related boundary formation
> support tetramerization controversy
> controversial forward signaling dominates
> export
> quit
```

---

## Real-World Writing Scenarios

### Scenario 1: Needing Citations While Writing the Introduction

```python
# Currently writing: "Previous studies have shown that cis-interactions..."

# Find support
assistant = AcademicWritingAssistant()
citations = assistant.find_references("cis interaction inhibits Eph signaling")

# Format citations
citation_text = assistant.suggest_citation_style(citations, "APA")
print(citation_text)
```

**Output**:
```
Kao and Kania (2011). Ephrin-Mediated cis-Attenuation of Eph Receptor 
Signaling Is Essential...

Carvalho et al. (2006). Silencing of EphA3 through a cis interaction 
with ephrinA5...
```

**Insert into the paper**:
```latex
Previous studies have shown that cis-interactions between Eph receptors 
and their ligands can attenuate receptor signaling (Kao \& Kania, 2011; 
Carvalho et al., 2006), suggesting a regulatory mechanism...
```

---

### Scenario 2: Raising a Controversy While Writing the Discussion

```python
# Want to discuss: "However, the role of cis-interaction remains debated"

result = assistant.check_controversial_claim(
    "cis-interaction is always inhibitory"
)

if result['controversy_level'] in ['high', 'moderate']:
    print("You can write a controversy paragraph!")
    print(f"Supporting papers: {result['supporting_papers']}")
    print(f"Opposing papers: {result['opposing_papers']}")
```

**Paper content**:
```latex
However, the role of cis-interaction in Eph signaling remains debated. 
While some studies suggest that cis-interactions primarily serve an 
inhibitory function (cite supporting papers), others have reported 
context-dependent effects (cite opposing papers). This discrepancy 
may reflect differences in...
```

---

### Scenario 3: Writing the Related Work Section

```python
# Find all related work on a topic
papers = assistant.find_related_work("tetramerization", n_papers=15)

# Group by year
by_year = {}
for p in papers:
    year = p['year']
    if year not in by_year:
        by_year[year] = []
    by_year[year].append(p)

# Output in chronological order
for year in sorted(by_year.keys()):
    print(f"\n{year}:")
    for p in by_year[year]:
        print(f"  - {p['authors']}: {p['title']}")
```

---

### Scenario 4: Real-Time Writing Assistance

```python
# Check as you write
assistant = AcademicWritingAssistant()

paragraph = """
Eph receptor signaling requires receptor clustering for full activation.
"""

# Fact check this statement
result = assistant.fact_check(paragraph)

if result['support_level'] == 'weak':
    print("⚠️  This statement may lack literature support")
    print(result['suggestion'])
    # You may need to revise the wording or add references
```

---

## Integration with LaTeX/Word

### Method 1: Quick Queries from the Command Line

Open a terminal while writing:
```bash
cd /Disk_2/claw_working_dir/ephrin_agentic_rag
python3 academic_writer.py --cite "your claim here"
```

Copy the citations into your paper.

### Method 2: Batch Processing with a Script

Create a `check_paragraphs.py`:
```python
#!/usr/bin/env python3
import sys
sys.path.insert(0, '/Disk_2/claw_working_dir/ephrin_agentic_rag')
from academic_writer import AcademicWritingAssistant

assistant = AcademicWritingAssistant()

paragraphs = [
    "Eph receptors require clustering for activation",
    "Cis-interaction is always inhibitory",
    "Reverse signaling occurs through ephrinB",
]

for p in paragraphs:
    result = assistant.fact_check(p)
    print(f"\n[{p}]")
    print(f"  Support level: {result['support_level']}")
```

### Method 3: Jupyter Notebook

```python
# In a notebook
%run /Disk_2/claw_working_dir/ephrin_agentic_rag/academic_writer.py
assistant = AcademicWritingAssistant()

# Then query at any time
assistant.find_references("your query")
```

---

## Citation Formats

Currently supported:
- **APA**: `Author (Year). Title.`
- **Vancouver**: `Author. Title. Year;`

```python
formatted = assistant.suggest_citation_style(citations, style="APA")
```

---

## Performance Tips

1. **Relevance threshold**: Default is 0.15; set it to 0.20+ if you need stricter results
2. **Number of results**: Default is 10 papers; can be increased or decreased
3. **Multi-hop queries**: Use `--multihop` or `MultiHopRAG` for complex comparisons

---

## Complete Example Script

```python
#!/usr/bin/env python3
"""Example: Writing a paragraph with academic support"""

import sys
sys.path.insert(0, '/Disk_2/claw_working_dir/ephrin_agentic_rag')
from academic_writer import AcademicWritingAssistant

def write_paragraph_with_support(topic, claim):
    """Write a paragraph with literature support"""
    
    assistant = AcademicWritingAssistant()
    
    # 1. Find references
    citations = assistant.find_references(claim, min_relevance=0.20)
    
    # 2. Format citations
    refs = "; ".join([f"{c.authors} et al., {c.year}" for c in citations[:3]])
    
    # 3. Generate the paragraph
    paragraph = f"""
{topic} plays a critical role in Eph-ephrin signaling. 
{claim} ({refs}). 
This mechanism has been implicated in various developmental processes...
"""
    
    print(paragraph)
    
    # 4. Fact check
    check = assistant.fact_check(claim)
    print(f"\n[Support level: {check['support_level']}, Confidence: {check['confidence']:.2f}]")

# Usage
write_paragraph_with_support(
    topic="Cis-interaction",
    claim="Cis-interaction between Eph receptors and ephrins can attenuate receptor signaling"
)
```

---

## FAQ

**Q: Which papers are in the knowledge base?**  
A: 199 Eph/Ephrin-related papers covering 1986-2025.

**Q: What does the relevance score mean?**  
A: 0.15-0.25: moderately relevant; 0.25+: highly relevant; below 0.15: weakly relevant

**Q: How do I update the knowledge base?**  
A: Run `python3 build_knowledge_base.py` to rebuild (if new papers have been added)

**Q: Are non-Eph/Ephrin topics supported?**  
A: Currently only papers in the knowledge base are supported, but you can query anything (you just may not find relevant literature)