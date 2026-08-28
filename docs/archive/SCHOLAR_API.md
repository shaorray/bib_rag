# 📚 Academic Writing Usage Guide

## Quick Invocation

### Method 1: Direct Queries from the Command Line (Recommended)

```bash
cd /Disk_2/claw_working_dir/ephrin_agentic_rag

# Find citations
python3 quick_query.py "cis interaction inhibits Eph signaling"

# Interactive writing assistance
python3 academic_writer.py

# Example interaction:
# > cite cis interaction inhibits signaling
# > check Eph receptor requires clustering
# > related axon guidance
# > quit
```

### Method 2: Python API

```python
import sys
sys.path.insert(0, '/Disk_2/claw_working_dir/ephrin_agentic_rag')

from rag_core import SimpleEmbedding, DocumentStore
from agentic_workflow import AgenticRAGWorkflow

# Initialize
doc_store = DocumentStore('ephrin_papers', 
    '/Disk_2/claw_working_dir/ephrin_agentic_rag/chroma_db')
embedder = SimpleEmbedding()

def retriever(query, k=8):
    emb = embedder.embed(query)
    return doc_store.query(emb, n_results=k)

workflow = AgenticRAGWorkflow(retriever)

# Query
result = workflow.run("cis interaction mechanism")
print(result['answer'])
```

---

## 📝 Paper Writing Scenarios

### Scenario 1: Finding Citations

**Problem**: You wrote "Cis-interaction inhibits Eph receptor signaling" and need citations

**Invocation**:
```bash
python3 academic_writer.py --cite "cis interaction inhibits Eph signaling"
```

**Output**:
```
[1] Kao and Kania (2011) - rel: 0.240
    Ephrin-Mediated cis-Attenuation of Eph Receptor Signaling...

[2] Carvalho et al. (2006) - rel: 0.193
    Silencing of EphA3 through a cis interaction...
```

**Usage in the paper**:
```latex
Cis-interaction inhibits Eph receptor signaling 
(Kao \& Kania, 2011; Carvalho et al., 2006).
```

---

### Scenario 2: Fact Checking

**Problem**: You are unsure whether "Eph receptors require clustering for activation" is accurate

**Invocation**:
```bash
python3 academic_writer.py --check "Eph receptors require clustering"
```

**Output**:
```
✓ Support level: moderate
  Suggestion: This statement has some literature support; consider consulting related literature further
```

---

### Scenario 3: Related Work

**Problem**: Writing a Related Work section and need literature on "axon guidance"

**Invocation**:
```bash
python3 academic_writer.py --related "axon guidance"
```

**Output**:
```
[1] Bush and Soriano (2009): Ephrin-B1 regulates axon guidance...
[2] Williams et al. (2003): Ephrin-B2 and EphB1 Mediate Retinal Axon Divergence...
```

---

### Scenario 4: Checking for Controversy

**Problem**: You want to discuss whether "cis-interaction is always inhibitory" is controversial

**Invocation**:
```bash
python3 academic_writer.py --controversial "cis-interaction is always inhibitory"
```

**Output**:
```
⚖️ Controversy level: moderate
  Supporting papers: 12
  Opposing papers: 5
  Suggestion: This claim is somewhat controversial; consider mentioning different viewpoints
```

**Usage in the paper**:
```latex
However, whether cis-interaction is exclusively inhibitory 
remains debated (cite supporting; cite opposing)...
```

---

### Scenario 5: Generating Paragraph Support

**Problem**: You need supporting material for a paragraph about "tetramerization"

**Invocation**:
```bash
python3 academic_writer.py --support "tetramerization"
```

**Output**:
```
### Tetramerization - Mechanism

[1] Falivelli et al. (2013) reported that tetrameric Eph receptor 
complexes are essential for full activation...

[Support level: 0.22]
```

---

## 🔧 Integration into the LaTeX Workflow

### Method 1: Makefile Shortcuts

Create a `Makefile` in your paper directory:

```makefile
# Find citations
cite:
	@cd /Disk_2/claw_working_dir/ephrin_agentic_rag && \
	python3 quick_query.py "$(filter-out $@,$(MAKECMDGOALS))"

# Check paragraphs
%:
	@:
```

Usage:
```bash
make cite cis interaction
```

### Method 2: VS Code Tasks

`.vscode/tasks.json`:
```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Find Citation",
      "type": "shell",
      "command": "cd /Disk_2/claw_working_dir/ephrin_agentic_rag && python3 quick_query.py '${input:claim}'",
      "problemMatcher": []
    }
  ]
}
```

### Method 3: Emacs/Vim Shortcuts

Emacs (init.el):
```elisp
(defun eph-cite (claim)
  "Find citation for claim"
  (interactive "sClaim: ")
  (shell-command
   (format "cd /Disk_2/claw_working_dir/ephrin_agentic_rag && python3 quick_query.py '%s'" claim)
   "*Eph Citations*"))

(global-set-key "\C-c\C-e" 'eph-cite)
```

---

## 📊 Batch Writing Assistance

Create a `write_helper.py`:

```python
#!/usr/bin/env python3
import sys
sys.path.insert(0, '/Disk_2/claw_working_dir/ephrin_agentic_rag')

from academic_writer import AcademicWritingAssistant

assistant = AcademicWritingAssistant()

# Claims to be checked in the paper
claims = [
    "Eph receptors require clustering for activation",
    "Cis-interaction is inhibitory",
    "Reverse signaling occurs through ephrinB",
    "Eph-ephrin signaling maintains tissue boundaries",
]

print("=== Paper Claim Verification Report ===\n")

for claim in claims:
    result = assistant.fact_check(claim)
    print(f"✓ {claim}")
    print(f"  Support level: {result['support_level']}")
    print(f"  Confidence: {result['confidence']:.2f}\n")
```

Run it:
```bash
python3 write_helper.py
```

---

## 💡 Usage Tips

### 1. Relevance Threshold

| Threshold | Meaning | Advice |
|------|------|------|
| > 0.25 | Highly relevant | Can be cited directly |
| 0.15-0.25 | Moderately relevant | Use with context in mind |
| < 0.15 | Weakly relevant | Consider revising the query terms |

### 2. Query Term Optimization

❌ Bad: `"interaction"`  
✅ Better: `"Eph ephrin cis interaction"`

❌ Bad: `"signaling"`  
✅ Better: `"reverse signaling ephrinB"`

### 3. Combined Queries

For complex claims, use multi-hop reasoning:
```bash
python3 query_interface.py --multihop "Compare cis and trans signaling"
```

---

## 📚 Knowledge Base Coverage

| Topic | Coverage |
|------|--------|
| Cis-interaction | 8 papers |
| Reverse signaling | 12 papers |
| Tetramerization | 5 papers |
| Axon guidance | 16 papers |
| Cancer/Metastasis | 24 papers |
| Cell segregation | 14 papers |

**Total**: 199 papers, 388 document chunks

---

## 🐛 Troubleshooting

**Problem**: "No documents found"  
**Solution**: Use more specific technical terms, e.g. `"EphB4 ephrinB2"` instead of `"protein interaction"`

**Problem**: Confidence is too low  
**Solution**: 
1. Check whether the query contains technical terms
2. Try synonyms, e.g. `"cis"` → `"cis interaction"` → `"cis attenuation"`
3. Use multi-hop mode to decompose complex queries

---

## 🎯 Best Practices

1. **Check as you write**: Look up citations every time you write a claim
2. **Fact check**: Verify uncertain statements with `--check` first
3. **Record citations**: Use `export` to save the citations you find
4. **Flag controversies**: Mark controversial claims with `--controversial`
5. **Related Work**: Periodically refresh your literature review with `--related`

---

## 📖 Full Documentation

- `README.md` - System architecture overview
- `USAGE.md` - Detailed usage instructions
- `ACADEMIC_USAGE.md` - Dedicated academic writing guide
- `QUICK_START.md` - Quick start