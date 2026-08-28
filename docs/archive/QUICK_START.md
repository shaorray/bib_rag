# Agentic RAG Quick Start

## Knowledge Base Status

✅ **Build complete** - 199 Eph/Ephrin papers, 388 document chunks

## Use It Now

```bash
cd /Disk_2/claw_working_dir/ephrin_agentic_rag

# Interactive query
python3 query_interface.py

# Single query example
python3 query_interface.py -q "cis interaction Eph"

# Multi-hop reasoning
python3 query_interface.py --multihop -q "Compare forward and reverse signaling"
```

## Example Query

```
❓ Query: Eph receptor evolution

📋 Answer:
[1] # EPH RECEPTOR SIGNALLING CASTS
   Source: Pasquale 2005

[2] # Eph receptor function is modulated by heterooligomerization
   Source: Janes et al. 2011
...

📊 Confidence: 0.32
🔍 Retrieved: 8 docs
```

## File Descriptions

| File | Purpose |
|------|---------|
| `build_knowledge_base.py` | Rebuild the knowledge base |
| `query_interface.py` | Interactive queries |
| `rag_core.py` | Core RAG components |
| `agentic_workflow.py` | Agentic RAG logic |
| `chroma_db/ephrin_papers.pkl` | Vector database (3MB) |

## Features

- ✅ Self-RAG: Self-assessment of relevance
- ✅ CRAG: Query rewriting and retry
- ✅ Multi-hop: Complex query decomposition
- ✅ Persistent storage
- 📦 199 papers / 388 document chunks