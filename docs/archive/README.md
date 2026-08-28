# Eph/Ephrin Agentic RAG Knowledge Base

🧠 An intelligent knowledge base system built on 180+ Eph/Ephrin papers

## Features

### Core RAG Features
- ✅ **Vector retrieval**: Sentence-Transformers embeddings + cosine similarity retrieval
- ✅ **Smart chunking**: Semantics-aware chunking that preserves contextual coherence
- ✅ **Metadata management**: Automatic extraction of paper titles, authors, and years

### Agentic Features
- 🔄 **Self-RAG**: Self-assessment of the relevance of retrieved documents
- 🔧 **CRAG (Corrective RAG)**: Automatically rewrites the query and retries when retrieval fails
- 🎯 **Adaptive routing**: Chooses between retrieval and direct answering based on question complexity
- 🕸️ **Multi-hop reasoning**: Supports complex queries requiring multi-step reasoning

## Architecture

```
User Query
    ↓
[Analyze] - Is retrieval needed?
    ↓ (Yes)              ↓ (No)
[Retrieve]              [Direct Answer]
    ↓
[Grade] - Relevance assessment (Self-RAG)
    ↓
┌─────────┬──────────┐
↓ (High)  ↓ (Medium) ↓ (Low)
[Generate] [Rewrite]  [Rewrite]
              ↓
         [Retrieve]  ←  CRAG loop (max 2 retries)
              ↓
         [Generate]
```

## Quick Start

### 1. Install Dependencies

```bash
cd /Disk_2/claw_working_dir/ephrin_agentic_rag
pip install -r requirements.txt
```

### 2. Build Knowledge Base

```bash
python3 build_knowledge_base.py
```

This will:
- Load 194 markdown papers from `/Disk_2/claw_working_dir/Ephrin_papers/review_output/markdown_round2`
- Generate embeddings (simple bag-of-words fallback when network unavailable)
- Create vector database at `./chroma_db`
- Run test queries
- Index ~5,700 document chunks (~1.9M words)

### 3. Interactive Query

```bash
python3 query_interface.py
```

Commands:
- Type your question directly
- `/multihop <query>` - Use multi-hop reasoning
- `/stats` - Show knowledge base statistics
- `/history` - Show query history
- `/exit` - Exit

### 4. Batch Queries

```bash
# Create queries.txt with one question per line
echo "What is cis-interaction in Eph signaling?" > queries.txt
echo "How does ephrin-B1 regulate axon guidance?" >> queries.txt

# Run batch
python3 query_interface.py --batch queries.txt
```

## File Structure

```
ephrin_agentic_rag/
├── build_knowledge_base.py    # Knowledge base build script
├── rag_core.py                # Core RAG components (embeddings, store, pipeline)
├── agentic_workflow.py        # Agentic RAG workflow
├── query_interface.py         # Interactive query interface
├── requirements.txt           # Dependencies
├── README.md                  # This file
├── chroma_db/                 # Vector database (created automatically)
├── paper_metadata.json        # Paper metadata (created automatically)
└── query_history.json         # Query history (created automatically)
```

## Key Components

### 1. DocumentStore
A simple filesystem-based vector store that supports:
- Persistent storage (Pickle)
- Cosine similarity retrieval
- Metadata management

### 2. AgenticRAGWorkflow
A LangGraph-based workflow:
- **analyze**: Analyze query complexity
- **retrieve**: Retrieve documents
- **grade**: Assess relevance
- **rewrite**: CRAG query rewriting
- **generate**: Generate answers

### 3. MultiHopRAG
Supports complex multi-step reasoning:
- Query decomposition
- Parallel processing of sub-queries
- Result synthesis

## Query Rewriting Strategy

CRAG rewriting uses the following strategies:
1. **Keyword expansion**: Add synonyms (e.g. "Eph" → "Eph receptor")
2. **Boolean OR**: Broaden retrieval scope
3. **Context addition**: Add background terms such as "Eph ephrin signaling"

## Customization

### Using a different embedding model

Modify in `rag_core.py`:
```python
self.embedder = SimpleEmbedding(model_name="your-model")
```

### Adjusting chunk size

Modify in `build_knowledge_base.py`:
```python
chunks = self._smart_chunk(content, chunk_size=1000, overlap=200)
```

### Adding LLM integration

Configure in `agentic_workflow.py`:
```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini")
```

## Performance

- **Embedding**: all-MiniLM-L6-v2 (384 dimensions)
- **Retrieval**: ~50ms per query (in-memory)
- **CRAG Retries**: max 2 attempts
- **Multi-hop**: Supports up to 4 sub-queries

## Troubleshooting

### No documents found
```bash
# Check if markdown files exist
ls /Disk_2/claw_working_dir/Ephrin_papers/review_output/markdown/

# Rebuild knowledge base
rm -rf /Disk_2/claw_working_dir/ephrin_agentic_rag/chroma_db
python3 build_knowledge_base.py
```

### Low relevance scores
- Check if query contains specific keywords
- Try `/multihop` for complex questions
- Use more specific terminology

## References

- **Self-RAG**: [Learning to Retrieve, Generate, and Critique](https://arxiv.org/abs/2310.11511)
- **CRAG**: [Corrective RAG](https://arxiv.org/abs/2401.15884)
- **LangGraph**: https://langchain-ai.github.io/langgraph/

## License

MIT License - Academic/Research Use
ademic/Research Use
e, Generate, and Critique](https://arxiv.org/abs/2310.11511)
- **CRAG**: [Corrective RAG](https://arxiv.org/abs/2401.15884)
- **LangGraph**: https://langchain-ai.github.io/langgraph/

## License

MIT License - Academic/Research Use