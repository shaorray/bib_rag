# bib_agentic_rag

Agentic RAG system for academic paper bibliography management.

## Features

- **Incremental paper addition**: Add papers to ChromaDB without rebuilding
- **bge-m3 embeddings**: 1024-dim multilingual embeddings via local server
- **Smart chunking**: Paragraph-level, skips reference sections
- **Section-aware**: Extracts Abstract, Introduction, Methods, Results, Discussion, Conclusion
- **Duplicate detection**: Content-hash based, auto-skips unchanged papers
- **Batch processing**: 100 papers/batch with automatic persist

## Architecture

```
papers_dir/
  ├── md/              # pymupdf4llm extracted markdown
  └── ...

bib_agentic_rag/
  ├── src/
  │   └── incremental_add_papers.py   # Main builder
  ├── data/
  │   └── incremental_metadata.json   # Paper registry
  └── chroma_db_new/                  # ChromaDB vector store
```

## Requirements

- Python 3.10+
- ChromaDB
- LangChain
- bge-m3 embedding server (port 11435)

## Usage

```bash
# 1. Start bge-m3 server
python3 /home/rui/.openclaw/agents/local-embedding/local_embedding_server.py \
  --port 11435 --model /Disk_bot/models/bge-m3

# 2. Dry run (preview)
cd src
python3 incremental_add_papers.py /Disk_bot/paper_lib/md --dry-run

# 3. Actual build
python3 incremental_add_papers.py /Disk_bot/paper_lib/md

# 4. Add more papers later (incremental)
python3 incremental_add_papers.py /Disk_bot/paper_lib/md/signaling
```

## License

MIT
