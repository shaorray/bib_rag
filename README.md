# bib_rag

Academic bibliography RAG system for evidence-based writing.

## Status

- **1,643 papers** indexed
- **18,239 chunks** embedded (paragraph-level)
- **Model**: bge-m3 (1024-dim, GPU via llama-server port 8081)

## Quick Start

### 1. Start embedding server

```bash
# llama-server bge-m3 must be running on port 8081
nohup ~/.local/bin/llama-server \
  -m /Disk_bot/models/embeddings/bge-m3-Q4_K_M.gguf \
  --port 8081 -c 8192 --embedding -ngl 999 \
  > /tmp/llama-bge-m3.log 2>&1 &
```

## Usage

```bash
cd /Disk_bot/Eph/bib_rag

# Quick search
python3 -B query_bib_rag.py "cis interaction mechanism"

# Find citations for a claim
python3 -B query_bib_rag.py --cite "Eph receptors promote tumor suppression" --top 3

# Write a paragraph with citations (uses -B to ensure fresh code)
python3 -B bib_rag_writer.py "Eph receptor signaling regulates cell segregation" --top 5 --style APA --output /path/to/output.odt

# Insert bibliography from inline citations in .odt
python3 -B bib_rag_zotero_odt_proper.py /path/to/file.odt APA
```

### Writing a Paragraph

```bash
python3 -B bib_rag_writer.py "your topic sentence" [--top N] [--style APA|Vancouver|Nature] [--output path.odt]
```

Example:
```bash
python3 -B bib_rag_writer.py "Eph receptor signaling regulates cell segregation through repulsion" \
  --top 5 --style APA --output /Disk_bot/writing/synthesis_eph.odt
```

This will:
1. Search bib_rag for relevant passages
2. Analyze key claims and terms
3. Synthesize a paragraph with proper in-text citations
4. Add formatted references at the end
5. Save to .odt

### 3. Add new papers (incremental)

```bash
cd src

# CPU fallback (slow but reliable)
python3 build_stable.py /path/to/new_papers -b 50

# GPU build (requires llama-server on 8081)
python3 build_gpu.py /path/to/new_papers -b 50
```

## File Structure

```
bib_rag/
├── chroma_db_new/          ← Working vector database (701MB, 18,239 chunks)
├── data/
│   ├── build_checkpoint.json      ← Resume point (empty after completion)
│   └── incremental_metadata.json  ← Paper registry (16,460 entries)
├── src/
│   ├── build_gpu.py        ← GPU build script (current, uses llama-server bge-m3)
│   ├── build_stable.py     ← CPU fallback build (sentence-transformers)
│   └── requirements.txt   ← Python deps
├── query_bib_rag.py        ← Quick query interface
├── README.md               ← This file
└── archive/                ← Obsolete scripts (safe to ignore)
```

## Writing Workflow

1. **Draft a claim**: Write your assertion
2. **Find evidence**: `python3 query_bib_rag.py --cite "your claim"`
3. **Verify**: Check the returned DOI + excerpt
4. **Cite**: Add to your paper's reference list

## License

MIT
