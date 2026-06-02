#!/usr/bin/env python3
"""
Add new papers to bib_rag index.

Workflow:
1. Extract PDFs → Markdown (if not already extracted)
2. Run hierarchical build to index new Markdown files

Usage:
    python3 -B add_papers.py /path/to/new/papers/
    python3 -B add_papers.py /path/to/new/papers/ --skip-extract  # if already markdown
    python3 -B add_papers.py /path/to/new/papers/ --batch-size 10

Requirements:
    - llama-server bge-m3 on port 8081 (for embeddings)
    - pymupdf4llm installed (for PDF extraction, unless --skip-extract)
"""

import os
import sys
import subprocess
from pathlib import Path

# Default directories
PAPER_LIB = Path("/Disk_bot/paper_lib/My Library")
PDF_DIR = PAPER_LIB / "pdf"
MD_DIR = PAPER_LIB / "md"
BIB_RAG_DIR = Path("/Disk_bot/Eph/bib_rag")


def run_extract(pdf_path):
    """Extract PDFs to markdown using pymupdf4llm."""
    import fitz
    import pymupdf4llm
    
    print(f"📄 Extracting: {pdf_path.name}")
    
    try:
        doc = fitz.open(str(pdf_path))
        md = pymupdf4llm.to_markdown(doc, write_images=False)
        doc.close()
        
        # Save to md dir with same relative path
        rel_path = pdf_path.relative_to(PDF_DIR)
        md_file = MD_DIR / rel_path.with_suffix('.md')
        md_file.parent.mkdir(parents=True, exist_ok=True)
        md_file.write_text(md, encoding='utf-8')
        
        print(f"   ✅ Saved: {md_file}")
        return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


def extract_new_pdfs(source_dir):
    """Find and extract PDFs not yet in md/."""
    pdf_files = list(Path(source_dir).rglob('*.pdf'))
    pdf_files = [p for p in pdf_files if not p.name.startswith('._')]
    
    print(f"🔍 Found {len(pdf_files)} PDF files")
    
    extracted = 0
    for pdf_path in sorted(pdf_files):
        # Check if already extracted
        rel_path = pdf_path.relative_to(source_dir)
        expected_md = MD_DIR / rel_path.with_suffix('.md')
        
        if expected_md.exists():
            print(f"   ⏭️  Skipping (already extracted): {pdf_path.name}")
            continue
        
        # Copy PDF to library if not already there
        target_pdf = PDF_DIR / rel_path
        if not target_pdf.exists():
            target_pdf.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(str(pdf_path), str(target_pdf))
            print(f"   📋 Copied to library: {target_pdf}")
        
        if run_extract(target_pdf):
            extracted += 1
    
    print(f"\n✅ Extracted {extracted} new papers")
    return extracted


def build_index(batch_size=10):
    """Run hierarchical build to index new markdown files."""
    print(f"\n🏗️  Building hierarchical index (batch_size={batch_size})...")
    
    build_script = BIB_RAG_DIR / "build_hierarchical_gpu.py"
    
    result = subprocess.run(
        [sys.executable, "-B", str(build_script), "--batch-size", str(batch_size)],
        cwd=str(BIB_RAG_DIR),
        capture_output=False,
        text=True
    )
    
    return result.returncode == 0


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Add new papers to bib_rag index")
    parser.add_argument("source", nargs="?", default=str(PDF_DIR),
                       help="Source directory with new PDFs (default: /Disk_bot/paper_lib/My Library/pdf)")
    parser.add_argument("--skip-extract", action="store_true",
                       help="Skip PDF extraction (use if already in md/)")
    parser.add_argument("--batch-size", type=int, default=10,
                       help="Build batch size (default: 10)")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("📚 bib_rag — Add New Papers")
    print("=" * 70)
    
    # Step 1: Extract PDFs
    if not args.skip_extract:
        print(f"\n📄 Step 1: Extracting PDFs from {args.source}")
        print("-" * 70)
        
        try:
            import fitz
            import pymupdf4llm
        except ImportError:
            print("❌ pymupdf4llm not installed. Install with:")
            print("   pip install pymupdf4llm")
            print("\n   Or use --skip-extract if markdown files already exist")
            return 1
        
        extract_new_pdfs(args.source)
    else:
        print("\n⏭️  Skipping extraction (using existing markdown files)")
    
    # Step 2: Build index
    print(f"\n🏗️  Step 2: Building hierarchical index")
    print("-" * 70)
    
    # Check embedding server
    import requests
    try:
        resp = requests.get("http://localhost:8081/health", timeout=5)
        if resp.status_code == 200:
            print("✅ Embedding server (port 8081) is running")
        else:
            print("⚠️  Embedding server responded but not healthy")
    except Exception:
        print("❌ Embedding server (port 8081) is not running!")
        print("   Start it with: bash /Disk_bot/start_llama_bge_m3.sh")
        return 1
    
    if build_index(args.batch_size):
        print("\n✅ Index build complete!")
    else:
        print("\n⚠️  Index build may have issues. Check logs above.")
    
    # Summary
    print(f"\n{'=' * 70}")
    print("📊 Summary")
    print(f"{'=' * 70}")
    
    from src.parent_store_manager import ParentStoreManager
    import chromadb
    
    pm = ParentStoreManager()
    stats = pm.get_stats()
    print(f"Parent chunks: {stats['total_parents']}")
    
    client = chromadb.PersistentClient(path=str(BIB_RAG_DIR / "chroma_db_new"))
    coll = client.get_collection("bib_rag_papers")
    print(f"Child embeddings: {coll.count()}")
    
    print(f"\n✅ All done! New papers are now searchable via agentic_query.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
