#!/usr/bin/env python3
"""
add_papers.py — Add new PDFs to bib_rag index

Pipeline:
  1. Extract PDF → Markdown (pymupdf4llm)
  2. Copy markdown to papers directory (bib_rag's source dir)
  3. Run hierarchical build (incremental — only new/changed files indexed)
  4. Verify with a quick query

Usage:
  python3 -B add_papers.py /path/to/paper.pdf
  python3 -B add_papers.py /path/to/dir/of/pdfs/
  python3 -B add_papers.py paper1.pdf paper2.pdf --skip-extract
  python3 -B add_papers.py /path/to/pdfs/ --batch-size 5

Prerequisites:
  - bge-m3 embedding server OR SentenceTransformers (build script loads CPU model)
  - pymupdf4llm installed (pip install pymupdf4llm)

Options:
  --skip-extract    Skip PDF→MD extraction (use if markdown already exists)
  --batch-size      Build batch size (default 50)
  --papers-dir      Target markdown directory (default: /Disk_bot/paper_lib/My Library/md)
  --verify QUERY    After adding, run a test query to verify
  --dry-run         Show what would be done without executing
"""

import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# ---- Configuration ----
KB_ROOT = Path("/Disk_bot/Eph/bib_rag")
DEFAULT_PAPERS_DIR = Path("/Disk_bot/paper_lib/My Library/md")
BUILD_SCRIPT = KB_ROOT / "src" / "build_hierarchical.py"
QUERY_SCRIPT = KB_ROOT / "src" / "query_bib_rag.py"

# ---- PDF Extraction ----

def extract_pdf_to_md(pdf_path: Path, output_dir: Path) -> Path | None:
    """Extract a single PDF to markdown using pymupdf4llm."""
    try:
        import pymupdf4llm
    except ImportError:
        print("❌ pymupdf4llm not installed. Run: pip install pymupdf4llm")
        return None

    md_filename = pdf_path.stem + ".md"
    md_path = output_dir / md_filename

    print(f"  📄 Extracting: {pdf_path.name}")
    try:
        md_text = pymupdf4llm.to_markdown(str(pdf_path))
        md_path.write_text(md_text, encoding='utf-8')
        print(f"     ✅ → {md_path.name} ({len(md_text):,} chars)")
        return md_path
    except Exception as e:
        print(f"     ❌ Extraction failed: {str(e)[:100]}")
        return None


def find_pdfs(input_paths: list[str]) -> list[Path]:
    """Resolve input paths to a list of PDF files."""
    pdfs = []
    for p in input_paths:
        path = Path(p)
        if path.is_file() and path.suffix.lower() == '.pdf':
            pdfs.append(path)
        elif path.is_dir():
            pdfs.extend(sorted(path.rglob('*.pdf')))
        else:
            print(f"⚠️  Skipping (not a PDF or directory): {p}")
    return pdfs


def find_existing_md(pdf_path: Path, papers_dir: Path) -> Path | None:
    """Check if markdown already exists in papers_dir."""
    # Try exact stem match
    md_name = pdf_path.stem + ".md"
    md_path = papers_dir / md_name
    if md_path.exists():
        return md_path
    # Try fuzzy match (first 40 chars of stem)
    stem_prefix = pdf_path.stem[:40].lower()
    for f in papers_dir.glob('*.md'):
        if stem_prefix in f.stem.lower():
            return f
    return None


# ---- Build ----

def run_build(papers_dir: Path, batch_size: int) -> bool:
    """Run hierarchical build script (incremental)."""
    print(f"\n🔧 Running hierarchical build (incremental)...")
    print(f"   Papers dir: {papers_dir}")
    print(f"   Batch size: {batch_size}")

    cmd = [
        sys.executable, "-B", str(BUILD_SCRIPT),
        "--papers-dir", str(papers_dir),
        "--batch-size", str(batch_size),
    ]

    try:
        result = subprocess.run(cmd, cwd=str(KB_ROOT))
        if result.returncode == 0:
            print("   ✅ Build completed successfully")
            return True
        else:
            print(f"   ❌ Build failed (exit code {result.returncode})")
            return False
    except Exception as e:
        print(f"   ❌ Build error: {e}")
        return False


# ---- Verification ----

def verify_query(query: str) -> bool:
    """Run a test query to verify new papers are searchable."""
    print(f"\n🔍 Verification query: \"{query}\"")
    cmd = [sys.executable, "-B", str(QUERY_SCRIPT), query, "--top", "3"]
    try:
        result = subprocess.run(cmd, cwd=str(KB_ROOT), capture_output=True, text=True, timeout=30)
        output = result.stdout
        if "Found" in output and "relevant" in output.lower():
            print(output[-500:])
            return True
        else:
            print(output[-500:])
            return False
    except Exception as e:
        print(f"   ⚠️  Query failed: {e}")
        return False


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(
        description="Add new PDFs to bib_rag index",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("inputs", nargs='+', help="PDF file(s) or directory containing PDFs")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip PDF→MD extraction (use existing markdown)")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Build batch size (default: 50)")
    parser.add_argument("--papers-dir", default=str(DEFAULT_PAPERS_DIR),
                        help=f"Target markdown directory (default: {DEFAULT_PAPERS_DIR})")
    parser.add_argument("--verify", metavar="QUERY", default=None,
                        help="Run a test query after adding")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without executing")
    args = parser.parse_args()

    papers_dir = Path(args.papers_dir)
    papers_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"📚 bib_rag — Add Papers")
    print(f"   Papers dir: {papers_dir}")
    print(f"   KB root:    {KB_ROOT}")
    print(f"   Time:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    # ---- Step 1: Find PDFs ----
    pdfs = find_pdfs(args.inputs)
    if not pdfs:
        print("❌ No PDF files found.")
        sys.exit(1)
    print(f"Found {len(pdfs)} PDF(s):\n")
    for p in pdfs:
        print(f"  • {p.name}")

    if args.dry_run:
        print("\n--dry-run: stopping here.")
        return

    # ---- Step 2: Extract / locate markdown ----
    print(f"\n{'─'*70}")
    print("Step 1: PDF → Markdown")
    print(f"{'─'*70}")

    md_files_added = []
    md_files_skipped = []

    for pdf in pdfs:
        if args.skip_extract:
            # Look for existing markdown
            existing = find_existing_md(pdf, papers_dir)
            if existing:
                print(f"  ⏭️  Already exists: {existing.name}")
                md_files_skipped.append(existing)
            else:
                print(f"  ⚠️  No markdown found for {pdf.name} (use without --skip-extract)")
            continue

        md_path = extract_pdf_to_md(pdf, papers_dir)
        if md_path:
            md_files_added.append(md_path)
        else:
            print(f"  ❌ Failed: {pdf.name}")

    total_new = len(md_files_added)
    total_skip = len(md_files_skipped)

    if total_new == 0 and total_skip == 0:
        print("\n❌ No markdown files produced. Nothing to add.")
        sys.exit(1)

    print(f"\n📊 Summary: {total_new} new, {total_skip} already indexed")

    # ---- Step 3: Build (incremental) ----
    print(f"\n{'─'*70}")
    print("Step 2: Hierarchical build (incremental)")
    print(f"{'─'*70}")

    success = run_build(papers_dir, args.batch_size)
    if not success:
        print("\n⚠️  Build had issues. Papers may not be fully indexed.")
        sys.exit(1)

    # ---- Step 4: Verify ----
    if args.verify:
        print(f"\n{'─'*70}")
        print("Step 3: Verification")
        print(f"{'─'*70}")
        verify_query(args.verify)

    # ---- Done ----
    print(f"\n{'='*70}")
    print(f"✅ Done! Added {total_new} paper(s) to bib_rag.")
    print(f"   Papers dir: {papers_dir}")
    print(f"   ChromaDB:   {KB_ROOT / 'chroma_db_new'}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()