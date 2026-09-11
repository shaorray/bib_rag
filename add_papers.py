#!/usr/bin/env python3
"""
add_papers.py — Add new PDFs to bib_rag index

Pipeline:
  1. Extract PDF → Markdown (pymupdf4llm)
  2. Copy markdown to papers directory (bib_rag's source dir)
  3. Index each markdown via src/index_single_paper.py (llama-server embedding)
  4. Verify with a quick query

Usage:
  python3 -B add_papers.py /path/to/paper.pdf
  python3 -B add_papers.py /path/to/dir/of/pdfs/
  python3 -B add_papers.py paper1.pdf paper2.pdf --skip-extract
  python3 -B add_papers.py /path/to/pdfs/ --batch-size 5

Prerequisites:
  - llama-server embedding endpoint on port 8081 (bge-m3)
  - pymupdf4llm installed (pip install pymupdf4llm)

Options:
  --skip-extract    Skip PDF→MD extraction (use if markdown already exists)
  --batch-size      Build batch size (default 50)
  --papers-dir      Target markdown directory (default: <active library>/md)
  --verify QUERY    After adding, run a test query to verify
  --dry-run         Show what would be done without executing
  --no-enrich       Skip metadata enrichment (Crossref/PubMed backfill)
  --enrich-offline  Enrichment regex-scan only, no network calls
"""

import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# ---- Configuration (Multi-KB aware) ----
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from kb_config import get_config, parse_kb_arg

# Strip --kb from argv before argparse
_argv = parse_kb_arg()

_CFG = get_config()
CODE_ROOT = Path(_CFG["code_root"])
KB_ROOT = Path(_CFG["data_root"])  # data dir (chroma, parent_store) for display
# Per-library papers dir (setup_library scaffolds <library>/md/); the shared
# Zotero-derived corpus remains available via explicit --papers-dir.
DEFAULT_PAPERS_DIR = KB_ROOT / "md"
QUERY_SCRIPT = CODE_ROOT / "src" / "query_bib_rag.py"

# Indexing is done by src/index_single_paper.py (uses llama-server embedding on
# port 8081, bypassing the broken SentenceTransformers import). build_hierarchical.py
# is retired — it depended on SentenceTransformer which no longer imports.
from index_single_paper import index_paper

# Metadata enrichment (Crossref/PubMed backfill of doi/pmid/title/journal/authors)
# at add-time. Imported lazily in enrich_step() so --no-enrich runs never pay
# the meta_audit import cost. See src/enrich_meta.py.
ENRICH_ONLINE = True   # --no-enrich sets False
ENRICH_NETWORK = True  # --enrich-offline sets False (regex-scan only)

def enrich_step(md_files: list, online: bool = True) -> dict:
    """Fill missing metadata (DOI/PMID/...) in each md via Crossref/PubMed.

    Writes a front-matter block (Title:/Authors:/.../doi:/PMID:/PMCID:) into
    the md BEFORE indexing, so chunking.extract_meta and hybrid_search pick
    the fields up without any changes on their side. Returns
    {md_filename: meta_dict} for run_build to pass into index_paper as
    meta_override (the chunker alone misses front-matter titles).
    Never raises — a failed lookup leaves the regex-scanned values in place
    and indexing proceeds.
    """
    out: dict = {}
    if not md_files:
        return out
    try:
        from enrich_meta import enrich_md, local_meta, upsert_frontmatter
    except ImportError:
        print("   ⚠️ enrich_meta unavailable — indexing without enrichment")
        return out
    if not online:
        # offline mode: local regex scan only (identifiers already in the text)
        for md in md_files:
            try:
                meta = local_meta(md)
                upsert_frontmatter(md, meta)
                out[md.name] = meta
            except Exception as e:
                print(f"   ⚠️ offline-enrich failed for {md.name}: {e}")
        return out
    for md in md_files:
        try:
            meta = enrich_md(md)
            out[md.name] = meta
            gained = [k for k in ("doi", "pmid", "pmcid", "title", "year", "journal", "authors")
                      if meta.get(k)]
            print(f"   🧪 enriched: {', '.join(gained) if gained else 'nothing found'}")
        except Exception as e:
            print(f"   ⚠️ enrich failed for {md.name} (continuing): {e}")
    return out

# ---- PDF Extraction ----

def _extract_one_pdf_subprocess(pdf_path: Path, md_path: Path) -> tuple[bool, str]:
    """Run pymupdf4llm extraction in an isolated subprocess.

    pymupdf 1.27's ONNX layout model (BoxRFDGNN) SEGFAULTS on some pages
    (observed 2026-09-10: Cell Research open-access PDF, JMG 2006 scan) —
    a native crash that cannot be caught with try/except in-process.
    Running per-PDF in a subprocess turns a fatal batch event into a
    one-file failure. ~200ms overhead per file, worth it for robustness.
    Returns (ok, diagnostic).
    """
    import subprocess
    # pymupdf4llm 1.27 layout mode runs an ONNX model that segfaults in this
    # environment (onnxruntime × pymupdf._mupdf interplay, 2026-09-10). The
    # classic pymupdf_rag path (use_layout(False)) extracts the same PDFs
    # without the ONNX dependency and without crashing.
    code = (
        "import pymupdf4llm, pathlib\n"
        "pymupdf4llm.use_layout(False)\n"
        f"md = pymupdf4llm.to_markdown({str(pdf_path)!r})\n"
        f"pathlib.Path({str(md_path)!r}).write_text(md, encoding='utf-8')\n"
    )
    try:
        r = subprocess.run(
            [sys.executable, "-B", "-c", code],
            capture_output=True, text=True, timeout=300,
            env={**os.environ, "OMP_NUM_THREADS": "1"},
        )
    except subprocess.TimeoutExpired:
        return False, "timeout after 300s"
    if r.returncode == 0 and md_path.exists() and md_path.stat().st_size > 0:
        return True, ""
    # negative returncode = killed by signal (segfault = -11); NEVER retry
    # in-process — the same native crash would take down the whole batch
    sig = f", killed by signal {-r.returncode}" if r.returncode < 0 else ""
    err = (r.stderr or "").strip().splitlines()
    err = err[-1][:120] if err else ""
    return False, f"subprocess exit {r.returncode}{sig} {err}"


def extract_pdf_to_md(pdf_path: Path, output_dir: Path) -> Path | None:
    """Extract a single PDF to markdown using pymupdf4llm (isolated subprocess).

    A crash-prone PDF fails alone; the batch continues. No in-process
    fallback by design (see _extract_one_pdf_subprocess docstring).
    """
    md_filename = pdf_path.stem + ".md"
    md_path = output_dir / md_filename

    print(f"  📄 Extracting: {pdf_path.name}")
    try:
        import pymupdf4llm  # noqa: F401  (availability check only)
    except ImportError:
        print("❌ pymupdf4llm not installed. Run: pip install pymupdf4llm")
        return None

    ok, diag = _extract_one_pdf_subprocess(pdf_path, md_path)
    if not ok:
        print(f"     ❌ Extraction failed in isolation ({diag}) — skipping file")
        if md_path.exists():
            md_path.unlink()   # remove any partial output
        return None
    size = md_path.stat().st_size
    if size < 500:
        print(f"     ⚠️ Suspiciously small ({size}B) — extraction likely failed")
        md_path.unlink()
        return None
    print(f"     ✅ → {md_path.name} ({size:,} bytes)")
    return md_path


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

def run_build(md_files: list, batch_size: int, meta_map: dict = None) -> bool:
    """Index each markdown file via src/index_single_paper.index_paper().

    Uses the llama-server embedding endpoint (port 8081) — the same bge-m3 model
    the retired build_hierarchical.py used, but without the broken
    SentenceTransformers import.

    meta_map: optional {md_filename: meta} from enrich_step — passed to
    index_paper so enrichment fields beat the chunker's filename fallbacks.
    """
    meta_map = meta_map or {}
    print(f"\n🔧 Indexing {len(md_files)} markdown file(s) via index_single_paper...")
    ok = 0
    for i, md in enumerate(md_files, 1):
        print(f"\n[{i}/{len(md_files)}]")
        try:
            if index_paper(md, meta_override=meta_map.get(Path(md).name)):
                ok += 1
        except Exception as e:
            print(f"   ❌ Index failed: {e}")
    print(f"\n   ✅ Indexed {ok}/{len(md_files)} successfully")
    return ok > 0


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
    parser.add_argument("--no-enrich", action="store_true",
                        help="Skip Crossref/PubMed metadata enrichment")
    parser.add_argument("--enrich-offline", action="store_true",
                        help="Enrichment regex-scan only (no network); implies enrichment on")
    args = parser.parse_args()

    global ENRICH_ONLINE, ENRICH_NETWORK
    if args.no_enrich:
        ENRICH_ONLINE = False
    if args.enrich_offline:
        ENRICH_ONLINE, ENRICH_NETWORK = True, False

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

    # ---- Step 2a: Metadata enrichment (before indexing) ----
    # All new + skipped markdown files (skipped = already extracted MD)
    md_to_index = md_files_added + md_files_skipped
    meta_map: dict = {}
    if ENRICH_ONLINE:
        print(f"\n{'─'*70}")
        print("Step 2a: Metadata enrichment (Crossref/PubMed)"
              + (" — offline regex-scan only" if not ENRICH_NETWORK else ""))
        print(f"{'─'*70}")
        meta_map = enrich_step(md_to_index, online=ENRICH_NETWORK)
    else:
        print("\n⏭️  Metadata enrichment skipped (--no-enrich)")

    # ---- Step 3: Index (via index_single_paper) ----
    print(f"\n{'─'*70}")
    print("Step 2: Indexing markdown files")
    print(f"{'─'*70}")

    success = run_build(md_to_index, args.batch_size, meta_map=meta_map)
    if not success:
        print("\n⚠️  Indexing had issues. Papers may not be fully indexed.")
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