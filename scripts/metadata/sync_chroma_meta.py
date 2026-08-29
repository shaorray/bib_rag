#!/usr/bin/env python3
"""
sync_chroma_meta.py — push parent_store meta fixes into chroma chunk metadata.

meta_audit.py fixes biblio fields (title/authors/year/journal/doi/pmid/pmcid)
in parent_store/*.json chunk `meta` blocks, but chroma chunk metadata keeps
the stale pre-fix values (e.g. filename-like titles, empty authors). This
script does a single paged scan of the chroma collection and batch-updates
ONLY the seven biblio fields from the current parent_store values — topics,
article_type, section, wc, hash, idx etc. are left untouched.

Single-pass scan + batched col.update() (same write pattern as apply_tags.py).

Usage:
    <name>-rag python3 scripts/metadata/sync_chroma_meta.py [--dry-run] [--batch 2000]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# --- repo bootstrap (works from any cwd) ---------------------------------
_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

try:
    from kb_config import get_config
except ImportError:  # direct src/ on path
    from kb_config import get_config  # noqa: F811

FIELDS = ("title", "authors", "year", "journal", "doi", "pmid", "pmcid")


def load_parent_meta(parent_dir: Path) -> dict:
    """source.md -> fixed meta dict (chunk[0].meta of each parent file)."""
    out = {}
    for p in parent_dir.glob("*.json"):
        try:
            chunks = json.loads(p.read_text())
            if isinstance(chunks, list) and chunks and isinstance(chunks[0].get("meta"), dict):
                out[chunks[0]["meta"].get("source") or chunks[0].get("source", "")] = chunks[0]["meta"]
        except (json.JSONDecodeError, OSError, IndexError, AttributeError):
            continue
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="only report, don't write")
    ap.add_argument("--batch", type=int, default=2000, help="chunks per update call")
    args = ap.parse_args()

    cfg = get_config()
    chroma_path = cfg["chroma_path"]
    collection_name = cfg["collection_name"]
    parent_dir = Path(cfg["parent_store_dir"])

    pmeta = load_parent_meta(parent_dir)
    print(f"loaded parent meta for {len(pmeta)} sources from {parent_dir}")

    import chromadb
    col = chromadb.PersistentClient(path=str(chroma_path)).get_collection(collection_name)
    total = col.count()
    print(f"chroma collection '{collection_name}': {total} chunks")

    t0 = time.time()
    updated = skipped_scan = missing_src = 0
    field_counts = {f: 0 for f in FIELDS}
    batch_ids, batch_metas = [], []

    def flush():
        nonlocal updated, batch_ids, batch_metas
        if not batch_ids:
            return
        if not args.dry_run:
            col.update(ids=batch_ids, metadatas=batch_metas)
        updated += len(batch_ids)
        batch_ids, batch_metas = [], []

    offset, page = 0, 5000
    while True:
        r = col.get(include=["metadatas"], limit=page, offset=offset)
        ids, metas = r["ids"] or [], r["metadatas"] or []
        if not ids:
            break
        for cid, m in zip(ids, metas):
            src = m.get("source", "")
            pm = pmeta.get(src)
            if pm is None:
                missing_src += 1
                continue
            patch = {}
            for f in FIELDS:
                new = str(pm.get(f, "") or "")
                old = m.get(f)
                old_s = str(old) if old is not None else ""
                if new != old_s:
                    patch[f] = new
                    field_counts[f] += 1
            if patch:
                newm = dict(m)
                newm.update(patch)
                batch_ids.append(cid)
                batch_metas.append(newm)
                if len(batch_ids) >= args.batch:
                    flush()
        offset += len(ids)
        print(f"...scanned {offset}/{total} ({time.time()-t0:.0f}s)", file=sys.stderr)
    flush()

    mode = "DRY-RUN (no writes)" if args.dry_run else "APPLIED"
    print(f"\n=== {mode} ===")
    print(f"chunks scanned:        {offset}")
    print(f"chunks updated:        {updated}")
    print(f"chunks w/o parent:     {missing_src}")
    print(f"field change counts:   {json.dumps(field_counts)}")
    print(f"elapsed:               {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()