#!/usr/bin/env python3
"""
Batch-index all Cadherin markdown papers into bib_rag ChromaDB via
src/index_single_paper.index_paper(). Skips files already in the checkpoint.
"""
import os, sys, time
sys.path.insert(0, '/Disk_bot/RAG/bib_rag/src')
from index_single_paper import index_paper

MD_DIR = "/Disk_bot/Eph/Cadherin_papers/md"

def main():
    mds = sorted(f for f in os.listdir(MD_DIR) if f.endswith('.md'))
    print(f"to index: {len(mds)} Cadherin md files")
    t0 = time.time()
    ok = fail = skip = 0
    for i, f in enumerate(mds, 1):
        path = os.path.join(MD_DIR, f)
        try:
            if index_paper(path):
                ok += 1
            else:
                skip += 1
        except Exception as e:
            fail += 1
            print(f"  ❌ {f}: {str(e)[:80]}", file=sys.stderr)
        if i % 20 == 0 or i == len(mds):
            el = time.time() - t0
            print(f"[{i}/{len(mds)}] ok={ok} skip={skip} fail={fail} ({el:.0f}s)", file=sys.stderr)
    print(f"\ndone: ok={ok} skip={skip} fail={fail}, {time.time()-t0:.0f}s total")

if __name__ == "__main__":
    main()
