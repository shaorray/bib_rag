#!/usr/bin/env python3
"""Retry indexing for Cadherin md files not yet in ChromaDB (after embed retry fix)."""
import os, sys, time
sys.path.insert(0, '/Disk_bot/Eph/bib_rag/src')
import chromadb
from index_single_paper import index_paper

MD_DIR = "/Disk_bot/Eph/Cadherin_papers/md"
CHROMA = "/Disk_bot/Eph/bib_rag/chroma_db_new"

def main():
    col = chromadb.PersistentClient(path=CHROMA).get_collection("bib_rag_papers")
    r = col.get(include=["metadatas"])
    sources = set(m.get("source", "") for m in r["metadatas"])
    mds = [f for f in os.listdir(MD_DIR) if f.endswith(".md")]
    missing = [f for f in mds if f not in sources]
    print(f"待重试: {len(missing)} 篇")
    t0 = time.time()
    ok = fail = 0
    for i, f in enumerate(missing, 1):
        path = os.path.join(MD_DIR, f)
        try:
            if index_paper(path):
                ok += 1
            else:
                fail += 1
                print(f"  skip {f}", file=sys.stderr)
        except Exception as e:
            fail += 1
            print(f"  ❌ {f}: {str(e)[:80]}", file=sys.stderr)
        if i % 10 == 0 or i == len(missing):
            print(f"[{i}/{len(missing)}] ok={ok} fail={fail} ({time.time()-t0:.0f}s)", file=sys.stderr)
    print(f"\n完成: ok={ok} fail={fail} 总耗时 {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
