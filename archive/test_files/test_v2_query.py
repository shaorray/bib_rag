#!/usr/bin/env python3
"""Quick test query for the rebuilt chroma_db_v2."""
import os, pickle, numpy as np
from sentence_transformers import SentenceTransformer

DB_PATH = "/Disk_2/claw_working_dir/ephrin_agentic_rag/chroma_db_v2/ephrin_papers_v2.pkl"

# Load
with open(DB_PATH, 'rb') as f:
    data = pickle.load(f)

# Load same model
os.environ['HF_HUB_OFFLINE'] = '1'
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',
                              cache_folder='/tmp/st_cache',
                              local_files_only=True,
                              trust_remote_code=True)

def query(q, k=5):
    emb = model.encode(q, normalize_embeddings=True)
    embs = np.array(data['embeddings']).astype(np.float32)
    sims = np.dot(embs, emb)
    top = np.argsort(sims)[-k:][::-1]
    print(f"\nQuery: '{q}'")
    for i, idx in enumerate(top):
        doc = data['documents'][idx].replace('\n', ' ')[:200]
        meta = data['metadatas'][idx]
        print(f"  {i+1}. [{meta.get('section','?')}] sim={sims[idx]:.4f}")
        print(f"      {doc}")

if __name__ == '__main__':
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else 'Eph receptor signaling mechanism'
    query(q)

# Example usage:
# python3 test_v2_query.py "forward signaling in axon guidance"
# python3 test_v2_query.py "ephrin-A5 knockout phenotype"
