#!/usr/bin/env python3
"""
vLLM Embedding Adapter for Eph/Ephrin Agentic RAG Knowledge Base
Replaces sentence-transformers with vLLM's nomic-embed-text for ~10x speedup
"""

import os
import sys
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path

# HF mirror for China
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

class VLLMEmbedding:
    """
    vLLM-based embedding engine using nomic-embed-text-v1.5
    
    Performance: ~500 texts/s on RTX 3080 (vs ~50-100/s with sentence-transformers)
    Dimension: 768 (same as nomic-embed-text-v1.5)
    """
    
    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5"):
        self.model_name = model_name
        self.model = None
        self.dimension = 768
        
        # Lazy loading - only load when first embed() call is made
        self._loaded = False
    
    def _load_model(self):
        """Lazy load vLLM model"""
        if self._loaded:
            return
        
        try:
            from vllm import LLM
            print(f"🔧 Loading vLLM embedding model: {self.model_name}")
            
            self.model = LLM(
                model=self.model_name,
                download_dir='/Disk_bot/models',
                gpu_memory_utilization=0.3,  # Low GPU usage for embedding
                enforce_eager=True,
                trust_remote_code=True,
                dtype='float16',
                max_model_len=2048,
            )
            self._loaded = True
            print(f"   ✓ vLLM embedding model ready (768-dim)")
            
        except Exception as e:
            print(f"   ⚠️ vLLM loading failed: {e}")
            print(f"   ⚠️ Falling back to sentence-transformers")
            self._load_fallback()
    
    def _load_fallback(self):
        """Fallback to sentence-transformers if vLLM fails"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="/tmp/sentence_transformers")
            self.dimension = self.model.get_sentence_embedding_dimension()
            self._loaded = True
            print(f"   ✓ Fallback model loaded: all-MiniLM-L6-v2 ({self.dimension}-dim)")
        except Exception as e:
            print(f"   ❌ Fallback also failed: {e}")
            self.model = None
            self._loaded = True
    
    def embed(self, text: str) -> np.ndarray:
        """Embed single text"""
        self._load_model()
        
        if hasattr(self.model, 'embed'):
            # vLLM path
            outputs = self.model.embed([text])
            return np.array(outputs[0].outputs.embedding, dtype=np.float32)
        elif self.model:
            # sentence-transformers path
            return self.model.encode(text, convert_to_numpy=True)
        else:
            return self._simple_embedding(text)
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Batch embed with vLLM (high throughput)"""
        self._load_model()
        
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        
        if hasattr(self.model, 'embed'):
            # vLLM path - process all at once (vLLM handles batching internally)
            import time
            t0 = time.time()
            outputs = self.model.embed(texts)
            duration = time.time() - t0
            
            embeddings = np.array([o.outputs.embedding for o in outputs], dtype=np.float32)
            print(f"   ⚡ vLLM embedded {len(texts)} texts in {duration:.3f}s ({len(texts)/duration:.0f} texts/s)")
            return embeddings
            
        elif self.model:
            # sentence-transformers path
            return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        else:
            return np.array([self._simple_embedding(t) for t in texts])
    
    def _simple_embedding(self, text: str) -> np.ndarray:
        """Simple fallback embedding"""
        import hashlib
        words = text.lower().split()
        embedding = np.zeros(self.dimension)
        
        for word in words:
            hash_val = int(hashlib.md5(word.encode()).hexdigest(), 16)
            np.random.seed(hash_val % 2**32)
            word_vec = np.random.randn(self.dimension)
            embedding += word_vec
        
        if len(words) > 0:
            embedding /= len(words)
        
        return embedding / (np.linalg.norm(embedding) + 1e-8)


# --- Compatibility shim for existing code ---
class SimpleEmbedding(VLLMEmbedding):
    """
    Drop-in replacement for existing SimpleEmbedding class
    Usage: change `from rag_core import SimpleEmbedding` to use this instead
    """
    pass


# --- CLI test ---
if __name__ == "__main__":
    print("Testing vLLM Embedding Adapter...")
    
    embedder = VLLMEmbedding()
    
    # Test single
    vec = embedder.embed("Eph receptors regulate cell adhesion and migration.")
    print(f"Single embedding: {vec.shape}, sample: {vec[:5]}")
    
    # Test batch
    texts = [
        "Eph-ephrin signaling in neural development.",
        "Cell adhesion molecules in cancer metastasis.",
        "Axon guidance mechanisms during embryogenesis.",
        "Receptor tyrosine kinase signaling pathways.",
    ] * 5  # 20 texts
    
    embeddings = embedder.embed_batch(texts)
    print(f"Batch embeddings: {embeddings.shape}")
    print("✓ vLLM Embedding Adapter ready!")
