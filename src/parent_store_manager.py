#!/usr/bin/env python3
"""
Parent Store Manager
Loads parent chunks from JSON files for retrieval-augmented generation.

Used by agent_tools.py to retrieve full parent context after child chunk search.
"""

import os, sys, json, glob
from pathlib import Path
from typing import List, Dict, Optional

# ─── Multi-KB config ─────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:  # bib_rag-package-try
    from .kb_config import get_config
except ImportError:  # flat (loose-script mode)
    from kb_config import get_config
_CFG = get_config()
PARENT_STORE_DIR = _CFG["parent_store_dir"]

class ParentStoreManager:
    """Manages parent chunk JSON files."""
    
    def __init__(self, store_dir: str = PARENT_STORE_DIR, create: bool = False):
        self.store_dir = store_dir
        if create:
            os.makedirs(store_dir, exist_ok=True)
    
    def load_content(self, parent_id: str) -> Optional[Dict]:
        """Load a single parent chunk by its parent_id."""
        # parent_id format: "source#section#hash"
        parts = parent_id.split('#')
        if len(parts) < 2:
            return None
        
        source = parts[0]
        safe_name = self._safe_filename(source)
        filepath = os.path.join(self.store_dir, f"{safe_name}.json")
        
        if not os.path.exists(filepath):
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                parents = json.load(f)
            
            for parent in parents:
                if parent.get('parent_id') == parent_id:
                    return parent
            
            return None
        except Exception as e:
            print(f"❌ Error loading parent {parent_id}: {e}")
            return None
    
    def load_content_many(self, parent_ids: List[str]) -> List[Dict]:
        """Load multiple parent chunks by their IDs."""
        results = []
        for pid in parent_ids:
            parent = self.load_content(pid)
            if parent:
                results.append(parent)
        return results
    
    def load_by_source(self, source: str) -> List[Dict]:
        """Load all parent chunks for a given source paper."""
        safe_name = self._safe_filename(source)
        filepath = os.path.join(self.store_dir, f"{safe_name}.json")
        
        if not os.path.exists(filepath):
            return []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    
    def list_sources(self) -> List[str]:
        """List all source papers in the parent store."""
        files = glob.glob(os.path.join(self.store_dir, "*.json"))
        return [os.path.basename(f)[:-5] for f in files]  # Remove .json
    
    def _safe_filename(self, source: str) -> str:
        """Convert source name to safe filename."""
        import re
        safe = re.sub(r'[^\w\-]', '_', source)
        return safe[:100]
    
    def get_stats(self) -> Dict:
        """Get statistics about the parent store."""
        files = glob.glob(os.path.join(self.store_dir, "*.json"))
        total_parents = 0
        for f in files:
            try:
                with open(f, 'r') as fh:
                    data = json.load(fh)
                    total_parents += len(data)
            except:
                pass
        
        return {
            'source_files': len(files),
            'total_parents': total_parents
        }

if __name__ == "__main__":
    # Test
    mgr = ParentStoreManager()
    stats = mgr.get_stats()
    print(f"Parent Store Stats: {stats}")
    
    if stats['source_files'] > 0:
        sources = mgr.list_sources()[:3]
        print(f"Sample sources: {sources}")
        
        # Try loading first source
        if sources:
            parents = mgr.load_by_source(sources[0])
            if parents:
                print(f"\nFirst parent from {sources[0]}:")
                print(f"  parent_id: {parents[0]['parent_id']}")
                print(f"  section: {parents[0]['section']}")
                print(f"  word_count: {parents[0]['word_count']}")
                print(f"  content_preview: {parents[0]['content'][:200]}...")
