#!/usr/bin/env python3
"""
Agent Tools for Hierarchical RAG

Two tools:
1. search_child_chunks(query, limit) → Hybrid search in ChromaDB, returns top-K children with parent_id
2. retrieve_parent_chunks(parent_id) → Load full parent context from JSON

Usage:
    from src.agent_tools import ToolFactory
    tools = ToolFactory(db_collection, llm_url)
    results = tools.search_child_chunks("Eph repulsion", 5)
    parent = tools.retrieve_parent_chunks("parent_id_here")
"""

import os
import requests
from typing import List, Dict, Optional
from src.parent_store_manager import ParentStoreManager

CHROMA_DB_PATH = "/Disk_bot/Eph/bib_rag/chroma_db_new"
BIB_RAG_EMBED_URL = "http://localhost:8081/v1/embeddings"

# --- Tool output budgets ---------------------------------------------------
# The agent subgraph re-prefills the WHOLE message history on every iteration.
# Parent chunks are large (median ~16k chars, p90 ~85k, max ~1.5M), so an
# unbounded `retrieve_parent_chunks` ToolMessage makes the next LLM prefill
# take minutes on a slow local model (e.g. Qwen3.8-27B). Cap each tool result
# so the per-iteration context stays bounded. Env-overridable:
#   RETRIEVE_PARENT_MAX_CHARS, SEARCH_RESULT_MAX_CHARS, MANY_PARENT_MAX_CHARS
def _budget(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default

RETRIEVE_PARENT_MAX_CHARS = _budget("RETRIEVE_PARENT_MAX_CHARS", 8000)   # ~2000 tokens
SEARCH_RESULT_MAX_CHARS = _budget("SEARCH_RESULT_MAX_CHARS", 800)        # per result
MANY_PARENT_MAX_CHARS = _budget("MANY_PARENT_MAX_CHARS", 1500)
SEARCH_MAX_LIMIT = _budget("SEARCH_MAX_LIMIT", 6)


def _clip(text: str, max_chars: int) -> str:
    """Truncate text to `max_chars`, marking the cut."""
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n…[truncated at {max_chars} chars]"


class ToolFactory:
    """Factory for creating agent tools with hierarchical retrieval."""
    
    def __init__(self, collection=None, llm_url: str = None):
        """
        Args:
            collection: ChromaDB collection (optional, auto-connects if None)
            llm_url: URL for LLM API (optional)
        """
        self.llm_url = llm_url
        self.parent_store = ParentStoreManager()
        
        # Auto-connect to ChromaDB if collection not provided
        if collection is None:
            from langchain_community.vectorstores import Chroma
            class Dummy:
                def embed_documents(self, texts): return [[0.0]*1024 for _ in texts]
                def embed_query(self, text): return [0.0]*1024
            self.db = Chroma(
                persist_directory=CHROMA_DB_PATH,
                embedding_function=Dummy(),
                collection_name="bib_rag_papers"
            )
            self.collection = self.db._collection
        else:
            self.collection = collection
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query using llama-server bge-m3."""
        try:
            resp = requests.post(
                BIB_RAG_EMBED_URL,
                headers={"Content-Type": "application/json"},
                json={"input": text, "model": "bge-m3"},
                timeout=30
            )
            emb = resp.json()["data"][0]["embedding"]
            norm = sum(x*x for x in emb) ** 0.5
            return [x/norm for x in emb] if norm > 0 else emb
        except Exception as e:
            print(f"❌ Embedding failed: {e}")
            return None
    
    def search_child_chunks(self, query: str, limit: int = 5, where: Optional[dict] = None) -> str:
        """
        Search for top-K most relevant child chunks in ChromaDB.

        Args:
            query: Search query string
            limit: Maximum number of results (default 5)
            where: Optional ChromaDB metadata filter dict, e.g.
                {"article_type": "review"} or
                {"$and": [{"article_type": "review"}, {"topic_eph-signaling": 1}]}
                Topic keywords are stored as boolean keys `topic_<kw>: 1`.

        Returns:
            Formatted string with child chunks and their parent_ids
        """
        try:
            # Embed query
            embedding = self.embed_query(query)
            if not embedding:
                return "EMBEDDING_ERROR: Could not generate query embedding"

            # Clamp the result count so a single search can't flood the context.
            limit = max(1, min(int(limit or 5), SEARCH_MAX_LIMIT))

            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=limit,
                where=where,
                include=["documents", "metadatas", "distances"]
            )

            if not results or not results["ids"][0]:
                return "NO_RELEVANT_CHUNKS"

            # Format results
            output_lines = []
            for i in range(len(results["ids"][0])):
                doc = results["documents"][0][i]
                meta = results["metadatas"][0][i]
                dist = results["distances"][0][i]
                sim = 1.0 / (1.0 + dist)

                parent_id = meta.get("parent_id", "unknown")
                source = meta.get("source", "unknown")
                section = meta.get("section", "unknown")

                output_lines.append(
                    f"--- RESULT {i+1} (similarity: {sim:.3f}) ---\n"
                    f"Source: {source}\n"
                    f"Section: {section}\n"
                    f"Parent ID: {parent_id}\n"
                    f"Content: {_clip(doc, SEARCH_RESULT_MAX_CHARS)}"
                )

            return "\n\n".join(output_lines)

        except Exception as e:
            return f"RETRIEVAL_ERROR: {str(e)}"
    
    def retrieve_parent_chunks(self, parent_id: str) -> str:
        """
        Retrieve full parent chunk by parent_id from JSON store.
        
        Args:
            parent_id: Parent chunk ID (format: "source#section#hash")
        
        Returns:
            Full parent content or error message
        """
        try:
            parent = self.parent_store.load_content(parent_id)
            
            if not parent:
                return f"NO_PARENT_DOCUMENT: {parent_id}"
            
            meta = parent.get('meta', {})
            content = parent.get('content', '')
            
            return (
                f"--- PARENT CHUNK ---\n"
                f"Source: {parent.get('source', 'unknown')}\n"
                f"Section: {parent.get('section', 'unknown')}\n"
                f"Parent ID: {parent_id}\n"
                f"Title: {meta.get('title', 'N/A')}\n"
                f"Authors: {meta.get('authors', 'N/A')}\n"
                f"Year: {meta.get('year', 'N/A')}\n"
                f"Word count: {parent.get('word_count', 0)}\n"
                f"\nContent:\n{_clip(content, RETRIEVE_PARENT_MAX_CHARS)}"
            )
            
        except Exception as e:
            return f"PARENT_RETRIEVAL_ERROR: {str(e)}"
    
    def retrieve_many_parents(self, parent_ids: List[str]) -> str:
        """Retrieve multiple parent chunks at once."""
        try:
            parents = self.parent_store.load_content_many(parent_ids)
            
            if not parents:
                return "NO_PARENT_DOCUMENTS"
            
            output_lines = []
            for i, parent in enumerate(parents, 1):
                meta = parent.get('meta', {})
                output_lines.append(
                    f"--- PARENT DOCUMENT {i} ---\n"
                    f"Source: {parent.get('source', 'unknown')}\n"
                    f"Section: {parent.get('section', 'unknown')}\n"
                    f"Parent ID: {parent.get('parent_id', 'N/A')}\n"
                    f"Title: {meta.get('title', 'N/A')}\n"
                    f"Year: {meta.get('year', 'N/A')}\n"
                    f"Word count: {parent.get('word_count', 0)}\n"
                    f"\nContent:\n{_clip(parent.get('content', ''), MANY_PARENT_MAX_CHARS)}"
                )
            
            return "\n\n".join(output_lines)
            
        except Exception as e:
            return f"PARENT_RETRIEVAL_ERROR: {str(e)}"


from langchain_core.tools import tool

def create_tools(collection=None):
    """Create LangChain-compatible tools for agent use.
    
    Returns list of @tool-decorated functions for ToolNode.
    """
    factory = ToolFactory(collection)
    
    @tool
    def search_child_chunks(query: str, limit: int = 5, where: Optional[dict] = None) -> str:
        """Search for relevant small child chunks in the document database. Returns excerpts with parent IDs for full context retrieval. Optional `where` is a ChromaDB metadata filter (e.g. {"article_type": "review"} or {"$and": [{"article_type": "review"}, {"topic_eph-signaling": 1}]})."""
        return factory.search_child_chunks(query, limit, where)
    
    @tool
    def retrieve_parent_chunks(parent_id: str) -> str:
        """Retrieve full parent chunk by parent_id. Use after search_child_chunks to get complete context."""
        return factory.retrieve_parent_chunks(parent_id)
    
    @tool
    def retrieve_many_parents(parent_ids: list) -> str:
        """Retrieve multiple parent chunks at once by their IDs."""
        return factory.retrieve_many_parents(parent_ids)
    
    return [search_child_chunks, retrieve_parent_chunks, retrieve_many_parents]


def create_tools_legacy(collection=None) -> List:
    """
    Legacy: Create tool dicts with name, description, and callable.
    
    Returns list of tool dicts for non-LangGraph use.
    """
    factory = ToolFactory(collection)
    
    return [
        {
            "name": "search_child_chunks",
            "description": "Search for relevant small child chunks in the document database. Returns excerpts with parent IDs for full context retrieval.",
            "func": factory.search_child_chunks
        },
        {
            "name": "retrieve_parent_chunks",
            "description": "Retrieve full parent chunk by parent_id. Use after search_child_chunks to get complete context.",
            "func": factory.retrieve_parent_chunks
        },
        {
            "name": "retrieve_many_parents",
            "description": "Retrieve multiple parent chunks at once by their IDs.",
            "func": factory.retrieve_many_parents
        }
    ]


if __name__ == "__main__":
    # Test the tools
    print("🧪 Testing agent tools...\n")
    
    tools = create_tools()
    factory = ToolFactory()
    
    # Test search
    print("1️⃣ search_child_chunks('Eph receptor signaling', 3):")
    result = factory.search_child_chunks("Eph receptor signaling", 3)
    print(result[:800] + "...\n")
    
    # Test parent store stats
    print("2️⃣ Parent store stats:")
    print(factory.parent_store.get_stats())
