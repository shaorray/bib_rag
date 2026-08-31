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

import os, sys
import requests
from typing import List, Dict, Optional

# ─── Multi-KB config ─────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:  # bib_rag-package-try
    from .kb_config import get_config
except ImportError:  # flat (loose-script mode)
    from kb_config import get_config
try:  # bib_rag-package-try
    from .parent_store_manager import ParentStoreManager
except ImportError:  # flat (loose-script mode)
    from parent_store_manager import ParentStoreManager
_CFG = get_config()
CHROMA_DB_PATH = _CFG["chroma_path"]
BIB_RAG_EMBED_URL = _CFG["embed_url"]

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


def _is_local_llm() -> bool:
    """True when the LLM_URL/LLM_MODEL points at a local llama-server slot
    (GGUF or :5015) — same detection as agent_nodes._resolve_limits. Local
    slots are 4096 tokens (`-c 8192 -np 2`), so tool budgets must default
    tighter than for cloud models with 32k+ windows."""
    model = os.environ.get("LLM_MODEL", "")
    llm_url = os.environ.get("LLM_URL", "")
    return model.endswith(".gguf") or ":5015" in llm_url


RETRIEVE_PARENT_MAX_CHARS = _budget("RETRIEVE_PARENT_MAX_CHARS",
                                    4000 if _is_local_llm() else 8000)  # ~1000/2000 tokens
SEARCH_RESULT_MAX_CHARS = _budget("SEARCH_RESULT_MAX_CHARS", 800)        # per result
MANY_PARENT_MAX_CHARS = _budget("MANY_PARENT_MAX_CHARS", 1500)
SEARCH_MAX_LIMIT = _budget("SEARCH_MAX_LIMIT", 6)
# Whole-call cap: with SEARCH_MAX_LIMIT=6 results × (800 chars + headers) a
# single search can emit ~5.5KB (~1400 tokens); Qwen issues up to 3 parallel
# searches in one round, so the ROUND budget is what matters for a 4096-token
# local slot: base prompts (~1650 tok measured) + Σ tool calls must stay
# under 4096 → search 2200, parent 4000 chars per call.
SEARCH_CALL_MAX_CHARS = _budget("SEARCH_CALL_MAX_CHARS", 2200)           # ~550 tokens


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
                collection_name=_CFG["collection_name"],
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
    
    def _post_fusion(self, query: str, fused: List[dict],
                     limit: int) -> List[dict]:
        """Shared post-fusion pipeline: rerank → citation boost → source cap.

        Both the first pass AND the broaden retry MUST go through this —
        a weak first pass that broadens still deserves cross-encoder
        precision and the diversity cap (broaden used to bypass all three,
        shipping raw fusion order: found by the v3 regression test).
        Each stage fails soft (env kill-switch / exception → skip stage).
        """
        # Cross-encoder rerank (bge-reranker-v2-m3 via /v1/rerank).
        # RRF is rank-only; the cross-encoder reintroduces absolute
        # (query, passage) relevance and catches terminology-misaligned
        # matches bi-encoders score poorly. NOTE: rerank must NOT cut to
        # limit here — the cap below does the final cut (cutting first
        # would let backfill restore the very duplicates the cap removes).
        if os.environ.get("RERANK", "1") != "0":
            try:
                try:
                    from .reranker import rerank_results
                except ImportError:  # src/ on sys.path directly (CLI/tests)
                    try:  # bib_rag-package-try
                        from .reranker import rerank_results
                    except ImportError:  # flat (loose-script mode)
                        from reranker import rerank_results
                fused = rerank_results(query, fused)
            except Exception:
                pass

        # Citation-graph proximity boost: anchors = top-2 reranked sources;
        # pool entries that are 1-hop citation neighbors get a gentle
        # multiplicative nudge (×1.05 cited-by / ×1.03 cites). Reranker
        # scores are logits, so the factors are deliberately small.
        if os.environ.get("RAG_CITE_BOOST", "1") != "0":
            try:
                try:
                    from .reference_graph import load_graph, neighbors
                except ImportError:
                    try:  # bib_rag-package-try
                        from .reference_graph import load_graph, neighbors
                    except ImportError:  # flat (loose-script mode)
                        from reference_graph import load_graph, neighbors
                g = load_graph()
                if g:
                    anchors = [e.get("source", "") for e in fused[:2]]
                    nbr: Dict[str, str] = {}
                    for a in anchors:
                        for s, info in neighbors(g, a).items():
                            d = (info or {}).get("direction", "")
                            if s not in nbr or d == "cited-by":
                                nbr[s] = d
                    for a in anchors:
                        nbr.pop(a, None)
                    if nbr:
                        for e in fused:
                            d = nbr.get(e.get("source", ""))
                            if d:
                                e["rerank_score"] = (
                                    e.get("rerank_score") or 0.0) * (
                                    1.05 if d == "cited-by" else 1.03)
                        fused.sort(
                            key=lambda x: -(x.get("rerank_score") or 0.0))
            except Exception:
                pass

        # Per-source diversity cap (env RAG_SOURCE_CAP, default 2): walk the
        # reranked pool keeping at most CAP entries per source until limit
        # is filled; backfill from skipped extras only when the pool
        # exhausts below limit (count contract).
        try:
            cap = max(1, int(os.environ.get("RAG_SOURCE_CAP", "2")))
            counts: Dict[str, int] = {}
            kept, skipped = [], []
            for e in fused:
                s = e.get("source", "")
                if len(kept) >= limit:
                    break
                if counts.get(s, 0) >= cap:
                    skipped.append(e)
                else:
                    counts[s] = counts.get(s, 0) + 1
                    kept.append(e)
            if len(kept) < limit:
                kept.extend(skipped[:limit - len(kept)])
            fused = kept
        except Exception:
            pass
        return fused

    def _format_results(self, fused: List[dict], broadened: bool = False,
                        broaden_reasons: Optional[List[str]] = None) -> str:
        """Format search results for the agent (shared by both paths)."""
        output_lines = []
        if broadened:
            output_lines.append(
                "[BROADENED] first-pass results were weak ("
                + "; ".join(broaden_reasons or [])
                + ") — results below include a widened re-search (where-filter "
                "dropped, limit widened).")
        for i, r in enumerate(fused):
            sim = r.get("similarity", 0.0)
            output_lines.append(
                f"--- RESULT {i+1} (similarity: {sim:.3f}, channels: {r.get('channels', 'vec')}) ---\n"
                f"Source: {r['source']}\n"
                f"Section: {r['section']}\n"
                f"Parent ID: {r['parent_id']}\n"
                f"Content: {_clip(r['text'], SEARCH_RESULT_MAX_CHARS)}"
            )
            # Whole-call budget: stop adding results once the formatted output
            # would exceed SEARCH_CALL_MAX_CHARS (parallel tool calls multiply
            # this — see the budget note at the top of this file).
            if sum(len(x) + 2 for x in output_lines) >= SEARCH_CALL_MAX_CHARS:
                output_lines.append(
                    f"…[call capped at {SEARCH_CALL_MAX_CHARS} chars; "
                    f"{len(fused) - i - 1} result(s) dropped — narrow the query or raise SEARCH_CALL_MAX_CHARS]")
                break
        return "\n\n".join(output_lines)

    def _vector_entries(self, results) -> List[dict]:
        """ChromaDB query result → entry dicts for RRF fusion."""
        out = []
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            out.append({
                "parent_id": meta.get("parent_id", "unknown"),
                "source": meta.get("source", "unknown"),
                "section": meta.get("section", "unknown"),
                "text": results["documents"][0][i],
                "similarity": 1.0 / (1.0 + results["distances"][0][i]),
            })
        return out

    def _vector_search(self, query: str, limit: int,
                       where: Optional[dict]):
        """Raw ChromaDB query. Returns None on embedding failure."""
        embedding = self.embed_query(query)
        if not embedding:
            return None
        return self.collection.query(
            query_embeddings=[embedding],
            n_results=limit,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

    def search_child_chunks(self, query: str, limit: int = 5, where: Optional[dict] = None) -> str:
        """
        Search for top-K most relevant child chunks in ChromaDB.

        Hybrid retrieval (DocsGPT/seerai mechanism): dense bge-m3 results are
        fused with BM25 (FTS5 index via hybrid_search.HybridIndex) using RRF
        (k=60). Gene symbols and receptor names (Myc, Notch1) are exact
        lexical signals that dense embeddings often miss; BM25 catches them.
        When the FTS index is empty or errors, falls back to dense-only
        silently — the agent never breaks because BM25 is missing.
        Env kill-switch: HYBRID_SEARCH=0 disables the BM25 channel.

        Broadened retry (zotero-redisearch-rag mechanism, see
        src/broaden.py): when the fused results trip a weakness signal
        (too few / too short / weak best score / narrative-starved), ONE
        automatic re-search runs with the where-filter dropped and a wider
        limit; results are prefixed with [BROADENED] + the triggering
        signals. Env kill-switch: BROADEN_RETRY=0.

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
            # Clamp the result count so a single search can't flood the context.
            limit = max(1, min(int(limit or 5), SEARCH_MAX_LIMIT))

            # Goldset ablation 2026-08-31: fetching limit*3 dense candidates
            # and letting the extra siblings into fusion DILUTED the
            # chunk-frequency signal (recall 0.742→0.725) — dense keeps the
            # tight limit; bm25 contributes its own limit*3 pool inside
            # HybridIndex.search, so the two channels stay complementary
            # (tight dense head + wider lexical recall) rather than symmetric.
            results = self._vector_search(query, limit, where)
            if results is None:
                return "EMBEDDING_ERROR: Could not generate query embedding"

            if not results or not results["ids"][0]:
                return "NO_RELEVANT_CHUNKS"

            vector_entries = self._vector_entries(results)

            # Hybrid fusion (BM25 + RRF) when enabled. The fused pool is
            # limit*3: fusion ranks cheaply, the cross-encoder below does
            # the precise ordering — rerank needs candidates to cut.
            if os.environ.get("HYBRID_SEARCH", "1") != "0":
                try:
                    from .hybrid_search import HybridIndex
                    fused = HybridIndex().search(query, vector_entries,
                                                 top_k=limit * 3)
                except Exception:
                    fused = vector_entries  # BM25 unavailable → dense-only
            else:
                fused = vector_entries

            # Shared post-fusion pipeline: rerank → cite-boost → source cap.
            # (See _post_fusion — both the first pass and the broaden retry
            # below go through it.)
            fused = self._post_fusion(query, fused, limit)

            # Broadened retry on weak first-pass (zero-LLM, once).
            if os.environ.get("BROADEN_RETRY", "1") != "0":
                try:
                    try:
                        from .broaden import (retrieval_metrics, should_broaden,
                                              plan_broadening, broaden_signature,
                                              or_split_query)
                    except ImportError:  # src/ on sys.path directly (CLI/tests)
                        try:  # bib_rag-package-try
                            from .broaden import retrieval_metrics, should_broaden, plan_broadening, broaden_signature, or_split_query
                        except ImportError:  # flat (loose-script mode)
                            from broaden import retrieval_metrics, should_broaden, plan_broadening, broaden_signature, or_split_query
                    metrics = retrieval_metrics(vector_entries)
                    weak, reasons = should_broaden(metrics)
                    sig = broaden_signature(query, {"drop_where": True,
                                                    "or_split": False})
                    if weak and sig not in getattr(self, "_broaden_seen", set()):
                        if not hasattr(self, "_broaden_seen"):
                            self._broaden_seen = set()
                        self._broaden_seen.add(sig)
                        plan = plan_broadening(query, had_where=bool(where),
                                               attempt=0)
                        if plan:
                            wide_limit = min(limit * 3, SEARCH_MAX_LIMIT)
                            alt_query = (or_split_query(query) or query
                                         if plan.get("or_split") else query)
                            alt_where = None if plan.get("drop_where") else where
                            alt = self._vector_search(alt_query, wide_limit,
                                                      alt_where)
                            if alt and alt["ids"][0]:
                                alt_entries = self._vector_entries(alt)
                                if os.environ.get("HYBRID_SEARCH", "1") != "0":
                                    try:
                                        try:
                                            from .hybrid_search import HybridIndex
                                        except ImportError:
                                            try:  # bib_rag-package-try
                                                from .hybrid_search import HybridIndex
                                            except ImportError:  # flat (loose-script mode)
                                                from hybrid_search import HybridIndex
                                        alt_fused = HybridIndex().search(
                                            alt_query, alt_entries,
                                            top_k=wide_limit)
                                    except Exception:
                                        alt_fused = alt_entries
                                else:
                                    alt_fused = alt_entries
                                alt_metrics = retrieval_metrics(alt_entries)
                                alt_weak, _ = should_broaden(alt_metrics)
                                # keep the better of the two passes.
                                # NOTE: rrf_fuse dedups by parent_id, so a
                                # "wider" search may FUSE to fewer entries
                                # than the first pass (parent-level dupes);
                                # a weak alt with equalish size must still
                                # win when the FIRST pass was the one judged
                                # weak — otherwise the widened re-search can
                                # never replace it and [BROADENED] never
                                # fires. Prefer the alt whenever it is not
                                # itself weak; fall back to a raw-count win.
                                if not alt_weak or (len(alt_fused) > len(fused)) \
                                        or (alt_weak and len(reasons) == 1
                                            and reasons[0].startswith("weak-best-score")
                                            and alt_metrics.get("best_similarity")
                                            and alt_metrics["best_similarity"]
                                            >= metrics.get("best_similarity", 0)):
                                    # alt is no worse on the triggering signal
                                    # (same-or-better best similarity) → swap
                                    # so the [BROADENED] header still reaches
                                    # the agent. The alt pool ALSO goes
                                    # through the full post-fusion pipeline
                                    # (rerank → cite-boost → cap) — broaden
                                    # used to bypass it, shipping raw fusion
                                    # order for exactly the queries that
                                    # were already weak.
                                    alt_fused = self._post_fusion(
                                        alt_query, alt_fused, limit)
                                    fused = alt_fused[:limit]
                                    return self._format_results(
                                        fused, broadened=True,
                                        broaden_reasons=reasons)
                    return self._format_results(fused)
                except Exception:
                    return self._format_results(fused)

            return self._format_results(fused)

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

    def snowball_search(self, source: str, direction: str = "forward") -> str:
        """Citation snowballing over the reference graph (Corvus mechanism).

        forward: papers in the KB citing `source`; backward: `source`'s
        references that are in the KB. Gracefully reports when the graph
        hasn't been built yet (scripts/build_reference_graph.py).
        """
        try:
            try:
                from .reference_graph import load_graph, snowball
            except ImportError:  # src/ on sys.path directly (CLI/tests)
                try:  # bib_rag-package-try
                    from .reference_graph import load_graph, snowball
                except ImportError:  # flat (loose-script mode)
                    from reference_graph import load_graph, snowball
            graph = load_graph()
            if graph is None:
                return ("NO_REFERENCE_GRAPH: reference graph not built yet. "
                        "Run scripts/build_reference_graph.py once to enable snowballing.")
            result = snowball(graph, source, direction=direction)
            if result.get("error"):
                return f"SNOWBALL_ERROR: {result['error']}"
            matches = result.get("matches", [])
            if not matches:
                return (f"NO_MATCHES: no in-library {'citing papers' if direction == 'forward' else 'resolved references'} "
                        f"for '{source}' (direction={direction})")
            lines = [f"--- SNOWBALL {direction.upper()} from '{result['query']}' ---"]
            for i, m in enumerate(matches, 1):
                if direction == "forward":
                    lines.append(
                        f"{i}. {m.get('title', 'unknown')} ({m.get('year', 'n.d.')})\n"
                        f"   Source: {m['source']}\n"
                        f"   Cited via: {m.get('via', '')}"
                    )
                else:
                    resolved = (f" → IN LIBRARY: {m['resolved_title']}"
                                if m.get("in_library") else " → not in library")
                    lines.append(f"{i}. {m.get('raw_ref', '')[:180]}{resolved}")
            return "\n".join(lines)
        except Exception as e:
            return f"SNOWBALL_ERROR: {str(e)}"

    def related_papers(self, source: str, k: int = 8) -> str:
        """"I just read X, what should I read next?" — ranked recommendations.

        Blends three signals: topic-keyword Jaccard (chroma `topics`),
        title+lead embedding cosine (bge-m3), and the citation-graph signal
        (in-library forward citations + bibliographic coupling).
        """
        try:
            try:
                from .related_papers import find_related
            except ImportError:  # src/ on sys.path directly (CLI/tests)
                try:  # bib_rag-package-try
                    from .related_papers import find_related
                except ImportError:  # flat (loose-script mode)
                    from related_papers import find_related
            result = find_related(source, k=k)
            if result.get("error"):
                return f"RELATED_ERROR: {result['error']}"
            matches = result.get("matches", [])
            if not matches:
                return (f"NO_MATCHES: no related papers found for "
                        f"'{source}' — every candidate scored below threshold.")
            lines = [f"--- RELATED PAPERS for '{result['query']}' "
                     f"({result.get('title', '')[:70]}) ---"]
            for i, m in enumerate(matches, 1):
                lines.append(
                    f"{i}. [{m.get('article_type') or '?'}] "
                    f"score={m['score']:.3f} "
                    f"(topics={m['topic_j']:.2f} emb={m['emb_cos']:.2f} "
                    f"graph={m['graph']:.2f}) "
                    f"{m['source']} ({m.get('year', 'n.d.')})\n"
                    f"   {m.get('title', '')[:100]}\n"
                    f"   why: {m.get('why', '')}"
                )
            return "\n".join(lines)
        except Exception as e:
            return f"RELATED_ERROR: {str(e)}"


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

    @tool
    def find_papers_citing(source: str) -> str:
        """Citation snowballing (forward): find papers IN the knowledge base that cite the given paper. Pass a source filename or paper title (fuzzy). Returns matched library papers with the citation evidence. Use to follow research influence forward in time."""
        return factory.snowball_search(source, direction="forward")

    @tool
    def get_paper_references(source: str) -> str:
        """Citation snowballing (backward): list the references of the given paper that are themselves in the knowledge base. Pass a source filename or paper title (fuzzy). Returns raw reference strings plus resolution to library papers when possible."""
        return factory.snowball_search(source, direction="backward")

    @tool
    def find_related_papers(source: str, k: int = 8) -> str:
        """Recommend papers related to a given library paper — "I just read X, what should I read next?". Pass a source filename, PMID-stem, or title fragment. Blends topic overlap, embedding similarity, and citation-graph signals; each hit carries a machine-readable `why` (shared topics, cites-it, bibliographic coupling). Use to build reading paths around a seed paper."""
        return factory.related_papers(source, k=k)

    return [search_child_chunks, retrieve_parent_chunks, retrieve_many_parents,
            find_papers_citing, get_paper_references, find_related_papers]


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
