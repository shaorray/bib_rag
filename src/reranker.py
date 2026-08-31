#!/usr/bin/env python3
"""
reranker.py — Cross-encoder reranking of hybrid retrieval results.

bge-reranker-v2-m3 (Q8_0) served by llama.cpp's /v1/rerank endpoint
(port 11436, launched via llama-server.sh bge-reranker-v2-m3). A
cross-encoder reads (query, passage) JOINTLY, so it catches the
terminology-misalignment misses bi-encoder retrieval cannot: the query
says "cell segregation", the passage says "sorting by differential
adhesion" — no lexical overlap, mediocre cosine, but the cross-encoder
reads them together and scores the pair as what it is: the same claim.

Design:
  - Post-fusion stage: rrf_fuse produces the candidate set, rerank
    REORDERS it. RRF deliberately throws away score magnitude (rank-only
    fusion); rerank reintroduces absolute (query, passage) relevance.
  - Ranks, not scores, are the contract: the output is sorted by
    relevance_score and the original RRF fields are preserved for
    display/debug.
  - Graceful degradation everywhere: server down / empty response /
    short candidate lists → the input ordering is returned unchanged.
    The agent never breaks because the reranker is missing.
  - Env kill-switch: RERANK=0 (default on when the service is up).
  - Context guard: candidate text is clipped to RERANK_MAX_CHARS
    (default 1000 ≈ 250 tokens; cross-encoders degrade on long text and
    the child chunks are 500 words).
"""
from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from typing import List, Optional

RERANK_URL = os.environ.get(
    'RERANK_URL', 'http://localhost:11436/v1/rerank')
RERANK_MODEL = os.environ.get('RERANK_MODEL', 'bge-reranker-v2-m3')
RERANK_MAX_CHARS = int(os.environ.get('RERANK_MAX_CHARS', '1000'))
RERANK_TIMEOUT = int(os.environ.get('RERANK_TIMEOUT', '30'))

_WARNED = {"n": 0}

def _warn_unavailable(detail: str) -> None:
    """One stderr warning per process when the rerank service fails.

    Reranking is an enhancement, never a dependency — but a persistent
    outage (dead endpoint, wrong port) must not silently degrade every
    query's ranking quality. First failure warns; the rest stay quiet.
    """
    if _WARNED["n"] >= 1:
        return
    _WARNED["n"] = 1
    import sys
    print(f"[reranker] UNAVAILABLE — serving fusion order instead "
          f"({detail}); set RERANK_URL or start the service. "
          f"(warning once per process)", file=sys.stderr)


def rerank_available() -> bool:
    """Probe the rerank service (cheap; cached by the caller if needed)."""
    try:
        req = urllib.request.Request(RERANK_URL.rsplit('/v1/rerank', 1)[0]
                                      + '/health')
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def rerank_results(query: str, results: List[dict],
                   top_k: Optional[int] = None) -> List[dict]:
    """Reorder `results` (list of retrieval dicts with 'text') by
    cross-encoder relevance to `query`.

    Never raises: any failure returns `results` unchanged. A 2-result
    list isn't worth a network round-trip — passthrough. Short lists keep
    their fusion order (RRF already did the work).
    """
    if not results or len(results) < 3:
        return results[:top_k] if top_k else results
    docs = [(r.get('text') or '')[:RERANK_MAX_CHARS] for r in results]
    try:
        body = json.dumps({
            'model': RERANK_MODEL, 'query': query, 'documents': docs,
            'top_n': len(docs),
        }).encode()
        req = urllib.request.Request(
            RERANK_URL, data=body,
            headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=RERANK_TIMEOUT) as r:
            out = json.loads(r.read())
        ranked = sorted(out['results'], key=lambda x: -x['relevance_score'])
        reordered = []
        for hit in ranked:
            e = dict(results[hit['index']])
            e['rerank_score'] = round(hit['relevance_score'], 4)
            e['rrf'] = results[hit['index']].get('rrf')   # keep RRF for debug
            reordered.append(e)
        return reordered[:top_k] if top_k else reordered
    except (urllib.error.URLError, KeyError, IndexError, ValueError,
            TimeoutError, json.JSONDecodeError) as e:
        _warn_unavailable(f"{type(e).__name__}: {e}")
        return results[:top_k] if top_k else results
    except Exception as e:
        # reranking is an enhancement, never a dependency
        _warn_unavailable(f"{type(e).__name__}: {e}")
        return results[:top_k] if top_k else results