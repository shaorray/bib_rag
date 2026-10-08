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
# Max pairs per submitted batch. The server packs several (query,doc) pairs
# into one physical batch; a big pool sent in one call can exceed the server's
# ubatch even when each individual pair fits. Splitting keeps each request
# small and independent (observed 2026-10-08: a 64-pair expanded pool tripped
# the server's 512-token ubatch; the server default is now 2048 but the client
# must not depend on a specific server config).
RERANK_BATCH = int(os.environ.get('RERANK_BATCH', '16'))

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


def _rerank_batch(query: str, batch: List[dict]) -> Optional[List[float]]:
    """Score one batch of docs. Returns a list of relevance scores aligned
    to `batch`, or None on failure (caller degrades)."""
    docs = [(r.get('text') or '')[:RERANK_MAX_CHARS] for r in batch]
    body = json.dumps({
        'model': RERANK_MODEL, 'query': query, 'documents': docs,
        'top_n': len(docs),
    }).encode()
    last_err: Optional[Exception] = None
    for attempt in range(2):
        try:
            req = urllib.request.Request(
                RERANK_URL, data=body,
                headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=RERANK_TIMEOUT) as r:
                out = json.loads(r.read())
            scores = [None] * len(batch)
            for hit in out['results']:
                scores[hit['index']] = hit['relevance_score']
            return scores
        except urllib.error.HTTPError as e:
            last_err = e
            # Transient 5xx (e.g. ubatch overflow, momentary overload):
            # back off briefly and retry once before degrading.
            if attempt == 0 and 500 <= getattr(e, 'code', 0) < 600:
                import time as _t
                _t.sleep(1.0)
                continue
        except (urllib.error.URLError, KeyError, IndexError, ValueError,
                TimeoutError, json.JSONDecodeError) as e:
            last_err = e
        except Exception as e:
            last_err = e
        break
    _warn_unavailable(f"{type(last_err).__name__}: {last_err}")
    return None


def rerank_results(query: str, results: List[dict],
                   top_k: Optional[int] = None) -> List[dict]:
    """Reorder `results` (list of retrieval dicts with 'text') by
    cross-encoder relevance to `query`.

    Never raises: any failure returns `results` unchanged. A 2-result
    list isn't worth a network round-trip — passthrough. Short lists keep
    their fusion order (RRF already did the work).

    The pool is submitted in batches of RERANK_BATCH pairs so a large pool
    cannot overflow the server's physical batch; if ANY batch fails, the
    whole call degrades to fusion order (a partially-reranked pool is worse
    than a consistently-ordered one).
    """
    if not results or len(results) < 3:
        return results[:top_k] if top_k else results

    scores: List[Optional[float]] = []
    step = max(1, RERANK_BATCH)
    for i in range(0, len(results), step):
        part = _rerank_batch(query, results[i:i + step])
        if part is None:
            # One failing batch → degrade the whole call (consistent order).
            return results[:top_k] if top_k else results
        scores.extend(part)

    order = sorted(range(len(results)), key=lambda i: -(scores[i] or -1e9))
    reordered = []
    for i in order:
        e = dict(results[i])
        if scores[i] is not None:
            e['rerank_score'] = round(scores[i], 4)
        e['rrf'] = results[i].get('rrf')   # keep RRF for debug
        reordered.append(e)
    return reordered[:top_k] if top_k else reordered