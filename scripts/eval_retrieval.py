#!/usr/bin/env python3
"""
eval_retrieval.py — source-level recall@k / MRR over the retrieval gold set.

Mirrors the PRODUCTION retrieval path exactly (agent_tools.search_child_chunks
minus formatting/broaden): _vector_search → _vector_entries →
HybridIndex().search (RRF fusion). No LLM in the loop.

Usage:
    python3 scripts/eval_retrieval.py [--kb eph_rag] [--k 6] [--json out.json]
Env switches respected: HYBRID_SEARCH=0 (dense-only), RERANK=... etc —
whatever the production code reads, so A/B runs use the same harness.
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, '/Disk_bot/RAG/bib_rag/src')

from kb_config import get_config, parse_kb_arg


def _source_key(s: str) -> str:
    """Chroma `source` values and parent_store filenames live in two naming
    domains ('Hall et al. - 2024 - …' vs 'Hall_et_al__-_2024_-_…'). Fold both
    through the SAME sanitize transform hybrid_search uses for lookups."""
    return re.sub(r"[^\\w\\-]", "_", s or "")[:100]


def load_gold(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    argv = parse_kb_arg()
    ap = argparse.ArgumentParser()
    ap.add_argument('--gold', default=None)
    ap.add_argument('--k', type=int, default=6)
    ap.add_argument('--limit-pool', type=int, default=0,
                    help='override dense candidate pool (0 = production)')
    ap.add_argument('--json', default=None, help='dump full per-query results')
    args = ap.parse_args(argv)

    cfg = get_config()
    gold_path = args.gold or os.path.join(cfg['data_dir'],
                                          'gold_retrieval.jsonl')
    rows = load_gold(gold_path)
    print(f'library : {cfg["kb_name"]}   gold: {len(rows)} queries   k={args.k}')

    # late imports so env vars set above are respected
    from agent_tools import ToolFactory
    from hybrid_search import HybridIndex

    tf = ToolFactory()
    idx = HybridIndex()

    def retrieve(query, limit):
        """Exact production path (no broaden, no formatting): dense limit
        candidates → RRF with bm25 limit*3 pool → top_k."""
        res = tf._vector_search(query, limit, None)
        if res is None:
            return []
        entries = tf._vector_entries(res)
        if os.environ.get('HYBRID_SEARCH', '1') != '0':
            try:
                out = idx.search(query, entries, top_k=limit * 3)
            except Exception:
                out = entries[:limit * 3]
        else:
            out = entries[:limit * 3]
        # rerank narrows the fused pool to top_k — mirrors
        # search_child_chunks (fused limit*3 → rerank → limit)
        if os.environ.get('RERANK', '1') != '0':
            try:
                from reranker import rerank_results
                return rerank_results(query, out, top_k=limit)
            except Exception:
                return out[:limit]
        return out[:limit]

    per_q, tiers = [], {}
    t0 = time.time()
    for r in rows:
        gold = {_source_key(g) for g in r['gold']}
        hits = retrieve(r['query'], args.k)
        sources = [_source_key(h.get('source', '')) for h in hits]
        # first rank where a gold source appears (source-level MRR)
        mrr, rank_hit = 0.0, 0
        for j, s in enumerate(sources, 1):
            if s in gold:
                mrr = 1.0 / j
                rank_hit = j
                break
        rec = 1.0 if rank_hit else 0.0
        per_q.append({'query': r['query'], 'tier': r.get('tier', '?'),
                     'gold': sorted(gold), 'sources': sources,
                     'recall': rec, 'mrr': mrr, 'rank': rank_hit})
        t = tiers.setdefault(r.get('tier', '?'), {'n': 0, 'rec': 0.0, 'mrr': 0.0})
        t['n'] += 1
        t['rec'] += rec
        t['mrr'] += mrr

    n = len(per_q)
    print(f'elapsed : {time.time()-t0:.0f}s   ({(time.time()-t0)/n:.1f}s/query)\n')
    print(f'{"tier":9} {"n":>3} {"recall@k":>9} {"MRR":>6}')
    for tname in sorted(tiers):
        t = tiers[tname]
        print(f'{tname:9} {t["n"]:>3} {t["rec"]/t["n"]:>9.3f} {t["mrr"]/t["n"]:>6.3f}')
    tot_rec = sum(q['recall'] for q in per_q) / n
    tot_mrr = sum(q['mrr'] for q in per_q) / n
    print(f'{"ALL":9} {n:>3} {tot_rec:>9.3f} {tot_mrr:>6.3f}')

    if args.json:
        json.dump({'k': args.k, 'n': n, 'recall': tot_rec, 'mrr': tot_mrr,
                   'per_query': per_q},
                  open(args.json, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print(f'\nper-query results -> {args.json}')

    # failure quick-view: the worst tier's misses
    misses = [q for q in per_q if not q['recall']]
    print(f'\nmisses: {len(misses)}/{n}')
    for q in misses[:12]:
        print(f'  [{q["tier"]:7}] {q["query"][:66]}')
        print(f'            gold: {q["gold"][0][:60] if q["gold"] else "?"}')
        print(f'            got : {q["sources"][0][:60] if q["sources"] else "∅"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())