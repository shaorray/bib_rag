#!/usr/bin/env python3
"""
tree_retrieve.py — reasoning-based tree retrieval over a bib_rag tree index.

The second half of bib_rag's tree mode: given a query and a tree index built
by tree_index.py, an LLM reasons its way down the hierarchy (the way a human
expert turns to the right section of a long report) and returns the matching
nodes plus their page text.

This is the vectorless counterpart to query_bib_rag.py: no embeddings, no
chunk similarity — explicit tree navigation with a traceable path.

Usage:
    python3 -B src/tree_retrieve.py "query" --source <stem> [--top 5]
    python3 -B src/tree_retrieve.py "query" --source <stem> --show-tree
    python3 -B src/tree_retrieve.py --list

Requires a tree built with: bibrag run src/tree_index.py <doc> [--mode flash]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:  # bib_rag-package-try
    from .kb_config import get_config, parse_kb_arg
except ImportError:  # flat (loose-script mode)
    from kb_config import get_config, parse_kb_arg

_argv = parse_kb_arg()
if _argv:
    sys.argv = [sys.argv[0]] + _argv
_CFG = get_config()

try:  # bib_rag-package-try
    from .tree_index import load_index, list_indexes, _llm_settings, _litellm_model_name
except ImportError:
    from tree_index import load_index, list_indexes, _llm_settings, _litellm_model_name


# ─── LLM call (OpenAI-compatible, same endpoint as the rest of bib_rag) ─────
def _llm_json(system: str, user: str, max_tokens: int = 2000) -> dict:
    import requests

    cfg = _llm_settings()
    url = cfg["url"].rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if cfg.get("key") and cfg["key"] != "not-required":
        headers["Authorization"] = f"Bearer {cfg['key']}"
    body = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    # local reasoning models: keep content non-empty
    if os.environ.get("TREE_NO_REASONING", "1") == "1":
        body["reasoning_effort"] = "none"
    r = requests.post(url, headers=headers, json=body, timeout=300)
    r.raise_for_status()
    data = r.json()
    text = data["choices"][0]["message"]["content"] or ""
    return _extract_json(text)


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass
    return {"raw": text}


# ─── tree navigation ───────────────────────────────────────────────────────
def _flatten(nodes: list, depth: int = 0) -> list[dict]:
    out = []
    for n in nodes or []:
        out.append({"node_id": n.get("node_id"), "title": n.get("title", ""),
                    "start_index": n.get("start_index"), "end_index": n.get("end_index"),
                    "summary": n.get("summary") or n.get("prefix_summary") or "",
                    "depth": depth, "has_children": bool(n.get("nodes"))})
        out.extend(_flatten(n.get("nodes") or [], depth + 1))
    return out


def _outline_lines(nodes: list) -> list[str]:
    """One formatted line per node (the outline's unit of pagination)."""
    lines = []
    for n in _flatten(nodes):
        pad = "  " * n["depth"]
        summ = (n["summary"] or "")[:160].replace("\n", " ")
        lines.append(f'{pad}[{n["node_id"]}] {n["title"]} '
                     f'(p{n["start_index"]}-{n["end_index"]})'
                     + (f' — {summ}' if summ else ""))
    return lines


def _outline_for_prompt(nodes: list, max_chars: int = 12000) -> str:
    return "\n".join(_outline_lines(nodes))[:max_chars]


def paginate_outline(nodes: list, part: int = 1, budget: int = 12000) -> dict:
    """Split the outline into ~budget-char pages instead of truncating it.

    Borrowed from PageIndex agent_tools._split_structure (MIT), adapted to
    line granularity: a long document's outline can exceed what one prompt
    should carry, and a hard truncation silently drops the back half of the
    tree — so the LLM can never navigate there. Pagination keeps every node
    reachable; the caller walks parts until has_more is false.

    Returns {'part', 'total_parts', 'has_more', 'text'}.
    """
    lines = _outline_lines(nodes)
    pages, cur, size = [], [], 0
    for line in lines:
        if cur and size + len(line) + 1 > budget:
            pages.append(cur)
            cur, size = [], 0
        cur.append(line)
        size += len(line) + 1
    if cur:
        pages.append(cur)
    pages = pages or [[""]]
    part = max(1, min(int(part), len(pages)))
    return {
        "part": part,
        "total_parts": len(pages),
        "has_more": part < len(pages),
        "text": "\n".join(pages[part - 1]),
    }


def _find_node(nodes: list, node_id: str):
    for n in nodes or []:
        if n.get("node_id") == node_id:
            return n
        hit = _find_node(n.get("nodes") or [], node_id)
        if hit:
            return hit
    return None


_SYSTEM = (
    "You are a document-retrieval reasoning agent. You are given the outline "
    "(tree) of ONE long document: each node has an id, title, page range, and "
    "a summary. Your job: select the node(s) most relevant to the user's query "
    "by reasoning over the outline — relevance, not surface similarity. Prefer "
    "the deepest node that directly covers the query; use parent nodes only if "
    "the query spans multiple sections. Return STRICT JSON:\n"
    '{"nodes": ["<node_id>", ...], "reason": "<one sentence>"}\n'
    "Return 1-5 node ids. If nothing is relevant, return an empty list."
)

# Grounding + citation contract for answers built from tree hits. Borrowed
# from PageIndex's LOCAL_CITATION_PROMPTS (MIT): answer only from what was
# actually read, never fill a gap from general knowledge, and tag every claim
# with the node + page span it came from so the reader can verify it.
CITATION_PROTOCOL = """\
GROUNDING
- Answer only from the tree nodes actually read via tree retrieval. State only what the retrieved node text supports.
- Never fill a gap from general knowledge. When the document does not answer the question, say so plainly.

CITATIONS
- Cite every statement supported by a node's text as a bracketed reference: [<node_id>, p. <start>-<end>] — the node id and page span exactly as returned.
- Never invent or alter a node id or page span. If a claim draws on several nodes, cite each (at most 3); beyond that, cite the single strongest node.
- A citation must reference the node whose OWN pages carry the claim, not an ancestor that merely contains it."""


def retrieve_by_node(node_id: str, stem: str, out_dir: str | None = None) -> dict:
    """Read ONE node's own pages directly (no reasoning) — the 'open this section'
    path. Returns the same hit shape as retrieve() so callers treat both alike."""
    manifest = load_index(stem, out_dir=out_dir)
    if manifest is None:
        raise FileNotFoundError(
            f"no tree index for {stem!r}; build one with "
            f"'bibrag run src/tree_index.py <doc>'")
    tree = manifest["tree"]
    node = _find_node(tree, str(node_id))
    if not node:
        return {"stem": stem, "source": manifest.get("source"),
                "query": None, "reason": None, "hits": []}
    pages = _load_pages(stem, out_dir)
    start, end = node.get("start_index"), node.get("end_index")
    hit = {
        "node_id": node.get("node_id"),
        "title": node.get("title"),
        "start_index": start, "end_index": end,
        "summary": node.get("summary") or node.get("prefix_summary"),
        "text": _slice_pages(pages, start, end),
    }
    return {"stem": stem, "source": manifest.get("source"),
            "query": None, "reason": None, "hits": [hit]}


def retrieve(query: str, stem: str, top: int = 5, out_dir: str | None = None,
             budget: int = 12000) -> dict:
    manifest = load_index(stem, out_dir=out_dir)
    if manifest is None:
        raise FileNotFoundError(
            f"no tree index for {stem!r}; build one with "
            f"'bibrag run src/tree_index.py <doc>'")

    tree = manifest["tree"]
    # Reason over the outline. For a tree longer than one prompt budget, walk
    # EVERY part and let the LLM pick from each, then merge the picks — a hard
    # truncation would silently hide the back half of the tree from reasoning
    # (the very failure pagination exists to prevent).
    picked: list[str] = []
    reasons: list[str] = []
    page_no, total_parts = 1, 1
    while page_no <= total_parts:
        page = paginate_outline(tree, part=page_no, budget=budget)
        total_parts = page["total_parts"]
        user = (f"Document: {manifest.get('source')}\n"
                f"(outline part {page['part']}/{page['total_parts']})\n\n"
                f"Outline:\n{page['text']}\n\nQuery: {query}")
        result = _llm_json(_SYSTEM, user)
        ids = result.get("nodes") or []
        if isinstance(ids, str):
            ids = [ids]
        for nid in ids:
            if nid not in picked:
                picked.append(nid)
        if result.get("reason"):
            reasons.append(str(result["reason"]))
        page_no += 1

    ids = picked[:top]
    pages = _load_pages(stem, out_dir)
    hits = []
    for nid in ids:
        node = _find_node(tree, str(nid))
        if not node:
            continue
        start, end = node.get("start_index"), node.get("end_index")
        text = _slice_pages(pages, start, end)
        hits.append({
            "node_id": node.get("node_id"),
            "title": node.get("title"),
            "start_index": start, "end_index": end,
            "summary": node.get("summary") or node.get("prefix_summary"),
            "text": text,
        })

    return {
        "stem": stem,
        "source": manifest.get("source"),
        "query": query,
        "reason": reasons[0] if reasons else None,
        "parts_searched": total_parts,
        "hits": hits,
    }


def _load_pages(stem: str, out_dir: str | None):
    try:  # bib_rag-package-try
        from .tree_index import _trees_dir
    except ImportError:
        from tree_index import _trees_dir
    p = _trees_dir(_CFG, out_dir) / f"{stem}.pages.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _slice_pages(pages, start, end) -> str:
    if isinstance(pages, str):
        return pages[:4000]
    if not isinstance(pages, list) or start is None:
        return ""
    lo = max(0, int(start) - 1)
    hi = int(end) if end else lo + 1
    return "\n\n".join(pages[lo:hi])[:8000]


# ─── CLI ────────────────────────────────────────────────────────────────────
def _cli() -> int:
    ap = argparse.ArgumentParser(description="Reasoning-based retrieval over a bib_rag tree index")
    ap.add_argument("query", nargs="?", help="the question")
    ap.add_argument("--source", required=False, help="tree stem (from tree_index.py)")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--show-tree", action="store_true", help="print the outline only")
    ap.add_argument("--json", action="store_true", help="emit raw JSON")
    ap.add_argument("--list", action="store_true", help="list available tree indexes")
    args = ap.parse_args()

    if args.list:
        rows = list_indexes(args.out)
        for r in rows:
            print(f"  {r['stem']:40s} {r['mode']:8s} {r['page_count']}p")
        return 0

    if not args.source:
        ap.error("--source <stem> is required (see --list)")
    if args.show_tree:
        m = load_index(args.source, out_dir=args.out)
        if not m:
            print(f"no tree index for {args.source!r}", file=sys.stderr)
            return 2
        print(_outline_for_prompt(m["tree"], max_chars=100000))
        return 0
    if not args.query:
        ap.error("need a query")

    res = retrieve(args.query, args.source, top=args.top, out_dir=args.out)
    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0
    print(f"Q: {args.query}")
    print(f"source: {res['source']}  ({res['stem']})")
    if res.get("parts_searched", 1) > 1:
        print(f"outline parts searched: {res['parts_searched']}")
    if res.get("reason"):
        print(f"reason: {res['reason']}")
    print()
    for h in res["hits"]:
        print(f"[{h['node_id']}] {h['title']}  (p{h['start_index']}-{h['end_index']})")
        if h.get("text"):
            print(h["text"][:1200])
        print("-" * 60)
    if not res["hits"]:
        print("(no matching node)")
    else:
        print()
        print(CITATION_PROTOCOL)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
