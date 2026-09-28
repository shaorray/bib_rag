#!/usr/bin/env python3
"""
build_gold_faithfulness.py — build the faithfulness gold set (Feature 3).

Reads gold_retrieval_v2.jsonl ({query, tier, gold: [filenames], gold_title})
and emits gold_faithfulness.jsonl rows:

    {id, query, expected_sources, key_fact, refusal_expected, notes}

  * For the first --limit non-refusal rows: key_fact = first clean sentence
    (>= --min-len chars) extracted verbatim from the expected gold source's
    parent chunks in the parent store. Rows whose source fails to load or
    that contain no clean sentence are SKIPPED and reported (never fabricated
    with an invented key_fact).
  * Then appends --refusal-n refusal_expected=true rows with EMPTY
    expected_sources and queries clearly absent from an Efnb2/ephrin
    mouse-development H&E library (off-topic biomedical queries).

Usage:
    python3 scripts/build_gold_faithfulness.py [--kb eph_rag] [--limit 25]
        [--out <path>] [--gold <path>] [--min-len 40] [--refusal-n 8]
"""
import argparse
import json
import os
import re
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
for _p in (_SRC, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from kb_config import get_config, parse_kb_arg  # noqa: E402


# ── Parent-store access (flat, env-free: explicit store_dir) ───────────────

def _safe_filename(source: str) -> str:
    """Same transform ParentStoreManager/_source_key use."""
    return re.sub(r"[^\w\-]", "_", source or "")[:100]


def load_parents_by_source(store_dir: str, source: str):
    """Return the parent-chunk dicts for one source file, [] when missing."""
    path = os.path.join(store_dir, f"{_safe_filename(source)}.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


# ── Clean-sentence extraction (verbatim, no fabrication) ───────────────────

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_MD_PREFIX = re.compile(r"^\s*(?:[#>\-*•]+\s*|\d+[.)\]]\s*)+")


def first_clean_sentence(text: str, min_len: int = 40) -> str:
    """First sentence of `text` that looks like real prose and is >= min_len.

    'Clean' = markdown/numbering prefixes stripped, ends with sentence
    punctuation, mostly alphabetic, not an ALL-CAPS header. Returns '' when
    nothing qualifies (caller then skips the row — never invents text).
    """
    for sent in _SENT_SPLIT.split(text or ""):
        s = _MD_PREFIX.sub("", sent.strip())
        if len(s) < min_len:
            continue
        if not re.search(r"[.!?][\"')\]]?$", s):
            continue
        letters = sum(ch.isalpha() for ch in s)
        if letters / len(s) < 0.55:
            continue
        if s.upper() == s:
            continue
        return s
    return ""


# ── Designed-refusal queries (absent from an Efnb2/ephrin H&E library) ─────

REFUSAL_QUERIES = [
    "What is the recommended first-line pharmacotherapy for type 2 diabetes mellitus?",
    "How does CAR-T cell therapy work in relapsed B-cell acute lymphoblastic leukemia?",
    "What are the CDC guidelines for influenza vaccination during pregnancy?",
    "Describe the pathophysiology of a vaso-occlusive crisis in sickle cell disease.",
    "Which antibiotics are first-line for community-acquired pneumonia in adults?",
    "What is the mechanism of action of metformin in lowering blood glucose?",
    "How is hepatitis E virus transmitted and what is its typical clinical course?",
    "What surgical approaches are used for aortic valve replacement in elderly patients?",
]


def build_rows(gold_path: str, store_dir: str, limit: int = 25,
               refusal_n: int = 8, min_len: int = 40):
    """Returns (rows, skipped). rows are dict records ready for JSONL;
    skipped = [{id, query, note}] for gold rows that could not be grounded."""
    gold = []
    with open(gold_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                gold.append(json.loads(line))

    rows, skipped = [], []
    n_nonrefusal = 0
    for i, r in enumerate(gold, 1):
        if n_nonrefusal >= limit:
            break
        q = (r.get("query") or "").strip()
        expected = [g for g in (r.get("gold") or []) if g]
        if not q or not expected:
            skipped.append({"id": f"fr-{i:04d}", "query": q,
                            "note": "missing query or gold sources"})
            continue

        key_fact = ""
        loaded_src = None
        for src in expected:
            parents = load_parents_by_source(store_dir, src)
            if not parents:
                continue
            for p in parents:
                key_fact = first_clean_sentence(p.get("content", ""), min_len)
                if key_fact:
                    loaded_src = src
                    break
            if key_fact:
                break

        if not key_fact:
            # skip-with-note: nothing loadable/extractable — never fabricate.
            skipped.append({"id": f"fr-{i:04d}", "query": q,
                            "note": f"gold source not loadable / no clean "
                                    f"sentence: {expected[0]}"})
            continue

        rows.append({
            "id": f"fr-{len(rows) + 1:04d}",
            "query": q,
            "expected_sources": expected,
            "key_fact": key_fact,
            "refusal_expected": False,
            "notes": f"key_fact from {loaded_src}",
        })
        n_nonrefusal += 1

    for j, q in enumerate(REFUSAL_QUERIES[:max(0, refusal_n)], 1):
        rows.append({
            "id": f"rf-{j:04d}",
            "query": q,
            "expected_sources": [],
            "key_fact": "",
            "refusal_expected": True,
            "notes": "designed refusal",
        })
    return rows, skipped


def main():
    argv = parse_kb_arg()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=25,
                    help="non-refusal gold rows to ground (default 25)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--gold", default=None)
    ap.add_argument("--min-len", type=int, default=40)
    ap.add_argument("--refusal-n", type=int, default=8)
    args = ap.parse_args(argv)

    try:
        cfg = get_config()
    except RuntimeError as e:
        print(f"kb_config: {e}", file=sys.stderr)
        return 3
    gold_path = args.gold or os.path.join(cfg["data_dir"],
                                          "gold_retrieval_v2.jsonl")
    out_path = args.out or os.path.join(cfg["data_dir"],
                                        "gold_faithfulness.jsonl")
    if not os.path.isfile(gold_path):
        print(f"gold file missing: {gold_path}", file=sys.stderr)
        return 3
    if not os.path.isdir(cfg["parent_store_dir"]):
        print(f"parent store missing: {cfg['parent_store_dir']}",
              file=sys.stderr)
        return 3

    rows, skipped = build_rows(gold_path, cfg["parent_store_dir"],
                               limit=args.limit, refusal_n=args.refusal_n,
                               min_len=args.min_len)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"gold source : {gold_path}")
    print(f"rows written: {len(rows)} ({len(rows) - args.refusal_n} grounded, "
          f"{args.refusal_n} designed-refusal) -> {out_path}")
    print(f"skipped     : {len(skipped)} (source not loadable / no clean sentence)")
    for s in skipped:
        print(f"  [{s['id']}] {s['query'][:60]}  :: {s['note'][:80]}")
    if len(rows) < 30:
        print(f"WARNING: only {len(rows)} rows (<30 target); raise --limit "
              f"or check the parent store.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())