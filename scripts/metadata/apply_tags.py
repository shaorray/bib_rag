#!/usr/bin/env python3
"""
Apply article_type + topics tags from classification CSV(s) into the ChromaDB
collection metadata. Preserves all existing metadata fields. Per chunk sharing
the same source it writes:

  article_type (str)          — "review" | "experimental" | "methods"
  topics       (JSON string)  — '["kw1","kw2","kw3"]'
  topic_<kw>   (int 1)        — one boolean key per keyword, so the agent can
                                  filter with where={"article_type": "review"}
                                  or {"$and": [{...}, {"topic_x": 1}]}
                                  (see agent_prompts.py Metadata filtering).

Performance (2026-08-29 benchmark, 481k-chunk library): the previous
per-source filtered-get + update path cost ~3.4 s/source (~140 min for a full
library; single-chunk updates ~46 ms each). This version does ONE offset
scroll over the collection and rewrites only changed chunks in batches of
--batch, ~0.7 ms/chunk: full pass ≈2 + update ≈5 min.

Existing topic_* boolean keys are stripped before re-writing, so re-running
with an updated CSV never leaves stale keywords behind.

Usage:
    /usr/bin/python3.10 -B scripts/metadata/apply_tags.py --csv outputs/tags.csv
    /usr/bin/python3.10 -B scripts/metadata/apply_tags.py --csv a.csv --csv b.csv --dry-run
"""
import argparse
import csv
import json
import sys
import time

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ (bib_utils, zotero_access)
from kb_config import get_config

_CFG = get_config()
CHROMA_DB_PATH = _CFG["chroma_path"]
COLLECTION = _CFG["collection_name"]


def load_tags(paths):
    """source -> {article_type, topics}; later CSVs win on duplicate sources."""
    tags = {}
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                src = row["source"]
                at = row.get("article_type", "")
                if at in ("ERROR", "", "?"):
                    continue
                tp = []
                raw = row.get("topics", "")
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, list):
                        tp = [str(x).strip().lower() for x in parsed if str(x).strip()][:5]
                except Exception:
                    tp = []
                tags[src] = {"article_type": at, "topics": tp}
    return tags


def parse_topics(raw):
    """topics cell -> (json_str, [keywords]); tolerates string-schema rows
    like '"experimental: a, b, c"' or already-parsed lists."""
    if isinstance(raw, list):
        tps = [str(x).strip().lower() for x in raw if str(x).strip()][:5]
        return json.dumps(tps, ensure_ascii=False), tps
    raw = (raw or "").strip()
    if not raw:
        return json.dumps([]), []
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise ValueError("not a list")
        tps = [str(x).strip().lower() for x in parsed if str(x).strip()][:5]
        return json.dumps(tps, ensure_ascii=False), tps
    except Exception:
        # salvage '"article_type: t1, t2, t3'" string form
        if ":" in raw:
            _at, _, rest = raw.partition(":")
            if _at.strip().lower() in ("review", "experimental", "methods"):
                rest = rest.strip().strip('"[]')
                tps = [x.strip().lower() for x in rest.split(",") if x.strip()][:5]
                return json.dumps(tps, ensure_ascii=False), tps
        return json.dumps([]), []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, action="append",
                    help="tags CSV from classify_papers.py (repeatable)")
    ap.add_argument("--dry-run", action="store_true", help="only report, don't write")
    ap.add_argument("--batch", type=int, default=2000, help="chunks per update call")
    args = ap.parse_args()

    tags = load_tags(args.csv)
    print(f"loaded {len(tags)} tagged sources from {len(args.csv)} CSV(s)")

    import chromadb
    col = chromadb.PersistentClient(path=CHROMA_DB_PATH).get_collection(COLLECTION)

    t0 = time.time()
    matched_sources = set()
    batch_ids, batch_metas = [], []
    updated = skipped_scan = 0
    skip_keys = set()
    offset, page = 0, 5000

    def flush():
        nonlocal updated, batch_ids, batch_metas
        if not batch_ids:
            return
        if not args.dry_run:
            col.update(ids=batch_ids, metadatas=batch_metas)
        updated += len(batch_ids)
        batch_ids, batch_metas = [], []
        if updated % 20000 < args.batch:
            print(f"...{updated} chunks ({time.time()-t0:.0f}s)", file=sys.stderr)

    while True:
        r = col.get(include=["metadatas"], limit=page, offset=offset)
        ids, metas = r["ids"] or [], r["metadatas"] or []
        if not ids:
            break
        for cid, m in zip(ids, metas):
            src = m.get("source", "")
            tag = tags.get(src)
            if tag is None:
                continue
            matched_sources.add(src)
            topics_json, tps = parse_topics(tag["topics"])
            new = dict(m)
            new["article_type"] = tag["article_type"]
            new["topics"] = topics_json
            # replace stale boolean topic keys
            if not skip_keys:
                for k in m:
                    if k.startswith("topic_"):
                        skip_keys.add(k)
            for k in skip_keys:
                new.pop(k, None)
            for kw in tps:
                if kw:
                    new[f"topic_{kw}"] = 1
            batch_ids.append(cid)
            batch_metas.append(new)
            if len(batch_ids) >= args.batch:
                flush()
        offset += page
    flush()

    untagged = set(tags) - matched_sources
    print(f"done: {updated} chunks across {len(matched_sources)} sources updated, "
          f"{len(untagged)} tagged sources had no chunks, {time.time()-t0:.0f}s")
    if untagged:
        preview = sorted(untagged)[:10]
        for s in preview:
            print(f"  [no chunks] {s}", file=sys.stderr)
    if args.dry_run:
        print("DRY RUN — no writes performed")


if __name__ == "__main__":
    main()