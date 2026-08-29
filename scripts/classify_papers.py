#!/usr/bin/env python3
"""
classify_papers.py — General-purpose paper classification for any RAG library.

Classifies papers into article_type (review/experimental/methods) + 3 topic
keywords using an LLM (local Qwen on 5015 by default, or cloud via --backend
cloud). Output is a resumable CSV (source, year, title, article_type, topics)
consumed by scripts/metadata/apply_tags.py — writing to ChromaDB is decoupled.

Three input modes (pick one):

  --titles-csv FILE   A CSV with a `source` column (the md filename as stored
                      in chroma metadata) plus title/year columns. Use this when
                      you have clean metadata (e.g. a PubMed labeled.csv with
                      title/authors/doi/year keyed by PMID.md).
                      Optional: --title-col / --year-col to rename columns.

  --md-dir DIR        One or more markdown directories. The script extracts the
                      title from the md content (via index_single_paper's
                      extract_meta) and uses the first ~200 words as a
                      pseudo-abstract — titles alone miss method papers, so
                      title+abstract is recommended.

  --from-chroma       Classify every source already indexed in the library
                      (title read from chunk metadata). Useful for tagging a
                      library that was built before classification existed.

Performance notes (benchmarked on local Qwen3.8-27B, 2 slots, 2026-08-29):
  - Decode throughput saturates ~55 tok/s; the only real lever is OUTPUT size,
    not prompt size or batch count. The default string-value schema
    ('"0": "experimental: t1, t2, t3"') emits ~17 tok/paper vs ~47 for the
    detailed array schema: ~3.2x faster (3.2-3.5 papers/s vs 0.7-1.2).
  - article_type agreement with the detailed schema measured 92% (48/48 papers
    re-classified both ways); topics shrink 3-5 -> 3 keywords.
  - Pass --schema detailed if you need the old 3-5 keyword array format
    (slower; kept for CSV-format compatibility).
  - --workers N dispatches N batches concurrently (default 2 = llama-server
    slot count). Extra workers beyond server slots add no throughput.

Domain adaptation: pass --domain-topics with a comma-separated list of seed
topics for your domain (style anchors only — the vocabulary is open).

Usage (with a wrapper, e.g. eph-rag / geo-rag):
    <name>-rag scripts/classify_papers.py --titles-csv metadata/labeled.csv \
        --resume outputs/tags.csv
    <name>-rag scripts/classify_papers.py --md-dir /path/to/md --resume outputs/tags.csv
    <name>-rag scripts/classify_papers.py --from-chroma --resume outputs/tags.csv
"""
import argparse
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from kb_config import get_config  # noqa: E402

_CFG = get_config()
OUT_DIR = _CFG["outputs_dir"]

TOPIC_EXAMPLES = [
    "signaling", "development", "cell-adhesion", "cell-migration",
    "morphogenesis", "cancer", "spatial-transcriptomics", "single-cell",
    "imaging", "computational", "in-silico",
]

# ─── Schemas ────────────────────────────────────────────────────────────────
# string (default): one JSON string value per paper — cheapest to decode.
PROMPT_STRING = (
    "Classify scientific papers. For each numbered paper output article_type "
    "(one of review|experimental|methods) and exactly 3 topics (lowercase, "
    "1-2 words each, standard terminology, comma-separated).\n"
    'Return ONLY compact JSON with numeric keys and STRING values. '
    'Example: {"0": "experimental: cell adhesion, migration, signaling", '
    '"1": "review: ephrin signaling, development"'
)

# detailed: the original {"article_type": ..., "topics": [...]} shape.
TOPIC_EXAMPLES_JSON = json.dumps(TOPIC_EXAMPLES, ensure_ascii=False)
PROMPT_DETAILED = f"""You are a scientific paper classifier. Given paper titles (optionally with abstracts), classify each into:
- article_type: one of "review" (survey/overview/perspective), "experimental"
  (hypothesis-driven lab research), "methods" (protocols/tools/benchmark/computational workflows).
- topics: a LIST of 3 to 5 concise keywords capturing the paper's scientific/technical
  themes. This is an OPEN vocabulary — invent accurate keywords as needed; do not force
  a paper into a topic that does not fit.

Normalization rules for topics:
- lowercase, nouns or noun phrases (e.g. "neural development", "cell adhesion").
- multi-word phrases: separate words with a space, do NOT use underscores.
- use standard, well-established domain terminology.
- 3 to 5 keywords per paper. Example style (not a limit, and not exclusive):
{TOPIC_EXAMPLES_JSON}

Rules:
- article_type is a single label.
- topics must be an array of 3-5 distinct strings. Never reuse the same keyword twice.

Return for each paper a compact JSON object of the form:
{{"<index>": {{"article_type": "...", "topics": ["...", "...", "..."]}}}}
Output ONLY valid JSON, no explanation.
"""


def parse_string_entry(v):
    """Parse a string-schema value 'experimental: t1, t2, t3' -> (type, [topics])."""
    if not isinstance(v, str) or ":" not in v:
        return "", []
    at, tps = v.split(":", 1)
    topics = [x.strip().lower() for x in tps.split(",") if x.strip()]
    return at.strip().lower(), topics[:5]


# Salvage regexes for truncated/broken model JSON (observed in the wild).
_STRING_ROW_RE = re.compile(r'"(\d+)"\s*:\s*"([^"{}]+?)"')
_DETAILED_ROW_RE = re.compile(
    r'"(\d+)"\s*:\s*\{\s*"article_type"\s*:\s*"([a-z]+)"\s*,?\s*'
    r'"topics"\s*:\s*\[(.*?)\]', re.S)


def salvage_string_rows(txt):
    """Recover {index: value-string} pairs from broken JSON output."""
    return {m.group(1): m.group(2) for m in _STRING_ROW_RE.finditer(txt)}


def salvage_detailed_rows(txt):
    """Recover {index: (article_type, topics_list)} from broken JSON output."""
    out = {}
    for m in _DETAILED_ROW_RE.finditer(txt):
        topics = [t.strip().strip('"\'') for t in m.group(3).split(",") if t.strip()]
        out[m.group(1)] = (m.group(2), topics)
    return out


def _extract_json_obj(content):
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(content[start:end + 1])
    except json.JSONDecodeError:
        return None


def classify_batch(papers, client, llm_model, schema="string", max_tokens_cap=8000):
    """Classify [(source, text), ...] in one LLM call.

    Returns {index:int -> (article_type, topics:list)} or {"error": msg}.
    Tolerant to truncated JSON: salvages per-row regex matches.
    """
    numbered = "\n".join(f"{i}: {t}" for i, (_, t) in enumerate(papers))
    prompt = PROMPT_STRING if schema == "string" else PROMPT_DETAILED
    # output budget: ~17 tok/paper (string) or ~50 tok/paper (detailed) + headroom
    per_paper = 20 if schema == "string" else 60
    max_tokens = min(per_paper * len(papers) + 300, max_tokens_cap)
    try:
        resp = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": numbered},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        content = resp.choices[0].message.content.strip()
        parsed = _extract_json_obj(content)
        out = {}
        if parsed is None:
            # broken json — salvage raw rows instead of failing the whole batch
            if schema == "string":
                parsed = salvage_string_rows(content)
                if not parsed:
                    return {"error": f"no salvagable rows: {content[:100]}"}
            else:
                salv = salvage_detailed_rows(content)
                if not salv:
                    return {"error": f"no salvagable rows: {content[:100]}"}
                return {int(k): (v[0], [t.lower() for t in v[1]])
                        for k, v in salv.items() if isinstance(v, tuple)}
        for k, v in parsed.items():
            try:
                idx = int(k)
            except (TypeError, ValueError):
                continue
            if schema == "string" and isinstance(v, str):
                at, tps = parse_string_entry(v)
            elif isinstance(v, dict):
                at = str(v.get("article_type", "")).strip().lower()
                raw_tps = v.get("topics", [])
                tps = []
                for x in raw_tps:
                    x = str(x).strip().lower()
                    if x and x not in tps:
                        tps.append(x)
            else:
                continue
            if at in ("review", "experimental", "methods") and tps:
                out[idx] = (at, tps[:5])
        return out if out else {"error": f"no valid entries: {content[:100]}"}
    except Exception as e:
        return {"error": str(e)}


def load_done(path):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done.add(row["source"])
    return done


def collect_from_titles_csv(path, title_col, year_col):
    """Clean-metadata mode: rows keyed by source with title/year columns."""
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            src = (row.get("source") or "").strip()
            if not src:
                continue
            title = (row.get(title_col) or "").strip()
            year = (row.get(year_col) or "").strip()
            # classify on title alone when we have no abstract — acceptable for
            # curated metadata, but md mode is preferred when abstracts matter
            out.append((src, title, year))
    return out


def collect_from_md_dirs(md_dirs):
    """md mode: extract title + pseudo-abstract from each file."""
    from index_single_paper import clean_text, truncate_at_references, extract_meta
    out = []
    for md_dir in md_dirs:
        d = Path(md_dir)
        files = sorted(f for f in d.iterdir() if f.suffix == ".md" and f.stat().st_size >= 500)
        print(f"[scan] {d}: {len(files)} md files")
        for f in files:
            text = f.read_text(encoding="utf-8", errors="ignore")
            cleaned = truncate_at_references(clean_text(text))
            if len(cleaned.strip()) < 500:
                continue
            meta = extract_meta(cleaned, f.name)
            title = meta.get("title", f.name)
            words = cleaned.split()
            pseudo_abstract = " ".join(words[:200])
            year = str(meta.get("year", ""))
            out.append((f.name, f"{title}. {pseudo_abstract}", year))
    return out


def collect_from_chroma():
    """Chroma mode: every indexed source, title from chunk metadata."""
    import chromadb
    col = chromadb.PersistentClient(path=_CFG["chroma_path"]).get_collection(_CFG["collection_name"])
    seen = {}
    offset, page = 0, 5000
    while True:
        r = col.get(include=["metadatas"], limit=page, offset=offset)
        if not r["ids"]:
            break
        for m in r["metadatas"]:
            src = m.get("source", "")
            if src and src not in seen:
                seen[src] = (m.get("title", ""), str(m.get("year", "")))
        offset += page
    return [(src, t, y) for src, (t, y) in sorted(seen.items())]


def main():
    ap = argparse.ArgumentParser(description="Classify papers in the active RAG library")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--titles-csv", help="CSV with clean metadata (source + title/year columns)")
    src.add_argument("--md-dir", action="append", help="markdown dir(s); repeatable")
    src.add_argument("--from-chroma", action="store_true",
                     help="classify every source already indexed in the library")
    ap.add_argument("--title-col", default="title", help="title column in --titles-csv")
    ap.add_argument("--year-col", default="year", help="year column in --titles-csv")
    ap.add_argument("--resume", default=os.path.join(OUT_DIR, "tags.csv"))
    ap.add_argument("--batch", type=int, default=64, help="papers per LLM call")
    ap.add_argument("--workers", type=int, default=2,
                    help="concurrent LLM calls (match llama-server slots)")
    ap.add_argument("--limit", type=int, help="classify only the first N (testing)")
    ap.add_argument("--backend", choices=["local", "cloud"], default="local")
    ap.add_argument("--schema", choices=["string", "detailed"], default="string",
                    help="string = fast compact output (default, ~3x faster); "
                         "detailed = legacy 3-5 keyword arrays")
    ap.add_argument("--domain-topics", help="comma-separated seed topics for your domain")
    args = ap.parse_args()

    global PROMPT_STRING, PROMPT_DETAILED
    if args.domain_topics:
        seeds = [t.strip() for t in args.domain_topics.split(",") if t.strip()]
        seed_line = "Example style (not a limit): " + json.dumps(seeds, ensure_ascii=False)
        PROMPT_STRING += "\n" + seed_line
        PROMPT_DETAILED = PROMPT_DETAILED.replace(TOPIC_EXAMPLES_JSON,
                                                  json.dumps(seeds, ensure_ascii=False))

    # LLM client
    if args.backend == "local":
        LOCAL_URL = "http://localhost:5015/v1"
        from library_config import get_setting
        LOCAL_MODEL = get_setting(_CFG["data_root"], "classify_model", "")
        if not LOCAL_MODEL:
            sys.exit("ERROR: --backend local needs a local GGUF path. Set it via: "
                     "(1) <library>/config.json settings.classify_model, or "
                     "(2) CLASSIFY_MODEL env var. Example: "
                     "CLASSIFY_MODEL=/path/to/model.gguf")
        llm_url, llm_model = LOCAL_URL, LOCAL_MODEL
    else:
        llm_url, llm_model = "http://localhost:11434/v1", "glm-5.2:cloud"

    from openai import OpenAI
    client = OpenAI(base_url=llm_url, api_key="ollama")

    # collect papers
    if args.titles_csv:
        papers = collect_from_titles_csv(args.titles_csv, args.title_col, args.year_col)
    elif args.md_dir:
        papers = collect_from_md_dirs(args.md_dir)
    else:
        papers = collect_from_chroma()

    done = load_done(args.resume)
    todo = [p for p in papers if p[0] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"papers: {len(papers)} total, {len(done)} already classified, {len(todo)} to do")
    if not todo:
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    new_file = not os.path.exists(args.resume)
    t0 = time.time()
    errs = done_count = 0

    def run_one(batch):
        """One LLM call with a single retry on failure."""
        papers_text = [(p[0], p[1]) for p in batch]
        got = classify_batch(papers_text, client, llm_model, schema=args.schema)
        if "error" in got and len(batch) > 1:
            got = classify_batch(papers_text, client, llm_model, schema=args.schema)
        return got

    with open(args.resume, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["source", "year", "title", "article_type", "topics"])
        if new_file:
            writer.writeheader()

        batches = [todo[i:i + args.batch] for i in range(0, len(todo), args.batch)]
        workers = max(1, min(args.workers, len(batches)))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            # ex.map preserves batch order so resume-CSV rows stay deterministic
            for bi, (batch, got) in enumerate(
                    zip(batches, ex.map(run_one, batches))):
                if "error" in got:
                    errs += len(batch)
                    for p in batch:
                        writer.writerow({"source": p[0], "year": p[2], "title": p[1],
                                         "article_type": "ERROR",
                                         "topics": str(got["error"])[:200]})
                    fh.flush()
                    continue
                for j, p in enumerate(batch):
                    at, tps = got.get(j, ("?", []))
                    writer.writerow({"source": p[0], "year": p[2], "title": p[1],
                                     "article_type": at,
                                     "topics": json.dumps(tps[:5])})
                    done_count += 1
                fh.flush()
                elapsed = time.time() - t0
                rate = done_count / elapsed if done_count else 0
                eta_min = (len(todo) - done_count) / rate / 60 if rate else 0
                print(f"  [{(bi + 1) * args.batch}/{len(todo)}] ok={done_count} "
                      f"err={errs} ({rate:.1f}/s, eta {eta_min:.0f} min)")

    print(f"done: {done_count} classified, {errs} failed "
          f"({time.time()-t0:.0f}s, {done_count/max(time.time()-t0, 0.01):.2f}/s)")


if __name__ == "__main__":
    main()