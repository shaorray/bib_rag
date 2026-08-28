#!/usr/bin/env python3
"""
classify_papers.py — General-purpose paper classification for any RAG library.

Classifies papers into article_type (review/experimental/methods) + 3-5 topic
keywords using an LLM (local Qwen on 5015 by default, or cloud via --backend
cloud). Output is a resumable CSV (source, year, title, article_type, topics)
consumed by scripts/apply_tags.py — writing to ChromaDB is decoupled.

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

Domain adaptation: pass --topics-example with a comma-separated list of seed
topics for your domain (style anchors only — the vocabulary is open). If you
need a *closed* controlled vocabulary instead, edit TOPIC_EXAMPLES + PROMPT.

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
import sys
import time
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
TOPIC_EXAMPLES_JSON = json.dumps(TOPIC_EXAMPLES, ensure_ascii=False)

PROMPT = f"""You are a scientific paper classifier. Given paper titles (optionally with abstracts), classify each into:
- article_type: one of "review" (综述/survey/overview/perspective), "experimental"
  (实验研究), "methods" (方法学/工具/benchmark/计算流程).
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


def classify_batch(papers, client, llm_model):
    """papers: list of (source, text_to_classify). Returns {index: {...}} or {"error": ...}."""
    try:
        numbered = "\n".join(f"{i}: {t}" for i, (_, t) in enumerate(papers))
        resp = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": "Classify these papers:\n" + numbered},
            ],
            temperature=0.0,
            max_tokens=3000,
        )
        content = resp.choices[0].message.content.strip()
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1:
            return {"error": f"no json: {content[:100]}"}
        parsed = json.loads(content[start:end + 1])
        return {k: v for k, v in parsed.items()
                if isinstance(v, dict) and "article_type" in v}
    except Exception as e:
        return {"error": str(e)}


def load_done(path):
    done = set()
    if not os.path.exists(path):
        return done
    import csv
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done.add(row["source"])
    return done


def collect_from_titles_csv(path, title_col, year_col):
    """Clean-metadata mode: rows keyed by source with title/year columns."""
    import csv
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
    ap.add_argument("--batch", type=int, default=4, help="papers per LLM call")
    ap.add_argument("--limit", type=int, help="classify only the first N (testing)")
    ap.add_argument("--backend", choices=["local", "cloud"], default="local")
    ap.add_argument("--domain-topics", help="comma-separated seed topics for your domain")
    args = ap.parse_args()

    # LLM client
    if args.backend == "local":
        LOCAL_URL = "http://localhost:5015/v1"
        LOCAL_MODEL = os.environ.get(
            "CLASSIFY_MODEL",
            "/Disk_bot/models/huihui_Qwen3.8-27B-abliterated-GGUF/Huihui-Qwen3.8-27B-abliterated-Q5_K_L.gguf")
        llm_url, llm_model = LOCAL_URL, LOCAL_MODEL
    else:
        llm_url, llm_model = "http://localhost:11434/v1", "glm-5.2:cloud"

    if args.domain_topics:
        global TOPIC_EXAMPLES_JSON, PROMPT
        seeds = [t.strip() for t in args.domain_topics.split(",") if t.strip()]
        TOPIC_EXAMPLES_JSON = json.dumps(seeds, ensure_ascii=False)
        PROMPT = PROMPT.split("\n\nNormalization rules")[0].replace(
            json.dumps(TOPIC_EXAMPLES, ensure_ascii=False), TOPIC_EXAMPLES_JSON) + \
            """

Normalization rules for topics:
- lowercase, nouns or noun phrases (e.g. "neural development", "cell adhesion").
- multi-word phrases: separate words with a space, do NOT use underscores.
- use standard, well-established domain terminology.
- 3 to 5 keywords per paper.

Rules:
- article_type is a single label.
- topics must be an array of 3-5 distinct strings. Never reuse the same keyword twice.

Return for each paper a compact JSON object of the form:
{"<index>": {"article_type": "...", "topics": ["...", "...", "..."]}}
Output ONLY valid JSON, no explanation.
"""

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
    t0 = __import__("time").time()
    errs = done_count = 0

    with open(args.resume, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["source", "year", "title", "article_type", "topics"])
        if new_file:
            writer.writeheader()
        for bi in range(0, len(todo), args.batch):
            batch = todo[bi:bi + args.batch]
            got = classify_batch([(p[0], p[1]) for p in batch], client, llm_model)
            if "error" in got:
                errs += len(batch)
                for p in batch:
                    writer.writerow({"source": p[0], "year": p[2], "title": p[1],
                                     "article_type": "ERROR", "topics": str(got["error"])[:200]})
                fh.flush()
                continue
            for j, p in enumerate(batch):
                entry = got.get(str(j), {})
                at = entry.get("article_type", "?")
                tps = entry.get("topics", [])
                seen_tp = []
                for x in tps:
                    x = str(x).strip().lower()
                    if x and x not in seen_tp:
                        seen_tp.append(x)
                writer.writerow({"source": p[0], "year": p[2], "title": p[1],
                                 "article_type": at, "topics": json.dumps(seen_tp[:5])})
                done_count += 1
            fh.flush()
            rate = done_count / (time.time() - t0) if done_count else 0
            print(f"  [{bi + len(batch)}/{len(todo)}] ok={done_count} err={errs} ({rate:.1f}/s)")


if __name__ == "__main__":
    main()