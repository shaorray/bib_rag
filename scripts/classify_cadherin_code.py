#!/usr/bin/env python3
"""
Classify cadherin_code review papers (article_type + open-vocab topics) using
LOCAL Qwen, feeding title + first ~200 words of md content as pseudo-abstract.

Unlike classify_cadherin.py which reads clean titles from labeled.csv, this
script reads md files directly, extracts title via index_single_paper's
extract_meta(), and takes the first ~200 words as abstract context.

Output: CSV with source, year, title, article_type, topics — same format
so backfill_cadherin.py / apply_tags.py can consume it.

Usage:
    /usr/bin/python3.10 -B scripts/classify_cadherin_code.py \
        --md-dir /Disk_bot/Eph/review_proj/cadherin_code/papers/md \
        --md-dir /Disk_bot/Eph/review_proj/cadherin_code/papers \
        --resume outputs/cadherin_code_tags.csv
"""
import argparse
import csv
import json
import os
import sys
import time

LOCAL_URL = "http://localhost:5015/v1"
LOCAL_MODEL = "/Disk_bot/models/huihui_Qwen3.8-27B-abliterated-GGUF/Huihui-Qwen3.8-27B-abliterated-Q5_K_L.gguf"
CLOUD_URL = "http://localhost:11434/v1"
CLOUD_MODEL = "glm-5.2:cloud"
OUT_DIR = "/Disk_bot/Eph/bib_rag/outputs"

sys.path.insert(0, "/Disk_bot/Eph/bib_rag/src")
from index_single_paper import clean_text, truncate_at_references, extract_meta

TOPIC_EXAMPLES = [
    "eph-signaling", "notch-signaling", "wnt-signaling", "shh-signaling",
    "tgf-bmp-signaling", "fgf-signaling", "retinoic-acid-signaling",
    "calcium-signaling", "rho-gtpase-signaling",
    "neural-development", "neural-guidance", "vascular-development",
    "cancer", "cell-adhesion", "cell-migration", "boundary-formation",
    "morphogenesis", "cell-fate",
    "spatial-transcriptomics", "single-cell", "imaging", "computational", "in-silico",
]
TOPIC_EXAMPLES_JSON = json.dumps(TOPIC_EXAMPLES, ensure_ascii=False)

PROMPT = f"""You are a scientific paper classifier. Given paper titles and abstracts, classify each into:
- article_type: one of "review" (综述/survey/overview/perspective), "experimental"
  (实验研究), "methods" (方法学/工具/benchmark/计算流程).
- topics: a LIST of 3 to 5 concise keywords capturing the paper's biological/technical
  themes. This is an OPEN vocabulary — invent accurate keywords as needed; do not force
  a paper into a topic that does not fit.

Normalization rules for topics:
- lowercase, nouns or noun phrases (e.g. "neural development", "cell adhesion").
- multi-word phrases: separate words with a space, do NOT use underscores.
- use standard, well-established domain terminology.
- 3 to 5 keywords per paper. Example style (not a limit, and not exclusive):
{TOPIC_EXAMPLES_JSON}

Rules:
- article_type is a single label. Use "methods" if the paper develops a new method,
  tool, model, simulation, or computational pipeline — even if the title doesn't say "method".
- Use "review" for surveys, perspectives, overviews, or papers that primarily synthesize
  existing literature rather than presenting new experiments.
- Use "experimental" for papers presenting new experimental findings (wet-lab or dry-lab).
- topics must be an array of 3-5 distinct strings. Never reuse the same keyword twice.

Return for each paper a compact JSON object of the form:
{{"<index>": {{"article_type": "...", "topics": ["...", "...", "..."]}}}}
Output ONLY valid JSON, no explanation.
"""


def extract_title_and_abstract(md_path):
    """Extract title + first ~200 words as pseudo-abstract from an md file."""
    text = open(md_path, encoding="utf-8", errors="ignore").read()
    if len(text.strip()) < 500:
        return "", "", ""
    cleaned = truncate_at_references(clean_text(text))
    if len(cleaned.strip()) < 500:
        return "", "", ""
    meta = extract_meta(cleaned, os.path.basename(md_path))
    title = meta.get("title", os.path.basename(md_path).replace(".md", ""))
    year = meta.get("year", "")
    # Take first ~200 words as abstract
    words = cleaned.split()[:200]
    abstract = " ".join(words)
    return title, year, abstract


def classify_batch(papers, client, llm_model):
    """papers = list of (source, title, abstract)"""
    try:
        numbered = "\n".join(
            f"{i}: {p[1]} | Abstract: {p[2]}"
            for i, p in enumerate(papers)
        )
        resp = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": "Classify these papers:\n" + numbered},
            ],
            temperature=0.0,
            max_tokens=4000,
        )
        content = resp.choices[0].message.content.strip()
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1:
            return {"error": f"no json: {content[:100]}"}
        parsed = json.loads(content[start:end + 1])
        out = {}
        for k, v in parsed.items():
            if isinstance(v, dict) and "article_type" in v:
                out[k] = v
        return out
    except Exception as e:
        return {"error": str(e)}


def load_done(path):
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as f:
        return {row["source"] for row in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md-dir", action="append", required=True,
                    help="Directory containing md files (can repeat)")
    ap.add_argument("--backend", choices=["cloud", "local"], default="local")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--resume", type=str,
                    default=os.path.join(OUT_DIR, "cadherin_code_tags.csv"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if args.backend == "local":
        llm_url, llm_model = LOCAL_URL, LOCAL_MODEL
    else:
        llm_url, llm_model = CLOUD_URL, CLOUD_MODEL

    os.makedirs(OUT_DIR, exist_ok=True)
    done_set = load_done(args.resume)

    from openai import OpenAI
    client = OpenAI(base_url=llm_url, api_key="ollama")

    # Collect md files from all dirs
    papers = []
    for md_dir in args.md_dir:
        for f in sorted(os.listdir(md_dir)):
            if not f.endswith(".md"):
                continue
            if f == "DOWNLOADED_PAPERS.md":
                continue
            if f in done_set:
                continue
            path = os.path.join(md_dir, f)
            if os.path.getsize(path) < 500:
                continue
            title, year, abstract = extract_title_and_abstract(path)
            if not title:
                continue
            papers.append((f, title, year, abstract))

    if args.limit:
        papers = papers[:args.limit]
    print(f"待分类: {len(papers)} 篇 (已跳过 {len(done_set)})", file=sys.stderr)

    new_file = not os.path.exists(args.resume)
    fh = open(args.resume, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=["source", "year", "title", "article_type", "topics"])
    if new_file:
        writer.writeheader()
    fh.flush()

    t0 = time.time()
    done = errs = 0
    batches = [papers[i:i + args.batch] for i in range(0, len(papers), args.batch)]
    for bi, batch in enumerate(batches):
        batch_papers = [(p[0], p[1], p[3]) for p in batch]  # (source, title, abstract)
        got = classify_batch(batch_papers, client, llm_model)
        if "error" in got:
            errs += len(batch)
            for p in batch:
                writer.writerow({"source": p[0], "year": p[1], "title": p[1],
                                 "article_type": "ERROR", "topics": got["error"][:200]})
            fh.flush()
            continue
        for j, p in enumerate(batch):
            entry = got.get(str(j)) or got.get(str(j + 1)) or {}
            at = entry.get("article_type", "?")
            tp = entry.get("topics", [])
            if isinstance(tp, str):
                tp = [tp]
            seen_tp = []
            for x in tp:
                x = str(x).strip().lower()
                if x and x not in seen_tp:
                    seen_tp.append(x)
            writer.writerow({"source": p[0], "year": p[1], "title": p[1],
                             "article_type": at, "topics": json.dumps(seen_tp[:5])})
            done += 1
        fh.flush()
        if (bi + 1) % 10 == 0 or (bi + 1) == len(batches):
            el = time.time() - t0
            print(f"[{done}/{len(papers)}] {el:.0f}s ({done/el:.1f}/s) errs={errs}",
                  file=sys.stderr)

    fh.close()
    print(f"完成: {done} 篇, {errs} 错误, 总耗时 {time.time()-t0:.0f}s", file=sys.stderr)
    print(f"结果: {args.resume}")


if __name__ == "__main__":
    main()