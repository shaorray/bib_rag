#!/usr/bin/env python3
"""
Classify Cadherin papers (article_type + open-vocab topics) using LOCAL Qwen,
reading clean titles from cadherin_labeled.csv (not the noisy md-extracted title).

Output: CSV with source (PMID.md), year, title, article_type, topics — same format
as classify_all.py so backfill_cadherin.py can consume it.

Usage:
    /usr/bin/python3.10 -B classify_cadherin.py \
        --labeled /Disk_bot/Eph/Cadherin_papers/metadata/cadherin_labeled.csv \
        --resume outputs/cadherin_tags.csv
"""
import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, "/Disk_bot/RAG/bib_rag/src")
from kb_config import get_config

_CFG = get_config()

LOCAL_URL = "http://localhost:5015/v1"
LOCAL_MODEL = "/Disk_bot/models/huihui_Qwen3.8-27B-abliterated-GGUF/Huihui-Qwen3.8-27B-abliterated-Q5_K_L.gguf"
CLOUD_URL = "http://localhost:11434/v1"
CLOUD_MODEL = "glm-5.2:cloud"
OUT_DIR = _CFG["outputs_dir"]

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

PROMPT = f"""You are a scientific paper classifier. Given paper titles, classify each into:
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
- article_type is a single label.
- topics must be an array of 3-5 distinct strings. Never reuse the same keyword twice.

Return for each paper a compact JSON object of the form:
{{"<index>": {{"article_type": "...", "topics": ["...", "...", "..."]}}}}
Output ONLY valid JSON, no explanation.
"""


def classify_batch(titles, client, llm_model):
    try:
        numbered = "\n".join(f"{i}: {t}" for i, t in enumerate(titles))
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
    ap.add_argument("--labeled", required=True)
    ap.add_argument("--backend", choices=["cloud", "local"], default="local")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--resume", type=str, default=os.path.join(OUT_DIR, "cadherin_tags.csv"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if args.backend == "local":
        llm_url, llm_model = LOCAL_URL, LOCAL_MODEL
    else:
        llm_url, llm_model = CLOUD_URL, CLOUD_MODEL

    os.makedirs(OUT_DIR, exist_ok=True)
    done_set = load_done(args.resume)

    from openai import OpenAI
    client = OpenAI(base_url=llm_url, api_key="not-required")

    # Load clean titles from labeled.csv, keyed by PMID — but ONLY for papers
    # whose md file actually exists (i.e. indexed into ChromaDB).
    md_dir = "/Disk_bot/Eph/Cadherin_papers/md"
    md_files = set()
    if os.path.isdir(md_dir):
        md_files = {f for f in os.listdir(md_dir) if f.endswith(".md")}
    papers = []
    with open(args.labeled, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pmid = row.get("pmid", "").strip()
            title = row.get("title", "").strip()
            if not pmid or not title:
                continue
            src = f"{pmid}.md"
            if src not in md_files:
                continue  # only classify papers we actually indexed
            papers.append({
                "source": src,
                "year": row.get("pub_year", "").strip(),
                "title": title,
            })
    papers = [p for p in papers if p["source"] not in done_set]
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
        titles = [p["title"] for p in batch]
        got = classify_batch(titles, client, llm_model)
        if "error" in got:
            errs += len(batch)
            for p in batch:
                writer.writerow({"source": p["source"], "year": p["year"], "title": p["title"],
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
            writer.writerow({"source": p["source"], "year": p["year"], "title": p["title"],
                             "article_type": at, "topics": json.dumps(seen_tp[:5])})
            done += 1
        fh.flush()
        if (bi + 1) % 25 == 0 or (bi + 1) == len(batches):
            el = time.time() - t0
            print(f"[{done}/{len(papers)}] {el:.0f}s ({done/el:.1f}/s) errs={errs}", file=sys.stderr)

    fh.close()
    print(f"完成: {done} 篇, {errs} 错误, 总耗时 {time.time()-t0:.0f}s", file=sys.stderr)
    print(f"结果: {args.resume}")


if __name__ == "__main__":
    main()
