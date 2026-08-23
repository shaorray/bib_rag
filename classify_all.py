#!/usr/bin/env python3
"""
Full-corpus classification using LOCAL Qwen3.8-27B (offline, no API cost).

Classifies every unique paper (by source) into article_type (review/experimental/
methods) + 3-5 topic keywords from the controlled vocabulary. Writes results to a
CSV (source, year, article_type, topics) so the run is resumable and decoupled from
writing to ChromaDB.

Usage:
    /usr/bin/python3.10 -B classify_all.py --batch 2 --resume out/tags.csv
    # or switch backend:
    /usr/bin/python3.10 -B classify_all.py --batch 4 --backend cloud --resume out/tags.csv
"""
import argparse
import csv
import json
import os
import sys
import time

import chromadb

CHROMA_DB_PATH = "/Disk_bot/Eph/bib_rag/chroma_db_new"
OUT_DIR = "/Disk_bot/Eph/bib_rag/outputs"
LOCAL_URL = "http://localhost:5015/v1"
LOCAL_MODEL = "/Disk_bot/models/huihui_Qwen3.8-27B-abliterated-GGUF/Huihui-Qwen3.8-27B-abliterated-Q5_K_L.gguf"
CLOUD_URL = "http://localhost:11434/v1"
CLOUD_MODEL = "glm-5.2:cloud"

# Examples to guide topic style (OPEN vocabulary — model may use these or invent
# new ones for papers from other domains). Kept for normalization consistency only.
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


def cleanest(title: str) -> str:
    if " - " in title:
        return title.split(" - ")[-1].strip()
    return title


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
    """Return set of sources already tagged (for resume)."""
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as f:
        return {row["source"] for row in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["cloud", "local"], default="local")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--resume", type=str, default=os.path.join(OUT_DIR, "tags.csv"))
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    if args.backend == "local":
        llm_url, llm_model = LOCAL_URL, LOCAL_MODEL
    else:
        llm_url, llm_model = CLOUD_URL, CLOUD_MODEL

    os.makedirs(OUT_DIR, exist_ok=True)
    done_set = load_done(args.resume)

    from openai import OpenAI
    client = OpenAI(base_url=llm_url, api_key="ollama")

    col = chromadb.PersistentClient(path=CHROMA_DB_PATH).get_collection("bib_rag_papers")
    r = col.get(include=["metadatas"])
    seen = {}
    for m in r["metadatas"]:
        src = m.get("source", "unknown")
        if src not in seen:
            seen[src] = {"title": m.get("title", ""), "year": m.get("year", "")}
    papers = [(src, meta, cleanest(meta["title"])) for src, meta in seen.items()]
    papers = [p for p in papers if p[0] not in done_set]  # skip done
    if args.limit:
        papers = papers[:args.limit]
    print(f"待分类: {len(papers)} 篇 (已跳过 {len(done_set)} 篇已处理)", file=sys.stderr)

    # append mode
    new_file = not os.path.exists(args.resume)
    fh = open(args.resume, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=["source", "year", "title", "article_type", "topics"])
    if new_file:
        writer.writeheader()
    fh.flush()

    t0 = time.time()
    done = 0
    errs = 0
    batches = [papers[i:i + args.batch] for i in range(0, len(papers), args.batch)]
    for bi, batch in enumerate(batches):
        titles = [t for _, _, t in batch]
        got = classify_batch(titles, client, llm_model)
        if "error" in got:
            errs += len(batch)
            print(f"[batch {bi}] ERROR: {got['error']}", file=sys.stderr)
            for src, meta, t in batch:
                writer.writerow({"source": src, "year": meta["year"], "title": t,
                                 "article_type": "ERROR", "topics": got["error"][:200]})
            fh.flush()
            continue
        for j, (src, meta, t) in enumerate(batch):
            entry = got.get(str(j)) or got.get(str(j + 1)) or {}
            at = entry.get("article_type", "?")
            tp = entry.get("topics", [])
            if isinstance(tp, str):
                tp = [tp]
            # OPEN vocabulary: keep model's own keywords, just dedupe + cap at 5
            seen_tp = []
            for x in tp:
                x = str(x).strip().lower()
                if x and x not in seen_tp:
                    seen_tp.append(x)
            tp = seen_tp[:5]
            writer.writerow({"source": src, "year": meta["year"], "title": t,
                             "article_type": at, "topics": json.dumps(tp)})
            done += 1
        fh.flush()
        if (bi + 1) % 25 == 0 or (bi + 1) == len(batches):
            el = time.time() - t0
            rate = done / el if el > 0 else 0
            print(f"[{done}/{len(papers)}] {el:.0f}s ({rate:.1f}/s) errs={errs}", file=sys.stderr)

    fh.close()
    print(f"完成: {done} 篇新分类, {errs} 错误, 总耗时 {time.time()-t0:.0f}s", file=sys.stderr)
    print(f"结果: {args.resume}")


if __name__ == "__main__":
    main()
