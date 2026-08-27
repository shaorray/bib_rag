#!/usr/bin/env python3
"""
Classify papers by article_type (review/experimental/methods) + 3-5 topic keywords
using cloud GLM, in BATCHES for speed. Output tags are meant for ChromaDB metadata
`where` filtering on retrieval.

Design notes:
  - topic is a CLOSED controlled vocabulary (not free-form) so tags are stable
    and usable for `where` filters. Each paper gets 3-5 topics.
  - Batching: N titles per LLM call (fewer calls, less reasoning overhead).
  - article_type: "review" | "experimental" | "methods" (single label).

Usage:
    /usr/bin/python3.10 -B scripts/classify_demo.py --sample 20 --batch 4
"""
import argparse
import json
import sys
import time

import chromadb

sys.path.insert(0, "/Disk_bot/RAG/bib_rag/src")
from kb_config import get_config

_CFG = get_config()
CHROMA_DB_PATH = _CFG["chroma_path"]
COLLECTION = _CFG["collection_name"]
# Local Qwen3.8-27B (OpenAI-compatible, llama-server on 5015)
LOCAL_URL = "http://localhost:5015/v1"
LOCAL_MODEL = "/Disk_bot/models/huihui_Qwen3.8-27B-abliterated-GGUF/Huihui-Qwen3.8-27B-abliterated-Q5_K_L.gguf"
# Cloud GLM via Ollama
CLOUD_URL = "http://localhost:11434/v1"
CLOUD_MODEL = "glm-5.2:cloud"

# Controlled topic vocabulary (medium granularity). Model must only pick from these.
TOPICS = [
    # signaling pathways
    "eph-signaling", "notch-signaling", "wnt-signaling", "shh-signaling",
    "tgf-bmp-signaling", "fgf-signaling", "retinoic-acid-signaling",
    "calcium-signaling", "rho-gtpase-signaling", "other-signaling",
    # biological processes
    "neural-development", "neural-guidance", "vascular-development",
    "cancer", "cell-adhesion", "cell-migration", "boundary-formation",
    "morphogenesis", "cell-fate",
    # domain / tech
    "spatial-transcriptomics", "single-cell", "imaging", "computational", "in-silico",
]

TOPIC_VOCAB_JSON = json.dumps(TOPICS, ensure_ascii=False)

PROMPT = f"""You are a scientific paper classifier. Given paper titles, classify each into:
- article_type: one of "review" (综述/survey/overview/perspective), "experimental"
  (实验研究), "methods" (方法学/工具/benchmark/计算流程).
- topics: a LIST of 3 to 5 keywords selected ONLY from this controlled vocabulary:
{TOPIC_VOCAB_JSON}

Rules:
- article_type is a single label; topics is a list of 3-5 distinct keywords from the
  vocabulary above. If none fits, use "other-signaling" or the closest generic term.
- Do NOT invent keywords outside the list.

Return for each paper a compact JSON object of the form:
{{"<index>": {{"article_type": "...", "topics": ["...", "...", "..."]}}}}
Output ONLY valid JSON, no explanation.
"""


def classify_batch(titles: list[str], client, llm_model: str) -> dict:
    """Classify a batch of titles. Returns {batch_index -> {article_type, topics}}."""
    try:
        numbered = "\n".join(f"{i}: {t}" for i, t in enumerate(titles))
        resp = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": "Classify these papers:\n" + numbered},
            ],
            temperature=0.0,
            max_tokens=3000,  # reasoning model needs room
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


def cleanest(title: str) -> str:
    """Collapse noisy source-style title to a clean short title."""
    if " - " in title:
        return title.split(" - ")[-1].strip()
    return title


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--backend", choices=["cloud", "local"], default="cloud",
                    help="cloud=GLM via Ollama (fast, needs network); local=Qwen3.8-27B (offline)")
    args = ap.parse_args()

    if args.backend == "local":
        llm_url, llm_model = LOCAL_URL, LOCAL_MODEL
    else:
        llm_url, llm_model = CLOUD_URL, CLOUD_MODEL

    from openai import OpenAI
    client = OpenAI(base_url=llm_url, api_key="ollama")

    col = chromadb.PersistentClient(path=CHROMA_DB_PATH).get_collection(COLLECTION)
    r = col.get(include=["metadatas"])
    seen = {}
    for m in r["metadatas"]:
        src = m.get("source", "unknown")
        if src not in seen:
            seen[src] = {"title": m.get("title", ""), "year": m.get("year", "")}
    papers = list(seen.items())
    print(f"总论文数: {len(papers)}，抽样 {args.sample} 篇，batch={args.batch}", file=sys.stderr)

    sample = papers[:args.sample]
    # Clean titles up front, keep alignment with source.
    sample_clean = [(src, meta, cleanest(meta["title"])) for src, meta in sample]

    t0 = time.time()
    results = []
    batches = [sample_clean[i:i + args.batch] for i in range(0, len(sample_clean), args.batch)]
    for bi, batch in enumerate(batches):
        titles = [t for _, _, t in batch]
        got = classify_batch(titles, client, llm_model)
        if "error" in got:
            print(f"[batch {bi}] ERROR: {got['error']}", file=sys.stderr)
            for src, meta, t in batch:
                results.append({"source": src, "title": t, "error": got["error"]})
            continue
        # map by index key
        for j, (src, meta, t) in enumerate(batch):
            key = str(j)
            entry = got.get(key) or got.get(str(j + 1))
            if entry:
                results.append({"source": src, "title": t,
                                "article_type": entry.get("article_type", "?"),
                                "topics": entry.get("topics", [])})
            else:
                results.append({"source": src, "title": t, "error": f"missing index {key}"})
        done = min((bi + 1) * args.batch, len(sample_clean))
        print(f"[batch {bi+1}/{len(batches)}] done {done}/{len(sample_clean)} ({time.time()-t0:.0f}s)", file=sys.stderr)
        time.sleep(0.3)

    print(f"\n=== 分类结果 ({len(results)} 篇, 耗时 {time.time()-t0:.0f}s) ===")
    from collections import Counter
    at = Counter(x.get("article_type", x.get("error", "?")) for x in results)
    tp = Counter()
    for x in results:
        for t in x.get("topics", []):
            tp[t] += 1
    print("文章类型分布:", dict(at))
    print("topic 词频 top30:", dict(tp.most_common(30)))
    print("---")
    for r in results:
        at = r.get("article_type", r.get("error", "?"))
        tp = ", ".join(r.get("topics", [])) or "-"
        print(f"{str(at):12} [{tp}] | {r['title'][:55]}")


if __name__ == "__main__":
    main()
