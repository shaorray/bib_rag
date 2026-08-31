#!/usr/bin/env python3
"""
normalize_topics.py — Collapse the open-vocabulary topic tags into a
controlled vocabulary with an alias map.

Why: classify_papers.py lets the LLM invent keywords freely, so the same
concept accumulates spelling variants ("cell adhesion" / "cell-adhesion" /
"cell-cell adhesion", "e-cadherin" vs "cadherins", ...). The agent's
where={"topic_eph-signaling": 1} filter silently misses papers tagged with a
variant. This tool builds the canonical vocabulary once and rewrites the
library (chroma topics + topic_<kw> boolean keys) to match.

Pipeline (all stages cached under <library>/outputs/topic_norm/):
  1. Read per-source topic lists from chroma (single offset scroll).
  2. Embed every distinct keyword (bge-m3, hyphens→spaces).
  3. Greedy COMPLETE-LINKAGE clustering on cosine sim (threshold 0.90;
     plain single-link union-find at 0.85 over-chains unrelated keywords).
  4. Canonical name per cluster:
       a. deterministic — identical token multiset (hyphen/space variants):
          prefer hyphenated form, then highest document frequency;
       b. if the cluster has semantically distinct members (size ≥2 after (a)):
          local Qwen picks the most standard established term.
  5. Emit topic_aliases.json (keyword → canonical), rewrite every affected
     source in chroma (per-chunk topic_* None-out + rewrite, matching
     apply_tags.py semantics) and rewrite the tags CSV if --tags given.

Re-running is idempotent: already-canonical vocabularies produce no writes.

Usage (with a wrapper, e.g. eph-rag / geo-rag):
    <name>-rag scripts/normalize_topics.py --dry-run
    <name>-rag scripts/normalize_topics.py --apply --tags outputs/tags_full.csv
"""
import argparse
import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/ (bib_utils)
try:  # bib_rag-package-try
    from .kb_config import get_config
except ImportError:  # flat (loose-script mode)
    from kb_config import get_config
try:  # bib_rag-package-try
    from .library_config import get_setting
except ImportError:  # flat (loose-script mode)
    from library_config import get_setting

_CFG = get_config()

SIM_THRESHOLD = 0.90        # complete-linkage cosine threshold
EMBED_BATCH = 256


def toks(t):
    """Token multiset key that ignores hyphen/space variants."""
    return tuple(sorted(t.replace("-", " ").split()))


def slug(t):
    """Canonical form: lowercase, words joined with single hyphens."""
    t = t.strip().lower().replace(" ", "-")
    return re.sub(r"[^a-z0-9-]", "", t)


def load_source_topics(col):
    """source -> [keywords] from chroma, plus which chunks carry which variant."""
    src_topics = {}
    sources_at = {}
    offset, page = 0, 5000
    while True:
        r = col.get(include=["metadatas"], limit=page, offset=offset)
        ids, metas = r["ids"] or [], r["metadatas"] or []
        if not ids:
            break
        for cid, m in zip(ids, metas):
            s = m.get("source", "")
            if not s:
                continue
            try:
                tps = json.loads(m.get("topics") or "[]")
            except Exception:
                tps = []
            if s not in src_topics:
                src_topics[s] = tps
                sources_at[s] = m.get("article_type", "")
            if tps and src_topics[s] != tps:
                print(f"[warn] {s}: chunk topic lists differ; keeping first",
                      file=sys.stderr)
        offset += page
    return src_topics, sources_at


def embed_keywords(terms, emb_url, use_raw=False):
    url = emb_url.replace("/v1/embeddings", "/embedding") if use_raw else emb_url
    out = {}
    for i in range(0, len(terms), EMBED_BATCH):
        batch = [t.replace("-", " ") for t in terms[i:i + EMBED_BATCH]]
        resp = _post_embeddings(url, batch)
        for j, d in enumerate(resp):
            out[terms[i + j]] = d
    return out


def _post_embeddings(url, batch, tries=3):
    import requests
    payload = {"input": batch, "model": "bge-m3"}
    last = None
    for a in range(tries):
        try:
            r = requests.post(url, json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and "data" in data:
                return [d["embedding"] for d in data["data"]]
            if isinstance(data, list):       # /embedding raw endpoint
                return data
        except Exception as e:
            last = e
            time.sleep(2 * (a + 1))
    raise RuntimeError(f"embedding failed after {tries} tries: {last}")


def complete_link_clusters(terms, emb, threshold=SIM_THRESHOLD):
    """Greedy complete-linkage: merge only if EVERY cross-pair ≥ threshold."""
    X = np.array([emb[t] for t in terms], dtype=np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    S = X @ X.T
    idx = {t: i for i, t in enumerate(terms)}
    members = {i: {i} for i in range(len(terms))}
    parent = list(range(len(terms)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    pairs = []
    n = len(terms)
    for i in range(n):
        row = S[i]
        for j in np.where(row > threshold)[0]:
            if j > i:
                pairs.append((row[j], i, int(j)))
    for sim, i, j in sorted(pairs, reverse=True):
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        if min(S[x, y] for x in members[ri] for y in members[rj]) >= threshold:
            parent[rj] = ri
            members[ri] |= members[rj]
            del members[rj]
    return [[terms[i] for i in c] for c in members.values()]


def deterministic_canon(cluster, vocab):
    """Hyphen/space variant resolution within one cluster (no LLM).

    Group members by token multiset; per group prefer hyphenated form,
    tie-break by document frequency then length. Returns (label_map, leftover)
    where leftover are keyword groups whose token multisets genuinely differ.
    """
    groups = defaultdict(list)
    for t in cluster:
        groups[toks(t)].append(t)
    label_map = {}
    leftover = {}
    for key, grp in groups.items():
        if len(grp) > 1:
            hyph = [t for t in grp if "-" in t]
            pick = hyph[0] if hyph else max(
                grp, key=lambda t: (vocab.get(t, 0), -len(t)))
            for t in grp:
                if t != pick:
                    label_map[t] = slug(pick)
            leftover[slug(pick)] = vocab.get(pick, 0)
        else:
            leftover[slug(grp[0])] = vocab.get(grp[0], 0)
    return label_map, leftover


def llm_canonical(leftover_terms, vocab, llm_url, llm_model):
    """For mixed clusters: pick ONE canonical label per multi-member cluster."""
    import requests
    aliases = {}

    def canon(names):
        lines = "\n".join(f"- {t} ({vocab.get(t, 0)} papers)"
                          for t in sorted(names, key=lambda x: -vocab.get(x, 0)))
        try:
            r = requests.post(llm_url, timeout=120, json={
                "model": llm_model,
                "messages": [
                    {"role": "system", "content":
                        "You merge duplicate scientific topic keywords into ONE "
                        "canonical label. Answer with ONLY the label: lowercase, "
                        "noun phrase, hyphens between words, max 3 words. Prefer "
                        "the most standard established term."},
                    {"role": "user", "content":
                        f"Members with paper counts:\n{lines}\n\nCanonical label:"}],
                "temperature": 0.0, "max_tokens": 30})
            lab = r.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            return None
        lab = lab.split("\n")[0].strip().strip('"').rstrip(".").lower()
        lab = re.sub(r"[^a-z0-9 -]", "", lab).strip().replace(" ", "-")
        return lab or None

    for cluster in leftover_terms:
        members = sorted(cluster, key=lambda x: -vocab.get(x, 0))
        lab = canon(cluster)
        if lab:
            for t in cluster:
                sl = slug(t)
                if sl != lab:
                    aliases[sl] = lab
        else:  # fall back to the most frequent member
            head = slug(members[0])
            for t in cluster:
                if slug(t) != head:
                    aliases[slug(t)] = head
    return aliases


def main():
    ap = argparse.ArgumentParser(
        description="Normalize topic keywords into a controlled vocabulary")
    ap.add_argument("--dry-run", action="store_true",
                    help="build + save alias map, don't touch chroma")
    ap.add_argument("--apply", action="store_true", help="rewrite chroma metadata")
    ap.add_argument("--tags", help="tags CSV to rewrite alongside chroma")
    ap.add_argument("--threshold", type=float, default=SIM_THRESHOLD)
    ap.add_argument("--batch", type=int, default=2000, help="chunks per update")
    ap.add_argument("--force-llm", action="store_true",
                    help="use the LLM even on single-member clusters (no)")
    args = ap.parse_args()
    if args.force_llm:
        sys.exit("--force-llm is intentionally unsupported: single-variant "
                 "keywords are left untouched.")

    import chromadb
    col = chromadb.PersistentClient(path=_CFG["chroma_path"]).get_collection(
        _CFG["collection_name"])

    t0 = time.time()
    src_topics, sources_at = load_source_topics(col)
    print(f"sources: {len(src_topics)} ({time.time()-t0:.0f}s)")

    vocab = Counter()
    for tps in src_topics.values():
        vocab.update(tps)
    terms = sorted(vocab)
    print(f"distinct keywords: {len(terms)}")

    outdir = Path(_CFG["outputs_dir"]) / "topic_norm"
    outdir.mkdir(parents=True, exist_ok=True)

    emb_cache = outdir / "keyword_embeddings.npz"
    if emb_cache.exists():
        z = np.load(emb_cache, allow_pickle=True)
        saved_t, saved_e = list(z["terms"]), z["emb"]
        emb = {t: saved_e[i] for i, t in enumerate(saved_t) if t in vocab}
        missing = [t for t in terms if t not in emb]
        if missing:
            emb.update(embed_keywords(missing, _CFG["embed_url"]))
            all_t = sorted(emb)
            np.savez(emb_cache, terms=all_t,
                     emb=np.array([emb[t] for t in all_t], dtype=np.float32))
    else:
        emb = embed_keywords(terms, _CFG["embed_url"])
        np.savez(emb_cache, terms=terms,
                 emb=np.array([emb[t] for t in terms], dtype=np.float32))
    print(f"embedded {len(emb)} keywords ({time.time()-t0:.0f}s)")

    clusters = complete_link_clusters(terms, emb, threshold=args.threshold)
    multi = sum(1 for c in clusters if len(c) > 1)
    print(f"clusters: {len(clusters)} total, {multi} with >1 member")

    # stage 4a: deterministic within-cluster variant folding
    aliases = {}
    llm_inputs = []
    for c in clusters:
        lm, leftover = deterministic_canon(c, vocab)
        aliases.update(lm)
        if len(c) > 1:
            names = sorted({*leftover, *{slug(t) for t in lm.values() if t}})
            groups = defaultdict(list)
            for t in c:
                groups[resolve_slug(t, aliases)].append(t)
            if len(groups) > 1:
                llm_inputs.append(
                    [max(g, key=lambda t: vocab.get(t, 0)) for g in groups.values()])
    # stage 4b: LLM canonical for remaining multi-name clusters
    llm_url = f"{_CFG.get('llm_url', 'http://localhost:5015/v1')}/chat/completions"
    llm_model = get_setting(_CFG["data_root"], "classify_model", "")
    if llm_model:
        llm_aliases = llm_canonical(llm_inputs, vocab, llm_url, llm_model)
    else:
        print("[warn] no classify_model set; using most-frequent-member fallback")
        llm_aliases = {}
    # apply LLM aliases AFTER deterministic (llm label chains on top)
    for k, v in llm_aliases.items():
        base = aliases.get(v, v)
        aliases[k] = base
    final_map = {t: resolve_slug(t, aliases) for t in terms}
    canon_freq = Counter()
    for t in terms:
        canon_freq[final_map[t]] += vocab[t]
    print(f"keywords {len(terms)} -> canonical topics {len(canon_freq)} "
          f"({time.time()-t0:.0f}s)")

    alias_path = outdir / "topic_aliases.json"
    alias_path.write_text(json.dumps(
        {"map": final_map, "canonical_frequencies": dict(canon_freq)},
        indent=1, ensure_ascii=False))
    print(f"alias map -> {alias_path}")

    # per-source new topic lists (drop exact-dup collapses, cap at 5)
    new_topics = {}
    for s, tps in src_topics.items():
        seen = []
        for t in tps:
            c = final_map.get(t, slug(t)) if t else ""
            if c and c not in seen:
                seen.append(c)
        new_topics[s] = sorted(seen)[:5]

    if args.dry_run or not args.apply:
        changed = sum(1 for s in new_topics
                      if sorted(src_topics[s]) != new_topics[s])
        print(f"DRY RUN: {changed}/{len(new_topics)} sources would change; "
              f"re-run with --apply to write")
        return

    # stage 5: rewrite chroma (apply_tags semantics: None-out topic_* then add)
    import chromadb.api  # noqa: F401
    batch_ids, batch_metas = [], []
    updated = 0
    offset, page = 0, 5000
    while True:
        r = col.get(include=["metadatas"], limit=page, offset=offset)
        ids, metas = r["ids"] or [], r["metadatas"] or []
        if not ids:
            break
        for cid, m in zip(ids, metas):
            s = m.get("source", "")
            tp = new_topics.get(s)
            if tp is None:
                continue
            new = dict(m)
            new["topics"] = json.dumps(tp, ensure_ascii=False)
            for k in [k for k in new if k.startswith("topic_")]:
                new[k] = None
            for kw in tp:
                new[f"topic_{kw}"] = 1
            batch_ids.append(cid)
            batch_metas.append(new)
            if len(batch_ids) >= args.batch:
                col.update(ids=batch_ids, metadatas=batch_metas)
                updated += len(batch_ids)
                batch_ids, batch_metas = [], []
                print(f"...{updated} chunks ({time.time()-t0:.0f}s)",
                      file=sys.stderr)
        offset += page
    if batch_ids:
        col.update(ids=batch_ids, metadatas=batch_metas)
        updated += len(batch_ids)
    print(f"chroma: {updated} chunks rewritten across {len(new_topics)} sources")

    if args.tags:
        rewrite_tags_csv(args.tags, final_map)


def rewrite_tags_csv(path, final_map):
    """Rewrite the topics column of a tags CSV through the alias map."""
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    for r in rows:
        try:
            tps = json.loads(r.get("topics") or "[]")
        except Exception:
            tps = []
        out = []
        for t in tps:
            c = final_map.get(str(t).strip().lower()) or slug(str(t))
            if c and c not in out:
                out.append(c)
        r["topics"] = json.dumps(sorted(out)[:5], ensure_ascii=False)
    fieldnames = ["source", "year", "title", "article_type", "topics"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"tags CSV rewritten: {path} ({len(rows)} rows)")


def resolve_slug(t, aliases):
    seen = set()
    cur = slug(t)
    while cur in aliases and cur not in seen:
        seen.add(cur)
        cur = aliases[cur]
    return cur


if __name__ == "__main__":
    main()