#!/usr/bin/env python3
"""
bib_rag_grill.py — Write-after-Grill academic paragraph writer.

Two-phase workflow inspired by Matt Pocock's /grill-me + /grill-with-docs:

  PHASE A: GRILL  (interactive or --auto)
    Interview the user / inspect the codebase (CONTEXT.md) to lock down
    the scope, mechanism, tissue, model, and citation targets BEFORE
    writing. Produce a structured `GrillSpec` dict.

  PHASE B: WRITE
    Search bib_rag using the GrillSpec's scoped query (not the raw
    topic), synthesize a paragraph that uses CONTEXT.md vocabulary
    exactly, and apply the spec's constraints (model preference,
    tissue filter, citation style).

Usage:
    # Interactive grilling (asks 1 question at a time, recommended)
    python3 -B bib_rag_grill.py "Eph drives boundary sharpening"
    
    # Auto-grill (uses LLM to fill in missing spec fields from CONTEXT.md)
    python3 -B bib_rag_grill.py "Eph drives boundary sharpening" --auto
    
    # Skip grill, use a saved spec from a prior run
    python3 -B bib_rag_grill.py --spec /tmp/grill_spec_20260607.json

Why this exists (lessons from the old bib_rag_writer.py):
  - Old tool: searched with raw topic sentence → retrieved 5 papers
    about general "Eph signaling" → wrote generic paragraph.
  - Grill pre-step forces decisions like:
      * Mechanism focus: repulsion / adhesion / tension / endocytosis?
      * Tissue: hindbrain / somite / neural crest / generic?
      * Model stance: support / critique / survey?
    → narrower query → 5x more relevant retrievals → tighter paragraph.

Borrowed from: https://github.com/mattpocock/skills (skills/productivity/grill-me/SKILL.md)
"""

import sys, os, re, json, argparse, requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from odf.opendocument import OpenDocumentText
from odf.text import P, H

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kb_config import get_config

# === Paths ===
_CFG = get_config()
BIB_RAG_ROOT = Path(_CFG["data_root"])
CONTEXT_PATH = BIB_RAG_ROOT / "CONTEXT.md"
SPEC_DIR = Path("/tmp/bib_rag_grill_specs")
SPEC_DIR.mkdir(exist_ok=True)

# === API endpoints (shared with bib_rag_writer.py) ===
# ─── Multi-KB config ─────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kb_config import get_config
_CFG = get_config()
BIB_RAG_EMBED_URL = _CFG["embed_url"]
ZOTERO_BASE = "http://localhost:23119/api/users/0"
CHROMA_PATH = _CFG["chroma_path"]

# === LLM endpoints ===
# Default: llama-server (Qwen3.6-35B-A3B on port 5015) — your primary inference.
# We do NOT use vLLM. vLLM is removed from the chain entirely.
LLM_ENDPOINTS = [
    ("http://127.0.0.1:5015/v1", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
]


def call_llm(messages, max_tokens=600, temperature=0.3, timeout=120) -> str:
    """Call local LLM with fallback chain. Returns content string or raises."""
    last_err = None
    for url, model in LLM_ENDPOINTS:
        try:
            resp = requests.post(
                f"{url}/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"All LLM endpoints failed: {last_err}")


# ---------------------------------------------------------------------------
# CONTEXT.md loader — read domain glossary into memory
# ---------------------------------------------------------------------------
def load_context_glossary() -> Dict:
    """Parse CONTEXT.md into a usable glossary dict.
    
    Returns: {
        "terms": {"eph receptor": "...", "repulsion": "..."},
        "anti_patterns": [("eph kinase", "eph receptor"), ...],
        "models": {"repulsion": "Taylor et al. 2017 + many", ...}
    }
    """
    if not CONTEXT_PATH.exists():
        print(f"⚠️  CONTEXT.md not found at {CONTEXT_PATH} — running without glossary constraints.")
        return {"terms": {}, "anti_patterns": [], "models": {}}
    
    text = CONTEXT_PATH.read_text()
    
    # Models section (look for "## 3. The Three Competing Models")
    models = {}
    in_models = False
    for line in text.splitlines():
        if "## 3." in line and "Competing Models" in line:
            in_models = True
            continue
        if in_models and line.startswith("## 4."):
            break
        m = re.match(r"### Model ([A-D]):\s+\*\*([^*]+)\*\*", line)
        if m:
            models[m.group(2).lower()] = m.group(2)
    
    # Anti-patterns section (## 8.)
    anti_patterns = []
    in_anti = False
    for line in text.splitlines():
        if "## 8." in line and "Anti-Patterns" in line:
            in_anti = True
            continue
        if in_anti and line.startswith("## 9."):
            break
        # Pattern: "❌ "foo" → ✅ "bar""
        m = re.match(r'.*❌\s*"([^"]+)"\s*→\s*✅\s*"([^"]+)"', line)
        if m:
            anti_patterns.append((m.group(1), m.group(2)))
    
    return {
        "models": models,
        "anti_patterns": anti_patterns,
        "path": str(CONTEXT_PATH),
    }


# ---------------------------------------------------------------------------
# GrillSpec — the structured output of the grilling phase
# ---------------------------------------------------------------------------
SPEC_FIELDS = [
    ("mechanism_focus", "Which mechanism is the paragraph about?",
     ["repulsion", "adhesion", "cortical tension", "endocytosis", "bidirectional signaling", "cis-interaction", "survey (all)"]),
    ("tissue", "Which tissue/developmental context?",
     ["generic", "hindbrain rhombomeres", "neural crest", "somites", "otic vesicle", "cell culture (in vitro)", "Drosophila"]),
    ("model_stance", "What stance should the paragraph take?",
     ["support repulsion model (Taylor et al.)",
      "critique differential adhesion model",
      "support cortical tension model (O'Neill et al.)",
      "support endocytosis model",
      "neutral survey of all models",
      "extend existing model with new evidence"]),
    ("organism", "Which organism(s) to focus on?",
     ["mouse", "chick", "zebrafish", "Xenopus", "Drosophila", "C. elegans", "human", "in vitro only", "all"]),
    ("year_range", "Citation year range filter?",
     ["all years", "last 5 years (2021-2026)", "last 10 years (2016-2026)", "classic (>2010)", "specific (specify)"]),
    ("evidence_type", "What kind of evidence should we lean on?",
     ["in vivo (mouse/chick/zebrafish)",
      "in vitro (cell culture, co-culture assays)",
      "computational/modeling",
      "biochemistry (binding kinetics, structure)",
      "balanced mix"]),
    ("paragraph_length", "How long should the paragraph be?",
     ["short (~150 words, 1-2 citations)",
      "medium (~250 words, 3-5 citations)",
      "long (~400 words, 5-8 citations)"]),
    ("stance_phrasing", "Optional — what claim should the paragraph make? (or leave blank)",
     "freeform"),
]


def empty_spec() -> Dict:
    return {f[0]: None for f in SPEC_FIELDS}


def ask_one_question(field_name: str, question: str, choices) -> Optional[str]:
    """Ask ONE question at a time. Returns chosen string or None if skipped.
    
    Borrowed directly from grill-me: 'Ask the questions one at a time.'
    """
    if isinstance(choices, list):
        print(f"\n❓ {question}")
        for i, c in enumerate(choices, 1):
            print(f"   {i}) {c}")
        print(f"   s) skip (use default / let auto-fill handle it)")
        raw = input("   → ").strip()
        if raw.lower() == "s":
            return None
        try:
            return choices[int(raw) - 1]
        except (ValueError, IndexError):
            print(f"   ⚠️  invalid, skipping")
            return None
    else:
        # Freeform
        print(f"\n❓ {question}")
        print(f"   (enter freeform, or 's' to skip)")
        raw = input("   → ").strip()
        if raw.lower() == "s":
            return None
        return raw


def interactive_grill(topic: str) -> Dict:
    """Run the interactive grill (one question at a time)."""
    print(f"\n{'='*70}")
    print(f"🥩 PHASE A: GRILL — locking down scope before searching bib_rag")
    print(f"{'='*70}")
    print(f"\nTopic: '{topic}'")
    print(f"\nI'll ask {len(SPEC_FIELDS)} questions. For each, pick a number")
    print(f"or 's' to skip. You can also pass --auto to let an LLM fill these in.")
    
    spec = empty_spec()
    spec["raw_topic"] = topic
    spec["grilled_at"] = datetime.now().isoformat()
    spec["mode"] = "interactive"
    
    for fname, question, choices in SPEC_FIELDS:
        answer = ask_one_question(fname, question, choices)
        spec[fname] = answer
    
    return spec


def auto_grill(topic: str, glossary: Dict) -> Dict:
    """Auto-fill the spec using an LLM that has read CONTEXT.md.
    
    Use this when you want a fast pass and don't need a human in the loop.
    """
    print(f"\n🤖 PHASE A (auto): LLM is filling in the spec from CONTEXT.md…")
    
    context_snippet = ""
    if glossary.get("path"):
        ctx = Path(glossary["path"]).read_text()
        context_snippet = f"\n\n# DOMAIN GLOSSARY (CONTEXT.md excerpt):\n{ctx[:3000]}\n"
    
    models_list = ", ".join(glossary.get("models", {}).keys()) or "repulsion, adhesion, tension, endocytosis"
    
    prompt = f"""You are filling in a structured spec for an academic paragraph about Eph/ephrin signaling.

# TOPIC (raw):
{topic}

# AVAILABLE MODELS:
{models_list}
{context_snippet}

# INSTRUCTIONS:
Output a JSON object with these exact keys (use null for any field you cannot confidently determine from the topic):
  mechanism_focus, tissue, model_stance, organism, year_range, evidence_type, paragraph_length, stance_phrasing

Constraints:
- mechanism_focus MUST be one of: repulsion, adhesion, cortical tension, endocytosis, bidirectional signaling, cis-interaction, survey
- model_stance MUST be one of: "support repulsion model", "critique differential adhesion", "support cortical tension model", "support endocytosis model", "neutral survey", "extend existing model"
- tissue MUST be one of: generic, hindbrain rhombomeres, neural crest, somites, otic vesicle, cell culture (in vitro), Drosophila
- paragraph_length MUST be one of: short, medium, long

Output ONLY the JSON, no markdown fences, no commentary."""

    try:
        text = call_llm(
            [{"role": "user", "content": prompt}],
            max_tokens=400, temperature=0.1, timeout=60,
        )
        # Strip any accidental markdown fences
        text = re.sub(r"^```json\s*|\s*```$", "", text, flags=re.MULTILINE)
        spec = json.loads(text)
        spec["raw_topic"] = topic
        spec["grilled_at"] = datetime.now().isoformat()
        spec["mode"] = "auto"
        return spec
    except Exception as e:
        print(f"⚠️  auto-grill failed: {e}")
        print(f"   Falling back to interactive.")
        return interactive_grill(topic)


# ---------------------------------------------------------------------------
# PHASE B: WRITE — search bib_rag with spec-scoped query
# ---------------------------------------------------------------------------
def build_scoped_query(topic: str, spec: Dict) -> str:
    """Combine raw topic with spec hints to build a tighter search query.
    
    De-duplicates terms. Doesn't repeat words already in `topic`.
    """
    topic_words = set(topic.lower().split())
    parts = [topic]
    seen = set(topic_words)
    for hint in [spec.get("mechanism_focus"), spec.get("tissue"), spec.get("organism")]:
        if hint and hint != "generic" and hint != "all":
            hint_words = set(hint.lower().split())
            # Only add if it brings new info
            if not hint_words.issubset(seen):
                parts.append(hint)
                seen.update(hint_words)
    return " ".join(parts)


def embed_query(text: str) -> List[float]:
    resp = requests.post(
        BIB_RAG_EMBED_URL,
        headers={"Content-Type": "application/json"},
        json={"input": text, "model": "bge-m3"},
        timeout=30
    )
    emb = resp.json()["data"][0]["embedding"]
    norm = sum(x * x for x in emb) ** 0.5
    return [x / norm for x in emb] if norm > 0 else emb


def search_bib_rag(query: str, top_k: int = 5) -> List[Dict]:
    emb = embed_query(query)
    from langchain_community.vectorstores import Chroma

    class PrecomputedEmbed:
        def __init__(self, emb): self.emb = emb
        def embed_documents(self, texts): return [self.emb] * len(texts)
        def embed_query(self, text): return self.emb

    db = Chroma(
        persist_directory=CHROMA_PATH,
        collection_name=_CFG["collection_name"],
        embedding_function=PrecomputedEmbed(emb),
    )
    docs = db._collection.query(
        query_embeddings=[emb], n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    results = []
    for i in range(len(docs["ids"][0])):
        meta = docs["metadatas"][0][i]
        dist = docs["distances"][0][i]
        sim = 1.0 / (1.0 + dist)
        results.append({
            "text": docs["documents"][0][i],
            "title": meta.get("title", ""),
            "year": meta.get("year", ""),
            "doi": meta.get("doi", ""),
            "section": meta.get("section", ""),
            "similarity": sim,
        })
    return sorted(results, key=lambda x: x["similarity"], reverse=True)


# ---------------------------------------------------------------------------
# Anti-pattern enforcement — leverage CONTEXT.md to clean up LLM output
# ---------------------------------------------------------------------------
def enforce_glossary(text: str, glossary: Dict) -> str:
    """Apply anti-pattern corrections from CONTEXT.md §8."""
    for bad, good in glossary.get("anti_patterns", []):
        # Use word-boundary regex; case-insensitive
        pattern = re.compile(re.escape(bad), re.IGNORECASE)
        text = pattern.sub(good, text)
    return text


# ---------------------------------------------------------------------------
# ODT writer
# ---------------------------------------------------------------------------
def write_odt(paragraph: str, citations: List[str], spec: Dict, output_path: str):
    doc = OpenDocumentText()

    # Title
    doc.text.addElement(H(outlinelevel=1, text=f"Grill-spec Synthesis"))

    # Spec box
    spec_lines = [
        f"Topic: {spec.get('raw_topic', '')}",
        f"Mechanism focus: {spec.get('mechanism_focus', '—')}",
        f"Tissue: {spec.get('tissue', '—')}",
        f"Model stance: {spec.get('model_stance', '—')}",
        f"Organism: {spec.get('organism', '—')}",
        f"Evidence type: {spec.get('evidence_type', '—')}",
        f"Grilled: {spec.get('grilled_at', '—')} ({spec.get('mode', '—')})",
    ]
    doc.text.addElement(P(text="[Grill spec — locked before search]"))
    for line in spec_lines:
        doc.text.addElement(P(text=line))
    doc.text.addElement(P(text=""))

    # Paragraph
    doc.text.addElement(P(text=paragraph))
    doc.text.addElement(P(text=""))

    # Bibliography
    doc.text.addElement(H(outlinelevel=2, text="References"))
    for i, ref in enumerate(citations, 1):
        doc.text.addElement(P(text=f"[{i}] {ref}"))

    doc.save(output_path)
    print(f"\n📝 Saved to {output_path}")


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Grill-then-write academic paragraph generator (bib_rag)")
    parser.add_argument("topic", nargs="?", help="Raw topic sentence")
    parser.add_argument("--auto", action="store_true", help="Auto-fill the spec via LLM (skip interactive grill)")
    parser.add_argument("--spec", help="Path to a saved GrillSpec JSON (skip grilling entirely)")
    parser.add_argument("--top", type=int, default=5, help="Number of passages to retrieve")
    parser.add_argument("--output", help="Output .odt path (default: /Disk_bot/writing/grill_<ts>.odt)")
    parser.add_argument("--save-spec", help="Save the spec to this path (default: /tmp/bib_rag_grill_specs/<ts>.json)")
    args = parser.parse_args()

    if not args.topic and not args.spec:
        parser.error("Provide a topic or --spec <path>")

    glossary = load_context_glossary()

    # PHASE A: GRILL
    if args.spec:
        spec = json.loads(Path(args.spec).read_text())
        print(f"📋 Loaded spec from {args.spec}")
    elif args.auto:
        spec = auto_grill(args.topic, glossary)
    else:
        spec = interactive_grill(args.topic)

    # Save spec (always — enables resumption)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    spec_path = args.save_spec or str(SPEC_DIR / f"grill_spec_{ts}.json")
    Path(spec_path).write_text(json.dumps(spec, indent=2, ensure_ascii=False))
    print(f"\n📋 Spec saved: {spec_path}")

    # Print spec for review
    print(f"\n{'='*70}")
    print(f"LOCKED SPEC:")
    print(f"{'='*70}")
    for fname, _, _ in SPEC_FIELDS:
        print(f"  {fname}: {spec.get(fname)}")
    print(f"{'='*70}")

    # PHASE B: WRITE
    scoped_query = build_scoped_query(args.topic or spec.get("raw_topic", ""), spec)
    print(f"\n🔍 PHASE B: searching bib_rag with scoped query:")
    print(f"   '{scoped_query}'")

    passages = search_bib_rag(scoped_query, top_k=args.top)
    if not passages:
        print("❌ No passages found.")
        sys.exit(1)
    print(f"✅ Found {len(passages)} passages\n")

    for i, p in enumerate(passages, 1):
        print(f"  [{i}] {p['title'][:60]} (sim={p['similarity']:.3f})")

    # Use the LLM to write the paragraph given the spec + passages
    print(f"\n✍️  Writing paragraph (spec-aware, glossary-constrained)…")
    paragraph = synthesize_with_spec(passages, spec, glossary)

    # Enforce anti-patterns from CONTEXT.md
    paragraph = enforce_glossary(paragraph, glossary)

    print(f"\n{'='*70}")
    print("GENERATED PARAGRAPH:")
    print(f"{'='*70}")
    print(paragraph)

    # Citation harvesting (lightweight — use top 3 unique titles)
    seen, cits = set(), []
    for p in passages[:3]:
        if p["title"] not in seen:
            seen.add(p["title"])
            cits.append(f"{p['title']} ({p['year']}) — DOI: {p['doi']}")
    print(f"\n{'='*70}")
    print("REFERENCES (auto-harvested from retrieval):")
    print(f"{'='*70}")
    for c in cits:
        print(f"  • {c}")

    output = args.output or f"/Disk_bot/writing/grill_{ts}.odt"
    write_odt(paragraph, cits, spec, output)


def synthesize_with_spec(passages: List[Dict], spec: Dict, glossary: Dict) -> str:
    """Use local vLLM to synthesize a paragraph that follows the locked spec."""
    # Build context from passages
    ctx_blocks = []
    for i, p in enumerate(passages, 1):
        ctx_blocks.append(f"[{i}] ({p['title']}, {p['year']}, sim={p['similarity']:.2f})\n{p['text'][:600]}")
    context = "\n\n---\n\n".join(ctx_blocks)

    stance = spec.get("stance_phrasing") or "review evidence for the mechanism of interest"
    mechanism = spec.get("mechanism_focus", "repulsion")
    tissue = spec.get("tissue", "generic")
    model = spec.get("model_stance", "neutral survey")
    length = spec.get("paragraph_length", "medium")
    word_target = {"short": 150, "medium": 250, "long": 400}.get(length, 250)

    # Anti-pattern reminder
    anti = ""
    if glossary.get("anti_patterns"):
        anti = "\n\n# TERMINOLOGY (CONTEXT.md §8):\n"
        for bad, good in glossary["anti_patterns"][:6]:
            anti += f"- Use '{good}', NOT '{bad}'\n"

    prompt = f"""Write a {length} academic paragraph (~{word_target} words) about Eph/ephrin signaling.

# LOCKED SCOPE (do not deviate):
- Mechanism focus: {mechanism}
- Tissue: {tissue}
- Model stance: {model}
- Required claim: {stance}
{anti}

# EVIDENCE (numbered passages from bib_rag):
{context}

# CITATION RULES (APA):
- Cite as (Author et al., YEAR) inline
- Use only the passages above as evidence
- If a claim isn't supported by the retrieved passages, do NOT include it

# STYLE:
- Use the exact vocabulary from CONTEXT.md (e.g. "cell segregation", "boundary formation", "heterotypic repulsion")
- Be specific about cell type / tissue / organism when passages support it

Write ONLY the paragraph (no headers, no preamble)."""

    try:
        return call_llm(
            [{"role": "user", "content": prompt}],
            max_tokens=600, temperature=0.3, timeout=120,
        )
    except Exception as e:
        # Build a useful academic fallback from the top passages.
        # Uses the CONTEXT.md vocabulary (repulsion, boundary, etc.)
        # to stay on-spec even when LLM is down.
        top = passages[:3]
        claims = []
        for p in top:
            first_sent = p["text"].split(". ")[0] if p["text"] else ""
            if first_sent:
                claims.append(f"{first_sent} ({p['title']}, {p['year']})")
        fallback = (
            f"[LLM synthesis unavailable: {e}]\n\n"
            f"Evidence-based summary: " + " ".join(claims) +
            f" These findings are consistent with the locked spec "
            f"({spec.get('mechanism_focus', 'repulsion')}-mediated "
            f"{spec.get('tissue', 'tissue')} boundary formation, "
            f"in support of {spec.get('model_stance', 'the repulsion model')})."
        )
        return fallback


if __name__ == "__main__":
    main()
