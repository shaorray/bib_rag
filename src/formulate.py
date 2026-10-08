#!/usr/bin/env python3
"""
formulate.py — Phase 2 of autonomous research: turn graph directions into
structured, testable hypotheses.

Phase 1 (dream.py) produces DIRECTIONS from the citation graph's shape — each
one a "thing worth asking about" anchored on real nodes, with no claim yet.
Phase 2 hands each direction to the LLM and asks it to write down what the
direction actually suggests as a CHECKABLE hypothesis: a claim, why the graph
made it plausible, and — critically — what evidence would confirm or refute it.

Design borrowed from:
  - idea-generation  (3-5 diverse ideas; rate Interestingness/Feasibility/
                      Novelty — here: Interesting/Feasible/Testable, because
                      "novelty" is decided in Phase 3 against the corpus)
  - novelty-assessment (harsh critic: the hypothesis must be falsifiable, not
                      a restatement of the graph signal)
  - autoresearch      (hypothesis-FIRST: state the expected outcome before
                      any verification; that is what makes the loop honest)

Contract:
    formulate(direction, context="") -> Hypothesis
      Hypothesis = {
        "id": str,                    # deterministic, derived from direction id
        "from_direction": str,        # Phase-1 direction id (audit link)
        "claim": str,                 # one falsifiable sentence
        "rationale": str,             # why the graph signal suggests it
        "expected_evidence": [str],   # what to look for in the corpus
        "falsifier": str,             # what would REFUTE it (harsh-critic)
        "scores": {"interesting","feasible","testable"},   # 1-10 each
        "status": "proposed",
      }
    formulate_all(directions, context="") -> List[Hypothesis]

    Never raises: LLM down / unparseable -> the direction is SKIPPED (not
    fabricated into a fake hypothesis). A skipped direction is reported back
    via the returned list length so the caller can tell.
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional

import requests

try:  # bib_rag-package-try
    from .llm_endpoint import llm_endpoint as _llm_endpoint, env_int as _env_int
except ImportError:  # flat (loose-script mode)
    try:
        from llm_endpoint import llm_endpoint as _llm_endpoint, env_int as _env_int
    except ImportError:  # src/ on sys.path directly (no package context)
        import os as _os, sys as _sys
        _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
        from llm_endpoint import llm_endpoint as _llm_endpoint, env_int as _env_int


SYSTEM_PROMPT = """You are a meticulous research scientist turning a structural clue from a \
citation graph into ONE testable hypothesis about a scientific literature corpus.

You are given a DIRECTION: a structural observation (a bridge paper between subfields, two \
same-topic papers that never cite each other, or an influential-but-under-cited paper). Your job \
is NOT to restate the observation — it is to propose the CHECKABLE scientific claim the \
observation hints at.

Return a JSON object with exactly these keys:
{
  "claim": ONE falsifiable sentence stating a SINGLE hypothesis (see the atomicity rule below),
  "rationale": one or two sentences on WHY the structural clue makes this claim plausible,
  "expected_evidence": a list of 2-4 short phrases naming what to look for in the corpus to support it (e.g. "co-occurrence of ephrin-B reverse signalling and RhoA activation"),
  "falsifier": one sentence: what observation in the corpus would REFUTE the claim,
  "scores": {"interesting": 1-10, "feasible": 1-10, "testable": 1-10}
}

ATOMICITY RULE (the most important rule):
- The claim must assert exactly ONE thing: ONE mechanism, ONE directionality, ONE condition.
- Do NOT conjoin multiple assertions with "and" / "while" / "thereby" / commas. A compound claim
  like "X activates Y and inhibits Z, independent of W" is FOUR claims — pick the ONE strongest
  and put the rest in rationale, NOT in the claim.
- Prefer a specific, directional form: "A mediates/requires/regulates B under C".
- A reader must be able to point at a single sentence of evidence and say yes/no to the claim.
- If the direction genuinely suggests several claims, choose the single most central one.

Other rules:
- The claim must be SPECIFIC and FALSIFIABLE. "X is important" is not a hypothesis. "X requires Y" is.
- Do NOT invent facts, numbers, or paper titles. The hypothesis is a PROPOSAL — it has no evidence yet; do not imply it does.
- Be a harsh critic when scoring: "testable" is low if the corpus cannot plausibly contain the discriminating evidence; "feasible" is low if verifying it would need experiments outside the literature.
- If the direction cannot yield a single testable claim, return {"claim": ""}.

Return ONLY the JSON object. No prose, no markdown fences."""


def _stable_hyp_id(direction_id: str) -> str:
    """Deterministic hypothesis id from the direction id (audit link)."""
    return f"H-{direction_id}"


def _build_message(direction: Dict, context: str) -> str:
    anchors = ", ".join(direction.get("anchors") or [])
    sig = json.dumps(direction.get("signal") or {}, ensure_ascii=False)
    parts = [
        f"DIRECTION TYPE: {direction.get('source', '')}",
        f"NOTE: {direction.get('note', '')}",
        f"ANCHOR PAPER(S): {anchors}",
        f"RAW SIGNAL: {sig}",
    ]
    if direction.get("topic"):
        parts.append(f"SHARED TOPIC TERMS: {direction['topic']}")
    if context:
        parts.append(f"RESEARCH QUESTION / CONTEXT:\n{context}")
    parts.append("Propose the testable hypothesis this direction suggests.")
    return "\n".join(parts)


# ─── parsing ────────────────────────────────────────────────────────────────
_JSON_OBJ = re.compile(r"\{.*\}", re.S)
_SCORE_KEYS = ("interesting", "feasible", "testable")


def _clamp_score(v) -> Optional[int]:
    try:
        n = int(round(float(v)))
    except (TypeError, ValueError):
        return None
    return max(1, min(10, n))


def _parse_hypothesis(text: str) -> Optional[Dict]:
    """Tolerant parse of the model's JSON. Returns None when the response is
    unusable (no claim / unparseable) — the caller then SKIPS the direction
    rather than fabricating one."""
    if not text:
        return None
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    candidates = [raw]
    m = _JSON_OBJ.search(raw)
    if m:
        candidates.append(m.group(0))
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        claim = str(obj.get("claim") or "").strip()
        if not claim:
            return None  # explicit "cannot yield a claim"
        ee = obj.get("expected_evidence") or []
        if isinstance(ee, str):
            ee = [ee]
        ee = [str(x).strip() for x in ee if str(x).strip()][:6]
        scores_raw = obj.get("scores") or {}
        scores = {}
        for k in _SCORE_KEYS:
            scores[k] = _clamp_score(scores_raw.get(k)) if isinstance(scores_raw, dict) else None
        return {
            "claim": claim,
            "rationale": str(obj.get("rationale") or "").strip(),
            "expected_evidence": ee,
            "falsifier": str(obj.get("falsifier") or "").strip(),
            "scores": scores,
        }
    return None


def _call_llm(direction: Dict, context: str, timeout: int, max_tokens: int) -> Optional[str]:
    url, model = _llm_endpoint()
    try:
        resp = requests.post(
            f"{url}/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _build_message(direction, context)},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.4,  # a little room to be generative, not wild
                "reasoning_effort": "none",
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception:
        return None


# ─── public API ──────────────────────────────────────────────────────────────
def formulate(direction: Dict, context: str = "",
              timeout: Optional[int] = None,
              max_tokens: Optional[int] = None) -> Optional[Dict]:
    """Turn ONE Phase-1 direction into a structured hypothesis.

    Returns the Hypothesis dict, or None when the LLM is unavailable or the
    direction yields no testable claim. Never raises, never fabricates.
    """
    if not direction or not direction.get("id"):
        return None
    if os.environ.get("FORMULATE", "1") == "0":
        return None
    timeout = timeout if timeout is not None else _env_int("FORMULATE_TIMEOUT", 60)
    max_tokens = max_tokens if max_tokens is not None else _env_int("FORMULATE_MAX_TOKENS", 500)

    content = _call_llm(direction, context, timeout, max_tokens)
    parsed = _parse_hypothesis(content or "")
    if not parsed:
        return None
    return {
        "id": _stable_hyp_id(direction["id"]),
        "from_direction": direction["id"],
        "from_source": direction.get("source", ""),
        "anchors": direction.get("anchors") or [],
        **parsed,
        "status": "proposed",
    }


def formulate_all(directions: List[Dict], context: str = "") -> List[Dict]:
    """Formulate hypotheses for a list of directions.

    Directions that fail (LLM down / no testable claim) are DROPPED — the
    returned list is shorter than the input, which is the honest signal that
    nothing was invented to fill the gap.
    """
    out: List[Dict] = []
    for d in directions or []:
        h = formulate(d, context=context)
        if h:
            out.append(h)
    return out


# ─── CLI (manual inspection) ─────────────────────────────────────────────────
def _cli() -> int:
    import sys

    def _load():
        try:
            from .kb_config import get_config
            from .dream import diverge
        except ImportError:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from kb_config import get_config
            from dream import diverge
        return get_config(), diverge

    question = sys.argv[1] if len(sys.argv) > 1 else ""
    cfg, diverge = _load()
    mpath = os.path.join(cfg["data_dir"], "graph_metrics.json")
    gpath = cfg.get("reference_graph_path") or os.path.join(
        cfg["data_dir"], "reference_graph.json")
    with open(mpath, encoding="utf-8") as f:
        doc = json.load(f)
    metrics = doc.get("metrics") or doc.get("nodes") or {}
    graph = json.load(open(gpath, encoding="utf-8")) if os.path.exists(gpath) else None

    dirs = diverge(question, metrics, graph)
    print(f"{len(dirs)} directions -> formulating hypotheses...\n")
    hyps = formulate_all(dirs, context=question)
    print(f"{len(hyps)} hypotheses from {len(dirs)} directions\n")
    for h in hyps:
        print(f"[{h['from_source']}] {h['id']}")
        print(f"  claim:   {h['claim']}")
        print(f"  why:     {h['rationale']}")
        print(f"  falsify: {h['falsifier']}")
        print(f"  expect:  {', '.join(h['expected_evidence'])}")
        print(f"  scores:  {h['scores']}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
