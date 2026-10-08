#!/usr/bin/env python3
"""
verify.py — Phase 3 of autonomous research: verify each proposed hypothesis
against the corpus, with the toolkit's own evidence rules.

Phases 1-2 produce PROPOSALS (graph directions -> structured hypotheses). A
proposal is worth nothing until something checks it. Phase 3 does that check
IN THE CORPUS — deterministically where possible, with a harsh-critic LLM
only for the final support/refute call.

Layered verification (fail-soft at every layer):

  L0  retrieval      — for each expected_evidence phrase, retrieve corpus
                       chunks (query_bib_rag.search). No hits anywhere =>
                       status "unsupported" (the corpus cannot speak to it).
  L1  whitelist      — the retrieved parent_ids become the session's
                       retrieval_keys; ONLY evidence pointing at those parents
                       can be "verified" (this is claim_ledger's rule, reused).
  L2  ledger         — build a claim with the retrieved parents as kind=chunk
                       evidence and run claim_ledger.validate_claims in STRICT
                       mode. A hypothesis that cannot produce even one
                       whitelisted evidence pointer CANNOT be verified.
  L3  critic (LLM)   — given the retrieved evidence text, decide support =
                       supported | contradicted | inconclusive, considering
                       the hypothesis's own falsifier. Refuted is a first-
                       class outcome (autoresearch's "honest recording").

Output status:
  verified      — L2 passed AND L3 says supported
  refuted       — L3 says contradicted (evidence against the claim)
  speculative   — evidence found but inconclusive, OR only external pointers
  unsupported   — nothing retrieved / nothing whitelisted (corpus silent)

Contract:
    verify(hypothesis, ...) -> Verification
      Verification = {
        "hypothesis_id", "status", "support",
        "n_evidence", "evidence": [{parent_id, source, snippet}],
        "critic": {verdict, reason},     # L3
        "ledger_ok", "ledger_errors": [...],  # L2
        "queries": [...],
      }
    verify_all(hypotheses) -> List[Verification]

    Never raises. A verification failure degrades the status, never fabricates
    support: an unverifiable hypothesis is reported as such.
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Tuple

import requests

try:  # bib_rag-package-try
    from .llm_endpoint import llm_endpoint as _llm_endpoint, env_int as _env_int
except ImportError:  # flat (loose-script mode)
    try:
        from llm_endpoint import llm_endpoint as _llm_endpoint, env_int as _env_int
    except ImportError:  # src/ on sys.path directly
        import os as _os, sys as _sys
        _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
        from llm_endpoint import llm_endpoint as _llm_endpoint, env_int as _env_int


STATUS_VERIFIED = "verified"
STATUS_REFUTED = "refuted"
STATUS_SPECULATIVE = "speculative"
STATUS_UNSUPPORTED = "unsupported"

# L0 relevance threshold: retrieved chunks below this similarity are treated
# as "no evidence" — the corpus did not really speak to the hypothesis; it
# merely matched surface words. Borrows the calibration logic from
# broaden.py (bge-m3 cosine-ish similarity = 1/(1+dist); strong hits >0.54,
# noise drifts toward 0.2). Without this, off-domain directions (e.g. a
# medical-imaging method paper on an Eph question) retrieve a pile of
# irrelevant chunks and sit in "speculative" forever instead of being
# cleanly reported unsupported.
def _min_sim() -> float:
    try:
        return float(os.environ.get("VERIFY_MIN_SIM", "0.54"))
    except (TypeError, ValueError):
        return 0.54


# ─── L0: retrieval ───────────────────────────────────────────────────────────
def _retrieve(query: str, top_k: int) -> List[dict]:
    """Corpus retrieval via the shared CLI path. Returns a list of
    {parent_id, source, snippet, similarity}. [] on any failure."""
    try:
        try:  # bib_rag-package-try
            from .query_bib_rag import search as _search
        except ImportError:
            from query_bib_rag import search as _search
        results = _search(query, top_k=top_k, expand=False, distill=False)
    except Exception:
        return []
    out = []
    for r in results or []:
        meta = r.get("metadata") or {}
        pid = meta.get("parent_id")
        if not pid:
            continue
        out.append({
            "parent_id": pid,
            "source": meta.get("source") or meta.get("title", ""),
            "snippet": (r.get("text") or "")[:400],
            "similarity": (1.0 / (1.0 + r["distance"])
                           if r.get("distance") is not None else None),
        })
    return out


def collect_evidence(hypothesis: Dict, top_k: int) -> Tuple[List[dict], List[str]]:
    """Retrieve evidence for every expected_evidence phrase.

    Returns (evidence, queries). Evidence is de-duplicated by parent_id
    (best similarity kept) and gated by the L0 relevance threshold: a chunk
    whose similarity is below VERIFY_MIN_SIM is not evidence — the corpus
    matched surface tokens, not the claim. Deterministic given the corpus +
    phrase order. When a hit carries no similarity (distance missing) it is
    kept (cannot judge → don't silently drop).
    """
    phrases = hypothesis.get("expected_evidence") or []
    # fall back to the claim itself when the model gave no phrases — still a
    # legitimate retrieval probe, never a fabrication.
    probes = [p for p in phrases if p] or [hypothesis.get("claim") or ""]
    probes = [p for p in probes if p]
    threshold = _min_sim()
    evidence: Dict[str, dict] = {}
    queries: List[str] = []
    for q in probes:
        queries.append(q)
        for e in _retrieve(q, top_k):
            sim = e.get("similarity")
            if sim is not None and threshold and sim < threshold:
                continue  # below relevance floor → not evidence
            prev = evidence.get(e["parent_id"])
            if prev is None or ((e.get("similarity") or 0) >
                                (prev.get("similarity") or 0)):
                evidence[e["parent_id"]] = e
    ev = sorted(evidence.values(),
                key=lambda x: (-(x.get("similarity") or 0), x["parent_id"]))
    return ev, queries


# ─── L1+L2: whitelist + claim_ledger strict validation ───────────────────────
def ledger_check(hypothesis: Dict, evidence: List[dict]) -> Tuple[bool, List[str]]:
    """Run the hypothesis's evidence through claim_ledger in strict mode.

    The retrieved parents form the session's retrieval_keys; a claim is built
    with every evidence pointer as kind=chunk. Returns (ok, errors). A claim
    with NO whitelisted evidence fails strict validation — which is exactly
    the rule we want: an unverifiable hypothesis must not be marked verified.
    """
    try:
        try:  # bib_rag-package-try
            from .claim_ledger import validate_claims
        except ImportError:
            from claim_ledger import validate_claims
    except Exception:
        return False, ["claim_ledger unavailable"]
    keys = {e["parent_id"] for e in evidence}
    claim = {
        "id": "c1",
        "text": hypothesis.get("claim") or "",
        "confidence": "moderate",
        "evidence": [{"kind": "chunk", "parent_id": e["parent_id"]}
                     for e in evidence],
    }
    try:
        res = validate_claims([claim], retrieval_keys=keys, strict=True)
    except Exception as e:
        return False, [f"ledger raised: {e}"]
    return res.ok, list(res.errors)


# ─── L3: harsh-critic verdict ────────────────────────────────────────────────
CRITIC_SYSTEM = """You are a harsh research critic. Given a HYPOTHESIS and the \
EVIDENCE retrieved from a scientific corpus, decide how the evidence bears on it.

Return a JSON object:
{"verdict": "supported" | "contradicted" | "inconclusive", "reason": one sentence}

Rules:
- "supported": the evidence directly and specifically backs the hypothesis.
- "contradicted": the evidence directly opposes it (pay attention to the
  FALSIFIER — if the evidence matches the falsifier, the verdict is contradicted).
- "inconclusive": evidence is tangential, empty, or only weakly related.
- Be HARSH. Shared topic words are NOT support. If the evidence does not
  actually state the claim, say inconclusive — never inflate.
- Judge ONLY from the provided evidence. Do not use outside knowledge.

Return ONLY the JSON object."""


def _critic(hypothesis: Dict, evidence: List[dict], timeout: int,
            max_tokens: int) -> Dict:
    """L3 verdict. Returns {"verdict": ..., "reason": ...}; verdict
    "inconclusive" on any failure (never fabricates a support)."""
    fallback = {"verdict": "inconclusive", "reason": "critic unavailable"}
    if os.environ.get("VERIFY_CRITIC", "1") == "0":
        return fallback
    url, model = _llm_endpoint()
    ev_text = "\n\n".join(
        f"[{i+1}] source={e.get('source','')}\n{e.get('snippet','')}"
        for i, e in enumerate(evidence[:6])) or "(no evidence retrieved)"
    user = (
        f"HYPOTHESIS:\n{hypothesis.get('claim','')}\n\n"
        f"FALSIFIER:\n{hypothesis.get('falsifier','')}\n\n"
        f"EVIDENCE:\n{ev_text}\n\n"
        f"How does the evidence bear on the hypothesis?"
    )
    try:
        resp = requests.post(
            f"{url}/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": CRITIC_SYSTEM},
                    {"role": "user", "content": user},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "reasoning_effort": "none",
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
    except Exception:
        return fallback
    obj = _parse_verdict(content or "")
    return obj or fallback


_JSON_OBJ = re.compile(r"\{.*\}", re.S)


def _parse_verdict(text: str) -> Optional[Dict]:
    if not text:
        return None
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    for cand in [raw] + ([m.group(0) for m in [_JSON_OBJ.search(raw)] if m]):
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        v = str(obj.get("verdict") or "").strip().lower()
        if v in ("supported", "contradicted", "inconclusive"):
            return {"verdict": v, "reason": str(obj.get("reason") or "").strip()}
    return None


# ─── public API ──────────────────────────────────────────────────────────────
def _decide_status(evidence: List[dict], ledger_ok: bool, critic: Dict) -> str:
    """Combine the layers into a status. The ORDER encodes the safety rule:
    no evidence -> unsupported; evidence but not whitelisted/ledger-failed ->
    unsupported (cannot verify without a real pointer); critic contradicted ->
    refuted; critic supported (and ledger ok) -> verified; else speculative.
    """
    if not evidence:
        return STATUS_UNSUPPORTED
    if not ledger_ok:
        # evidence exists but none of it is a whitelisted, valid pointer
        return STATUS_UNSUPPORTED
    verdict = (critic or {}).get("verdict")
    if verdict == "contradicted":
        return STATUS_REFUTED
    if verdict == "supported":
        return STATUS_VERIFIED
    return STATUS_SPECULATIVE


def verify(hypothesis: Dict, top_k: Optional[int] = None) -> Dict:
    """Verify one hypothesis against the corpus. Never raises."""
    if not hypothesis or not hypothesis.get("id"):
        return {"hypothesis_id": None, "status": STATUS_UNSUPPORTED,
                "support": 0, "n_evidence": 0, "evidence": [],
                "critic": {"verdict": "inconclusive",
                           "reason": "no hypothesis"},
                "ledger_ok": False, "ledger_errors": ["no hypothesis"],
                "queries": []}
    top_k = top_k if top_k is not None else _env_int("VERIFY_TOP_K", 5)
    timeout = _env_int("VERIFY_TIMEOUT", 60)
    max_tokens = _env_int("VERIFY_MAX_TOKENS", 200)

    evidence, queries = collect_evidence(hypothesis, top_k)
    ledger_ok, ledger_errors = ledger_check(hypothesis, evidence)
    critic = _critic(hypothesis, evidence, timeout, max_tokens)
    status = _decide_status(evidence, ledger_ok, critic)

    return {
        "hypothesis_id": hypothesis["id"],
        "status": status,
        "support": len(evidence),
        "n_evidence": len(evidence),
        "evidence": [
            {"parent_id": e["parent_id"], "source": e.get("source", ""),
             "snippet": e.get("snippet", ""),
             "similarity": e.get("similarity")}
            for e in evidence[:8]
        ],
        "critic": critic,
        "ledger_ok": ledger_ok,
        "ledger_errors": ledger_errors,
        "queries": queries,
    }


def verify_all(hypotheses: List[Dict], top_k: Optional[int] = None) -> List[Dict]:
    """Verify a list of hypotheses; return one Verification per hypothesis
    (no hypothesis is dropped here — a failed verification is DATA, not a
    reason to hide the hypothesis)."""
    return [verify(h, top_k=top_k) for h in (hypotheses or [])]


# ─── CLI ─────────────────────────────────────────────────────────────────────
def _cli() -> int:
    import sys
    try:
        from .kb_config import get_config
        from .dream import diverge
        from .formulate import formulate_all
    except ImportError:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from kb_config import get_config
        from dream import diverge
        from formulate import formulate_all
    question = sys.argv[1] if len(sys.argv) > 1 else ""
    cfg = get_config()
    mpath = os.path.join(cfg["data_dir"], "graph_metrics.json")
    gpath = cfg.get("reference_graph_path") or os.path.join(
        cfg["data_dir"], "reference_graph.json")
    doc = json.load(open(mpath, encoding="utf-8"))
    metrics = doc.get("metrics") or doc.get("nodes") or {}
    graph = json.load(open(gpath, encoding="utf-8")) if os.path.exists(gpath) else None

    dirs = diverge(question, metrics, graph)
    hyps = formulate_all(dirs, context=question)
    print(f"{len(dirs)} directions -> {len(hyps)} hypotheses -> verifying...\n")
    vers = verify_all(hyps)
    from collections import Counter
    tally = Counter(v["status"] for v in vers)
    for h, v in zip(hyps, vers):
        print(f"[{v['status']:<11}] {h['id']}  ({v['n_evidence']} evidence, "
              f"critic={v['critic']['verdict']})")
        print(f"    {h['claim'][:100]}")
    print(f"\nstatus tally: {dict(tally)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
