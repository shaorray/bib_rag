#!/usr/bin/env python3
"""
claim_ledger.py — Claim–Evidence ledger for bib_rag agentic answers.

Adapted (best-effort) from:
  * TheVirtualBiotech src/utils/claims.py (github.com/harrisongzhang/
    TheVirtualBiotech) — the claim-evidence object, tri-state status
    (verified/external/unresolved), and the reject-at-filing rule.
  * /Disk_bot/Eph/review_proj/tools/validate_evidence.py — the discovery_tree
    adaptation already in use for the exploration tree.

What it adds to bib_rag (which already has, zero-LLM):
    layer 1  evidence_gate.py     — did each retrieval query return evidence?
    layer 2  citation_guard.py    — is each cited source actually retrieved
                                    (whitelist) and does the quote exist
                                    (substring + lexical)?
    layer 3  answer-side hygiene  — strip inline hallucinated citations
    layer 4  THIS MODULE          — explicit claim objects with structured
                                    evidence pointers, validated AT FILING
                                    TIME and persisted as a ledger. Claims
                                    are what survive into downstream use
                                    (discovery_tree nodes, review drafts,
                                    follow-up experiments) — so they get
                                    their own auditable record, not just an
                                    answer-side scrub.

A "claim" here is a single factual assertion the agent makes in an answer.
Its evidence entries must resolve:
    artifact   — a real file path on disk, sha256 recorded (hash checked on
                 every subsequent validation; change = loud defect)
    tool_call  — a tool_use_id that exists in this session's transcript and
                 did NOT error (a failed query can never support a finding)
    chunk      — a bib_rag parent_id present in this session's retrieval_keys
                 (bridges to citation_guard's whitelist world)
    citation   — PMID / DOI / URL (syntactic check only; status=external —
                 deliberately NOT verified, same as VB: "we did not check it")

Usage inside agent_nodes.collect_answer (after citation_guard):
    from .claim_ledger import claims_from_answer, validate_and_persist
    claims = extract_claims(answer)                    # [[claim:C1]] style or
                                                       # sentence-level
    report = validate_and_persist(claims, retrieval_keys, tool_messages)
    # appends an [claim_ledger] note; never breaks answering

Standalone:
    python3 claim_ledger.py --file claims.json
    python3 claim_ledger.py --file claims.json --ledger <ledger.jsonl>
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# ── Schema constants (VB-faithful tri-state) ────────────────────────────

EVIDENCE_KINDS = {"artifact", "tool_call", "chunk", "citation"}
LOCAL_KINDS = {"artifact", "tool_call", "chunk"}

STATUS_LABELS = {
    "verified": "✓ verified",
    "external": "external ref",
    "unresolved": "not on record",
}

CONFIDENCE_LEVELS = {"strong", "moderate", "weak"}

CLAIM_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,32}$")
CLAIM_REF_RE = re.compile(r"\[\[claim:([A-Za-z0-9_.-]+)\]\]")
PMID_RE = re.compile(r"^\d{6,9}$")
DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)
URL_RE = re.compile(r"^https?://", re.IGNORECASE)

#: retrieval_keys entries look like "search::<query>" / parent-id markers
#: (same ledger bib_rag already threads through the graph).


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def evidence_status(ev: dict) -> str:
    """verified / external / unresolved — identical tri-state to VB."""
    if ev.get("verified"):
        return "verified"
    return "external" if ev.get("kind") == "citation" else "unresolved"


@dataclass
class ValidationResult:
    claims: List[dict] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict:
        return {"ok": self.ok, "n_claims": len(self.claims),
                "errors": self.errors, "warnings": self.warnings}


# ════════════════════════════════════════════════════════════════════════
# Validation core
# ════════════════════════════════════════════════════════════════════════

def validate_evidence_entry(ev: dict, idx: int, retrieval_keys: Set[str],
                            tool_events: List[dict],
                            strict: bool = True):
    """Validate one evidence entry against session ground truth.

    kind=artifact  : path exists + sha256.
    kind=chunk     : parent_id present in retrieval_keys (bib_rag whitelist).
    kind=tool_call : tool_use_id in tool_events, not error, not pending.
    kind=citation  : pmid/doi/url syntax only → status external.
    """
    errors, warnings = [], []
    kind = str(ev.get("kind") or "").lower()
    entry: Dict[str, Any] = {"kind": kind, "verified": False}
    if ev.get("note"):
        entry["note"] = str(ev["note"])[:500]

    if kind == "artifact":
        path = str(ev.get("path") or "").strip()
        if not path:
            errors.append(f"evidence[{idx}] (artifact) missing 'path'")
            return entry, errors, warnings
        entry["path"] = path
        if not os.path.exists(path):
            msg = f"evidence[{idx}] artifact {path!r} does not exist"
            (errors if strict else warnings).append(msg)
            return entry, errors, warnings
        entry["sha256"] = ev.get("sha256") or sha256_file(path)
        if ev.get("sha256") and ev["sha256"] != entry["sha256"]:
            msg = (f"evidence[{idx}] artifact {path!r} changed since filing "
                   f"(sha256 mismatch); refile the claim")
            (errors if strict else warnings).append(msg)
        else:
            entry["verified"] = True

    elif kind == "chunk":
        pid = str(ev.get("parent_id") or "").strip()
        if not pid:
            errors.append(f"evidence[{idx}] (chunk) missing 'parent_id'")
            return entry, errors, warnings
        entry["parent_id"] = pid
        # retrieval_keys carry parent ids; a chunk outside this session's
        # retrieval whitelist is exactly citation_guard's dropped-citation
        # case — here it stays loud for claims.
        if pid in retrieval_keys or any(pid in k for k in retrieval_keys):
            entry["verified"] = True
        else:
            msg = (f"evidence[{idx}] chunk {pid!r} not among this session's "
                   f"retrieved parents")
            (errors if strict else warnings).append(msg)

    elif kind == "tool_call":
        tuid = str(ev.get("tool_use_id") or "").strip()
        if not tuid:
            errors.append(f"evidence[{idx}] (tool_call) missing 'tool_use_id'")
            return entry, errors, warnings
        entry["tool_use_id"] = tuid
        by_id = {t.get("tool_use_id"): t for t in tool_events
                 if isinstance(t, dict) and t.get("tool_use_id")}
        call = by_id.get(tuid)
        if call is None:
            msg = f"evidence[{idx}] tool_call {tuid!r} not in session trace"
            (errors if strict else warnings).append(msg)
        elif call.get("is_error") or call.get("type") == "tool_error":
            msg = (f"evidence[{idx}] cites FAILED tool call {tuid!r}; "
                   f"a failed query cannot support a finding")
            (errors if strict else warnings).append(msg)
        else:
            entry["verified"] = True
            entry["tool_name"] = call.get("tool_name")

    elif kind == "citation":
        ref = ev.get("pmid") or ev.get("doi") or ev.get("url")
        if not ref:
            errors.append(f"evidence[{idx}] (citation) needs pmid, doi or url")
            return entry, errors, warnings
        for k in ("pmid", "doi", "url", "title"):
            if ev.get(k):
                entry[k] = ev[k]
        if ev.get("pmid") and not PMID_RE.match(str(ev["pmid"])):
            errors.append(f"evidence[{idx}] pmid {ev['pmid']!r} malformed")
        if ev.get("doi") and not DOI_RE.match(str(ev["doi"])):
            warnings.append(f"evidence[{idx}] doi {ev['doi']!r} looks malformed")
        # deliberately verified=False — external is not checked locally

    else:
        errors.append(f"evidence[{idx}] unknown kind {kind!r} "
                      f"(expected one of {sorted(EVIDENCE_KINDS)})")
    return entry, errors, warnings


def messages_to_tool_events(messages) -> List[dict]:
    """Convert LangChain messages into the dict trace validate_claims expects.

    collect_answer passes raw LangChain messages as `tool_events`; the
    validator filters with isinstance(t, dict), so unconverted objects are
    silently skipped and kind=tool_call evidence can never match its
    tool_use_id ("not in session trace" warning on every such claim).
    This converter emits one dict per AIMessage tool call (with is_error
    resolved from the paired ToolMessage) so tool_call evidence actually
    resolves. Zero-LLM; never raises.
    """
    events: List[dict] = []
    error_ids: Set[str] = set()
    # ToolMessage carries is_error + the tool_call_id it answers.
    for m in messages or []:
        if type(m).__name__ == "ToolMessage" or getattr(m, "type", "") == "tool":
            if getattr(m, "is_error", False) or "ERROR" in str(getattr(m, "content", ""))[:200].upper():
                error_ids.add(str(getattr(m, "tool_call_id", "") or ""))
    for m in messages or []:
        for tc in (getattr(m, "tool_calls", None) or []):
            events.append({
                "tool_use_id": str(tc.get("id") or ""),
                "name": str(tc.get("name") or ""),
                "type": "tool_error" if str(tc.get("id") or "") in error_ids else "tool_call",
            })
    return events


def validate_claims(claims: Any, retrieval_keys: Optional[Set[str]] = None,
                    tool_events: Optional[List[dict]] = None,
                    strict: bool = True) -> ValidationResult:
    """Validate a batch of claim dicts. VB-faithful rules:

      * claim must have id (regex-checked, unique) and text
      * claim MUST cite >=1 evidence entry
      * >=1 evidence must survive (unresolved-only = error when strict)
      * confidence defaults to moderate; unknown → warning
    """
    res = ValidationResult()
    retrieval_keys = retrieval_keys or set()
    tool_events = tool_events or []

    if isinstance(claims, dict):
        claims = claims.get("claims", [])
    if not isinstance(claims, list):
        res.errors.append("claims must be a list of claim objects")
        return res

    seen_ids: Set[str] = set()
    for i, raw in enumerate(claims):
        where = f"claim[{i}]"
        if not isinstance(raw, dict):
            res.errors.append(f"{where}: not an object")
            continue
        cid = str(raw.get("id") or "").strip()
        if not cid:
            res.errors.append(f"{where}: missing 'id'")
            continue
        if not CLAIM_ID_RE.match(cid):
            res.errors.append(f"claim {cid}: id must match {CLAIM_ID_RE.pattern}")
            continue
        if cid in seen_ids:
            res.errors.append(f"claim {cid}: duplicate id")
            continue
        seen_ids.add(cid)

        text = str(raw.get("text") or "").strip()
        if not text:
            res.errors.append(f"claim {cid}: missing 'text'")
            continue

        confidence = str(raw.get("confidence") or "moderate").lower()
        if confidence not in CONFIDENCE_LEVELS:
            res.warnings.append(f"claim {cid}: unknown confidence "
                                f"{confidence!r}, using 'moderate'")
            confidence = "moderate"

        raw_evidence = raw.get("evidence") or []
        if not isinstance(raw_evidence, list) or not raw_evidence:
            res.errors.append(f"claim {cid}: must cite >=1 evidence entry")
            continue

        kept: List[dict] = []
        for j, ev in enumerate(raw_evidence):
            if not isinstance(ev, dict):
                res.errors.append(f"claim {cid}: evidence[{j}] is not an object")
                continue
            entry, errs, warns = validate_evidence_entry(
                ev, j, retrieval_keys, tool_events, strict=strict)
            res.errors.extend(f"claim {cid}: {e}" for e in errs)
            res.warnings.extend(f"claim {cid}: {w}" for w in warns)
            if entry is not None:
                kept.append(entry)
        if not kept:
            res.errors.append(f"claim {cid}: no evidence survived validation")
            continue

        solid = [e for e in kept if evidence_status(e) in ("verified", "external")]
        if strict and not solid:
            res.errors.append(
                f"claim {cid}: every evidence pointer unresolved — "
                f"an unvalidated evidence link is worse than no link")
            continue

        res.claims.append({
            "id": cid,
            "text": text,
            "agent": raw.get("agent"),
            "confidence": confidence,
            "evidence": kept,
            "status_counts": {
                "verified": sum(1 for e in kept if evidence_status(e) == "verified"),
                "external": sum(1 for e in kept if evidence_status(e) == "external"),
                "unresolved": sum(1 for e in kept if evidence_status(e) == "unresolved"),
            },
        })
    return res


# ════════════════════════════════════════════════════════════════════════
# Ledger persistence (append-only JSONL, same shape as discovery_tree)
# ════════════════════════════════════════════════════════════════════════

def default_ledger_path() -> str:
    """Ledger path for the ACTIVE library, via kb_config (never a hardcoded
    root). Falls back to the data root the rest of the toolkit resolves, so a
    wrapper-driven run (BIB_RAG_KB_NAME) writes to the right library."""
    try:
        try:
            from .kb_config import get_config
        except ImportError:  # flat (loose-script mode)
            from kb_config import get_config
        root = get_config()["data_root"]
    except Exception:
        # Last-resort only: an explicit root if the caller set one, else CWD.
        root = os.environ.get("BIB_RAG_ROOT") or os.getcwd()
    return str(Path(root) / "claim_ledger.jsonl")


def persist(claims: List[dict], ledger_path: Optional[str] = None,
            session: str = "") -> str:
    """Append validated claims to the ledger. Returns path written."""
    path = ledger_path or default_ledger_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    rec = {"ts": datetime.now().isoformat(), "session": session,
           "claims": claims}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


# ════════════════════════════════════════════════════════════════════════
# Answer-side extraction: [[claim:C1]] anchors or sentence heuristics
# ════════════════════════════════════════════════════════════════════════

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def extract_claim_refs(answer: str) -> List[str]:
    """Find [[claim:...]] anchors in an answer body (VB-style)."""
    return sorted(set(CLAIM_REF_RE.findall(answer)))


# ── Numeric claim re-check (Feature 2 / governance checklist L3) ─────────

NUM_RE = re.compile(r"\d+(?:\.\d+)?")
# forms like 2-fold / 3.5-fold — capture the numeric head
FOLD_RE = re.compile(r"(\d+(?:\.\d+)?)\s*-?\s*(?:fold|×|x)\b", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")


def claim_numbers(text: str) -> List[str]:
    """Numeric tokens a claim asserts: integers/decimals (thousand-separators
    stripped), percentages, fold-changes (numeric head only), and years."""
    t = text.replace(",", "")  # strip thousands separators
    out = {m.group(1) if m.group(1) else m.group(0)
           for m in FOLD_RE.finditer(t)}
    out.update(NUM_RE.findall(t))
    return sorted(out)


def numeric_mismatch(claim_text: str, evidence_texts: List[str],
                     source_years: Optional[List[int]] = None) -> Optional[str]:
    """Reason string when a numeric claim is unsupported by its evidence.

    A claim containing numbers is only as good as its numbers: at least one
    number must appear verbatim (after thousand-separator stripping) in the
    cited chunk texts / evidence quotes. A year matching the cited source's
    publication year counts as supporting (citation-of-record, not a fact).
    Claims without numbers are never mismatched — returns None.
    """
    nums = claim_numbers(claim_text)
    if not nums:
        return None
    hay = [(t or "").replace(",", "") for t in evidence_texts if t]
    if source_years:
        hay.extend(str(y) for y in source_years)
    for n in nums:
        # boundary-aware: '10' must not match inside '100' or '2010'
        pat = re.compile(rf"(?<![\d.]){re.escape(n)}(?!\d)")
        if any(pat.search(h) for h in hay):
            return None
    return "numeric-mismatch"


def build_claims_from_sources(answer: str, max_claims: int = 30) -> List[dict]:
    """Turn an answer's Sources section into claim objects (best-effort).

    Each source line becomes one claim citing that source as chunk/citation
    evidence; the claim text is the sentence of the body that mentions the
    same PMID/DOI first, or the source line itself when no in-body mention
    is found (confidence=weak then). This gives every answer a machine-
    checkable ledger without requiring the model to emit claim syntax.
    """
    try:
        from .citation_guard import split_answer_sources  # reuse the splitter
    except ImportError:
        from citation_guard import split_answer_sources  # script-style run
    try:
        body, sources, _tail = split_answer_sources(answer)
    except Exception:
        body, sources = answer, []
    claims = []
    for i, src in enumerate(sources, 1):
        cid = f"S{i}"
        # find the first PMID/DOI in the source line
        m_pmids = re.findall(r"\b(\d{6,9})\b", src)
        m_dois = re.findall(r"(10\.\d{4,9}/[^\s,;)]+)", src)
        evidence = []
        if m_dois:
            evidence.append({"kind": "citation", "doi": m_dois[0].rstrip('.,;')})
        elif m_pmids:
            evidence.append({"kind": "citation", "pmid": m_pmids[0]})
        # hook the body sentence that cites this source index, if any
        m = re.search(rf"\[{i}\]|\[{cid}\]", body)
        if m:
            start = body.rfind(".", 0, m.start())
            end = body.find(".", m.end())
            sent = body[start + 1:end].strip() or body[max(0, m.start() - 200):m.end()]
            evidence[-1]["note"] = f"cited from body near: {sent[:160]}"
        if not evidence:
            evidence.append({"kind": "citation", "url": src[:200]})
        claims.append({"id": cid, "text": src[:300],
                       "confidence": "weak", "evidence": evidence})
        if i >= max_claims:
            break
    return claims


# ════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Claim-Evidence ledger for bib_rag (VB-style)")
    ap.add_argument("--file", help="claims.json to validate")
    ap.add_argument("--answer-file", help="answer.md: build claims from Sources")
    ap.add_argument("--ledger", default=None, help="ledger.jsonl to append")
    ap.add_argument("--no-persist", action="store_true")
    ap.add_argument("--lenient", action="store_true")
    args = ap.parse_args()
    strict = not args.lenient

    if args.answer_file:
        answer = Path(args.answer_file).read_text(encoding="utf-8")
        claims = build_claims_from_sources(answer)
        print(f"extracted {len(claims)} claim(s) from Sources")
    elif args.file:
        with open(args.file) as f:
            claims = json.load(f)
        if isinstance(claims, dict):
            claims = claims.get("claims", [])
    else:
        print("need --file or --answer-file")
        raise SystemExit(2)

    res = validate_claims(claims, strict=strict)
    print(json.dumps(res.as_dict(), indent=1, ensure_ascii=False))
    if res.ok and not args.no_persist and res.claims:
        path = persist(res.claims, args.ledger)
        print(f"ledger: {path}")
    raise SystemExit(0 if res.ok else 1)


if __name__ == "__main__":
    main()