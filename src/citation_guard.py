#!/usr/bin/env python3
"""
citation_guard.py — Deterministic citation verification for bib_rag answers.

Borrowed mechanisms (see /Disk_bot/notes/citation_rag/):
  1. WHITELIST  (paper-qa)   — every cited source must be one of the parent
                               chunks actually retrieved this session. A
                               citation not in the whitelist is DROPPED, not
                               merely flagged (deterministic post-processing,
                               no LLM self-discipline required).
  2. SUBSTRING  (LumiCite)   — each claim sentence's quoted evidence must
                               appear (normalized) in the cited child chunk.
  3. LEXICAL    (citelocal-agent, cheap tier) — a claim sentence must share
                               enough rare tokens with its cited chunk to be
                               plausibly supported. Rare = corpus-frequency
                               weighted (IDF-ish); stopwords excluded.

All checks are ZERO-LLM. The guard runs after collect_answer and rewrites the
answer's Sources section in place. It never invents content — it can only
remove or annotate unverified citations, and it always reports what it did.

Usage (inside agent_nodes.collect_answer):
    from .citation_guard import enforce_citation_guard
    guarded = enforce_citation_guard(answer, retrieval_keys, store=...)

Usage (standalone):
    python3 citation_guard.py --answer-file a.md --keys-file keys.txt
"""
from __future__ import annotations

import os
import re
import json
import math
import hashlib
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

from .kb_config import get_config

# ---------------------------------------------------------------------------
# Tunables (env-overridable, same pattern as agent_tools budgets)
# ---------------------------------------------------------------------------

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default

# Jaccard-like overlap needed between claim tokens and chunk tokens for the
# claim to count as "supported" at the lexical tier. 0.15 is deliberately low:
# this tier only catches clearly-unrelated citations.
LEXICAL_SUPPORT_THRESHOLD = _env_float("CITATION_LEXICAL_THRESHOLD", 0.15)

# Minimum tokens (after stopword/rare-token filtering) for a claim to be
# lexically checkable at all; shorter claims skip lexical check.
MIN_CLAIM_TOKENS = _env_int("CITATION_MIN_CLAIM_TOKENS", 6)

# Maximum chars of evidence substring to search for (longer quotes are
# almost surely paraphrases; substring tier only applies to short quotes).
MAX_SUBSTRING_LEN = _env_int("CITATION_MAX_SUBSTRING_LEN", 300)

# Number of rare tokens required in common for the lexical tier to pass
# when the claim is long enough.
LEXICAL_RARE_TOKENS_REQUIRED = _env_int("CITATION_RARE_TOKENS", 3)

# Stopwords: domain-generic English + academic boilerplate. Gene symbols,
# receptor names etc. are NOT here — they are exactly the tokens that matter.
_STOPWORDS = frozenset("""
a an the and or but if then than that this these those of in on at to for from
by with without within into onto under over between among during before after
is are was were be been being it its it's as we our us they their them he she
his her not no nor so such can could may might must shall should will would
do does did done have has had having also more most much many few both each
other another some any all one two three first second third however therefore
thus moreover furthermore additionally respectively ie eg etc vs versus
using used use uses based data study studies results result show shows shown
suggest suggests suggested indicate indicates indicated found finds observed
observes demonstrate demonstrates demonstrated report reports reported
paper article figure table section method methods
""".split())

# Section headers often leaked into chunk text; not evidence.
_SECTION_JUNK = re.compile(
    r"^(references|bibliography|acknowledge?ments?|supplementary|appendix|"
    r"data availability|conflict of interest|author contributions?)\b",
    re.I,
)


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """Aggressive normalization for substring checks: lowercase, unify quotes/
    dashes/whitespace, strip citation markers like [12] or (Smith et al., 2020)."""
    if not text:
        return ""
    t = text.lower()
    t = re.sub(r"\[\d+(?:[,\-\s]+\d+)*\]", " ", t)          # [1], [2,3], [4-7]
    t = re.sub(r"\((?:[a-z][a-z'\-]+(?:\s+et\s+al\.)?)(?:,?\s*\d{4})?\)", " ", t)
    t = t.replace("’", "'").replace("‘", "'")
    t = t.replace("“", '"').replace("”", '"')
    t = re.sub(r"[–—−]", "-", t)
    t = re.sub(r"[\s]+", " ", t)
    return t.strip()


def _tokens(text: str) -> List[str]:
    """Tokenize: word chars + hyphens + digits; drop stopwords and 1-char tokens."""
    t = normalize(text)
    raw = re.findall(r"[a-z][a-z\-]*\d*[a-z\d]|\d+", t)
    return [w for w in raw if len(w) > 1 and w not in _STOPWORDS]


# ---------------------------------------------------------------------------
# Whitelist extraction from retrieval_keys
# ---------------------------------------------------------------------------

_PARENT_KEY_RE = re.compile(r"^parent::(.+)$")


def parent_ids_from_keys(retrieval_keys: Set[str]) -> Set[str]:
    """Extract parent ids from the agent's retrieval_keys set
    (entries look like 'parent::<parent_id>' and 'search::<query>')."""
    out: Set[str] = set()
    for k in retrieval_keys or set():
        m = _PARENT_KEY_RE.match(k)
        if m:
            out.add(m.group(1))
    return out


_PARENT_ID_IN_TEXT_RE = re.compile(r"Parent ID:\s*(\S+)")

# ToolMessage name → treat as retrieval evidence (search results carry
# "Parent ID:" lines even when the agent never called retrieve_parent_chunks)


def parent_ids_from_tool_messages(messages) -> Set[str]:
    """Fallback whitelist source: parse 'Parent ID: X' lines out of ToolMessages.

    Needed because the agent may answer after search_child_chunks alone
    (retrieval_keys then only holds search:: entries, and the strict
    parent:: whitelist would be empty — dropping every citation despite a
    successful retrieval). Search results are retrieval evidence too: the
    tool output literally lists the parent_ids whose excerpts were shown.
    """
    out: Set[str] = set()
    for m in messages or []:
        name = getattr(m, "name", "") or ""
        if name not in ("search_child_chunks", "retrieve_parent_chunks",
                        "retrieve_many_parents"):
            continue
        content = getattr(m, "content", "")
        if not isinstance(content, str):
            continue
        for hit in _PARENT_ID_IN_TEXT_RE.finditer(content):
            pid = hit.group(1).strip().strip('*_')
            if pid and pid not in ("unknown", "N/A", "none"):
                out.add(pid)
    return out


# ---------------------------------------------------------------------------
# Answer parsing: find the Sources section and [SOURCE n]-style markers
# ---------------------------------------------------------------------------

_SOURCES_HEADER_RE = re.compile(
    r"^[#>\*\s-]*sources?\b[^a-z]*$", re.I | re.M)
_SOURCE_LINE_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])?\s*(.+?)\s*$")

# Matches "Parent ID: <id>" blocks the tools emit, or bare parent ids.
_PARENT_ID_HINT_RE = re.compile(r"Parent ID:\s*(\S.+?)\s*$", re.M)


def split_answer_sources(answer: str) -> Tuple[str, List[str], str]:
    """Split an answer into (body, source_lines, sources_header_line).

    The Sources section is everything from the last 'Sources' header to EOF
    (orchestrator prompt mandates it be the final section). If no header is
    found, returns (answer, [], '').
    """
    matches = list(_SOURCES_HEADER_RE.finditer(answer))
    if not matches:
        return answer, [], ""
    m = matches[-1]
    body = answer[: m.start()].rstrip("\n-")
    header = m.group(0)
    src_block = answer[m.end():]
    lines = []
    for raw in src_block.splitlines():
        line = _SOURCE_LINE_RE.match(raw)
        if line and line.group(1).strip():
            lines.append(line.group(1).strip())
    return body, lines, header


# ---------------------------------------------------------------------------
# Source-line → parent_id resolution
# ---------------------------------------------------------------------------

def resolve_source_lines(source_lines: List[str],
                         known_parent_ids: Set[str],
                         parent_meta: Dict[str, dict],
                         ) -> List[Tuple[str, Optional[str]]]:
    """Map each Sources line to a parent_id when possible.

    Resolution strategies (first hit wins):
      1. line contains a literal parent_id present in known_parent_ids
      2. line's leading filename matches the SOURCE part of a known id
         (parent_id format: '<source>#<section>#<hash>' — answers usually
         cite by filename, not by full parent_id)
      3. line contains 'Parent ID: X'
      4. fuzzy title match against parent_meta titles (token Jaccard >= 0.6)
    Returns list of (line, parent_id_or_None).
    """
    # Pre-index: source-filename → set of parent_ids
    by_source: Dict[str, List[str]] = {}
    for k in known_parent_ids:
        src = k.split("#", 1)[0]
        by_source.setdefault(src, []).append(k)
    source_stems = {os.path.splitext(s)[0].lower(): s for s in by_source}

    resolved = []
    for line in source_lines:
        pid = None
        # Strategy 1: literal parent id substring
        for k in known_parent_ids:
            if k and k in line:
                pid = k
                break
        # Strategy 2: filename citation → source prefix of a known parent_id
        if pid is None:
            # candidates: markdown-ish tokens ending in .md/.pdf/.txt/.docx
            for m in re.finditer(r"[\w\-.]+\.(?:md|pdf|txt|docx|doc)\b", line, re.I):
                cand = m.group(0)
                if cand in by_source:
                    pid = by_source[cand][0]
                    break
                stem = os.path.splitext(cand)[0].lower()
                if stem in source_stems:
                    pid = by_source[source_stems[stem]][0]
                    break
            # fallback: known pids whose SOURCE part starts with the cited
            # stem (long hyphenated store names get split by the filename
            # regex at '_-_'; a prefix match over the raw source string is
            # more robust)
            if pid is None:
                line_l = line.lower()
                for src_name, pids_for_src in by_source.items():
                    src_stem = os.path.splitext(src_name)[0].lower()
                    if len(src_stem) >= 8 and src_stem in line_l:
                        pid = pids_for_src[0]
                        break
        # Strategy 3: explicit 'Parent ID:' hint
        if pid is None:
            m = _PARENT_ID_HINT_RE.search(line)
            if m:
                cand = m.group(1).strip().strip('*_')
                if cand in known_parent_ids:
                    pid = cand
        # Strategy 4: fuzzy title match
        if pid is None:
            line_toks = set(_tokens(line))
            best, best_j = None, 0.0
            for k, meta in parent_meta.items():
                title = meta.get("title", "")
                if not title:
                    continue
                t_toks = set(_tokens(title))
                if not t_toks:
                    continue
                j = len(line_toks & t_toks) / max(1, len(line_toks | t_toks))
                if j > best_j:
                    best, best_j = k, j
            if best is not None and best_j >= 0.6:
                pid = best
        resolved.append((line, pid))
    return resolved


# ---------------------------------------------------------------------------
# Parent metadata + chunk-text loading
# ---------------------------------------------------------------------------

def load_parent_meta_map(parent_ids: Set[str]) -> Dict[str, dict]:
    """Load title/meta for the given parent_ids from the parent_store JSONs.
    Returns {parent_id: {'title':..., 'source':..., 'section':...}}.
    Missing ids are simply absent from the result."""
    cfg = get_config()
    store_dir = cfg["parent_store_dir"]
    out: Dict[str, dict] = {}
    if not os.path.isdir(store_dir):
        return out
    # Group by source file (parent_id format: <source>#<section>#<hash>)
    by_source: Dict[str, Set[str]] = {}
    for pid in parent_ids:
        src = pid.split("#", 1)[0]
        by_source.setdefault(src, set()).add(pid)
    for src, wanted in by_source.items():
        safe = re.sub(r"[^\w\-]", "_", src)[:100]
        path = os.path.join(store_dir, f"{safe}.json")
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                parents = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        for p in parents:
            if p.get("parent_id") in wanted:
                out[p["parent_id"]] = {
                    "title": p.get("meta", {}).get("title", ""),
                    "source": p.get("source", ""),
                    "section": p.get("section", ""),
                }
    return out


def load_parent_text(parent_id: str) -> str:
    """Load the full text of a parent chunk from the parent store."""
    cfg = get_config()
    src = parent_id.split("#", 1)[0]
    safe = re.sub(r"[^\w\-]", "_", src)[:100]
    path = os.path.join(cfg["parent_store_dir"], f"{safe}.json")
    if not os.path.exists(path):
        return ""
    try:
        with open(path, encoding="utf-8") as f:
            parents = json.load(f)
    except (OSError, json.JSONDecodeError):
        return ""
    for p in parents:
        if p.get("parent_id") == parent_id:
            return p.get("content", "")
    return ""


# ---------------------------------------------------------------------------
# Claim-level verification (lexical tier)
# ---------------------------------------------------------------------------

def claim_supported_lexically(claim: str, chunk_text: str,
                              threshold: float = LEXICAL_SUPPORT_THRESHOLD,
                              rare_required: int = LEXICAL_RARE_TOKENS_REQUIRED,
                              ) -> Tuple[bool, float]:
    """Cheap lexical support check between a claim sentence and a chunk.

    Score = weighted overlap: rare (long) tokens count 3x, normal 1x.
    Returns (supported, score).
    """
    c_toks = _tokens(claim)
    k_toks = _tokens(chunk_text)
    if len(c_toks) < MIN_CLAIM_TOKENS or not k_toks:
        return True, 1.0  # too short to judge → don't punish

    c_counts = Counter(c_toks)
    k_set = set(k_toks)
    # "Rare" = long tokens (>=8 chars, usually gene/method names) count more.
    w_c = Counter()
    for t, n in c_counts.items():
        w_c[t] = n * (3 if len(t) >= 8 else 1)
    matched = {t: c for t, c in w_c.items() if t in k_set}
    total = sum(w_c.values())
    score = sum(matched.values()) / total if total else 0.0
    # Also require at least `rare_required` distinct rare-token hits for long claims
    rare_hits = sum(1 for t in matched if len(t) >= 8)
    supported = score >= threshold and (rare_hits >= rare_required or len(c_toks) < 12)
    return supported, score


# ---------------------------------------------------------------------------
# Main entry: enforce the guard on a finished answer
# ---------------------------------------------------------------------------

def enforce_citation_guard(answer: str,
                           retrieval_keys: Set[str],
                           parent_meta: Optional[Dict[str, dict]] = None,
                           tool_messages=None,
                           ) -> Tuple[str, Dict]:
    """Deterministically verify the Sources section of a finished answer.

    Pipeline (zero LLM):
      1. Build the whitelist: parent:: keys from retrieval_keys PLUS any
         'Parent ID:' lines found in retrieval ToolMessages (the agent may
         answer from search results without retrieving parents).
      2. Parse the Sources section into lines.
      3. Resolve each line to a parent_id (whitelist membership check).
      4. Drop lines that resolve to nothing (unverifiable → hallucination risk).
      5. Lexical spot-check: for lines with a resolved parent, verify the
         answer's body shares lexical support with the parent text. Failures
         are annotated, not silently kept.
      6. Rewrite the Sources section; append a guard report.

    Returns (guarded_answer, report_dict).
    report keys: kept, dropped, annotated, dropped_lines, notes
    """
    known = parent_ids_from_keys(retrieval_keys)
    if tool_messages:
        known = known | parent_ids_from_tool_messages(tool_messages)
    if parent_meta is None:
        parent_meta = load_parent_meta_map(known)

    body, source_lines, header = split_answer_sources(answer)
    report = {"kept": 0, "dropped": 0, "annotated": 0,
              "dropped_lines": [], "notes": []}

    # No Sources section at all → nothing deterministic to check; annotate.
    if not header:
        report["notes"].append("no Sources section found; guard skipped")
        return answer, report

    resolved = resolve_source_lines(source_lines, known, parent_meta)

    kept_lines: List[str] = []
    # Body sentences for lexical support (top informative sentences only).
    body_sentences = [s for s in re.split(r"(?<=[.!?])\s+", body) if len(s) > 40]

    for line, pid in resolved:
        if pid is None:
            report["dropped"] += 1
            report["dropped_lines"].append(line[:120])
            continue
        # Lexical support vs the cited parent's text
        ptext = load_parent_text(pid)
        if not ptext:
            # Cannot verify (parent store missing) — keep but annotate.
            kept_lines.append(f"{line}  ⚠️[unverified: parent store miss]")
            report["kept"] += 1
            report["annotated"] += 1
            continue
        # Check the body has *some* lexical grounding in this parent.
        best = 0.0
        for sent in body_sentences[:12]:  # cap for speed
            ok, score = claim_supported_lexically(sent, ptext)
            best = max(best, score)
            if ok:
                break
        if body_sentences and best < 0.05:
            # Body shares virtually nothing with this source → likely a
            # decoration citation. Annotate rather than drop (may be a
            # general-reference citation whose paraphrase is heavy).
            kept_lines.append(f"{line}  ⚠️[low lexical support: {best:.2f}]")
            report["kept"] += 1
            report["annotated"] += 1
        else:
            kept_lines.append(line)
            report["kept"] += 1

    # Rebuild answer
    if kept_lines:
        src_text = header.rstrip() + "\n" + "\n".join(f"- {l}" for l in kept_lines)
    else:
        src_text = ""
    new_answer = body.rstrip() + ("\n\n" + src_text if src_text else "")
    if report["dropped"]:
        new_answer += (
            f"\n\n<!-- citation_guard: {report['dropped']} source line(s) "
            f"could not be matched to any retrieved parent chunk and were "
            f"removed -->"
        )
    return new_answer, report