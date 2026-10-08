#!/usr/bin/env python3
"""
autoresearch.py — autonomous "dreaming" research loop for bib_rag.

Chains the four phases into one CLI entry point:

    Phase 0  SETUP       freeze run_id = hash(graph snapshot, seed, question)
    Phase 1  DREAM       dream.diverge      — graph-driven directions (no LLM)
    Phase 2  FORMULATE   formulate.formulate_all — structured hypotheses
    Phase 3  VERIFY      verify.verify_all  — corpus check + ledger + critic
    Phase 4  CONSOLIDATE persist everything append-only; write final_report.md

A run is a DIRECTORY: <library>/autoresearch/<run_id>/. Everything is
append-only JSONL so a run is auditable and re-runnable: the same run_id
recomputed against the same graph snapshot reproduces the same direction set
(Phase 1 is deterministic); the LLM phases are recorded verbatim.

This module is deliberately SEPARATE from the query/agentic pipeline — it
never touches query_bib_rag.py or agentic_graph.py, it only CALLS their
public functions. "Autonomous research" is a different activity from
"literature retrieval" and must not be entangled with it.

Usage (via the console command):
    bibrag autoresearch "your research question" [--rounds 3] [--n-each 5]
                        [--no-anchor] [--out DIR]

Outputs:
    run_state.json        — run_id, seed, phases, counts, fingerprint
    dream_candidates.jsonl— Phase 1 directions (append)
    hypotheses.jsonl      — Phase 2 hypotheses   (append)
    verification.jsonl    — Phase 3 verifications (append)
    final_report.md       — Phase 4 human summary
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from typing import Dict, List, Optional

try:  # bib_rag-package-try
    from .kb_config import get_config
except ImportError:  # flat (loose-script mode)
    try:
        from kb_config import get_config
    except ImportError:  # src/ on sys.path directly
        import os as _os, sys as _sys
        _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
        from kb_config import get_config


def _autoresearch_dir(cfg: Dict, run_id: str) -> str:
    d = os.path.join(cfg["data_root"], "autoresearch", run_id)
    os.makedirs(d, exist_ok=True)
    return d


def _append_jsonl(path: str, rows: List[dict]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _load_metrics_graph(cfg: Dict):
    """graph_metrics.json (centralities/community) + citation_graph.json
    (titles/edges). Either may be missing → (metrics, None)."""
    import json as _json
    mp = os.path.join(cfg["data_dir"], "graph_metrics.json")
    metrics: Dict = {}
    if os.path.exists(mp):
        try:
            doc = _json.load(open(mp, encoding="utf-8"))
            metrics = doc.get("metrics") or doc.get("nodes") or {}
        except Exception:
            metrics = {}
    gp = cfg.get("reference_graph_path") or os.path.join(
        cfg["data_dir"], "reference_graph.json")
    cg = os.path.join(cfg["data_dir"], "citation_graph.json")
    graph = None
    for path in (gp, cg):
        if os.path.exists(path):
            try:
                graph = _json.load(open(path, encoding="utf-8"))
                break
            except Exception:
                continue
    return metrics, graph


def run(question: str, rounds: int = 3, n_each: int = 5,
        anchor: bool = True, out_dir: Optional[str] = None,
        seed: int = 7) -> Dict:
    """Execute the four phases and persist the run. Returns the run_state.

    `rounds` controls how many dream→formulate→verify cycles run. Each round
    is independent (same question/graph) but writes to the same append-only
    logs; round N's hypotheses are deduped against round N-1's by hypothesis
    id so repeated rounds don't duplicate the same deterministic directions.
    """
    try:
        from . import dream, formulate, verify
    except ImportError:
        import os as _os, sys as _sys
        _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
        import dream, formulate, verify

    cfg = get_config()
    metrics, graph = _load_metrics_graph(cfg)

    # ── Phase 0: SETUP ────────────────────────────────────────────────────
    fp = dream.graph_fingerprint(metrics, graph)
    run_id = f"{fp}-s{seed}"
    if out_dir:
        run_path = out_dir
        os.makedirs(run_path, exist_ok=True)
    else:
        run_path = _autoresearch_dir(cfg, run_id)

    state = {
        "run_id": run_id,
        "question": question,
        "seed": seed,
        "graph_fingerprint": fp,
        "n_metrics_nodes": len(metrics),
        "library": cfg.get("kb_name"),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "phases": {"setup": "done"},
        "rounds_requested": rounds,
        "rounds_done": 0,
        "counts": {"directions": 0, "hypotheses": 0, "verifications": 0},
    }
    _write_state(run_path, state)

    # ── Phases 1-3 per round ──────────────────────────────────────────────
    seen_hyp_ids: set = set()
    all_hyps: List[dict] = []
    all_vers: List[dict] = []
    dir_rows, hyp_rows, ver_rows = [], [], []

    for rnd in range(1, rounds + 1):
        dirs = dream.diverge(question, metrics, graph, n_each=n_each,
                             anchor=anchor)
        dir_rows.extend({"round": rnd, **d} for d in dirs)

        hyps = formulate.formulate_all(dirs, context=question)
        new_hyps = [h for h in hyps if h["id"] not in seen_hyp_ids]
        for h in new_hyps:
            seen_hyp_ids.add(h["id"])
            hyp_rows.append({"round": rnd, **h})

        vers = verify.verify_all(new_hyps)
        for v in vers:
            ver_rows.append({"round": rnd, **v})
        all_hyps.extend(new_hyps)
        all_vers.extend(vers)

        # a round that produced NO new hypotheses means the deterministic
        # directions are exhausted AND the LLM added nothing — further rounds
        # would only pad the logs. Stop BEFORE counting this round as done,
        # so rounds_done reflects rounds that actually contributed.
        if not new_hyps:
            break

        state["rounds_done"] = rnd
        state["counts"] = {"directions": len(dir_rows),
                           "hypotheses": len(hyp_rows),
                           "verifications": len(ver_rows)}
        _write_state(run_path, state)

    # ── Phase 4: CONSOLIDATE ──────────────────────────────────────────────
    _append_jsonl(os.path.join(run_path, "dream_candidates.jsonl"), dir_rows)
    _append_jsonl(os.path.join(run_path, "hypotheses.jsonl"), hyp_rows)
    _append_jsonl(os.path.join(run_path, "verification.jsonl"), ver_rows)

    report = _render_report(state, all_hyps, all_vers)
    with open(os.path.join(run_path, "final_report.md"), "w", encoding="utf-8") as f:
        f.write(report)

    state["phases"]["dream"] = "done"
    state["phases"]["formulate"] = "done"
    state["phases"]["verify"] = "done"
    state["phases"]["consolidate"] = "done"
    state["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    state["run_path"] = run_path
    _write_state(run_path, state)
    return state


def _write_state(run_path: str, state: Dict) -> None:
    with open(os.path.join(run_path, "run_state.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _render_report(state: Dict, hyps: List[dict], vers: List[dict]) -> str:
    """Assemble final_report.md — honest, structured, audit-linked."""
    by_id = {v["hypothesis_id"]: v for v in vers}
    tally = Counter(v["status"] for v in vers)
    lines: List[str] = []
    lines.append(f"# Autonomous Research — {state['question']}\n")
    lines.append(f"- **run_id**: `{state['run_id']}`")
    lines.append(f"- **library**: {state.get('library')}")
    lines.append(f"- **graph fingerprint**: `{state['graph_fingerprint']}` "
                 f"({state['n_metrics_nodes']} nodes)")
    lines.append(f"- **rounds**: {state['rounds_done']}/{state['rounds_requested']}")
    lines.append(f"- **generated**: {state.get('started')} → {state.get('finished')}\n")

    lines.append("## Status tally\n")
    for st in ("verified", "refuted", "speculative", "unsupported"):
        lines.append(f"- **{st}**: {tally.get(st, 0)}")
    lines.append("")

    # group hypotheses by status, verified first — the honest ordering
    order = {"verified": 0, "refuted": 1, "speculative": 2, "unsupported": 3}
    ranked = sorted(hyps, key=lambda h: order.get(
        (by_id.get(h["id"]) or {}).get("status", "unsupported"), 9))

    lines.append("## Hypotheses\n")
    for h in ranked:
        v = by_id.get(h["id"]) or {}
        st = v.get("status", "?")
        lines.append(f"### [{st}] {h['id']}  ({h.get('from_source','?')})\n")
        lines.append(f"**Claim.** {h.get('claim','')}\n")
        if h.get("rationale"):
            lines.append(f"*Why the graph suggested it:* {h['rationale']}\n")
        if h.get("falsifier"):
            lines.append(f"*Falsifier:* {h['falsifier']}\n")
        sc = h.get("scores") or {}
        lines.append("Scores: "
                     + ", ".join(f"{k}={sc.get(k)}" for k in
                                 ("interesting", "feasible", "testable")) + "\n")
        lines.append(f"Anchors: {', '.join(h.get('anchors') or [])}\n")
        if v:
            lines.append(f"Verification: {v.get('n_evidence', 0)} evidence "
                         f"chunk(s); critic={((v.get('critic') or {}).get('verdict'))}"
                         f" — {(v.get('critic') or {}).get('reason','')}\n")
            if v.get("ledger_errors"):
                lines.append(f"Ledger: {len(v['ledger_errors'])} error(s)\n")
        lines.append("")

    lines.append("---\n")
    lines.append("_Method: deterministic graph divergence (Phase 1, no LLM) → "
                 "LLM hypothesis formulation (Phase 2) → corpus verification "
                 "with claim_ledger strict mode + harsh critic (Phase 3). "
                 "`verified` requires whitelisted evidence; unverifiable "
                 "hypotheses are reported as `unsupported`, never inflated._\n")
    return "\n".join(lines)


# ─── CLI ─────────────────────────────────────────────────────────────────────
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="bibrag autoresearch",
        description="Autonomous 'dreaming' research loop over the library.")
    ap.add_argument("question")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--n-each", type=int, default=5)
    ap.add_argument("--no-anchor", action="store_true",
                    help="disable domain anchoring (pure structural dream)")
    ap.add_argument("--out", default=None, help="run directory (default: <lib>/autoresearch/<run_id>)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    st = run(args.question, rounds=args.rounds, n_each=args.n_each,
             anchor=not args.no_anchor, out_dir=args.out, seed=args.seed)
    c = st["counts"]
    print(f"run_id: {st['run_id']}")
    print(f"rounds: {st['rounds_done']}/{st['rounds_requested']}")
    print(f"directions={c['directions']}  hypotheses={c['hypotheses']}  "
          f"verifications={c['verifications']}")
    print(f"report: {os.path.join(st['run_path'], 'final_report.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
