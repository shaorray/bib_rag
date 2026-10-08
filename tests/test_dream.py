#!/usr/bin/env python3
"""test_dream.py — regression tests for src/dream.py (autonomous-research
Phase 1: deterministic graph divergence).

Pure: no network, no LLM, no real library. Fixtures are inline mini-graphs
whose structure is hand-computable, so the tests pin the ALGORITHM (bridge
detection, uncited-pair logic, orphan ranking, determinism), not a snapshot.

Run:  /usr/bin/python3 tests/test_dream.py
"""
import os
import sys
import tempfile
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="dreamtest_")
os.environ["BIB_RAG_ROOT"] = _TMP
os.environ.setdefault("HOME", "/home/rui")
os.makedirs(os.path.join(_TMP, "parent_store"), exist_ok=True)
os.makedirs(os.path.join(_TMP, "data"), exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import dream  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append((name, detail))


def M(comm, btw=0.0, rcr=0.0, icc=0, cc=0, pr=0.0, title="", topic=""):
    return {"community_id": comm, "betweenness": btw, "rcr": rcr,
            "in_corpus_cited_by": icc, "citation_count": cc,
            "pagerank": pr, "title": title, "topic": topic}


# ---------------------------------------------------------------------------
# D1 bridges: only high-betweenness nodes; needs >=2 communities
# ---------------------------------------------------------------------------
print("D1 bridges")
metrics = {
    "a.md": M(1, btw=0.5, title="ephrin signalling in axon guidance"),
    "b.md": M(2, btw=0.0001, title="retinal mapping"),
    "c.md": M(2, btw=0.0, title="cell sorting"),
}
out = dream._d1_bridges(metrics, set(), n_each=5)
check("returns bridge candidates", len(out) >= 1)
check("top bridge is the high-betweenness node", out[0]["anchors"] == ["a.md"])
check("bridge carries its betweenness", out[0]["signal"]["betweenness"] == 0.5)
check("zero-betweenness nodes excluded",
      all(c["signal"]["betweenness"] > 0 for c in out))

# single community → no bridges (can't span)
check("one community → no bridges",
      dream._d1_bridges({"x.md": M(1, btw=0.9)}, set(), 5) == [])

# question focus: default mode RANKS (keeps all, boosts matches);
# "filter" mode is the strict variant.
only = dream._d1_bridges(metrics, dream._tokens("axon guidance"), 5)
check("rank mode keeps matching bridge", any(c["anchors"] == ["a.md"] for c in only))
kept = dream._d1_bridges(metrics, dream._tokens("volcano petrology"), 5)
check("rank mode keeps non-matching bridge (boost, not filter)",
      any(c["anchors"] == ["a.md"] for c in kept))
filt = dream._d1_bridges(metrics, dream._tokens("volcano petrology"), 5,
                         focus="filter")
check("filter mode drops non-matching bridge", filt == [])


# ---------------------------------------------------------------------------
# D2 uncited pairs: same community + shared tokens + NOT adjacent
# ---------------------------------------------------------------------------
print("D2 uncited pairs")
graph = {"edges": [["a.md", "x.md"]]}  # a-x cite each other
metrics2 = {
    "a.md": M(1, title="ephrin receptor repulsion axon guidance"),
    "b.md": M(1, title="ephrin receptor repulsion cell migration"),
    "x.md": M(1, title="ephrin receptor repulsion axon guidance"),
    "c.md": M(2, title="unrelated geology subduction"),
}
out = dream._d2_uncited(metrics2, graph, set(), n_each=10)
pairs = {tuple(sorted(c["anchors"])) for c in out}
check("finds a non-cited same-community pair", ("a.md", "b.md") in pairs)
check("excludes adjacent pair (a,x cite each other)",
      not any("a.md" in p and "x.md" in p for p in pairs))
check("excludes cross-community pair (a,c)",
      not any("a.md" in p and "c.md" in p for p in pairs))
check("requires >=2 shared tokens",
      all(len(c["signal"]["shared_tokens"]) >= 2 for c in out))
check("no graph → no pairs", dream._d2_uncited(metrics2, None, set(), 10) == [])
check("no metrics → no pairs", dream._d2_uncited({}, graph, set(), 10) == [])

# determinism: identical inputs → identical output
out2 = dream._d2_uncited(metrics2, graph, set(), n_each=10)
check("D2 deterministic", [c["id"] for c in out] == [c["id"] for c in out2])


# ---------------------------------------------------------------------------
# D3 orphans: high RCR * low in-corpus citations
# ---------------------------------------------------------------------------
print("D3 orphans")
metrics3 = {
    "famous_neglected.md": M(1, rcr=8.0, icc=0, cc=500, title="ephrin gradient"),
    "famous_used.md": M(1, rcr=8.0, icc=20, cc=500, title="ephrin gradient"),
    "semi_neglected.md": M(1, rcr=4.0, icc=10, cc=300, title="ephrin gradient"),
    "obscure.md": M(1, rcr=0.1, icc=0, cc=2, title="ephrin oddity"),
    "zero.md": M(1, rcr=0.0, icc=0, cc=0, title="empty"),
}
out = dream._d3_orphans(metrics3, set(), n_each=10)
ids = [c["anchors"][0] for c in out]
check("fully-cited high-RCR node excluded (neglect=0)",
      "famous_used.md" not in ids)
check("neglected high-RCR ranks first", ids and ids[0] == "famous_neglected.md")
check("semi-cited ranks below fully-neglected",
      ids.index("famous_neglected.md") < ids.index("semi_neglected.md"))
check("zero-signal node excluded", "zero.md" not in ids)
check("neglect score = rcr/rcr_max * (1 - icc/icc_max)",
      abs(out[0]["signal"]["neglect_score"] - 1.0) < 1e-9,
      f"got {out[0]['signal']['neglect_score']}")

# question focus
foc = dream._d3_orphans(metrics3, dream._tokens("ephrin gradient"), 10)
check("focus keeps matching orphan", any(c["anchors"][0] == "famous_neglected.md" for c in foc))


# ---------------------------------------------------------------------------
# fingerprint: stable, changes when the partition changes
# ---------------------------------------------------------------------------
print("graph_fingerprint")
fp1 = dream.graph_fingerprint(metrics3, graph)
fp2 = dream.graph_fingerprint(metrics3, graph)
check("fingerprint deterministic", fp1 == fp2)
metrics3b = dict(metrics3)
metrics3b["famous_neglected.md"] = M(3, rcr=8.0, icc=0)  # community moved
check("fingerprint changes on community drift",
      dream.graph_fingerprint(metrics3b, graph) != fp1)


# ---------------------------------------------------------------------------
# diverge(): full contract
# ---------------------------------------------------------------------------
print("diverge()")
merged_metrics = {**metrics, **metrics2, **metrics3}
cands = dream.diverge("ephrin receptor repulsion", merged_metrics, graph, n_each=3)
check("diverge returns a list", isinstance(cands, list))
check("every candidate has the required keys",
      all({"id", "source", "anchors", "signal", "topic", "note"} <= set(c) for c in cands))
check("sources are from the three families",
      all(c["source"] in ("bridge", "uncited", "orphan") for c in cands))
check("ids unique", len({c["id"] for c in cands}) == len(cands))
check("deterministic across calls",
      [c["id"] for c in cands] ==
      [c["id"] for c in dream.diverge("ephrin receptor repulsion", merged_metrics, graph, n_each=3)])
check("empty metrics → empty (no raise)", dream.diverge("q", {}, None) == [])

# every direction is auditable: it names at least one real graph node
all_nodes = set(merged_metrics)
check("every direction anchors on a real node",
      all(c["anchors"] and all(a in all_nodes for a in c["anchors"]) for c in cands))

print()
if FAILURES:
    print(f"✗ {len(FAILURES)} FAILURES:")
    for name, detail in FAILURES:
        print(f"  - {name}: {detail}")
    sys.exit(1)
print("✓ all checks passed")
