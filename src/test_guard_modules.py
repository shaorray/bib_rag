#!/usr/bin/env python3
"""
test_guard_modules.py — Offline unit tests for the six-gap hardening work
(citation_guard / zotero_match / evidence_gate / hybrid_search / chunking
captions / evaluate.citation_faithfulness). NO LLM, NO network, NO ChromaDB
writes: everything runs against tmp fixtures.

Run:  python3 -m pytest src/test_guard_modules.py -v   (or plain python3)
"""
import os
import sys
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# citation_guard
# ---------------------------------------------------------------------------

def test_normalize_strips_citation_markers():
    from src.citation_guard import normalize
    t = normalize("Ephb1 binds ephrin-B1 [12,13] (Parker et al., 2021).")
    assert "12" not in t.replace("ephrin-b1", "")  # bracket refs gone
    assert "[" not in t and "et al" not in t
    print("  ✓ normalize strips [n] and (Author, year)")


def test_parent_ids_from_keys():
    from src.citation_guard import parent_ids_from_keys
    keys = {"parent::a.md#results#h1", "search::ephb1 query", "search::another"}
    assert parent_ids_from_keys(keys) == {"a.md#results#h1"}
    print("  ✓ parent_ids_from_keys extracts only parent:: entries")


def test_split_answer_sources():
    from src.citation_guard import split_answer_sources
    ans = "Body text.\n\n---\n**Sources:**\n- a.md\n- b.md\n"
    body, lines, header = split_answer_sources(ans)
    assert "Sources" in header and lines == ["a.md", "b.md"], (header, lines)
    assert body.strip() == "Body text."
    # no header
    body2, lines2, header2 = split_answer_sources("just body, no sources")
    assert lines2 == [] and header2 == ""
    print("  ✓ split_answer_sources parses Sources section")


def test_enforce_guard_drops_unknown_source(tmp_parent_store):
    """A source line that resolves to NO known parent must be dropped."""
    import src.citation_guard as cg
    pid = "fake_paper.md#results#abc123"
    # fake parent store
    os.makedirs(tmp_parent_store, exist_ok=True)
    with open(os.path.join(tmp_parent_store, "fake_paper_md.json"), "w") as f:
        json.dump([{"parent_id": pid, "source": "fake_paper.md", "section": "results",
                    "content": "Ephb1 receptor signaling drives axon repulsion in "
                               "the developing hindbrain through ephrin-B1 binding "
                               "and contact-mediated repulsion responses.",
                    "word_count": 20, "char_count": 150,
                    "meta": {"title": "Ephb1 drives axon repulsion", "year": "2021"}}], f)

    keys = {f"parent::{pid}", "search::ephb1"}
    answer = ("Ephb1 receptor signaling drives axon repulsion through ephrin-B1. "
              "Ephb1 knockdown disrupts hindbrain boundary formation significantly.\n\n"
              "---\n**Sources:**\n"
              f"- fake_paper.md (Ephb1 drives axon repulsion, 2021)\n"
              "- hallucinated_paper_that_was_never_retrieved.md\n")

    # point the guard at the fake store
    orig = cg.get_config
    cg.get_config = lambda: {"parent_store_dir": os.path.dirname(
        os.path.join(os.path.abspath(__file__)))}  # placeholder, patched below

    # Simpler: monkeypatch load_parent_text/meta paths via env-free injection
    calls = {"meta": {pid: {"title": "Ephb1 drives axon repulsion", "source": "fake_paper.md", "section": "results"}},
             "text": "Ephb1 receptor signaling drives axon repulsion in the developing "
                     "hindbrain through ephrin-B1 binding and contact-mediated repulsion."}
    cg.get_config = orig  # restore
    orig_meta, orig_text = cg.load_parent_meta_map, cg.load_parent_text
    cg.load_parent_meta_map = lambda ids: calls["meta"]
    cg.load_parent_text = lambda pid_: calls["text"]
    try:
        guarded, report = cg.enforce_citation_guard(answer, keys)
        assert report["dropped"] == 1, report
        assert "hallucinated_paper" not in guarded
        assert report["kept"] == 1
    finally:
        cg.load_parent_meta_map = orig_meta
        cg.load_parent_text = orig_text = orig_text  # noqa
        cg.load_parent_text = orig_text
    print("  ✓ enforce_citation_guard drops unverifiable source line")


def test_claim_supported_lexically():
    from src.citation_guard import claim_supported_lexically
    chunk = ("Ephb1 receptor signaling drives axon repulsion in the developing "
             "hindbrain through ephrin-B1 binding and contact-mediated repulsion.")
    ok, s = claim_supported_lexically(
        "Ephb1 signaling drives axon repulsion in the hindbrain via ephrin-B1.", chunk)
    assert ok, s
    ok2, s2 = claim_supported_lexically(
        "Quantum dots emit photons under laser excitation in semiconductor physics.", chunk)
    assert not ok2, s2
    print(f"  ✓ lexical support: related={s:.2f} unrelated={s2:.2f}")


# ---------------------------------------------------------------------------
# zotero_match
# ---------------------------------------------------------------------------

def test_zotero_match_rejects_wrong_paper():
    from src.zotero_match import verify_zotero_hit, pick_best_hit
    ok, s, r = verify_zotero_hit(
        "A Mechanical Model of Cell Segregation Driven by Differential Adhesion",
        "10.1371/journal.pone.0043226",
        {"title": "Interplay of Eph-Ephrin Signalling and Cadherin Function",
         "doi": "10.3389/fcell.2021.784039"})
    assert not ok and r == "doi-conflict", (ok, r)
    # pick_best_hit skips wrong candidates
    best = pick_best_hit(
        [{"key": "A", "title": "Interplay of Eph-Ephrin Signalling", "doi": ""},
         {"key": "B", "title": "A mechanical model of cell segregation driven by differential adhesion",
          "doi": "10.1371/journal.pone.0043226"}],
        "A Mechanical Model of Cell Segregation Driven by Differential Adhesion",
        "10.1371/journal.pone.0043226")
    assert best and best["key"] == "B"
    none = pick_best_hit([{"key": "X", "title": "quantum dot photonics", "doi": ""}],
                         "Eph receptor signaling", "")
    assert none is None
    print("  ✓ zotero_match: DOI conflict rejected; pick_best_hit correct")


# ---------------------------------------------------------------------------
# evidence_gate
# ---------------------------------------------------------------------------

class _Msg:
    def __init__(self, content, name="search_child_chunks"):
        self.content = content
        self.name = name


def test_evidence_gate_no_evidence():
    from src.evidence_gate import evidence_coverage, gap_instruction, coverage_block
    msgs = [_Msg("NO_RELEVANT_CHUNKS"), _Msg("RETRIEVAL_ERROR: timeout")]
    keys = {"search::ephb1 boundary formation", "search::ephrin-B1 repulsion"}
    cov = evidence_coverage(msgs, keys)
    assert cov["no_evidence"] is True
    assert len(cov["queries"]) == 2
    gi = gap_instruction(cov)
    assert "EVIDENCE GATE" in gi and "NO retrieval" in gi
    cb = coverage_block(cov)
    assert "Evidence coverage" in cb
    print("  ✓ evidence_gate: all-empty retrievals → no_evidence + instructions")


def test_evidence_gate_productive():
    from src.evidence_gate import evidence_coverage
    msgs = [_Msg("--- RESULT 1 (similarity: 0.8) ---\nSource: x.md\nContent: Ephb1..."),
            _Msg("--- PARENT CHUNK ---\nContent: ephrin signaling")]
    keys = {"search::ephb1"}
    cov = evidence_coverage(msgs, keys)
    assert cov["no_evidence"] is False and cov["n_results_total"] >= 2
    print("  ✓ evidence_gate: productive retrievals pass")


# ---------------------------------------------------------------------------
# hybrid_search (RRF fusion — offline, synthetic rankings)
# ---------------------------------------------------------------------------

def test_rrf_fusion():
    from src.hybrid_search import HybridIndex
    idx = HybridIndex(fts_path=":memory:")  # in-memory; search() degrades gracefully
    vec = [{"parent_id": "a#s#1", "source": "a.md", "section": "results",
            "text": "Ephb1 binds ephrin-B1", "similarity": 0.9},
           {"parent_id": "b#s#h", "source": "b.md", "section": "intro",
            "text": "Cadherin adhesion", "similarity": 0.7}]
    bm = [{"parent_id": "b#s#h", "source": "b.md", "section": "intro",
           "text": "Cadherin adhesion", "bm25": 12.0},
          {"parent_id": "c#s#h", "source": "c.md", "section": "results",
           "text": "Ephrin-B1 reverse signaling", "bm25": 9.0}]
    fused = idx.rrf_fuse(vec, bm, top_k=3)
    # 'b' appears in BOTH channels → should rank #1
    assert fused[0]["parent_id"] == "b#s#h", fused
    assert fused[0]["channels"] == "bm25+vec"
    assert len(fused) == 3
    print("  ✓ RRF fusion: dual-channel chunk ranks first")


def test_fts_escape():
    from src.hybrid_search import _fts_escape
    assert _fts_escape('Ephb1 "quoted" (term):') == "Ephb1 quoted term"
    print("  ✓ FTS query escaping")


def test_fts_index_roundtrip(tmpdir):
    """Build a tiny FTS index from a fake source and search it."""
    os.environ["BIB_RAG_ROOT"] = str(tmpdir)
    os.environ.pop("BIB_RAG_KB_NAME", None)
    import importlib
    import src.kb_config as kc
    importlib.reload(kc)
    os.makedirs(os.path.join(str(tmpdir), "parent_store"), exist_ok=True)
    parents = [{"parent_id": "t.md#results#h1", "source": "t.md", "section": "results",
                "content": "Ephb1 mutant embryos show failed axon guidance. " * 10,
                "word_count": 60, "char_count": 500, "meta": {"title": "T"}}]
    with open(os.path.join(str(tmpdir), "parent_store", "t_md.json"), "w") as f:
        json.dump(parents, f)
    # reload hybrid_search with the new root
    import src.hybrid_search as hs
    importlib.reload(hs)
    idx = hs.HybridIndex()
    n = idx.upsert_source("t_md")
    assert n > 0
    hits = idx.bm25_search("Ephb1 mutant axon", limit=5)
    assert hits and "Ephb1" in hits[0]["text"]
    # gene symbol exact hit (the whole point of the BM25 channel)
    hits2 = idx.bm25_search("Mab21l2", limit=5)  # absent gene → empty OK
    print(f"  ✓ FTS roundtrip: {n} children, query hits={len(hits)}, miss={len(hits2)}")


# ---------------------------------------------------------------------------
# chunking captions
# ---------------------------------------------------------------------------

def test_caption_extraction():
    from src.chunking import extract_captions, create_parent_chunks
    text = ("RESULTS\n\n" + "Body text. " * 40 + "\n\n"
            "Figure 3: Ephb1 knockdown disrupts boundary formation. Scale bar 50 um.\n\n"
            "Table 2: Adhesion coefficients across conditions.")
    caps = extract_captions(text)
    labels = {(c["section"], c["label"]) for c in caps}
    assert ("figure_caption", "3") in labels
    assert ("table_caption", "2") in labels
    parents = create_parent_chunks(text, "t.md", {"title": "T", "year": "2021"})
    ctypes = {p.get("chunk_type") for p in parents}
    assert "figure_caption" in ctypes and "table_caption" in ctypes and "section" in ctypes
    # caption parents bypass MIN_PARENT_SIZE
    caps_parents = [p for p in parents if p.get("chunk_type", "").endswith("caption")]
    assert all(len(p["content"]) >= 20 for p in caps_parents)
    print(f"  ✓ captions: {len(caps)} extracted, atomic parents created")


def test_caption_rejects_inline_mentions():
    from src.chunking import extract_captions
    text = ("As shown in Figure 3, the cells migrate.\n\n"
            "RESULTS\n\nBody paragraphs continue here without captions. " * 5)
    assert extract_captions(text) == []
    print("  ✓ inline 'Figure 3' mentions are NOT captured as captions")


# ---------------------------------------------------------------------------
# evaluate.citation_faithfulness (reuses guard, offline)
# ---------------------------------------------------------------------------

def test_citation_faithfulness_scoring():
    from src.evaluate import citation_faithfulness
    import src.citation_guard as cg
    pid = "fake_paper.md#results#abc"
    keys = {f"parent::{pid}"}
    ans = ("Ephb1 drives axon repulsion through ephrin-B1 binding.\n\n"
           "---\n**Sources:**\n"
           "- fake_paper.md (Ephb1 study, 2021)\n- made_up_source.md\n")
    orig_meta, orig_text = cg.load_parent_meta_map, cg.load_parent_text
    cg.load_parent_meta_map = lambda ids: {pid: {"title": "Ephb1 study", "source": "fake_paper.md", "section": "results"}}
    cg.load_parent_text = lambda p: ("Ephb1 receptor signaling drives axon repulsion in the "
                                     "developing hindbrain through ephrin-B1 binding.")
    try:
        r = citation_faithfulness(ans, keys)
    finally:
        cg.load_parent_meta_map = orig_meta
        cg.load_parent_text = orig_text
    assert r["n_source_lines"] == 2 and r["n_whitelisted"] == 1 and r["n_dropped"] == 1
    assert r["whitelist_rate"] == 0.5
    print("  ✓ citation_faithfulness: 1/2 grounded, rate=0.5")


# ---------------------------------------------------------------------------
# fixture
# ---------------------------------------------------------------------------

def tmp_parent_store():
    import tempfile
    return tempfile.mkdtemp(prefix="bib_rag_test_")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            if fn.__code__.co_argcount == 0:
                fn()
            else:
                import tempfile
                fn(tempfile.mkdtemp(prefix="bib_rag_fts_"))
        except Exception as e:
            failed += 1
            print(f"  ✗ {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} tests passed")
    sys.exit(1 if failed else 0)