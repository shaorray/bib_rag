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


def test_zotero_match_multi_identifier():
    """P3 paperIdentity: PMID/PMCID exact-match verification via
    verify_zotero_hit_ids; registry conflicts reject regardless of title."""
    from src.zotero_match import verify_zotero_hit_ids, pick_best_hit
    title = "Eph receptor signalling in axon guidance"
    # PMID agrees despite garbage title → accept
    ok, _s, r = verify_zotero_hit_ids(
        title, {"pmid": "PMID: 34526773"},
        {"title": "completely different words", "pmid": "34526773", "doi": ""})
    assert ok and r == "pmid-agrees", (ok, r)
    # PMID conflict → reject even with identical titles
    ok, _s, r = verify_zotero_hit_ids(
        title, {"pmid": "34526773"},
        {"title": title, "pmid": "99999999", "doi": ""})
    assert not ok and r == "pmid-conflict", (ok, r)
    # PMCID via PMC-link form
    ok, _s, r = verify_zotero_hit_ids(
        title, {"pmcid": "PMC3452677"},
        {"title": "whatever", "pmcid": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3452677/"})
    assert ok and r == "pmcid-agrees", (ok, r)
    # DOI still two-tier inside the ids variant
    ok, _s, r = verify_zotero_hit_ids(
        title, {"doi": "10.1016/j.ydbio.2021.01.002"},
        {"title": title, "doi": "10.1016/J.YDBIO.2021.01.002"})
    assert ok and r == "doi-agrees", (ok, r)
    ok, _s, r = verify_zotero_hit_ids(
        title, {"doi": "10.1016/j.ydbio.2021.01.002"},
        {"title": title, "doi": "10.1038/s41586-020-2649-2"})
    assert not ok and r == "doi-conflict", (ok, r)
    # no identifiers at all → falls back to title threshold
    ok, _s, r = verify_zotero_hit_ids(
        title, {}, {"title": title, "doi": "", "pmid": "", "pmcid": ""})
    assert ok and r == "title-match", (ok, r)
    # pick_best_hit honors the ids variant through its hit dicts
    best = pick_best_hit(
        [{"key": "A", "title": "unrelated quantum optics", "doi": ""},
         {"key": "B", "title": "something else entirely", "doi": "",
          "pmid": "34526773"}],
        title, "", query_ids={"pmid": "34526773"})
    assert best and best["key"] == "B", best
    print("  ✓ zotero_match ids: PMID/PMCID exact-match + conflict reject + fallback")


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
# B2: CJK bigram tokenization in FTS
# ---------------------------------------------------------------------------

def test_cjk_bigram_indexing(tmpdir):
    """CJK text must be findable by sub-phrase queries (bigram channel)."""
    os.environ["BIB_RAG_ROOT"] = str(tmpdir)
    os.environ.pop("BIB_RAG_KB_NAME", None)
    import importlib
    import src.kb_config as kc
    importlib.reload(kc)
    os.makedirs(os.path.join(str(tmpdir), "parent_store"), exist_ok=True)
    parents = [{"parent_id": "cjk.md#results#h1", "source": "cjk.md",
                "section": "results",
                "content": "轴突导向因子在 Ephb1 突变体中表达失调，导致神经回路异常。",
                "word_count": 20, "char_count": 60,
                "meta": {"title": "中文测试"}}]
    with open(os.path.join(str(tmpdir), "parent_store", "cjk_md.json"), "w") as f:
        json.dump(parents, f)
    import src.hybrid_search as hs
    importlib.reload(hs)
    idx = hs.HybridIndex()
    n = idx.upsert_source("cjk_md")
    assert n > 0
    # sub-phrase of the indexed sentence — impossible without the CJK channel
    hits = idx.bm25_search("轴突导向", limit=5)
    assert hits, "CJK sub-phrase query returned nothing"
    assert "轴突" in hits[0]["text"] or "轴突导向" in hits[0]["text"]
    # full mixed query (latin + cjk) must not error and must hit
    hits2 = idx.bm25_search("Ephb1 突变体", limit=5)
    assert hits2, "mixed latin+CJK query returned nothing"
    # absent phrase → empty
    hits3 = idx.bm25_search("转录组测序", limit=5)
    assert hits3 == []
    print(f"  ✓ CJK bigram: {n} children, 轴突导向→{len(hits)} hits, "
          f"mixed→{len(hits2)}, absent→0")


def test_cjk_prepare_unit():
    from src.hybrid_search import _cjk_prepare
    assert _cjk_prepare("轴突导向") == "轴突 突导 导向"
    assert _cjk_prepare("中") == "中"
    assert _cjk_prepare("Ephb1 only") == ""
    print("  ✓ _cjk_prepare bigram segmentation")


# ---------------------------------------------------------------------------
# B5: identifier normalization
# ---------------------------------------------------------------------------

def test_identifier_normalization():
    from src.identifiers import (normalize_doi, normalize_arxiv,
                                 normalize_pmid, doi_prefix_agree,
                                 detect_identifier)
    # URL-wrapped, case-insensitive, version-suffixed → same canonical DOI
    assert (normalize_doi("https://doi.org/10.1016/J.YDBIO.2021.01.002")
            == "10.1016/j.ydbio.2021.01.002")
    assert (normalize_doi("doi:10.1016/j.ydbio.2021.01.002v2")
            == "10.1016/j.ydbio.2021.01.002")
    # Oxford-style DOIs legitimately end in "v<digits>" — NOT a version suffix
    assert normalize_doi("10.1093/nar/gkv370") == "10.1093/nar/gkv370"
    assert normalize_doi("10.1093/cvr/cvr154") == "10.1093/cvr/cvr154"
    # digit-preceded version markers still stripped
    assert normalize_doi("10.1016/j.ydbio.2021.01.002.v2") == "10.1016/j.ydbio.2021.01.002"
    assert normalize_doi("not a doi") is None
    assert normalize_arxiv("arXiv:2103.12345v2") == "2103.12345"
    assert normalize_pmid("PMID: 34526773") == "34526773"
    assert detect_identifier("10.1038/s41586-020-2649-2") == (
        "doi", "10.1038/s41586-020-2649-2")
    # P3a: PMCID — canonical PMC<digits>, bare digits stay PMID
    from src.identifiers import normalize_pmcid
    assert normalize_pmcid("PMCID: PMC3452677") == "PMC3452677"
    assert normalize_pmcid("https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3452677/") == "PMC3452677"
    assert normalize_pmcid("pmc 3452677") == "PMC3452677"   # space + case
    assert normalize_pmcid("34526773") is None              # bare digits → PMID
    assert detect_identifier("PMCID: PMC3452677") == ("pmcid", "PMC3452677")
    assert detect_identifier("https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3452677/")[0] == "pmcid"
    assert detect_identifier("34526773") == ("pmid", "34526773")
    print("  ✓ identifier normalization: doi/arxiv/pmid/pmcid canonical forms")


def test_guard_redirect_loop():
    """P3b: collect_answer redirects when ALL Sources lines are hallucinated;
    ships (annotated) once the GUARD_REDIRECT_MAX budget is exhausted."""
    from langchain_core.messages import AIMessage
    from src.agent_nodes import collect_answer

    def run_once(retries, sources_lines, env=None):
        os.environ.setdefault("CITATION_GUARD", "1")
        saved = {k: os.environ.get(k) for k in (env or {})}
        os.environ.update(env or {})
        try:
            body = ("The axon guidance result.\n\n---\n**Sources:**\n"
                    + "\n".join(f"- {l}" for l in sources_lines))
            state = {
                "messages": [AIMessage(content=body, id="m1")],
                "retrieval_keys": set(),          # nothing verifiable
                "question": "q", "question_index": 0,
                "guard_retries": retries,
            }
            return collect_answer(state)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    # first attempt: all lines dropped, budget available → redirect
    out = run_once(0, ["Hallucinated_2020_ghost.md"])
    assert out.get("guard_redirect") is True, out
    assert out["guard_retries"] == 1 and out["final_answer"] == ""
    assert "agent_answers" not in out  # absent = unchanged (LangGraph partial update)
    # feedback injected, failed answer removed
    kinds = [type(m).__name__ for m in out["messages"]]
    assert "HumanMessage" in kinds, kinds
    assert "citation policy" in out["messages"][-1].content
    # retry budget exhausted → ships with note, no redirect
    out2 = run_once(1, ["Hallucinated_2020_ghost.md"])
    assert out2.get("guard_redirect", False) in (False, None) or not out2.get("guard_redirect"), out2
    assert out2["final_answer"] and out2["agent_answers"], out2
    assert "redirected" in out2["guard_note"], out2["guard_note"]
    # clean answer (whitelisted source) → no redirect, ships normally
    state3 = {
        "messages": [AIMessage(
            content="The axon guidance result.\n\n---\n**Sources:**\n- good_paper.md",
            id="m3")],
        "retrieval_keys": {"parent::good_paper.md"},
        "question": "q", "question_index": 0, "guard_retries": 0,
    }
    out3 = collect_answer(state3)
    assert not out3.get("guard_redirect"), out3
    assert out3["final_answer"] and out3["agent_answers"], out3
    print("  ✓ guard redirect: all-hallucinated → retry w/ feedback; budget cap ships")


def test_guard_redirect_e2e_subgraph():
    """P3b end-to-end: fake LLM returns a hallucinated answer then an honest
    one; the compiled subgraph must redirect once, retry, and ship with the
    redirect recorded in guard_note. Catches state-channel persistence bugs
    (guard_redirect stuck True → GraphRecursionError) that node-level tests
    cannot see."""
    from langchain_core.messages import AIMessage
    from src.agentic_graph import create_agent_graph

    answers = [
        AIMessage(content="Claim A.\n\n---\n**Sources:**\n- Ghost_Paper_2020.md", id="a1"),
        AIMessage(content="The evidence does not support a specific claim.", id="a2"),
    ]

    class FakeLLMWithTools:
        def __init__(self, seq):
            self.seq = list(seq)
            self.i = 0

        def bind_tools(self, tools, **k):
            return self

        def invoke(self, *a, **k):
            msg = self.seq[min(self.i, len(self.seq) - 1)]
            self.i += 1
            return msg

    graph = create_agent_graph(FakeLLMWithTools(answers), [])
    sub = graph.nodes["agent"].bound
    state = {
        "messages": [("user", "What drives axon repulsion?")],
        "question": "What drives axon repulsion?", "question_index": 0,
        "retrieval_keys": set(), "context_summary": "",
    }
    result = sub.invoke(state, config={"recursion_limit": 50})
    assert result.get("guard_retries") == 1, result.get("guard_retries")
    assert len(result.get("agent_answers", [])) == 1
    assert "redirected" in result.get("guard_note", ""), result.get("guard_note")
    print("  ✓ guard redirect e2e: subgraph loop redirect→retry→ship, note preserved")


def test_doi_match_journal_mates():
    """Same-journal DOIs must NOT count as the same paper (regression for
    the prefix-length bug), while truncated metadata still matches."""
    from src.zotero_match import doi_match
    assert doi_match("https://doi.org/10.1016/J.YDBIO.2021.01.002",
                     "doi:10.1016/j.ydbio.2021.01.002") is True
    # journal-mates: identical registrant+journal path, different item id
    assert doi_match("10.1016/j.ydbio.2021.01.002",
                     "10.1016/j.ydbio.2019.05.007") is False
    assert doi_match("10.1038/s41586-020-2649-2",
                     "10.1038/s41591-021-01299-x") is False
    # truncated metadata (short prefix of the full DOI) → agree
    assert doi_match("10.1016/j.ydbio.2021.01",
                     "10.1016/j.ydbio.2021.01.002") is True
    assert doi_match("", "10.1016/x") is None
    print("  ✓ doi_match: canonical + journal-mate separation + truncation")


def test_reference_graph_doi_resolution(tmpdir):
    """_resolve_source must find papers by bare/URL-wrapped DOI."""
    import src.reference_graph as rg
    graph = {"papers": {
        "wilkinson_2021.md": {"title": "Eph signalling in development",
                              "year": "2021",
                              "doi": "https://doi.org/10.1242/dev.199555"},
    }, "edges": []}
    assert rg._resolve_source(graph, "10.1242/dev.199555") == "wilkinson_2021.md"
    assert (rg._resolve_source(
        graph, "doi:10.1242/dev.199555v1") == "wilkinson_2021.md")
    assert rg._resolve_source(graph, "10.9999/not-there") is None
    print("  ✓ reference_graph: DOI-based paper resolution")


# ---------------------------------------------------------------------------
# B1: broadened retry signals
# ---------------------------------------------------------------------------

def test_broaden_signals():
    from src.broaden import (retrieval_metrics, should_broaden,
                             plan_broadening, or_split_query)
    # weak: single table-debris chunk with low similarity
    weak = [{"text": "Fig 3a | 45.2 | 12.1 |", "similarity": 0.31}]
    weak_ok, weak_reasons = should_broaden(retrieval_metrics(weak))
    assert weak_ok and len(weak_reasons) >= 2
    # strong: prose chunks with decent similarity — represents a HEALTHY
    # result, so must pass with realistic live-corpus scores. Garbage
    # queries floor at ~0.49-0.53; real queries hit 0.56+ (see broaden.py
    # MIN_SIM calibration note). Use well-above-threshold sims here.
    strong = [{"text": "Ephb1 mutant embryos show failed axon guidance at the "
                       "midline of the developing hindbrain. " * 2,
               "similarity": 0.66},
              {"text": "In contrast, ephrin-B1 Fc clustering rescued the "
                       "repulsive response robustly in explant cultures. " * 2,
               "similarity": 0.64}]
    assert not should_broaden(retrieval_metrics(strong))[0]
    # borderline-real: sparse topic must not trip S3 alone; give it enough
    # chars that only S3 is in play (live sparse queries score ~0.58 best)
    borderline = [{"text": "Guidance of the pronephric duct involves repulsive "
                           "cues from the somitic mesoderm during migration and "
                           "epithelial tubule extension. " * 2,
                   "similarity": 0.58},
                  {"text": "In zebrafish, Wnt11r signaling polarizes the "
                           "pronephric duct cells as they migrate rostrally. " * 2,
                   "similarity": 0.57}]
    assert not should_broaden(retrieval_metrics(borderline))[0]
    # ladder: attempt 0 → widen only; attempt 1 → +or-split; attempt 2 → stop
    p0 = plan_broadening("q", True, 0)
    p1 = plan_broadening("q", True, 1)
    assert p0 and not p0["or_split"] and p1 and p1["or_split"]
    assert plan_broadening("q", True, 2) is None
    # or-split: rarest (longest) terms win; single term → ''
    split = or_split_query("the role of Ephb1 in axon guidance")
    assert "Ephb1" in split and "guidance" in split
    assert or_split_query("Ephb1") == ""
    print("  ✓ broaden: 4-signal detection + ladder + or-split")


def test_broaden_disabled_by_default_env(tmpdir):
    """BROADEN_RETRY=0 must leave search output untouched (no [BROADENED])."""
    os.environ["BROADEN_RETRY"] = "0"
    try:
        # pure formatting check via _format_results
        from src.agent_tools import ToolFactory
        f = ToolFactory.__new__(ToolFactory)  # skip __init__ (no chroma here)
        out = f._format_results(
            [{"parent_id": "a#s#1", "source": "a.md", "section": "results",
              "text": "hello world content", "similarity": 0.5}])
        assert "BROADENED" not in out and "RESULT 1" in out
        out2 = f._format_results(
            [{"parent_id": "a#s#1", "source": "a.md", "section": "r",
              "text": "x", "similarity": 0.2}],
            broadened=True, broaden_reasons=["few-results(1<2)"])
        assert out2.startswith("[BROADENED]") and "few-results" in out2
    finally:
        os.environ.pop("BROADEN_RETRY", None)
    print("  ✓ broaden formatting: [BROADENED] header + kill-switch path")


# ---------------------------------------------------------------------------
# P1a/P1b: answer-side hygiene + strict normalization (paper-qa/CogDoc/seerai)
# ---------------------------------------------------------------------------

def test_normalize_strict_variants():
    """NFKC + dash/space/quote folding (seerai grounding rules)."""
    from src.citation_guard import _normalize_strict
    t = _normalize_strict("Ephb1\u2010ephrin‑B1  \u00a0 binding „quotes” and ﬁ ligature")
    assert "\u2010" not in t and "\u2011" not in t and "\u00a0" not in t
    assert '"' in t          # low/high double quotes → "
    assert "fi" in t         # NFKC unfolds the ﬁ ligature
    t2 = _normalize_strict("it\u2019s a \u2018single\u2019 one")
    assert "it's" in t2 and "'single'" in t2
    print("  ✓ _normalize_strict folds dashes/spaces/quotes/ligatures")


def test_strip_trailing_refs():
    from src.citation_guard import strip_trailing_refs
    assert strip_trailing_refs("Binds ephrin-B1 (see Fig. 2a)") == "Binds ephrin-B1"
    assert strip_trailing_refs("Boundary cells form (Table S3)") == "Boundary cells form"
    assert strip_trailing_refs("Levels rise (Fig. 1b) (Fig. 2a)") == "Levels rise"
    assert strip_trailing_refs("No trailing ref here") == "No trailing ref here"
    # non-trailing refs are untouched
    assert strip_trailing_refs("(Fig. 2a) shows levels") == "(Fig. 2a) shows levels"
    print("  ✓ strip_trailing_refs removes trailing figure/table refs")


def test_scan_malformed_citation_tokens():
    from src.citation_guard import scan_malformed_citation_tokens
    bad = scan_malformed_citation_tokens(
        "uses [e id:5] and [evidence id123] and [E1:P3] mixes")
    assert "[e id:5]" in bad and "[evidence id123]" in bad and "[E1:P3]" in bad, bad
    # known-good shapes must NOT be flagged
    good = scan_malformed_citation_tokens(
        "cites [12], [3,4], [PMID:12345678], [Smith 2020], [Figure 2]")
    assert good == [], good
    print(f"  ✓ malformed tokens detected: {bad}")


def test_strip_inline_hallucinated_citations():
    from src.citation_guard import strip_inline_hallucinated_citations
    known = {"Parkers_2021_ephrin_axon_guidance.md#results#abc",
             "Chen_2019_ephrin.md#intro#def"}
    meta = {}
    body = ("Ephb1 drives repulsion (Parkers et al., 2021) via ephrin-B1 (Chen, 2019) "
            "and also (Fakester et al., 1999) nonsense (Bogus, 2030).")
    cleaned, removed = strip_inline_hallucinated_citations(body, known, meta)
    # whitelisted citations kept
    assert "(Parkers et al., 2021)" in cleaned and "(Chen, 2019)" in cleaned
    # hallucinated ones stripped
    assert "(Fakester et al., 1999)" not in cleaned and "(Bogus, 2030)" not in cleaned
    assert set(removed) == {"(Fakester et al., 1999)", "(Bogus, 2030)"}, removed
    # year-mismatch with a known surname is still hallucinated
    _, removed2 = strip_inline_hallucinated_citations(
        "Claims (Parkers, 1999) here.", {"Parkers_2021_ephrin_axon_guidance.md#results#abc"}, meta)
    assert removed2 == ["(Parkers, 1999)"], removed2
    print(f"  ✓ inline hallucinated citations stripped, whitelisted kept: {removed}")


def test_strip_inline_noop_when_no_parents():
    """Empty whitelist → no-op (never nuke every citation)."""
    from src.citation_guard import strip_inline_hallucinated_citations
    body = "All (Nobody, 2001) citations (Who et al., 2002) stay."
    cleaned, removed = strip_inline_hallucinated_citations(body, set(), {})
    assert cleaned == body and removed == []
    print("  ✓ no whitelisted parents → inline stripping disabled (safe no-op)")


def test_strip_inline_kill_switch():
    from src.citation_guard import strip_inline_hallucinated_citations
    body = "A (Fakester et al., 1999) B."
    os.environ["CITATION_GUARD_ANSWER_SIDE"] = "0"
    try:
        cleaned, removed = strip_inline_hallucinated_citations(body, set(), {"x": {}})
        assert cleaned == body and removed == []
    finally:
        os.environ.pop("CITATION_GUARD_ANSWER_SIDE", None)
    print("  ✓ CITATION_GUARD_ANSWER_SIDE=0 kill-switch works")


def test_enforce_answer_side_hygiene_end_to_end():
    from src.citation_guard import enforce_answer_side_hygiene
    known = {"Parkers_2021_ephrin_axon_guidance.md#results#abc"}
    meta = {"Parkers_2021_ephrin_axon_guidance.md#results#abc":
            {"title": "Ephb1 drives axon repulsion", "source":
             "Parkers_2021_ephrin_axon_guidance.md", "section": "results"}}
    ans = ("Ephb1 drives repulsion (Parkers et al., 2021) and nonsense "
           "(Fakester et al., 1999) here [e id:5].\n\n---\n**Sources:**\n"
           "- Parkers_2021_ephrin_axon_guidance.md\n")
    out, report = enforce_answer_side_hygiene(ans, known, meta)
    assert "(Fakester et al., 1999)" not in out
    assert "(Parkers et al., 2021)" in out          # whitelisted → kept
    assert "**Sources:**" in out and "Parkers_2021_ephrin_axon_guidance.md" in out
    assert len(report["stripped_inline"]) == 1
    assert report["malformed_tokens"] == ["[e id:5]"]
    # double spaces from removals are collapsed
    assert "  " not in out.split("\n\n")[0]
    print("  ✓ enforce_answer_side_hygiene end-to-end: strip + flag + Sources intact")


# ---------------------------------------------------------------------------
# P2a: bibtex_export (offline paths only — no network in tests)
# ---------------------------------------------------------------------------

def test_bibtex_key_generation():
    from src.bibtex_export import make_bibtex_key
    k1 = make_bibtex_key("Lupiáñez", "2015", "Disruptions of the basal lamina")
    assert k1 == "lupianez_2015_disruptions", k1
    k2 = make_bibtex_key("Lupiañez", "2015", "Disruptions of the basal lamina",
                         existing={k1})
    assert k2 == "lupianez_2015_disruptionsa", k2  # collision suffix (seerai rule)
    # gene names with digits stay whole
    k3 = make_bibtex_key("Parkers", "2021", "Ephb1 drives axon repulsion")
    assert k3 == "parkers_2021_ephb1", k3
    print("  ✓ bibtex key: diacritic fold + collision suffix + digit-words")


def test_bibtex_from_meta():
    from src.bibtex_export import bibtex_from_meta, bibtex_fields
    e = bibtex_from_meta({
        "title": "Ephb1 drives axon repulsion", "year": "2021",
        "authors": "Paulson AF; Fang X", "journal": "Dev Cell",
        "doi": "10.1016/j.ydbio.2021.01.002",
        "source": "Parkers_2021_ephrin_axon_guidance.md"})
    assert e and "@article{parkers_2021_ephb1," in e
    f = bibtex_fields(e)
    assert f["author"] == "Paulson, AF and Fang, X"       # PubMed → BibTeX authors
    assert f["journal"] == "Dev Cell" and f["year"] == "2021"
    assert f["doi"] == "10.1016/j.ydbio.2021.01.002"
    # every field except the last ends with a comma (valid BibTeX)
    body_lines = [l for l in e.splitlines() if "=" in l]
    for l in body_lines[:-1]:
        assert l.rstrip().endswith(","), l
    assert not body_lines[-1].rstrip().endswith(",")
    # no title → None
    assert bibtex_from_meta({"year": "2021"}) is None
    print("  ✓ bibtex_from_meta: PubMed authors converted, valid commas, no-title None")


def test_bibtex_fields_brace_balanced():
    from src.bibtex_export import bibtex_fields
    f = bibtex_fields('@a{x, title = {The {Eph} receptor {T}}, year = {2020}}')
    assert f["title"] == "The {Eph} receptor {T}", f   # nested braces preserved
    print("  ✓ bibtex_fields handles nested braces")


def test_bibtex_fill_missing_fields():
    from src.bibtex_export import _fill_missing_fields, bibtex_fields
    bib = "@article{x2020,\n  title = {T},\n  year = {2020}\n}"
    f = bibtex_fields(_fill_missing_fields(bib, {"author": "Last, First", "journal": "J Bio"}))
    assert f.get("author") == "Last, First" and f.get("journal") == "J Bio"
    assert f.get("title") == "T" and f.get("year") == "2020"  # existing untouched
    print("  ✓ missing fields filled from meta, existing preserved")


def test_bibtex_key_rewrite():
    from src.bibtex_export import _rewrite_key
    rw = _rewrite_key("@article{CrossrefOriginal, title = {X}, year = {2020}}", "neal_2021_ets")
    assert "@article{neal_2021_ets," in rw
    print("  ✓ Crossref key rewritten to canonical form")


def test_bibtex_author_lastname():
    from src.bibtex_export import author_lastname
    assert author_lastname(author_lead="Parkers, J.") == "Parkers"
    assert author_lastname(author_lead="Parker Van Der Berg") == "Berg"
    assert author_lastname(meta={"source": "Chen_2019_ephrin.md"}) == "Chen"
    print("  ✓ author_lastname: Crossref style + filename fallback")


def test_bibtex_meta_cleanup(tmpdir):
    """load_paper_meta + bibtex_from_meta must not inherit scrape noise:
    filename-prefix titles, multi-line journal fragments, Zotero-flattened
    author lists."""
    from src.bibtex_export import load_paper_meta, bibtex_from_meta
    store = str(tmpdir)
    with open(os.path.join(store, "Smith_et_al__-_2019_-_A_Great_Paper_md.json"), "w") as f:
        json.dump([{"parent_id": "x#full#y", "source": "Smith_et_al__-_2019_-_A_Great_Paper.md",
                    "section": "full_text", "content": "x", "word_count": 1, "char_count": 1,
                    "meta": {
                        "title": "Smith et al. - 2019 - A Great Paper",
                        "authors": "Smith, John, Doe, Jane, Roe, Richard",
                        "journal": "Research article \n\nNeuroscience \n\nS",
                        "year": "2019", "doi": "10.1234/xyz"}}], f)
    meta = load_paper_meta(store, "Smith_et_al__-_2019_-_A_Great_Paper.md")
    assert meta["title"] == "A Great Paper", meta["title"]
    e = bibtex_from_meta(meta, set())
    assert e is not None
    from src.bibtex_export import bibtex_fields
    f = bibtex_fields(e)
    assert f["author"] == "Smith, John and Doe, Jane and Roe, Richard", f["author"]
    assert "journal" not in f, f        # scrape junk dropped → @misc
    assert e.startswith("@misc{")
    # legit journal passes through
    meta2 = dict(meta, journal="Nature Neuroscience")
    e2 = bibtex_from_meta(meta2, set())
    assert "journal = {Nature Neuroscience}" in e2, e2
    print("  ✓ meta cleanup: filename title stripped, Zotero authors joined, journal noise dropped")


def test_bibtex_load_paper_meta_dual_paths(tmpdir):
    """Accepts both source filename and store filename; PMID-style stores too."""
    from src.bibtex_export import load_paper_meta
    store = str(tmpdir)
    with open(os.path.join(store, "10068468_md.json"), "w") as f:
        json.dump([{"parent_id": "10068468.md#full_text#abc", "source": "10068468.md",
                    "section": "full_text", "content": "x", "word_count": 1,
                    "char_count": 1,
                    "meta": {"title": "A Xenopus paper", "authors": "Paulson AF",
                             "year": "1999", "journal": "", "doi": "10.1006/dbio.1998.9158"}}], f)
    m1 = load_paper_meta(store, "10068468.md")
    m2 = load_paper_meta(store, "10068468_md.json")
    assert m1["doi"] == m2["doi"] == "10.1006/dbio.1998.9158", (m1, m2)
    assert m1["year"] == "1999"
    print("  ✓ load_paper_meta resolves both filename conventions")


def test_bibtex_export_offline_batch(tmpdir):
    from src.bibtex_export import export_answers_bib
    store = str(tmpdir)
    with open(os.path.join(store, "fake_paper_md.json"), "w") as f:
        json.dump([{"parent_id": "fake_paper.md#results#abc", "source": "fake_paper.md",
                    "section": "results", "content": "x", "word_count": 1, "char_count": 10,
                    "meta": {"title": "Ephb1 drives axon repulsion",
                             "authors": "Parkers J; Chen L", "year": "2021",
                             "journal": "Dev Cell", "doi": "10.1016/j.ydbio.2021.01.002"}}], f)
    out = os.path.join(str(tmpdir), "refs.bib")
    res = export_answers_bib(["fake_paper.md"], out, offline=True, store_dir=store)
    assert res["written"] == 1 and res["skipped"] == 0, res
    text = open(out).read()
    assert "@article{fake_2021_ephb1," in text and "author = {Parkers, J and Chen, L}" in text
    # missing paper → skipped, not crash
    res2 = export_answers_bib(["missing_paper.md"], os.path.join(str(tmpdir), "r2.bib"),
                              offline=True, store_dir=store)
    assert res2["written"] == 0 and res2["skipped"] == 1
    print("  ✓ offline batch export: written + skipped paths")


# ---------------------------------------------------------------------------
# P2b: retraction_watch (offline — temp snapshot CSV, never the live one)
# ---------------------------------------------------------------------------

_RETRACTION_CSV = (
    "Record ID,Title,Journal,RetractionDate,RetractionDOI,"
    "OriginalPaperDOI,RetractionNature,Reason\n"
    "1,Real retraction,Cardiovasc Res,3/19/2012,10.1093/cvr/cvs087,"
    "10.1093/cvr/cvr154,Retraction,Unreliable Data\n"
    "2,Arabidopsis ATX1,Plant J,12/15/2015,10.1093/nar/gkv1489,"
    "10.1093/nar/gkm464,Correction,Duplication of/in Image\n"
)

# The false-positive pair that motivated the identifiers.py fix:
# 10.1093/nar/gkv370 (Oxford NAR, in the library) vs gkv1489 (snapshot row) —
# the old `v\d+$` strip collided both to bare "10.1093/nar/gk".


def test_retraction_load_and_is_retracted(tmpdir):
    from src.retraction_watch import load_retractions, is_retracted
    snap = os.path.join(str(tmpdir), "rw.csv")
    with open(snap, "w") as f:
        f.write(_RETRACTION_CSV)
    retr = load_retractions(snapshot=snap, cache_days=-1)
    assert "10.1093/cvr/cvr154" in retr
    assert is_retracted("https://doi.org/10.1093/CVR/CVR154", retr)  # case+prefix
    assert not is_retracted("10.1093/nar/gkv370", retr)   # Oxford DOI intact
    assert not is_retracted("10.1093/nar/gk", retr)       # no bare-prefix hit
    assert is_retracted("10.1093/nar/gkm464", retr)
    print("  ✓ retraction load + is_retracted: case/prefix tolerant, Oxford DOIs intact")


def test_retraction_check_sources(tmpdir):
    from src.retraction_watch import check_sources
    store = str(tmpdir)
    with open(os.path.join(store, "retracted_md.json"), "w") as f:
        json.dump([{"parent_id": "retracted.md#full#x", "source": "retracted.md",
                    "section": "full_text", "content": "x", "word_count": 1,
                    "char_count": 1,
                    "meta": {"title": "Cardiovasc retraction case",
                             "doi": "10.1093/cvr/cvr154"}}], f)
    with open(os.path.join(store, "clean_md.json"), "w") as f:
        json.dump([{"parent_id": "clean.md#full#x", "source": "clean.md",
                    "section": "full_text", "content": "x", "word_count": 1,
                    "char_count": 1,
                    "meta": {"title": "Clean Oxford paper",
                             "doi": "10.1093/nar/gkv370"}}], f)
    # empty snapshot set → no hits
    hits = check_sources(store_dir=store, retractions=set())
    assert hits == {}, hits
    # retracted DOI flagged; clean Oxford DOI untouched by the normalize fix
    hits = check_sources(store_dir=store, retractions={"10.1093/cvr/cvr154"})
    assert set(hits) == {"Cardiovasc retraction case"}, hits
    assert hits["Cardiovasc retraction case"]["doi"] == "10.1093/cvr/cvr154"
    print("  ✓ check_sources: retracted flagged, Oxford DOI untouched")


def test_retraction_doctor_smoke(tmpdir):
    """check_retractions wiring: missing-snapshot INFO, OK path, kill-switch."""
    import importlib.util
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "doctor", os.path.join(repo, "scripts", "doctor.py"))
    assert spec is not None and spec.loader is not None
    doctor = importlib.util.module_from_spec(spec)
    _saved_doctor = sys.modules.get("doctor")
    sys.modules["doctor"] = doctor  # required before exec_module (@dataclass lookup)
    try:
        spec.loader.exec_module(doctor)
        cfg = {"parent_store_dir": str(tmpdir), "data_dir": str(tmpdir)}
        # kill-switch (checked before any filesystem access)
        os.environ["RETRACTION_CHECK"] = "0"
        try:
            r = doctor.check_retractions(cfg)
            assert r[0].status == doctor.INFO and "RETRACTION_CHECK" in r[0].message
        finally:
            os.environ["RETRACTION_CHECK"] = "1"
        # force snapshot_path so the test never depends on the live snapshot
        sys.path.insert(0, os.path.join(repo, "src"))
        import retraction_watch as rw  # bare module — the one doctor's import binds
        orig = rw.snapshot_path
        try:
            rw.snapshot_path = lambda: os.path.join(str(tmpdir), "absent.csv")
            r = doctor.check_retractions(cfg)
            assert r[0].status == doctor.INFO and "--update" in r[0].message, r
            # empty store + snapshot present, nothing matches → OK
            snap = os.path.join(str(tmpdir), "rw.csv")
            with open(snap, "w") as f:
                f.write(_RETRACTION_CSV)
            rw.snapshot_path = lambda: snap
            r = doctor.check_retractions(cfg)
            assert r[0].status == doctor.OK and "0 retracted" in r[0].message, r
        finally:
            rw.snapshot_path = orig
    finally:
        if _saved_doctor is not None:
            sys.modules["doctor"] = _saved_doctor
        else:
            sys.modules.pop("doctor", None)
    print("  ✓ doctor check_retractions: missing-snapshot INFO + OK + kill-switch")


# ---------------------------------------------------------------------------
# Full-功能实测 (2026-08-28) 抓到的四个 bug 的回归测试
# ---------------------------------------------------------------------------

def test_zotero_tokens_split_glued_words():
    """Zotero PDF-title scrapes lose spaces ('PerturbsXenopusGastrulation',
    'p120ctn1A'); token similarity must split camelCase/letter-digit
    boundaries BEFORE lowercasing or the true paper is rejected (real-data
    bug: sim 0.308 < 0.55 → correct hit rejected)."""
    from src.zotero_match import title_similarity
    q = ("Misexpression of the catenin p120(ctn)1A perturbs Xenopus "
         "gastrulation")
    scraped = ("Misexpression of the Catenin p120ctn1A PerturbsXenopus"
               "Gastrulation But Does Not Elicit Wnt-Directed Axis "
               "Specification")
    sim = title_similarity(q, scraped)
    assert sim >= 0.55, f"glued-title similarity still too low: {sim:.3f}"
    # discriminative power must survive: unrelated papers stay far apart
    assert title_similarity("Eph receptor signalling in axon guidance",
                            "Ephrin signalling in angiogenesis") < 0.55
    assert title_similarity("a randomized trial of drug X",
                            "B cell receptor signalling") < 0.1
    print(f"  ✓ title tokens split glued Zotero scrapes (sim={sim:.3f})")


def test_retraction_cache_single_load(tmpdir):
    """The 66MB snapshot used to be re-parsed on EVERY is_retracted() call
    (1.5s × 1270 DOIs ≈ 33min measured). The module memo makes repeat calls
    O(1); explicit `snapshot=` bypasses it."""
    import importlib
    import src.retraction_watch as rw
    snap = os.path.join(str(tmpdir), "rw.csv")
    with open(snap, "w") as f:
        f.write(_RETRACTION_CSV)
    # first call with explicit snapshot bypasses memo and returns a set
    r1 = rw.load_retractions(snapshot=snap, cache_days=-1)
    assert "10.1093/cvr/cvr154" in r1
    # default-path call populates the memo; second call is the SAME object
    orig_path = rw.snapshot_path
    try:
        rw.snapshot_path = lambda: snap
        rw._retraction_cache = None
        a = rw.load_retractions()
        b = rw.load_retractions()
        assert a is b, "memoized set should be returned by identity"
        assert "10.1093/cvr/cvr154" in a
    finally:
        rw.snapshot_path = orig_path
        rw._retraction_cache = None
    print("  ✓ retraction_watch: module memo returns the cached set")


def test_retraction_check_sources_default_loads(tmpdir):
    """check_sources(retractions=None) used to be a silent no-op
    (`retractions is not None and ...` never fired). The default call must
    load the snapshot itself and flag the retracted paper."""
    from src.retraction_watch import check_sources
    import src.retraction_watch as rw
    store = str(tmpdir)
    with open(os.path.join(store, "retracted_md.json"), "w") as f:
        json.dump([{"parent_id": "r.md#full#x", "source": "r.md",
                    "section": "full_text", "content": "x", "word_count": 1,
                    "char_count": 1,
                    "meta": {"title": "Cardiovasc retraction case",
                             "doi": "10.1093/cvr/cvr154"}}], f)
    snap = os.path.join(str(tmpdir), "rw.csv")
    with open(snap, "w") as f:
        f.write(_RETRACTION_CSV)
    orig_path = rw.snapshot_path
    try:
        rw.snapshot_path = lambda: snap
        rw._retraction_cache = None
        hits = check_sources(store_dir=store)  # retractions=None!
        assert set(hits) == {"Cardiovasc retraction case"}, hits
    finally:
        rw.snapshot_path = orig_path
        rw._retraction_cache = None
    print("  ✓ check_sources(None) loads the snapshot itself (no silent no-op)")


def test_search_call_budget_cap():
    """A single search call with 6 results × 800-char excerpts ≈ 5.5KB;
    TWO parallel calls then blew the 4096-token local slot (llama-server
    400 'exceed_context_size_error'). The formatted call output must be
    capped by SEARCH_CALL_MAX_CHARS."""
    from src.agent_tools import _clip
    import src.agent_tools as at
    fused = [{"source": f"s{i}.md", "section": "full_text",
              "parent_id": f"p{i}", "similarity": 0.9,
              "text": "x" * 800, "channels": "vec"}
             for i in range(6)]
    # poke the private method through a stub factory (no Chroma needed)
    ToolFactory = at.ToolFactory
    factory = ToolFactory.__new__(ToolFactory)  # skip __init__ (no DB)
    out = factory._format_results(fused)
    # cap is checked AFTER appending a result → worst case = cap + one result
    assert len(out) <= at.SEARCH_CALL_MAX_CHARS + 900, (
        f"call output {len(out)} exceeds budget {at.SEARCH_CALL_MAX_CHARS}")
    assert "call capped" in out  # transparent drop marker
    print(f"  ✓ search call budget: {len(out)} chars < "
          f"{at.SEARCH_CALL_MAX_CHARS} cap, drop marker present")


def test_local_parallel_tool_call_cap():
    """Qwen issues 2-3 parallel tool calls per round; their combined outputs
    (plus prompts ≈1650 tok) blew the 4096-token local slot. The orchestrator
    must cap to 1 call/round when LLM points at a local slot — and leave
    cloud (11434) rounds untouched."""
    import os
    from src import agent_nodes as an

    class FakeAIMessage:
        def __init__(self, n):
            self.tool_calls = [{"name": "search_child_chunks", "args": {"query": f"q{i}"},
                                "id": f"call_{i}"} for i in range(n)]

    orig_url, orig_model = os.environ.get("LLM_URL"), os.environ.get("LLM_MODEL")
    try:
        os.environ["LLM_URL"] = "http://localhost:5015/v1"
        resp = an._cap_parallel_tool_calls(FakeAIMessage(3))
        assert len(resp.tool_calls) == 1, resp.tool_calls
        # cloud round untouched (fresh instance — the cap mutates in place)
        os.environ["LLM_URL"] = "http://localhost:11434/v1"
        resp2 = an._cap_parallel_tool_calls(FakeAIMessage(3))
        assert len(resp2.tool_calls) == 3, resp2.tool_calls
        # single call never touched
        resp2 = an._cap_parallel_tool_calls(FakeAIMessage(1))
        assert len(resp2.tool_calls) == 1
    finally:
        if orig_url is None:
            os.environ.pop("LLM_URL", None)
        else:
            os.environ["LLM_URL"] = orig_url
        if orig_model is not None:
            os.environ["LLM_MODEL"] = orig_model
    print("  ✓ local slot: parallel tool calls capped to 1/round; cloud untouched")


def test_agent_tools_local_budget_default():
    """RETRIEVE_PARENT_MAX_CHARS must default to 4000 (not 8000) when the
    env points at the local 5015 slot — 2×8000-char parents ≈ 4500 tokens
    alone, over a 4096-token slot."""
    import importlib
    from src import agent_tools as at
    orig_url = os.environ.get("LLM_URL")
    try:
        os.environ["LLM_URL"] = "http://localhost:5015/v1"
        importlib.reload(at)
        assert at.RETRIEVE_PARENT_MAX_CHARS == 4000, at.RETRIEVE_PARENT_MAX_CHARS
        assert at._is_local_llm() is True
    finally:
        if orig_url is None:
            os.environ.pop("LLM_URL", None)
        else:
            os.environ["LLM_URL"] = orig_url
        importlib.reload(at)  # restore cloud-default state for other tests
    print("  ✓ agent_tools: local LLM → tighter parent budget default (4000)")


def test_retrieval_keys_harvest_parent_ids_from_search_results():
    """The evidence ledger must carry parent:: keys even when the agent only
    calls search_child_chunks (never retrieve_parent_chunks). Search
    ToolMessages list 'Parent ID:' lines (agent_tools._format_results), and
    evaluate.citation_faithfulness sees only the keys — without harvesting,
    every Sources line scores whitelist_rate 0.0 despite grounded retrieval
    (the live citation guard is unaffected: it also parses tool messages)."""
    import src.agent_nodes as an_mod
    from src.agent_nodes import should_compress_context
    from langchain_core.messages import AIMessage, ToolMessage

    msgs = [
        AIMessage(content="", tool_calls=[{
            "name": "search_child_chunks",
            "args": {"query": "Ephrin B1 forward signaling"}, "id": "t1"}]),
        ToolMessage(content=(
            "--- RESULT 1 (similarity: 0.611, channels: vec) ---\n"
            "Parent ID: Davy_et_al_2004_Ephrin-B1_md#results#abc123\n"
            "Content: ..."), name="search_child_chunks", tool_call_id="t1"),
    ]
    state = {"messages": list(msgs), "retrieval_keys": set()}
    saved = an_mod.MAX_ITERATIONS
    an_mod.MAX_ITERATIONS = 10  # force the token-threshold branch
    try:
        cmd = should_compress_context(state)
    finally:
        an_mod.MAX_ITERATIONS = saved
    keys = cmd.update["retrieval_keys"]
    assert "parent::Davy_et_al_2004_Ephrin-B1_md#results#abc123" in keys, sorted(keys)
    assert any(k.startswith("search::") for k in keys), keys
    print("  ✓ retrieval ledger: parent:: keys harvested from search ToolMessages")


def test_retrieval_keys_reject_malformed_parent_args():
    """When the model calls retrieve_parent_chunks with garbage args (author
    surnames etc. → tool returns NO_PARENT_DOCUMENT), the ledger must NOT
    record parent::<garbage>: such keys pollute the whitelist AND satisfy the
    harvest gate, so genuine 'Parent ID:' lines from ToolMessages never land.
    Regression: live Qwen run produced parent::Kim/parent::Das/... keys and
    evaluate.citation_faithfulness resolved every Sources line via bare
    substring match (fake whitelist_rate 1.0, empty lexical_scores)."""
    import src.agent_nodes as an_mod
    from src.agent_nodes import should_compress_context
    from langchain_core.messages import AIMessage, ToolMessage

    msgs = [
        AIMessage(content="", tool_calls=[{
            "name": "search_child_chunks",
            "args": {"query": "Eph receptor axon guidance"}, "id": "t1"}]),
        ToolMessage(content=(
            "--- RESULT 1 (similarity: 0.611, channels: vec) ---\n"
            "Parent ID: Davy_et_al_2004_Ephrin-B1_md#results#abc123\n"
            "Content: ..."), name="search_child_chunks", tool_call_id="t1"),
        AIMessage(content="", tool_calls=[{
            "name": "retrieve_parent_chunks",
            "args": {"parent_id": "Kim"}, "id": "t2"}]),  # garbage arg
        ToolMessage(content="NO_PARENT_DOCUMENT: Kim",
                    name="retrieve_parent_chunks", tool_call_id="t2"),
    ]
    state = {"messages": list(msgs), "retrieval_keys": set()}
    saved = an_mod.MAX_ITERATIONS
    an_mod.MAX_ITERATIONS = 10  # force the token-threshold branch
    try:
        cmd = should_compress_context(state)
    finally:
        an_mod.MAX_ITERATIONS = saved
    keys = cmd.update["retrieval_keys"]
    assert not any(k == "parent::Kim" for k in keys), sorted(keys)
    assert "parent::Davy_et_al_2004_Ephrin-B1_md#results#abc123" in keys, sorted(keys)
    print("  ✓ retrieval ledger: malformed parent args rejected, harvest gate not fooled")




# ---------------------------------------------------------------------------
# fixture
# ---------------------------------------------------------------------------

import pytest  # test dep; the __main__ runner below never needs it


@pytest.fixture
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