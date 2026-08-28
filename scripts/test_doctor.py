#!/usr/bin/env python3
"""Doctor unit tests — fully offline, tmp library, no real data touched."""
import json
import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

import doctor  # noqa: E402


def _make_tiny_library(tmp):
    """parent_store with 2 sources + chroma sqlite stub + fts + refgraph.
    Naming mirrors production: chroma/fts store RAW source 'paper_a.md',
    parent_store filenames are sanitized 'paper_a_md.json'."""
    ps = os.path.join(tmp, "parent_store")
    os.makedirs(ps, exist_ok=True)
    for name in ("paper_a", "paper_b"):
        with open(os.path.join(ps, name + "_md.json"), "w") as f:
            json.dump([{"parent_id": f"{name}.md#results#h",
                        "source": f"{name}.md", "section": "results",
                        "content": "Ephb1 signaling content " * 20,
                        "word_count": 40, "meta": {"title": name,
                                                   "year": "2021",
                                                   "doi": ""}}], f)
    data = os.path.join(tmp, "data")
    os.makedirs(data, exist_ok=True)
    os.makedirs(os.path.join(tmp, "chroma_db_new"), exist_ok=True)
    # minimal chroma stub with matching metadata schema
    chroma = sqlite3.connect(os.path.join(tmp, "chroma_db_new",
                                          "chroma.sqlite3"))
    chroma.execute("CREATE TABLE embeddings (id INTEGER PRIMARY KEY)")
    chroma.execute("""CREATE TABLE embedding_metadata (
        id INTEGER, key TEXT, string_value TEXT,
        PRIMARY KEY (id, key))""")
    for i, name in enumerate(("paper_a", "paper_b")):
        chroma.execute("INSERT INTO embeddings VALUES (?)", (i + 1,))
        chroma.execute("INSERT INTO embedding_metadata VALUES (?,?,?)",
                       (i + 1, "source", f"{name}.md"))
        chroma.execute("INSERT INTO embedding_metadata VALUES (?,?,?)",
                       (i + 1, "parent_id", f"{name}.md#results#h"))
    chroma.commit()
    chroma.close()
    # fts index
    fts = sqlite3.connect(os.path.join(data, "fts_index.db"))
    fts.execute("""CREATE VIRTUAL TABLE child_fts USING fts5(
        text, cjk_text, parent_id UNINDEXED, source UNINDEXED,
        section UNINDEXED, chunk_idx UNINDEXED)""")
    fts.execute("INSERT INTO child_fts VALUES ('ephb text', '', 'p', 'paper_a.md', 'results', 0)")
    fts.commit()
    fts.close()
    # reference graph with a dup DOI + a junk DOI
    g = {"papers": {
        "paper_a": {"title": "paper a", "year": "2021", "doi": "10.1000/aaa"},
        "paper_b": {"title": "paper b", "year": "2021", "doi": "10.1016/"},
    }, "edges": [{"from": "paper_a", "to_author": "X", "to_year": "2020"}]}
    with open(os.path.join(data, "reference_graph.json"), "w") as f:
        json.dump(g, f)
    return ps, data


def test_doctor_offline_all_checks():
    tmp = tempfile.mkdtemp(prefix="bib_rag_doctor_")
    os.environ["BIB_RAG_ROOT"] = tmp
    os.environ.pop("BIB_RAG_KB_NAME", None)
    import importlib
    import kb_config
    importlib.reload(kb_config)
    importlib.reload(doctor)
    _make_tiny_library(tmp)
    cfg = doctor.get_config()
    cfg["parent_store_disabled_dir"] = os.path.join(tmp, "parent_store_disabled")
    cfg["chroma_sqlite"] = os.path.join(tmp, "chroma_db_new", "chroma.sqlite3")
    # fixture sanity: chroma stores RAW source names ("paper_a.md") while
    # parent_store files are sanitized ("paper_a") — exactly like production
    results = []
    for fn in (lambda: doctor.check_built_indexes(cfg),
               lambda: doctor.check_parent_store(cfg),
               lambda: doctor.check_index_drift(cfg),
               lambda: doctor.check_reference_graph(cfg),
               lambda: doctor.check_doi_quality(cfg),
               lambda: doctor.check_disk(cfg)):
        results.extend(fn())
    names = {r.name for r in results}
    assert "parent_store_integrity" in names
    assert "fts_coverage" in names
    assert "reference_graph" in names
    assert "doi_quality" in names
    # no FAILs on the healthy fixture
    fails = [r for r in results if r.status == doctor.FAIL]
    assert not fails, [(r.name, r.message) for r in fails]
    # doi_quality should flag the junk 10.1016/ DOI
    doi = next(r for r in results if r.name == "doi_quality")
    assert "junk" in doi.message
    report = doctor.format_report(results, strict=False)
    assert "summary" in report
    print(f"  ✓ doctor offline checks: {len(results)} results, 0 fail")


def test_doctor_detects_corruption():
    tmp = tempfile.mkdtemp(prefix="bib_rag_doctor2_")
    os.environ["BIB_RAG_ROOT"] = tmp
    os.environ.pop("BIB_RAG_KB_NAME", None)
    import importlib
    import kb_config
    importlib.reload(kb_config)
    importlib.reload(doctor)
    ps, data = _make_tiny_library(tmp)
    # corrupt a parent_store file
    with open(os.path.join(ps, "paper_a.json"), "w") as f:
        f.write("{not json")
    cfg = doctor.get_config()
    res = doctor.check_parent_store(cfg)
    assert res[0].status == doctor.FAIL, res
    print("  ✓ doctor detects corrupt parent_store → FAIL")


def test_checkresult_demotion():
    r = doctor.CheckResult("doi_quality", doctor.WARN, "msg", "fix")
    assert r.demote_if_noise(strict=False).status == doctor.INFO
    assert r.demote_if_noise(strict=True).status == doctor.WARN
    r2 = doctor.CheckResult("chroma_orphans", doctor.WARN, "x")
    assert r2.demote_if_noise(strict=False).status == doctor.WARN
    print("  ✓ CheckResult demotion: doi_quality→INFO by default, strict shows")


if __name__ == "__main__":
    test_doctor_offline_all_checks()
    test_doctor_detects_corruption()
    test_checkresult_demotion()
    print("\n3/3 doctor tests passed")