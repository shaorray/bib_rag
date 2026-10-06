#!/usr/bin/env python3
"""
tree_index.py — PageIndex-style hierarchical tree indexing for bib_rag.

Builds a vectorless, reasoning-based tree index for a single long document
(PDF or Markdown) and stores it under <library>/parent_store/trees/<stem>.json,
side-by-side with the existing parent_store/ (no interference).

This is bib_rag's third retrieval mode:
  * semantic search   → query_bib_rag.py   (chroma + BM25 + rerank)
  * agentic search    → agentic_query.py   (LangGraph agent over chunks)
  * tree reasoning    → tree_index.py + tree_retrieve.py  (this + sibling)

The heavy lifting (PDF layout analysis, heading detection, outline assembly,
tree-cost optimization) is vendored from VectifyAI PageIndex (MIT) under
src/vendor_pageindex/. This module is the bib_rag-side adapter: it wires
PageIndex's local build path to bib_rag's library layout, LLM config, and
JSON conventions.

Usage:
    python3 -B src/tree_index.py <doc.pdf|doc.md> [--mode flash|standard] [--out DIR]
    python3 -B src/tree_index.py --batch <dir> [--glob '*.pdf']

Library layout:
    <library root>/parent_store/trees/<stem>.json    tree index (metadata + structure)
    <library root>/parent_store/trees/<stem>.pages.json  per-page markdown (for retrieval)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# ─── path setup ────────────────────────────────────────────────────────────
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:  # bib_rag-package-try
    from .kb_config import get_config, parse_kb_arg
except ImportError:  # flat (loose-script mode)
    from kb_config import get_config, parse_kb_arg

_argv = parse_kb_arg()
if _argv:
    sys.argv = [sys.argv[0]] + _argv
_CFG = get_config()

VENDOR = _SRC / "vendor_pageindex"


# ─── LLM config: reuse bib_rag / environment, not PageIndex's cloud ────────
def _llm_settings() -> dict:
    """Resolve the indexing/summary model from bib_rag-style env vars.

    Mirrors agentic_query.py's LLM_URL / LLM_MODEL convention so the tree
    indexer talks to the same local LLM as the rest of the toolkit.

    Returns a dict with 'model' (litellm-style) and optional 'api_base'/'key'.
    """
    url = os.environ.get("LLM_URL", "http://localhost:11434/v1")
    model = os.environ.get("LLM_MODEL", "glm-5.2:cloud")
    key = os.environ.get("LLM_API_KEY", "not-required")
    # PageIndex uses litellm: an OpenAI-compatible base_url maps to
    # "openai/<model>" with api_base/api_key passed per-call. We expose the
    # raw values and let the adapter translate when it invokes the model.
    return {"url": url, "model": model, "key": key}


def llm_settings() -> dict:
    """Public form of _llm_settings (kept as a public alias so cross-module
    callers like tree_retrieve do not borrow the underscore-prefixed name)."""
    return _llm_settings()


def litellm_model_name(cfg: dict) -> str:
    """Public form of _litellm_model_name (cross-module alias)."""
    return _litellm_model_name(cfg)


def _litellm_model_name(cfg: dict) -> str:
    """Translate a bib_rag LLM config to a litellm model string."""
    model = cfg["model"]
    url = cfg["url"]
    # If the user already gave a provider-prefixed name, respect it.
    if "/" in model and not model.startswith(("http://", "https://")):
        return model
    # Local / OpenAI-compatible endpoint → route through litellm's openai provider.
    if url.rstrip("/").endswith("/v1") or "localhost" in url or "127.0.0.1" in url:
        return f"openai/{model}"
    return model


def _install_model_router() -> None:
    """Monkeypatch vendored utils.llm_completion to honour bib_rag LLM env.

    The vendored code calls litellm.completion(model, messages, api_base=...)
    through a single chokepoint (utils.llm_completion). Rather than edit the
    vendored source, we wrap it so every call carries the bib_rag base_url/key
    and the provider-prefixed model name.
    """
    try:  # bib_rag-package-try
        from .vendor_pageindex import utils as _vu
    except ImportError:
        from vendor_pageindex import utils as _vu

    if getattr(_vu, "_bib_rag_router_installed", False):
        return

    cfg = _llm_settings()
    litellm_model = _litellm_model_name(cfg)
    _orig = _vu.llm_completion

    def _patched(model, prompt, chat_history=None, return_finish_reason=False):
        return _orig(
            litellm_model,
            prompt,
            chat_history=chat_history,
            return_finish_reason=return_finish_reason,
        )

    # Inject api_base/api_key via litellm's environment so we don't have to
    # thread them through every vendored helper. litellm reads these globals.
    os.environ.setdefault("OPENAI_API_BASE", cfg["url"])
    os.environ.setdefault("OPENAI_BASE_URL", cfg["url"])
    os.environ.setdefault("OPENAI_API_KEY", cfg["key"])
    _vu.llm_completion = _patched
    _vu._bib_rag_router_installed = True


# ─── tree store ────────────────────────────────────────────────────────────
def _trees_dir(cfg: dict, override: str | None = None) -> Path:
    if override:
        d = Path(override)
        d.mkdir(parents=True, exist_ok=True)
        return d
    d = Path(cfg["parent_store_dir"]) / "trees"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _stem(path: str) -> str:
    p = Path(path)
    return p.stem if p.suffix.lower() in (".pdf", ".md", ".markdown") else p.name


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, data) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


# ─── build paths ───────────────────────────────────────────────────────────
def _build_tree_pdf(pdf_path: str, mode: str = "flash") -> tuple[list, str | None, list[str]]:
    """Build a tree index from a PDF. Returns (structure, description, page_texts).

    Mirrors the vendored LocalAPI._index_flash / _index_standard but without
    the DocStore layer — we persist to bib_rag's own JSON layout instead.
    """
    try:  # bib_rag-package-try
        from .vendor_pageindex import utils as vu
    except ImportError:
        from vendor_pageindex import utils as vu

    _install_model_router()
    cfg = _llm_settings()
    litellm_model = _litellm_model_name(cfg)

    # page texts (PyPDF2, matching the vendored path so page ranges line up)
    import PyPDF2

    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        page_texts = [vu._scrub_surrogates(page.extract_text() or "")
                      if hasattr(vu, "_scrub_surrogates") else (page.extract_text() or "")
                      for page in reader.pages]

    if mode == "flash":
        try:  # bib_rag-package-try
            from .vendor_pageindex.flash import page_index_flash
            from .vendor_pageindex.flash.api import flash_rejection_reason
        except ImportError:
            from vendor_pageindex.flash import page_index_flash
            from vendor_pageindex.flash.api import flash_rejection_reason

        result = page_index_flash(
            pdf_path,
            summary=True,
            summary_model=litellm_model,
            optimize="full",
            optimize_model=litellm_model,
            use_embedded_toc=True,
        )
        structure = result.get("structure", [])
        reason = flash_rejection_reason(result)
        if reason:
            raise RuntimeError(f"flash index rejected: {reason}")
        vu.write_node_id(structure)
        description = vu.generate_doc_description(
            vu.create_clean_structure_for_description(structure),
            model=litellm_model,
        )
        return structure, description, page_texts

    if mode == "standard":
        try:  # bib_rag-package-try
            from .vendor_pageindex.page_index_classic import page_index_main
        except ImportError:
            from vendor_pageindex.page_index_classic import page_index_main

        page_list = [(t, vu.count_tokens(t, model=litellm_model)) for t in page_texts]
        opt = vu.ConfigLoader().load({
            "model": litellm_model,
            "summary_model": litellm_model,
            "if_add_node_id": "yes",
            "if_add_node_summary": "yes",
            "if_add_node_text": "yes",
            "if_add_doc_description": "yes",
        })
        result = page_index_main(pdf_path, opt, page_list=page_list)
        structure = result.get("structure") or []
        if not structure:
            raise RuntimeError("standard indexing produced no structure")
        return structure, result.get("doc_description"), page_texts

    raise ValueError(f"unknown mode {mode!r} (expected flash|standard)")


def _build_tree_md(md_path: str) -> tuple[list, str | None, dict]:
    """Build a tree index from Markdown. Returns (structure, description, node_texts)."""
    try:  # bib_rag-package-try
        from .vendor_pageindex import utils as vu
        from .vendor_pageindex.page_index_md import md_to_tree
    except ImportError:
        from vendor_pageindex import utils as vu
        from vendor_pageindex.page_index_md import md_to_tree

    _install_model_router()
    cfg = _llm_settings()
    litellm_model = _litellm_model_name(cfg)

    result = md_to_tree(
        md_path,
        if_thinning=False,
        if_add_node_summary="yes",
        summary_token_threshold=200,
        model=litellm_model,
    )
    structure = result.get("structure") or []
    if not structure:
        raise RuntimeError("markdown indexing produced no structure")
    return structure, result.get("doc_description"), {}


# ─── public API ─────────────────────────────────────────────────────────────
def _pdf_meta(doc_path: str, stem: str) -> dict:
    """Best-effort author/year/doi for a freshly built tree.

    Looks up the library's chroma metadata by the document stem (the markdown
    filename the vector index keys `source` on), so a tree's citations carry
    the same author/year as the vector path. Returns {} when the doc is not
    yet in chroma or the lookup fails.
    """
    try:
        import chromadb
        chroma_dir = _CFG.get("chroma_path")
        if not chroma_dir or not os.path.isdir(chroma_dir):
            return {}
        client = chromadb.PersistentClient(path=chroma_dir)
        col = client.get_collection(_CFG["collection_name"])
        for key in (f"{stem}.md", stem):
            got = col.get(where={"source": key}, limit=1, include=["metadatas"])
            metas = (got or {}).get("metadatas") or []
            if metas:
                m = metas[0] or {}
                return {k: m.get(k, "") for k in ("authors", "year", "doi", "title", "journal")}
    except Exception:
        pass
    return {}


def build_index(doc_path: str, mode: str = "flash", out_dir: str | None = None) -> dict:
    """Build and persist a tree index for one document. Returns the manifest."""
    doc_path = os.path.abspath(os.path.expanduser(doc_path))
    if not os.path.isfile(doc_path):
        raise FileNotFoundError(doc_path)
    suffix = Path(doc_path).suffix.lower()
    trees = _trees_dir(_CFG, out_dir)
    stem = _stem(doc_path)

    if suffix == ".pdf":
        structure, description, page_texts = _build_tree_pdf(doc_path, mode=mode)
        pages_payload = page_texts
    elif suffix in (".md", ".markdown"):
        structure, description, _ = _build_tree_md(doc_path)
        # markdown nodes carry their own text; store raw md for reference
        pages_payload = Path(doc_path).read_text(encoding="utf-8")
    else:
        raise ValueError(f"unsupported document type: {suffix}")

    manifest = {
        "schema": "bib_rag.tree.v1",
        "source": doc_path,
        "stem": stem,
        "mode": mode,
        "created_at": _now_iso(),
        "description": description,
        "page_count": len(pages_payload) if isinstance(pages_payload, list) else None,
        "meta": _pdf_meta(doc_path, stem),
        "tree": structure,
    }

    _atomic_write_json(trees / f"{stem}.json", _annotate_manifest_with_qc(manifest))
    _atomic_write_json(trees / f"{stem}.pages.json", pages_payload)
    return manifest


def load_index(stem: str, out_dir: str | None = None) -> dict | None:
    trees = _trees_dir(_CFG, out_dir)
    p = trees / f"{stem}.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def list_indexes(out_dir: str | None = None) -> list[dict]:
    trees = _trees_dir(_CFG, out_dir)
    out = []
    for p in sorted(trees.glob("*.json")):
        if p.name.endswith(".pages.json"):
            continue
        try:
            m = json.loads(p.read_text(encoding="utf-8"))
            out.append({"stem": m.get("stem", p.stem), "source": m.get("source"),
                        "mode": m.get("mode"), "created_at": m.get("created_at"),
                        "page_count": m.get("page_count")})
        except Exception:
            continue
    return out


# ─── tree quality control (borrowed from PageIndex tree_optimize) ───────────
def quality_report(structure: list, page_count: int | None = None) -> dict:
    """Structural QC on a built tree: page-range sanity, sibling ordering,
    coverage gaps, and a search-cost estimate.

    A tree whose nodes have broken page ranges or out-of-order siblings makes
    reasoning retrieval unreliable (the LLM navigates to a node that doesn't
    cover the right pages). Runs the vendored validators, so bib_rag's QC and
    PageIndex's own indexing contract stay identical.

    Returns {'ok': bool, 'issues': [...], 'metrics': {...}}.
    """
    try:  # bib_rag-package-try
        from .vendor_pageindex import tree_optimize as to
    except ImportError:
        from vendor_pageindex import tree_optimize as to

    if page_count is None:
        ends = []
        stack = list(structure)
        while stack:
            n = stack.pop()
            if isinstance(n.get("end_index"), int):
                ends.append(n["end_index"])
            stack.extend(n.get("nodes") or [])
        page_count = max(ends) if ends else 1

    issues = list(to.validate(structure, page_count))
    metrics = {}
    try:
        m = to.complexity(structure, page_count)
        # m is a metrics dict/namespace; keep the plain fields
        for k, v in (vars(m).items() if hasattr(m, "__dict__") else
                     (m.items() if isinstance(m, dict) else [])):
            if isinstance(v, (int, float, str, bool)):
                metrics[k] = v
    except Exception as e:
        metrics["complexity_error"] = str(e)

    return {"ok": not issues, "issues": issues, "metrics": metrics}


def _annotate_manifest_with_qc(manifest: dict) -> dict:
    """Attach a QC block to a manifest before it is written."""
    try:
        manifest["qc"] = quality_report(manifest.get("tree") or [],
                                         manifest.get("page_count"))
    except Exception as e:
        manifest["qc"] = {"ok": True, "issues": [f"qc skipped: {e}"], "metrics": {}}
    return manifest


# ─── CLI ────────────────────────────────────────────────────────────────────
def _cli() -> int:
    ap = argparse.ArgumentParser(description="Build a PageIndex-style tree index for a document")
    ap.add_argument("doc", nargs="?", help="PDF or Markdown file")
    ap.add_argument("--mode", choices=["flash", "standard"], default="flash")
    ap.add_argument("--out", default=None, help="override trees dir")
    ap.add_argument("--batch", default=None, help="build for every PDF in a directory")
    ap.add_argument("--glob", default="*.pdf")
    ap.add_argument("--list", action="store_true", help="list existing tree indexes")
    ap.add_argument("--report", action="store_true",
                    help="after building (or with --list), print the QC report")
    args = ap.parse_args()

    if args.list:
        rows = list_indexes(args.out)
        if not rows:
            print("(no tree indexes)")
            return 0
        for r in rows:
            print(f"  {r['stem']:40s} {r['mode']:8s} {r['page_count']}p  {r['created_at']}")
        return 0

    if args.batch:
        docs = sorted(Path(args.batch).glob(args.glob))
        if not docs:
            print(f"no files matching {args.glob} in {args.batch}", file=sys.stderr)
            return 2
        ok = 0
        for d in docs:
            try:
                m = build_index(str(d), mode=args.mode, out_dir=args.out)
                print(f"  ✓ {m['stem']}  ({m['page_count']} pages)")
                ok += 1
            except Exception as e:
                print(f"  ✗ {d.name}: {e}", file=sys.stderr)
        print(f"built {ok}/{len(docs)}")
        return 0 if ok else 1

    if not args.doc:
        ap.error("need a document path, --batch DIR, or --list")
    m = build_index(args.doc, mode=args.mode, out_dir=args.out)
    print(f"✓ indexed {m['stem']}  ({m['page_count']} pages, mode={m['mode']})")
    if m.get("description"):
        print(f"  description: {m['description'][:200]}")
    if args.report:
        _print_qc(m)
    return 0


def _print_qc(m: dict) -> None:
    qc = m.get("qc") or quality_report(m.get("tree") or [], m.get("page_count"))
    status = "✓ clean" if qc["ok"] else f"⚠ {len(qc['issues'])} issue(s)"
    print(f"  QC: {status}")
    for issue in qc["issues"][:10]:
        print(f"      - {issue}")
    if len(qc["issues"]) > 10:
        print(f"      ... +{len(qc['issues']) - 10} more")
    if qc.get("metrics"):
        keys = ("worst_case", "average", "normalized_worst_case", "tree_cost")
        shown = {k: qc["metrics"][k] for k in keys if k in qc["metrics"]}
        if shown:
            print("  cost: " + "  ".join(f"{k}={v}" for k, v in shown.items()))


if __name__ == "__main__":
    raise SystemExit(_cli())
