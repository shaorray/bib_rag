#!/usr/bin/env python3
"""
backfill_all.py — Orchestrated metadata backfill for the active RAG library.

Runs the fixation layers in fallback order, stopping at the first layer that
succeeds for each paper:

  0. BibTeX snapshot   — if a My Library.bib is provided/found, run
                         bib_to_parent_store.py (triple-match DOI backfill).
  1. Live Zotero       — if zotero_access.available() (MCP or local HTTP),
                         match remaining papers via zotero_search/zotero_item.
  2. Remote registries — Crossref / PubMed / OpenAlex (via meta_audit's
                         clients) for papers still missing fields.
  3. Final proof-read  — meta_audit.py round: verify DOIs resolve, correct
                         year/journal/authors, confidence-gated writes.

Papers are tracked in a per-library status ledger
(`data/backfill_status.csv`: source, title, year, doi, authors, status per
layer) so re-runs skip already-fixed papers and you can audit what each
layer contributed.

Usage:
    <name>-rag scripts/metadata/backfill_all.py --dry-run          # plan + probe
    <name>-rag scripts/metadata/backfill_all.py --apply            # full run
    <name>-rag scripts/metadata/backfill_all.py --bib "/path/My Library.bib" --apply
    <name>-rag scripts/metadata/backfill_all.py --limit 20 --apply # small trial
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIR.parent / "src"))
sys.path.insert(0, str(SCRIPTS_DIR))

from kb_config import get_config  # noqa: E402
from library_config import get_setting  # noqa: E402

_CFG = get_config()
LEDGER_FIELDS = ["source", "layer", "title", "year", "doi", "authors",
                 "status", "detail"]


# ── per-paper helpers ───────────────────────────────────────────────────────

def papers_needing_fix(limit=0):
    """Parent-store files whose meta is missing doi or title, i.e. candidates."""
    pm_dir = Path(_CFG["parent_store_dir"])
    out = []
    for f in sorted(pm_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        meta = {}
        for chunk in data:
            m = chunk.get("meta", {}) if isinstance(chunk, dict) else {}
            if m:
                meta = m
                break
        title = (meta.get("title") or "").strip()
        doi = (meta.get("doi") or "").strip()
        # heuristic noise signature: title looks like a journal header
        noisy = bool(title) and any(
            c.isdigit() for c in title[:12]) and ("(" in title[:40])
        if not doi or not title or noisy:
            out.append({"file": f.name, "meta": meta, "title": title,
                        "doi": doi, "noisy": noisy})
    if limit:
        out = out[:limit]
    return out


def norm_title(t):
    import re
    return re.sub(r"[^a-z0-9]+", "", (t or "").lower())[:80]


# ── layer 0: BibTeX snapshot ───────────────────────────────────────────────

def layer_bib(bib_path, candidates, dry_run):
    """Reuse bib_to_parent_store's matcher for per-paper resolution."""
    from bib_utils import parse_bib_entries
    from bib_to_parent_store import match_paper_to_entry, match_paper_by_title
    entries = parse_bib_entries(Path(bib_path))
    print(f"[layer 0] BibTeX snapshot: {len(entries)} entries from {bib_path}")
    import re
    results = []
    for c in candidates:
        title = c["title"]
        year = str(c["meta"].get("year", "") or "")
        # derive lastname from filename stem ("Author_et_al_-_..." / "Boström_...")
        stem = c["file"].replace(".md", "")
        first = re.split(r"[_\-]", stem)[0]
        lastname = first if first and not first.isdigit() else ""
        entry, status = None, "no_match"
        if title and year:
            entry, status = match_paper_by_title(title, year, entries,
                                                 paper_abstract_norm="")
        if entry is None and lastname:
            entry, status = match_paper_to_entry(
                (lastname, year, (title or "").lower()),
                entries, paper_abstract_norm="")
        if entry and entry.get("doi"):
            results.append({"source": c["file"], "layer": "bib",
                            "title": entry.get("title", title),
                            "year": entry.get("year", year),
                            "doi": entry.get("doi", ""),
                            "authors": entry.get("author", ""),
                            "status": "matched", "detail": f"bib:{status}"})
        else:
            results.append({"source": c["file"], "layer": "bib",
                            "title": title, "year": year, "doi": "",
                            "authors": "", "status": "unmatched",
                            "detail": "no bib entry passed matching"})
    return results


# ── layer 1: live Zotero ────────────────────────────────────────────────────

def layer_zotero(candidates, dry_run):
    import zotero_access
    if not zotero_access.available():
        print("[layer 1] Zotero unreachable (no MCP, no HTTP) — skipping")
        return []
    print("[layer 1] Zotero reachable — searching remaining papers")
    results = []
    for c in candidates:
        title = c["title"]
        # noisy titles (journal headers) produce garbage zotero searches — skip them
        if c.get("noisy"):
            results.append({"source": c["file"], "layer": "zotero",
                            "title": title, "year": "", "doi": "",
                            "authors": "", "status": "unmatched",
                            "detail": "skipped: noisy title (journal header)"})
            continue
        items = zotero_access.zotero_search(title, limit=1) if title else []
        if not items:
            results.append({"source": c["file"], "layer": "zotero",
                            "title": title, "year": "", "doi": "",
                            "authors": "", "status": "unmatched",
                            "detail": "no zotero hit"})
            continue
        it = items[0]
        # fetch full item for doi/authors if the search result is thin
        if not it.get("doi") and it.get("key"):
            full = zotero_access.zotero_item(it["key"]) or {}
            it.update({k: v for k, v in full.items() if v and not it.get(k)})
        results.append({"source": c["file"], "layer": "zotero",
                        "title": it.get("title", title),
                        "year": it.get("year", ""),
                        "doi": it.get("doi", ""),
                        "authors": it.get("authors", ""),
                        "status": "matched" if it.get("doi") or it.get("title") else "unmatched",
                        "detail": "zotero item"})
    return results


# ── layer 2: remote registries (via meta_audit clients) ────────────────────

def layer_remote(candidates, dry_run):
    from meta_audit import (CrossrefClient, PubmedClient, OpenAlexClient,
                            DEFAULT_CROSSREF_MAILTO)
    cr = CrossrefClient(mailto=DEFAULT_CROSSREF_MAILTO)
    pm = PubmedClient()
    oa = OpenAlexClient()
    results = []
    for c in candidates:
        title, doi = c["title"], c["doi"]
        # noisy titles (journal headers) pollute registry search — use a cleaned form
        clean_title = title
        if c.get("noisy"):
            import re as _re
            t = _re.sub(r"[*_]+", "", title)
            t = _re.sub(r"^[A-Z][A-Za-z\s&]+\s+\d+", "", t)  # leading journal+volume
            clean_title = t.strip() or title
        rec, via = None, ""
        if doi and not doi.startswith("unknown"):
            rec = cr.verify_doi(doi)      # single dict or None
            if rec:
                via = "crossref-doi"
        if rec is None and title:
            hits = cr.search(clean_title)  # list of dicts
            if hits:
                rec, via = hits[0], "crossref-title"
        if rec is None and title:
            hits = oa.search(clean_title)  # list of dicts
            if hits:
                rec, via = hits[0], "openalex-title"
        if rec is None and title:
            pmids = pm.search(clean_title) # list of PMIDs
            if pmids:
                via = "pubmed-title"
                rec = {"title": title, "doi": "", "year": "", "journal": "",
                       "authors": "", "pmid": pmids[0]}
        if rec:
            results.append({"source": c["file"], "layer": f"remote:{via}",
                            "title": rec.get("title", title),
                            "year": str(rec.get("year", "")),
                            "doi": rec.get("doi", doi),
                            "authors": rec.get("authors", ""),
                            "status": "matched", "detail": via})
        else:
            results.append({"source": c["file"], "layer": "remote",
                            "title": title, "year": "", "doi": doi,
                            "authors": "", "status": "unmatched",
                            "detail": "no registry record"})
    return results


# ── layer 3: meta_audit proof-read round ────────────────────────────────────

def layer_audit(dry_run, bib, limit):
    """Run meta_audit.py as the final proof-reading round (its own confidence
    gating + backups apply)."""
    cmd = [sys.executable, "-B", str(SCRIPTS_DIR / "metadata" / "meta_audit.py"),
           "--bib", str(bib)]
    if dry_run:
        print("[layer 3] meta_audit has its own dry-run/apply semantics — run with "
              "--apply at the end for the final proof-read round, or invoke it now:")
        print("          " + " ".join(cmd))
        return 0
    print("[layer 3] running meta_audit.py proof-read round...")
    return subprocess.call(cmd)


# ── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Orchestrated metadata backfill")
    ap.add_argument("--bib", default=None,
                    help="My Library.bib path (layer 0). Default: library config / BIB_RAG_BIB_PATH")
    ap.add_argument("--apply", action="store_true",
                    help="write fixes (default: dry-run, ledger only)")
    ap.add_argument("--limit", type=int, default=0, help="fix only first N papers")
    ap.add_argument("--skip-layers", default="",
                    help="comma list to skip: bib,zotero,remote")
    args = ap.parse_args()
    skip = {s.strip() for s in args.skip_layers.split(",") if s.strip()}

    bib = args.bib or get_setting(_CFG["data_root"], "bib_path", "")
    ledger_path = Path(_CFG["data_dir"]) / "backfill_status.csv"

    candidates = papers_needing_fix(limit=args.limit)
    print(f"library : {_CFG['kb_name']} ({_CFG['data_root']})")
    print(f"mode    : {'APPLY' if args.apply else 'DRY-RUN (pass --apply to write fixes)'}")
    print(f"papers needing metadata fix: {len(candidates)}")
    if not candidates:
        print("nothing to do")
        return

    # previously-fixed papers (skip on re-run)
    done = set()
    if ledger_path.exists():
        with open(ledger_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("status") == "matched":
                    done.add(row["source"])
    todo = [c for c in candidates if c["file"] not in done]
    print(f"previously fixed (skipping): {len(done)} | to process: {len(todo)}")
    if not todo:
        print("all candidates already have a matched layer recorded")
        return

    ledger_rows = []
    remaining = list(todo)

    # layer 0: BibTeX snapshot
    if "bib" not in skip and bib and Path(bib).exists():
        t0 = time.time()
        r = layer_bib(bib, remaining, args.apply)
        matched = [x for x in r if x["status"] == "matched"]
        print(f"[layer 0] BibTeX: {len(matched)}/{len(remaining)} matched ({time.time()-t0:.0f}s)")
        ledger_rows.extend(r)
        remaining = [c for c in remaining if c["file"] not in {x['source'] for x in matched}]
    else:
        print("[layer 0] no BibTeX snapshot provided — skipping (pass --bib to use it)")

    # layer 1: live Zotero
    if "zotero" not in skip and remaining:
        t0 = time.time()
        r = layer_zotero(remaining, args.apply)
        matched = [x for x in r if x["status"] == "matched"]
        print(f"[layer 1] Zotero: {len(matched)}/{len(remaining)} matched ({time.time()-t0:.0f}s)")
        ledger_rows.extend(r)
        remaining = [c for c in remaining if c["file"] not in {x['source'] for x in matched}]
    else:
        print("[layer 1] skipped (already fixed, or excluded)")

    # layer 2: remote registries
    if "remote" not in skip and remaining:
        t0 = time.time()
        r = layer_remote(remaining, args.apply)
        matched = [x for x in r if x["status"] == "matched"]
        print(f"[layer 2] Remote registries: {len(matched)}/{len(remaining)} matched ({time.time()-t0:.0f}s)")
        ledger_rows.extend(r)
        remaining = [c for c in remaining if c["file"] not in {x['source'] for x in matched}]
    else:
        print("[layer 2] skipped or nothing left")

    # leftovers
    for c in remaining:
        ledger_rows.append({"source": c["file"], "layer": "none",
                            "title": c["title"], "year": "", "doi": "",
                            "authors": "", "status": "unresolved",
                            "detail": "no layer produced a record"})

    # write ledger
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not ledger_path.exists()
    with open(ledger_path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=LEDGER_FIELDS)
        if new_file:
            w.writeheader()
        w.writerows(ledger_rows)
    print(f"\nledger: {ledger_path} (+{len(ledger_rows)} rows)")

    # apply chroma/parent_store writes for matched rows
    if args.apply:
        matched = [r for r in ledger_rows if r["status"] == "matched" and r["layer"] != "none"]
        print(f"applying {len(matched)} matched records to chroma/parent_store...")
        import chromadb
        from chunking import atomic_json_dump
        col = chromadb.PersistentClient(path=_CFG["chroma_path"]).get_collection(
            _CFG["collection_name"])
        by_source = {r["source"]: r for r in matched}
        # chroma chunks
        offset, page, n_upd = 0, 2000, 0
        while True:
            res = col.get(limit=page, offset=offset, include=["metadatas"])
            ids = res["ids"]
            if not ids:
                break
            new_metas = []
            for m in res["metadatas"]:
                src = m.get("source", "")
                new = dict(m)
                rec = by_source.get(src)
                if rec:
                    for fld in ("title", "year", "doi", "authors"):
                        if rec.get(fld):
                            new[fld] = rec[fld]
                new_metas.append(new)
            col.update(ids=ids, metadatas=new_metas)
            n_upd += len(ids)
            offset += page
            if len(ids) < page:
                break
        print(f"[apply] chroma metadata updated on {n_upd} chunks")
        # parent_store meta too
        pm_dir = Path(_CFG["parent_store_dir"])
        for f in pm_dir.glob("*.json"):
            stem = f.stem
            src = stem + ".md" if not stem.endswith(".md") else stem
            rec = by_source.get(src) or by_source.get(stem)
            if not rec:
                continue
            data = json.loads(f.read_text(encoding="utf-8"))
            changed = False
            for chunk in data:
                m = chunk.get("meta") if isinstance(chunk, dict) else None
                if m is None:
                    continue
                for fld in ("title", "year", "doi", "authors"):
                    if rec.get(fld) and m.get(fld) != rec[fld]:
                        m[fld] = rec[fld]
                        changed = True
            if changed:
                atomic_json_dump(data, f)
        print("[apply] parent_store meta updated")
        # layer 3
        if bib:
            layer_audit(dry_run=False, bib=bib, limit=args.limit)
    else:
        print("\n(dry-run — no writes. Re-run with --apply to write matched records,")
        print(" then layer 3 meta_audit runs as the final proof-read round.)")


if __name__ == "__main__":
    main()