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
import os
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
        # Candidacy = a real metadata GAP (no DOI, or no title). A noisy title
        # ALONE is not a gap: e.g. 'Ca(2+) signalling...' trips the heuristic
        # but is a real paper title with valid doi/authors/year, and would
        # otherwise resurface in every single run forever. `noisy` still rides
        # along so layers can adapt their matching.
        if not doi or not title:
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
    from bind_zotero import bind_paper
    entries = parse_bib_entries(Path(bib_path))
    print(f"[layer 0] BibTeX snapshot: {len(entries)} entries from {bib_path}")
    import re
    results = []
    for c in candidates:
        rec = bind_paper(Path(c["file"]) if Path(c["file"]).exists()
                         else Path(_CFG["parent_store_dir"]) / c["file"],
                         entries, c["meta"])
        if rec["status"] == "matched":
            results.append({"source": c["file"], "layer": "bib",
                            "title": rec["title"], "year": rec["year"],
                            "doi": rec["doi"], "authors": rec["authors"],
                            "status": "matched",
                            "detail": f"bib:{rec['match_type']}"})
        else:
            results.append({"source": c["file"], "layer": "bib",
                            "title": c["title"], "year": c["meta"].get("year", ""),
                            "doi": "", "authors": "", "status": "unmatched",
                            "detail": f"no bib binding ({rec['match_type']})"})
    return results


# ── layer 1: live Zotero ────────────────────────────────────────────────────

def layer_zotero(candidates, dry_run):
    """Match candidates against a FULL bulk pull of the local Zotero library.

    Measured (2026-08, eph_rag host): the per-paper search endpoint costs
    ~10.8s/call (server-side title search), so 623 candidates serialize to
    ~112 min — while paginating /items (500/page) pulls the whole library
    (8.5k items) in ~6s with DOI present on 97.5% of entries. A one-time
    bulk pull + in-memory normalized-title matching replaces the per-paper
    searches entirely. Falls back to per-paper zotero_search when the bulk
    endpoint is unavailable.
    """
    import zotero_access
    if not zotero_access.available():
        print("[layer 1] Zotero unreachable (no MCP, no HTTP) — skipping")
        return []

    # ── build in-memory library index: bulk pull via local HTTP ──
    lib_index = []  # list of normalized item dicts
    try:
        from zotero_access import HTTP_BASE, _http_json, _normalize_item
        start, pulled, t0 = 0, 0, time.time()
        while True:
            page = _http_json("/items", {"itemType": "-attachment", "limit": 500,
                                         "format": "json", "start": start})
            if not isinstance(page, list) or not page:
                break
            for it in page:
                d = it.get("data") or {}
                if d.get("itemType") in ("attachment", "note", "annotation"):
                    continue
                norm = _normalize_item(d, it.get("key") or "")
                if norm.get("title"):
                    lib_index.append(norm)
            pulled += len(page)
            start += 500
            if len(page) < 500:
                break
        print(f"[layer 1] bulk-pulled {pulled} items ({len(lib_index)} with titles) "
              f"in {time.time()-t0:.0f}s — matching in memory")
    except Exception as ex:
        print(f"[layer 1] bulk pull failed ({type(ex).__name__}) — falling back "
              "to per-paper search (slow)")
        lib_index = None

    def _match_lib(title):
        """In-memory title match against the bulk-pulled index."""
        from bib_to_parent_store import normalize_paper_title
        nt = normalize_paper_title(title)
        if not nt:
            return None
        for it in lib_index or []:
            bt = normalize_paper_title(it.get("title", ""))
            if bt and (bt[:20] in nt or nt[:20] in bt):
                return it
        return None

    results = []
    for c in candidates:
        title = c["title"]
        if c.get("noisy"):
            results.append({"source": c["file"], "layer": "zotero",
                            "title": title, "year": "", "doi": "",
                            "authors": "", "status": "unmatched",
                            "detail": "skipped: noisy title (journal header)"})
            continue
        it = None
        if lib_index and title:
            it = _match_lib(title)
        elif title:
            items = zotero_access.zotero_search(title, limit=1)
            it = items[0] if items else None
        if not it:
            results.append({"source": c["file"], "layer": "zotero",
                            "title": title, "year": "", "doi": "",
                            "authors": "", "status": "unmatched",
                            "detail": "no zotero hit"})
            continue
        results.append({"source": c["file"], "layer": "zotero",
                        "title": it.get("title", title),
                        "year": it.get("year", ""),
                        "doi": it.get("doi", ""),
                        "authors": it.get("authors", ""),
                        "status": "matched" if it.get("doi") or it.get("title") else "unmatched",
                        "detail": "zotero bulk" if lib_index else "zotero item"})
    return results


# ── layer 2: remote registries (via meta_audit clients) ────────────────────

def layer_remote(candidates, dry_run):
    """Remote registries (Crossref / OpenAlex / PubMed) with thread-pool fan-out.

    Each candidate walks its own fallback chain (crossref-doi → crossref-title
    → openalex-title → pubmed-title); the registry clients are stateless HTTP
    wrappers (a throttle timestamp + counters), so candidates run in parallel
    while each chain stays sequential. Measured serial cost ~4.3s/paper
    (worst case, 3 registry fallbacks) → thread pool cuts wall time ~Nx.
    """
    from concurrent.futures import ThreadPoolExecutor
    from meta_audit import (CrossrefClient, PubmedClient, OpenAlexClient,
                            DEFAULT_CROSSREF_MAILTO)
    cr = CrossrefClient(mailto=DEFAULT_CROSSREF_MAILTO)
    pm = PubmedClient()
    oa = OpenAlexClient()

    def resolve_one(c):
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
            return {"source": c["file"], "layer": f"remote:{via}",
                    "title": rec.get("title", title),
                    "year": str(rec.get("year", "")),
                    "doi": rec.get("doi", doi),
                    "authors": rec.get("authors", ""),
                    "status": "matched", "detail": via}
        return {"source": c["file"], "layer": "remote",
                "title": title, "year": "", "doi": doi,
                "authors": "", "status": "unmatched",
                "detail": "no registry record"}

    # Crossref polite-pool throttle lives on the shared client; 4 workers keep
    # total request rate well under the 50/s polite ceiling (bottleneck is
    # per-request latency anyway).
    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(resolve_one, candidates))
    return results


# ── layer 3: meta_audit proof-read round ────────────────────────────────────

def layer_audit(dry_run, bib, limit):
    """Run meta_audit.py as the final proof-read round (its own confidence
    gating + backups apply).

    --limit is forwarded so a trial backfill doesn't trigger a full-library
    audit; --resume lets an interrupted proof-read continue where it left off
    (meta_audit maintains its own progress file). An apply-mode proof-read on
    the full library is still a long job — it is opt-in via BACKFILL_AUDIT=full.
    """
    cmd = [sys.executable, "-B", str(SCRIPTS_DIR / "metadata" / "meta_audit.py"),
           "--bib", str(bib), "--resume"]
    if limit:
        cmd += ["--limit", str(limit)]
    if os.environ.get("BACKFILL_AUDIT", "").strip().lower() == "full":
        print("[layer 3] BACKFILL_AUDIT=full — running meta_audit over the whole library (--apply)")
        return subprocess.call(cmd + ["--apply"])
    print("[layer 3] meta_audit proof-read round deferred (would audit the whole "
          "library). Run explicitly when wanted:")
    print("          " + " ".join(cmd + ["--apply"]))
    return 0


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

    # Per-layer failure ledger: skip a candidate only when EVERY layer this
    # run will attempt has already failed for it (and on 'unresolved', which
    # means no layer produced a record). Rows with status=matched must NOT
    # block reprocessing — a dry-run records matched without writing
    # parent_store, and papers_needing_fix() naturally re-includes anything
    # still missing meta; blocking on matched would permanently drop those
    # papers from every future --apply run.
    failed_layers = {}   # source -> set of layers that failed (bib/zotero/remote)
    unresolved = set()
    if ledger_path.exists():
        with open(ledger_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                src = row.get("source") or ""
                status = row.get("status") or ""
                if status == "unresolved":
                    unresolved.add(src)
                elif status == "unmatched":
                    layer = (row.get("layer") or "").split(":")[0]
                    if layer in ("bib", "zotero", "remote"):
                        failed_layers.setdefault(src, set()).add(layer)

    def _attempt_layers():
        layers = set()
        if "bib" not in skip and bib and Path(bib).exists():
            layers.add("bib")
        if "zotero" not in skip:
            layers.add("zotero")
        if "remote" not in skip:
            layers.add("remote")
        return layers

    attempt = _attempt_layers()
    todo = [c for c in candidates
            if c["file"] not in unresolved
            and not attempt.issubset(failed_layers.get(c["file"], set()))]
    n_blocked = len([c for c in candidates if c["file"] in unresolved]) + \
        len([c for c in candidates if c["file"] not in unresolved
             and attempt.issubset(failed_layers.get(c["file"], set()))])
    print(f"to process: {len(todo)} | fully-failed previously (skipping): {n_blocked}")

    ledger_rows = []
    remaining = list(todo)

    # layer 0: BibTeX snapshot
    if "bib" not in skip and bib and Path(bib).exists():
        t0 = time.time()
        r = layer_bib(bib, remaining, args.apply)
        matched = [x for x in r if x["status"] == "matched"]
        print(f"[layer 0] BibTeX: {len(matched)}/{len(remaining)} matched ({time.time()-t0:.0f}s)")
        ledger_rows.extend(r)
        # Only drain papers whose matched record is DOI-complete; a match
        # without a DOI still needs the registries (layer 2), so it stays in
        # the flow.
        complete = {x["source"] for x in matched if x.get("doi")}
        remaining = [c for c in remaining if c["file"] not in complete]
    else:
        print("[layer 0] no BibTeX snapshot provided — skipping (pass --bib to use it)")

    # layer 1: live Zotero
    if "zotero" not in skip and remaining:
        t0 = time.time()
        r = layer_zotero(remaining, args.apply)
        matched = [x for x in r if x["status"] == "matched"]
        n_doi = len([x for x in matched if x.get("doi")])
        print(f"[layer 1] Zotero: {len(matched)}/{len(remaining)} matched "
              f"({n_doi} with DOI, {time.time()-t0:.0f}s)")
        ledger_rows.extend(r)
        # Title-only matches (Zotero items often lack DOI) must NOT drain —
        # layer 2 still needs to fill the DOI from Crossref/OpenAlex/PubMed.
        complete = {x["source"] for x in matched if x.get("doi")}
        remaining = [c for c in remaining if c["file"] not in complete]
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

        def chroma_source_variants(fn: str):
            """parent-store filename -> possible chroma `source` values.

            Ledger `source` keys are parent-store filenames ('X_md.json' or
            'Author_Year_Title.json' stems included); chroma metadata `source`
            values are '.md' names where the PMID-style '_md' suffix becomes
            '.md' (measured: stem '10087273_md' -> chroma '10087273.md').
            Name-style stems map to '<stem>.md' directly.
            """
            stem = fn[:-5] if fn.endswith(".json") else fn
            out = {stem, stem + ".md"}
            if stem.endswith("_md"):
                out.add(stem[:-3] + ".md")
            return sorted(out)

        want = sorted({v for s in by_source for v in chroma_source_variants(s)})
        # ── chroma chunks: filtered read (no full scan) ──
        found = {}
        page = 100
        for i in range(0, len(want), page):
            chunk_keys = want[i:i + page]
            res = col.get(where={"source": {"$in": chunk_keys}},
                          include=["metadatas"], limit=10000)
            for cid, m in zip(res["ids"] or [], res["metadatas"] or []):
                found[cid] = dict(m)
        print(f"[apply] chroma: located {len(found)} chunks for {len(want)} source variants")
        n_upd = 0
        BATCH = 500
        ids_batch, meta_batch = [], []
        for cid, m in found.items():
            src = m.get("source", "")
            rec = by_source.get(src)
            if rec is None:
                # resolved via a variant — map back through the ledger by stem
                stem = src[:-3] if src.endswith(".md") else src
                for alt in chroma_source_variants(stem):
                    if alt in by_source:
                        rec = by_source[alt]
                        break
            if rec is None:
                continue
            new = dict(m)
            for fld in ("title", "year", "doi", "authors"):
                if rec.get(fld):
                    new[fld] = rec[fld]
            if new != m:  # only update chunks whose metadata actually changes
                ids_batch.append(cid)
                meta_batch.append(new)
                n_upd += 1
                if len(ids_batch) >= BATCH:
                    col.update(ids=ids_batch, metadatas=meta_batch)
                    ids_batch, meta_batch = [], []
        if ids_batch:
            col.update(ids=ids_batch, metadatas=meta_batch)
        print(f"[apply] chroma metadata updated on {n_upd} chunks (filtered locate + changed-only write)")
        # parent_store meta too — only touch files that actually matched
        pm_dir = Path(_CFG["parent_store_dir"])
        n_pm = 0
        for fname, rec in by_source.items():
            f = pm_dir / fname
            if not f.exists():
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
                n_pm += 1
        print(f"[apply] parent_store meta updated on {n_pm} files")
        # layer 3
        if bib:
            layer_audit(dry_run=False, bib=bib, limit=args.limit)
    else:
        print("\n(dry-run — no writes. Re-run with --apply to write matched records,")
        print(" then layer 3 meta_audit runs as the final proof-read round.)")


if __name__ == "__main__":
    main()