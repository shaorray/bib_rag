#!/usr/bin/env python3
"""zotero_access.py — bib_rag's Zotero access layer.

Priority (so bib_rag reaches Zotero whenever it needs to):
  1. Zotero MCP server  (`zotero-mcp serve --transport stdio`, from the
     `zotero-mcp-server` PyPI package, https://github.com/54yyyu/zotero-mcp)
     — structured JSON via the `zotero_search_items` / `zotero_get_item_metadata`
       MCP tools (format=json).
  2. Zotero local HTTP API (http://localhost:23119) — dependency-free fallback,
     requires the Zotero desktop local API to be enabled.
  3. Graceful None/[] when neither is reachable (Zotero desktop not running,
     server not installed, ...). Consumers must never crash on an empty Zotero.

Consumers:
  - scripts/meta_audit.py            (ZoteroClient corroboration source)
  - src/bib_rag_writer*.py           (search_zotero, citation formatting)

Env knobs:
  BIB_RAG_ZOTERO_MCP=0    force-disable the MCP path (HTTP fallback only)
  BIB_RAG_ZOTERO_URL=...  override the local HTTP API base
                          (default http://localhost:23119/api/users/0)
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import re
import select
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

log = logging.getLogger("zotero_access")

HTTP_BASE = os.environ.get("BIB_RAG_ZOTERO_URL",
                           "http://localhost:23119/api/users/0")
HTTP_TIMEOUT = 5
MCP_TOOL_TIMEOUT = 30
MCP_START_TIMEOUT = 60
MCP_BINARY = "zotero-mcp"

_mcp: Optional["MCPClient"] = None   # None = not tried, False = unavailable
_mcp_disabled = os.environ.get("BIB_RAG_ZOTERO_MCP", "").strip().lower() in ("0", "false", "no")


# ---------------------------------------------------------------------------
# Normalization (Zotero item -> bib_rag flat dict)
# ---------------------------------------------------------------------------

def _normalize_item(data: Dict[str, Any], key: str = "") -> Dict[str, str]:
    """Flatten a Zotero item's `data` dict into bib_rag's string shape."""
    creators = data.get("creators") or []
    authors = "; ".join(
        f"{c.get('lastName', '')},{c.get('firstName', '')}".strip(",")
        for c in creators
        if c.get("lastName") or c.get("firstName")
    )
    date = data.get("date") or ""
    m = re.search(r"(19\d{2}|20\d{2})", date)
    year = m.group(1) if m else ""
    return {
        "key": key or data.get("key") or "",
        "title": data.get("title") or "",
        "doi": data.get("DOI") or "",
        "year": year,
        "date": date,
        "authors": authors,
        "journal": data.get("publicationTitle") or "",
        "volume": data.get("volume") or "",
        "issue": data.get("issue") or "",
        "pages": data.get("pages") or "",
        "item_type": data.get("itemType") or "",
    }


def display_authors(authors_str: str, max_display: int = 3) -> str:
    """`Doe,Jane;Smith,John` -> `Jane Doe and John Smith` (writers' format)."""
    names = [a.strip() for a in (authors_str or "").split(";") if a.strip()]
    if not names:
        return "Unknown"
    disp = []
    for n in names:
        if "," in n:
            last, first = [p.strip() for p in n.split(",", 1)]
            disp.append(f"{first} {last}".strip())
        else:
            disp.append(n)
    if len(disp) > max_display:
        return f"{disp[0]} et al."
    if len(disp) == 1:
        return disp[0]
    return ", ".join(disp[:-1]) + f" and {disp[-1]}"


# ---------------------------------------------------------------------------
# Zotero local HTTP API (fallback path)
# ---------------------------------------------------------------------------

def _http_json(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    url = HTTP_BASE + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
            return json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError,
            json.JSONDecodeError, TimeoutError, OSError) as ex:
        log.debug("zotero http %s failed: %s", path, ex)
        return None


def _http_search(query: str, limit: int = 3) -> List[Dict[str, str]]:
    data = _http_json("/items", {"q": query, "limit": limit, "itemType": "-attachment"})
    if not isinstance(data, list):
        return []
    return [_normalize_item(it.get("data") or {}, it.get("key") or "")
            for it in data]


def _http_item(key: str) -> Optional[Dict[str, str]]:
    it = _http_json("/items/" + urllib.parse.quote(key))
    if not isinstance(it, dict):
        return None
    return _normalize_item(it.get("data") or {}, it.get("key") or key)


# ---------------------------------------------------------------------------
# Zotero MCP server (primary path) — minimal stdio JSON-RPC client
# ---------------------------------------------------------------------------

class MCPError(Exception):
    pass


class MCPClient:
    """Newline-delimited JSON-RPC client for `zotero-mcp serve --transport stdio`."""

    def __init__(self, timeout: int = MCP_TOOL_TIMEOUT):
        self.timeout = timeout
        self.proc: Optional[subprocess.Popen] = None
        self._id = 0

    def start(self) -> "MCPClient":
        env = dict(os.environ)
        env.setdefault("ZOTERO_LOCAL", "true")
        env.setdefault("ZOTERO_MCP_SCHEMA_REFRESH", "0")  # no network schema refresh
        env.setdefault("ZOTERO_MCP_TOOLSETS", "none")     # core tools only
        self.proc = subprocess.Popen(
            [MCP_BINARY, "serve", "--transport", "stdio"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1, env=env,
        )
        try:
            self._request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "bib_rag", "version": "0.1"},
            }, timeout=MCP_START_TIMEOUT)
            self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        except Exception:
            self.close()
            raise
        return self

    def _send(self, obj: Dict[str, Any]) -> None:
        if self.proc is None or self.proc.poll() is not None:
            raise MCPError("zotero-mcp process not running")
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()

    def _read(self, timeout: float) -> Dict[str, Any]:
        end = time.time() + timeout
        while time.time() < end:
            r, _, _ = select.select([self.proc.stdout], [], [], 1)
            if r:
                line = self.proc.stdout.readline()
                if line and line.strip():
                    try:
                        return json.loads(line)
                    except json.JSONDecodeError:
                        continue
        raise MCPError("timeout waiting for zotero-mcp")

    def _request(self, method: str, params: Dict[str, Any],
                 timeout: Optional[int] = None) -> Dict[str, Any]:
        self._id += 1
        mid = self._id
        self._send({"jsonrpc": "2.0", "id": mid, "method": method, "params": params})
        deadline = time.time() + (timeout or self.timeout)
        while time.time() < deadline:
            msg = self._read(deadline - time.time())
            if msg.get("id") == mid:
                if "error" in msg:
                    raise MCPError("MCP error: " + json.dumps(msg["error"])[:300])
                return msg
            # otherwise a notification / unrelated response — keep reading
        raise MCPError(f"no response for {method}")

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        try:
            return self._call_tool_once(name, arguments)
        except (MCPError, OSError, ValueError) as ex:
            log.debug("zotero-mcp %s failed (%s) — restarting once", name, ex)
            self.close()
            self.start()
            return self._call_tool_once(name, arguments)

    def _call_tool_once(self, name: str, arguments: Dict[str, Any]) -> Any:
        resp = self._request("tools/call", {"name": name, "arguments": arguments})
        result = resp.get("result", {})
        if result.get("isError"):
            raise MCPError("tool error: " + json.dumps(result)[:300])
        sc = result.get("structuredContent")
        if sc is not None:
            return sc
        texts = [c.get("text", "") for c in result.get("content", [])
                 if c.get("type") == "text"]
        return "\n".join(texts)

    def close(self) -> None:
        if self.proc is not None:
            try:
                self.proc.terminate()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=5)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None


_ITEM_KEY_RE = re.compile(r"\*\*Item Key:\*\*\s*(\S+)")
_AUTHORS_RE = re.compile(r"\*\*Authors:\*\*\s*(.*)$", re.M)
_DATE_RE = re.compile(r"\*\*Date:\*\*\s*(.*)$", re.M)


def _parse_search_markdown(text: str) -> List[Dict[str, str]]:
    """Parse `zotero_search_items` markdown output into lightweight items."""
    out: List[Dict[str, str]] = []
    sections = re.split(r"(?m)^##\s*\d+\.\s+", text or "")
    for sec in sections[1:]:
        lines = sec.splitlines()
        title = lines[0].strip() if lines else ""
        m = _ITEM_KEY_RE.search(sec)
        if not m:
            continue
        a = _AUTHORS_RE.search(sec)
        d = _DATE_RE.search(sec)
        out.append({
            "key": m.group(1).strip(),
            "title": title,
            "authors": a.group(1).strip() if a else "",
            "date": d.group(1).strip() if d else "",
            "year": "",
            "doi": "",
            "journal": "", "volume": "", "issue": "", "pages": "",
            "item_type": "",
        })
    return out


def _get_mcp() -> Optional[MCPClient]:
    global _mcp
    if _mcp_disabled:
        return None
    if _mcp is None:
        try:
            _mcp = MCPClient().start()
            log.info("zotero-mcp connected (Zotero MCP v0 stdio)")
        except Exception as ex:
            _mcp = False
            log.info("zotero-mcp unavailable (%s) — falling back to local HTTP API", ex)
    return _mcp if _mcp else None


def close_mcp() -> None:
    global _mcp
    if isinstance(_mcp, MCPClient):
        _mcp.close()
    _mcp = None


atexit.register(close_mcp)


def _mcp_search(query: str, limit: int = 3) -> List[Dict[str, str]]:
    cli = _get_mcp()
    if cli is None:
        return []
    out = cli.call_tool("zotero_search_items", {"query": query, "limit": limit})
    # The tool returns structuredContent = {"result": "<markdown>"} — unwrap it
    if isinstance(out, dict):
        r = out.get("result")
        text = r if isinstance(r, str) else json.dumps(out, ensure_ascii=False)
    elif isinstance(out, str):
        text = out
    else:
        text = json.dumps(out, ensure_ascii=False)
    return _parse_search_markdown(text)


def _mcp_item(key: str) -> Optional[Dict[str, str]]:
    cli = _get_mcp()
    if cli is None:
        return None
    out = cli.call_tool("zotero_get_item_metadata",
                        {"item_key": key, "format": "json"})
    item: Any = out
    if isinstance(out, dict) and "result" in out:
        item = out["result"]
    elif isinstance(out, str):
        try:
            item = json.loads(out)
        except json.JSONDecodeError:
            return None
    if not isinstance(item, dict):
        return None
    return _normalize_item(item.get("data") or {}, item.get("key") or key)


# ---------------------------------------------------------------------------
# Public API (never raises — empty results on failure)
# ---------------------------------------------------------------------------

def zotero_search(query: str, limit: int = 3) -> List[Dict[str, str]]:
    """Search Zotero by title/creator/year. Lightweight items (key/title/...)."""
    try:
        items = _mcp_search(query, limit)
        if items:
            return items
    except Exception as ex:
        log.debug("zotero MCP search failed (%s) — HTTP fallback", ex)
    return _http_search(query, limit)


def zotero_item(key: str) -> Optional[Dict[str, str]]:
    """Full metadata for one Zotero item key, or None."""
    try:
        item = _mcp_item(key)
        if item:
            return item
    except Exception as ex:
        log.debug("zotero MCP metadata failed (%s) — HTTP fallback", ex)
    return _http_item(key)


def available() -> bool:
    """True if Zotero is reachable through either path."""
    try:
        if _get_mcp() is not None:
            return True
    except Exception:
        pass
    return _http_json("/items", {"limit": 1}) is not None
