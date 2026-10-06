#!/usr/bin/env python3
"""
page_spec.py — parse page specifications like "1-3,7,9-12" into page lists.

Borrowed from PageIndex's agent_tools._expand_pages (MIT). One parser for the
whole toolkit: the agent tool surface and any CLI consumer share it, so page
handling stays consistent.

Guarantees:
  * sorts + dedups (overlapping ranges don't double-count)
  * bounds each part arithmetically before materializing (a spec like
    "1-1000000000" fails fast instead of allocating billions of ints)
  * three error codes: 'invalid' | 'too_many' | 'nonpositive'
"""

from __future__ import annotations

import re

MAX_REQUESTED_PAGES = 50

_PAGE_SPEC_RE = re.compile(r"\s*\d+\s*(?:-\s*\d+\s*)?(?:,\s*\d+\s*(?:-\s*\d+\s*)?)*")


class PageSpecError(ValueError):
    """Raised for a malformed or oversized page specification."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def expand_pages(pages) -> list[int]:
    """Expand '1-3,7' into a sorted, deduplicated page list.

    Raises PageSpecError with code 'invalid', 'too_many', or 'nonpositive'.
    """
    if not isinstance(pages, str):
        raise PageSpecError("invalid", f"Invalid page specification: {pages!r}")
    if not _PAGE_SPEC_RE.fullmatch(pages):
        raise PageSpecError("invalid", f"Invalid page specification '{pages}'")

    too_many = (f"Page specification '{pages}' spans more than "
                f"{MAX_REQUESTED_PAGES} pages; request a narrower range")
    expanded: set[int] = set()
    for part in pages.split(","):
        part = part.strip()
        try:
            if "-" in part:
                start, end = (int(x) for x in part.split("-", 1))
                if start > end:
                    raise PageSpecError("invalid",
                                        f"Invalid range '{part}': start must be <= end")
            else:
                start = end = int(part)
        except PageSpecError:
            raise
        except ValueError as exc:
            raise PageSpecError("invalid",
                                f"Invalid page specification '{pages}'") from exc
        if end - start + 1 > MAX_REQUESTED_PAGES:
            raise PageSpecError("too_many", too_many)
        expanded.update(range(start, end + 1))
        if len(expanded) > MAX_REQUESTED_PAGES:
            raise PageSpecError("too_many", too_many)

    if min(expanded) < 1:
        raise PageSpecError("nonpositive",
                            "Invalid page numbers. Page numbers must be positive integers")
    return sorted(expanded)


def format_page_spec(pages: list[int]) -> str:
    """Render a sorted page list back to compact spec form ('1-3,5,7-9')."""
    if not pages:
        return ""
    pages = sorted(set(int(p) for p in pages))
    runs, start, prev = [], pages[0], pages[0]
    for p in pages[1:]:
        if p == prev + 1:
            prev = p
            continue
        runs.append((start, prev))
        start = prev = p
    runs.append((start, prev))
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)


if __name__ == "__main__":
    import sys
    for spec in sys.argv[1:] or ["1-3,7,9-12"]:
        try:
            print(f"{spec} → {expand_pages(spec)}  (round-trip: {format_page_spec(expand_pages(spec))})")
        except PageSpecError as e:
            print(f"{spec} → ERROR[{e.code}]: {e}")
