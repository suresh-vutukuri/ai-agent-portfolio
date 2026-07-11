"""Small collection helpers - sandbox file for demoing the PR review agent."""

from __future__ import annotations


def dedupe(items: list[str]) -> list[str]:
    """Return `items` with duplicates removed, preserving order."""
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            result.append(item)
    return result
