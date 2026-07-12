"""Maps claims in the writer's final report back to the URL(s) they cite.

Relies on the writer_agent's prompt (see agents/config/tasks.yaml, synthesize_task)
instructing it to follow every factual claim with an inline citation in the exact
form "[Source: URL]". This module parses that convention back out of the report
text; it does not itself enforce citation - the enforcement is the task prompt.
"""

from __future__ import annotations

import re
from typing import Any

CITATION_PATTERN = re.compile(r"\[Source:\s*(https?://[^\]\s]+)\]")


def extract_citations(report: str) -> list[dict[str, Any]]:
    """Pull each cited claim out of the report, paired with its source URL(s).

    Args:
        report: The writer agent's final markdown report text.

    Returns:
        One dict per non-empty, citation-bearing line, each with:
          - "claim": the line's text with "[Source: URL]" markers stripped out.
          - "sources": the list of URLs cited on that line, in order.
    """
    claims: list[dict[str, Any]] = []
    for line in report.splitlines():
        line = line.strip()
        if not line:
            continue
        urls = CITATION_PATTERN.findall(line)
        if not urls:
            continue
        claim_text = CITATION_PATTERN.sub("", line).strip()
        claims.append({"claim": claim_text, "sources": urls})
    return claims


def build_source_index(report: str) -> dict[str, list[str]]:
    """Invert the report's citations into a URL -> claims-it-supports index.

    Args:
        report: The writer agent's final markdown report text.

    Returns:
        A dict mapping each cited URL to the list of claim texts it supports,
        in the order they appear in the report.
    """
    index: dict[str, list[str]] = {}
    for citation in extract_citations(report):
        for url in citation["sources"]:
            index.setdefault(url, []).append(citation["claim"])
    return index


def uncited_lines(report: str) -> list[str]:
    """Find substantive lines in the report body that carry no citation.

    Skips markdown structural lines (headings, list markers, horizontal rules,
    blank lines) since those aren't factual claims. Stops at the "Sources"
    section, since its lines are a bare URL list rather than claims.

    Args:
        report: The writer agent's final markdown report text.

    Returns:
        The substantive lines, in order, that have no "[Source: URL]" marker.
    """
    uncited: list[str] = []
    for line in report.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if lowered.startswith("# sources") or lowered.startswith("## sources"):
            break
        if stripped.startswith("#"):
            continue
        if CITATION_PATTERN.search(stripped):
            continue
        uncited.append(stripped)
    return uncited


def citation_coverage(report: str) -> float:
    """Compute the fraction of substantive report lines that carry a citation.

    Args:
        report: The writer agent's final markdown report text.

    Returns:
        num_cited_lines / (num_cited_lines + num_uncited_lines), or 1.0 if the
        report has no substantive lines at all.
    """
    num_cited = len(extract_citations(report))
    num_uncited = len(uncited_lines(report))
    total = num_cited + num_uncited
    return num_cited / total if total else 1.0
