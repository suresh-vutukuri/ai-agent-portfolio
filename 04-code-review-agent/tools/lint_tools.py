"""Wraps ruff and bandit as subprocess-based tools returning structured issues."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class RuffIssue:
    """One style/lint issue reported by ruff.

    Attributes:
        line: 1-indexed line number the issue was found on.
        rule_code: Ruff's rule code, e.g. "F401".
        message: Human-readable description of the issue.
    """

    line: int
    rule_code: str
    message: str


@dataclass
class BanditIssue:
    """One security issue reported by bandit.

    Attributes:
        line: 1-indexed line number the issue was found on.
        severity: Bandit's severity rating, e.g. "LOW", "MEDIUM", "HIGH".
        issue_text: Human-readable description of the issue.
    """

    line: int
    severity: str
    issue_text: str


def run_ruff(file_path: str) -> List[RuffIssue]:
    """Run `ruff check` on a file and return its issues as structured data.

    Args:
        file_path: Path to the Python file to lint.

    Returns:
        One RuffIssue per issue ruff found, in the order ruff reported them.
        An empty list means ruff found nothing to flag.

    Raises:
        FileNotFoundError: If file_path doesn't exist.
        RuntimeError: If the `ruff` executable isn't installed/on PATH, or
            its output couldn't be parsed as JSON.
    """
    if not Path(file_path).is_file():
        raise FileNotFoundError(f"No such file: {file_path}")

    try:
        result = subprocess.run(
            ["ruff", "check", "--output-format=json", file_path],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as e:
        raise RuntimeError("ruff is not installed or not found on PATH. Install it with `pip install ruff`.") from e

    # ruff exits with status 1 when it finds issues - that's normal, not a failure.
    try:
        issues = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Could not parse ruff output as JSON: {result.stderr or result.stdout}") from e

    return [
        RuffIssue(line=issue["location"]["row"], rule_code=issue["code"], message=issue["message"])
        for issue in issues
    ]


def run_bandit(file_path: str) -> List[BanditIssue]:
    """Run bandit on a file and return its security issues as structured data.

    Args:
        file_path: Path to the Python file to scan.

    Returns:
        One BanditIssue per issue bandit found, in the order bandit reported
        them. An empty list means bandit found nothing to flag.

    Raises:
        FileNotFoundError: If file_path doesn't exist.
        RuntimeError: If the `bandit` executable isn't installed/on PATH, or
            its output couldn't be parsed as JSON.
    """
    if not Path(file_path).is_file():
        raise FileNotFoundError(f"No such file: {file_path}")

    try:
        result = subprocess.run(
            ["bandit", "-f", "json", file_path],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as e:
        raise RuntimeError("bandit is not installed or not found on PATH. Install it with `pip install bandit`.") from e

    # bandit exits with status 1 when it finds issues - that's normal, not a failure.
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Could not parse bandit output as JSON: {result.stderr or result.stdout}") from e

    return [
        BanditIssue(line=issue["line_number"], severity=issue["issue_severity"], issue_text=issue["issue_text"])
        for issue in data["results"]
    ]
