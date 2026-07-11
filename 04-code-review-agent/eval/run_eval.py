"""Precision/recall eval: does the review agent flag known injected issues?

Runs each synthetic test diff in test_diffs/ through agent/reviewer.py's
lint + review pipeline, then checks whether the review text flags each
issue listed in test_diffs/expected_issues.json. review_pr() only accepts
GitHub PR coordinates, so this reuses reviewer's private lint/query helpers
directly with a synthetic PRDiff built from each diff's own added lines
(every fixture here is a "new file" diff, so its added lines are the whole
file). Computes precision and recall per diff and in aggregate, and writes
eval/results/scorecard.csv and eval/results/summary.json.
"""

from __future__ import annotations

import asyncio
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent
TEST_DIFFS_DIR = EVAL_DIR / "test_diffs"
EXPECTED_ISSUES_PATH = TEST_DIFFS_DIR / "expected_issues.json"
RESULTS_DIR = EVAL_DIR / "results"

# agent/ and tools/ are plain sibling dirs, not packages
sys.path.insert(0, str(PROJECT_ROOT / "agent"))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import reviewer  # noqa: E402
from diff_parser import parse_diff  # noqa: E402
from github_client import PRDiff  # noqa: E402

# Keywords looked for in a matching review bullet, per injected issue type.
# This is a heuristic over free-form LLM prose, not an exact grader - it can
# miss phrasing it doesn't anticipate, or (rarely) match an unrelated bullet
# that happens to share a file, line number, and keyword by coincidence.
ISSUE_TYPE_KEYWORDS: Dict[str, List[str]] = {
    "sql_injection": ["sql injection", "sql", "injection"],
    "hardcoded_secret": ["hardcoded", "hard-coded", "secret", "credential", "password"],
    "bare_except": ["bare except", "except:", "bare `except`", "broad except"],
    "unused_import": ["unused import", "unused"],
    "off_by_one": ["off-by-one", "off by one", "boundary", "index"],
}

SCORECARD_FIELDNAMES = ["diff_file", "num_expected", "num_flagged", "true_positives", "precision", "recall"]


def _load_expected_issues() -> List[Dict[str, Any]]:
    """Read the labeled test diffs from expected_issues.json.

    Returns:
        The list of test-case dicts (diff_file, expected_issues), in file order.
    """
    with open(EXPECTED_ISSUES_PATH, encoding="utf-8") as f:
        return json.load(f)


def _review_diff_text(diff_text: str) -> str:
    """Run reviewer.py's lint + review pipeline directly on a diff string.

    Bypasses review_pr()/fetch_pr_diff() (which require a real GitHub PR)
    by reusing reviewer's private helpers with a synthetic PRDiff built
    from the diff's own added lines.

    Args:
        diff_text: A unified diff for one synthetic test file.

    Returns:
        The agent's markdown review.

    Raises:
        RuntimeError: If the Claude Code CLI isn't installed or the Claude
            Agent SDK query fails (propagated from reviewer._run_review_query).
    """
    changed_files = parse_diff(diff_text)
    changed_files_content = {
        cf.filename: "\n".join(line.content for line in cf.added_lines) + "\n" for cf in changed_files
    }
    pr_diff = PRDiff(diff_text=diff_text, changed_files=changed_files_content)

    findings = reviewer._lint_python_files(pr_diff, changed_files)
    findings_text = reviewer._format_findings(findings)
    return asyncio.run(reviewer._run_review_query(diff_text, findings_text))


def _extract_bullets(review_text: str, sections: Optional[Set[str]] = None) -> List[str]:
    """Pull every bullet line out of the given markdown sections.

    Args:
        review_text: The agent's markdown review.
        sections: Lowercase heading names to collect bullets from. Defaults
            to {"critical", "warnings", "suggestions"} (every findings section).

    Returns:
        One string per bullet point found under a matching "## <Heading>"
        heading, in document order.
    """
    if sections is None:
        sections = {"critical", "warnings", "suggestions"}
    bullets: List[str] = []
    in_target_section = False
    for line in review_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            heading = stripped[3:].strip().lower()
            in_target_section = heading in sections
            continue
        if in_target_section and stripped.startswith(("- ", "* ")):
            bullets.append(stripped)
    return bullets


def _extract_critical_bullets(review_text: str) -> List[str]:
    """Pull bullet lines out of only the "## Critical" section.

    Args:
        review_text: The agent's markdown review.

    Returns:
        One string per bullet point found under the "## Critical" heading,
        in document order.
    """
    return _extract_bullets(review_text, {"critical"})


def _bullet_matches(bullet: str, expected: Dict[str, Any]) -> bool:
    """Check whether one review bullet plausibly covers one expected issue.

    A match requires the bullet to mention the expected file, the expected
    line number as a standalone token, and at least one keyword associated
    with the expected issue type.

    Args:
        bullet: One bullet line from _extract_bullets.
        expected: One entry from expected_issues.json (file, line, issue_type).

    Returns:
        True if the bullet appears to cover the expected issue.
    """
    lower = bullet.lower()
    if expected["file"].lower() not in lower:
        return False
    if not re.search(rf"\b{re.escape(str(expected['line']))}\b", bullet):
        return False
    keywords = ISSUE_TYPE_KEYWORDS.get(expected["issue_type"], [])
    return any(keyword in lower for keyword in keywords)


def _score_review(review_text: str, expected_issues: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Match a review's bullets against one diff's expected issues.

    Args:
        review_text: The agent's markdown review for one diff.
        expected_issues: That diff's expected_issues entries.

    Returns:
        A dict with num_flagged (total bullets in the review), num_expected,
        true_positives (expected issues matched by some bullet), precision
        (true_positives / num_flagged), recall (true_positives /
        num_expected), and the same true_positives/num_flagged pair scoped
        to only the "## Critical" section (critical_true_positives,
        critical_num_flagged). precision/recall are None when their
        denominator is zero.
    """
    bullets = _extract_bullets(review_text)
    critical_bullets = _extract_critical_bullets(review_text)

    matched_bullet_indices: Set[int] = set()
    true_positives = 0
    matched_critical_indices: Set[int] = set()
    critical_true_positives = 0

    for expected in expected_issues:
        for i, bullet in enumerate(bullets):
            if i in matched_bullet_indices:
                continue
            if _bullet_matches(bullet, expected):
                matched_bullet_indices.add(i)
                true_positives += 1
                break
        for i, bullet in enumerate(critical_bullets):
            if i in matched_critical_indices:
                continue
            if _bullet_matches(bullet, expected):
                matched_critical_indices.add(i)
                critical_true_positives += 1
                break

    num_flagged = len(bullets)
    num_expected = len(expected_issues)
    critical_num_flagged = len(critical_bullets)

    return {
        "num_flagged": num_flagged,
        "num_expected": num_expected,
        "true_positives": true_positives,
        "precision": true_positives / num_flagged if num_flagged else None,
        "recall": true_positives / num_expected if num_expected else None,
        "critical_num_flagged": critical_num_flagged,
        "critical_true_positives": critical_true_positives,
    }


def evaluate_diff(test_case: Dict[str, Any]) -> Dict[str, Any]:
    """Run one labeled test diff through the reviewer and score it.

    Args:
        test_case: One entry from expected_issues.json (diff_file, expected_issues).

    Returns:
        A row dict matching SCORECARD_FIELDNAMES, plus critical_num_flagged
        and critical_true_positives (used only to roll up critical_precision/
        critical_recall in summarize(); not written to the CSV).
    """
    diff_path = TEST_DIFFS_DIR / test_case["diff_file"]
    diff_text = diff_path.read_text(encoding="utf-8")

    review_text = _review_diff_text(diff_text)
    scores = _score_review(review_text, test_case["expected_issues"])

    return {
        "diff_file": test_case["diff_file"],
        "num_expected": scores["num_expected"],
        "num_flagged": scores["num_flagged"],
        "true_positives": scores["true_positives"],
        "precision": scores["precision"],
        "recall": scores["recall"],
        "critical_num_flagged": scores["critical_num_flagged"],
        "critical_true_positives": scores["critical_true_positives"],
    }


def _write_scorecard_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    """Dump one row per test diff to a CSV file.

    Args:
        rows: Per-diff result dicts, as produced by evaluate_diff. May carry
            extra keys (e.g. critical_num_flagged) beyond SCORECARD_FIELDNAMES;
            those are dropped here rather than written to the CSV.
        path: Where to write the CSV; parent directories get created if missing.

    Returns:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SCORECARD_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: List[Optional[float]]) -> Optional[float]:
    """Average the non-None values in a list.

    Args:
        values: Numbers (or None) to average.

    Returns:
        The mean of the non-None values, or None if there are none.
    """
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Roll the per-diff rows up into aggregate precision/recall.

    Reports critical_precision/critical_recall as the headline metrics -
    scoped to only the "## Critical" section, since that's what the reviewer
    actually intends as must-fix findings, and is far less diluted by the
    legitimate-but-unlabeled extra issues (style nits, missing test
    coverage, etc.) a thorough reviewer also raises under Warnings/
    Suggestions. micro_precision/macro_precision (pooled across every
    Critical/Warnings/Suggestions bullet) are kept alongside for
    transparency into that broader, noisier picture.

    Micro-averages pool raw counts across every diff (appropriate since our
    diffs carry different numbers of expected issues); macro-averages are
    the mean of each diff's own precision/recall.

    Args:
        rows: Per-diff result dicts, as produced by evaluate_diff.

    Returns:
        A dict with total counts plus critical, micro, and macro precision/recall.
    """
    total_expected = sum(row["num_expected"] for row in rows)
    total_flagged = sum(row["num_flagged"] for row in rows)
    total_true_positives = sum(row["true_positives"] for row in rows)
    total_critical_flagged = sum(row["critical_num_flagged"] for row in rows)
    total_critical_true_positives = sum(row["critical_true_positives"] for row in rows)

    return {
        "num_diffs": len(rows),
        "total_expected_issues": total_expected,
        "total_flagged_issues": total_flagged,
        "total_true_positives": total_true_positives,
        "critical_precision": total_critical_true_positives / total_critical_flagged if total_critical_flagged else None,
        "critical_recall": total_critical_true_positives / total_expected if total_expected else None,
        "micro_precision": total_true_positives / total_flagged if total_flagged else None,
        "micro_recall": total_true_positives / total_expected if total_expected else None,
        "macro_precision": _mean([row["precision"] for row in rows]),
        "macro_recall": _mean([row["recall"] for row in rows]),
    }


def main() -> None:
    """Evaluate every test diff and write the scorecard + summary.

    Returns:
        None. Writes eval/results/scorecard.csv and eval/results/summary.json,
        and prints the summary to stdout as it goes.
    """
    test_cases = _load_expected_issues()
    print(f"Loaded {len(test_cases)} test diff(s) from {EXPECTED_ISSUES_PATH}")

    rows = []
    for test_case in test_cases:
        print(f"  Evaluating {test_case['diff_file']}...")
        rows.append(evaluate_diff(test_case))

    scorecard_path = RESULTS_DIR / "scorecard.csv"
    summary_path = RESULTS_DIR / "summary.json"

    _write_scorecard_csv(rows, scorecard_path)
    summary = summarize(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote {scorecard_path}")
    print(f"Wrote {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
