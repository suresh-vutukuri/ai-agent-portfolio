"""Runs the labeled test queries through the triage crew and scores routing accuracy.

Each query is run via run_triage(), which already logs which specialist's
tools fired to logs/handoff_log.jsonl (see agents/handoff_logger.py). This
reads that same record back right after each run rather than tracking tool
calls a second time - nesting a second event-bus scope around run_triage()
would silently drop its internal listener (see track_tool_calls's docstring).
Compares the logged agent_invoked against the query's expected_specialist
label, and writes eval/results/routing_scorecard.csv (per-query) and
eval/results/summary.json (aggregate accuracy).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent
TEST_QUERIES_PATH = EVAL_DIR / "test_queries.json"
RESULTS_DIR = EVAL_DIR / "results"

# agents/ is a plain sibling dir, not a package
sys.path.insert(0, str(PROJECT_ROOT / "agents"))

from handoff_logger import read_last_handoff  # noqa: E402
from run_triage import run_triage  # noqa: E402

SCORECARD_FIELDNAMES = [
    "id",
    "query",
    "expected_specialist",
    "actual_specialist",
    "correct",
    "delegated_to",
    "tools_called",
]


def _load_test_queries(path: Path) -> List[dict[str, Any]]:
    """Read the labeled test queries from JSON.

    Args:
        path: Path to test_queries.json.

    Returns:
        The list of query dicts, in file order.
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_query(example: dict[str, Any]) -> dict[str, Any]:
    """Run one labeled query through the triage crew and score its routing.

    Args:
        example: One entry from test_queries.json (id, query, expected_specialist).

    Returns:
        A row dict matching SCORECARD_FIELDNAMES.
    """
    run_triage(example["query"])
    record = read_last_handoff()
    actual_specialist = record["agent_invoked"]
    tool_calls = record["tool_calls"]
    delegations = record.get("delegations", [])

    return {
        "id": example["id"],
        "query": example["query"],
        "expected_specialist": example["expected_specialist"],
        "actual_specialist": actual_specialist,
        "correct": actual_specialist == example["expected_specialist"],
        "delegated_to": "; ".join(delegations),
        "tools_called": "; ".join(tool_calls),
    }


def _write_scorecard_csv(rows: List[dict[str, Any]], path: Path) -> None:
    """Dump one row per query to a CSV file.

    Args:
        rows: Per-query result dicts, as produced by evaluate_query.
        path: Where to write the CSV; parent directories get created if missing.

    Returns:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SCORECARD_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: List[dict[str, Any]]) -> dict[str, Any]:
    """Roll the per-query rows up into an aggregate routing accuracy score.

    Args:
        rows: Per-query result dicts, as produced by evaluate_query.

    Returns:
        A dict with the total query count, correct count, and routing
        accuracy (num_correct / num_queries).
    """
    num_queries = len(rows)
    num_correct = sum(1 for row in rows if row["correct"])
    return {
        "num_queries": num_queries,
        "num_correct": num_correct,
        "routing_accuracy": num_correct / num_queries if num_queries else None,
    }


def main() -> None:
    """Evaluate the whole test set and write the scorecard + summary.

    Returns:
        None. Writes eval/results/routing_scorecard.csv and
        eval/results/summary.json, and prints the summary to stdout as it goes.
    """
    test_queries = _load_test_queries(TEST_QUERIES_PATH)
    print(f"Loaded {len(test_queries)} test quer(y/ies) from {TEST_QUERIES_PATH}")

    rows = []
    for example in test_queries:
        print(f"  Evaluating {example['id']}...")
        rows.append(evaluate_query(example))

    scorecard_path = RESULTS_DIR / "routing_scorecard.csv"
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
