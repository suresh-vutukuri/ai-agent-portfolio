"""Evaluation harness: compares compute_htf_bias() against a hand-labeled golden dataset.

This only exercises the deterministic analysis pipeline (fetch_htf_bars,
compute_htf_bias, find_entry_candidates) - no LLM calls, so it's free, fast,
and repeatable.

Known limitation: yfinance's 5-minute intraday data only reaches back ~60
days from *today*, not 60 days from an arbitrary historical end_date, so it
can't reliably re-fetch a temporally accurate LTF window for golden-dataset
dates that are themselves close to 60 days old. The candidate sanity check
therefore runs find_entry_candidates() against *current* LTF data alongside
each entry's *historical* HTF bias - a structural sanity check of the
candidate-ranking logic, not a true point-in-time backtest of LTF candidates.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, TypedDict

# data/ and analysis/ are sibling project folders, not installed packages -
# make their modules importable the same flat way the rest of this project
# does (see graph/nodes.py for the same pattern).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _subdir in ("data", "analysis"):
    _path = str(_PROJECT_ROOT / _subdir)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from htf_bias import compute_htf_bias  # noqa: E402
from htf_data import fetch_htf_bars  # noqa: E402
from ltf_candidates import DEFAULT_NEAR_TERM_THRESHOLD_PCT, find_entry_candidates  # noqa: E402
from ltf_data import fetch_ltf_bars  # noqa: E402

GOLDEN_DATASET_PATH = Path(__file__).resolve().parent / "golden_dataset.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
SUMMARY_PATH = RESULTS_DIR / "summary.json"

HTF_PERIOD_DAYS = 60


class GoldenEntry(TypedDict):
    """One hand-labeled golden-dataset row."""

    date: str
    ticker: str
    expected_bias: str
    notes: str


def _load_golden_dataset(path: Path) -> list[GoldenEntry]:
    """Load the hand-labeled golden dataset from disk.

    Args:
        path: Path to golden_dataset.json.

    Returns:
        The list of golden entries.
    """
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _check_candidate_sanity(candidate: dict, near_term_threshold_pct: float) -> dict:
    """Run basic structural sanity checks on a single top near-term candidate.

    Checks:
      - An invalidation level can be derived from the zone (bottom for a
        bullish zone, top for a bearish zone - the opposite side of the zone
        from entry, per the ICT invalidation concept) and the zone itself is
        well-formed (top strictly greater than bottom).
      - The candidate's own distance_from_current_price['pct'] is actually
        <= near_term_threshold_pct - i.e. it's genuinely near-term and this
        isn't a tiering/ranking bug slipping a standing-reference zone in.

    Args:
        candidate: An EntryCandidate dict from find_entry_candidates().
        near_term_threshold_pct: The threshold the candidate was tiered
            against.

    Returns:
        {'invalidation_level': float, 'invalidation_exists': bool,
         'within_near_term_threshold': bool, 'passed': bool}
    """
    top, bottom = candidate["top"], candidate["bottom"]
    zone_well_formed = top > bottom

    invalidation_level = bottom if candidate["type"] == "bullish" else top
    invalidation_exists = zone_well_formed and invalidation_level is not None

    distance_pct = candidate["distance_from_current_price"]["pct"]
    within_threshold = distance_pct <= near_term_threshold_pct

    return {
        "invalidation_level": invalidation_level,
        "invalidation_exists": invalidation_exists,
        "within_near_term_threshold": within_threshold,
        "passed": invalidation_exists and within_threshold,
    }


def _evaluate_entry(entry: GoldenEntry) -> dict:
    """Run the deterministic pipeline for one golden-dataset entry and score it.

    Args:
        entry: One {date, ticker, expected_bias, notes} golden entry.

    Returns:
        A per-entry result dict with the computed bias, whether it matched
        the label, the top near-term candidate (if any) and its sanity
        check, and any errors hit along the way.
    """
    result: dict = {
        "date": entry["date"],
        "ticker": entry["ticker"],
        "expected_bias": entry["expected_bias"],
        "notes": entry["notes"],
        "computed_bias": None,
        "confidence": None,
        "bias_match": False,
        "top_near_term_candidate": None,
        "candidate_sanity": None,
        "errors": [],
    }

    try:
        htf_df = fetch_htf_bars(
            ticker=entry["ticker"], period_days=HTF_PERIOD_DAYS, end_date=entry["date"]
        )
        bias_result = compute_htf_bias(htf_df)
    except Exception as exc:
        result["errors"].append(f"htf_bias: {exc}")
        return result

    result["computed_bias"] = bias_result["bias"]
    result["confidence"] = bias_result["confidence"]
    result["bias_match"] = bias_result["bias"] == entry["expected_bias"]

    try:
        # See module docstring: current LTF data, not a point-in-time fetch -
        # a structural sanity check, not a temporally accurate backtest.
        ltf_df = fetch_ltf_bars(ticker=entry["ticker"])
        candidates = find_entry_candidates(ltf_df, bias_result)
        near_term = [c for c in candidates if c["proximity_tier"] == "near-term"]
        if near_term:
            top_candidate = near_term[0]
            result["top_near_term_candidate"] = top_candidate
            result["candidate_sanity"] = _check_candidate_sanity(
                top_candidate, DEFAULT_NEAR_TERM_THRESHOLD_PCT
            )
    except Exception as exc:
        result["errors"].append(f"ltf_candidates: {exc}")

    return result


def run_eval() -> dict:
    """Run the full eval harness over the golden dataset and write a summary.

    Returns:
        The summary dict - the same one written to eval/results/summary.json.
    """
    golden_entries = _load_golden_dataset(GOLDEN_DATASET_PATH)
    per_entry_results = [_evaluate_entry(entry) for entry in golden_entries]

    bias_total = len(per_entry_results)
    bias_correct = sum(1 for r in per_entry_results if r["bias_match"])

    checked = [r for r in per_entry_results if r["candidate_sanity"] is not None]
    sanity_passed = sum(1 for r in checked if r["candidate_sanity"]["passed"])
    skipped = bias_total - len(checked)

    summary = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_entries": bias_total,
        "bias_accuracy": {
            "correct": bias_correct,
            "total": bias_total,
            "rate": round(bias_correct / bias_total, 4) if bias_total else 0.0,
        },
        "candidate_sanity": {
            "passed": sanity_passed,
            "checked": len(checked),
            "skipped_no_near_term_candidate": skipped,
            "rate": round(sanity_passed / len(checked), 4) if checked else None,
        },
        "per_entry_results": per_entry_results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with SUMMARY_PATH.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    return summary


if __name__ == "__main__":
    result_summary = run_eval()

    bias_stats = result_summary["bias_accuracy"]
    print(
        f"Bias accuracy: {bias_stats['correct']}/{bias_stats['total']} "
        f"({bias_stats['rate']:.1%})"
    )

    sanity_stats = result_summary["candidate_sanity"]
    if sanity_stats["checked"]:
        print(
            f"Candidate sanity: {sanity_stats['passed']}/{sanity_stats['checked']} passed "
            f"({sanity_stats['rate']:.1%}), "
            f"{sanity_stats['skipped_no_near_term_candidate']} skipped (no near-term candidate)"
        )
    else:
        print("Candidate sanity: no entries had a near-term candidate to check")

    print(f"Full results written to {SUMMARY_PATH}")
