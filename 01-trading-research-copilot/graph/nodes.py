"""LangGraph node functions wrapping the data-fetch and analysis modules.

Each node reads what it needs from the shared state and returns only the
keys it updates - LangGraph merges that partial dict back into the overall
state. Every node wraps its work in try/except and appends a message to
state['errors'] on failure instead of raising, so one bad fetch or a rate
limit doesn't crash the whole pipeline. Downstream nodes check for missing
upstream inputs and skip their own work gracefully rather than raising.
"""

from __future__ import annotations

import sys
from pathlib import Path

# data/ and analysis/ are sibling project folders, not installed packages -
# make their modules importable the same flat way the rest of this project
# does (see e.g. htf_bias.py importing swing_structure.py directly).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _subdir in ("data", "analysis"):
    _path = str(_PROJECT_ROOT / _subdir)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from htf_bias import compute_htf_bias  # noqa: E402
from htf_data import fetch_htf_bars  # noqa: E402
from ltf_candidates import find_entry_candidates  # noqa: E402
from ltf_data import fetch_ltf_bars  # noqa: E402

from state import CopilotState


def fetch_htf_node(state: CopilotState) -> dict:
    """Fetch 1H OHLCV bars for state['ticker'] into state['htf_df'].

    Args:
        state: Current graph state; only 'ticker' is required.

    Returns:
        {'htf_df': DataFrame} on success, or {'errors': [...]} with this
        failure appended (e.g. invalid ticker, yfinance returning no data).
    """
    try:
        htf_df = fetch_htf_bars(ticker=state["ticker"])
        return {"htf_df": htf_df}
    except Exception as exc:
        return {"errors": state["errors"] + [f"fetch_htf_node: {exc}"]}


def compute_htf_bias_node(state: CopilotState) -> dict:
    """Compute the HTF bias from state['htf_df'] into state['htf_bias'].

    ICT concept: this is where the pipeline settles on an overall directional
    bias (bullish/bearish/neutral) from 1H swing structure, BOS/CHoCH, FVGs,
    and order blocks - see analysis/htf_bias.py for the underlying logic.

    Args:
        state: Current graph state; requires 'htf_df' to have been set by
            fetch_htf_node.

    Returns:
        {'htf_bias': dict} on success. If 'htf_df' is missing (the fetch
        step failed) or bias computation raises, appends to
        {'errors': [...]} instead of crashing.
    """
    if state.get("htf_df") is None:
        return {
            "errors": state["errors"]
            + ["compute_htf_bias_node: skipped, no htf_df (fetch_htf_node likely failed)"]
        }
    try:
        bias = compute_htf_bias(state["htf_df"])
        return {"htf_bias": bias}
    except Exception as exc:
        return {"errors": state["errors"] + [f"compute_htf_bias_node: {exc}"]}


def fetch_ltf_node(state: CopilotState) -> dict:
    """Fetch 5-min OHLCV bars for state['ticker'] into state['ltf_df'].

    Args:
        state: Current graph state; only 'ticker' is required.

    Returns:
        {'ltf_df': DataFrame} on success, or {'errors': [...]} with this
        failure appended.
    """
    try:
        ltf_df = fetch_ltf_bars(ticker=state["ticker"])
        return {"ltf_df": ltf_df}
    except Exception as exc:
        return {"errors": state["errors"] + [f"fetch_ltf_node: {exc}"]}


def find_ltf_candidates_node(state: CopilotState) -> dict:
    """Find ranked LTF entry candidates aligned with the HTF bias.

    ICT concept: this is the "trade with the higher timeframe" step - LTF
    FVGs/order blocks only survive as candidates when their direction matches
    state['htf_bias']['bias'], then get ranked by whether a liquidity sweep
    preceded them - see analysis/ltf_candidates.py.

    Args:
        state: Current graph state; requires 'ltf_df' and 'htf_bias' to have
            been set by earlier nodes.

    Returns:
        {'ltf_candidates': list[dict]} on success. If 'ltf_df' or 'htf_bias'
        is missing, or candidate detection raises, appends to
        {'errors': [...]} instead of crashing.
    """
    if state.get("ltf_df") is None or state.get("htf_bias") is None:
        return {
            "errors": state["errors"]
            + ["find_ltf_candidates_node: skipped, missing ltf_df or htf_bias"]
        }
    try:
        candidates = find_entry_candidates(state["ltf_df"], state["htf_bias"])
        return {"ltf_candidates": candidates}
    except Exception as exc:
        return {"errors": state["errors"] + [f"find_ltf_candidates_node: {exc}"]}
