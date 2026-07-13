"""Demo: run the trading research copilot for ES=F and NQ=F, printing both research notes.

Usage:
    python demo.py

Requires only OPENAI_API_KEY (and optionally LANGCHAIN_TRACING_V2/
LANGCHAIN_API_KEY/LANGCHAIN_PROJECT for LangSmith tracing) in a .env file at
the project root - no other setup needed. Runs today's live market data
through the same LangGraph pipeline as graph/run_copilot.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

# build_graph.py/state.py live in graph/, a sibling folder to this script,
# not an installed package - make them importable the same flat way the rest
# of this project does (see graph/nodes.py for the same pattern).
sys.path.insert(0, str(Path(__file__).resolve().parent / "graph"))

from build_graph import build_graph  # noqa: E402
from state import CopilotState  # noqa: E402

DEMO_TICKERS: list[str] = ["ES=F", "NQ=F"]


def _run_for_ticker(ticker: str) -> None:
    """Run the copilot graph for one ticker and print its research note (or errors).

    Args:
        ticker: Futures ticker, e.g. 'ES=F' or 'NQ=F'.

    Returns:
        None. Prints any pipeline warnings/errors, then the synthesis output
        (or a fallback message if no output was produced).
    """
    initial_state: CopilotState = {
        "ticker": ticker,
        "htf_df": None,
        "ltf_df": None,
        "htf_bias": None,
        "ltf_candidates": None,
        "synthesis_output": None,
        "errors": [],
    }

    graph = build_graph()
    final_state = graph.invoke(initial_state)

    if final_state.get("errors"):
        print("--- Pipeline warnings/errors ---")
        for error in final_state["errors"]:
            print(f"  - {error}")
        print()

    output = final_state.get("synthesis_output")
    print(output if output else "No synthesis output was produced. See errors above.")


def main() -> None:
    """Run the copilot for every ticker in DEMO_TICKERS, printing each research note.

    Returns:
        None.
    """
    for i, ticker in enumerate(DEMO_TICKERS, start=1):
        print(f"{'=' * 70}")
        print(f"Example {i}/{len(DEMO_TICKERS)}: {ticker}")
        print("-" * 70)
        _run_for_ticker(ticker)
        print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
