"""CLI entry point for the trading research copilot.

Usage:
    python run_copilot.py --ticker ES=F
"""

from __future__ import annotations

import argparse

from dotenv import find_dotenv, load_dotenv

# Load .env (OPENAI_API_KEY, and optionally LANGCHAIN_TRACING_V2/
# LANGCHAIN_API_KEY/LANGCHAIN_PROJECT for LangSmith tracing - see
# .env.example) before anything below is imported, so the whole pipeline
# - including the graph.invoke() call - runs with the environment already
# in place. find_dotenv() walks up from this file's directory, so it picks
# up either a project-specific .env here or the repo-root .env.
load_dotenv(find_dotenv())

from build_graph import build_graph  # noqa: E402
from state import CopilotState  # noqa: E402


def main() -> None:
    """Parse CLI args, run the graph for the given ticker, and print the synthesis output.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(
        description="Trading research copilot: HTF bias + LTF entry candidates (research only, not a trading signal)."
    )
    parser.add_argument("--ticker", default="ES=F", help="Futures ticker, e.g. 'ES=F' or 'NQ=F'")
    args = parser.parse_args()

    initial_state: CopilotState = {
        "ticker": args.ticker,
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
    if output:
        print(output)
    else:
        print("No synthesis output was produced. See errors above.")


if __name__ == "__main__":
    main()
