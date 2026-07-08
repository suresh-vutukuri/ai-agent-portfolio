"""CLI entry point: run a customer query through the multi-agent triage crew.

Usage:
    python run_triage.py "My invoice INV-5002 shows overdue but I paid it last week"
"""

from __future__ import annotations

import sys
from typing import List

from crew import TriageCrew
from handoff_logger import log_handoff, track_tool_calls


def run_triage(query: str) -> str:
    """Kick off the triage crew with a customer query and return the final response.

    Every run is logged to logs/handoff_log.jsonl via handoff_logger.log_handoff.
    Which specialist handled the query is captured via
    handoff_logger.track_tool_calls, which listens on CrewAI's event bus
    rather than Crew(step_callback=...) - see that function's docstring for
    why step_callback doesn't reliably fire for tool calls under native
    function calling.

    Args:
        query: The customer's raw support query.

    Returns:
        The manager-reviewed, finalized response text.
    """
    tool_calls: List[str] = []
    delegations: List[str] = []
    crew = TriageCrew().crew()

    with track_tool_calls(tool_calls, delegations):
        result = crew.kickoff(inputs={"query": query})

    response = result.raw
    log_handoff(query=query, tool_calls=tool_calls, response=response, delegations=delegations)

    return response


def main() -> None:
    """Grab the query from argv, run it through the crew, and print the answer.

    Returns:
        None.

    Raises:
        SystemExit: If no query was passed on the command line.
    """
    if len(sys.argv) < 2:
        print('Usage: python run_triage.py "<customer query>"')
        raise SystemExit(1)

    query = " ".join(sys.argv[1:])
    response = run_triage(query)
    print(response)


if __name__ == "__main__":
    main()
