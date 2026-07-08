"""Demo: run a handful of example customer queries through the triage crew.

Usage:
    python demo.py

Requires only OPENAI_API_KEY in a .env file at the project root - no other
setup needed. Each example references a sample ID that actually exists in
tools/billing_tools.py, tools/tech_tools.py, or tools/returns_tools.py, so
the specialist tools return real (mock) data instead of a "not found"
fallback.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# agents/ is a plain sibling dir, not a package
sys.path.insert(0, str(PROJECT_ROOT / "agents"))

from run_triage import run_triage  # noqa: E402

# Sample IDs below are pulled straight from the mock data in tools/*.py:
#   billing_tools.py  -> INV-5002 (overdue invoice), CUST-2001 (payment history)
#   tech_tools.py     -> ERR_AUTH_401, auth-service (degraded status)
#   returns_tools.py  -> ORD-1001 (delivered, within refund window),
#                        ORD-1002 (delivered, past refund window)
EXAMPLE_QUERIES: list[str] = [
    "Can you check the status of invoice INV-5002? I thought I already paid it.",
    "Can you pull up my last few payments? My customer ID is CUST-2001.",
    "I'm getting error code ERR_AUTH_401 every time I try to log in - is auth-service having issues?",
    "I'd like to return order ORD-1001 - is it still eligible for a refund?",
    "It's been a while since I placed order ORD-1002 - can I still get my money back?",
]


def main() -> None:
    """Run each example query through the triage crew and print the result.

    Returns:
        None. Prints the query and the crew's final response for every
        entry in EXAMPLE_QUERIES, separated by a divider line.
    """
    for i, query in enumerate(EXAMPLE_QUERIES, start=1):
        print(f"{'=' * 70}")
        print(f"Example {i}/{len(EXAMPLE_QUERIES)}")
        print(f"Query: {query}")
        print("-" * 70)
        response = run_triage(query)
        print(f"Response: {response}")
        print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
