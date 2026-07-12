"""Demo: run a few example research topics through the deep research crew.

Usage:
    python demo.py

Requires only OPENAI_API_KEY and TAVILY_API_KEY in a .env file at the
project root - no other setup needed. Each report is printed and saved to
outputs/, same as running run_research.py directly.
"""

from __future__ import annotations

from run_research import run_research, save_report

EXAMPLE_TOPICS: list[str] = [
    "The current state of small modular nuclear reactors",
    "How large language model context windows have scaled since 2023",
    "The economics of vertical farming at commercial scale",
]


def main() -> None:
    """Run each example topic through the research crew, printing and saving each report.

    Returns:
        None. Prints every topic and its resulting report, and saves each
        report to outputs/, separated by a divider line.
    """
    for i, topic in enumerate(EXAMPLE_TOPICS, start=1):
        print(f"{'=' * 70}")
        print(f"Example {i}/{len(EXAMPLE_TOPICS)}")
        print(f"Topic: {topic}")
        print("-" * 70)
        report = run_research(topic)
        path = save_report(topic, report)
        print(report)
        print(f"\nSaved report to {path}")
        print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
