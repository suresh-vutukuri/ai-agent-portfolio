"""CLI entry point: run the deep research crew on a topic and save a cited
markdown report.

Usage:
    python run_research.py --topic "The current state of small modular nuclear reactors"
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# agents/ is a plain sibling dir, not a package
sys.path.insert(0, str(PROJECT_ROOT / "agents"))

from crew import ResearchCrew  # noqa: E402


def run_research(topic: str) -> str:
    """Kick off the research crew for a topic and return the final markdown report.

    Args:
        topic: The research topic to investigate.

    Returns:
        The writer agent's final markdown report, with inline [Source: URL]
        citations.
    """
    crew = ResearchCrew().crew()
    result = crew.kickoff(inputs={"topic": topic})
    return result.raw


def _slugify(topic: str) -> str:
    """Turn a topic string into a filesystem-safe slug for the output filename.

    Args:
        topic: The raw research topic.

    Returns:
        A lowercase, hyphen-separated slug truncated to 60 characters.
    """
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in topic)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")[:60] or "report"


def save_report(topic: str, report: str, outputs_dir: Path = OUTPUTS_DIR) -> Path:
    """Save the markdown report to outputs/<timestamp>-<topic-slug>.md.

    Args:
        topic: The research topic (used to build the filename).
        report: The final markdown report text.
        outputs_dir: Directory to save the report in; created if missing.

    Returns:
        The path the report was written to.
    """
    outputs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = outputs_dir / f"{timestamp}-{_slugify(topic)}.md"
    path.write_text(report, encoding="utf-8")
    return path


def main() -> None:
    """Parse --topic, run the crew, print the report, and save it to outputs/.

    Returns:
        None.

    Raises:
        SystemExit: If --topic was not provided.
    """
    parser = argparse.ArgumentParser(description="Run the deep research crew on a topic.")
    parser.add_argument("--topic", required=True, help="The research topic to investigate.")
    args = parser.parse_args()

    report = run_research(args.topic)
    print(report)

    path = save_report(args.topic, report)
    print(f"\nSaved report to {path}")


if __name__ == "__main__":
    main()
