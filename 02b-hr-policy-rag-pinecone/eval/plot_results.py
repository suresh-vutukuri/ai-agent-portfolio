"""Bar chart of the aggregate eval scores, saved to eval/results/scorecard.png."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

EVAL_DIR = Path(__file__).resolve().parent
SUMMARY_PATH = EVAL_DIR / "results" / "summary.json"
OUTPUT_PATH = EVAL_DIR / "results" / "scorecard.png"

# (summary key, label, scale) - scale normalizes the 1-5 judge scores onto the
# same 0-1 bar height as the 0-1 metrics; bars are still annotated with the
# original-scale value.
_METRICS: list[tuple[str, str, float]] = [
    ("recall_at_k", "Recall@k", 1.0),
    ("groundedness", "Groundedness (/5)", 5.0),
    ("citation_accuracy", "Citation Accuracy", 1.0),
    ("answer_relevance", "Answer Relevance (/5)", 5.0),
    ("negative_control_refusal_accuracy", "Negative Control\nRefusal Accuracy", 1.0),
]


def load_summary(path: Path = SUMMARY_PATH) -> dict[str, Optional[float]]:
    """Load the aggregate scores written by run_eval.py.

    Args:
        path: Path to summary.json.

    Returns:
        The parsed summary dict.
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def plot_summary(summary: dict[str, Optional[float]], output_path: Path = OUTPUT_PATH) -> None:
    """Draw the aggregate scores as a bar chart and save it as a PNG.

    All bars share a 0-1 axis: the two 1-5 judge scores get divided by 5 for
    the bar height, but each bar is still annotated with its real-scale value.

    Args:
        summary: Aggregate summary dict, as produced by run_eval.summarize.
        output_path: Where to save the PNG; parent directories get created
            if they don't exist.

    Returns:
        None.
    """
    labels: list[str] = []
    heights: list[float] = []
    raw_labels: list[str] = []

    for key, label, scale in _METRICS:
        value = summary.get(key)
        labels.append(label)
        if value is None:
            heights.append(0.0)
            raw_labels.append("n/a")
        else:
            heights.append(value / scale)
            raw_labels.append(f"{value:.2f}" if scale == 1.0 else f"{value:.2f}/5")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(labels, heights, color="#4C72B0")

    for bar, raw_label in zip(bars, raw_labels):
        ax.annotate(
            raw_label,
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Normalized score (0-1)")
    ax.set_title(
        f"HR Policy RAG — Eval Scorecard "
        f"(n={summary.get('num_scored', 0)} scored, "
        f"{summary.get('num_negative_control', 0)} negative control)"
    )
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Load the eval summary and render the scorecard chart.

    Returns:
        None.
    """
    summary = load_summary()
    plot_summary(summary)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
