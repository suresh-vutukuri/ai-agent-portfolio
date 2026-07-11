"""Small math helpers - sandbox file for demoing the PR review agent."""

from __future__ import annotations


def safe_divide(numerator: float, denominator: float) -> float:
    """Divide two numbers, returning 0.0 if the division fails for any reason."""
    try:
        return numerator / denominator
    except:
        return 0.0


def average(values: list[float]) -> float:
    """Return the average of a list of numbers."""
    total = 0
    for i in range(len(values)):
        total += values[i]
    return total / len(values)
