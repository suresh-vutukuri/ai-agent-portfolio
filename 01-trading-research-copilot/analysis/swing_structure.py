"""Swing high/low (fractal) detection for HTF market-structure analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def detect_swings(df: pd.DataFrame, lookback: int = 3) -> pd.DataFrame:
    """Detect swing highs and swing lows using a fractal-style rule.

    ICT/price-action concept: a swing high is a local peak - a bar whose High
    is greater than the High of `lookback` bars on *both* sides of it (and
    symmetrically, a swing low is a local trough for Lows). These pivots are
    the raw building blocks of market-structure analysis (see
    structure_shifts.detect_bos_choch), since BOS/CHoCH events are defined
    relative to the most recent swing points.

    Because a swing needs bars on both sides to confirm it, the most recent
    `lookback` bars can never be confirmed as swings yet (there isn't enough
    future price action) - they're always flagged False, same as the first
    `lookback` bars (no history on their left side).

    Args:
        df: OHLCV DataFrame, chronologically sorted (as from
            htf_data.fetch_htf_bars).
        lookback: Number of bars required on each side for a pivot to count
            as a swing. Higher values find fewer, more significant swings.

    Returns:
        A copy of `df` with two added boolean columns:
          - 'swing_high': True where the bar is a confirmed swing high.
          - 'swing_low': True where the bar is a confirmed swing low.
    """
    out = df.copy()
    n = len(out)
    highs = out["High"].to_numpy()
    lows = out["Low"].to_numpy()

    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)

    for i in range(lookback, n - lookback):
        # A swing high/low must be the strict extreme among all bars in the
        # window on *both* sides - ties don't count as a confirmed pivot.
        left_highs = highs[i - lookback : i]
        right_highs = highs[i + 1 : i + lookback + 1]
        if highs[i] > left_highs.max() and highs[i] > right_highs.max():
            swing_high[i] = True

        left_lows = lows[i - lookback : i]
        right_lows = lows[i + 1 : i + lookback + 1]
        if lows[i] < left_lows.min() and lows[i] < right_lows.min():
            swing_low[i] = True

    out["swing_high"] = swing_high
    out["swing_low"] = swing_low
    return out
