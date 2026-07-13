"""Liquidity sweep (stop-hunt) detection.

ICT concept: resting liquidity sits just beyond swing points - stop-loss and
breakout orders cluster above swing highs ("buy-side liquidity") and below
swing lows ("sell-side liquidity"). A sweep (a.k.a. stop hunt / liquidity
grab) is when a candle's wick pierces through one of those levels -
triggering the resting orders - but the candle then closes back on the other
side of the level, rejecting the move. That rejection is read as a signal in
the *opposite* direction of the wick: sweeping buy-side liquidity above a high
and closing back below it is bearish (the breakout was a trap), while
sweeping sell-side liquidity below a low and closing back above it is
bullish.
"""

from __future__ import annotations

from typing import Literal, TypedDict

import pandas as pd


class SweepEvent(TypedDict):
    """A single liquidity sweep/stop-hunt event."""

    timestamp: pd.Timestamp
    level: float
    direction: Literal["bullish", "bearish"]
    swept_swing_timestamp: pd.Timestamp


def detect_liquidity_sweeps(
    df: pd.DataFrame, swings: pd.DataFrame, lookback: int = 20
) -> list[SweepEvent]:
    """Detect liquidity sweeps of recent swing highs/lows.

    For each confirmed swing high, scans forward up to `lookback` bars for a
    candle whose High pierces above the swing level but whose Close comes
    back below it - a buy-side liquidity sweep, read as bearish. Symmetrically,
    for each swing low, scans for a candle whose Low pierces below the level
    but closes back above it - a sell-side liquidity sweep, read as bullish.
    Only the first qualifying candle after each swing is recorded: once a
    level's resting liquidity has been swept, a later re-test of that same
    old level isn't a fresh sweep of it.

    Args:
        df: OHLCV DataFrame, chronologically sorted.
        swings: DataFrame from swing_structure.detect_swings(df), with
            'swing_high'/'swing_low' flag columns aligned to df's index.
        lookback: Max number of bars after a swing point to keep checking for
            a sweep of that swing's level. Bounds how "recent" a swing must
            be to still count as live, resting liquidity.

    Returns:
        A list of SweepEvent dicts, sorted chronologically by the timestamp
        of the sweeping candle.
    """
    idx = df.index
    highs, lows, closes = df["High"], df["Low"], df["Close"]
    n = len(df)

    events: list[SweepEvent] = []

    swing_highs = swings[swings["swing_high"]]
    for swing_ts, row in swing_highs.iterrows():
        level = float(row["High"])
        start_pos = idx.get_loc(swing_ts) + 1
        end_pos = min(start_pos + lookback, n)
        for pos in range(start_pos, end_pos):
            if highs.iloc[pos] > level and closes.iloc[pos] < level:
                events.append(
                    {
                        "timestamp": idx[pos],
                        "level": level,
                        "direction": "bearish",
                        "swept_swing_timestamp": swing_ts,
                    }
                )
                break  # level is swept; stop looking for further sweeps of it

    swing_lows = swings[swings["swing_low"]]
    for swing_ts, row in swing_lows.iterrows():
        level = float(row["Low"])
        start_pos = idx.get_loc(swing_ts) + 1
        end_pos = min(start_pos + lookback, n)
        for pos in range(start_pos, end_pos):
            if lows.iloc[pos] < level and closes.iloc[pos] > level:
                events.append(
                    {
                        "timestamp": idx[pos],
                        "level": level,
                        "direction": "bullish",
                        "swept_swing_timestamp": swing_ts,
                    }
                )
                break

    events.sort(key=lambda event: event["timestamp"])
    return events
