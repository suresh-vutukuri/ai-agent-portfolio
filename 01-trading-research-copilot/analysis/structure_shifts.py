"""Break of Structure (BOS) / Change of Character (CHoCH) detection.

ICT market-structure concepts:
  - BOS (Break of Structure): price closes beyond the most recent swing point
    *in the direction of the prevailing trend* - e.g. a new higher high in an
    uptrend, or a new lower low in a downtrend. Confirms the trend is
    continuing.
  - CHoCH (Change of Character): price closes beyond the most recent swing
    point *against* the prevailing trend - e.g. price breaks below the last
    higher low in an uptrend. Flags a potential trend reversal.
"""

from __future__ import annotations

from typing import Literal, Optional, TypedDict

import pandas as pd


class StructureEvent(TypedDict):
    """A single BOS or CHoCH event."""

    type: Literal["BOS", "CHoCH"]
    direction: Literal["bullish", "bearish"]
    price: float
    timestamp: pd.Timestamp


def _ordered_swing_points(swings: pd.DataFrame) -> list[tuple[pd.Timestamp, float, str]]:
    """Collapse the swing-flag DataFrame into a strictly alternating high/low sequence.

    Fractal swing detection (swing_structure.detect_swings) can flag two
    swing highs in a row with no swing low in between (e.g. a stair-step up
    move). Market-structure analysis only cares about alternating pivots, so
    when consecutive swings share a type we keep the more extreme one
    (the higher high / the lower low) and drop the other - it wasn't a
    meaningful structural pivot.

    Args:
        swings: DataFrame produced by detect_swings, with 'swing_high' /
            'swing_low' boolean columns and High/Low price columns.

    Returns:
        A chronological list of (timestamp, price, kind) tuples, `kind` being
        'high' or 'low', guaranteed to alternate between the two.
    """
    points: list[tuple[pd.Timestamp, float, str]] = []
    for ts, row in swings.iterrows():
        if row.get("swing_high"):
            points.append((ts, float(row["High"]), "high"))
        if row.get("swing_low"):
            points.append((ts, float(row["Low"]), "low"))
    points.sort(key=lambda p: p[0])

    alternating: list[tuple[pd.Timestamp, float, str]] = []
    for point in points:
        if alternating and alternating[-1][2] == point[2]:
            # Same type as the last kept swing - keep whichever is the more
            # extreme pivot, discard the other.
            if point[2] == "high" and point[1] > alternating[-1][1]:
                alternating[-1] = point
            elif point[2] == "low" and point[1] < alternating[-1][1]:
                alternating[-1] = point
        else:
            alternating.append(point)
    return alternating


def detect_bos_choch(df: pd.DataFrame, swings: pd.DataFrame) -> Optional[StructureEvent]:
    """Find the most recent Break of Structure (BOS) or Change of Character (CHoCH).

    Walks the bars in chronological order, tracking the most recent
    not-yet-broken swing high and swing low. Whenever a bar's Close breaks
    past one of those levels, that's a structure event:
      - Breaking the swing high while flat/bullish (or the swing low while
        flat/bearish) is a BOS - the existing trend continuing.
      - Breaking the swing high while the prevailing trend is bearish (or the
        swing low while bullish) is a CHoCH - a potential reversal.
    Once a level is broken it's "consumed" (cleared) so the same break can't
    fire twice; the trend flips to match the direction of the break.

    Args:
        df: OHLCV DataFrame (as produced by htf_data.fetch_htf_bars), used
            for its Close prices to detect the actual structure breaks.
        swings: The DataFrame returned by detect_swings(df), with
            'swing_high'/'swing_low' flag columns aligned to df's index.

    Returns:
        A StructureEvent for the most recent BOS/CHoCH, or None if the swing
        history is too short to establish any break yet.
    """
    pivots = _ordered_swing_points(swings)
    if len(pivots) < 2:
        return None

    trend: Optional[Literal["bullish", "bearish"]] = None
    last_swing_high: Optional[tuple[pd.Timestamp, float]] = None
    last_swing_low: Optional[tuple[pd.Timestamp, float]] = None
    last_event: Optional[StructureEvent] = None

    pivot_iter = iter(pivots)
    next_pivot = next(pivot_iter, None)

    for ts, row in df.iterrows():
        # Register any swing pivot confirmed as of this bar before checking
        # for a break, so a level is available to be broken by later bars.
        while next_pivot is not None and next_pivot[0] <= ts:
            p_ts, p_price, p_kind = next_pivot
            if p_kind == "high":
                last_swing_high = (p_ts, p_price)
            else:
                last_swing_low = (p_ts, p_price)
            next_pivot = next(pivot_iter, None)

        close = float(row["Close"])

        if last_swing_high is not None and close > last_swing_high[1]:
            event_type: Literal["BOS", "CHoCH"] = "CHoCH" if trend == "bearish" else "BOS"
            last_event = {
                "type": event_type,
                "direction": "bullish",
                "price": last_swing_high[1],
                "timestamp": ts,
            }
            trend = "bullish"
            last_swing_high = None  # consumed - wait for the next swing high to form
        elif last_swing_low is not None and close < last_swing_low[1]:
            event_type = "CHoCH" if trend == "bullish" else "BOS"
            last_event = {
                "type": event_type,
                "direction": "bearish",
                "price": last_swing_low[1],
                "timestamp": ts,
            }
            trend = "bearish"
            last_swing_low = None  # consumed - wait for the next swing low to form

    return last_event
