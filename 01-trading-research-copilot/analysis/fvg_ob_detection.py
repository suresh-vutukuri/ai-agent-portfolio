"""Fair Value Gap (FVG) and Order Block (OB) detection - ICT-style imbalance and origin zones."""

from __future__ import annotations

from typing import Literal, TypedDict

import pandas as pd


class Zone(TypedDict):
    """A price zone (FVG or order block) with its mitigation status."""

    type: Literal["bullish", "bearish"]
    top: float
    bottom: float
    timestamp: pd.Timestamp
    mitigated: bool


def _mark_mitigation(zones: list[Zone], df: pd.DataFrame) -> None:
    """Flag each zone 'mitigated' once a later bar's range trades back into it.

    ICT concept: a zone (FVG or order block) is "mitigated"/"filled" once
    price returns and trades through it - at that point it's considered
    used up and no longer a meaningful reference level. Overlap is checked
    bar-by-bar: a later bar overlaps the zone if its Low is at/below the
    zone's top *and* its High is at/above the zone's bottom.

    Mutates each zone dict in place.
    """
    idx = df.index
    highs, lows = df["High"], df["Low"]
    for zone in zones:
        start_pos = idx.get_loc(zone["timestamp"]) + 1
        later_lows = lows.iloc[start_pos:]
        later_highs = highs.iloc[start_pos:]
        overlaps = (later_lows <= zone["top"]) & (later_highs >= zone["bottom"])
        zone["mitigated"] = bool(overlaps.any())


def detect_fvgs(df: pd.DataFrame, check_mitigation: bool = True) -> list[Zone]:
    """Detect Fair Value Gaps (FVGs): 3-candle price imbalances.

    ICT concept: an FVG is a 3-candle pattern where the middle candle moves so
    fast/far that it leaves a gap between candle 1 and candle 3 that never
    traded on candle 2 itself - an "imbalance" the market often revisits
    ("fills") later.
      - Bullish FVG: candle 1's High < candle 3's Low. Gap zone =
        [candle1.High, candle3.Low].
      - Bearish FVG: candle 1's Low > candle 3's High. Gap zone =
        [candle3.High, candle1.Low].

    Args:
        df: OHLCV DataFrame, chronologically sorted.
        check_mitigation: If True, mark each zone 'mitigated' once a later
            bar's range trades back into it (the gap has been "filled").

    Returns:
        A list of Zone dicts in chronological order, one per detected gap,
        each carrying its price boundaries, the timestamp of the confirming
        third candle, and whether it has since been mitigated.
    """
    zones: list[Zone] = []
    highs, lows = df["High"], df["Low"]
    idx = df.index
    n = len(df)

    for i in range(2, n):
        c1_high, c1_low = highs.iloc[i - 2], lows.iloc[i - 2]
        c3_high, c3_low = highs.iloc[i], lows.iloc[i]

        if c1_high < c3_low:
            zones.append(
                {
                    "type": "bullish",
                    "top": float(c3_low),
                    "bottom": float(c1_high),
                    "timestamp": idx[i],
                    "mitigated": False,
                }
            )
        elif c1_low > c3_high:
            zones.append(
                {
                    "type": "bearish",
                    "top": float(c1_low),
                    "bottom": float(c3_high),
                    "timestamp": idx[i],
                    "mitigated": False,
                }
            )

    if check_mitigation:
        _mark_mitigation(zones, df)

    return zones


def detect_order_blocks(
    df: pd.DataFrame,
    impulse_lookback: int = 10,
    impulse_multiplier: float = 2.0,
    check_mitigation: bool = True,
) -> list[Zone]:
    """Detect Order Blocks (OBs): the last opposing candle before a strong impulsive move.

    ICT concept: an order block marks where large ("smart money") orders
    likely accumulated just before price launched away impulsively, so the
    zone is treated as a level price may return to before continuing in the
    impulse's direction.
      - Bullish OB: the last down-close (bearish) candle immediately before a
        strong up-move. Zone = that candle's [Low, High].
      - Bearish OB: the last up-close (bullish) candle immediately before a
        strong down-move. Zone = that candle's [Low, High].

    "Strong impulsive move" is approximated here as a candle whose body
    (|Close - Open|) exceeds `impulse_multiplier` times the average body size
    of the preceding `impulse_lookback` candles - a simple stand-in for the
    displacement/momentum ICT traders look for, without requiring a full ATR
    or liquidity-sweep model.

    Args:
        df: OHLCV DataFrame, chronologically sorted.
        impulse_lookback: Number of preceding candles used to compute the
            average body-size baseline for what counts as "strong".
        impulse_multiplier: How many times the baseline average body size an
            impulse candle's body must exceed to qualify.
        check_mitigation: If True, mark each zone 'mitigated' once price
            trades back into it.

    Returns:
        A list of Zone dicts in chronological order.
    """
    zones: list[Zone] = []
    opens, closes = df["Open"], df["Close"]
    highs, lows = df["High"], df["Low"]
    idx = df.index
    n = len(df)

    bodies = (closes - opens).abs()

    for i in range(impulse_lookback + 1, n):
        baseline = bodies.iloc[i - impulse_lookback : i].mean()
        if baseline == 0:
            continue

        impulse_body = bodies.iloc[i]
        if impulse_body < impulse_multiplier * baseline:
            continue  # not a strong enough displacement to mark an origin

        impulse_is_bullish = closes.iloc[i] > opens.iloc[i]
        prior = i - 1
        prior_is_bearish = closes.iloc[prior] < opens.iloc[prior]
        prior_is_bullish = closes.iloc[prior] > opens.iloc[prior]

        if impulse_is_bullish and prior_is_bearish:
            zones.append(
                {
                    "type": "bullish",
                    "top": float(highs.iloc[prior]),
                    "bottom": float(lows.iloc[prior]),
                    "timestamp": idx[prior],
                    "mitigated": False,
                }
            )
        elif not impulse_is_bullish and prior_is_bullish:
            zones.append(
                {
                    "type": "bearish",
                    "top": float(highs.iloc[prior]),
                    "bottom": float(lows.iloc[prior]),
                    "timestamp": idx[prior],
                    "mitigated": False,
                }
            )

    if check_mitigation:
        _mark_mitigation(zones, df)

    return zones
