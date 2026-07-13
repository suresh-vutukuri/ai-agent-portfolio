"""Orchestrates HTF (1-hour) bias detection: structure + FVGs + order blocks -> one bias read."""

from __future__ import annotations

from typing import Literal, Optional, TypedDict

import pandas as pd

from fvg_ob_detection import Zone, detect_fvgs, detect_order_blocks
from structure_shifts import StructureEvent, detect_bos_choch
from swing_structure import detect_swings

# "Near" an FVG/OB zone without being inside it, expressed as a fraction of
# current price (e.g. 0.0015 == 0.15%, ~11 points on a ~7600 ES=F print).
PROXIMITY_PCT = 0.0015

BASE_CONFIDENCE_WITH_EVENT = 0.5
BASE_CONFIDENCE_NO_EVENT = 0.3
CHOCH_CONFIDENCE_PENALTY = 0.05
ALIGNED_ZONE_CONFIDENCE_BOOST = 0.15


class HTFBiasResult(TypedDict):
    """Structured HTF bias output consumed by the graph/entry agent."""

    bias: Literal["bullish", "bearish", "neutral"]
    last_structure_event: Optional[StructureEvent]
    active_fvgs: list[Zone]
    active_order_blocks: list[Zone]
    confidence: float


def _is_near_or_inside(price: float, top: float, bottom: float, proximity_pct: float) -> bool:
    """True if `price` sits inside [bottom, top], or within proximity_pct of either edge."""
    if bottom <= price <= top:
        return True
    tolerance = price * proximity_pct
    return (bottom - tolerance) <= price <= (top + tolerance)


def compute_htf_bias(df: pd.DataFrame) -> HTFBiasResult:
    """Compute the higher-timeframe (1H) directional bias for the bias agent.

    Combines four ICT-style structure reads into one summary:
      1. Swing highs/lows (swing_structure.detect_swings) - the raw pivots
         everything else is built from.
      2. The most recent BOS/CHoCH (structure_shifts.detect_bos_choch) - this
         is what actually sets the directional bias.
      3. Fair Value Gaps (fvg_ob_detection.detect_fvgs) - unmitigated
         imbalances price may be drawn back to.
      4. Order Blocks (fvg_ob_detection.detect_order_blocks) - unmitigated
         origin zones of prior impulsive moves.

    `confidence` starts from a baseline (higher if a structure event exists
    at all, slightly discounted for a CHoCH since a reversal is less
    confirmed than a continuation) and is boosted when the current price is
    sitting inside or near an active FVG/OB whose own direction agrees with
    the structural bias - i.e. multiple ICT signals lining up rather than
    just one.

    Args:
        df: 1H OHLCV DataFrame, as produced by htf_data.fetch_htf_bars.

    Returns:
        An HTFBiasResult: {bias, last_structure_event, active_fvgs,
        active_order_blocks, confidence}.
    """
    swings = detect_swings(df)
    structure_event = detect_bos_choch(df, swings)
    fvgs = detect_fvgs(df)
    order_blocks = detect_order_blocks(df)

    active_fvgs = [zone for zone in fvgs if not zone["mitigated"]]
    active_order_blocks = [zone for zone in order_blocks if not zone["mitigated"]]

    bias: Literal["bullish", "bearish", "neutral"]
    if structure_event is None:
        bias = "neutral"
        confidence = BASE_CONFIDENCE_NO_EVENT
    else:
        bias = structure_event["direction"]
        confidence = BASE_CONFIDENCE_WITH_EVENT
        if structure_event["type"] == "CHoCH":
            confidence -= CHOCH_CONFIDENCE_PENALTY

    if bias != "neutral" and not df.empty:
        last_price = float(df["Close"].iloc[-1])
        aligned_zones = [zone for zone in (active_fvgs + active_order_blocks) if zone["type"] == bias]
        for zone in aligned_zones:
            if _is_near_or_inside(last_price, zone["top"], zone["bottom"], PROXIMITY_PCT):
                confidence += ALIGNED_ZONE_CONFIDENCE_BOOST
                break  # one confirming zone is enough to award the boost

    confidence = max(0.0, min(1.0, confidence))

    return {
        "bias": bias,
        "last_structure_event": structure_event,
        "active_fvgs": active_fvgs,
        "active_order_blocks": active_order_blocks,
        "confidence": round(confidence, 3),
    }
