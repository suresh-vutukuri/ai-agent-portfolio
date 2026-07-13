"""Orchestrates LTF (5-minute) entry-candidate detection.

ICT approach: trade *with* the higher-timeframe bias, looking for lower-
timeframe zones (FVGs/order blocks) in that same direction - ideally ones a
liquidity sweep set up just before, the classic "sweep liquidity, then reverse
into a discount/premium zone" entry model. This module only ranks
historical-context candidates for research; it does not size, place, or
execute anything (see the portfolio README for that disclaimer).
"""

from __future__ import annotations

from typing import Literal, Optional, TypedDict

import pandas as pd

from fvg_ob_detection import Zone, detect_fvgs, detect_order_blocks
from liquidity_sweeps import SweepEvent, detect_liquidity_sweeps
from structure_shifts import StructureEvent, detect_bos_choch
from swing_structure import detect_swings

# How many bars before a zone formed a same-direction sweep still counts as
# having "set up" that zone, rather than being an unrelated older event.
SWEEP_LOOKBACK_BARS = 10

# A zone within this % of current price is "near-term" - close enough that
# price could reach it soon, making it an actionable entry read right now.
# Anything further out is still a legitimate unmitigated level, just further
# away ("standing-reference") - more useful as a future target/reference than
# something to act on immediately. ~1-2% is a reasonable default for ES/NQ
# intraday ranges; callers can override per-instrument/session volatility.
DEFAULT_NEAR_TERM_THRESHOLD_PCT = 1.5

BASE_STRENGTH = 1.0
# Must exceed SWEPT_LIQUIDITY_STRENGTH_BONUS so that even a near-term zone
# with no sweep outranks a standing-reference zone that does have one - a
# distant zone shouldn't win just because it also had a liquidity sweep.
NEAR_TERM_STRENGTH_BONUS = 1.5
SWEPT_LIQUIDITY_STRENGTH_BONUS = 1.0


class PriceDistance(TypedDict):
    """Distance from current price to a zone's nearest edge."""

    points: float
    pct: float


class EntryCandidate(TypedDict):
    """A single ranked LTF entry candidate zone."""

    type: Literal["bullish", "bearish"]
    zone_kind: Literal["fvg", "order_block"]
    top: float
    bottom: float
    timestamp: pd.Timestamp
    swept_liquidity_before: bool
    distance_from_current_price: PriceDistance
    proximity_tier: Literal["near-term", "standing-reference"]
    strength: float
    ltf_structure_event: Optional[StructureEvent]


def _swept_liquidity_before(
    zone: Zone, sweeps: list[SweepEvent], df: pd.DataFrame, window_bars: int
) -> bool:
    """True if a same-direction sweep happened shortly before this zone formed.

    ICT concept: a liquidity sweep (grabbing resting stops) is what often
    fuels the impulsive move that leaves behind an FVG or order block - so a
    zone preceded by a matching sweep is a better-supported setup than one
    that appeared without one.
    """
    idx = df.index
    zone_pos = idx.get_loc(zone["timestamp"])
    for sweep in sweeps:
        if sweep["direction"] != zone["type"]:
            continue
        sweep_pos = idx.get_loc(sweep["timestamp"])
        if sweep_pos <= zone_pos and (zone_pos - sweep_pos) <= window_bars:
            return True
    return False


def _distance_from_current_price(current_price: float, top: float, bottom: float) -> PriceDistance:
    """Distance (points and %) from current price to a zone's nearest edge.

    Zero if current price is already trading inside [bottom, top] - the
    useful question is how far price has to move to reach the zone at all,
    not its distance to some interior point.
    """
    if bottom <= current_price <= top:
        points = 0.0
    elif current_price < bottom:
        points = bottom - current_price
    else:
        points = current_price - top
    pct = (points / current_price) * 100 if current_price else 0.0
    return {"points": round(points, 4), "pct": round(pct, 4)}


def find_entry_candidates(
    df: pd.DataFrame,
    htf_bias: dict,
    near_term_threshold_pct: float = DEFAULT_NEAR_TERM_THRESHOLD_PCT,
) -> list[EntryCandidate]:
    """Find and rank LTF (5-min) entry candidates aligned with the HTF bias.

    ICT approach: only look for entries *with* the higher-timeframe trend - a
    bullish HTF bias means only bullish ("discount") zones are worth
    considering as potential long entries, and a bearish bias means only
    bearish ("premium") zones for potential shorts.

    Among those, candidates are ranked on two factors:
      - Proximity: a zone within `near_term_threshold_pct` of the current
        price is "near-term" - close enough to be an actionable read right
        now. Further-out unmitigated zones are "standing-reference": still
        legitimate levels, but more useful as future targets/context than as
        an immediate entry, so they rank below every near-term candidate
        regardless of any other factor.
      - Liquidity sweeps: within a proximity tier, zones that a liquidity
        sweep set up just before they formed rank higher, since the sweep
        suggests the move that created the zone was fueled by a genuine
        stop-run rather than a random, shallow push.

    Args:
        df: 5-min OHLCV DataFrame, as produced by ltf_data.fetch_ltf_bars.
        htf_bias: The dict returned by htf_bias.compute_htf_bias() for the
            same instrument - only its 'bias' key is used, to filter which
            direction of LTF zone is worth considering at all.
        near_term_threshold_pct: Max distance from current price (as a % of
            current price) for a zone to be classified "near-term" rather
            than "standing-reference".

    Returns:
        A list of EntryCandidate dicts, strongest first (all near-term
        candidates before any standing-reference one). Empty if the HTF bias
        is 'neutral', if no LTF zones align with the bias, or if `df` is
        empty.
    """
    bias_direction = htf_bias.get("bias", "neutral")
    if bias_direction not in ("bullish", "bearish") or df.empty:
        return []  # no directional edge from HTF, or nothing to measure against

    current_price = float(df["Close"].iloc[-1])

    swings = detect_swings(df)
    ltf_structure_event = detect_bos_choch(df, swings)
    fvgs = detect_fvgs(df)
    order_blocks = detect_order_blocks(df)
    sweeps = detect_liquidity_sweeps(df, swings)

    tagged_zones: list[tuple[Zone, Literal["fvg", "order_block"]]] = [
        (zone, "fvg") for zone in fvgs
    ] + [(zone, "order_block") for zone in order_blocks]

    candidates: list[EntryCandidate] = []
    for zone, zone_kind in tagged_zones:
        # Only unmitigated ("live") zones are still valid reference levels,
        # and only ones matching the HTF bias direction are worth trading with.
        if zone["mitigated"] or zone["type"] != bias_direction:
            continue

        distance = _distance_from_current_price(current_price, zone["top"], zone["bottom"])
        proximity_tier: Literal["near-term", "standing-reference"] = (
            "near-term" if distance["pct"] <= near_term_threshold_pct else "standing-reference"
        )
        swept_before = _swept_liquidity_before(zone, sweeps, df, SWEEP_LOOKBACK_BARS)

        strength = BASE_STRENGTH
        if proximity_tier == "near-term":
            strength += NEAR_TERM_STRENGTH_BONUS
        if swept_before:
            strength += SWEPT_LIQUIDITY_STRENGTH_BONUS

        candidates.append(
            {
                "type": zone["type"],
                "zone_kind": zone_kind,
                "top": zone["top"],
                "bottom": zone["bottom"],
                "timestamp": zone["timestamp"],
                "swept_liquidity_before": swept_before,
                "distance_from_current_price": distance,
                "proximity_tier": proximity_tier,
                "strength": strength,
                "ltf_structure_event": ltf_structure_event,
            }
        )

    # Strongest first; break ties by recency since a fresher zone is more
    # likely to still be relevant to current price action. The strength
    # weighting above already guarantees every near-term candidate sorts
    # ahead of every standing-reference one.
    candidates.sort(key=lambda candidate: (candidate["strength"], candidate["timestamp"]), reverse=True)
    return candidates
