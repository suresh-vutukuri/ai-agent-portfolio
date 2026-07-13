"""Fetches higher-timeframe (1-hour) OHLCV bars for the bias agent."""

from __future__ import annotations

from typing import Optional, Union

import pandas as pd
import yfinance as yf

from data_utils import retry_on_empty, validate_ohlcv

ET_TZ = "America/New_York"

DateLike = Union[str, pd.Timestamp]


@retry_on_empty(retries=3, delay_seconds=2.0)
def _download_period(ticker: str, period_days: int) -> pd.DataFrame:
    return yf.Ticker(ticker).history(period=f"{period_days}d", interval="1h")


@retry_on_empty(retries=3, delay_seconds=2.0)
def _download_range(ticker: str, start: str, end: str) -> pd.DataFrame:
    return yf.Ticker(ticker).history(start=start, end=end, interval="1h")


def fetch_htf_bars(
    ticker: str = "ES=F",
    period_days: int = 60,
    end_date: Optional[DateLike] = None,
) -> pd.DataFrame:
    """Fetch 1-hour OHLCV bars for the higher-timeframe (HTF) bias agent.

    Args:
        ticker: yfinance ticker symbol, e.g. 'ES=F' (E-mini S&P 500) or
            'NQ=F' (E-mini Nasdaq-100).
        period_days: How many trailing days of history to request, counting
            back from `end_date` (or from now, if `end_date` is None).
        end_date: If given, fetch a historical window ending at end-of-day
            (23:59:59 US/Eastern) on this date instead of the most recent
            `period_days`. Accepts an ISO date string (e.g. '2026-06-15') or
            a pandas Timestamp. This is what lets the eval harness reproduce
            the HTF bias "as of" a historical date without leaking bars from
            after that cutoff into the computation.

    Returns:
        A cleaned DataFrame with columns Open, High, Low, Close, Volume, a
        tz-aware datetime index (US/Eastern), sorted chronologically with no
        NaN rows or duplicate timestamps, and (if `end_date` was given) no
        bars after end-of-day on `end_date`.

    Raises:
        ValueError: If yfinance returns no data (invalid ticker, or the
            requested window has no trading bars, e.g. a market holiday), if
            trimming to `end_date` leaves nothing, or the resulting data
            fails validation.
    """
    cutoff: Optional[pd.Timestamp] = None

    if end_date is None:
        raw = _download_period(ticker, period_days)
    else:
        cutoff = pd.Timestamp(end_date)
        cutoff = cutoff.tz_localize(ET_TZ) if cutoff.tz is None else cutoff.tz_convert(ET_TZ)
        cutoff = cutoff.normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)
        start = (cutoff - pd.Timedelta(days=period_days)).strftime("%Y-%m-%d")
        # yfinance's `end` is exclusive-ish, so request one extra day and
        # trim to the exact cutoff explicitly below rather than relying on it.
        end = (cutoff.normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        raw = _download_range(ticker, start, end)

    if raw.empty:
        window_note = f" ending {end_date}" if end_date is not None else ""
        raise ValueError(
            f"yfinance returned no 1h bars for ticker '{ticker}' for the requested "
            f"{period_days}-day window{window_note}. Check that the ticker is valid "
            f"(e.g. 'ES=F', 'NQ=F') and that the requested window isn't entirely a "
            f"market holiday or closure."
        )

    df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(ET_TZ)
    else:
        df.index = df.index.tz_convert(ET_TZ)

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    if cutoff is not None:
        df = df[df.index <= cutoff]
        if df.empty:
            raise ValueError(
                f"No 1h bars for '{ticker}' remained after trimming to "
                f"end_date={end_date} (cutoff {cutoff}). The requested window "
                f"may fall outside the data yfinance actually returned."
            )

    validate_ohlcv(df, ticker)

    return df
