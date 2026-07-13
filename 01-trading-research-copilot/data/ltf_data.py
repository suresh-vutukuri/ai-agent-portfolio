"""Fetches lower-timeframe (5-minute) OHLCV bars for the entry agent."""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from data_utils import retry_on_empty, validate_ohlcv

ET_TZ = "America/New_York"

# yfinance caps 5-minute intraday history at 60 days; 55 leaves a small buffer
# so a slow request (or timezone rounding at the edge of the window) doesn't
# tip over the limit and get rejected.
MAX_INTRADAY_DAYS = 60


def fetch_ltf_bars(ticker: str = "ES=F", period_days: int = 55) -> pd.DataFrame:
    """Fetch 5-minute OHLCV bars for the lower-timeframe (LTF) entry agent.

    Args:
        ticker: yfinance ticker symbol, e.g. 'ES=F' (E-mini S&P 500) or
            'NQ=F' (E-mini Nasdaq-100).
        period_days: How many trailing days of history to request. Must stay
            under yfinance's 60-day limit for 5-minute intraday data; the
            default of 55 leaves a small buffer.

    Returns:
        A cleaned DataFrame with columns Open, High, Low, Close, Volume, a
        tz-aware datetime index (US/Eastern), sorted chronologically with no
        NaN rows or duplicate timestamps.

    Raises:
        ValueError: If period_days exceeds yfinance's intraday limit, if
            yfinance returns no data (invalid ticker, or the requested window
            has no trading bars, e.g. a market holiday), or the resulting
            data fails validation.
    """
    if period_days > MAX_INTRADAY_DAYS:
        raise ValueError(
            f"period_days={period_days} exceeds yfinance's {MAX_INTRADAY_DAYS}-day "
            f"limit for 5-minute intraday data. Use {MAX_INTRADAY_DAYS} or fewer."
        )

    raw = _download(ticker, period_days)

    if raw.empty:
        raise ValueError(
            f"yfinance returned no 5m bars for ticker '{ticker}' over the last "
            f"{period_days} day(s). Check that the ticker is valid (e.g. 'ES=F', "
            f"'NQ=F') and that the requested window isn't entirely a market "
            f"holiday or closure."
        )

    df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(ET_TZ)
    else:
        df.index = df.index.tz_convert(ET_TZ)

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    validate_ohlcv(df, ticker)

    return df


@retry_on_empty(retries=3, delay_seconds=2.0)
def _download(ticker: str, period_days: int) -> pd.DataFrame:
    return yf.Ticker(ticker).history(period=f"{period_days}d", interval="5m")
