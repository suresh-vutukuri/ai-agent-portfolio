"""Shared helpers for fetching and validating OHLCV bars from yfinance."""

from __future__ import annotations

import functools
import time
import warnings
from typing import Callable, TypeVar

import pandas as pd

REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")

T = TypeVar("T", bound=pd.DataFrame)


def validate_ohlcv(df: pd.DataFrame, ticker: str, max_gap_hours: float = 72.0) -> None:
    """Validate an OHLCV DataFrame's shape and chronological continuity.

    Args:
        df: DataFrame expected to have a datetime index and OHLCV columns.
        ticker: Ticker symbol, used only to make error/warning messages useful.
        max_gap_hours: Largest gap between consecutive bars still considered a
            normal market closure. CME futures (ES=F, NQ=F) trade nearly 24x5,
            so most gaps are weekends (~49h); 72h leaves buffer for long
            weekends/holidays without masking a genuine data gap.

    Raises:
        ValueError: If the DataFrame is empty, missing required OHLCV
            columns, or its index isn't sorted in chronological order.
    """
    if df.empty:
        raise ValueError(f"No OHLCV data to validate for '{ticker}': DataFrame is empty.")

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"'{ticker}' data is missing required column(s): {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    if not df.index.is_monotonic_increasing:
        raise ValueError(f"'{ticker}' data is not sorted in chronological order.")

    gaps = df.index.to_series().diff().dropna()
    oversized = gaps[gaps > pd.Timedelta(hours=max_gap_hours)]
    if not oversized.empty:
        warnings.warn(
            f"'{ticker}' has {len(oversized)} gap(s) larger than {max_gap_hours}h "
            f"between bars (largest: {oversized.max()}). This can be normal around "
            f"holidays, but double-check if unexpected.",
            stacklevel=2,
        )


def retry_on_empty(
    retries: int = 3, delay_seconds: float = 2.0
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator that retries a yfinance-calling function if it returns an empty DataFrame.

    yfinance occasionally returns an empty frame on transient rate limits
    rather than raising an exception; this retries with a fixed delay before
    giving up and handing back whatever (possibly empty) result it last got.

    Args:
        retries: Number of attempts before giving up.
        delay_seconds: Seconds to wait between attempts.

    Returns:
        A decorator wrapping a zero-or-more-argument function that returns a
        pandas DataFrame.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> T:
            result: T | None = None
            for attempt in range(1, retries + 1):
                result = func(*args, **kwargs)
                if result is not None and not result.empty:
                    return result
                if attempt < retries:
                    time.sleep(delay_seconds)
            return result

        return wrapper

    return decorator
