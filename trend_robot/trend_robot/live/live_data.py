"""As-of price loading for the live runner.

Thin wrappers over the research data layer that fetch the price panel an order
plan should be built from, *as of* a given date. The functions REUSE the
research-side helpers (:func:`run_research._date_window` and
:func:`run_research._load_prices`) so the live path observes exactly the same
data contract, source-selection logic and synthetic fallback as the backtest.

Anti-look-ahead
---------------
:func:`prices_asof` GUARANTEES the returned frame has NO rows after ``asof``:
the order plan is built only from closes up to and including ``asof`` (mirroring
the backtest's ``shift(1)`` discipline), so orders are what you would place
*after* ``asof`` (next session).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from trend_robot.config import Config

# Reuse the research helpers verbatim so live/research data parity holds.
import run_research

__all__ = ["prices_asof", "last_prices"]


def prices_asof(
    cfg: Config,
    asof: str,
    cache_dir: str | Path,
    *,
    prefer_yfinance: bool = False,
    history_years: int = 15,
) -> tuple[pd.DataFrame, str]:
    """Load the adjusted-close panel as of ``asof`` (no rows after ``asof``).

    The lookback window is computed via :func:`run_research._date_window`
    (ending at ``asof``) and the panel is fetched via
    :func:`run_research._load_prices`, which uses the deterministic synthetic
    provider by default and prefers cached Yahoo only when ``prefer_yfinance``
    is set (falling back to synthetic if the download is unusable -- so the
    dry-run keeps working offline).

    Parameters
    ----------
    cfg:
        Typed configuration (universe, lookbacks, seed).
    asof:
        Inclusive "as of" date (``"YYYY-MM-DD"``). The returned frame contains
        no rows strictly after this date.
    cache_dir:
        Directory for the parquet price cache.
    prefer_yfinance:
        Prefer live cached Yahoo prices (``True``) or go straight to the
        deterministic synthetic provider (``False``, the default).
    history_years:
        Calendar years of history to request before ``asof``.

    Returns
    -------
    tuple[pandas.DataFrame, str]
        The price panel (rows <= ``asof``) and the data-source label
        (``"yfinance"`` / ``"synthetic"``).
    """
    start, end = run_research._date_window(history_years, asof)
    prices, data_source = run_research._load_prices(
        cfg, start, end, cache_dir, prefer_yfinance=prefer_yfinance
    )

    # Hard guarantee: never expose any row after asof, regardless of provider.
    asof_ts = pd.Timestamp(asof)
    prices = prices.loc[prices.index <= asof_ts]

    # Trim any TRAILING all-NaN rows. A brand-new bar (e.g. today's date) is
    # often present in the source with no published close yet; that phantom tail
    # row would otherwise become the "latest" bar and yield an all-NaN signal
    # row -> a spurious FLAT book (the live runner would wrongly go to cash).
    # Interior NaN gaps (the explicit data-contract gaps) are preserved.
    has_data = prices.notna().any(axis=1)
    if has_data.any():
        last_real = has_data[has_data].index[-1]
        prices = prices.loc[:last_real]
    return prices, data_source


def last_prices(prices: pd.DataFrame) -> pd.Series:
    """Return the last valid (non-NaN) close for each column.

    Parameters
    ----------
    prices:
        Adjusted-close panel (rows=dates, cols=symbols), possibly with trailing
        ``NaN`` gaps per column.

    Returns
    -------
    pandas.Series
        Indexed by symbol, the most recent non-NaN close per column. Columns
        that are entirely ``NaN`` map to ``NaN``.
    """
    if prices.shape[1] == 0:
        return pd.Series(dtype="float64")
    # ffill then take the last row -> last valid observation per column.
    return prices.ffill().iloc[-1]
