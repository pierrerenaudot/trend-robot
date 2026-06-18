"""Yahoo Finance price provider.

Implements :class:`DataProvider` against ``yfinance``, returning *adjusted*
close prices and degrading GRACEFULLY when downloads fail. Yahoo is rate-limited
in this environment (HTTP 429): on any download failure the provider logs a
warning and returns whatever it can (a partial frame, or an empty frame), and
NEVER raises so the research pipeline keeps running.

A parquet disk cache (via :class:`trend_robot.data.provider.CachedProvider`) is
composed on top through :meth:`YFinanceProvider.cached`, so identical requests
do not re-download.

Honors the data contract in :mod:`trend_robot.data.provider`:
  * tz-naive trading-day index,
  * columns=tickers (request order),
  * adjusted closes,
  * explicit ``NaN`` gaps (NO silent forward-fill),
  * never any data after ``end``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .provider import CachedProvider

__all__ = ["YFinanceProvider"]

_LOGGER = logging.getLogger(__name__)


class YFinanceProvider:
    """Adjusted-close price provider backed by ``yfinance``.

    Parameters
    ----------
    pause:
        Optional courtesy pause (seconds) reserved for future per-request
        throttling; unused at research scale but kept for forward-compat.
    """

    def __init__(self, pause: float = 0.0) -> None:
        self._pause = float(pause)

    def cached(self, cache_dir: str | Path) -> CachedProvider:
        """Return this provider wrapped in a parquet-caching :class:`CachedProvider`.

        Parameters
        ----------
        cache_dir:
            Directory under which downloaded frames are cached as parquet.

        Returns
        -------
        CachedProvider
            A provider that serves identical requests from disk, downloading at
            most once per distinct request.
        """
        return CachedProvider(self, cache_dir)

    def get_prices(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        """Download adjusted-close prices for ``tickers`` over ``[start, end]``.

        On any failure (rate limit / network / parse), logs a warning and
        returns the best available frame (possibly empty). Never raises.

        Parameters
        ----------
        tickers:
            Ticker symbols to download.
        start, end:
            Inclusive ISO date bounds (``"YYYY-MM-DD"``).

        Returns
        -------
        pandas.DataFrame
            Tz-naive trading-day index, columns=``tickers`` (request order),
            adjusted closes with explicit ``NaN`` gaps; no data after ``end``.
            An all-``NaN``/empty frame is returned if nothing could be fetched.
        """
        tickers = list(tickers)
        try:
            import yfinance as yf  # imported lazily so the package imports offline

            # ``auto_adjust=True`` -> the returned "Close" is the *adjusted*
            # close (dividends/splits applied), matching the data contract.
            raw = yf.download(
                tickers=tickers,
                start=start,
                # yfinance treats ``end`` as exclusive; bump by one day so the
                # requested ``end`` date itself is included, then we re-clip.
                end=(pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                auto_adjust=True,
                actions=False,
                progress=False,
                threads=False,
                group_by="column",
            )
        except Exception as exc:  # noqa: BLE001 - rate limit/network: degrade
            _LOGGER.warning(
                "yfinance download failed for %s (%s..%s): %s. "
                "Returning empty frame (graceful degradation).",
                tickers,
                start,
                end,
                exc,
            )
            return self._empty(tickers, start, end)

        return self._normalize(raw, tickers, start, end)

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------
    def _normalize(
        self,
        raw: pd.DataFrame,
        tickers: list[str],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Coerce a raw yfinance frame into the canonical data contract.

        Handles both the single-ticker (flat columns) and multi-ticker
        (MultiIndex columns) layouts, extracts the adjusted "Close", aligns to
        the requested ticker set/order, makes the index tz-naive, and clips off
        anything after ``end``.
        """
        if raw is None or len(raw) == 0:
            return self._empty(tickers, start, end)

        try:
            close = self._extract_close(raw, tickers)
        except Exception as exc:  # noqa: BLE001 - malformed payload: degrade
            _LOGGER.warning(
                "Failed to parse yfinance payload for %s: %s. Returning empty.",
                tickers,
                exc,
            )
            return self._empty(tickers, start, end)

        # Align to the requested universe/order; absent tickers -> NaN columns
        # (explicit gaps, NO forward-fill).
        close = close.reindex(columns=tickers)

        # tz-naive trading-day index; never any data after ``end``.
        close.index = pd.to_datetime(close.index).tz_localize(None)
        end_ts = pd.Timestamp(end)
        close = close.loc[close.index <= end_ts]
        close.index.name = "date"
        return close.sort_index().astype("float64")

    @staticmethod
    def _extract_close(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
        """Extract the (adjusted) Close columns as a tickers-keyed frame."""
        if isinstance(raw.columns, pd.MultiIndex):
            # Layout is (field, ticker) under group_by="column".
            if "Close" in raw.columns.get_level_values(0):
                close = raw.xs("Close", axis=1, level=0)
            else:  # pragma: no cover - defensive against layout changes
                close = raw.xs("Close", axis=1, level=1)
            return close
        # Single-ticker download: flat columns with a "Close" field.
        single = raw["Close"]
        if isinstance(single, pd.Series):
            single = single.to_frame(name=tickers[0])
        return single

    @staticmethod
    def _empty(tickers: list[str], start: str, end: str) -> pd.DataFrame:
        """Return a contract-shaped empty frame (no rows, ticker columns)."""
        index = pd.DatetimeIndex([], name="date")
        return pd.DataFrame(columns=list(tickers), index=index, dtype="float64")
