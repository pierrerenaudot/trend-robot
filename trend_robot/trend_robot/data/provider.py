"""Data-provider contract and disk-cache helper.

Defines the :class:`DataProvider` :class:`typing.Protocol` that every concrete
price source (Yahoo Finance, synthetic, ...) must satisfy, plus a small parquet
cache helper used by providers to avoid repeated downloads, and a
:class:`CachedProvider` wrapper that memoizes any provider's responses to disk.

DATA CONTRACT (enforced by all providers):
  * Return a :class:`pandas.DataFrame` indexed by date (tz-naive trading days).
  * Columns are the requested tickers; values are *adjusted* close prices
    (dividends/splits applied).
  * Gaps are explicit (``NaN``) -- NO silent forward-fill.
  * Never return data after ``end`` (no look-ahead / future leakage).

Public API
----------
- :class:`DataProvider`   -- the provider contract (Protocol).
- :func:`make_cache_key`  -- stable key from a (tickers, start, end) request.
- :func:`cache_path`      -- map a cache key to a parquet path.
- :func:`read_cache`      -- read a cached frame (or ``None`` if absent).
- :func:`write_cache`     -- persist a frame to parquet atomically.
- :class:`CachedProvider` -- transparent parquet-caching wrapper.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd

__all__ = [
    "DataProvider",
    "make_cache_key",
    "cache_path",
    "read_cache",
    "write_cache",
    "CachedProvider",
]

_LOGGER = logging.getLogger(__name__)


@runtime_checkable
class DataProvider(Protocol):
    """Protocol every price provider must implement.

    Implementations must honor the data contract described in the module
    docstring (adjusted closes, tz-naive index, explicit NaN gaps, no future
    data beyond ``end``).
    """

    def get_prices(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        """Return adjusted-close prices for ``tickers`` over ``[start, end]``.

        Parameters
        ----------
        tickers:
            List of ticker symbols to fetch.
        start, end:
            Inclusive ISO date bounds (``"YYYY-MM-DD"``).

        Returns
        -------
        pandas.DataFrame
            Indexed by tz-naive trading days, columns=``tickers``, values are
            adjusted close prices with explicit ``NaN`` gaps.
        """
        ...


# ---------------------------------------------------------------------------
# Cache keying & parquet helpers
# ---------------------------------------------------------------------------
def make_cache_key(tickers: list[str], start: str, end: str) -> str:
    """Build a stable, collision-resistant cache key for a price request.

    The key is invariant to the *order* in which tickers are requested (so
    ``["SPY", "TLT"]`` and ``["TLT", "SPY"]`` share a cache entry) and is a
    short hexadecimal digest safe to use as a filename.

    Parameters
    ----------
    tickers:
        Ticker symbols requested.
    start, end:
        Inclusive ISO date bounds (``"YYYY-MM-DD"``).

    Returns
    -------
    str
        A 16-character hex digest identifying the request.
    """
    normalized = ",".join(sorted(str(t) for t in tickers))
    payload = f"{normalized}|{start}|{end}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def cache_path(cache_dir: str | Path, key: str) -> Path:
    """Return the parquet cache file path for a given cache ``key``.

    Parameters
    ----------
    cache_dir:
        Directory under which cached parquet files are stored.
    key:
        Stable identifier for the cached payload (e.g. a hash of the request).

    Returns
    -------
    pathlib.Path
        ``<cache_dir>/<key>.parquet``.
    """
    return Path(cache_dir) / f"{key}.parquet"


def read_cache(path: str | Path) -> pd.DataFrame | None:
    """Read a cached price frame from parquet, or ``None`` if absent/unreadable.

    Reading never raises on a missing or corrupt cache file: a corrupt file is
    treated as a miss (logged), so a damaged cache degrades to a re-fetch rather
    than crashing the pipeline.

    Parameters
    ----------
    path:
        Filesystem path to a parquet cache file.

    Returns
    -------
    pandas.DataFrame | None
        The cached frame (DatetimeIndex, tz-naive) or ``None`` on a miss.
    """
    p = Path(path)
    if not p.is_file():
        return None
    try:
        df = pd.read_parquet(p)
    except Exception as exc:  # noqa: BLE001 - any read failure is a cache miss
        _LOGGER.warning("Ignoring unreadable cache file %s: %s", p, exc)
        return None
    # Parquet preserves the index name but we re-assert tz-naive datetimes to
    # guarantee the contract regardless of how the file was written.
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def write_cache(df: pd.DataFrame, path: str | Path) -> None:
    """Persist a price frame to parquet at ``path`` (atomic, best-effort).

    The parent directory is created if needed. The write is atomic: data is
    written to a temporary sibling file and then renamed into place, so a
    concurrent or interrupted write never leaves a half-written cache entry.
    Failures are logged and swallowed -- caching is an optimization and must
    never crash the pipeline.

    Parameters
    ----------
    df:
        Price frame to persist (DatetimeIndex, columns=tickers).
    path:
        Destination parquet path.
    """
    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        df.to_parquet(tmp)
        os.replace(tmp, p)
    except Exception as exc:  # noqa: BLE001 - caching must never crash callers
        _LOGGER.warning("Failed to write cache file %s: %s", p, exc)


# ---------------------------------------------------------------------------
# Transparent caching wrapper
# ---------------------------------------------------------------------------
class CachedProvider:
    """Wrap any :class:`DataProvider`, memoizing responses to a parquet cache.

    Identical ``get_prices`` requests (same tickers ignoring order, same
    ``start``/``end``) are served from disk on subsequent calls, so the wrapped
    provider is invoked at most once per distinct request. This is what lets a
    rate-limited source (Yahoo) avoid re-downloading, and lets tests assert that
    a cache hit avoids recompute.

    Parameters
    ----------
    provider:
        The underlying provider to wrap (must satisfy :class:`DataProvider`).
    cache_dir:
        Directory where parquet cache files are stored.
    """

    def __init__(self, provider: DataProvider, cache_dir: str | Path) -> None:
        self._provider = provider
        self._cache_dir = Path(cache_dir)

    def get_prices(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        """Return cached prices if present, else fetch, cache, and return.

        Parameters
        ----------
        tickers:
            Ticker symbols to fetch.
        start, end:
            Inclusive ISO date bounds (``"YYYY-MM-DD"``).

        Returns
        -------
        pandas.DataFrame
            Adjusted-close frame honoring the data contract. Cached results are
            re-projected onto the requested ``tickers`` (and their order) so the
            order-independent cache key never changes the returned column order.
        """
        key = make_cache_key(tickers, start, end)
        path = cache_path(self._cache_dir, key)

        cached = read_cache(path)
        if cached is not None:
            return self._project(cached, tickers)

        fetched = self._provider.get_prices(tickers, start, end)
        # Only persist non-empty frames: an empty frame typically signals a
        # transient failure (e.g. rate limit) we want to retry next time.
        if not fetched.empty:
            write_cache(fetched, path)
        return fetched

    @staticmethod
    def _project(df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
        """Re-order/select cached columns to match the requested ``tickers``.

        Missing tickers (not present in the cached frame) appear as all-``NaN``
        columns, preserving the explicit-gap contract.
        """
        return df.reindex(columns=list(tickers))
