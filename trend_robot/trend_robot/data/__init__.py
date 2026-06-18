"""Data layer: price providers and disk caching.

Exposes the :class:`DataProvider` protocol (the contract every provider must
satisfy), the parquet cache helper / :class:`CachedProvider` wrapper, and the
two concrete providers:

  * :class:`SyntheticProvider` -- deterministic seeded GBM prices for offline
    runs and tests (Yahoo is rate-limited in this environment).
  * :class:`YFinanceProvider`  -- live adjusted closes via ``yfinance`` with a
    parquet cache and graceful degradation on failure.
"""

from __future__ import annotations

from .provider import (
    CachedProvider,
    DataProvider,
    cache_path,
    make_cache_key,
    read_cache,
    write_cache,
)
from .synthetic_provider import SyntheticProvider
from .yfinance_provider import YFinanceProvider

__all__ = [
    "DataProvider",
    "CachedProvider",
    "make_cache_key",
    "cache_path",
    "read_cache",
    "write_cache",
    "SyntheticProvider",
    "YFinanceProvider",
]
