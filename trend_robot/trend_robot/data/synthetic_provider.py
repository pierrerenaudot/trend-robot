"""Deterministic synthetic price provider.

Generates reproducible (seeded) synthetic adjusted-close series so that tests,
smoke tests and offline runs never depend on live, rate-limited downloads
(Yahoo returns HTTP 429 in this environment).

Each ticker follows a geometric Brownian motion (GBM) with per-ticker drift and
volatility deterministically derived from the ticker symbol and the global
seed, so the same (ticker, seed, date range) always yields identical prices.

Implements :class:`DataProvider` and honors the data contract in
:mod:`trend_robot.data.provider` (tz-naive trading-day index, columns=tickers,
adjusted closes, NO future data beyond ``end``).
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

__all__ = ["SyntheticProvider"]

# Number of trading periods per year used to scale the daily drift/vol of the
# GBM. This is a synthetic-data generation detail (not a market parameter):
# the strategy/backtest annualization is governed independently by
# ``cfg.periods_per_year``.
_TRADING_DAYS_PER_YEAR: int = 252


class SyntheticProvider:
    """Deterministic GBM price provider for offline runs and tests.

    Prices are generated on the pandas business-day calendar (``"B"``) between
    ``start`` and ``end`` (inclusive), shared across all tickers so columns are
    calendar-aligned. The series are fully reproducible: identical ``seed`` and
    request bounds produce bit-identical frames.

    Parameters
    ----------
    seed:
        Global integer seed (typically ``cfg.seed``). Combined with each ticker
        symbol to produce that ticker's independent path.
    initial_price:
        Starting price for every series (default ``100.0``). Purely a synthetic
        generation convenience; no market value is implied.
    annual_drift:
        Baseline annualized log-drift around which per-ticker drift is jittered.
    annual_vol:
        Baseline annualized volatility around which per-ticker vol is jittered.
    """

    def __init__(
        self,
        seed: int = 42,
        *,
        initial_price: float = 100.0,
        annual_drift: float = 0.05,
        annual_vol: float = 0.15,
    ) -> None:
        self._seed = int(seed)
        self._initial_price = float(initial_price)
        self._annual_drift = float(annual_drift)
        self._annual_vol = float(annual_vol)

    # -- reproducible per-ticker parameterization --------------------------
    def _ticker_seed(self, ticker: str) -> int:
        """Derive a stable 32-bit RNG seed from ``(global seed, ticker)``."""
        payload = f"{self._seed}:{ticker}".encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        return int.from_bytes(digest[:4], "big")

    def _ticker_params(self, ticker: str) -> tuple[float, float]:
        """Return deterministic ``(annual_drift, annual_vol)`` for ``ticker``.

        Both are jittered around the provider baselines using a per-ticker RNG,
        so different tickers have distinct (but reproducible) risk/return
        profiles. Volatility is floored to stay strictly positive.
        """
        rng = np.random.default_rng(self._ticker_seed(ticker))
        drift = self._annual_drift + rng.normal(0.0, 0.04)
        vol = self._annual_vol * float(rng.uniform(0.6, 1.6))
        return drift, max(vol, 1e-3)

    def get_prices(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        """Return deterministic synthetic adjusted-close prices.

        Parameters
        ----------
        tickers:
            Ticker symbols to generate.
        start, end:
            Inclusive ISO date bounds (``"YYYY-MM-DD"``). The index spans the
            business days in ``[start, end]``; nothing after ``end`` is emitted.

        Returns
        -------
        pandas.DataFrame
            Tz-naive business-day index, columns=``tickers`` (in request order),
            values are strictly-positive GBM adjusted closes. Returns an empty
            (column-typed) frame if the date range contains no business days.
        """
        # Business-day calendar shared by every ticker (calendar alignment).
        index = pd.bdate_range(start=start, end=end)
        # Defensive: drop anything strictly after ``end`` (no future data).
        end_ts = pd.Timestamp(end)
        index = index[index <= end_ts]
        index = pd.DatetimeIndex(index).tz_localize(None)
        index.name = "date"

        if len(index) == 0:
            return pd.DataFrame(columns=list(tickers), index=index, dtype="float64")

        n = len(index)
        dt = 1.0 / _TRADING_DAYS_PER_YEAR
        data: dict[str, np.ndarray] = {}

        for ticker in tickers:
            mu, sigma = self._ticker_params(ticker)
            rng = np.random.default_rng(self._ticker_seed(ticker))
            # GBM log-returns: (mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z.
            shocks = rng.standard_normal(n)
            log_rets = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks
            log_rets[0] = 0.0  # first observation anchors at the initial price
            prices = self._initial_price * np.exp(np.cumsum(log_rets))
            data[ticker] = prices

        df = pd.DataFrame(data, index=index, columns=list(tickers))
        return df.astype("float64")
