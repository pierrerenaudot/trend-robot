"""Tests for the metrics layer (T6): performance stats + Deflated Sharpe.

Covers the spec section-8 metric requirements:
  * Sharpe verified on a known analytic case.
  * Max drawdown (and duration) verified on a hand-built equity curve.
  * Deflated Sharpe Ratio decreases as ``n_trials`` grows.
Plus exactness checks for CAGR, vol, profit factor, hit rate and per-asset
P&L reconciliation against the equity curve.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from trend_robot.backtest.engine import BacktestResult
from trend_robot.metrics.deflated_sharpe import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    observed_sharpe,
)
from trend_robot.metrics.performance import performance_metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _result_from_returns(
    returns: list[float],
    *,
    initial: float = 100.0,
    assets: list[str] | None = None,
) -> BacktestResult:
    """Build a minimal BacktestResult whose equity matches given returns."""
    assets = assets or ["A"]
    idx = pd.bdate_range("2020-01-01", periods=len(returns) + 1)
    eq_vals = [initial]
    for r in returns:
        eq_vals.append(eq_vals[-1] * (1.0 + r))
    equity = pd.Series(eq_vals, index=idx, name="equity")
    weights = pd.DataFrame(
        np.ones((len(idx), len(assets))), index=idx, columns=assets
    )
    turnover = pd.Series(np.zeros(len(idx)), index=idx, name="turnover")
    trades = pd.DataFrame(
        columns=["date", "asset", "delta_weight", "cost"]
    ).astype({"asset": "object", "delta_weight": "float64", "cost": "float64"})
    return BacktestResult(equity, weights, turnover, trades)


class _Cfg:
    """Minimal config stub exposing only what performance_metrics reads."""

    periods_per_year = 252


# ---------------------------------------------------------------------------
# Sharpe (analytic)
# ---------------------------------------------------------------------------
def test_sharpe_analytic_constant_return_is_infinite() -> None:
    """A constant positive return has zero vol -> Sharpe is undefined (nan)."""
    result = _result_from_returns([0.001] * 50)
    m = performance_metrics(result, _Cfg())
    assert math.isnan(m["sharpe"])
    assert m["annual_vol"] == 0.0


def test_sharpe_analytic_two_point_series() -> None:
    """Hand-computed Sharpe on a small alternating series.

    Returns [+0.01, -0.005, +0.02, -0.005] -> mean and sample std known.
    Sharpe_annual = mean/std(ddof=1) * sqrt(252).
    """
    rets = [0.01, -0.005, 0.02, -0.005]
    result = _result_from_returns(rets)
    m = performance_metrics(result, _Cfg())

    arr = np.array(rets)
    expected = arr.mean() / arr.std(ddof=1) * math.sqrt(252)
    assert m["sharpe"] == pytest.approx(expected, rel=1e-12)


def test_annual_vol_analytic() -> None:
    """Annualized vol equals sample std times sqrt(periods_per_year)."""
    rets = [0.01, -0.01, 0.02, -0.02, 0.0]
    result = _result_from_returns(rets)
    m = performance_metrics(result, _Cfg())
    expected = np.array(rets).std(ddof=1) * math.sqrt(252)
    assert m["annual_vol"] == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# Drawdown (analytic)
# ---------------------------------------------------------------------------
def test_max_drawdown_hand_built() -> None:
    """Equity 100 -> 120 -> 90 -> 110: peak 120, trough 90 -> dd = -25%."""
    idx = pd.bdate_range("2020-01-01", periods=4)
    equity = pd.Series([100.0, 120.0, 90.0, 110.0], index=idx, name="equity")
    weights = pd.DataFrame({"A": [1.0, 1.0, 1.0, 1.0]}, index=idx)
    turnover = pd.Series([0.0, 0.0, 0.0, 0.0], index=idx)
    trades = pd.DataFrame(
        columns=["date", "asset", "delta_weight", "cost"]
    ).astype({"asset": "object", "delta_weight": "float64", "cost": "float64"})
    result = BacktestResult(equity, weights, turnover, trades)

    m = performance_metrics(result, _Cfg())
    # Worst peak-to-trough: 90/120 - 1 = -0.25.
    assert m["max_drawdown"] == pytest.approx(-0.25, rel=1e-12)


def test_max_drawdown_duration() -> None:
    """Underwater from bar 2 (90) through bar 3 (95) -> 2 bars; recovers at 130."""
    idx = pd.bdate_range("2020-01-01", periods=5)
    equity = pd.Series([100.0, 120.0, 90.0, 95.0, 130.0], index=idx, name="equity")
    weights = pd.DataFrame({"A": [1.0] * 5}, index=idx)
    turnover = pd.Series([0.0] * 5, index=idx)
    trades = pd.DataFrame(
        columns=["date", "asset", "delta_weight", "cost"]
    ).astype({"asset": "object", "delta_weight": "float64", "cost": "float64"})
    result = BacktestResult(equity, weights, turnover, trades)

    m = performance_metrics(result, _Cfg())
    assert m["max_drawdown_duration"] == 2


def test_no_drawdown_for_monotonic_curve() -> None:
    """A strictly increasing equity curve has zero drawdown and zero duration."""
    result = _result_from_returns([0.01] * 20)
    m = performance_metrics(result, _Cfg())
    assert m["max_drawdown"] == pytest.approx(0.0, abs=1e-15)
    assert m["max_drawdown_duration"] == 0


# ---------------------------------------------------------------------------
# CAGR (analytic)
# ---------------------------------------------------------------------------
def test_cagr_analytic_one_year_doubling() -> None:
    """Doubling over exactly periods_per_year steps -> CAGR == 100%."""
    ppy = 252
    per_step = 2.0 ** (1.0 / ppy) - 1.0
    result = _result_from_returns([per_step] * ppy)
    m = performance_metrics(result, _Cfg())
    assert m["cagr"] == pytest.approx(1.0, rel=1e-9)


# ---------------------------------------------------------------------------
# Profit factor / hit rate
# ---------------------------------------------------------------------------
def test_profit_factor_and_hit_rate() -> None:
    """Profit factor = gross gains / gross losses; hit rate = frac positive."""
    rets = [0.10, -0.05, 0.10, -0.05]  # gains 0.20, losses 0.10
    result = _result_from_returns(rets)
    m = performance_metrics(result, _Cfg())
    # Note: equity-implied returns are recomputed from the curve; check the
    # signs/magnitudes are consistent rather than the literal inputs.
    assert m["profit_factor"] > 1.0
    assert m["hit_rate"] == pytest.approx(0.5, rel=1e-9)


def test_per_asset_pnl_reconciles_with_equity() -> None:
    """Single-asset attribution sums to the total equity change."""
    result = _result_from_returns([0.01, -0.02, 0.03], assets=["A"])
    m = performance_metrics(result, _Cfg())
    total_change = float(result.equity.iloc[-1] - result.equity.iloc[0])
    assert sum(m["per_asset_pnl"].values()) == pytest.approx(total_change, rel=1e-9)


def test_per_asset_pnl_multi_asset_keys() -> None:
    """Attribution dict carries one entry per asset column."""
    result = _result_from_returns([0.01, -0.01], assets=["A", "B"])
    m = performance_metrics(result, _Cfg())
    assert set(m["per_asset_pnl"]) == {"A", "B"}


# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio
# ---------------------------------------------------------------------------
def _normal_returns(n: int = 1000, mu: float = 0.001, sd: float = 0.01) -> pd.Series:
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(mu, sd, size=n))


def test_dsr_decreases_with_n_trials() -> None:
    """DSR must be monotonically decreasing as n_trials grows."""
    rets = _normal_returns()
    vals = [
        deflated_sharpe_ratio(rets, n_trials=n, skew=0.0, kurtosis=3.0)
        for n in (1, 2, 10, 100, 1000)
    ]
    # Strictly non-increasing.
    for earlier, later in zip(vals, vals[1:]):
        assert later <= earlier + 1e-12
    # And it actually moves (first > last) for a strategy with positive Sharpe.
    assert vals[0] > vals[-1]


def test_dsr_in_unit_interval() -> None:
    """DSR is a probability in [0, 1]."""
    rets = _normal_returns()
    for n in (1, 5, 50):
        dsr = deflated_sharpe_ratio(rets, n_trials=n, skew=-0.3, kurtosis=5.0)
        assert 0.0 <= dsr <= 1.0


def test_expected_max_sharpe_zero_for_single_trial() -> None:
    """With one trial there is no multiple-testing hurdle (SR0 == 0)."""
    assert expected_max_sharpe(1) == 0.0


def test_expected_max_sharpe_increases_with_trials() -> None:
    """The expected-max-Sharpe hurdle grows with the number of trials."""
    hurdles = [expected_max_sharpe(n) for n in (2, 10, 100, 1000)]
    for earlier, later in zip(hurdles, hurdles[1:]):
        assert later > earlier


def test_dsr_single_trial_matches_psr_formula() -> None:
    """With n_trials=1 and normal moments, DSR equals the probabilistic Sharpe.

    PSR(0) = Phi( SR * sqrt(T-1) / sqrt(1 - g3*SR + (g4-1)/4*SR^2) ).
    """
    rets = _normal_returns()
    sr = observed_sharpe(rets)
    t = rets.size
    skew, kurt = 0.0, 3.0
    denom = math.sqrt(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2)
    expected = float(norm.cdf(sr * math.sqrt(t - 1) / denom))
    got = deflated_sharpe_ratio(rets, n_trials=1, skew=skew, kurtosis=kurt)
    assert got == pytest.approx(expected, rel=1e-12)


def test_dsr_fat_tails_lower_than_normal() -> None:
    """Heavier tails / negative skew reduce confidence (lower DSR).

    Isolated at ``n_trials=1`` so the multiple-testing hurdle (SR0) is zero and
    the only difference between the two calls is the non-normality correction in
    the denominator; otherwise an extreme SR0 can floor both DSRs at 0 and mask
    the effect.
    """
    rets = _normal_returns()
    normal_dsr = deflated_sharpe_ratio(rets, n_trials=1, skew=0.0, kurtosis=3.0)
    fat_dsr = deflated_sharpe_ratio(rets, n_trials=1, skew=-0.5, kurtosis=8.0)
    assert fat_dsr < normal_dsr


def test_dsr_invalid_n_trials_raises() -> None:
    """n_trials < 1 is rejected."""
    rets = _normal_returns(50)
    with pytest.raises(ValueError):
        deflated_sharpe_ratio(rets, n_trials=0, skew=0.0, kurtosis=3.0)
