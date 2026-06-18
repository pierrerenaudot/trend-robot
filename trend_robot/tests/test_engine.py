"""Backtest-engine tests (spec section 8): cost application + consistency.

* **Cost application** -- higher per-side cost (and higher turnover) must
  produce a strictly larger total cost and a strictly lower final equity. A
  strategy that only survives at zero/low cost is fragile (spec section 5).
* **Engine consistency** -- a constant unit weight on a single asset with zero
  costs must replicate that asset's cumulative return exactly (the engine adds
  no spurious P&L).

All data is deterministic and offline; nothing touches a live download.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from trend_robot.backtest.engine import run_backtest
from trend_robot.config import Config
from trend_robot.signals.tsmom import tsmom_signal
from trend_robot.portfolio.sizing import target_weights


def _with(cfg: Config, **overrides: object) -> Config:
    """A re-validated copy of ``cfg`` with the given overrides applied."""
    return dataclasses.replace(cfg, **overrides)


# ---------------------------------------------------------------------------
# Engine consistency: w=1 single asset, zero costs => equity == cum return.
# ---------------------------------------------------------------------------
def test_zero_cost_unit_weight_replicates_asset_cumulative_return(
    cfg: Config,
) -> None:
    """Constant weight 1 on one asset, 0 bps cost => equity tracks the asset.

    Daily rebalance keeps the held weight pinned at 1.0 (no drift away from
    target), so the marked-to-market equity must equal
    ``initial_capital * cumprod(1 + r_asset)`` to floating tolerance.
    """
    rng = np.random.default_rng(cfg.seed)
    idx = pd.bdate_range("2020-01-01", periods=120)
    steps = rng.normal(0.0003, 0.01, size=len(idx))
    level = 100.0 * np.cumprod(1.0 + steps)
    prices = pd.DataFrame({"A": level}, index=idx)

    tgt = pd.DataFrame({"A": [1.0] * len(idx)}, index=idx)
    zero_cost = _with(cfg, cost_bps_per_side=0.0, rebalance="daily")
    res = run_backtest(prices, tgt, zero_cost)

    asset_ret = prices["A"].pct_change(fill_method=None).fillna(0.0)
    expected = float(cfg.initial_capital) * (1.0 + asset_ret).cumprod()

    pd.testing.assert_series_equal(
        res.equity, expected.rename("equity"), check_exact=False, rtol=1e-12
    )
    # No costs => no trades charged after the initial establishment beyond the
    # first bar, and total charged cost is exactly zero.
    assert float(res.trades["cost"].sum()) == pytest.approx(0.0, abs=0.0)


def test_zero_cost_unit_weight_final_equity_matches_total_return(
    cfg: Config,
) -> None:
    """Final equity equals initial_capital times the asset's total return."""
    idx = pd.bdate_range("2020-01-01", periods=60)
    # Deterministic, hand-built up-then-down path.
    level = np.concatenate(
        [np.linspace(100.0, 130.0, 30), np.linspace(130.0, 110.0, 30)]
    )
    prices = pd.DataFrame({"A": level}, index=idx)
    tgt = pd.DataFrame({"A": [1.0] * len(idx)}, index=idx)

    res = run_backtest(prices, tgt, _with(cfg, cost_bps_per_side=0.0, rebalance="daily"))
    total_return = level[-1] / level[0]  # first bar return is 0 (pct_change NaN->0)
    assert res.equity.iloc[-1] == pytest.approx(
        float(cfg.initial_capital) * total_return, rel=1e-12
    )


# ---------------------------------------------------------------------------
# Cost application: higher cost => higher total cost => lower equity.
# ---------------------------------------------------------------------------
def test_higher_cost_bps_lowers_final_equity_monotonically(
    synthetic_prices: pd.DataFrame,
    synthetic_returns: pd.DataFrame,
    cfg: Config,
) -> None:
    """Replaying the SAME book at rising cost levels: cost up, equity down.

    Uses the real signal->sizing book so turnover is realistic. Holding the book
    fixed isolates the cost effect: only ``cost_bps_per_side`` changes.
    """
    signals = tsmom_signal(synthetic_prices, cfg.lookbacks, cfg.direction)
    weights = target_weights(signals, synthetic_returns, cfg)

    levels = [0.0, cfg.cost_bps_per_side, *cfg.cost_stress_levels]
    levels = sorted(set(levels))
    assert len(levels) >= 3  # 0 + base + at least one stress level

    final_equities: list[float] = []
    total_costs: list[float] = []
    for bps in levels:
        res = run_backtest(synthetic_prices, weights, _with(cfg, cost_bps_per_side=bps))
        final_equities.append(float(res.equity.iloc[-1]))
        total_costs.append(float(res.trades["cost"].sum()))

    # Total charged cost is strictly increasing in the per-side rate (turnover
    # is identical across runs because the book is identical).
    for lo, hi in zip(total_costs, total_costs[1:]):
        assert hi > lo
    # Final equity is strictly decreasing in cost.
    for lo, hi in zip(final_equities, final_equities[1:]):
        assert hi < lo
    # Zero-cost total cost is exactly zero.
    assert total_costs[0] == pytest.approx(0.0, abs=1e-12)


def test_higher_turnover_increases_total_cost(cfg: Config) -> None:
    """At a fixed cost rate, a higher-turnover schedule costs more.

    Two exogenous target schedules on flat prices (so weights never drift):
    one rebalances to the same book repeatedly (low turnover after setup), the
    other flips the book every bar (high turnover). Same cost rate => the
    flipping book must incur strictly more total cost and end lower.
    """
    idx = pd.bdate_range("2020-01-01", periods=20)
    prices = pd.DataFrame({"A": [100.0] * len(idx)}, index=idx)  # flat => no drift

    steady = pd.DataFrame({"A": [0.5] * len(idx)}, index=idx)
    flipping_vals = [0.5 if i % 2 == 0 else -0.5 for i in range(len(idx))]
    flipping = pd.DataFrame({"A": flipping_vals}, index=idx)

    costed = _with(cfg, cost_bps_per_side=10.0, rebalance="daily")
    steady_res = run_backtest(prices, steady, costed)
    flip_res = run_backtest(prices, flipping, costed)

    steady_turnover = float(steady_res.turnover.sum())
    flip_turnover = float(flip_res.turnover.sum())
    assert flip_turnover > steady_turnover

    steady_cost = float(steady_res.trades["cost"].sum())
    flip_cost = float(flip_res.trades["cost"].sum())
    assert flip_cost > steady_cost

    # On flat prices the only thing moving equity is cost, so more cost => lower.
    assert flip_res.equity.iloc[-1] < steady_res.equity.iloc[-1]


def test_turnover_and_trades_reconcile_on_rebalance(cfg: Config) -> None:
    """Trade-ledger cost reconciles with turnover * cost_rate * equity-at-trade.

    Single rebalance on the first bar from a flat book to a unit book: the one
    recorded trade's cost must equal ``cost_rate * |delta| * initial_capital``.
    """
    idx = pd.bdate_range("2020-01-01", periods=5)
    prices = pd.DataFrame({"A": [100.0] * len(idx)}, index=idx)
    tgt = pd.DataFrame({"A": [1.0] * len(idx)}, index=idx)

    bps = 10.0
    res = run_backtest(prices, tgt, _with(cfg, cost_bps_per_side=bps, rebalance="daily"))

    # Lagged target => first executed book is at bar 1 (delta = 1.0 from flat).
    first_trades = res.trades[res.trades["date"] == idx[1]]
    assert len(first_trades) == 1
    expected_cost = (bps * 1e-4) * 1.0 * float(cfg.initial_capital)
    assert float(first_trades["cost"].iloc[0]) == pytest.approx(expected_cost, rel=1e-12)
