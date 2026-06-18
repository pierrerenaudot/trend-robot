"""No-look-ahead tests (spec section 8).

Proves the zero-look-ahead principle on all three decision layers:

* the SIGNAL (:func:`trend_robot.signals.tsmom.tsmom_signal`),
* the SIZING (:func:`trend_robot.portfolio.sizing.target_weights`),
* the ENGINE (:func:`trend_robot.backtest.engine.run_backtest`).

The technique is *truncate-and-compare*: compute the quantity on the full
history, then recompute on a prefix that drops every observation strictly after
some cut date ``t``. A decision made at ``t`` must be byte-for-byte identical in
both runs -- if it changed, future information had leaked in.

All data is deterministic and offline (``SyntheticProvider`` / hand-built
frames); nothing here touches a live download.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trend_robot.backtest.engine import run_backtest
from trend_robot.config import Config
from trend_robot.portfolio.sizing import target_weights
from trend_robot.signals.tsmom import tsmom_signal


# ---------------------------------------------------------------------------
# SIGNAL: tsmom_signal at t is invariant to dropping data after t.
# ---------------------------------------------------------------------------
def test_signal_no_lookahead_truncate_and_compare(
    synthetic_prices: pd.DataFrame, cfg: Config
) -> None:
    """Signal values up to the cut date are unchanged when the future is removed."""
    full = tsmom_signal(synthetic_prices, cfg.lookbacks, cfg.direction)

    # Cut roughly 80% in, leaving a meaningful future tail to discard.
    cut_pos = int(len(synthetic_prices) * 0.8)
    cut_date = synthetic_prices.index[cut_pos]

    truncated_prices = synthetic_prices.iloc[: cut_pos + 1]
    truncated = tsmom_signal(truncated_prices, cfg.lookbacks, cfg.direction)

    full_head = full.loc[:cut_date]
    # Same shape (no rows fabricated, none dropped) and identical values incl NaN.
    assert list(truncated.index) == list(full_head.index)
    pd.testing.assert_frame_equal(truncated, full_head, check_exact=True)


def test_signal_single_row_decision_uses_only_past(cfg: Config) -> None:
    """The signal at a chosen date is identical whether or not later prices exist.

    Hand-built monotone-up series: with all positive lookback returns the sign
    is +1 on every defined horizon, so the long_short signal is +1. Appending a
    crash *after* the decision date must not alter that decision.
    """
    idx = pd.bdate_range("2018-01-01", periods=400)
    rising = pd.DataFrame(
        {"X": np.linspace(100.0, 300.0, len(idx))}, index=idx
    )
    decision_date = idx[300]

    full = tsmom_signal(rising, cfg.lookbacks, cfg.direction)

    # Replace everything strictly after the decision date with a violent crash.
    crashed = rising.copy()
    crashed.iloc[301:] = 1.0  # future collapse that must be ignored at t=300
    with_future = tsmom_signal(crashed, cfg.lookbacks, cfg.direction)

    assert full.loc[decision_date, "X"] == pytest.approx(
        with_future.loc[decision_date, "X"], abs=0.0
    )
    # Sanity: this date is fully defined and unambiguously long.
    assert full.loc[decision_date, "X"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# SIZING: target_weights at t is invariant to dropping data after t.
# ---------------------------------------------------------------------------
def test_sizing_no_lookahead_truncate_and_compare(
    synthetic_prices: pd.DataFrame,
    synthetic_returns: pd.DataFrame,
    cfg: Config,
) -> None:
    """Target weights up to the cut date do not change when the future is removed."""
    signals = tsmom_signal(synthetic_prices, cfg.lookbacks, cfg.direction)
    full_w = target_weights(signals, synthetic_returns, cfg)

    cut_pos = int(len(synthetic_prices) * 0.8)
    cut_date = synthetic_prices.index[cut_pos]

    sig_trunc = signals.iloc[: cut_pos + 1]
    ret_trunc = synthetic_returns.iloc[: cut_pos + 1]
    trunc_w = target_weights(sig_trunc, ret_trunc, cfg)

    full_head = full_w.loc[:cut_date]
    assert list(trunc_w.index) == list(full_head.index)
    # EWMA vol/cov are causal -> truncating the future must not move any weight.
    pd.testing.assert_frame_equal(trunc_w, full_head, check_exact=True)


# ---------------------------------------------------------------------------
# ENGINE: weights are shift(1)-lagged; a bar's own/later target is never used.
# ---------------------------------------------------------------------------
def test_engine_held_weights_are_shift1_lagged(cfg: Config) -> None:
    """On daily rebalance the held book at t equals the target decided at t-1.

    Construct a target schedule that is flat for many bars then jumps to a fixed
    book on a single date. With daily rebalancing and zero drift influence on
    the *target* (targets are exogenous here), the realized held weight on a bar
    must equal the previous bar's target -- never the current bar's.
    """
    idx = pd.bdate_range("2020-01-01", periods=10)
    # Flat prices => zero returns => no drift, so held == lagged target exactly.
    prices = pd.DataFrame(
        {"A": [100.0] * len(idx), "B": [100.0] * len(idx)}, index=idx
    )
    tgt = pd.DataFrame(0.0, index=idx, columns=["A", "B"])
    # Decide a book at position 4; it must only be *held* from position 5.
    tgt.iloc[4] = [0.5, -0.3]

    daily_cfg = _with(cfg, rebalance="daily")
    res = run_backtest(prices, tgt, daily_cfg)

    held = res.weights
    # Before the decision is executed (positions 0..4) the book is flat.
    assert (held.iloc[:5].to_numpy() == 0.0).all()
    # The decision made at t=4 is first held at t=5 (one-bar lag), not at t=4.
    assert held.iloc[5, held.columns.get_loc("A")] == pytest.approx(0.5)
    assert held.iloc[5, held.columns.get_loc("B")] == pytest.approx(-0.3)


def test_engine_no_lookahead_truncate_and_compare(
    synthetic_prices: pd.DataFrame,
    synthetic_returns: pd.DataFrame,
    cfg: Config,
) -> None:
    """Equity/held-weights up to the cut bar are unchanged when the future drops.

    A full causal pipeline (signal -> sizing -> engine) recomputed on a prefix
    must reproduce the equity curve and held book bar-for-bar up to the cut.
    """
    signals = tsmom_signal(synthetic_prices, cfg.lookbacks, cfg.direction)
    weights = target_weights(signals, synthetic_returns, cfg)
    full = run_backtest(synthetic_prices, weights, cfg)

    cut_pos = int(len(synthetic_prices) * 0.8)
    cut_date = synthetic_prices.index[cut_pos]

    px_trunc = synthetic_prices.iloc[: cut_pos + 1]
    w_trunc = weights.iloc[: cut_pos + 1]
    trunc = run_backtest(px_trunc, w_trunc, cfg)

    # Equity path identical up to and including the cut bar.
    pd.testing.assert_series_equal(
        trunc.equity, full.equity.loc[:cut_date], check_exact=True
    )
    # Held weights identical up to the cut bar.
    pd.testing.assert_frame_equal(
        trunc.weights, full.weights.loc[:cut_date], check_exact=True
    )


def test_engine_future_target_does_not_affect_past_equity(cfg: Config) -> None:
    """Mutating targets strictly after a bar leaves that bar's equity unchanged.

    Same exogenous-target setup, but instead of truncating we *overwrite* the
    future portion of the target schedule. Equity at every bar up to the change
    must be bit-identical, proving the bar never consults its own/later target.
    """
    idx = pd.bdate_range("2020-01-01", periods=12)
    rng = np.random.default_rng(cfg.seed)
    # Deterministic mild random walk so returns are non-trivial (drift matters).
    steps = rng.normal(0.0, 0.01, size=len(idx))
    level = 100.0 * np.cumprod(1.0 + steps)
    prices = pd.DataFrame({"A": level}, index=idx)

    tgt = pd.DataFrame({"A": [0.5] * len(idx)}, index=idx)
    base = run_backtest(prices, tgt, _with(cfg, rebalance="daily"))

    change_pos = 7
    change_date = idx[change_pos]
    tgt_future = tgt.copy()
    tgt_future.iloc[change_pos:] = -1.0  # flip the book in the future only
    perturbed = run_backtest(prices, tgt_future, _with(cfg, rebalance="daily"))

    # The lagged target means the change at change_pos is first executed at
    # change_pos+1; everything up to change_date is therefore untouched.
    pd.testing.assert_series_equal(
        base.equity.loc[:change_date],
        perturbed.equity.loc[:change_date],
        check_exact=True,
    )


def _with(cfg: Config, **overrides: object) -> Config:
    """Local helper: a re-validated copy of ``cfg`` with overrides applied."""
    import dataclasses

    return dataclasses.replace(cfg, **overrides)
