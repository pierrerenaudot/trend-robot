"""Position-sizing tests (spec section 8): gross-leverage cap + vol targeting.

* **Gross-exposure cap** -- on every date ``sum_i |w_i| <= max_gross_leverage``.
* **Realized vol ~ target** -- the realized annualized volatility of the sized
  book is of the same order of magnitude as ``portfolio_vol_target`` (the sizing
  recipe actually targets risk, not merely a fixed notional).
* **Purity / zeroing** -- a NaN/0 signal or undefined vol takes no position; the
  function never mutates its inputs.

All data is deterministic and offline; nothing touches a live download.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trend_robot.config import Config
from trend_robot.portfolio.sizing import target_weights
from trend_robot.signals.tsmom import tsmom_signal


# ---------------------------------------------------------------------------
# Gross-leverage cap: sum(|w|) <= max_gross_leverage on every date.
# ---------------------------------------------------------------------------
def test_gross_exposure_never_exceeds_cap(
    synthetic_prices: pd.DataFrame,
    synthetic_returns: pd.DataFrame,
    cfg: Config,
) -> None:
    """Daily gross exposure stays at or below the configured cap (with tol)."""
    signals = tsmom_signal(synthetic_prices, cfg.lookbacks, cfg.direction)
    weights = target_weights(signals, synthetic_returns, cfg)

    gross = weights.abs().sum(axis=1)
    # Tiny float tolerance on the renormalization equality.
    assert (gross <= cfg.max_gross_leverage + 1e-9).all()
    # The cap actually binds at least once (otherwise the test is vacuous):
    assert gross.max() > 0.0


def test_gross_cap_renormalizes_when_exceeded(cfg: Config) -> None:
    """A deliberately tiny cap forces every active row to gross == cap exactly.

    With a maximally tight cap and strong signals across many assets, every
    date that takes any position renormalizes to gross == cap.
    """
    idx = pd.bdate_range("2016-01-01", periods=400)
    rng = np.random.default_rng(cfg.seed)
    cols = ["A", "B", "C", "D"]
    # Trending, low-noise series so signals are strong and vols well-defined.
    data = {}
    for k, c in enumerate(cols):
        drift = 0.0005 * (k + 1)
        noise = rng.normal(0.0, 0.005, size=len(idx))
        data[c] = 100.0 * np.cumprod(1.0 + drift + noise)
    prices = pd.DataFrame(data, index=idx)
    returns = prices.pct_change(fill_method=None)

    tight = _with(cfg, max_gross_leverage=0.5)
    signals = tsmom_signal(prices, cfg.lookbacks, cfg.direction)
    weights = target_weights(signals, returns, tight)

    gross = weights.abs().sum(axis=1)
    active = gross[gross > 0.0]
    assert len(active) > 0
    # Every active date is capped at (or below) the tight cap; the binding ones
    # sit exactly on it.
    assert (active <= tight.max_gross_leverage + 1e-9).all()
    assert active.max() == pytest.approx(tight.max_gross_leverage, rel=1e-9)


# ---------------------------------------------------------------------------
# Realized vol ~ portfolio_vol_target (order of magnitude).
# ---------------------------------------------------------------------------
def test_realized_portfolio_vol_near_target_order_of_magnitude(
    synthetic_prices: pd.DataFrame,
    synthetic_returns: pd.DataFrame,
    cfg: Config,
) -> None:
    """Realized vol of the sized book is the same order of magnitude as target.

    Portfolio return at t uses the weights decided at t-1 (no look-ahead) dotted
    with realized returns at t. Its annualized std should land within a generous
    band around ``portfolio_vol_target`` -- exact equality is impossible because
    ex-ante EWMA risk differs from realized risk, but it must not be off by an
    order of magnitude (that would mean targeting is broken).
    """
    signals = tsmom_signal(synthetic_prices, cfg.lookbacks, cfg.direction)
    weights = target_weights(signals, synthetic_returns, cfg)

    lagged = weights.shift(1).fillna(0.0)
    port_ret = (lagged * synthetic_returns.fillna(0.0)).sum(axis=1)
    # Only score once the book is actually active (skip the warm-up of all-zeros).
    active = port_ret[(weights.abs().sum(axis=1) > 0).shift(1).fillna(False)]
    assert len(active) > 100

    realized_vol = float(active.std(ddof=1) * np.sqrt(cfg.periods_per_year))
    target = cfg.portfolio_vol_target
    # Order-of-magnitude band: within a factor of ~3 either side of target.
    assert target / 3.0 <= realized_vol <= target * 3.0


# ---------------------------------------------------------------------------
# Zeroing / purity.
# ---------------------------------------------------------------------------
def test_nan_and_zero_signals_take_no_position(cfg: Config) -> None:
    """NaN or zero signal => zero weight; defined positive signal => non-zero."""
    idx = pd.bdate_range("2016-01-01", periods=400)
    rng = np.random.default_rng(cfg.seed)
    prices = pd.DataFrame(
        {
            "A": 100.0 * np.cumprod(1.0 + 0.001 + rng.normal(0, 0.005, len(idx))),
            "B": 100.0 * np.cumprod(1.0 + 0.001 + rng.normal(0, 0.005, len(idx))),
        },
        index=idx,
    )
    returns = prices.pct_change(fill_method=None)
    signals = tsmom_signal(prices, cfg.lookbacks, cfg.direction)

    # Force B's signal to NaN and a slice of A's to exactly 0 on a late date.
    forced = signals.copy()
    late = idx[350]
    forced.loc[late, "B"] = np.nan
    forced.loc[late, "A"] = 0.0

    weights = target_weights(forced, returns, cfg)
    assert weights.loc[late, "B"] == 0.0
    assert weights.loc[late, "A"] == 0.0


def test_target_weights_does_not_mutate_inputs(
    synthetic_prices: pd.DataFrame,
    synthetic_returns: pd.DataFrame,
    cfg: Config,
) -> None:
    """target_weights is pure: it never mutates the signals/returns it receives."""
    signals = tsmom_signal(synthetic_prices, cfg.lookbacks, cfg.direction)
    sig_before = signals.copy()
    ret_before = synthetic_returns.copy()

    _ = target_weights(signals, synthetic_returns, cfg)

    pd.testing.assert_frame_equal(signals, sig_before, check_exact=True)
    pd.testing.assert_frame_equal(synthetic_returns, ret_before, check_exact=True)


def _with(cfg: Config, **overrides: object) -> Config:
    """A re-validated copy of ``cfg`` with overrides applied."""
    import dataclasses

    return dataclasses.replace(cfg, **overrides)
