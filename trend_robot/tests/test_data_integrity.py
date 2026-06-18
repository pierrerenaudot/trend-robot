"""Data-integrity tests (spec section 8): provider/contract format + NaN handling.

Verifies the data contract from :mod:`trend_robot.data.provider`:
  * tz-naive ``DatetimeIndex`` of trading days,
  * columns are exactly the requested tickers, in request order,
  * adjusted-close-like values (strictly positive, finite) for the synthetic
    provider,
  * no data after ``end`` (no future leakage),
  * explicit ``NaN`` gaps are *preserved* by the signal layer (not silently
    filled),
  * determinism: identical (seed, request) -> identical frame.

The synthetic provider is used throughout; nothing here hits a live download
(Yahoo is rate-limited / HTTP 429 in this environment).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trend_robot.config import Config
from trend_robot.data.synthetic_provider import SyntheticProvider
from trend_robot.signals.tsmom import tsmom_signal


def test_synthetic_provider_contract_shape(cfg: Config) -> None:
    """Index is tz-naive datetimes; columns equal requested tickers, in order."""
    provider = SyntheticProvider(seed=cfg.seed)
    df = provider.get_prices(cfg.universe, "2019-01-01", "2019-06-30")

    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.tz is None  # tz-naive
    assert list(df.columns) == list(cfg.universe)  # exact tickers, request order
    assert df.index.is_monotonic_increasing


def test_synthetic_provider_values_are_positive_finite(cfg: Config) -> None:
    """Adjusted-close-like values: all strictly positive and finite (no NaN)."""
    provider = SyntheticProvider(seed=cfg.seed)
    df = provider.get_prices(["SPY", "TLT"], "2019-01-01", "2019-12-31")

    arr = df.to_numpy()
    assert np.isfinite(arr).all()
    assert (arr > 0.0).all()


def test_synthetic_provider_no_data_after_end(cfg: Config) -> None:
    """The provider never emits a date strictly after ``end`` (no look-ahead)."""
    end = "2020-03-31"
    provider = SyntheticProvider(seed=cfg.seed)
    df = provider.get_prices(["SPY"], "2020-01-01", end)
    assert df.index.max() <= pd.Timestamp(end)


def test_synthetic_provider_is_deterministic(cfg: Config) -> None:
    """Identical (seed, request) yields a bit-identical frame (reproducibility)."""
    p1 = SyntheticProvider(seed=cfg.seed)
    p2 = SyntheticProvider(seed=cfg.seed)
    a = p1.get_prices(cfg.universe, "2018-01-01", "2018-12-31")
    b = p2.get_prices(cfg.universe, "2018-01-01", "2018-12-31")
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_synthetic_provider_seed_changes_output(cfg: Config) -> None:
    """A different seed produces a different (deterministic) path."""
    a = SyntheticProvider(seed=cfg.seed).get_prices(["SPY"], "2018-01-01", "2018-12-31")
    b = SyntheticProvider(seed=cfg.seed + 1).get_prices(
        ["SPY"], "2018-01-01", "2018-12-31"
    )
    assert not a["SPY"].equals(b["SPY"])


def test_signal_preserves_nan_gaps_no_silent_fill(cfg: Config) -> None:
    """An explicit NaN price gap propagates to a NaN signal (no forward-fill).

    The data contract forbids silent gap-filling. Inject a NaN into an otherwise
    valid series at a date where the signal would otherwise be defined; the
    signal at that date must become NaN rather than reusing a stale price.
    """
    idx = pd.bdate_range("2017-01-01", periods=400)
    level = np.linspace(100.0, 200.0, len(idx))
    prices = pd.DataFrame({"X": level}, index=idx)

    full = tsmom_signal(prices, cfg.lookbacks, cfg.direction)
    gap_date = idx[300]
    assert not np.isnan(full.loc[gap_date, "X"])  # defined before we punch a hole

    holed = prices.copy()
    holed.loc[gap_date, "X"] = np.nan
    holed_sig = tsmom_signal(holed, cfg.lookbacks, cfg.direction)

    # The gap date itself is now undefined (NaN P_t -> NaN signal), proving the
    # gap was not silently filled.
    assert np.isnan(holed_sig.loc[gap_date, "X"])


def test_returns_from_contract_handle_nan_without_inf(cfg: Config) -> None:
    """pct_change on a frame with a NaN gap yields NaN (not inf) at the gap.

    Confirms downstream return computation tolerates explicit gaps gracefully:
    NaN in, NaN out -- never an infinity that would poison risk estimates.
    """
    idx = pd.bdate_range("2020-01-01", periods=10)
    s = pd.DataFrame({"X": [100.0, 101.0, np.nan, 103.0, 104.0,
                            105.0, 106.0, 107.0, 108.0, 109.0]}, index=idx)
    rets = s.pct_change(fill_method=None)
    arr = rets["X"].to_numpy()
    assert not np.isinf(arr).any()
    # The gap row and the row immediately after it are NaN (undefined returns).
    assert np.isnan(rets.loc[idx[2], "X"])
    assert np.isnan(rets.loc[idx[3], "X"])


def test_empty_range_returns_contract_shaped_empty(cfg: Config) -> None:
    """A range with no business days yields an empty, correctly-typed frame."""
    provider = SyntheticProvider(seed=cfg.seed)
    # A single weekend day -> no business days in range.
    df = provider.get_prices(["SPY"], "2021-01-02", "2021-01-03")
    assert df.empty
    assert list(df.columns) == ["SPY"]
