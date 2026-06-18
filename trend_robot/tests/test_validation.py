"""Validation-harness tests (spec section 6).

Exercises the T9 validation module end to end on deterministic, offline data
(no live downloads -- Yahoo is HTTP 429 here). Coverage:

* **Locked split** (6.1) -- the last ``1 - train_test_ratio`` of the history is
  the untouched OOS test set, the leading fraction is train; the two slices are
  contiguous, disjoint and exhaustive.
* **Walk-forward** (6.2) -- window lengths come from
  ``wf_*_years * periods_per_year``; the concatenated per-window test track is
  contiguous and non-overlapping.
* **Purged CV + embargo** (6.3) -- purging removes train samples whose
  information window overlaps the test fold (purged counts grow with the
  horizon); the embargo removes ``round(cv_embargo * n)`` samples after each
  fold.
* **Trials counter** (6.4) -- :class:`TrialCounter` increments and de-dups, and
  ``int(counter)`` feeds :func:`deflated_sharpe_ratio` with a DSR that decreases
  monotonically as the trials count grows.

Everything flows from the project :class:`Config` (loaded from ``config.yaml``)
and the deterministic :class:`SyntheticProvider`; no market values are
hard-coded here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trend_robot.config import Config
from trend_robot.metrics.deflated_sharpe import deflated_sharpe_ratio
from trend_robot.validation import (
    PurgedKFold,
    TrialCounter,
    concat_test_segments,
    purged_cv_splits,
    train_test_split,
    walk_forward_splits,
)
from trend_robot.validation.splits import _years_to_bars

from .conftest import make_config


# ---------------------------------------------------------------------------
# Fixtures: a deterministic business-day index long enough to exercise the
# walk-forward windows (train + test years) on the real config.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def long_index() -> pd.DatetimeIndex:
    """A 12-year tz-naive business-day index (deterministic, offline)."""
    return pd.bdate_range("2008-01-01", "2019-12-31")


# ===========================================================================
# 6.1 -- Locked train/test split
# ===========================================================================
def test_locked_split_last_fraction_is_test(
    long_index: pd.DatetimeIndex, cfg: Config
) -> None:
    """Test set is the last ``1 - ratio`` fraction; train the first ``ratio``."""
    train, test = train_test_split(long_index, cfg)
    n = len(long_index)
    boundary = int(n * cfg.train_test_ratio)

    # Train is exactly the first ``ratio`` fraction (within integer rounding).
    assert len(train) == boundary
    assert len(test) == n - boundary
    # The train fraction never exceeds the configured ratio.
    assert len(train) / n <= cfg.train_test_ratio + 1e-12
    # The test fraction is approximately (1 - ratio).
    assert len(test) / n == pytest.approx(1.0 - cfg.train_test_ratio, abs=1.0 / n)


def test_locked_split_disjoint_contiguous_exhaustive(
    long_index: pd.DatetimeIndex, cfg: Config
) -> None:
    """Train/test are disjoint, time-ordered and together cover the index."""
    train, test = train_test_split(long_index, cfg)

    # Disjoint.
    assert train.intersection(test).empty
    # Exhaustive: their concatenation is exactly the original index.
    rebuilt = train.append(test)
    pd.testing.assert_index_equal(rebuilt, long_index)
    # Contiguous & ordered: every train date precedes every test date.
    if len(train) and len(test):
        assert train.max() < test.min()


def test_locked_split_accepts_dataframe(
    long_index: pd.DatetimeIndex, cfg: Config
) -> None:
    """Passing a DataFrame splits on its index, identically to the index."""
    frame = pd.DataFrame({"x": np.arange(len(long_index))}, index=long_index)
    train_df, test_df = train_test_split(frame, cfg)
    train_ix, test_ix = train_test_split(long_index, cfg)
    pd.testing.assert_index_equal(train_df, train_ix)
    pd.testing.assert_index_equal(test_df, test_ix)


# ===========================================================================
# 6.2 -- Walk-forward windows
# ===========================================================================
def test_walk_forward_window_lengths_from_config(
    long_index: pd.DatetimeIndex, cfg: Config
) -> None:
    """Train/test window lengths equal ``wf_*_years * periods_per_year``."""
    windows = walk_forward_splits(long_index, cfg)
    assert windows, "expected at least one complete walk-forward window"

    train_bars = _years_to_bars(cfg.wf_train_years, cfg.periods_per_year)
    test_bars = _years_to_bars(cfg.wf_test_years, cfg.periods_per_year)
    step_bars = _years_to_bars(cfg.wf_step_years, cfg.periods_per_year)

    for w in windows:
        assert len(w.train_index) == train_bars
        assert len(w.test_index) == test_bars

    # Folds are numbered 0..k and the train windows advance by exactly ``step``.
    assert [w.fold for w in windows] == list(range(len(windows)))
    if len(windows) >= 2:
        pos0 = long_index.get_loc(windows[0].train_index[0])
        pos1 = long_index.get_loc(windows[1].train_index[0])
        assert pos1 - pos0 == step_bars


def test_walk_forward_train_test_adjacent_no_overlap(
    long_index: pd.DatetimeIndex, cfg: Config
) -> None:
    """Each test window starts right after its train window (no gap/overlap)."""
    windows = walk_forward_splits(long_index, cfg)
    for w in windows:
        # No bar appears in both train and test of the same fold.
        assert pd.Index(w.train_index).intersection(pd.Index(w.test_index)).empty
        train_end_pos = long_index.get_loc(w.train_index[-1])
        test_start_pos = long_index.get_loc(w.test_index[0])
        assert test_start_pos == train_end_pos + 1


def test_walk_forward_concatenated_track_contiguous_non_overlapping(
    long_index: pd.DatetimeIndex, cfg: Config
) -> None:
    """Concatenated test segments form one contiguous, non-overlapping track."""
    windows = walk_forward_splits(long_index, cfg)
    track = concat_test_segments(windows)

    # Non-overlapping: no duplicates, and the total length is the sum of the
    # per-window test lengths (default step == test ==> perfect tiling).
    assert track.is_unique
    assert len(track) == sum(len(w.test_index) for w in windows)

    # Strictly increasing (sorted) and contiguous in index positions: the OOS
    # track is an unbroken run of consecutive bars in the original index.
    assert track.is_monotonic_increasing
    positions = long_index.get_indexer(track)
    assert (np.diff(positions) == 1).all()


def test_walk_forward_empty_when_history_too_short(cfg: Config) -> None:
    """Histories shorter than one train+test block yield no windows."""
    short = pd.bdate_range("2020-01-01", "2020-06-30")
    assert walk_forward_splits(short, cfg) == []
    assert concat_test_segments([]).empty


# ===========================================================================
# 6.3 -- Purged CV + embargo
# ===========================================================================
def test_purged_cv_folds_partition_the_index(cfg: Config) -> None:
    """Test folds are a contiguous, exhaustive partition of all positions."""
    n = 1000
    n_splits = 5
    splits = purged_cv_splits(n, cfg, n_splits=n_splits, horizon=0)
    assert len(splits) == n_splits

    all_test = np.concatenate([s.test_positions for s in splits])
    np.testing.assert_array_equal(np.sort(all_test), np.arange(n))
    # Each fold's test block is itself contiguous.
    for s in splits:
        assert (np.diff(s.test_positions) == 1).all()


def test_purging_grows_with_horizon(cfg: Config) -> None:
    """Larger information-window horizons purge strictly more train samples."""
    n = 2000
    n_splits = 5
    counts: list[int] = []
    for horizon in (0, 21, 63, 126, 252):
        splits = purged_cv_splits(n, cfg, n_splits=n_splits, horizon=horizon)
        counts.append(sum(int(s.purged_positions.size) for s in splits))

    # Zero horizon purges nothing (no train sample's window overlaps a fold).
    assert counts[0] == 0
    # Purged counts are numeric and monotonically non-decreasing in horizon,
    # and strictly grow once the horizon is positive.
    assert all(b >= a for a, b in zip(counts, counts[1:]))
    assert counts[-1] > counts[1] > 0


def test_purging_removes_overlapping_train_samples(cfg: Config) -> None:
    """Every purged sample's info window actually overlaps its test fold."""
    n = 1000
    horizon = 50
    splits = purged_cv_splits(n, cfg, n_splits=4, horizon=horizon)
    for s in splits:
        test_start = int(s.test_positions[0])
        test_end = int(s.test_positions[-1])
        # Purged samples are removed from the train set...
        assert not np.isin(s.purged_positions, s.train_positions).any()
        # ...and each one's [i-h, i+h] window genuinely intersects the fold.
        for i in s.purged_positions:
            assert (i + horizon) >= test_start and (i - horizon) <= test_end


def test_embargo_removes_expected_count_after_each_fold(cfg: Config) -> None:
    """Embargo blocks ``round(cv_embargo * n)`` samples after each test fold."""
    n = 2000
    embargo_frac = 0.05  # large enough that the block is unambiguous
    cfg_emb = make_config(cfg, cv_embargo=embargo_frac)
    embargo_len = int(round(embargo_frac * n))
    assert embargo_len > 0

    splits = purged_cv_splits(n, cfg_emb, n_splits=5, horizon=0)
    for s in splits:
        test_end = int(s.test_positions[-1])
        # Expected embargo block = the ``embargo_len`` positions immediately
        # after the fold that exist and are not themselves in the test fold.
        expected = [
            p
            for p in range(test_end + 1, test_end + embargo_len + 1)
            if p < n and p not in set(s.test_positions.tolist())
        ]
        np.testing.assert_array_equal(
            np.sort(s.embargoed_positions), np.array(sorted(expected))
        )
        # Embargoed samples are not in the resulting train set.
        assert not np.isin(s.embargoed_positions, s.train_positions).any()
    # The last fold ends at the final bar, so its embargo block is empty.
    assert splits[-1].embargoed_positions.size == 0


def test_zero_embargo_removes_nothing(cfg: Config) -> None:
    """With ``cv_embargo == 0`` no sample is embargoed."""
    cfg0 = make_config(cfg, cv_embargo=0.0)
    splits = purged_cv_splits(1000, cfg0, n_splits=5, horizon=0)
    assert all(s.embargoed_positions.size == 0 for s in splits)


def test_purged_kfold_validates_arguments(cfg: Config) -> None:
    """Constructor/`split` reject out-of-range arguments."""
    with pytest.raises(ValueError):
        PurgedKFold(n_splits=1, embargo=cfg.cv_embargo)
    with pytest.raises(ValueError):
        PurgedKFold(n_splits=5, embargo=1.0)
    with pytest.raises(ValueError):
        PurgedKFold(n_splits=5, embargo=cfg.cv_embargo, horizon=-1)
    with pytest.raises(ValueError):
        PurgedKFold(n_splits=5, embargo=cfg.cv_embargo).split(n_samples=3)


# ===========================================================================
# 6.4 -- Trials counter feeding the Deflated Sharpe Ratio
# ===========================================================================
def test_trial_counter_increments_and_dedups() -> None:
    """De-duplicating counter counts distinct configs once; clamps to >= 1."""
    counter = TrialCounter(deduplicate=True)
    assert counter.raw_count == 0
    assert int(counter) == 1  # clamped for DSR validity before any record

    assert counter.record(("a", 1)) == 1
    counter.record(("a", 1))  # duplicate -> no increment
    assert counter.raw_count == 1
    assert counter.record(("b", 2)) == 2
    assert int(counter) == counter.n_trials == 2

    counter.record_many([("c", 3), ("a", 1), ("d", 4)])  # one dup ("a",1)
    assert counter.raw_count == 4  # c and d are new; a is a repeat


def test_trial_counter_no_dedup_counts_every_call() -> None:
    """With ``deduplicate=False`` every record call increments."""
    counter = TrialCounter(deduplicate=False)
    for _ in range(5):
        counter.record(("same", 0))
    assert counter.raw_count == 5
    assert int(counter) == 5


def test_int_counter_feeds_dsr_monotone_decreasing(
    synthetic_prices: pd.DataFrame,
) -> None:
    """DSR decreases monotonically as the trials count from the counter grows."""
    # Deterministic strategy-like return series from synthetic prices.
    returns = synthetic_prices.iloc[:, 0].pct_change(fill_method=None).dropna()
    skew = float(returns.skew())
    kurtosis = float(returns.kurt()) + 3.0  # pandas kurt is excess

    counter = TrialCounter()
    dsr_values: list[float] = []
    for cfg_key in [("m", 21), ("m", 63), ("m", 126), ("m", 252), ("m", 504)]:
        counter.record(cfg_key)
        dsr = deflated_sharpe_ratio(
            returns,
            n_trials=int(counter),  # the counter feeds DSR directly
            skew=skew,
            kurtosis=kurtosis,
        )
        dsr_values.append(dsr)

    assert all(np.isfinite(v) for v in dsr_values)
    # More trials => higher hurdle => non-increasing DSR; and it strictly falls
    # somewhere between the 1-trial and 5-trial extremes.
    assert all(b <= a + 1e-12 for a, b in zip(dsr_values, dsr_values[1:]))
    assert dsr_values[-1] < dsr_values[0]
