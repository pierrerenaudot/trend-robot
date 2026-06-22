"""Tests for the advanced multiple-testing layer (spec 6.4).

Covers White's Reality Check (2000) and Hansen's SPA test (2005) on top of the
stationary bootstrap (Politis & Romano, 1994). All data is synthetic and seeded;
nothing touches the network.

Test matrix
-----------
* :func:`stationary_bootstrap_indices` -- shape, value range, determinism, the
  i.i.d. limit and the wrap-around / block-continuity behaviour.
* All-null candidates (zero-mean noise) => *large* p-values most of the time.
* One genuinely superior candidate among many nulls => *small* p-values.
* Data-snooping penalty: adding useless candidates raises White's p-value for a
  fixed good model (monotone-ish).
* SPA's consistent p-value is no larger than White's (bracketing + power).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trend_robot.validation.multiple_testing import (
    hansens_spa,
    stationary_bootstrap_indices,
    whites_reality_check,
)


# ---------------------------------------------------------------------------
# Synthetic performance-matrix builders
# ---------------------------------------------------------------------------
def _null_perf(t: int, k: int, *, seed: int, scale: float = 0.01) -> np.ndarray:
    """``(t, k)`` matrix of i.i.d. zero-mean Gaussian noise (all-null)."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, scale, size=(t, k))


def _perf_with_one_winner(
    t: int,
    k: int,
    *,
    seed: int,
    edge: float,
    scale: float = 0.01,
) -> np.ndarray:
    """All-null matrix except column 0, which has a positive mean ``edge``."""
    arr = _null_perf(t, k, seed=seed, scale=scale)
    arr[:, 0] += edge
    return arr


# ---------------------------------------------------------------------------
# stationary_bootstrap_indices
# ---------------------------------------------------------------------------
def test_bootstrap_indices_shape_and_range() -> None:
    rng = np.random.default_rng(0)
    n, n_boot = 50, 37
    idx = stationary_bootstrap_indices(n, avg_block=8.0, n_boot=n_boot, rng=rng)
    assert idx.shape == (n_boot, n)
    assert idx.dtype.kind == "i"
    assert idx.min() >= 0
    assert idx.max() < n


def test_bootstrap_indices_deterministic_given_seed() -> None:
    a = stationary_bootstrap_indices(60, 10.0, 25, np.random.default_rng(123))
    b = stationary_bootstrap_indices(60, 10.0, 25, np.random.default_rng(123))
    np.testing.assert_array_equal(a, b)

    # Different seed -> (almost surely) different draws.
    c = stationary_bootstrap_indices(60, 10.0, 25, np.random.default_rng(999))
    assert not np.array_equal(a, c)


def test_bootstrap_indices_iid_limit_avg_block_one() -> None:
    """avg_block == 1 => p == 1 => every step restarts (pure i.i.d. resample)."""
    rng = np.random.default_rng(7)
    idx = stationary_bootstrap_indices(40, avg_block=1.0, n_boot=200, rng=rng)
    # With p == 1, consecutive indices are independent: the fraction of
    # "advance" transitions (next == (prev+1) mod n) should be ~ 1/n, far below
    # what a blocky bootstrap (avg_block >> 1) would show.
    advance = (idx[:, 1:] == (idx[:, :-1] + 1) % 40).mean()
    assert advance < 0.10


def test_bootstrap_indices_blocks_preserve_continuity() -> None:
    """Large avg_block keeps long runs of consecutive (wrap-around) indices."""
    rng = np.random.default_rng(7)
    idx = stationary_bootstrap_indices(40, avg_block=20.0, n_boot=200, rng=rng)
    advance = (idx[:, 1:] == (idx[:, :-1] + 1) % 40).mean()
    # Expected advance fraction ~ 1 - p = 1 - 1/20 = 0.95.
    assert advance > 0.85
    # Wrap-around must occur (a block crossing the end -> index 0 after n-1).
    wrapped = ((idx[:, :-1] == 39) & (idx[:, 1:] == 0)).any()
    assert wrapped


# ---------------------------------------------------------------------------
# White's Reality Check -- null behaviour
# ---------------------------------------------------------------------------
def test_white_all_null_large_pvalue_typical() -> None:
    """All-null candidates => large p-values most of the time across seeds."""
    p_values = []
    for s in range(20):
        perf = _null_perf(400, 10, seed=s)
        res = whites_reality_check(perf, avg_block=5.0, n_boot=500, seed=100 + s)
        p_values.append(res["p_value"])
    p_values = np.asarray(p_values)
    # The Reality Check is conservative under the null; the vast majority of
    # seeds should fail to reject at the 10% level.
    assert (p_values > 0.10).mean() >= 0.7
    assert np.median(p_values) > 0.30


# ---------------------------------------------------------------------------
# White's Reality Check -- power
# ---------------------------------------------------------------------------
def test_white_one_winner_small_pvalue() -> None:
    """One genuinely superior candidate among nulls => small p-value, found."""
    perf = _perf_with_one_winner(600, 10, seed=1, edge=0.004, scale=0.01)
    res = whites_reality_check(perf, avg_block=5.0, n_boot=2000, seed=11)
    assert res["p_value"] < 0.05
    assert res["k_best"] == 0  # the planted winner is column 0


def test_white_accepts_dataframe_and_matches_array() -> None:
    perf = _perf_with_one_winner(300, 6, seed=2, edge=0.005)
    df = pd.DataFrame(perf, columns=[f"strat_{i}" for i in range(perf.shape[1])])
    r_arr = whites_reality_check(perf, avg_block=5.0, n_boot=500, seed=3)
    r_df = whites_reality_check(df, avg_block=5.0, n_boot=500, seed=3)
    assert r_arr["p_value"] == r_df["p_value"]
    # ``DataFrame.to_numpy`` yields a non-C-contiguous buffer, so the mean can
    # differ from the array path at ~1e-16 -- compare with a tolerance.
    assert r_arr["statistic"] == pytest.approx(r_df["statistic"], abs=1e-12)
    assert r_arr["k_best"] == r_df["k_best"]


def test_white_is_deterministic() -> None:
    perf = _perf_with_one_winner(300, 8, seed=4, edge=0.003)
    a = whites_reality_check(perf, avg_block=6.0, n_boot=400, seed=5)
    b = whites_reality_check(perf, avg_block=6.0, n_boot=400, seed=5)
    assert a == b


# ---------------------------------------------------------------------------
# Data-snooping penalty: more useless candidates -> larger White p-value
# ---------------------------------------------------------------------------
def test_white_data_snooping_penalty_monotone_ish() -> None:
    """Adding null candidates around a fixed good model raises White's p-value.

    The single good column (fixed across runs) is padded with an increasing
    number of independent null columns. White's bootstrap maximum is taken over
    more candidates, so the null distribution shifts right and the p-value of
    the *same* good model grows -- the data-snooping penalty.
    """
    rng = np.random.default_rng(20)
    t = 500
    # A modest (not overwhelming) edge: clearly significant alone, but vulnerable
    # to the snooping penalty once many useless candidates are added.
    good = rng.normal(0.0, 0.01, size=(t, 1)) + 0.0020  # fixed winning column

    p_values = []
    extra_counts = [0, 5, 20, 60, 150]
    for n_extra in extra_counts:
        if n_extra:
            noise = np.random.default_rng(1000 + n_extra).normal(
                0.0, 0.01, size=(t, n_extra)
            )
            perf = np.hstack([good, noise])
        else:
            perf = good
        res = whites_reality_check(perf, avg_block=5.0, n_boot=1500, seed=77)
        p_values.append(res["p_value"])

    p_values = np.asarray(p_values)
    # Penalty: with no extra candidates the good model is clearly significant;
    # with many useless candidates it is penalised heavily.
    assert p_values[0] < 0.05
    assert p_values[-1] > p_values[0]
    # Monotone-ish: the overall trend is increasing (allow tiny bootstrap
    # wiggle between adjacent steps but require a strong positive correlation
    # with the number of added candidates).
    corr = np.corrcoef(extra_counts, p_values)[0, 1]
    assert corr > 0.85


# ---------------------------------------------------------------------------
# Hansen's SPA
# ---------------------------------------------------------------------------
def test_spa_one_winner_small_pvalue() -> None:
    perf = _perf_with_one_winner(600, 10, seed=1, edge=0.004, scale=0.01)
    res = hansens_spa(perf, avg_block=5.0, n_boot=2000, seed=11)
    assert res["p_value"] < 0.05
    assert 0.0 <= res["p_value_lower"] <= res["p_value"] <= res["p_value_upper"]


def test_spa_all_null_large_pvalue_typical() -> None:
    p_values = []
    for s in range(20):
        perf = _null_perf(400, 10, seed=s)
        res = hansens_spa(perf, avg_block=5.0, n_boot=500, seed=200 + s)
        p_values.append(res["p_value"])
    p_values = np.asarray(p_values)
    assert (p_values > 0.10).mean() >= 0.6
    assert np.median(p_values) > 0.20


def test_spa_no_more_conservative_than_white() -> None:
    """SPA's consistent p-value should be <= White's on a clear case.

    Hansen's consistent re-centring drops significantly-inferior candidates from
    the snooping pool, so SPA is (weakly) more powerful than White's Reality
    Check, which is exactly SPA's *upper* p-value. Use a case padded with badly
    losing nulls so the two differ.
    """
    rng = np.random.default_rng(31)
    t = 600
    good = rng.normal(0.0, 0.01, size=(t, 1)) + 0.0035
    # Many strongly *inferior* candidates (negative mean): these should be
    # excluded by SPA's consistent threshold but inflate White's threshold.
    losers = rng.normal(-0.004, 0.01, size=(t, 40))
    perf = np.hstack([good, losers])

    white = whites_reality_check(perf, avg_block=5.0, n_boot=2000, seed=88)
    spa = hansens_spa(perf, avg_block=5.0, n_boot=2000, seed=88)

    # The SPA upper p-value mirrors White's full re-centring; the consistent
    # p-value is no larger and here strictly smaller (losers dropped).
    assert spa["p_value"] <= spa["p_value_upper"]
    assert spa["p_value"] <= white["p_value"] + 1e-9
    assert spa["p_value"] < 0.05


def test_spa_is_deterministic() -> None:
    perf = _perf_with_one_winner(300, 8, seed=4, edge=0.003)
    a = hansens_spa(perf, avg_block=6.0, n_boot=400, seed=5)
    b = hansens_spa(perf, avg_block=6.0, n_boot=400, seed=5)
    assert a == b
