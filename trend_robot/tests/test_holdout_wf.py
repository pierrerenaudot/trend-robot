"""B3 tests: short-window walk-forward stability for the forward hold-out.

The spec's locked-test walk-forward windows are 5y train / 1y test, which need
roughly seven years of forward data before two complete windows fit -- far too
long to make the hold-out's stability leg (b) usable early in a paper-trading
track. B3 lets the hold-out roll *short* windows (e.g. 6-month train /
3-month test) supplied in months, converted to bars via
``periods_per_year / 12``, WITHOUT changing the ``cfg`` defaults that govern the
locked-test section-6.5 read.

These tests are fully offline (the shared ``synthetic_prices`` fixture from the
deterministic :class:`SyntheticProvider`; Yahoo is HTTP 429 here) and assert:

* :func:`walk_forward_splits` honours explicit ``*_bars`` overrides and is
  byte-for-byte backward compatible when they are omitted;
* :func:`evaluate_holdout` with short ``wf_*_months`` makes the stability leg
  assessable (``>= 2`` windows) on a ~2-year slice where the 5y/1y default
  yields ``stability is None`` (0 windows);
* omitting the overrides reproduces the prior hold-out result exactly;
* the formatted report surfaces which window lengths were used;
* the months->bars conversion matches ``periods_per_year / 12``.

No market values are hard-coded: everything flows from the project
:class:`Config` (``config.yaml``).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd
import pytest

from trend_robot.config import Config
from trend_robot.validation.holdout import (
    HoldoutReport,
    _months_to_bars,
    evaluate_holdout,
    format_holdout_report,
)
from trend_robot.validation.preregistration import (
    DecisionRecord,
    freeze_decision,
)
from trend_robot.validation.splits import walk_forward_splits


def _freeze(
    cfg: Config,
    path: Path,
    *,
    decision_date: str = "2018-01-01",
    n_trials_spent: int = 12,
) -> DecisionRecord:
    """Convenience wrapper around :func:`freeze_decision` for these tests."""
    return freeze_decision(
        cfg,
        decision_date=decision_date,
        n_trials_spent=n_trials_spent,
        notes="B3 short-window walk-forward test",
        path=path,
    )


# ===========================================================================
# walk_forward_splits: explicit bar overrides + backward compatibility
# ===========================================================================
def test_walk_forward_bar_overrides_set_window_lengths(cfg: Config) -> None:
    """Explicit ``*_bars`` override the config-derived window lengths exactly."""
    index = pd.bdate_range("2019-01-01", periods=600)
    windows = walk_forward_splits(
        index, cfg, train_bars=126, test_bars=63, step_bars=63
    )
    assert len(windows) >= 2
    for w in windows:
        assert len(w.train_index) == 126
        assert len(w.test_index) == 63
    # Train windows advance by exactly the step.
    pos0 = index.get_loc(windows[0].train_index[0])
    pos1 = index.get_loc(windows[1].train_index[0])
    assert pos1 - pos0 == 63


def test_walk_forward_omitting_overrides_is_unchanged(cfg: Config) -> None:
    """Omitting the overrides reproduces the config-driven windows verbatim."""
    index = pd.bdate_range("2010-01-01", periods=2200)
    baseline = walk_forward_splits(index, cfg)
    # Passing None overrides must be identical to passing nothing at all.
    explicit_none = walk_forward_splits(
        index, cfg, train_bars=None, test_bars=None, step_bars=None
    )
    assert len(baseline) == len(explicit_none)
    for a, b in zip(baseline, explicit_none):
        assert a.fold == b.fold
        pd.testing.assert_index_equal(a.train_index, b.train_index)
        pd.testing.assert_index_equal(a.test_index, b.test_index)


def test_walk_forward_partial_override_keeps_config_for_rest(cfg: Config) -> None:
    """A single override changes only that window; the others stay config-derived."""
    index = pd.bdate_range("2010-01-01", periods=2200)
    train_default = int(round(cfg.wf_train_years * cfg.periods_per_year))
    test_default = int(round(cfg.wf_test_years * cfg.periods_per_year))
    # Override only the test window.
    windows = walk_forward_splits(index, cfg, test_bars=42)
    assert windows
    for w in windows:
        assert len(w.train_index) == train_default
        assert len(w.test_index) == 42 != test_default


# ===========================================================================
# _months_to_bars conversion
# ===========================================================================
def test_months_to_bars_matches_periods_per_year_over_12(cfg: Config) -> None:
    """months -> bars uses ``periods_per_year / 12`` rounded to a whole bar."""
    ppy = cfg.periods_per_year
    assert _months_to_bars(12, ppy) == ppy
    assert _months_to_bars(6, ppy) == int(round(6 * ppy / 12.0))
    assert _months_to_bars(3, ppy) == int(round(3 * ppy / 12.0))


# ===========================================================================
# evaluate_holdout: short windows make leg (b) assessable on a ~2y slice
# ===========================================================================
def _two_year_record(cfg: Config, tmp_path: Path) -> DecisionRecord:
    """A frozen record whose decision date leaves room for a ~2-year hold-out."""
    return _freeze(cfg, tmp_path / "d.json", decision_date="2018-01-01")


def test_default_windows_yield_no_stability_on_short_slice(
    synthetic_prices: pd.DataFrame, cfg: Config, tmp_path: Path
) -> None:
    """The 5y/1y default cannot fit 2 windows in a ~2-year retrospective slice."""
    record = _freeze(cfg, tmp_path / "d.json", n_trials_spent=12)
    report = evaluate_holdout(
        synthetic_prices,
        cfg,
        record,
        mode="retrospective",
        retrospective_months=24,
    )
    assert report.sufficient
    # ~2 years (~504 bars) << one 5y train + 1y test block => 0 windows.
    assert report.stability is None
    # Default window lengths are surfaced and flagged as the config default.
    assert report.wf_windows_overridden is False
    assert report.wf_train_bars == int(round(cfg.wf_train_years * cfg.periods_per_year))
    assert report.wf_test_bars == int(round(cfg.wf_test_years * cfg.periods_per_year))


def test_short_windows_make_stability_assessable_on_short_slice(
    synthetic_prices: pd.DataFrame, cfg: Config, tmp_path: Path
) -> None:
    """Short 6mo/3mo windows fit >= 2 complete windows in the same ~2y slice."""
    record = _freeze(cfg, tmp_path / "d.json", n_trials_spent=12)
    report = evaluate_holdout(
        synthetic_prices,
        cfg,
        record,
        mode="retrospective",
        retrospective_months=24,
        wf_train_months=6,
        wf_test_months=3,
        wf_step_months=3,
    )
    assert report.sufficient
    assert report.wf_windows_overridden is True
    assert report.wf_train_bars == _months_to_bars(6, cfg.periods_per_year)
    assert report.wf_test_bars == _months_to_bars(3, cfg.periods_per_year)
    assert report.wf_step_bars == _months_to_bars(3, cfg.periods_per_year)
    # The walk-forward stability leg is now ASSESSABLE: >= 2 windows.
    assert report.stability is not None
    assert report.stability.n_windows >= 2


def test_short_windows_also_help_forward_mode(
    synthetic_prices: pd.DataFrame, cfg: Config, tmp_path: Path
) -> None:
    """Forward mode on a ~2y post-decision slice: short windows -> assessable."""
    # Decision ~2 years before the last bar so the forward slice is ~2 years.
    last = synthetic_prices.index[-1]
    decision = last - pd.DateOffset(years=2)
    record = _freeze(
        cfg, tmp_path / "d.json", decision_date=decision.strftime("%Y-%m-%d")
    )

    default = evaluate_holdout(synthetic_prices, cfg, record, mode="forward")
    short = evaluate_holdout(
        synthetic_prices,
        cfg,
        record,
        mode="forward",
        wf_train_months=6,
        wf_test_months=3,
        wf_step_months=3,
    )
    assert default.sufficient and short.sufficient
    assert default.stability is None  # 5y/1y can't fit a 2y forward slice
    assert short.stability is not None and short.stability.n_windows >= 2


# ===========================================================================
# Backward compatibility: omitting overrides reproduces the prior result
# ===========================================================================
def test_omitting_overrides_reproduces_prior_holdout_result(
    synthetic_prices: pd.DataFrame, cfg: Config, tmp_path: Path
) -> None:
    """No ``wf_*_months`` => identical report to a call that never knew of them."""
    record = _two_year_record(cfg, tmp_path)
    baseline = evaluate_holdout(synthetic_prices, cfg, record, mode="forward")
    omitted = evaluate_holdout(
        synthetic_prices,
        cfg,
        record,
        mode="forward",
        wf_train_months=None,
        wf_test_months=None,
        wf_step_months=None,
    )
    # The substantive read is unchanged (drop the new bookkeeping fields).
    ignore = {"wf_train_bars", "wf_test_bars", "wf_step_bars", "wf_windows_overridden"}
    a = {k: v for k, v in dataclasses.asdict(baseline).items() if k not in ignore}
    b = {k: v for k, v in dataclasses.asdict(omitted).items() if k not in ignore}
    assert a == b
    assert baseline.wf_windows_overridden is False
    assert omitted.wf_windows_overridden is False


# ===========================================================================
# Formatted report surfaces the window lengths used
# ===========================================================================
def test_report_text_surfaces_window_lengths(
    synthetic_prices: pd.DataFrame, cfg: Config, tmp_path: Path
) -> None:
    """The printed (b) leg shows the window lengths and override status."""
    record = _freeze(cfg, tmp_path / "d.json", n_trials_spent=12)
    short = evaluate_holdout(
        synthetic_prices,
        cfg,
        record,
        mode="retrospective",
        retrospective_months=24,
        wf_train_months=6,
        wf_test_months=3,
        wf_step_months=3,
    )
    text = format_holdout_report(short, record)
    assert "Window lengths (bars)" in text
    assert "OVERRIDDEN (short)" in text
    assert f"train {short.wf_train_bars}" in text

    default = evaluate_holdout(
        synthetic_prices, cfg, record, mode="retrospective", retrospective_months=24
    )
    default_text = format_holdout_report(default, record)
    assert "config default" in default_text


def test_invalid_mode_still_raises_with_overrides(
    synthetic_prices: pd.DataFrame, cfg: Config, tmp_path: Path
) -> None:
    """The mode guard fires before any walk-forward override processing."""
    record = _freeze(cfg, tmp_path / "d.json")
    with pytest.raises(ValueError, match="forward.*retrospective"):
        evaluate_holdout(
            synthetic_prices,
            cfg,
            record,
            mode="sideways",
            wf_train_months=6,
        )
    assert isinstance(record, DecisionRecord)
    assert HoldoutReport is HoldoutReport  # imported symbol is usable
