"""Tests for pre-registration + the pristine forward hold-out harness.

Synthetic-only: every test builds prices from the seeded
:class:`SyntheticProvider` (via the shared ``synthetic_prices`` fixture) or a
hand-built panel. Nothing here touches a live, rate-limited download.

Covered (per the milestone spec):
  * strategy_fingerprint / config_hash determinism + sensitivity;
  * freeze_decision/load_decision round-trip; tamper-evident overwrite refusal;
  * verify_config_matches True/False;
  * evaluate_holdout forward mode (strict-after cutoff; sufficiency boundaries);
  * retrospective mode (last N months; mode label);
  * no-look-ahead (forward-slice metrics invariant to pre-hold-out bars);
  * dsr_carried <= dsr_preregistered;
  * run_holdout.main(['--mode','retrospective', ...]) end-to-end (no network).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trend_robot.config import Config
from trend_robot.validation.holdout import evaluate_holdout
from trend_robot.validation.preregistration import (
    DecisionRecord,
    PreRegistrationError,
    config_hash,
    freeze_decision,
    load_decision,
    strategy_fingerprint,
    verify_config_matches,
)

# The shared ``cfg`` / ``synthetic_prices`` fixtures come from tests/conftest.py.


def _make_cfg(base: Config, **overrides: object) -> Config:
    """Return a re-validated copy of ``base`` with ``overrides`` applied."""
    return dataclasses.replace(base, **overrides)


def _freeze(
    cfg: Config,
    path: Path,
    *,
    decision_date: str = "2020-01-01",
    n_trials_spent: int = 12,
    notes: str = "test freeze",
    overwrite: bool = False,
) -> DecisionRecord:
    """Convenience wrapper around :func:`freeze_decision` for the tests."""
    return freeze_decision(
        cfg,
        decision_date=decision_date,
        n_trials_spent=n_trials_spent,
        notes=notes,
        path=path,
        overwrite=overwrite,
    )


# ---------------------------------------------------------------------------
# Fingerprint + hash
# ---------------------------------------------------------------------------
def test_fingerprint_same_cfg_same_hash(cfg: Config) -> None:
    """Identical configs produce identical fingerprints and hashes."""
    fp1 = strategy_fingerprint(cfg)
    fp2 = strategy_fingerprint(_make_cfg(cfg))  # a fresh equal copy
    assert fp1 == fp2
    assert config_hash(fp1) == config_hash(fp2)


def test_fingerprint_only_strategy_fields(cfg: Config) -> None:
    """The fingerprint excludes operational (non-strategy) fields."""
    fp = strategy_fingerprint(cfg)
    assert "initial_capital" not in fp
    assert "train_test_ratio" not in fp
    assert "wf_train_years" not in fp
    # Strategy-defining fields are present.
    for key in ("universe", "direction", "rebalance", "lookbacks", "seed"):
        assert key in fp


def test_operational_change_does_not_change_hash(cfg: Config) -> None:
    """Changing an operational-only field leaves the strategy hash unchanged."""
    other = _make_cfg(cfg, initial_capital=cfg.initial_capital + 1000.0)
    assert config_hash(strategy_fingerprint(cfg)) == config_hash(
        strategy_fingerprint(other)
    )


def test_direction_change_changes_hash(cfg: Config) -> None:
    """Changing ``direction`` changes the fingerprint hash."""
    flipped = "long_short" if cfg.direction == "long_only" else "long_only"
    other = _make_cfg(cfg, direction=flipped)
    assert config_hash(strategy_fingerprint(cfg)) != config_hash(
        strategy_fingerprint(other)
    )


def test_lookbacks_change_changes_hash(cfg: Config) -> None:
    """Changing ``lookbacks`` changes the fingerprint hash."""
    other = _make_cfg(cfg, lookbacks=[*cfg.lookbacks, 504])
    assert config_hash(strategy_fingerprint(cfg)) != config_hash(
        strategy_fingerprint(other)
    )


def test_config_hash_is_key_order_independent(cfg: Config) -> None:
    """The hash is invariant to dict key insertion order (sort_keys=True)."""
    fp = strategy_fingerprint(cfg)
    shuffled = dict(reversed(list(fp.items())))
    assert config_hash(fp) == config_hash(shuffled)


# ---------------------------------------------------------------------------
# Freeze / load / overwrite
# ---------------------------------------------------------------------------
def test_freeze_load_round_trip(cfg: Config, tmp_path: Path) -> None:
    """A frozen record reloads to an equal :class:`DecisionRecord`."""
    path = tmp_path / "decision_record.json"
    record = _freeze(cfg, path, n_trials_spent=7, notes="round-trip")
    assert path.is_file()
    loaded = load_decision(path)
    assert loaded == record
    assert loaded.decision_date == "2020-01-01"
    assert loaded.n_trials_spent == 7
    assert loaded.config_hash == config_hash(strategy_fingerprint(cfg))
    assert loaded.config_fingerprint == strategy_fingerprint(cfg)


def test_freeze_refuses_to_overwrite_differing_record(
    cfg: Config, tmp_path: Path
) -> None:
    """Re-freezing a DIFFERENT strategy at the same path raises (tamper-evident)."""
    path = tmp_path / "decision_record.json"
    _freeze(cfg, path)
    flipped = "long_short" if cfg.direction == "long_only" else "long_only"
    other = _make_cfg(cfg, direction=flipped)
    with pytest.raises(PreRegistrationError, match="DIFFERENT strategy hash"):
        _freeze(other, path)


def test_freeze_overwrite_flag_allows_replacement(
    cfg: Config, tmp_path: Path
) -> None:
    """The explicit overwrite flag permits replacing a differing record."""
    path = tmp_path / "decision_record.json"
    _freeze(cfg, path)
    flipped = "long_short" if cfg.direction == "long_only" else "long_only"
    other = _make_cfg(cfg, direction=flipped)
    record = _freeze(other, path, overwrite=True)
    reloaded = load_decision(path)
    assert reloaded.config_hash == record.config_hash
    assert reloaded.config_hash == config_hash(strategy_fingerprint(other))


def test_freeze_identical_record_allowed_without_overwrite(
    cfg: Config, tmp_path: Path
) -> None:
    """Re-freezing the SAME strategy (same hash) is allowed without overwrite."""
    path = tmp_path / "decision_record.json"
    first = _freeze(cfg, path, notes="first")
    second = _freeze(cfg, path, notes="second")  # same hash -> allowed
    assert second.config_hash == first.config_hash


def test_freeze_rejects_bad_date(cfg: Config, tmp_path: Path) -> None:
    """A malformed decision date is rejected at freeze time."""
    path = tmp_path / "decision_record.json"
    with pytest.raises(PreRegistrationError, match="ISO"):
        _freeze(cfg, path, decision_date="not-a-date")


def test_freeze_rejects_nonpositive_trials(cfg: Config, tmp_path: Path) -> None:
    """``n_trials_spent`` must be >= 1."""
    path = tmp_path / "decision_record.json"
    with pytest.raises(PreRegistrationError, match="n_trials_spent"):
        _freeze(cfg, path, n_trials_spent=0)


def test_load_missing_record_raises(tmp_path: Path) -> None:
    """Loading a non-existent record raises a clear error."""
    with pytest.raises(PreRegistrationError, match="not found"):
        load_decision(tmp_path / "nope.json")


# ---------------------------------------------------------------------------
# verify_config_matches
# ---------------------------------------------------------------------------
def test_verify_config_matches_true_then_false(
    cfg: Config, tmp_path: Path
) -> None:
    """Matching cfg verifies True; a fingerprinted change verifies False."""
    path = tmp_path / "decision_record.json"
    record = _freeze(cfg, path)
    assert verify_config_matches(cfg, record) is True

    drifted = _make_cfg(cfg, vol_window=cfg.vol_window + 5)
    assert verify_config_matches(drifted, record) is False


# ---------------------------------------------------------------------------
# evaluate_holdout -- forward mode
# ---------------------------------------------------------------------------
def test_forward_holdout_only_bars_strictly_after_decision(
    cfg: Config, synthetic_prices: pd.DataFrame, tmp_path: Path
) -> None:
    """Forward hold-out contains only bars strictly after the decision date."""
    # Choose a decision date that lands inside the panel and is an actual bar,
    # so we can assert STRICT inequality (the bar itself must be excluded).
    mid = synthetic_prices.index[len(synthetic_prices) // 2]
    decision_date = mid.strftime("%Y-%m-%d")
    record = _freeze(
        cfg, tmp_path / "d.json", decision_date=decision_date, n_trials_spent=3
    )

    report = evaluate_holdout(synthetic_prices, cfg, record, mode="forward")
    assert report.mode == "forward"
    # The boundary bar (== decision date) must NOT be in the hold-out.
    assert report.holdout_start is not None
    assert report.holdout_start > mid
    # Every hold-out bar is strictly after the cutoff.
    expected = synthetic_prices.index[synthetic_prices.index > mid]
    assert report.n_holdout_bars == len(expected)


def test_forward_holdout_decision_at_last_bar_is_insufficient(
    cfg: Config, synthetic_prices: pd.DataFrame, tmp_path: Path
) -> None:
    """A decision date at/after the last bar yields 0 forward bars -> insufficient."""
    last = synthetic_prices.index[-1]
    record = _freeze(
        cfg,
        tmp_path / "d.json",
        decision_date=last.strftime("%Y-%m-%d"),
    )
    report = evaluate_holdout(synthetic_prices, cfg, record, mode="forward")
    assert report.n_holdout_bars == 0
    assert report.sufficient is False
    assert report.metrics == {}
    assert report.holdout_start is None
    # Insufficient reports do not produce a retain verdict.
    assert report.retain_preregistered is False
    assert report.retain_carried is False


def test_forward_holdout_early_decision_is_sufficient_with_metrics(
    cfg: Config, synthetic_prices: pd.DataFrame, tmp_path: Path
) -> None:
    """An early-enough decision date yields a sufficient hold-out with metrics."""
    # Decision well before the panel end so >= 1 year of bars follow.
    early = synthetic_prices.index[0].strftime("%Y-%m-%d")
    record = _freeze(
        cfg, tmp_path / "d.json", decision_date=early, n_trials_spent=10
    )
    report = evaluate_holdout(synthetic_prices, cfg, record, mode="forward")
    assert report.sufficient is True
    assert report.n_holdout_bars >= report.min_bars
    # Metrics computed on the slice.
    assert "sharpe" in report.metrics
    assert "cagr" in report.metrics
    # Both DSR values are finite probabilities in [0, 1].
    assert 0.0 <= report.dsr_preregistered <= 1.0
    assert 0.0 <= report.dsr_carried <= 1.0


def test_min_bars_defaults_to_one_year(
    cfg: Config, synthetic_prices: pd.DataFrame, tmp_path: Path
) -> None:
    """Default ``min_bars`` is one year (cfg.periods_per_year)."""
    record = _freeze(
        cfg, tmp_path / "d.json", decision_date=synthetic_prices.index[-1].strftime("%Y-%m-%d")
    )
    report = evaluate_holdout(synthetic_prices, cfg, record, mode="forward")
    assert report.min_bars == cfg.periods_per_year


# ---------------------------------------------------------------------------
# evaluate_holdout -- retrospective mode
# ---------------------------------------------------------------------------
def test_retrospective_mode_selects_last_n_months(
    cfg: Config, synthetic_prices: pd.DataFrame, tmp_path: Path
) -> None:
    """Retrospective mode selects ~ the last N months and labels itself."""
    record = _freeze(cfg, tmp_path / "d.json", n_trials_spent=8)
    months = 18
    report = evaluate_holdout(
        synthetic_prices,
        cfg,
        record,
        mode="retrospective",
        retrospective_months=months,
    )
    assert report.mode == "retrospective"
    # Window starts roughly N months before the last bar (strict-after offset).
    last = synthetic_prices.index[-1]
    expected_start_after = last - pd.DateOffset(months=months)
    assert report.holdout_start is not None
    assert report.holdout_start > expected_start_after
    assert report.holdout_end == last
    # Sanity: an 18-month window of business days is on the order of ~390 bars.
    assert 300 <= report.n_holdout_bars <= 480


def test_retrospective_dsr_carried_le_preregistered(
    cfg: Config, synthetic_prices: pd.DataFrame, tmp_path: Path
) -> None:
    """On the same slice, the carried DSR (more trials) <= the pre-registered DSR."""
    record = _freeze(cfg, tmp_path / "d.json", n_trials_spent=25)
    report = evaluate_holdout(
        synthetic_prices,
        cfg,
        record,
        mode="retrospective",
        retrospective_months=36,
    )
    assert report.sufficient is True
    assert np.isfinite(report.dsr_preregistered)
    assert np.isfinite(report.dsr_carried)
    # More trials -> higher hurdle -> lower (or equal) DSR.
    assert report.dsr_carried <= report.dsr_preregistered + 1e-12


# ---------------------------------------------------------------------------
# No look-ahead: forward-slice metrics invariant to pre-hold-out bars
# ---------------------------------------------------------------------------
def test_no_lookahead_holdout_metrics_invariant_to_prehistory(
    cfg: Config, synthetic_prices: pd.DataFrame, tmp_path: Path
) -> None:
    """Hold-out metrics depend only on data up to each hold-out date.

    Because weights are computed on the FULL history with causal signals,
    truncating the panel's *leading* bars (while keeping enough warm-up before
    the hold-out so the longest lookback is still defined) must not change the
    metrics measured on the forward slice. We demonstrate this by evaluating the
    same forward hold-out on the full panel and on a panel whose earliest year
    has been dropped.
    """
    # Decision near the end so the hold-out is short but the warm-up before it is
    # long (>> max lookback), guaranteeing the dropped leading bars are not part
    # of any hold-out date's causal window.
    decision = synthetic_prices.index[-130]  # ~ last ~half year of bars follow
    decision_date = decision.strftime("%Y-%m-%d")
    record = _freeze(
        cfg, tmp_path / "d.json", decision_date=decision_date, n_trials_spent=5
    )

    full = synthetic_prices
    # Drop the first ~year of leading bars; everything the hold-out dates depend
    # on (their trailing max-lookback window) is fully retained.
    trimmed = synthetic_prices.iloc[cfg.periods_per_year :]
    assert trimmed.index[0] < decision  # warm-up still precedes the hold-out

    # Use min_bars=1 so the short forward slice is evaluated in both cases.
    rep_full = evaluate_holdout(
        full, cfg, record, mode="forward", min_bars=1
    )
    rep_trim = evaluate_holdout(
        trimmed, cfg, record, mode="forward", min_bars=1
    )

    assert rep_full.sufficient and rep_trim.sufficient
    assert rep_full.n_holdout_bars == rep_trim.n_holdout_bars
    # Metrics on the forward slice are identical regardless of dropped prehistory.
    for key in ("sharpe", "cagr", "max_drawdown", "annual_vol"):
        a = rep_full.metrics[key]
        b = rep_trim.metrics[key]
        assert a == pytest.approx(b, rel=1e-9, abs=1e-12)
    assert rep_full.dsr_preregistered == pytest.approx(
        rep_trim.dsr_preregistered, rel=1e-9, abs=1e-12
    )


def test_holdout_dsr_carried_le_preregistered_forward(
    cfg: Config, synthetic_prices: pd.DataFrame, tmp_path: Path
) -> None:
    """Forward-mode carried DSR <= pre-registered DSR on the same slice."""
    early = synthetic_prices.index[0].strftime("%Y-%m-%d")
    record = _freeze(
        cfg, tmp_path / "d.json", decision_date=early, n_trials_spent=20
    )
    report = evaluate_holdout(synthetic_prices, cfg, record, mode="forward")
    assert report.sufficient is True
    assert report.dsr_carried <= report.dsr_preregistered + 1e-12


def test_invalid_mode_raises(
    cfg: Config, synthetic_prices: pd.DataFrame, tmp_path: Path
) -> None:
    """An unknown mode is rejected."""
    record = _freeze(cfg, tmp_path / "d.json")
    with pytest.raises(ValueError, match="mode"):
        evaluate_holdout(synthetic_prices, cfg, record, mode="sideways")


# ---------------------------------------------------------------------------
# run_holdout.main end-to-end (no network)
# ---------------------------------------------------------------------------
def test_run_holdout_main_retrospective_end_to_end(
    cfg: Config, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``run_holdout.main`` runs end-to-end in retrospective mode on synthetic data.

    Uses the default (synthetic) data source -- no ``--live`` -- so no network is
    touched. Asserts a report is printed and a non-zero exit is not raised.
    """
    import run_holdout

    # Freeze against the project config so verify_config_matches passes (no drift
    # warning), using the real config.yaml the conftest cfg fixture loaded.
    project_root = Path(run_holdout.__file__).resolve().parent
    config_path = project_root / "config.yaml"
    decision_path = tmp_path / "decision_record.json"
    _freeze(cfg, decision_path, decision_date="2018-01-01", n_trials_spent=12)

    rc = run_holdout.main(
        [
            "--config",
            str(config_path),
            "--decision",
            str(decision_path),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--history-years",
            "8",
            "--end",
            "2021-12-31",
            "--mode",
            "retrospective",
            "--retrospective-months",
            "24",
            "--log-level",
            "WARNING",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "NON-PRISTINE RETROSPECTIVE" in out
    assert "SECTION 6.5" in out
    # No drift warning expected (config matches the frozen record).
    assert "CONFIG DRIFT" not in out


def test_run_holdout_main_forward_end_to_end(
    cfg: Config, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``run_holdout.main`` runs end-to-end in forward (pristine) mode.

    A decision date early in the synthetic window yields a sufficient forward
    hold-out, so the pristine banner and a verdict are printed.
    """
    import run_holdout

    project_root = Path(run_holdout.__file__).resolve().parent
    config_path = project_root / "config.yaml"
    decision_path = tmp_path / "decision_record.json"
    _freeze(cfg, decision_path, decision_date="2016-01-01", n_trials_spent=12)

    rc = run_holdout.main(
        [
            "--config",
            str(config_path),
            "--decision",
            str(decision_path),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--history-years",
            "8",
            "--end",
            "2021-12-31",
            "--mode",
            "forward",
            "--log-level",
            "WARNING",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "PRISTINE FORWARD HOLD-OUT" in out
    assert "SECTION 6.5" in out
