"""T10 tests: cost sensitivity (spec 5) + final section-6.5 verdict.

Exercises the cost-stress replay and the locked-test-set validation report on
deterministic, offline data (``SyntheticProvider``; Yahoo is HTTP 429 here).
Coverage mirrors the spec:

* **Cost sensitivity (spec 5)** -- replaying the *same* strategy at higher costs
  charges strictly more total cost and yields lower final equity and Sharpe; the
  comparison table has one row per distinct cost level (base + stress) and never
  mutates the frozen :class:`Config`.
* **Final validation (spec 6.4 / 6.5 / 11)** -- the verdict is measured ONLY on
  the locked test slice from :func:`train_test_split`; ``retain`` is the AND of
  "DSR clearly positive" and "walk-forward stable"; raising ``n_trials`` never
  raises the DSR; and the printed report carries the section-11 human-judgment
  note.

Everything flows from the project :class:`Config` (``config.yaml``); no market
values are hard-coded here.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from trend_robot.config import Config
from trend_robot.portfolio.sizing import target_weights
from trend_robot.signals.tsmom import tsmom_signal
from trend_robot.validation import (
    cost_stress_table,
    cost_stress_test,
    evaluate_final_validation,
    format_final_report,
)
from trend_robot.validation.final_report import FinalValidationReport
from trend_robot.validation.splits import train_test_split

from .conftest import make_config


# ---------------------------------------------------------------------------
# Helper: the (frozen) strategy weights for the synthetic universe, computed
# once via the real signal -> sizing pipeline (no strategy logic duplicated).
# ---------------------------------------------------------------------------
def _weights(prices: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Strategy target weights from the real signal/sizing pipeline."""
    returns = prices.pct_change(fill_method=None)
    signals = tsmom_signal(prices, cfg.lookbacks, direction=cfg.direction)
    return target_weights(signals, returns, cfg)


# ===========================================================================
# Section 5 -- cost sensitivity
# ===========================================================================
def test_cost_stress_table_shape_and_levels(
    synthetic_prices: pd.DataFrame, cfg: Config
) -> None:
    """One row per distinct cost level (base + stress); table is well-formed."""
    weights = _weights(synthetic_prices, cfg)
    rows = cost_stress_test(synthetic_prices, weights, cfg)

    expected_levels = sorted(
        {float(cfg.cost_bps_per_side), *(float(x) for x in cfg.cost_stress_levels)}
    )
    assert [r.cost_bps_per_side for r in rows] == expected_levels

    # Exactly one row is flagged as the configured base cost.
    assert sum(1 for r in rows if r.is_base) == 1
    base_row = next(r for r in rows if r.is_base)
    assert base_row.cost_bps_per_side == float(cfg.cost_bps_per_side)

    table = cost_stress_table(rows)
    assert list(table.index) == expected_levels
    assert list(table.columns) == [
        "is_base",
        "cagr",
        "sharpe",
        "max_drawdown",
        "total_cost",
        "final_equity",
    ]
    assert len(table) == len(expected_levels)


def test_higher_cost_means_more_total_cost_and_lower_equity(
    synthetic_prices: pd.DataFrame, cfg: Config
) -> None:
    """Spec 5: higher cost on identical turnover => more cost, lower equity."""
    weights = _weights(synthetic_prices, cfg)
    rows = cost_stress_test(synthetic_prices, weights, cfg)
    # Rows are ascending in cost; metrics must degrade monotonically with cost.
    assert len(rows) >= 2
    total_costs = [r.total_cost for r in rows]
    final_eq = [r.final_equity for r in rows]
    sharpes = [r.sharpe for r in rows]

    assert all(b > a for a, b in zip(total_costs, total_costs[1:]))
    assert all(b < a for a, b in zip(final_eq, final_eq[1:]))
    # Sharpe is non-increasing as costs rise (a drag, never a boost).
    assert all(b <= a + 1e-12 for a, b in zip(sharpes, sharpes[1:]))


def test_cost_stress_does_not_mutate_config(
    synthetic_prices: pd.DataFrame, cfg: Config
) -> None:
    """The frozen base config is never mutated by the cost-stress replay."""
    snapshot = dataclasses.asdict(cfg)
    weights = _weights(synthetic_prices, cfg)
    cost_stress_test(synthetic_prices, weights, cfg)
    assert dataclasses.asdict(cfg) == snapshot
    assert cfg.cost_bps_per_side == snapshot["cost_bps_per_side"]


def test_cost_stress_base_matches_direct_backtest(
    synthetic_prices: pd.DataFrame, cfg: Config
) -> None:
    """The base-cost stress row equals a direct backtest at the base cost."""
    from trend_robot.backtest.engine import run_backtest
    from trend_robot.metrics.performance import performance_metrics

    weights = _weights(synthetic_prices, cfg)
    result = run_backtest(synthetic_prices, weights, cfg)
    metrics = performance_metrics(result, cfg)

    rows = cost_stress_test(synthetic_prices, weights, cfg)
    base_row = next(r for r in rows if r.is_base)
    assert base_row.total_cost == float(metrics["total_cost"])
    assert base_row.sharpe == float(metrics["sharpe"])
    assert base_row.final_equity == float(result.equity.iloc[-1])


# ===========================================================================
# Section 6.4 / 6.5 / 11 -- final validation report
# ===========================================================================
def test_final_report_measured_on_locked_test_set_only(
    synthetic_prices: pd.DataFrame, cfg: Config
) -> None:
    """The report's test slice is exactly the locked OOS split (spec 6.1)."""
    weights = _weights(synthetic_prices, cfg)
    _, test_index = train_test_split(synthetic_prices.index, cfg)

    report = evaluate_final_validation(
        synthetic_prices, weights, cfg, n_trials=1
    )
    assert isinstance(report, FinalValidationReport)
    assert report.n_test_bars == len(test_index)
    assert report.test_start == test_index[0]
    assert report.test_end == test_index[-1]


def test_final_verdict_is_and_of_dsr_and_stability(
    synthetic_prices: pd.DataFrame, cfg: Config
) -> None:
    """Spec 6.5: retain ONLY if DSR clearly positive AND walk-forward stable."""
    weights = _weights(synthetic_prices, cfg)
    report = evaluate_final_validation(
        synthetic_prices, weights, cfg, n_trials=1
    )
    assert report.stability is not None
    expected = bool(
        report.dsr_clearly_positive and report.stability.is_stable
    )
    assert report.retain is expected
    # dsr_clearly_positive must agree with the threshold comparison.
    assert report.dsr_clearly_positive == bool(
        np.isfinite(report.test_deflated_sharpe)
        and report.test_deflated_sharpe > report.dsr_threshold
    )


def test_more_trials_never_raises_dsr(
    synthetic_prices: pd.DataFrame, cfg: Config
) -> None:
    """Spec 6.4: a higher multiple-testing count cannot raise the DSR hurdle."""
    weights = _weights(synthetic_prices, cfg)
    dsr_1 = evaluate_final_validation(
        synthetic_prices, weights, cfg, n_trials=1
    ).test_deflated_sharpe
    dsr_50 = evaluate_final_validation(
        synthetic_prices, weights, cfg, n_trials=50
    ).test_deflated_sharpe
    assert np.isfinite(dsr_1) and np.isfinite(dsr_50)
    assert dsr_50 <= dsr_1 + 1e-12


def test_final_report_text_contains_verdict_and_section11_note(
    synthetic_prices: pd.DataFrame, cfg: Config
) -> None:
    """The printed report carries an explicit verdict + the spec-11 note."""
    weights = _weights(synthetic_prices, cfg)
    report = evaluate_final_validation(
        synthetic_prices, weights, cfg, n_trials=1
    )
    text = format_final_report(report)

    assert "SECTION 6.5 RECOMMENDATION" in text
    assert ("RETAIN" in text) or ("REJECT" in text)
    # Section-11: explicitly a human judgment; never auto-deploy / re-tune.
    assert "HUMAN judgment" in text
    assert "never auto-deploys" in text
    assert "re-tunes on the locked test set" in text
    # The DSR, its trials-corrected hurdle and n_trials are reported.
    assert "Deflated Sharpe" in text
    assert f"n_trials={report.n_trials}" in text


def test_final_validation_accepts_trial_counter(
    synthetic_prices: pd.DataFrame, cfg: Config
) -> None:
    """`evaluate_final_validation` accepts a TrialCounter for n_trials."""
    from trend_robot.validation import TrialCounter

    counter = TrialCounter()
    counter.record(("cfgA",))
    counter.record(("cfgB",))
    weights = _weights(synthetic_prices, cfg)
    report = evaluate_final_validation(
        synthetic_prices, weights, cfg, n_trials=counter
    )
    assert report.n_trials == counter.n_trials == 2
