"""Validation entry point: cost sensitivity + locked-test-set verdict (T10).

This is the *honest read-out* the spec demands at the end of a research effort
(spec sections 5, 6.4, 6.5 and 11). It runs end-to-end on the same
(synthetic-by-default, reproducible) data as :mod:`run_research`, computes the
strategy's target weights **once**, and then:

1. **Cost sensitivity (spec section 5).** Replays the identical backtest at the
   base cost and at every level in ``cfg.cost_stress_levels`` and prints a
   comparison table (CAGR / Sharpe / max DD / total cost / final equity). A
   strategy that only survives at low costs is fragile.

2. **Final validation report (spec section 6.5).** On the LOCKED out-of-sample
   test set (the last ``1 - train_test_ratio`` of history, carved off with
   :func:`trend_robot.validation.train_test_split`), it computes the Deflated
   Sharpe Ratio -- corrected with the ``n_trials`` multiple-testing count -- and
   the walk-forward Sharpe stability, then prints an explicit RETAIN/REJECT
   recommendation plus the mandatory section-11 note that the final decision is
   a HUMAN judgment and the system never auto-deploys or re-tunes on the test
   set.

Reproducibility is mandatory: the global RNGs are seeded from ``cfg.seed`` and
the synthetic provider is itself seeded, so a default invocation is fully
deterministic. No market values are hard-coded; everything flows from the typed
:class:`Config` loaded from ``config.yaml``. The locked test set is read exactly
once, for the final verdict -- never for tuning.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from run_research import _date_window, _load_prices
from trend_robot.config import Config, load_config, set_global_seed
from trend_robot.portfolio.sizing import target_weights
from trend_robot.signals.tsmom import tsmom_signal
from trend_robot.validation import (
    FinalValidationReport,
    TrialCounter,
    cost_stress_table,
    cost_stress_test,
    evaluate_final_validation,
    format_final_report,
)
from trend_robot.validation.stress import CostStressRow

_PROJECT_ROOT = Path(__file__).resolve().parent
_DEFAULT_CONFIG = _PROJECT_ROOT / "config.yaml"
_DEFAULT_CACHE_DIR = _PROJECT_ROOT / ".cache"

# Operational backtest length (calendar years of history), not a market value.
# Long enough that several walk-forward windows (5y train + 1y test) fit inside
# the locked test set's portion of history.
_DEFAULT_HISTORY_YEARS: int = 15

_LOGGER = logging.getLogger("trend_robot.run_validation")


@dataclass
class ValidationRun:
    """Everything one validation run produces.

    Attributes
    ----------
    cfg:
        The validated configuration used.
    data_source:
        Which provider supplied the prices (``"synthetic"`` / ``"yfinance"``).
    n_trials:
        The multiple-testing trial count fed into the Deflated Sharpe.
    cost_stress:
        Cost-sensitivity rows (one per cost level), ascending by cost.
    cost_stress_df:
        The same rows as a tidy comparison :class:`pandas.DataFrame`.
    final_report:
        The section-6.5 final validation verdict on the locked test set.
    """

    cfg: Config
    data_source: str
    n_trials: int
    cost_stress: list[CostStressRow]
    cost_stress_df: pd.DataFrame
    final_report: FinalValidationReport


def _record_base_trial(cfg: Config) -> TrialCounter:
    """Build a :class:`TrialCounter` recording this run's base configuration.

    Mirrors :mod:`run_research`: the strategy-relevant subset of the typed
    :class:`Config` is recorded as one trial, so ``n_trials`` is genuinely
    derived from what was searched rather than a hard-coded literal.

    Parameters
    ----------
    cfg:
        The validated configuration.

    Returns
    -------
    TrialCounter
        A counter with the base configuration recorded (``n_trials == 1``).
    """
    counter = TrialCounter()
    counter.record(
        (
            cfg.direction,
            cfg.rebalance,
            tuple(cfg.lookbacks),
            cfg.vol_window,
            cfg.asset_vol_target,
            cfg.portfolio_vol_target,
            cfg.max_gross_leverage,
            cfg.kelly_fraction,
        )
    )
    return counter


def run_validation(
    config_path: str | Path = _DEFAULT_CONFIG,
    *,
    cache_dir: str | Path = _DEFAULT_CACHE_DIR,
    history_years: int = _DEFAULT_HISTORY_YEARS,
    end: str | None = None,
    prefer_yfinance: bool = False,
    n_trials: int | None = None,
) -> ValidationRun:
    """Run cost sensitivity + the locked-test-set validation end-to-end.

    Loads + validates the config, seeds the RNGs, loads prices (deterministic
    synthetic by default), computes the strategy's target weights once, then runs
    the cost-stress replay (spec 5) and the section-6.5 final validation report
    on the locked out-of-sample test set (spec 6.5). The test set is used exactly
    once, for the verdict; nothing is re-tuned on it (spec 11).

    Parameters
    ----------
    config_path:
        Path to ``config.yaml``.
    cache_dir:
        Directory for the parquet price cache.
    history_years:
        Calendar years of price history to request (backtest length).
    end:
        Inclusive end date (``"YYYY-MM-DD"``); defaults to today.
    prefer_yfinance:
        Prefer cached live Yahoo (NOT reproducible) or go straight to the
        deterministic synthetic provider (the default).
    n_trials:
        Override the multiple-testing trial count. ``None`` (default) derives it
        from a :class:`TrialCounter` recording this run's base configuration
        (``n_trials == 1``). Set this to the true number of configurations tested
        across the whole research campaign to impose the correct DSR hurdle.

    Returns
    -------
    ValidationRun
        The full validation bundle (config, source, trials, cost stress, final
        report).
    """
    cfg = load_config(config_path)
    set_global_seed(cfg.seed)  # seed BEFORE any (synthetic) data generation.

    start, end_resolved = _date_window(history_years, end)
    _LOGGER.info(
        "TSMOM VALIDATION run | window %s..%s | universe=%s | rebalance=%s | "
        "direction=%s | seed=%d",
        start,
        end_resolved,
        cfg.universe,
        cfg.rebalance,
        cfg.direction,
        cfg.seed,
    )

    # --- Data (deterministic synthetic by default). -----------------------
    prices, data_source = _load_prices(
        cfg, start, end_resolved, cache_dir, prefer_yfinance=prefer_yfinance
    )

    # --- Strategy logic computed ONCE; reused for every replay/segment. ---
    returns = prices.pct_change(fill_method=None)
    signals = tsmom_signal(prices, cfg.lookbacks, direction=cfg.direction)
    weights = target_weights(signals, returns, cfg)

    # --- Multiple-testing trial count (spec 6.4). -------------------------
    if n_trials is None:
        trials = _record_base_trial(cfg).n_trials
    else:
        trials = max(1, int(n_trials))

    # --- (1) Cost sensitivity (spec section 5). ---------------------------
    cost_rows = cost_stress_test(prices, weights, cfg)
    cost_df = cost_stress_table(cost_rows)

    # --- (2) Final section-6.5 validation on the locked test set. ---------
    final_report = evaluate_final_validation(
        prices, weights, cfg, n_trials=trials
    )

    return ValidationRun(
        cfg=cfg,
        data_source=data_source,
        n_trials=trials,
        cost_stress=cost_rows,
        cost_stress_df=cost_df,
        final_report=final_report,
    )


def _print_cost_stress(run: ValidationRun) -> None:
    """Print the cost-sensitivity comparison table (spec section 5)."""
    print("=" * 70)
    print(f"COST SENSITIVITY  (spec section 5)  [data source: {run.data_source}]")
    print("-" * 70)
    header = (
        f"  {'bps/side':>9}  {'':<5}  {'CAGR':>9}  {'Sharpe':>8}  "
        f"{'MaxDD':>9}  {'TotalCost':>11}  {'FinalEq':>11}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in run.cost_stress:
        tag = "base" if row.is_base else "str"
        print(
            f"  {row.cost_bps_per_side:>9.2f}  {tag:<5}  "
            f"{row.cagr:>9.4f}  {row.sharpe:>8.4f}  "
            f"{row.max_drawdown:>9.4f}  {row.total_cost:>11.2f}  "
            f"{row.final_equity:>11.2f}"
        )
    print("-" * 70)
    # A quick fragility read-out: how much the Sharpe erodes from base to the
    # highest stressed cost level.
    base = next((r for r in run.cost_stress if r.is_base), run.cost_stress[0])
    worst = run.cost_stress[-1]
    print(
        f"  Sharpe @ base ({base.cost_bps_per_side:.0f} bps) = {base.sharpe:.4f}"
        f"  ->  @ {worst.cost_bps_per_side:.0f} bps = {worst.sharpe:.4f}"
    )
    if worst.sharpe <= 0.0 < base.sharpe:
        print(
            "  WARNING: Sharpe turns non-positive under stressed costs -- "
            "the edge looks COST-FRAGILE."
        )
    print("=" * 70)
    print()


def main(argv: list[str] | None = None) -> ValidationRun:
    """CLI entry point: run cost stress + locked-test-set validation.

    Parameters
    ----------
    argv:
        Optional argument vector (defaults to ``sys.argv``).

    Returns
    -------
    ValidationRun
        The completed validation bundle.
    """
    parser = argparse.ArgumentParser(
        description=(
            "TSMOM validation -- cost sensitivity (spec 5) + final "
            "locked-test-set verdict (spec 6.5)."
        )
    )
    parser.add_argument(
        "--config",
        default=str(_DEFAULT_CONFIG),
        help="Path to config.yaml (default: project-root config.yaml).",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(_DEFAULT_CACHE_DIR),
        help="Directory for the parquet price cache (default: ./.cache).",
    )
    parser.add_argument(
        "--history-years",
        type=int,
        default=_DEFAULT_HISTORY_YEARS,
        help=f"Years of price history (default: {_DEFAULT_HISTORY_YEARS}).",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Inclusive end date YYYY-MM-DD (default: today).",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help=(
            "Override the multiple-testing trial count fed into the Deflated "
            "Sharpe (default: derived from the base config = 1). Set to the "
            "true number of configurations searched to raise the hurdle."
        ),
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help=(
            "Opt in to live cached Yahoo Finance prices (falls back to "
            "synthetic on failure). NOTE: not reproducible across fresh "
            "fetches. Default is the deterministic, seeded synthetic provider."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    run = run_validation(
        config_path=args.config,
        cache_dir=args.cache_dir,
        history_years=args.history_years,
        end=args.end,
        prefer_yfinance=args.live,
        n_trials=args.n_trials,
    )

    # --- Print both deliverables. -----------------------------------------
    _print_cost_stress(run)
    print(format_final_report(run.final_report))

    return run


if __name__ == "__main__":
    main()
