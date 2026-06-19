"""Forward (pristine) / retrospective (non-pristine) hold-out entry point.

This CLI consumes a *frozen* pre-registration record (see
:mod:`trend_robot.validation.preregistration`) and reads section 6.5 on the
hold-out slice it defines:

* ``--mode forward`` (default) -- the PRISTINE forward hold-out: bars dated
  strictly after the record's ``decision_date``. Until enough such bars exist,
  the report says "accrue more data".
* ``--mode retrospective`` -- a clearly-labelled NON-PRISTINE diagnostic on the
  last ``--retrospective-months`` months of data, so a number can be produced
  today on existing history.

Flow
----
``load_config`` -> ``set_global_seed`` -> ``load_decision`` ->
``verify_config_matches`` (warn loudly on drift) -> ``_load_prices`` (synthetic
by default; opt-in cached Yahoo via ``--live``) -> ``evaluate_holdout`` ->
print ``format_holdout_report``.

No market values are hard-coded: every strategy/backtest parameter flows from
``config.yaml`` through the typed :class:`~trend_robot.config.Config`. Only the
operational window (history length, end date, cache dir) is set on the CLI.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

# Reuse the research pipeline's data-window + price-loading helpers so the
# synthetic fallback (Yahoo 429) and cache semantics are identical.
from run_research import _date_window, _load_prices
from trend_robot.config import load_config, set_global_seed
from trend_robot.validation.holdout import evaluate_holdout, format_holdout_report
from trend_robot.validation.preregistration import (
    load_decision,
    verify_config_matches,
)

_PROJECT_ROOT = Path(__file__).resolve().parent
_DEFAULT_CONFIG = _PROJECT_ROOT / "config.yaml"
_DEFAULT_CACHE_DIR = _PROJECT_ROOT / ".cache"
_DEFAULT_DECISION = _PROJECT_ROOT / "decision_record.json"

# Default research window: how many calendar years of history to pull. This is
# an operational length (so the no-look-ahead warm-up has enough pre-hold-out
# history), not a market value.
_DEFAULT_HISTORY_YEARS: int = 15

_LOGGER = logging.getLogger("trend_robot.run_holdout")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: evaluate the frozen decision on its hold-out slice.

    Parameters
    ----------
    argv:
        Optional argument vector (defaults to ``sys.argv``). Provided so tests
        can drive the CLI in-process without spawning a subprocess.

    Returns
    -------
    int
        Process exit code (``0`` on success).
    """
    parser = argparse.ArgumentParser(
        description=(
            "TSMOM forward/retrospective hold-out -- read section 6.5 on bars "
            "after a frozen pre-registration decision."
        )
    )
    parser.add_argument(
        "--config",
        default=str(_DEFAULT_CONFIG),
        help="Path to config.yaml (default: project-root config.yaml).",
    )
    parser.add_argument(
        "--decision",
        default=str(_DEFAULT_DECISION),
        help=(
            "Path to the frozen decision_record.json "
            "(default: project-root decision_record.json)."
        ),
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
        "--mode",
        choices=("forward", "retrospective"),
        default="forward",
        help=(
            "forward = PRISTINE hold-out (bars after the decision date); "
            "retrospective = NON-PRISTINE diagnostic on the last N months "
            "(default: forward)."
        ),
    )
    parser.add_argument(
        "--retrospective-months",
        type=int,
        default=12,
        help="Months of data for retrospective mode (default: 12).",
    )
    parser.add_argument(
        "--min-bars",
        type=int,
        default=None,
        help=(
            "Minimum hold-out bars required for a verdict "
            "(default: one year = cfg.periods_per_year)."
        ),
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help=(
            "Opt in to live cached Yahoo Finance prices (falls back to "
            "synthetic on failure). Default is the deterministic synthetic "
            "provider."
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

    # --- Config + reproducibility -----------------------------------------
    cfg = load_config(args.config)
    set_global_seed(cfg.seed)

    # --- Frozen pre-registration ------------------------------------------
    record = load_decision(args.decision)
    _LOGGER.info(
        "Loaded frozen decision: date=%s hash=%s n_trials_spent=%d",
        record.decision_date,
        record.config_hash,
        record.n_trials_spent,
    )

    # --- Drift check: warn LOUDLY if the live config no longer matches. ----
    if not verify_config_matches(cfg, record):
        _LOGGER.warning(
            "CONFIG DRIFT DETECTED: the current config does NOT match the frozen "
            "decision (hash %s). The pristine forward read is INVALID until the "
            "config is restored or a new decision is deliberately re-registered.",
            record.config_hash,
        )
        print(
            "\n*** WARNING: CONFIG DRIFT -- current config does not match the "
            "frozen decision hash. The forward hold-out below is NOT a valid "
            "pristine read for the current config. ***\n"
        )

    # --- Data (synthetic by default; opt-in cached Yahoo with fallback). ---
    start, end_resolved = _date_window(args.history_years, args.end)
    prices, data_source = _load_prices(
        cfg, start, end_resolved, args.cache_dir, prefer_yfinance=args.live
    )
    _LOGGER.info(
        "Hold-out run | window %s..%s | source=%s | mode=%s",
        start,
        end_resolved,
        data_source,
        args.mode,
    )

    # --- Evaluate + print -------------------------------------------------
    report = evaluate_holdout(
        prices,
        cfg,
        record,
        mode=args.mode,
        retrospective_months=args.retrospective_months,
        min_bars=args.min_bars,
    )
    print(format_holdout_report(report, record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
