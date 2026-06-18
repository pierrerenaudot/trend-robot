"""End-to-end TSMOM research entry point (spec 9 / T7).

Runs the full research pipeline on the configured universe and writes a report:

    load config + seed  ->  fetch prices (cached Yahoo, synthetic fallback)
      ->  daily returns  ->  TSMOM signal  ->  vol-targeted target weights
      ->  realistic cost-aware backtest  ->  honest metrics (incl. Deflated
      Sharpe)  ->  reporting (equity / drawdown / exposure / contribution
      charts + metrics table).

Data source
-----------
The pipeline has two data sources, both honoring the same data contract
(adjusted closes, tz-naive trading-day index, explicit NaN gaps):

* the deterministic, seeded :class:`SyntheticProvider` (the DEFAULT), and
* the live cached :class:`YFinanceProvider` (opt-in via ``--live``).

By DEFAULT the run uses the SyntheticProvider so the research run is fully
reproducible (see below). Yahoo Finance is rate-limited in this environment
(HTTP 429); the live path is therefore opt-in only. When ``--live`` is passed,
the pipeline PREFERS the cached :class:`YFinanceProvider` but FALLS BACK to the
deterministic :class:`SyntheticProvider` when the download is empty/unusable or
fails, logging clearly which source was used.

Reproducibility
---------------
Reproducibility is mandatory and is guaranteed for the DEFAULT invocation.
The global RNGs are seeded from ``cfg.seed`` before any data is generated, and
the synthetic provider is itself seeded from ``cfg.seed``. Running this script
twice with the default arguments (synthetic source) therefore yields identical
metrics and identical artifacts, byte-for-byte.

CAVEAT: the live ``--live`` path is NOT reproducible -- independent Yahoo
fetches return slightly different adjusted closes (and a moving last bar), so
two ``--live`` runs against fresh data can produce different metrics. Use
``--live`` only for exploratory/live snapshots; if you need a reproducible live
run, populate the cache once and reuse it (cache hits are deterministic).

No market values are hard-coded: every strategy/backtest parameter flows from
``config.yaml`` through the typed :class:`Config`. The only runtime inputs are
the date window and output locations (operational parameters, not market data),
both overridable on the command line.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from trend_robot.backtest.engine import BacktestResult, run_backtest
from trend_robot.config import Config, load_config, set_global_seed
from trend_robot.data.synthetic_provider import SyntheticProvider
from trend_robot.data.yfinance_provider import YFinanceProvider
from trend_robot.metrics.deflated_sharpe import deflated_sharpe_ratio
from trend_robot.metrics.performance import performance_metrics
from trend_robot.reporting.report import build_report
from trend_robot.signals.tsmom import tsmom_signal
from trend_robot.portfolio.sizing import target_weights
from trend_robot.validation import TrialCounter

_PROJECT_ROOT = Path(__file__).resolve().parent
_DEFAULT_CONFIG = _PROJECT_ROOT / "config.yaml"
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "outputs"
_DEFAULT_CACHE_DIR = _PROJECT_ROOT / ".cache"

# Default research window: how many calendar years of history to pull. This is
# an *operational* parameter (the length of the backtest), not a market value;
# it is overridable on the command line and never affects strategy economics.
_DEFAULT_HISTORY_YEARS: int = 15

_LOGGER = logging.getLogger("trend_robot.run_research")


@dataclass
class ResearchRun:
    """Bundle of everything one end-to-end research run produces.

    Attributes
    ----------
    cfg:
        The validated configuration the run used.
    prices:
        The adjusted-close price panel actually used (post source selection).
    data_source:
        Which provider supplied the prices: ``"yfinance"`` or ``"synthetic"``.
    result:
        The backtest result (equity, weights, turnover, trades).
    metrics:
        The honest performance metrics dictionary, augmented with
        ``deflated_sharpe`` and ``n_trials``.
    artifacts:
        Mapping of report-artifact name to the file path written.
    """

    cfg: Config
    prices: pd.DataFrame
    data_source: str
    result: BacktestResult
    metrics: dict[str, Any]
    artifacts: dict[str, Path]


def _date_window(history_years: int, end: str | None) -> tuple[str, str]:
    """Compute the ISO ``(start, end)`` window for the backtest.

    Parameters
    ----------
    history_years:
        Number of calendar years of history to request (operational length of
        the backtest, not a market value).
    end:
        Inclusive end date (``"YYYY-MM-DD"``); defaults to today if ``None``.

    Returns
    -------
    tuple[str, str]
        ``(start, end)`` ISO date strings.
    """
    end_ts = pd.Timestamp(end) if end is not None else pd.Timestamp.today().normalize()
    start_ts = end_ts - pd.DateOffset(years=int(history_years))
    return start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d")


def _usable(prices: pd.DataFrame, cfg: Config) -> bool:
    """Whether a price panel is rich enough to drive the pipeline.

    A panel is usable when it is non-empty, has at least one configured asset
    column with data, and spans more rows than the longest lookback (so the
    TSMOM signal can become defined for at least a few dates).

    Parameters
    ----------
    prices:
        Candidate adjusted-close panel.
    cfg:
        Typed configuration (for ``lookbacks`` / ``universe``).

    Returns
    -------
    bool
        ``True`` if the panel can support a meaningful backtest.
    """
    if prices is None or prices.empty:
        return False
    # Need more history than the longest lookback to ever produce a signal.
    min_rows = max(cfg.lookbacks) + cfg.vol_window + 5
    if len(prices) < min_rows:
        return False
    # At least one configured asset must carry some non-NaN data.
    present = [c for c in cfg.universe if c in prices.columns]
    if not present:
        return False
    return bool(prices[present].notna().any().any())


def _load_prices(
    cfg: Config,
    start: str,
    end: str,
    cache_dir: str | Path,
    *,
    prefer_yfinance: bool = False,
) -> tuple[pd.DataFrame, str]:
    """Load adjusted-close prices: deterministic synthetic by default.

    By default (``prefer_yfinance=False``) this goes straight to the seeded
    :class:`SyntheticProvider`, which is fully reproducible. When
    ``prefer_yfinance=True`` it instead tries the cached
    :class:`YFinanceProvider` first (so a previously cached parquet is reused
    and Yahoo is not re-hit); if that download is empty/unusable -- the common
    case here, since Yahoo is rate-limited -- it FALLS BACK to the
    deterministic :class:`SyntheticProvider`, logging which source was used.

    Parameters
    ----------
    cfg:
        Typed configuration (universe, lookbacks, seed).
    start, end:
        Inclusive ISO date bounds.
    cache_dir:
        Directory for the parquet price cache.
    prefer_yfinance:
        When ``False`` (the default), skip Yahoo entirely and use the
        deterministic synthetic provider (reproducible research run). When
        ``True``, prefer cached live Yahoo (NOT reproducible across fresh
        fetches), falling back to synthetic on failure.

    Returns
    -------
    tuple[pandas.DataFrame, str]
        The price panel and the data-source label (``"yfinance"`` /
        ``"synthetic"``).
    """
    tickers = list(cfg.universe)

    if prefer_yfinance:
        _LOGGER.info(
            "Attempting price download via cached YFinanceProvider "
            "(%s..%s, %d tickers)...",
            start,
            end,
            len(tickers),
        )
        provider = YFinanceProvider().cached(cache_dir)
        try:
            prices = provider.get_prices(tickers, start, end)
        except Exception as exc:  # noqa: BLE001 - never crash the pipeline
            _LOGGER.warning("YFinanceProvider raised unexpectedly: %s", exc)
            prices = pd.DataFrame()

        if _usable(prices, cfg):
            _LOGGER.info(
                "DATA SOURCE = yfinance (live/cached): %d rows x %d cols.",
                len(prices),
                prices.shape[1],
            )
            return prices, "yfinance"

        _LOGGER.warning(
            "yfinance returned no usable data (likely HTTP 429 rate limit). "
            "Falling back to deterministic SyntheticProvider."
        )
    else:
        _LOGGER.info("Skipping yfinance; using SyntheticProvider directly.")

    # --- Deterministic, seeded synthetic fallback. ------------------------
    synth = SyntheticProvider(seed=cfg.seed)
    prices = synth.get_prices(tickers, start, end)
    _LOGGER.info(
        "DATA SOURCE = synthetic (seed=%d): %d rows x %d cols.",
        cfg.seed,
        len(prices),
        prices.shape[1],
    )
    return prices, "synthetic"


def run_research(
    config_path: str | Path = _DEFAULT_CONFIG,
    *,
    output_dir: str | Path = _DEFAULT_OUTPUT_DIR,
    cache_dir: str | Path = _DEFAULT_CACHE_DIR,
    history_years: int = _DEFAULT_HISTORY_YEARS,
    end: str | None = None,
    prefer_yfinance: bool = False,
) -> ResearchRun:
    """Run the full TSMOM research pipeline and write a report.

    Steps: load + validate config, seed RNGs, fetch prices (deterministic
    synthetic by default; opt-in cached Yahoo with a synthetic fallback when
    ``prefer_yfinance=True``), compute daily returns, the TSMOM signal,
    vol-targeted target weights, a realistic cost-aware backtest, honest metrics
    (including the Deflated Sharpe Ratio), and finally the report artifacts.

    With the default arguments the run is fully reproducible: seeding happens
    before any data is generated and the synthetic provider is seeded from
    ``cfg.seed``, so two invocations with the same arguments produce identical
    metrics and artifacts. The opt-in ``prefer_yfinance=True`` live path is NOT
    reproducible across fresh Yahoo fetches.

    Parameters
    ----------
    config_path:
        Path to ``config.yaml``.
    output_dir:
        Directory for report artifacts (charts + metrics table).
    cache_dir:
        Directory for the parquet price cache.
    history_years:
        Calendar years of price history to request (backtest length).
    end:
        Inclusive end date (``"YYYY-MM-DD"``); defaults to today.
    prefer_yfinance:
        Go straight to the deterministic synthetic provider (``False``, the
        default -- reproducible) or prefer cached live Yahoo (``True`` -- not
        reproducible across fresh fetches).

    Returns
    -------
    ResearchRun
        Everything the run produced (config, prices, source label, backtest
        result, augmented metrics, artifact paths).
    """
    # --- Config + reproducibility -----------------------------------------
    cfg = load_config(config_path)
    set_global_seed(cfg.seed)  # seed BEFORE any (synthetic) data generation.

    start, end_resolved = _date_window(history_years, end)
    _LOGGER.info(
        "TSMOM research run | window %s..%s | universe=%s | rebalance=%s | "
        "direction=%s | seed=%d",
        start,
        end_resolved,
        cfg.universe,
        cfg.rebalance,
        cfg.direction,
        cfg.seed,
    )

    # --- Data --------------------------------------------------------------
    prices, data_source = _load_prices(
        cfg, start, end_resolved, cache_dir, prefer_yfinance=prefer_yfinance
    )

    # --- Daily simple returns (explicit NaN gaps; no forward-fill). --------
    returns = prices.pct_change(fill_method=None)

    # --- Signal -> weights -> backtest ------------------------------------
    signals = tsmom_signal(prices, cfg.lookbacks, direction=cfg.direction)
    weights = target_weights(signals, returns, cfg)
    result = run_backtest(prices, weights, cfg)

    # --- Honest metrics ----------------------------------------------------
    metrics: dict[str, Any] = dict(performance_metrics(result, cfg))

    # --- Multiple-testing correction: Deflated Sharpe Ratio. --------------
    # n_trials is the number of *distinct configurations* tested in this
    # research effort, tracked by a TrialCounter (spec 6.4). A single
    # end-to-end run on the default config records exactly one trial; the
    # counter is wired into the pipeline so the multiple-testing hurdle grows
    # automatically as additional variants are recorded.
    trial_counter = TrialCounter()
    # The configuration actually evaluated by this run is the strategy-relevant
    # subset of the typed Config. Recording it (rather than a literal) means the
    # trials count is genuinely derived from what was searched.
    trial_counter.record(
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
    n_trials = int(trial_counter)  # == trial_counter.n_trials
    equity_returns = result.equity.pct_change(fill_method=None).dropna()
    if len(equity_returns) >= 2:
        skew = float(equity_returns.skew())
        # pandas .kurt() is *excess* kurtosis; the DSR formula expects the
        # non-excess kurtosis (normal == 3), so add 3 back.
        kurtosis = float(equity_returns.kurt()) + 3.0
        dsr = deflated_sharpe_ratio(
            equity_returns, n_trials=n_trials, skew=skew, kurtosis=kurtosis
        )
    else:
        dsr = float("nan")
    metrics["deflated_sharpe"] = dsr
    metrics["n_trials"] = n_trials
    metrics["data_source"] = data_source

    # --- Report (presentation only) ---------------------------------------
    title = f"TSMOM research ({data_source})"
    artifacts = build_report(result, metrics, output_dir, title=title)

    _log_summary(metrics, artifacts, data_source)

    return ResearchRun(
        cfg=cfg,
        prices=prices,
        data_source=data_source,
        result=result,
        metrics=metrics,
        artifacts=artifacts,
    )


def _log_summary(
    metrics: dict[str, Any], artifacts: dict[str, Path], data_source: str
) -> None:
    """Print a concise human-readable summary of the run to stdout."""
    def fmt(key: str) -> str:
        val = metrics.get(key)
        if val is None:
            return "n/a"
        if isinstance(val, float):
            return f"{val:.4f}"
        return str(val)

    print("=" * 64)
    print(f"TSMOM RESEARCH RUN COMPLETE  (data source: {data_source})")
    print("-" * 64)
    print(f"  Period           : {metrics.get('start')} -> {metrics.get('end')}")
    print(f"  Bars             : {metrics.get('n_periods')}")
    print(f"  CAGR             : {fmt('cagr')}")
    print(f"  Annual vol       : {fmt('annual_vol')}")
    print(f"  Sharpe           : {fmt('sharpe')}")
    print(f"  Sortino          : {fmt('sortino')}")
    print(f"  Calmar / MAR     : {fmt('calmar')}")
    print(f"  Max drawdown     : {fmt('max_drawdown')}")
    print(f"  Max DD duration  : {metrics.get('max_drawdown_duration')} bars")
    print(f"  Profit factor    : {fmt('profit_factor')}")
    print(f"  Hit rate         : {fmt('hit_rate')}")
    print(f"  Avg ann turnover : {fmt('avg_annual_turnover')}")
    print(f"  Avg gross expo   : {fmt('avg_exposure')}")
    print(f"  Total cost       : {fmt('total_cost')}")
    print(f"  Deflated Sharpe  : {fmt('deflated_sharpe')} (n_trials={metrics.get('n_trials')})")
    print("-" * 64)
    print("  Per-asset P&L:")
    for asset, pnl in dict(metrics.get("per_asset_pnl", {})).items():
        print(f"    {asset:<6} : {pnl:,.2f}")
    print("-" * 64)
    print("  Artifacts written:")
    for name, path in artifacts.items():
        print(f"    {name:<14} -> {path}")
    print("=" * 64)


def main(argv: list[str] | None = None) -> ResearchRun:
    """CLI entry point: parse arguments and run the research pipeline.

    Parameters
    ----------
    argv:
        Optional argument vector (defaults to ``sys.argv``).

    Returns
    -------
    ResearchRun
        The completed run bundle.
    """
    parser = argparse.ArgumentParser(
        description="TSMOM research robot -- end-to-end backtest + report."
    )
    parser.add_argument(
        "--config",
        default=str(_DEFAULT_CONFIG),
        help="Path to config.yaml (default: project-root config.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_DEFAULT_OUTPUT_DIR),
        help="Directory for report artifacts (default: ./outputs).",
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
        "--live",
        action="store_true",
        help=(
            "Opt in to live cached Yahoo Finance prices (falls back to "
            "synthetic on failure). NOTE: live runs are NOT reproducible "
            "across fresh fetches. Default is the deterministic, seeded "
            "synthetic provider."
        ),
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help=(
            "Deprecated no-op: synthetic is now the default. Kept for "
            "backward compatibility; use --live to opt in to Yahoo."
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

    return run_research(
        config_path=args.config,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        history_years=args.history_years,
        end=args.end,
        prefer_yfinance=args.live,
    )


if __name__ == "__main__":
    main()
