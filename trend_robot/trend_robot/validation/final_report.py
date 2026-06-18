"""Final section-6.5 validation report on the locked test set.

This module assembles the *honest verdict* the spec demands at the very end of a
research effort (spec sections 6.4 / 6.5 / 11). Nothing here re-tunes anything:
it only evaluates a single, already-frozen strategy configuration on data it has
never been fitted to.

What it computes
----------------
1. **Locked train/test split (spec 6.1).** The history is split with
   :func:`trend_robot.validation.train_test_split`; everything below is measured
   on the *test* slice only -- the locked out-of-sample set that development
   never touched.

2. **Deflated Sharpe Ratio on the test set (spec 6.4).** The strategy is
   backtested on the test slice; its net-of-cost per-bar returns feed
   :func:`trend_robot.metrics.deflated_sharpe.deflated_sharpe_ratio`, corrected
   with the multiple-testing ``n_trials`` count from a
   :class:`trend_robot.validation.TrialCounter`. More trials => higher hurdle =>
   lower DSR. "Clearly positive" is read as a DSR comfortably above ``0.5`` (the
   probability that the true Sharpe is positive after correction); the threshold
   is configurable.

3. **Walk-forward stability (spec 6.2).** Rolling
   :func:`trend_robot.validation.walk_forward_splits` windows are evaluated and
   the per-window Sharpe ratios are collected. Stability is judged from the
   dispersion of those window Sharpes (how many windows are positive, and the
   spread relative to the mean). A track carried by a single lucky window is
   *not* stable.

4. **Retain / reject verdict (spec 6.5).** A variant is retained **only if**
   the DSR is clearly positive **and** the walk-forward track is stable. The
   decision and the section-11 note (this is a *human* judgment; the system
   never auto-deploys and never re-tunes on the locked test set) are printed
   verbatim.

No market values are hard-coded: the universe, costs, split ratios and
walk-forward windows all come from the typed :class:`Config`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from trend_robot.backtest.engine import run_backtest
from trend_robot.config import Config
from trend_robot.metrics.deflated_sharpe import deflated_sharpe_ratio
from trend_robot.metrics.performance import performance_metrics
from trend_robot.validation.splits import (
    WalkForwardWindow,
    train_test_split,
    walk_forward_splits,
)
from trend_robot.validation.trials import TrialCounter

__all__ = [
    "FinalValidationReport",
    "WalkForwardStability",
    "evaluate_final_validation",
    "format_final_report",
]

# Section-6.5 success thresholds. These are *judgment* thresholds for the
# automated verdict, not market values: they parameterize what "clearly
# positive DSR" and "stable walk-forward" mean for the printed recommendation.
# The human reviewer retains final say (spec 11).
_DEFAULT_DSR_THRESHOLD: float = 0.60
_DEFAULT_MIN_POSITIVE_WINDOW_FRACTION: float = 0.50
_DEFAULT_MAX_SHARPE_DISPERSION: float = 2.0


def _backtest_returns_on(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    cfg: Config,
    index: pd.Index,
) -> tuple[pd.Series, dict]:
    """Backtest the (pre-computed) weights restricted to ``index``.

    The strategy logic is *not* re-run; the supplied ``target_weights`` are
    sliced to ``index`` and replayed through the engine, so the segment is
    evaluated exactly as it would have been in production over that window.

    Parameters
    ----------
    prices:
        Full adjusted-close panel.
    target_weights:
        Full target-weight panel (already computed by the strategy).
    cfg:
        Typed configuration.
    index:
        The slice of dates to evaluate on (e.g. the locked test set or one
        walk-forward test window).

    Returns
    -------
    tuple[pandas.Series, dict]
        The per-bar net-of-cost equity returns on the slice and the performance
        metrics dictionary for the slice.
    """
    px = prices.loc[index]
    weights = target_weights.reindex(index=index)
    result = run_backtest(px, weights, cfg)
    metrics = performance_metrics(result, cfg)
    returns = result.equity.pct_change(fill_method=None)
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    return returns, metrics


def _deflated_sharpe_on_returns(returns: pd.Series, n_trials: int) -> float:
    """Deflated Sharpe of a per-bar return series, corrected for ``n_trials``.

    The skew and (non-excess) kurtosis are estimated from the returns
    themselves; pandas reports *excess* kurtosis, so ``3`` is added back to
    match the DSR formula's convention.

    Parameters
    ----------
    returns:
        Per-bar net-of-cost strategy returns.
    n_trials:
        Multiple-testing trial count (the DSR hurdle grows with this).

    Returns
    -------
    float
        Deflated Sharpe Ratio in ``[0, 1]`` (``nan`` if undefined).
    """
    if len(returns) < 2:
        return float("nan")
    skew = float(returns.skew())
    kurtosis = float(returns.kurt()) + 3.0  # pandas .kurt() is excess kurtosis
    return deflated_sharpe_ratio(
        returns, n_trials=n_trials, skew=skew, kurtosis=kurtosis
    )


@dataclass(frozen=True)
class WalkForwardStability:
    """Walk-forward stability summary on the locked test set (spec 6.2).

    Attributes
    ----------
    n_windows:
        Number of complete walk-forward windows evaluated.
    window_sharpes:
        Per-window annualized Sharpe ratios (chronological order).
    window_returns:
        Per-window total (cumulative) net-of-cost return.
    mean_sharpe:
        Mean of the per-window Sharpes.
    std_sharpe:
        Sample standard deviation of the per-window Sharpes.
    positive_fraction:
        Fraction of windows with a strictly positive Sharpe.
    dispersion:
        ``std_sharpe / |mean_sharpe|`` (coefficient of variation); ``inf`` when
        the mean is ~0. Lower means more stable.
    is_stable:
        Whether the track passes the stability criterion (enough positive
        windows and bounded dispersion).
    """

    n_windows: int
    window_sharpes: list[float]
    window_returns: list[float]
    mean_sharpe: float
    std_sharpe: float
    positive_fraction: float
    dispersion: float
    is_stable: bool


def _walk_forward_stability(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    cfg: Config,
    windows: list[WalkForwardWindow],
    *,
    min_positive_fraction: float,
    max_dispersion: float,
) -> WalkForwardStability:
    """Evaluate per-window Sharpe dispersion across walk-forward windows.

    Parameters
    ----------
    prices, target_weights, cfg:
        Strategy inputs (weights pre-computed and reused per window).
    windows:
        Walk-forward windows to evaluate (their ``test_index`` slices).
    min_positive_fraction:
        Minimum fraction of windows that must have a positive Sharpe for the
        track to be considered stable.
    max_dispersion:
        Maximum tolerated Sharpe coefficient of variation
        (``std / |mean|``) for stability.

    Returns
    -------
    WalkForwardStability
        The stability summary (see the dataclass).
    """
    sharpes: list[float] = []
    win_returns: list[float] = []
    for window in windows:
        returns, metrics = _backtest_returns_on(
            prices, target_weights, cfg, window.test_index
        )
        sharpe = float(metrics["sharpe"])
        sharpes.append(sharpe)
        # Total compounded net-of-cost return over the window.
        win_returns.append(float((1.0 + returns).prod() - 1.0))

    finite = np.array([s for s in sharpes if np.isfinite(s)], dtype="float64")
    n_windows = len(windows)
    if finite.size == 0:
        return WalkForwardStability(
            n_windows=n_windows,
            window_sharpes=sharpes,
            window_returns=win_returns,
            mean_sharpe=float("nan"),
            std_sharpe=float("nan"),
            positive_fraction=float("nan"),
            dispersion=float("inf"),
            is_stable=False,
        )

    mean_sharpe = float(finite.mean())
    std_sharpe = float(finite.std(ddof=1)) if finite.size >= 2 else 0.0
    positive_fraction = float((finite > 0.0).mean())
    if abs(mean_sharpe) > 1e-9:
        dispersion = float(std_sharpe / abs(mean_sharpe))
    else:
        dispersion = float("inf")

    is_stable = (
        n_windows >= 2
        and mean_sharpe > 0.0
        and positive_fraction >= min_positive_fraction
        and dispersion <= max_dispersion
    )
    return WalkForwardStability(
        n_windows=n_windows,
        window_sharpes=sharpes,
        window_returns=win_returns,
        mean_sharpe=mean_sharpe,
        std_sharpe=std_sharpe,
        positive_fraction=positive_fraction,
        dispersion=dispersion,
        is_stable=is_stable,
    )


@dataclass(frozen=True)
class FinalValidationReport:
    """The section-6.5 verdict on the locked test set.

    Attributes
    ----------
    n_test_bars:
        Number of bars in the locked out-of-sample test set.
    test_start, test_end:
        First/last test-set dates (or ``None`` if empty).
    n_trials:
        Multiple-testing trial count fed into the DSR.
    test_metrics:
        Full performance-metrics dict on the locked test set.
    test_deflated_sharpe:
        Deflated Sharpe Ratio on the locked test set, corrected for
        ``n_trials``.
    dsr_threshold:
        Threshold above which the DSR is treated as "clearly positive".
    dsr_clearly_positive:
        Whether ``test_deflated_sharpe`` exceeds ``dsr_threshold``.
    stability:
        Walk-forward stability summary.
    retain:
        Final automated recommendation: retain only if the DSR is clearly
        positive AND the walk-forward track is stable.
    """

    n_test_bars: int
    test_start: pd.Timestamp | None
    test_end: pd.Timestamp | None
    n_trials: int
    test_metrics: dict = field(repr=False)
    test_deflated_sharpe: float = float("nan")
    dsr_threshold: float = _DEFAULT_DSR_THRESHOLD
    dsr_clearly_positive: bool = False
    stability: WalkForwardStability | None = None
    retain: bool = False


def evaluate_final_validation(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    cfg: Config,
    *,
    n_trials: int | TrialCounter,
    dsr_threshold: float = _DEFAULT_DSR_THRESHOLD,
    min_positive_window_fraction: float = _DEFAULT_MIN_POSITIVE_WINDOW_FRACTION,
    max_sharpe_dispersion: float = _DEFAULT_MAX_SHARPE_DISPERSION,
) -> FinalValidationReport:
    """Run the section-6.5 final validation on the locked test set.

    The locked test slice (spec 6.1) is carved off with
    :func:`train_test_split`; all metrics below are measured there. The Deflated
    Sharpe Ratio (spec 6.4) is computed on the test-set net-of-cost returns and
    corrected for ``n_trials``; walk-forward stability (spec 6.2) is computed
    from the dispersion of per-window Sharpes over the *whole* history's rolling
    windows. The variant is retained only if the DSR is clearly positive AND the
    walk-forward track is stable (spec 6.5).

    The pre-computed ``target_weights`` are reused throughout: no strategy logic
    is re-run and, critically, nothing is re-tuned on the test set (spec 11).

    Parameters
    ----------
    prices:
        Full adjusted-close panel.
    target_weights:
        Full target-weight panel from the (already frozen) strategy.
    cfg:
        Typed configuration (split ratio, walk-forward windows, costs, ...).
    n_trials:
        Multiple-testing trial count, or a :class:`TrialCounter` (its
        ``n_trials`` is used). Larger raises the DSR hurdle.
    dsr_threshold:
        DSR above which the result is "clearly positive" (default ``0.60``).
    min_positive_window_fraction:
        Minimum fraction of walk-forward windows that must be positive for
        stability (default ``0.50``).
    max_sharpe_dispersion:
        Maximum tolerated per-window Sharpe coefficient of variation for
        stability (default ``2.0``).

    Returns
    -------
    FinalValidationReport
        The full verdict (see the dataclass).
    """
    trials = int(n_trials) if not isinstance(n_trials, int) else n_trials

    # --- Locked train/test split (spec 6.1); evaluate ONLY on test. --------
    _, test_index = train_test_split(prices.index, cfg)

    if len(test_index) >= 2:
        test_returns, test_metrics = _backtest_returns_on(
            prices, target_weights, cfg, test_index
        )
        dsr = _deflated_sharpe_on_returns(test_returns, trials)
        test_start = test_index[0]
        test_end = test_index[-1]
    else:
        test_returns = pd.Series(dtype="float64")
        test_metrics = {}
        dsr = float("nan")
        test_start = test_index[0] if len(test_index) else None
        test_end = test_index[-1] if len(test_index) else None

    dsr_clearly_positive = bool(np.isfinite(dsr) and dsr > dsr_threshold)

    # --- Walk-forward stability (spec 6.2). --------------------------------
    windows = walk_forward_splits(prices.index, cfg)
    stability = _walk_forward_stability(
        prices,
        target_weights,
        cfg,
        windows,
        min_positive_fraction=min_positive_window_fraction,
        max_dispersion=max_sharpe_dispersion,
    )

    # --- Section-6.5 verdict: retain ONLY if both conditions hold. ---------
    retain = bool(dsr_clearly_positive and stability.is_stable)

    return FinalValidationReport(
        n_test_bars=int(len(test_index)),
        test_start=test_start,
        test_end=test_end,
        n_trials=trials,
        test_metrics=test_metrics,
        test_deflated_sharpe=dsr,
        dsr_threshold=dsr_threshold,
        dsr_clearly_positive=dsr_clearly_positive,
        stability=stability,
        retain=retain,
    )


def _fmt(value: object) -> str:
    """Format a metric value for the printed report (floats to 4 dp)."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if not np.isfinite(value):
            return str(value)
        return f"{value:.4f}"
    return str(value)


def format_final_report(report: FinalValidationReport) -> str:
    """Render the section-6.5 verdict as a human-readable text block.

    The output includes the locked-test-set summary, the Deflated Sharpe (with
    its trials-corrected hurdle), the per-window walk-forward Sharpes, an
    explicit RETAIN/REJECT recommendation, and the mandatory section-11 note
    that the final call is a human judgment and the system never auto-deploys or
    re-tunes on the locked test set.

    Parameters
    ----------
    report:
        The evaluated :class:`FinalValidationReport`.

    Returns
    -------
    str
        A multi-line, ready-to-print report.
    """
    lines: list[str] = []
    bar = "=" * 70
    lines.append(bar)
    lines.append("FINAL VALIDATION REPORT  (spec section 6.5 -- LOCKED TEST SET)")
    lines.append("-" * 70)
    lines.append(
        f"  Locked OOS test set : {report.test_start} -> {report.test_end} "
        f"({report.n_test_bars} bars)"
    )
    lines.append(f"  Trials (n_trials)   : {report.n_trials}")
    lines.append("")

    # --- (a) Deflated Sharpe ----------------------------------------------
    m = report.test_metrics
    lines.append("  (a) Deflated Sharpe Ratio on the locked test set (spec 6.4)")
    lines.append(f"        Test Sharpe (annualized) : {_fmt(m.get('sharpe'))}")
    lines.append(f"        Test CAGR                : {_fmt(m.get('cagr'))}")
    lines.append(f"        Test max drawdown        : {_fmt(m.get('max_drawdown'))}")
    lines.append(
        f"        Deflated Sharpe (DSR)    : {_fmt(report.test_deflated_sharpe)} "
        f"(threshold {report.dsr_threshold:.2f}, n_trials={report.n_trials})"
    )
    verdict_a = "PASS" if report.dsr_clearly_positive else "FAIL"
    lines.append(f"        => DSR clearly positive  : {verdict_a}")
    lines.append("")

    # --- (b) Walk-forward stability ---------------------------------------
    s = report.stability
    lines.append("  (b) Walk-forward stability (spec 6.2)")
    if s is None or s.n_windows == 0:
        lines.append(
            "        No complete walk-forward windows fit in the history; "
            "stability cannot be assessed."
        )
        verdict_b = "FAIL"
    else:
        lines.append(f"        Windows evaluated        : {s.n_windows}")
        sharpe_str = ", ".join(_fmt(x) for x in s.window_sharpes)
        lines.append(f"        Per-window Sharpe        : [{sharpe_str}]")
        lines.append(f"        Mean / std Sharpe        : {_fmt(s.mean_sharpe)}"
                     f" / {_fmt(s.std_sharpe)}")
        lines.append(f"        Positive-window fraction : {_fmt(s.positive_fraction)}")
        lines.append(f"        Sharpe dispersion (cv)   : {_fmt(s.dispersion)}")
        verdict_b = "PASS" if s.is_stable else "FAIL"
    lines.append(f"        => Walk-forward stable    : {verdict_b}")
    lines.append("")

    # --- Final verdict -----------------------------------------------------
    lines.append("-" * 70)
    recommendation = "RETAIN" if report.retain else "REJECT (DO NOT DEPLOY)"
    lines.append(f"  SECTION 6.5 RECOMMENDATION : {recommendation}")
    lines.append(
        "  Criterion: retain ONLY if the Deflated Sharpe is clearly positive "
        "AND"
    )
    lines.append("             the walk-forward performance is stable.")
    lines.append("")
    # --- Section-11 human-judgment note -----------------------------------
    lines.append(
        "  NOTE (spec 11): This RETAIN/REJECT recommendation is an automated"
    )
    lines.append(
        "  summary, NOT a decision. The final call is a HUMAN judgment. This"
    )
    lines.append(
        "  system never auto-deploys and never re-tunes on the locked test set;"
    )
    lines.append(
        "  the test set was used here exactly once, for this final read-out."
    )
    lines.append(bar)
    return "\n".join(lines)
