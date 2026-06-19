"""Pristine forward (and non-pristine retrospective) hold-out evaluator.

This is the *evaluation* half of the forward-hold-out protocol; the *freeze*
half lives in :mod:`trend_robot.validation.preregistration`.

Why it exists
-------------
The section-6.5 ``RETAIN`` verdict on the locked test set is contaminated once
a variant was *selected* using exploration that peeked at all historical data.
After that peek, no past window is pristine. The only genuinely untouched
out-of-sample evidence is the FUTURE: freeze the decision now, then read
section 6.5 again on bars that arrive **strictly after** the decision date.
That forward slice is exactly what the paper-trading track accrues over time.

Two modes
---------
* ``mode='forward'`` -- **PRISTINE.** The hold-out is the set of bars dated
  strictly after ``record.decision_date``. Until enough such bars exist, the
  report says so (``sufficient=False``) and the caller prints "accrue more
  data" rather than a verdict.
* ``mode='retrospective'`` -- **NON-PRISTINE.** The hold-out is the last
  ``retrospective_months`` months of available data. This is *not* untouched
  (those bars were visible during selection), so it is loudly labelled as a
  diagnostic-only check that produces a number today.

No look-ahead
-------------
Mirroring :mod:`trend_robot.validation.final_report`, target weights are
computed once on the **full** price history (each date's signal/sizing uses only
prior prices), and only then sliced to the hold-out index before
:func:`run_backtest`. The weight at a hold-out date therefore depends only on
data up to that date -- prepending or dropping pre-hold-out bars cannot change
the hold-out metrics.

Deflated Sharpe at two trial counts
-----------------------------------
On a sufficient hold-out the DSR is computed twice on the slice's net-of-cost
returns:

* ``dsr_preregistered`` at ``n_trials=1`` -- a single pre-registered test on
  fresh data (no multiple-testing inflation), and
* ``dsr_carried`` at ``n_trials=record.n_trials_spent`` -- conservatively
  carrying the selection search's multiple-testing hurdle into the forward read.

Because the hurdle grows with ``n_trials``, ``dsr_carried <= dsr_preregistered``
for the same slice.

No market values are hard-coded: the universe, costs and walk-forward windows
flow from the typed :class:`Config`; the DSR threshold defaults to ``0.60`` to
match :mod:`trend_robot.validation.final_report`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from trend_robot.config import Config
from trend_robot.portfolio.sizing import target_weights
from trend_robot.signals.tsmom import tsmom_signal
from trend_robot.validation.final_report import (
    WalkForwardStability,
    _backtest_returns_on,
    _deflated_sharpe_on_returns,
    _walk_forward_stability,
)
from trend_robot.validation.preregistration import DecisionRecord
from trend_robot.validation.splits import walk_forward_splits

__all__ = [
    "HoldoutReport",
    "compute_full_history_weights",
    "evaluate_holdout",
    "format_holdout_report",
]

# DSR threshold above which the deflated Sharpe is read as "clearly positive".
# Mirrors trend_robot.validation.final_report (spec 6.5) -- a judgment threshold
# for the printed recommendation, not a market value.
_DEFAULT_DSR_THRESHOLD: float = 0.60

# Walk-forward stability thresholds, mirrored from final_report so the forward
# stability leg is judged on the same terms as the locked-test read.
_MIN_POSITIVE_WINDOW_FRACTION: float = 0.50
_MAX_SHARPE_DISPERSION: float = 2.0


def compute_full_history_weights(prices: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Compute target weights over the FULL price history (no look-ahead).

    Runs the standard ``signal -> sizing`` pipeline (matching
    :func:`run_research.run_research`) on the whole panel. Each date's signal
    and sizing are causal (they use only prior prices), so weights at any date
    depend only on data up to that date. Slicing these to a hold-out window
    therefore leaks no future information.

    Parameters
    ----------
    prices:
        Full adjusted-close panel.
    cfg:
        Typed configuration (``lookbacks``, ``direction``, sizing knobs).

    Returns
    -------
    pandas.DataFrame
        Target weights indexed exactly like ``prices``.
    """
    returns = prices.pct_change(fill_method=None)
    signals = tsmom_signal(prices, cfg.lookbacks, direction=cfg.direction)
    return target_weights(signals, returns, cfg)


def _forward_index(prices: pd.DataFrame, decision_date: str) -> pd.Index:
    """Index of bars dated strictly after ``decision_date`` (forward hold-out)."""
    cutoff = pd.Timestamp(decision_date)
    return prices.index[prices.index > cutoff]


def _retrospective_index(
    prices: pd.DataFrame, cfg: Config, retrospective_months: int
) -> pd.Index:
    """Index of the last ``retrospective_months`` months of available data.

    Uses a calendar :class:`pandas.DateOffset` from the last available bar so
    the window length is independent of trading-day density.
    """
    if len(prices.index) == 0:
        return prices.index[:0]
    last = prices.index[-1]
    start = last - pd.DateOffset(months=int(retrospective_months))
    return prices.index[prices.index > start]


@dataclass(frozen=True)
class HoldoutReport:
    """Result of a forward / retrospective hold-out read (spec 6.5 / 11).

    Attributes
    ----------
    mode:
        ``'forward'`` (PRISTINE -- genuinely untouched) or ``'retrospective'``
        (NON-PRISTINE -- a diagnostic on already-seen data).
    decision_date:
        The frozen decision date (ISO string) from the pre-registration record.
    holdout_start, holdout_end:
        First/last hold-out dates (``None`` if the hold-out is empty).
    n_holdout_bars:
        Number of bars in the hold-out slice.
    min_bars:
        Minimum bars required to produce a verdict (default one year of bars).
    sufficient:
        Whether ``n_holdout_bars >= min_bars``. When ``False`` the caller should
        print "accrue more data" instead of a verdict.
    metrics:
        Performance-metrics dict on the hold-out slice (minimal/empty when not
        sufficient).
    dsr_threshold:
        Threshold above which a DSR is "clearly positive".
    dsr_preregistered:
        DSR on the slice at ``n_trials=1`` (single pre-registered test).
    dsr_carried:
        DSR on the slice at ``n_trials=record.n_trials_spent`` (conservative,
        carrying the selection search). ``<= dsr_preregistered`` by construction.
    stability:
        Walk-forward stability over the hold-out slice if ``>= 2`` complete
        windows fit, else ``None`` ("not yet assessable").
    retain_preregistered:
        ``(dsr_preregistered > threshold) AND (stability is None or
        stability.is_stable)``.
    retain_carried:
        ``(dsr_carried > threshold) AND (stability is None or
        stability.is_stable)``.
    n_trials_carried:
        The ``n_trials_spent`` carried from the record (for reporting).
    """

    mode: str
    decision_date: str
    holdout_start: pd.Timestamp | None
    holdout_end: pd.Timestamp | None
    n_holdout_bars: int
    min_bars: int
    sufficient: bool
    metrics: dict = field(repr=False)
    dsr_threshold: float = _DEFAULT_DSR_THRESHOLD
    dsr_preregistered: float = float("nan")
    dsr_carried: float = float("nan")
    stability: WalkForwardStability | None = None
    retain_preregistered: bool = False
    retain_carried: bool = False
    n_trials_carried: int = 1


def evaluate_holdout(
    prices: pd.DataFrame,
    cfg: Config,
    record: DecisionRecord,
    *,
    mode: str = "forward",
    retrospective_months: int | None = None,
    min_bars: int | None = None,
    dsr_threshold: float = _DEFAULT_DSR_THRESHOLD,
) -> HoldoutReport:
    """Run the section-6.5 read on the post-decision (or retrospective) slice.

    Weights are computed once over the full history (no look-ahead, mirroring
    :func:`trend_robot.validation.final_report.evaluate_final_validation`), then
    sliced to the hold-out index and replayed through the engine. On a
    sufficient slice the DSR is computed at both ``n_trials=1`` (pre-registered)
    and ``n_trials=record.n_trials_spent`` (carried), and walk-forward stability
    is assessed only if ``>= 2`` complete windows fit the slice.

    Parameters
    ----------
    prices:
        Full adjusted-close panel (the more history, the better the no-look-ahead
        warm-up for the hold-out weights).
    cfg:
        Typed configuration (must match the frozen ``record`` -- the caller is
        responsible for verifying that with
        :func:`trend_robot.validation.preregistration.verify_config_matches`).
    record:
        The frozen :class:`DecisionRecord` (supplies the decision date and the
        carried ``n_trials_spent``).
    mode:
        ``'forward'`` (PRISTINE; bars strictly after the decision date) or
        ``'retrospective'`` (NON-PRISTINE; last ``retrospective_months`` months).
    retrospective_months:
        Window length for retrospective mode (defaults to ``12``). Ignored in
        forward mode.
    min_bars:
        Minimum hold-out bars required for a verdict. Defaults to one year,
        ``cfg.periods_per_year``.
    dsr_threshold:
        DSR above which the result is "clearly positive" (default ``0.60``).

    Returns
    -------
    HoldoutReport
        The hold-out read (see the dataclass).

    Raises
    ------
    ValueError
        If ``mode`` is not ``'forward'`` or ``'retrospective'``.
    """
    if mode not in ("forward", "retrospective"):
        raise ValueError(
            f"'mode' must be 'forward' or 'retrospective', got {mode!r}."
        )

    if min_bars is None:
        min_bars = int(cfg.periods_per_year)
    min_bars = int(min_bars)

    n_trials_carried = int(record.n_trials_spent)

    # --- No look-ahead: weights on the FULL history, then slice. -----------
    full_weights = compute_full_history_weights(prices, cfg)

    # --- Determine the hold-out index by mode. -----------------------------
    if mode == "forward":
        holdout_index = _forward_index(prices, record.decision_date)
    else:
        months = 12 if retrospective_months is None else int(retrospective_months)
        holdout_index = _retrospective_index(prices, cfg, months)

    n_holdout = int(len(holdout_index))
    holdout_start = holdout_index[0] if n_holdout else None
    holdout_end = holdout_index[-1] if n_holdout else None
    sufficient = n_holdout >= min_bars

    # --- Insufficient data: short-circuit with a minimal report. -----------
    if not sufficient:
        return HoldoutReport(
            mode=mode,
            decision_date=record.decision_date,
            holdout_start=holdout_start,
            holdout_end=holdout_end,
            n_holdout_bars=n_holdout,
            min_bars=min_bars,
            sufficient=False,
            metrics={},
            dsr_threshold=dsr_threshold,
            dsr_preregistered=float("nan"),
            dsr_carried=float("nan"),
            stability=None,
            retain_preregistered=False,
            retain_carried=False,
            n_trials_carried=n_trials_carried,
        )

    # --- Sufficient: slice + backtest + metrics on the hold-out only. ------
    holdout_returns, holdout_metrics = _backtest_returns_on(
        prices, full_weights, cfg, holdout_index
    )

    # --- DSR at two trial counts on the SAME slice. ------------------------
    dsr_pre = _deflated_sharpe_on_returns(holdout_returns, n_trials=1)
    dsr_carried = _deflated_sharpe_on_returns(
        holdout_returns, n_trials=n_trials_carried
    )

    # --- Walk-forward stability over the hold-out slice (>= 2 windows). ----
    windows = walk_forward_splits(holdout_index, cfg)
    if len(windows) >= 2:
        stability: WalkForwardStability | None = _walk_forward_stability(
            prices,
            full_weights,
            cfg,
            windows,
            min_positive_fraction=_MIN_POSITIVE_WINDOW_FRACTION,
            max_dispersion=_MAX_SHARPE_DISPERSION,
        )
    else:
        # Not enough hold-out history to roll >= 2 complete windows: the
        # walk-forward stability leg is "not yet assessable".
        stability = None

    # --- Section-6.5 verdict at each trial count. --------------------------
    # With stability=None the walk-forward leg is not yet assessable; we do not
    # block on it (it cannot be evaluated), so the verdict rests on the DSR.
    stable_leg = stability is None or stability.is_stable
    retain_pre = bool(np.isfinite(dsr_pre) and dsr_pre > dsr_threshold and stable_leg)
    retain_carried = bool(
        np.isfinite(dsr_carried) and dsr_carried > dsr_threshold and stable_leg
    )

    return HoldoutReport(
        mode=mode,
        decision_date=record.decision_date,
        holdout_start=holdout_start,
        holdout_end=holdout_end,
        n_holdout_bars=n_holdout,
        min_bars=min_bars,
        sufficient=True,
        metrics=holdout_metrics,
        dsr_threshold=dsr_threshold,
        dsr_preregistered=dsr_pre,
        dsr_carried=dsr_carried,
        stability=stability,
        retain_preregistered=retain_pre,
        retain_carried=retain_carried,
        n_trials_carried=n_trials_carried,
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


def format_holdout_report(report: HoldoutReport, record: DecisionRecord) -> str:
    """Render a hold-out read as a human-readable block (spec 6.5 / 11).

    The banner makes the pristine-vs-retrospective distinction *unmissable*; the
    body shows the frozen decision hash/date/trials, the data-sufficiency status
    (e.g. ``"126/252 bars -- need 126 more"``), the DSR at both ``n_trials``
    settings, the walk-forward leg (or that it is "not yet assessable"), and the
    mandatory section-11 note that this is a human judgment and the system never
    auto-deploys.

    Parameters
    ----------
    report:
        The evaluated :class:`HoldoutReport`.
    record:
        The frozen :class:`DecisionRecord` (for the banner header).

    Returns
    -------
    str
        A multi-line, ready-to-print report.
    """
    bar = "=" * 70
    lines: list[str] = [bar]

    # --- Unmissable pristine-vs-retrospective banner. ----------------------
    if report.mode == "forward":
        lines.append("  *** PRISTINE FORWARD HOLD-OUT  (genuinely untouched OOS) ***")
        lines.append(
            "  These bars arrived STRICTLY AFTER the frozen decision date, so"
        )
        lines.append(
            "  they were NOT visible during variant selection. This is the only"
        )
        lines.append("  truly out-of-sample evidence for the frozen strategy.")
    else:
        lines.append(
            "  !!! NON-PRISTINE RETROSPECTIVE CHECK  (NOT untouched OOS) !!!"
        )
        lines.append(
            "  These bars were VISIBLE during variant selection, so this is a"
        )
        lines.append(
            "  DIAGNOSTIC ONLY -- it produces a number today but is NOT a clean"
        )
        lines.append(
            "  out-of-sample verdict. For the pristine read, use mode='forward'."
        )
    lines.append("-" * 70)

    # --- Frozen decision provenance. ---------------------------------------
    lines.append("  Pre-registered decision")
    lines.append(f"        Decision date     : {record.decision_date}")
    lines.append(f"        Config hash       : {record.config_hash}")
    lines.append(f"        Trials spent      : {record.n_trials_spent}")
    lines.append(f"        Frozen at (UTC)   : {record.created_at}")
    if record.notes:
        lines.append(f"        Notes             : {record.notes}")
    lines.append("")

    # --- Data sufficiency. -------------------------------------------------
    lines.append("  Hold-out window")
    lines.append(
        f"        Slice             : {report.holdout_start} -> "
        f"{report.holdout_end}"
    )
    deficit = max(0, report.min_bars - report.n_holdout_bars)
    status = "SUFFICIENT" if report.sufficient else "INSUFFICIENT"
    sufficiency_line = (
        f"        Bars              : {report.n_holdout_bars}/{report.min_bars} "
        f"({status})"
    )
    if not report.sufficient:
        sufficiency_line += f" -- need {deficit} more"
    lines.append(sufficiency_line)
    lines.append("")

    if not report.sufficient:
        lines.append(
            "  Not enough hold-out data yet to render a section-6.5 verdict."
        )
        lines.append(
            "  ACCRUE MORE DATA: let the paper-trading / forward track run until"
        )
        lines.append(
            f"  at least {report.min_bars} post-decision bars have accumulated, "
            "then re-run."
        )
        lines.append("")
        lines.append(_section11_note())
        lines.append(bar)
        return "\n".join(lines)

    # --- (a) Deflated Sharpe at both trial settings. -----------------------
    m = report.metrics
    lines.append("  (a) Deflated Sharpe Ratio on the hold-out slice (spec 6.4)")
    lines.append(f"        Hold-out Sharpe (ann.)   : {_fmt(m.get('sharpe'))}")
    lines.append(f"        Hold-out CAGR            : {_fmt(m.get('cagr'))}")
    lines.append(f"        Hold-out max drawdown    : {_fmt(m.get('max_drawdown'))}")
    lines.append(
        f"        DSR (pre-registered)     : {_fmt(report.dsr_preregistered)} "
        f"(n_trials=1, threshold {report.dsr_threshold:.2f})"
    )
    lines.append(
        f"        DSR (carried)            : {_fmt(report.dsr_carried)} "
        f"(n_trials={report.n_trials_carried}, threshold "
        f"{report.dsr_threshold:.2f})"
    )
    lines.append(
        "        Note: 'carried' applies the selection search's multiple-testing"
    )
    lines.append(
        "        hurdle, so it is <= the pre-registered DSR (conservative)."
    )
    lines.append("")

    # --- (b) Walk-forward stability over the hold-out. ---------------------
    s = report.stability
    lines.append("  (b) Walk-forward stability over the hold-out (spec 6.2)")
    if s is None:
        lines.append(
            "        Fewer than 2 complete walk-forward windows fit the hold-out;"
        )
        lines.append(
            "        walk-forward stability is NOT YET ASSESSABLE on this slice."
        )
    else:
        lines.append(f"        Windows evaluated        : {s.n_windows}")
        sharpe_str = ", ".join(_fmt(x) for x in s.window_sharpes)
        lines.append(f"        Per-window Sharpe        : [{sharpe_str}]")
        lines.append(
            f"        Mean / std Sharpe        : {_fmt(s.mean_sharpe)} "
            f"/ {_fmt(s.std_sharpe)}"
        )
        lines.append(
            f"        Positive-window fraction : {_fmt(s.positive_fraction)}"
        )
        lines.append(f"        Sharpe dispersion (cv)   : {_fmt(s.dispersion)}")
        verdict_b = "PASS" if s.is_stable else "FAIL"
        lines.append(f"        => Walk-forward stable    : {verdict_b}")
    lines.append("")

    # --- Verdicts at both trial settings. ----------------------------------
    lines.append("-" * 70)
    rec_pre = "RETAIN" if report.retain_preregistered else "REJECT (DO NOT DEPLOY)"
    rec_carried = "RETAIN" if report.retain_carried else "REJECT (DO NOT DEPLOY)"
    lines.append(f"  SECTION 6.5 (pre-registered, n_trials=1) : {rec_pre}")
    lines.append(
        f"  SECTION 6.5 (carried, n_trials={report.n_trials_carried})"
        f"{' ' * max(1, 12 - len(str(report.n_trials_carried)))}: {rec_carried}"
    )
    lines.append(
        "  Criterion: retain ONLY if the Deflated Sharpe is clearly positive AND"
    )
    if s is None:
        lines.append(
            "             the walk-forward track is stable. Here the walk-forward"
        )
        lines.append(
            "             leg is NOT YET ASSESSABLE (too short), so the verdict"
        )
        lines.append("             rests on the DSR alone -- treat as provisional.")
    else:
        lines.append("             the walk-forward performance is stable.")
    lines.append("")
    lines.append(_section11_note())
    lines.append(bar)
    return "\n".join(lines)


def _section11_note() -> str:
    """The mandatory section-11 human-judgment / no-auto-deploy note."""
    return (
        "  NOTE (spec 11): This RETAIN/REJECT read is an automated summary, NOT a\n"
        "  decision. The final call is a HUMAN judgment. This system never\n"
        "  auto-deploys and never re-tunes on the hold-out; a pristine forward\n"
        "  read is consumed once, on data accrued after the frozen decision date."
    )
