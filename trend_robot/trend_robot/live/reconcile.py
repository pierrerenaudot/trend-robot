"""Book reconciliation: is the broker's book where the strategy expects it?

Motivated by a real incident (July 2026): the paper account was reset mid-month
AFTER the monthly submission had been recorded. The idempotence guard then
correctly refused to re-trade ("period already submitted"), leaving a FLAT book
for a month -- silently, with green scheduled runs. Nothing compared the
broker's actual positions to what the strategy believed it held.

This module provides that comparison. :func:`reconcile_book` is a PURE function
that measures the L1 gap between the broker's current weights and the target
weights, and flags the two anomaly shapes we care about:

* ``book_flat_but_target_invested`` -- the exact July failure: targets say
  "hold risk" but the broker book is empty;
* ``materially_off_target`` -- the total absolute weight gap exceeds a
  tolerance chosen to absorb normal intra-period drift (a monthly book drifts
  a few percent, not tens of percent).

The caller (``run_live.py``) decides WHEN a gap is an anomaly: it is expected
right before a rebalance submission, but NOT on a run that skips because the
period was already traded -- there the book should sit near target, and a large
gap means the earlier submission's fills are gone (reset, rejection, manual
liquidation) and a human must be alerted.

Pure: no I/O, no broker calls, no hard-coded market values.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from trend_robot.live.broker import Position

__all__ = ["ReconcileReport", "reconcile_book", "format_reconcile_report"]

# Default L1 tolerance: total |target - current| weight gap tolerated before
# the book is called materially off-target. Normal drift between monthly
# rebalances is a few percent of L1; a vanished book shows up as the full
# target gross (e.g. ~0.7). 0.25 sits comfortably between the two regimes.
_DEFAULT_TOLERANCE: float = 0.25

# Gross-weight floor under which a book is considered flat (dust-safe zero).
_FLAT_EPS: float = 1e-6


@dataclass(frozen=True)
class ReconcileReport:
    """Outcome of comparing the broker book to the target book.

    Attributes
    ----------
    current_weights:
        Realized weight per symbol (``qty * last_price / equity``) for the
        union of held and targeted symbols.
    target_weights:
        Target weight per symbol over the same union.
    deviations:
        ``target - current`` per symbol.
    l1_deviation:
        ``sum(|target - current|)`` -- the headline gap measure.
    max_deviation:
        Largest single-symbol absolute gap.
    target_gross:
        ``sum(|target|)``.
    current_gross:
        ``sum(|current|)``.
    tolerance:
        The L1 tolerance the report was evaluated against.
    book_flat_but_target_invested:
        ``True`` when the broker book is (near) empty while the target carries
        non-trivial gross exposure -- the July-incident signature.
    materially_off_target:
        ``True`` when ``l1_deviation > tolerance``.
    anomaly:
        ``True`` when either flag above is set. Whether an anomaly is *fatal*
        is the caller's decision (it depends on whether a rebalance was
        expected to have already happened).
    symbols:
        The union of symbols considered, sorted (presentation convenience).
    """

    current_weights: dict[str, float]
    target_weights: dict[str, float]
    deviations: dict[str, float]
    l1_deviation: float
    max_deviation: float
    target_gross: float
    current_gross: float
    tolerance: float
    book_flat_but_target_invested: bool
    materially_off_target: bool
    anomaly: bool
    symbols: list[str] = field(default_factory=list)


def reconcile_book(
    target_w: pd.Series,
    positions: dict[str, Position],
    last_px: pd.Series,
    equity: float,
    *,
    tolerance: float = _DEFAULT_TOLERANCE,
) -> ReconcileReport:
    """Measure how far the broker book sits from the target book.

    Parameters
    ----------
    target_w:
        Target weights indexed by symbol (today's frozen-strategy book).
    positions:
        Current broker positions keyed by symbol.
    last_px:
        Last known price per symbol (used to value held quantities; symbols
        with no usable price fall back to the position's reported
        ``market_value``).
    equity:
        Account equity used to express positions as weights.
    tolerance:
        L1 gap above which the book is flagged materially off-target. Must be
        positive.

    Returns
    -------
    ReconcileReport
        The structured comparison (see the dataclass). Never raises on empty
        inputs: an empty target with an empty book reconciles trivially.

    Raises
    ------
    ValueError
        If ``tolerance`` is not positive or ``equity`` is not finite/positive
        while positions exist (a zero-equity account with holdings cannot be
        expressed in weights).
    """
    if tolerance <= 0.0:
        raise ValueError(f"'tolerance' must be positive, got {tolerance}.")
    equity = float(equity)
    if positions and (not pd.notna(equity) or equity <= 0.0):
        raise ValueError(
            f"'equity' must be positive to weigh {len(positions)} open "
            f"position(s), got {equity}."
        )

    symbols = sorted(
        {str(s) for s in target_w.index} | {str(s) for s in positions}
    )

    current: dict[str, float] = {}
    target: dict[str, float] = {}
    deviations: dict[str, float] = {}
    for sym in symbols:
        tw = float(target_w.get(sym, 0.0))
        if pd.isna(tw):
            tw = 0.0

        pos = positions.get(sym)
        if pos is None or equity <= 0.0:
            cw = 0.0
        else:
            price = float(last_px.get(sym, float("nan")))
            if pd.notna(price) and price > 0.0:
                cw = (float(pos.qty) * price) / equity
            else:
                # No usable price: fall back to the broker-reported value.
                cw = float(pos.market_value) / equity

        current[sym] = cw
        target[sym] = tw
        deviations[sym] = tw - cw

    l1 = float(sum(abs(d) for d in deviations.values()))
    max_dev = float(max((abs(d) for d in deviations.values()), default=0.0))
    target_gross = float(sum(abs(w) for w in target.values()))
    current_gross = float(sum(abs(w) for w in current.values()))

    flat_vs_invested = (
        current_gross <= _FLAT_EPS and target_gross > tolerance
    )
    off_target = l1 > tolerance

    return ReconcileReport(
        current_weights=current,
        target_weights=target,
        deviations=deviations,
        l1_deviation=l1,
        max_deviation=max_dev,
        target_gross=target_gross,
        current_gross=current_gross,
        tolerance=float(tolerance),
        book_flat_but_target_invested=flat_vs_invested,
        materially_off_target=off_target,
        anomaly=bool(flat_vs_invested or off_target),
        symbols=symbols,
    )


def format_reconcile_report(report: ReconcileReport) -> str:
    """Render a reconciliation report as a compact, readable text block."""
    lines: list[str] = []
    status = "ANOMALY" if report.anomaly else "OK"
    lines.append(
        f"RECONCILIATION [{status}]  L1 gap {report.l1_deviation:.4f} "
        f"(tolerance {report.tolerance:.2f})  "
        f"gross current {report.current_gross:.4f} vs target "
        f"{report.target_gross:.4f}"
    )
    if report.book_flat_but_target_invested:
        lines.append(
            "  !! book is FLAT while the target is invested -- the recorded "
            "submission's positions are gone (account reset / rejected or "
            "cancelled fills / manual liquidation)."
        )
    if report.materially_off_target and not report.book_flat_but_target_invested:
        lines.append(
            "  !! book is materially off-target beyond normal drift."
        )
    header = f"  {'SYMBOL':<8}{'TARGET_W':>10}{'CURRENT_W':>11}{'GAP':>10}"
    lines.append(header)
    for sym in report.symbols:
        lines.append(
            f"  {sym:<8}{report.target_weights[sym]:>10.4f}"
            f"{report.current_weights[sym]:>11.4f}"
            f"{report.deviations[sym]:>10.4f}"
        )
    return "\n".join(lines)
