"""Backtest layer: execution engine and transaction-cost model.

Exposes the marked-to-market, look-ahead-free :func:`run_backtest` engine and
its :class:`BacktestResult` container, plus the pure transaction-cost functions
(:func:`linear_cost`, :func:`impact_cost`) used to charge rebalancing turnover.
"""

from __future__ import annotations

from trend_robot.backtest.costs import bps_to_fraction, impact_cost, linear_cost
from trend_robot.backtest.engine import BacktestResult, run_backtest

__all__ = [
    "BacktestResult",
    "run_backtest",
    "linear_cost",
    "impact_cost",
    "bps_to_fraction",
]
