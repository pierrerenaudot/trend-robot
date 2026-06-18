"""Metrics layer: performance statistics and the Deflated Sharpe Ratio."""

from __future__ import annotations

from trend_robot.metrics.deflated_sharpe import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    observed_sharpe,
)
from trend_robot.metrics.performance import performance_metrics

__all__ = [
    "performance_metrics",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "observed_sharpe",
]
