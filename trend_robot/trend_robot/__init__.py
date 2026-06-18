"""TSMOM research robot — time-series momentum research & validation package.

Scope: research + validation only (data layer, TSMOM signal, vol-targeted
portfolio, realistic backtest, honest metrics incl. Deflated Sharpe, and a
rigorous validation harness). Out of scope: live execution / OMS / broker.

The single typed entry point for configuration is :mod:`trend_robot.config`.
"""

from __future__ import annotations

from .config import Config, ConfigError, load_config, set_global_seed

__all__ = ["Config", "ConfigError", "load_config", "set_global_seed"]

__version__ = "0.1.0"
