"""Typed configuration for the TSMOM research robot.

This module is the single typed entry point for all strategy/backtest/
validation parameters. Values are loaded from ``config.yaml`` into the
:class:`Config` dataclass and *validated* eagerly so that any malformed
configuration fails fast with a clear, actionable error message.

NO market values are hard-coded in the codebase: everything flows from the
YAML file through this typed :class:`Config`.

Public API
----------
- :class:`Config`            -- typed container, one field per parameter.
- :func:`load_config`        -- parse + validate YAML into a :class:`Config`.
- :func:`set_global_seed`    -- seed ``numpy`` and ``random`` for determinism.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import yaml

__all__ = ["Config", "ConfigError", "load_config", "set_global_seed"]

# Allowed categorical values (kept here so validation has a single source).
_VALID_DIRECTIONS: frozenset[str] = frozenset({"long_short", "long_only"})
_VALID_REBALANCE: frozenset[str] = frozenset({"daily", "weekly", "monthly"})


class ConfigError(ValueError):
    """Raised when a configuration file is missing fields or is invalid."""


@dataclass(frozen=True)
class Config:
    """Immutable, typed container for every robot parameter.

    One field maps to exactly one key in ``config.yaml``. The dataclass is
    frozen so that, once validated, the configuration cannot be mutated in
    flight (which would break reproducibility guarantees).

    Attributes
    ----------
    initial_capital:
        Starting capital of the backtest (currency units).
    universe:
        Tickers traded. Must be a non-empty list of unique strings.
    direction:
        ``"long_short"`` or ``"long_only"`` (negative signals truncated).
    rebalance:
        Rebalancing cadence: ``"daily"``, ``"weekly"`` or ``"monthly"``.
    lookbacks:
        TSMOM lookback horizons in trading days; averaged across horizons.
    vol_window:
        Window (days) for ex-ante volatility estimation (EWMA com).
    asset_vol_target:
        Per-asset annualized volatility target (e.g. ``0.10`` = 10%).
    portfolio_vol_target:
        Portfolio-level annualized volatility target.
    max_gross_leverage:
        Cap on ``sum(|w_i|)``; weights renormalized if exceeded.
    kelly_fraction:
        Fractional-Kelly scaler applied to the vol-targeted weights.
    cost_bps_per_side:
        Transaction cost in basis points charged per side on traded notional.
    cost_stress_levels:
        Higher cost levels (bps/side) replayed for sensitivity analysis.
    periods_per_year:
        Trading periods per year used for annualization (e.g. 252).
    train_test_ratio:
        Fraction of history used for development; the remainder is the
        locked out-of-sample test set. Must lie in the open interval (0, 1).
    wf_train_years:
        Walk-forward training window length in years.
    wf_test_years:
        Walk-forward testing window length in years.
    wf_step_years:
        Walk-forward step (roll) length in years.
    cv_embargo:
        Embargo fraction (of total samples) applied after each CV test fold.
        Must lie in the half-open interval [0, 1).
    seed:
        Global integer seed for ``numpy`` and ``random``.
    """

    initial_capital: float
    universe: list[str]
    direction: str
    rebalance: str
    lookbacks: list[int]
    vol_window: int
    asset_vol_target: float
    portfolio_vol_target: float
    max_gross_leverage: float
    kelly_fraction: float
    cost_bps_per_side: float
    cost_stress_levels: list[float]
    periods_per_year: int
    train_test_ratio: float
    wf_train_years: int
    wf_test_years: int
    wf_step_years: int
    cv_embargo: float
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate every field; raise :class:`ConfigError` on any violation."""
        _validate(self)


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------
def _as_float(value: Any, name: str) -> float:
    """Coerce ``value`` to ``float`` or raise a clear :class:`ConfigError`."""
    try:
        return float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ConfigError(f"'{name}' must be a number, got {value!r}.") from exc


def _as_int(value: Any, name: str) -> int:
    """Coerce ``value`` to ``int`` (rejecting non-integral floats)."""
    if isinstance(value, bool):  # bool is a subclass of int; reject explicitly
        raise ConfigError(f"'{name}' must be an integer, got bool {value!r}.")
    if isinstance(value, float) and not value.is_integer():
        raise ConfigError(f"'{name}' must be an integer, got {value!r}.")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"'{name}' must be an integer, got {value!r}.") from exc


def _as_str_list(value: Any, name: str) -> list[str]:
    """Coerce ``value`` to a list of strings."""
    if not isinstance(value, (list, tuple)):
        raise ConfigError(f"'{name}' must be a list, got {type(value).__name__}.")
    return [str(v) for v in value]


def _as_int_list(value: Any, name: str) -> list[int]:
    """Coerce ``value`` to a list of ints."""
    if not isinstance(value, (list, tuple)):
        raise ConfigError(f"'{name}' must be a list, got {type(value).__name__}.")
    return [_as_int(v, f"{name}[{i}]") for i, v in enumerate(value)]


def _as_float_list(value: Any, name: str) -> list[float]:
    """Coerce ``value`` to a list of floats."""
    if not isinstance(value, (list, tuple)):
        raise ConfigError(f"'{name}' must be a list, got {type(value).__name__}.")
    return [_as_float(v, f"{name}[{i}]") for i, v in enumerate(value)]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _validate(cfg: Config) -> None:
    """Validate a :class:`Config`, raising :class:`ConfigError` on failure.

    Checks (non-exhaustive list mirrored from the spec):
      * ratios in the open interval (0, 1);
      * positive windows / lookbacks / capital / leverage;
      * non-empty, unique universe;
      * ``direction`` and ``rebalance`` in their allowed sets;
      * embargo fraction in [0, 1);
      * non-negative costs and a sane Kelly fraction.
    """
    # --- Capital -----------------------------------------------------------
    if cfg.initial_capital <= 0:
        raise ConfigError(
            f"'initial_capital' must be positive, got {cfg.initial_capital}."
        )

    # --- Universe ----------------------------------------------------------
    if not cfg.universe:
        raise ConfigError("'universe' must be a non-empty list of tickers.")
    if any(not isinstance(t, str) or not t.strip() for t in cfg.universe):
        raise ConfigError("'universe' tickers must be non-empty strings.")
    if len(set(cfg.universe)) != len(cfg.universe):
        raise ConfigError(f"'universe' contains duplicate tickers: {cfg.universe}.")

    # --- Categoricals ------------------------------------------------------
    if cfg.direction not in _VALID_DIRECTIONS:
        raise ConfigError(
            f"'direction' must be one of {sorted(_VALID_DIRECTIONS)}, "
            f"got {cfg.direction!r}."
        )
    if cfg.rebalance not in _VALID_REBALANCE:
        raise ConfigError(
            f"'rebalance' must be one of {sorted(_VALID_REBALANCE)}, "
            f"got {cfg.rebalance!r}."
        )

    # --- Lookbacks ---------------------------------------------------------
    if not cfg.lookbacks:
        raise ConfigError("'lookbacks' must be a non-empty list.")
    if any(lb <= 0 for lb in cfg.lookbacks):
        raise ConfigError(f"'lookbacks' must all be positive, got {cfg.lookbacks}.")

    # --- Windows -----------------------------------------------------------
    if cfg.vol_window <= 0:
        raise ConfigError(f"'vol_window' must be positive, got {cfg.vol_window}.")
    if cfg.periods_per_year <= 0:
        raise ConfigError(
            f"'periods_per_year' must be positive, got {cfg.periods_per_year}."
        )

    # --- Vol targets -------------------------------------------------------
    if cfg.asset_vol_target <= 0:
        raise ConfigError(
            f"'asset_vol_target' must be positive, got {cfg.asset_vol_target}."
        )
    if cfg.portfolio_vol_target <= 0:
        raise ConfigError(
            f"'portfolio_vol_target' must be positive, "
            f"got {cfg.portfolio_vol_target}."
        )

    # --- Leverage / Kelly --------------------------------------------------
    if cfg.max_gross_leverage <= 0:
        raise ConfigError(
            f"'max_gross_leverage' must be positive, got {cfg.max_gross_leverage}."
        )
    if cfg.kelly_fraction <= 0:
        raise ConfigError(
            f"'kelly_fraction' must be positive, got {cfg.kelly_fraction}."
        )

    # --- Costs -------------------------------------------------------------
    if cfg.cost_bps_per_side < 0:
        raise ConfigError(
            f"'cost_bps_per_side' must be non-negative, got {cfg.cost_bps_per_side}."
        )
    if any(c < 0 for c in cfg.cost_stress_levels):
        raise ConfigError(
            f"'cost_stress_levels' must all be non-negative, "
            f"got {cfg.cost_stress_levels}."
        )

    # --- Train/test ratio --------------------------------------------------
    if not (0.0 < cfg.train_test_ratio < 1.0):
        raise ConfigError(
            f"'train_test_ratio' must be in the open interval (0, 1), "
            f"got {cfg.train_test_ratio}."
        )

    # --- Walk-forward ------------------------------------------------------
    if cfg.wf_train_years <= 0:
        raise ConfigError(
            f"'wf_train_years' must be positive, got {cfg.wf_train_years}."
        )
    if cfg.wf_test_years <= 0:
        raise ConfigError(
            f"'wf_test_years' must be positive, got {cfg.wf_test_years}."
        )
    if cfg.wf_step_years <= 0:
        raise ConfigError(
            f"'wf_step_years' must be positive, got {cfg.wf_step_years}."
        )

    # --- Embargo -----------------------------------------------------------
    if not (0.0 <= cfg.cv_embargo < 1.0):
        raise ConfigError(
            f"'cv_embargo' must be in the half-open interval [0, 1), "
            f"got {cfg.cv_embargo}."
        )

    # --- Seed --------------------------------------------------------------
    if cfg.seed < 0:
        raise ConfigError(f"'seed' must be a non-negative integer, got {cfg.seed}.")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
# Required YAML keys -> coercion function. ``seed`` has a default so it is
# treated as optional below.
_COERCERS: dict[str, Any] = {
    "initial_capital": lambda v: _as_float(v, "initial_capital"),
    "universe": lambda v: _as_str_list(v, "universe"),
    "direction": lambda v: str(v),
    "rebalance": lambda v: str(v),
    "lookbacks": lambda v: _as_int_list(v, "lookbacks"),
    "vol_window": lambda v: _as_int(v, "vol_window"),
    "asset_vol_target": lambda v: _as_float(v, "asset_vol_target"),
    "portfolio_vol_target": lambda v: _as_float(v, "portfolio_vol_target"),
    "max_gross_leverage": lambda v: _as_float(v, "max_gross_leverage"),
    "kelly_fraction": lambda v: _as_float(v, "kelly_fraction"),
    "cost_bps_per_side": lambda v: _as_float(v, "cost_bps_per_side"),
    "cost_stress_levels": lambda v: _as_float_list(v, "cost_stress_levels"),
    "periods_per_year": lambda v: _as_int(v, "periods_per_year"),
    "train_test_ratio": lambda v: _as_float(v, "train_test_ratio"),
    "wf_train_years": lambda v: _as_int(v, "wf_train_years"),
    "wf_test_years": lambda v: _as_int(v, "wf_test_years"),
    "wf_step_years": lambda v: _as_int(v, "wf_step_years"),
    "cv_embargo": lambda v: _as_float(v, "cv_embargo"),
    "seed": lambda v: _as_int(v, "seed"),
}


def load_config(path: str | Path) -> Config:
    """Load and validate a :class:`Config` from a YAML file.

    Parameters
    ----------
    path:
        Filesystem path to ``config.yaml``.

    Returns
    -------
    Config
        A validated, immutable configuration object.

    Raises
    ------
    ConfigError
        If the file is missing, not a YAML mapping, has unknown/missing
        required keys, or any value fails validation.
    """
    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise ConfigError(f"Config file not found: {cfg_path}")

    try:
        with cfg_path.open("r", encoding="utf-8") as fh:
            raw: Any = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse YAML at {cfg_path}: {exc}") from exc

    if raw is None:
        raise ConfigError(f"Config file is empty: {cfg_path}")
    if not isinstance(raw, dict):
        raise ConfigError(
            f"Config root must be a mapping, got {type(raw).__name__} at {cfg_path}."
        )

    # Determine which keys are required (every dataclass field without a default).
    field_names = {f.name for f in fields(Config)}
    # `seed` carries a real default; everything else is mandatory.
    required = field_names - {"seed"}

    unknown = set(raw) - field_names
    if unknown:
        raise ConfigError(
            f"Unknown config key(s): {sorted(unknown)}. "
            f"Allowed keys: {sorted(field_names)}."
        )

    missing = required - set(raw)
    if missing:
        raise ConfigError(f"Missing required config key(s): {sorted(missing)}.")

    kwargs: dict[str, Any] = {}
    for key, value in raw.items():
        kwargs[key] = _COERCERS[key](value)

    # Construction triggers __post_init__ -> _validate.
    return Config(**kwargs)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_global_seed(seed: int) -> None:
    """Seed the global RNGs for deterministic, reproducible runs.

    Seeds both the standard library ``random`` module and ``numpy``'s legacy
    global random state. Call this once at the start of any run/backtest.

    Parameters
    ----------
    seed:
        Non-negative integer seed (typically ``cfg.seed``).

    Raises
    ------
    ConfigError
        If ``seed`` is not a non-negative integer.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ConfigError(f"'seed' must be a non-negative integer, got {seed!r}.")
    random.seed(seed)
    np.random.seed(seed)
