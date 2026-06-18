"""Shared pytest fixtures/helpers for the TSMOM robot test suite.

All fixtures use deterministic, offline data only (``SyntheticProvider`` or
hand-built frames). Nothing in the test suite ever touches a live, rate-limited
download (Yahoo returns HTTP 429 in this environment).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd
import pytest

from trend_robot.config import Config, load_config, set_global_seed
from trend_robot.data.synthetic_provider import SyntheticProvider

# Project root = the directory that contains ``config.yaml`` (one level above
# this ``tests`` package). Resolved from ``__file__`` so the suite is runnable
# regardless of the invoking CWD.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"


@pytest.fixture(scope="session")
def cfg() -> Config:
    """The validated project :class:`Config` loaded from ``config.yaml``.

    No market values are hard-coded in tests: everything flows from the same
    YAML the production pipeline uses.
    """
    config = load_config(_CONFIG_PATH)
    set_global_seed(config.seed)
    return config


def make_config(base: Config, **overrides: object) -> Config:
    """Return a copy of ``base`` with ``overrides`` applied (re-validated).

    A thin wrapper over :func:`dataclasses.replace` so tests can vary a single
    parameter (e.g. ``cost_bps_per_side``) without re-typing the whole config or
    hard-coding values.
    """
    return dataclasses.replace(base, **overrides)


@pytest.fixture(scope="session")
def synthetic_prices(cfg: Config) -> pd.DataFrame:
    """Deterministic synthetic adjusted-close panel over the full universe.

    Spans enough history that every configured lookback (including 252) becomes
    defined, so signal/sizing/engine all have real positions to exercise.
    """
    provider = SyntheticProvider(seed=cfg.seed)
    return provider.get_prices(cfg.universe, "2015-01-01", "2021-12-31")


@pytest.fixture(scope="session")
def synthetic_returns(synthetic_prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns of :func:`synthetic_prices` (NaN gaps preserved)."""
    return synthetic_prices.pct_change(fill_method=None)
