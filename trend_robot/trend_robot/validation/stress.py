"""Transaction-cost sensitivity analysis (spec section 5).

A strategy that is only profitable at unrealistically low transaction costs is
fragile. The spec therefore makes a cost **sensitivity test** mandatory: replay
*exactly the same* backtest at the base cost and at every elevated level in
``cfg.cost_stress_levels`` (e.g. ``[5, 10]`` bps per side) and compare how the
key economics degrade as costs rise.

Design notes
------------
* The strategy logic (signal -> weights) is computed **once** and reused across
  all cost levels. Only the backtest is replayed, so the *only* thing that
  varies between rows of the comparison table is the cost charged on turnover.
  This isolates the cost sensitivity cleanly and avoids duplicating any strategy
  logic.
* Cost is varied by constructing a **copy** of the frozen :class:`Config` with a
  different ``cost_bps_per_side`` via :func:`dataclasses.replace` (the original
  config is never mutated -- ``Config`` is frozen for reproducibility).
* No market values are hard-coded: the base cost and the stress levels both come
  from the typed :class:`Config`.

This module is pure with respect to its inputs: it never mutates ``prices``,
``target_weights`` or ``cfg``, and performs no I/O.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import pandas as pd

from trend_robot.backtest.engine import run_backtest
from trend_robot.config import Config
from trend_robot.metrics.performance import performance_metrics

__all__ = ["CostStressRow", "cost_stress_test", "cost_stress_table"]


@dataclass(frozen=True)
class CostStressRow:
    """Key economics of one cost-level replay (spec section 5).

    Attributes
    ----------
    cost_bps_per_side:
        Transaction cost (basis points per side) used for this replay.
    is_base:
        ``True`` for the strategy's configured base cost, ``False`` for an
        elevated stress level.
    cagr:
        Compound annual growth rate of the net-of-cost equity curve.
    sharpe:
        Annualized Sharpe ratio of the net-of-cost returns.
    max_drawdown:
        Worst peak-to-trough decline (a non-positive fraction).
    total_cost:
        Total transaction cost charged across the whole backtest (currency
        units).
    final_equity:
        Marked-to-market equity at the end of the backtest (currency units).
    """

    cost_bps_per_side: float
    is_base: bool
    cagr: float
    sharpe: float
    max_drawdown: float
    total_cost: float
    final_equity: float


def _replay_at_cost(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    cfg: Config,
    cost_bps_per_side: float,
    *,
    is_base: bool,
) -> CostStressRow:
    """Replay the backtest at one cost level and extract key metrics.

    Parameters
    ----------
    prices:
        Adjusted-close price panel (the backtest data contract).
    target_weights:
        Pre-computed target weights (strategy logic is *not* re-run).
    cfg:
        Base typed configuration.
    cost_bps_per_side:
        Cost (bps per side) to charge for this replay.
    is_base:
        Whether this level is the strategy's configured base cost.

    Returns
    -------
    CostStressRow
        The key economics observed at this cost level.
    """
    # Vary ONLY the cost; Config is frozen, so build a copy (never mutate).
    cfg_at_cost = dataclasses.replace(cfg, cost_bps_per_side=float(cost_bps_per_side))
    result = run_backtest(prices, target_weights, cfg_at_cost)
    metrics = performance_metrics(result, cfg_at_cost)

    final_equity = (
        float(result.equity.iloc[-1]) if result.equity.size else float("nan")
    )
    return CostStressRow(
        cost_bps_per_side=float(cost_bps_per_side),
        is_base=is_base,
        cagr=float(metrics["cagr"]),
        sharpe=float(metrics["sharpe"]),
        max_drawdown=float(metrics["max_drawdown"]),
        total_cost=float(metrics["total_cost"]),
        final_equity=final_equity,
    )


def cost_stress_test(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    cfg: Config,
) -> list[CostStressRow]:
    """Replay the same backtest at the base and stressed cost levels (spec 5).

    The strategy's target weights are taken as a given input and reused for every
    replay, so the only thing that changes between rows is the transaction cost
    charged on turnover. The base level is ``cfg.cost_bps_per_side``; the
    stressed levels are ``cfg.cost_stress_levels``. Duplicate levels (e.g. a
    stress level equal to the base) are de-duplicated, and the rows are returned
    in ascending cost order.

    Parameters
    ----------
    prices:
        Adjusted-close price panel used by the backtest.
    target_weights:
        Target weights produced by the strategy (signal -> sizing). Computed
        once by the caller and reused across all cost levels.
    cfg:
        Typed configuration providing the base cost and the stress levels. It is
        never mutated; a frozen copy with a different cost is used per level.

    Returns
    -------
    list[CostStressRow]
        One row per distinct cost level, ascending by cost, each flagged as base
        or stressed.
    """
    base = float(cfg.cost_bps_per_side)
    # Preserve the base flag while de-duplicating: a stress level equal to the
    # base collapses onto the base row.
    levels: dict[float, bool] = {base: True}
    for lvl in cfg.cost_stress_levels:
        levels.setdefault(float(lvl), False)

    rows = [
        _replay_at_cost(prices, target_weights, cfg, level, is_base=is_base)
        for level, is_base in levels.items()
    ]
    rows.sort(key=lambda row: row.cost_bps_per_side)
    return rows


def cost_stress_table(rows: list[CostStressRow]) -> pd.DataFrame:
    """Assemble cost-stress rows into a tidy comparison :class:`DataFrame`.

    Parameters
    ----------
    rows:
        Rows from :func:`cost_stress_test`.

    Returns
    -------
    pandas.DataFrame
        One row per cost level with columns ``cost_bps_per_side``, ``is_base``,
        ``cagr``, ``sharpe``, ``max_drawdown``, ``total_cost`` and
        ``final_equity``, indexed by ``cost_bps_per_side``.
    """
    records: list[dict[str, Any]] = [
        {
            "cost_bps_per_side": row.cost_bps_per_side,
            "is_base": row.is_base,
            "cagr": row.cagr,
            "sharpe": row.sharpe,
            "max_drawdown": row.max_drawdown,
            "total_cost": row.total_cost,
            "final_equity": row.final_equity,
        }
        for row in rows
    ]
    table = pd.DataFrame.from_records(
        records,
        columns=[
            "cost_bps_per_side",
            "is_base",
            "cagr",
            "sharpe",
            "max_drawdown",
            "total_cost",
            "final_equity",
        ],
    )
    if not table.empty:
        table = table.set_index("cost_bps_per_side")
    return table
