"""Honest performance metrics for a backtest (spec 3.5 / 7).

Turns a :class:`~trend_robot.backtest.engine.BacktestResult` into a flat
dictionary of *honest* performance statistics plus a per-asset profit-and-loss
attribution. "Honest" here means: realized (net-of-cost) returns only, standard
annualization driven entirely by ``cfg`` (no hard-coded market constants), and
no silent smoothing of drawdowns or losing streaks.

Reported statistics
-------------------
* **CAGR** -- geometric (compound) annual growth rate of equity.
* **annual_vol** -- annualized volatility of periodic (per-bar) returns.
* **sharpe** -- annualized Sharpe ratio (zero risk-free rate, the
  research-appropriate pessimistic convention).
* **sortino** -- like Sharpe but penalizing only downside deviation.
* **calmar** (a.k.a. MAR) -- CAGR divided by the absolute max drawdown.
* **max_drawdown** -- worst peak-to-trough equity decline (a negative number).
* **max_drawdown_duration** -- longest peak-to-recovery span, in bars.
* **profit_factor** -- gross gains / gross losses across bars.
* **hit_rate** -- fraction of bars with a strictly positive return.
* **avg_annual_turnover** -- mean one-sided turnover scaled to a year.
* **avg_exposure** -- average gross exposure ``sum_i |w_i|`` carried.
* **n_periods** / **start** / **end** -- bookkeeping.
* **per_asset_pnl** -- per-asset realized P&L attribution (see below).

This module is pure and side-effect-free: it performs no I/O, hard-codes no
market values (annualization comes from ``cfg.periods_per_year``), and never
mutates its inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trend_robot.backtest.engine import BacktestResult
from trend_robot.config import Config

__all__ = ["performance_metrics"]


def _equity_returns(equity: pd.Series) -> pd.Series:
    """Per-bar simple returns implied by an equity curve.

    Parameters
    ----------
    equity:
        Marked-to-market equity series (strictly the realized, net-of-cost
        curve produced by the engine).

    Returns
    -------
    pandas.Series
        Simple returns ``equity_t / equity_{t-1} - 1`` with the leading
        ``NaN`` dropped. Non-finite values (from a zero/NaN predecessor) are
        dropped so downstream statistics stay well defined.
    """
    rets = equity.pct_change(fill_method=None)
    rets = rets.replace([np.inf, -np.inf], np.nan).dropna()
    return rets


def _cagr(equity: pd.Series, periods_per_year: int) -> float:
    """Compound annual growth rate from first to last equity point.

    Parameters
    ----------
    equity:
        Equity curve.
    periods_per_year:
        Annualization factor (bars per year, e.g. ``cfg.periods_per_year``).

    Returns
    -------
    float
        Geometric annual growth rate, or ``nan`` when it is undefined
        (fewer than two points, non-positive endpoints).
    """
    if equity.size < 2:
        return float("nan")
    start = float(equity.iloc[0])
    end = float(equity.iloc[-1])
    if start <= 0.0 or end <= 0.0:
        return float("nan")
    n_periods = equity.size - 1  # number of return steps
    years = n_periods / float(periods_per_year)
    if years <= 0.0:
        return float("nan")
    return float((end / start) ** (1.0 / years) - 1.0)


def _annual_vol(returns: pd.Series, periods_per_year: int) -> float:
    """Annualized volatility of per-bar returns (sample std, ``ddof=1``)."""
    if returns.size < 2:
        return float("nan")
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def _sharpe(returns: pd.Series, periods_per_year: int) -> float:
    """Annualized Sharpe ratio with a zero risk-free rate.

    Computed as ``mean(returns) / std(returns) * sqrt(periods_per_year)``
    using the sample standard deviation (``ddof=1``). Returns ``nan`` when the
    return series is too short or has zero dispersion.
    """
    if returns.size < 2:
        return float("nan")
    sd = returns.std(ddof=1)
    if sd == 0.0 or not np.isfinite(sd):
        return float("nan")
    return float(returns.mean() / sd * np.sqrt(periods_per_year))


def _sortino(returns: pd.Series, periods_per_year: int) -> float:
    """Annualized Sortino ratio (downside-deviation denominator).

    The downside deviation is the root-mean-square of the *negative* returns
    (returns below the zero target), using ``ddof=0`` over those observations.
    Returns ``nan`` if there is no downside dispersion to normalize against.
    """
    if returns.size < 2:
        return float("nan")
    downside = returns[returns < 0.0]
    if downside.empty:
        return float("nan")
    # Root-mean-square of negative returns about the zero target.
    dd = float(np.sqrt(np.mean(np.square(downside.to_numpy()))))
    if dd == 0.0 or not np.isfinite(dd):
        return float("nan")
    return float(returns.mean() / dd * np.sqrt(periods_per_year))


def _drawdown_curve(equity: pd.Series) -> pd.Series:
    """Drawdown series ``equity / running_max - 1`` (<= 0 everywhere)."""
    running_max = equity.cummax()
    return equity / running_max - 1.0


def _max_drawdown(equity: pd.Series) -> float:
    """Worst peak-to-trough decline as a non-positive fraction.

    Returns ``0.0`` for a non-decreasing curve and ``nan`` for an empty curve.
    """
    if equity.empty:
        return float("nan")
    dd = _drawdown_curve(equity)
    return float(dd.min())


def _max_drawdown_duration(equity: pd.Series) -> int:
    """Longest underwater stretch, measured in bars.

    The duration is the maximum number of consecutive bars during which equity
    sits strictly below a prior running peak (peak-to-recovery). A curve that
    never draws down has duration ``0``.

    Parameters
    ----------
    equity:
        Equity curve.

    Returns
    -------
    int
        Longest underwater run length in bars (``0`` if never underwater).
    """
    if equity.empty:
        return 0
    dd = _drawdown_curve(equity).to_numpy()
    underwater = dd < 0.0
    longest = 0
    current = 0
    for flag in underwater:
        if flag:
            current += 1
            if current > longest:
                longest = current
        else:
            current = 0
    return int(longest)


def _profit_factor(returns: pd.Series) -> float:
    """Gross gains divided by gross losses across per-bar returns.

    Returns ``inf`` when there are gains but no losses, ``0.0`` when there are
    losses but no gains, and ``nan`` when there is neither.
    """
    if returns.empty:
        return float("nan")
    gains = float(returns[returns > 0.0].sum())
    losses = float(-returns[returns < 0.0].sum())
    if losses == 0.0:
        return float("inf") if gains > 0.0 else float("nan")
    return float(gains / losses)


def _hit_rate(returns: pd.Series) -> float:
    """Fraction of bars with a strictly positive return."""
    if returns.empty:
        return float("nan")
    return float((returns > 0.0).mean())


def _avg_annual_turnover(turnover: pd.Series, periods_per_year: int) -> float:
    """Mean per-bar one-sided turnover scaled to an annual figure.

    Parameters
    ----------
    turnover:
        Per-date one-sided turnover (fraction of equity) from the engine;
        zero on non-rebalance dates.
    periods_per_year:
        Annualization factor.

    Returns
    -------
    float
        ``mean(turnover) * periods_per_year`` (``nan`` for an empty series).
    """
    if turnover.empty:
        return float("nan")
    return float(turnover.mean() * periods_per_year)


def _avg_exposure(weights: pd.DataFrame) -> float:
    """Average gross exposure ``sum_i |w_i|`` carried across bars."""
    if weights.empty:
        return float("nan")
    gross = weights.abs().sum(axis=1)
    return float(gross.mean())


def _per_asset_pnl(result: BacktestResult) -> dict[str, float]:
    """Per-asset realized P&L attribution (currency units).

    The realized portfolio P&L of bar ``t`` (already net of any cost charged
    into that bar, since it is read straight off the equity curve) is split
    across assets by each asset's share of the signed dollar exposure carried
    into the bar, ``equity_{t-1} * w_i(t-1)``. Summing the per-asset
    contributions therefore reconciles exactly with the total change in equity:
    ``sum_i pnl_i == equity_last - equity_first``. Using *signed* exposure means
    longs and shorts attribute with the correct sign.

    Parameters
    ----------
    result:
        Backtest result carrying ``equity`` and held ``weights``.

    Returns
    -------
    dict[str, float]
        Mapping ``asset -> realized P&L`` (currency units). Empty if there are
        no assets or no return bars.
    """
    weights = result.weights
    equity = result.equity
    if weights.empty or equity.size < 2:
        return {col: 0.0 for col in weights.columns}

    # Capital deployed into each asset at the *start* of each bar is
    # equity_{t-1} * w_i(t-1). The realized portfolio P&L of bar t is
    # equity_t - (equity_{t-1} adjusted for cost). We attribute the gross
    # portfolio move proportionally to each asset's signed dollar exposure.
    eq = equity.to_numpy(dtype="float64")
    w = weights.to_numpy(dtype="float64")  # held weights, as of each close

    # Dollar exposure carried INTO bar t is equity_{t-1} * w(t-1).
    prev_eq = eq[:-1]  # equity_{t-1}
    prev_w = w[:-1]  # w(t-1)
    dollar_exposure = prev_w * prev_eq[:, None]  # shape (T-1, n_assets)

    # Realized portfolio P&L of bar t (net of any cost charged into the bar).
    port_pnl = eq[1:] - eq[:-1]  # shape (T-1,)

    # Split each bar's P&L across assets by their share of signed exposure.
    # Where total signed exposure is ~0, fall back to no attribution for that
    # bar (the move is then cost/cash drift, not asset-driven).
    total_exposure = dollar_exposure.sum(axis=1)  # signed sum
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(
            np.abs(total_exposure)[:, None] > 1e-12,
            dollar_exposure / total_exposure[:, None],
            0.0,
        )
    contrib = share * port_pnl[:, None]  # (T-1, n_assets)
    contrib = np.where(np.isfinite(contrib), contrib, 0.0)

    totals = contrib.sum(axis=0)
    return {col: float(totals[j]) for j, col in enumerate(weights.columns)}


def performance_metrics(result: BacktestResult, cfg: Config) -> dict:
    """Compute honest performance statistics for a backtest result.

    All return-based statistics use the realized, net-of-cost equity curve;
    annualization is driven by ``cfg.periods_per_year`` (no hard-coded market
    constants). The function is pure: it neither mutates ``result`` nor performs
    any I/O.

    Parameters
    ----------
    result:
        Output of :func:`trend_robot.backtest.engine.run_backtest` (equity,
        held weights, turnover, trade ledger).
    cfg:
        Typed configuration; ``periods_per_year`` sets annualization.

    Returns
    -------
    dict
        Flat mapping of metric name to value. Keys:
        ``cagr``, ``annual_vol``, ``sharpe``, ``sortino``, ``calmar``
        (alias ``mar``), ``max_drawdown``, ``max_drawdown_duration``,
        ``profit_factor``, ``hit_rate``, ``avg_annual_turnover``,
        ``avg_exposure``, ``n_periods``, ``start``, ``end``, ``total_cost``,
        and ``per_asset_pnl`` (a ``dict`` of asset -> realized P&L).
        Undefined statistics are reported as ``nan`` rather than raising.

    Raises
    ------
    TypeError
        If ``result`` is not a :class:`BacktestResult`.
    """
    if not isinstance(result, BacktestResult):
        raise TypeError(
            f"'result' must be a BacktestResult, got {type(result).__name__}."
        )

    ppy = int(cfg.periods_per_year)
    equity = result.equity.astype("float64")
    returns = _equity_returns(equity)

    cagr = _cagr(equity, ppy)
    max_dd = _max_drawdown(equity)

    # Calmar / MAR: CAGR over the magnitude of the worst drawdown.
    if np.isfinite(cagr) and np.isfinite(max_dd) and max_dd < 0.0:
        calmar = float(cagr / abs(max_dd))
    elif np.isfinite(cagr) and max_dd == 0.0:
        calmar = float("inf") if cagr > 0.0 else float("nan")
    else:
        calmar = float("nan")

    total_cost = (
        float(result.trades["cost"].sum())
        if not result.trades.empty
        else 0.0
    )

    metrics: dict = {
        "cagr": cagr,
        "annual_vol": _annual_vol(returns, ppy),
        "sharpe": _sharpe(returns, ppy),
        "sortino": _sortino(returns, ppy),
        "calmar": calmar,
        "mar": calmar,  # MAR ratio is an alias of Calmar.
        "max_drawdown": max_dd,
        "max_drawdown_duration": _max_drawdown_duration(equity),
        "profit_factor": _profit_factor(returns),
        "hit_rate": _hit_rate(returns),
        "avg_annual_turnover": _avg_annual_turnover(result.turnover, ppy),
        "avg_exposure": _avg_exposure(result.weights),
        "n_periods": int(returns.size),
        "start": equity.index[0] if equity.size else None,
        "end": equity.index[-1] if equity.size else None,
        "total_cost": total_cost,
        "per_asset_pnl": _per_asset_pnl(result),
    }
    return metrics
