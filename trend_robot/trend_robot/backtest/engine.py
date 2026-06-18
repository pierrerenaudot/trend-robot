"""Realistic, look-ahead-free backtest engine (spec 4 / 5).

Translates a stream of *target* portfolio weights into a marked-to-market equity
curve, charging transaction costs on rebalancing turnover. The engine is
deliberately framework-free (no heavy backtest library) and reproducible.

Mechanics
---------
* **No look-ahead.** Target weights computed at date ``t`` are *shifted by at
  least one bar* (``shift(1)``) before they can be acted upon, so the book held
  over ``(t, t+1]`` was decided using only information available at ``t``. A
  weight observed at ``t`` is therefore first *executed* at ``t+1``.
* **Rebalance cadence.** The book is only re-set toward the (lagged) target on
  the rebalance dates implied by ``cfg.rebalance`` (``daily`` / ``weekly`` /
  ``monthly``). Between rebalances the held weights *drift* with realized
  returns (mark-to-market): a winning asset's weight grows, a losing one's
  shrinks -- no trading occurs and therefore no cost is charged.
* **Costs on turnover.** On each rebalance the per-asset traded notional is
  ``|w_target_i - w_drifted_i| * equity``; the linear cost model
  (:mod:`trend_robot.backtest.costs`) charges ``cost_bps_per_side`` on it. Costs
  are deducted from equity, so higher turnover -> higher cost -> lower equity.
* **Mark to market.** Equity compounds by the portfolio return
  ``r_p(t) = sum_i w_i(t-1) * r_i(t)`` each bar (weights as of the prior close),
  net of any cost incurred when rebalancing into that bar.

Weights are interpreted as fractions of *current* portfolio equity (a
self-financing book); ``sum_i |w_i|`` is the gross leverage. Anything not in a
position is implicitly un-invested (earns nothing), which is the pessimistic,
research-appropriate convention.

This module performs no market I/O and hard-codes no market values: every
parameter (cadence, costs, annualization) flows from the typed :class:`Config`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trend_robot.backtest.costs import bps_to_fraction
from trend_robot.config import Config

__all__ = ["BacktestResult", "run_backtest"]


# Map the configured cadence to a pandas period code used to bucket dates.
# A new bucket (relative to the previous bar) marks a rebalance opportunity.
_CADENCE_FREQ: dict[str, str] = {
    "daily": "D",
    "weekly": "W",
    "monthly": "M",
}


@dataclass
class BacktestResult:
    """Container for the output of a single backtest run.

    Attributes
    ----------
    equity:
        Marked-to-market portfolio equity, indexed by trading day. Starts from
        ``cfg.initial_capital`` and compounds net of costs.
    weights:
        Realized *held* weights per asset and date (the book actually carried
        into each bar, after lagging, drift and rebalancing). Same columns as
        the input ``prices``.
    turnover:
        Per-date one-sided turnover as a fraction of equity,
        ``sum_i |w_target_i - w_drifted_i|`` (``0`` on non-rebalance dates).
    trades:
        Long-format ledger with one row per executed asset trade and columns
        ``["date", "asset", "delta_weight", "cost"]`` (``delta_weight`` is the
        change in weight; ``cost`` is the currency cost charged on it).
    """

    equity: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    trades: pd.DataFrame


def _rebalance_mask(index: pd.DatetimeIndex, rebalance: str) -> np.ndarray:
    """Boolean mask flagging the rebalance dates for a cadence.

    A date is a rebalance date when it falls in a new calendar bucket (day /
    week / month, per ``rebalance``) relative to the previous bar. The very
    first usable bar is always a rebalance (the book is established there).

    Parameters
    ----------
    index:
        Tz-naive trading-day index of the backtest.
    rebalance:
        Cadence string (``"daily"``, ``"weekly"`` or ``"monthly"``).

    Returns
    -------
    numpy.ndarray
        Boolean array aligned to ``index``; ``True`` on rebalance dates.

    Raises
    ------
    ValueError
        If ``rebalance`` is not a recognized cadence.
    """
    if rebalance not in _CADENCE_FREQ:
        raise ValueError(
            f"'rebalance' must be one of {sorted(_CADENCE_FREQ)}, "
            f"got {rebalance!r}."
        )

    n = len(index)
    mask = np.zeros(n, dtype=bool)
    if n == 0:
        return mask

    if rebalance == "daily":
        # Every bar is a rebalance opportunity.
        mask[:] = True
        return mask

    # Bucket each date into its calendar period; a change of period since the
    # previous bar marks a new rebalance.
    periods = index.to_period(_CADENCE_FREQ[rebalance])
    codes = periods.asi8  # integer period codes, monotonic in time
    mask[0] = True
    mask[1:] = codes[1:] != codes[:-1]
    return mask


def run_backtest(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    cfg: Config,
) -> BacktestResult:
    """Run a marked-to-market, cost-aware backtest with no look-ahead.

    Parameters
    ----------
    prices:
        Adjusted-close prices (the data contract from
        :mod:`trend_robot.data.provider`): tz-naive trading-day index, one
        column per asset, explicit ``NaN`` gaps. Drives realized returns.
    target_weights:
        Desired per-asset weights (fractions of equity) on each date, e.g. from
        :func:`trend_robot.portfolio.sizing.target_weights`. They are reindexed
        onto ``prices`` and **lagged by one bar** before execution, so the book
        held over a bar was decided strictly before that bar. ``NaN`` targets
        are treated as ``0`` (no position).
    cfg:
        Typed configuration providing ``initial_capital``, ``rebalance`` and
        ``cost_bps_per_side``.

    Returns
    -------
    BacktestResult
        Equity curve, realized held weights, per-date turnover and the trade
        ledger (see :class:`BacktestResult`).

    Raises
    ------
    TypeError
        If ``prices`` or ``target_weights`` is not a :class:`pandas.DataFrame`.
    ValueError
        If ``cfg.rebalance`` is not a recognized cadence.

    Notes
    -----
    The engine is deterministic and side-effect-free: it never mutates its
    inputs and performs no I/O. Identical inputs always produce an identical
    :class:`BacktestResult`.
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError(
            f"'prices' must be a pandas DataFrame, got {type(prices).__name__}."
        )
    if not isinstance(target_weights, pd.DataFrame):
        raise TypeError(
            "'target_weights' must be a pandas DataFrame, got "
            f"{type(target_weights).__name__}."
        )

    index = prices.index
    columns = prices.columns

    # --- Degenerate panels: return contract-shaped empties. ----------------
    if prices.shape[0] == 0 or prices.shape[1] == 0:
        equity = pd.Series(dtype="float64", index=index, name="equity")
        weights = pd.DataFrame(index=index, columns=columns, dtype="float64")
        turnover = pd.Series(dtype="float64", index=index, name="turnover")
        trades = pd.DataFrame(
            columns=["date", "asset", "delta_weight", "cost"]
        ).astype(
            {"asset": "object", "delta_weight": "float64", "cost": "float64"}
        )
        return BacktestResult(equity, weights, turnover, trades)

    px = prices.astype("float64")

    # --- Realized per-asset returns; NaN gaps -> 0 (asset earns nothing). ---
    # A missing/NaN price means the asset is not investable that bar; we
    # neither profit nor lose on a position we could not realistically hold.
    asset_returns = px.pct_change(fill_method=None)
    asset_returns = asset_returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # --- No look-ahead: align targets to the price grid and lag by 1 bar. --
    # The weight decided at t only governs the book held *into* t+1.
    tgt = target_weights.reindex(index=index, columns=columns)
    tgt = tgt.astype("float64").fillna(0.0)
    lagged_target = tgt.shift(1)  # first row -> NaN -> treated as flat below
    lagged_target.iloc[0] = 0.0

    rebal = _rebalance_mask(index, cfg.rebalance)
    cost_rate = bps_to_fraction(float(cfg.cost_bps_per_side))
    initial_capital = float(cfg.initial_capital)

    n = len(index)
    asset_list = list(columns)

    equity_vals = np.empty(n, dtype="float64")
    turnover_vals = np.zeros(n, dtype="float64")
    held_weights = np.zeros((n, len(asset_list)), dtype="float64")

    trade_dates: list[pd.Timestamp] = []
    trade_assets: list[str] = []
    trade_deltas: list[float] = []
    trade_costs: list[float] = []

    ret_arr = asset_returns.to_numpy(dtype="float64")
    tgt_arr = lagged_target.to_numpy(dtype="float64")

    # Weights carried *into* the current bar (as of the previous close). The
    # book starts flat; the first rebalance establishes positions.
    current_w = np.zeros(len(asset_list), dtype="float64")
    equity = initial_capital

    for t in range(n):
        # 1) Decide the book for bar t. On a rebalance date we trade from the
        #    drifted weights toward the lagged target, charging cost on the
        #    one-sided turnover. Between rebalances we hold (no trade, no cost).
        bar_cost_fraction = 0.0
        if rebal[t]:
            desired = tgt_arr[t]
            delta = desired - current_w
            turnover = float(np.abs(delta).sum())
            turnover_vals[t] = turnover
            bar_cost_fraction = cost_rate * turnover

            if turnover > 0.0:
                equity_at_trade = equity
                for j, asset in enumerate(asset_list):
                    dw = float(delta[j])
                    if dw == 0.0:
                        continue
                    notional = abs(dw) * equity_at_trade
                    trade_dates.append(index[t])
                    trade_assets.append(asset)
                    trade_deltas.append(dw)
                    trade_costs.append(cost_rate * notional)

            current_w = desired.copy()

        # 2) Mark to market over bar t using the book held into the bar.
        #    Cost (if any) is paid up-front, reducing the capital that compounds.
        port_ret = float(np.dot(current_w, ret_arr[t]))
        equity = equity * (1.0 - bar_cost_fraction) * (1.0 + port_ret)

        equity_vals[t] = equity
        held_weights[t] = current_w

        # 3) Drift the held weights with realized returns for the next bar
        #    (mark-to-market): w_i grows/shrinks with its asset's gross return,
        #    renormalized by the realized portfolio gross return. Uninvested
        #    cash earns nothing, so the gross factor is 1 + port_ret.
        gross = 1.0 + port_ret
        if gross != 0.0 and np.isfinite(gross):
            current_w = current_w * (1.0 + ret_arr[t]) / gross
        # If gross is zero/non-finite (a total wipeout edge case) keep weights
        # as-is; equity already reflects the move.

    equity_series = pd.Series(equity_vals, index=index, name="equity")
    weights_df = pd.DataFrame(held_weights, index=index, columns=columns)
    turnover_series = pd.Series(turnover_vals, index=index, name="turnover")

    trades_df = pd.DataFrame(
        {
            "date": trade_dates,
            "asset": trade_assets,
            "delta_weight": trade_deltas,
            "cost": trade_costs,
        }
    ).astype(
        {"asset": "object", "delta_weight": "float64", "cost": "float64"}
    )

    return BacktestResult(
        equity=equity_series,
        weights=weights_df,
        turnover=turnover_series,
        trades=trades_df,
    )
