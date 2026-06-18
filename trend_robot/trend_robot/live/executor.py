"""Order planning: reconcile a target book against current positions.

The planner :func:`plan_orders` is a PURE function that turns a target weight
vector, the current positions and last prices into a deterministic list of
:class:`OrderIntent` objects -- the trades that move the book from "current" to
"target". It performs the same notional/share arithmetic a live OMS would, with
guards for minimum trade size, whole-share rounding and the gross-leverage cap.

Nothing here sends an order; the runner decides whether to preview (dry-run) or
submit. All money values flow from the caller (``equity``, ``last_px``) and the
typed :class:`Config`; nothing is hard-coded.
"""

from __future__ import annotations

import math

import pandas as pd

from trend_robot.config import Config
from trend_robot.live.broker import OrderIntent, Position

__all__ = ["plan_orders", "summarize_plan"]


def _truncate(value: float) -> float:
    """Truncate ``value`` toward zero to a whole number (signed)."""
    return float(math.trunc(value))


def plan_orders(
    target_w: pd.Series,
    positions: dict[str, Position],
    last_px: pd.Series,
    equity: float,
    cfg: Config,
    *,
    min_trade_notional: float = 0.0,
    allow_fractional: bool = False,
) -> list[OrderIntent]:
    """Plan the rebalance orders that move current -> target.

    For each symbol in the union of the target index and the current positions:

    * ``target_notional = target_w[sym] * equity``;
    * ``cur_notional = cur_qty * price`` where ``cur_qty`` is the held quantity
      (``0`` if none) and ``price = last_px[sym]``;
    * ``delta_notional = target_notional - cur_notional``;
    * skip (reason ``"below_min_trade"``) if ``abs(delta_notional) <
      min_trade_notional``;
    * ``qty_delta = delta_notional / price``; unless ``allow_fractional`` it is
      truncated toward zero to whole shares; skip (reason ``"rounds_to_zero"``)
      if it becomes ``0``;
    * ``side`` is ``"buy"`` for a positive delta, ``"sell"`` otherwise; the
      :class:`OrderIntent` carries ``abs(qty)``, the est. price, the traded
      notional, the target/current weights and a reason (``"close"`` when the
      target weight is ``0`` and a position was held, else ``"rebalance"``).

    A held symbol that is absent from (or zero in) the target is sold to flat.

    Parameters
    ----------
    target_w:
        Today's target weights, indexed by symbol.
    positions:
        Current positions keyed by symbol.
    last_px:
        Last known price per symbol (indexed by symbol).
    equity:
        Account equity used to translate weights into notionals.
    cfg:
        Typed configuration (``max_gross_leverage`` for the guard).
    min_trade_notional:
        Trades with an absolute delta notional below this are skipped.
    allow_fractional:
        When ``False`` (default), order quantities are truncated to whole shares.

    Returns
    -------
    list[OrderIntent]
        Orders sorted by symbol (deterministic). Skipped symbols are omitted.

    Raises
    ------
    ValueError
        If the target's gross exposure exceeds ``cfg.max_gross_leverage``
        (the sizing layer should already enforce this; this is a safety net).
    """
    equity = float(equity)

    # --- Gross-leverage guard (sizing should already enforce it). ----------
    gross = float(target_w.abs().sum()) if len(target_w) else 0.0
    if gross > float(cfg.max_gross_leverage) + 1e-9:
        raise ValueError(
            f"Target gross exposure {gross:.6f} exceeds max_gross_leverage "
            f"{cfg.max_gross_leverage:.6f}; refusing to plan orders. (Sizing "
            f"should already cap this -- check upstream weight computation.)"
        )

    symbols = sorted(set(target_w.index) | set(positions.keys()))
    intents: list[OrderIntent] = []

    for sym in symbols:
        tw = float(target_w.get(sym, 0.0))
        if pd.isna(tw):
            tw = 0.0

        pos = positions.get(sym)
        cur_qty = float(pos.qty) if pos is not None else 0.0

        price = float(last_px.get(sym, float("nan")))
        if not math.isfinite(price) or price <= 0.0:
            # No usable price -> cannot size a trade for this symbol; skip it.
            continue

        target_notional = tw * equity
        cur_notional = cur_qty * price
        cur_weight = (cur_notional / equity) if equity > 0.0 else 0.0
        delta_notional = target_notional - cur_notional

        if abs(delta_notional) < float(min_trade_notional):
            continue  # below_min_trade

        qty_delta = delta_notional / price
        if not allow_fractional:
            qty_delta = _truncate(qty_delta)
        if qty_delta == 0.0:
            continue  # rounds_to_zero

        side = "buy" if qty_delta > 0.0 else "sell"
        had_position = cur_qty != 0.0
        reason = "close" if (tw == 0.0 and had_position) else "rebalance"
        abs_qty = abs(qty_delta)

        intents.append(
            OrderIntent(
                symbol=sym,
                side=side,
                qty=abs_qty,
                est_price=price,
                notional=abs_qty * price,
                target_weight=tw,
                current_weight=cur_weight,
                reason=reason,
            )
        )

    return intents


def summarize_plan(
    intents: list[OrderIntent],
    target_w: pd.Series,
    cfg: Config,
) -> dict[str, float | int]:
    """Summarize a plan: counts, buy/sell notionals, gross exposure, est. cost.

    Parameters
    ----------
    intents:
        The planned orders (from :func:`plan_orders`).
    target_w:
        Today's target weights (for the gross-exposure figure).
    cfg:
        Typed configuration (``cost_bps_per_side`` for the cost estimate).

    Returns
    -------
    dict
        ``n_orders``, ``total_buy_notional``, ``total_sell_notional``,
        ``gross_exposure`` (``sum|target_w|``) and ``est_cost``
        (``cost_bps_per_side / 1e4 * sum(notional)``).
    """
    total_buy = sum(i.notional for i in intents if i.side == "buy")
    total_sell = sum(i.notional for i in intents if i.side == "sell")
    total_notional = total_buy + total_sell
    gross_exposure = float(target_w.abs().sum()) if len(target_w) else 0.0
    est_cost = (float(cfg.cost_bps_per_side) / 1e4) * total_notional
    return {
        "n_orders": len(intents),
        "total_buy_notional": float(total_buy),
        "total_sell_notional": float(total_sell),
        "gross_exposure": gross_exposure,
        "est_cost": float(est_cost),
    }
