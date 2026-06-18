"""Transaction-cost model (spec 3.4 / 5).

Pure, side-effect-free cost functions used by the backtest engine to charge
trading frictions on rebalancing turnover.

Two components are modeled:

1. **Linear (spread/commission) cost** -- the mandatory term:

       cost = cost_bps_per_side * |delta notional|

   where ``cost_bps_per_side`` is expressed in *basis points* (1 bp = 1e-4)
   charged on the absolute traded notional of every side of a trade. This is
   the realistic, pessimistic baseline the engine always applies.

2. **Square-root market impact** -- the optional term, present in code for
   future calibration but *negligible/disabled by default* at research scale:

       impact = c * sigma_i * sqrt(|delta notional| / ADV)

   where ``sigma_i`` is the asset's (e.g. daily) volatility, ``ADV`` its
   average daily traded volume (in the same notional units as the trade) and
   ``c`` a dimensionless calibration constant. With ``c = 0`` (the default) the
   impact term vanishes, so research-scale backtests are unaffected while the
   capability remains wired in for later use.

All functions are vectorized over array-like inputs, perform no I/O, and never
mutate their arguments. Notionals are treated as absolute values internally, so
the sign of a trade never affects the (always non-negative) cost it incurs.
"""

from __future__ import annotations

import numpy as np

__all__ = ["bps_to_fraction", "linear_cost", "impact_cost"]

# 1 basis point expressed as a plain fraction (1 bp = 0.01% = 1e-4).
_BPS: float = 1e-4


def bps_to_fraction(bps: float) -> float:
    """Convert a basis-point figure to a plain fraction.

    Parameters
    ----------
    bps:
        Cost in basis points (e.g. ``2`` for 2 bps). One basis point equals
        ``1e-4``.

    Returns
    -------
    float
        The equivalent fraction (``bps * 1e-4``).
    """
    return float(bps) * _BPS


def linear_cost(
    delta_notional: np.ndarray | float,
    cost_bps_per_side: float,
) -> np.ndarray | float:
    """Linear per-side transaction cost on traded notional (the spec term).

    Implements ``cost = cost_bps_per_side * |delta notional|`` with the
    basis-point rate converted to a fraction. The cost is charged on the
    *absolute* traded notional, so it is always non-negative regardless of trade
    direction. This is the mandatory cost component the backtest engine applies
    on every rebalance.

    Parameters
    ----------
    delta_notional:
        Traded notional per asset (currency units). May be a scalar or any
        array-like; its sign is ignored (absolute value is used).
    cost_bps_per_side:
        Transaction cost in basis points charged per side (from
        ``cfg.cost_bps_per_side`` or a stress level). Must be non-negative.

    Returns
    -------
    numpy.ndarray | float
        Non-negative cost in currency units, broadcast to the shape of
        ``delta_notional`` (a Python ``float`` for scalar input).

    Raises
    ------
    ValueError
        If ``cost_bps_per_side`` is negative.
    """
    if cost_bps_per_side < 0:
        raise ValueError(
            f"'cost_bps_per_side' must be non-negative, got {cost_bps_per_side}."
        )
    rate = bps_to_fraction(cost_bps_per_side)
    abs_notional = np.abs(np.asarray(delta_notional, dtype="float64"))
    cost = rate * abs_notional
    # Preserve scalar-in / scalar-out ergonomics.
    if np.isscalar(delta_notional) or np.ndim(delta_notional) == 0:
        return float(cost)
    return cost


def impact_cost(
    delta_notional: np.ndarray | float,
    sigma: np.ndarray | float,
    adv: np.ndarray | float,
    c: float = 0.0,
) -> np.ndarray | float:
    """Optional square-root market-impact cost (calibratable, off by default).

    Implements ``impact = c * sigma_i * sqrt(|delta notional| / ADV)``. This
    term captures price impact that grows with the square root of trade size
    relative to average daily volume; it is **disabled by default** (``c = 0``)
    so research-scale backtests are unchanged, but the implementation is fully
    present so it can be calibrated and enabled later without code changes.

    The impact is always non-negative. Where ``ADV`` is non-positive or
    non-finite, impact is treated as ``0`` (no volume reference -> no estimate)
    rather than producing infinities, so the term degrades gracefully.

    Parameters
    ----------
    delta_notional:
        Traded notional per asset (currency units); sign ignored.
    sigma:
        Per-asset volatility used to scale impact (e.g. daily return vol), in
        the same convention the caller calibrated ``c`` against.
    adv:
        Average daily traded volume per asset, expressed as a notional in the
        same currency units as ``delta_notional``.
    c:
        Dimensionless impact-calibration constant. ``0.0`` (default) disables
        the term entirely. Must be non-negative.

    Returns
    -------
    numpy.ndarray | float
        Non-negative impact cost in currency units, broadcast across inputs (a
        Python ``float`` for all-scalar input).

    Raises
    ------
    ValueError
        If ``c`` is negative.
    """
    if c < 0:
        raise ValueError(f"impact constant 'c' must be non-negative, got {c}.")

    scalar_in = (
        np.isscalar(delta_notional)
        and np.isscalar(sigma)
        and np.isscalar(adv)
    )

    dn = np.abs(np.asarray(delta_notional, dtype="float64"))
    sig = np.asarray(sigma, dtype="float64")
    advol = np.asarray(adv, dtype="float64")

    # Fast exit (and exact zero) when the term is disabled.
    if c == 0.0:
        zeros = np.zeros(np.broadcast(dn, sig, advol).shape, dtype="float64")
        return 0.0 if scalar_in else zeros

    # Participation rate |delta notional| / ADV; guard non-positive/NaN ADV.
    valid_adv = np.isfinite(advol) & (advol > 0.0)
    participation = np.where(valid_adv, dn / np.where(valid_adv, advol, 1.0), 0.0)
    impact = float(c) * np.abs(sig) * np.sqrt(participation)
    impact = np.where(np.isfinite(impact), impact, 0.0)

    if scalar_in:
        return float(impact)
    return impact
