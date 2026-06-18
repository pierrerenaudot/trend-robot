"""Time-series momentum (TSMOM) signal (spec 3.2).

Pure, side-effect-free signal generation. The public entry point
:func:`tsmom_signal` maps a panel of adjusted-close prices to a per-asset,
per-date momentum signal in ``[-1, 1]``.

Formula
-------
For each asset and each lookback horizon ``L`` (in trading days):

    s_L(t) = sign(P_t / P_{t-L} - 1)        (== sign of the L-day return)

The final signal is the simple mean of the per-horizon signs across all
lookbacks, which is itself bounded in ``[-1, 1]`` (each term is in
``{-1, 0, +1}``)::

    s(t) = mean_L s_L(t)

Direction
---------
* ``"long_short"`` -- signal kept as-is in ``[-1, 1]``.
* ``"long_only"``  -- negative signals truncated to zero: ``s = max(s, 0)``.

No look-ahead
-------------
``P_{t-L}`` is obtained via :meth:`pandas.DataFrame.shift`, so the signal at
date ``t`` depends only on prices observed up to and including ``t``. No future
information ever enters the computation. (Execution-side lagging of the signal
before trading is the backtest engine's responsibility, not this function's.)

Insufficient history / gaps
---------------------------
A horizon is undefined at ``t`` when ``P_{t-L}`` does not exist (the first ``L``
rows) or when either ``P_t`` or ``P_{t-L}`` is ``NaN`` (an explicit price gap).
To avoid biasing the cross-horizon mean with a partial subset of horizons, the
final signal at ``t`` is ``NaN`` unless *every* requested lookback is defined at
``t``. This is the look-ahead-safe convention: the signal becomes available only
once the longest lookback has enough history. Input ``NaN`` price gaps therefore
propagate to ``NaN`` signals (no silent forward-fill), honoring the data
contract in :mod:`trend_robot.data.provider`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["tsmom_signal"]

_VALID_DIRECTIONS: frozenset[str] = frozenset({"long_short", "long_only"})


def tsmom_signal(
    prices: pd.DataFrame,
    lookbacks: list[int],
    direction: str = "long_short",
) -> pd.DataFrame:
    """Compute the cross-horizon TSMOM signal for a panel of prices.

    This is a **pure function**: it does not mutate ``prices`` and performs no
    I/O. The same inputs always yield the same output.

    Parameters
    ----------
    prices:
        Adjusted-close prices, indexed by tz-naive trading days with one column
        per asset (the data contract from
        :class:`trend_robot.data.provider.DataProvider`). Explicit ``NaN`` gaps
        are allowed and are propagated (never silently filled).
    lookbacks:
        Momentum lookback horizons in trading days (e.g. ``[21, 63, 126,
        252]``). Must be a non-empty list of positive integers.
    direction:
        ``"long_short"`` (default) keeps the signal in ``[-1, 1]``;
        ``"long_only"`` truncates negative signals to ``0`` via ``max(s, 0)``.

    Returns
    -------
    pandas.DataFrame
        Same index and columns as ``prices``. Values lie in ``[-1, 1]`` for
        ``"long_short"`` (or ``[0, 1]`` for ``"long_only"``). Dates/assets with
        insufficient history or an underlying price gap are ``NaN``.

    Raises
    ------
    TypeError
        If ``prices`` is not a :class:`pandas.DataFrame`.
    ValueError
        If ``lookbacks`` is empty or contains non-positive values, or if
        ``direction`` is not one of ``{"long_short", "long_only"}``.
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError(
            f"'prices' must be a pandas DataFrame, got {type(prices).__name__}."
        )
    if not lookbacks:
        raise ValueError("'lookbacks' must be a non-empty list of positive ints.")
    if any(int(lb) <= 0 for lb in lookbacks):
        raise ValueError(f"'lookbacks' must all be positive, got {lookbacks}.")
    if direction not in _VALID_DIRECTIONS:
        raise ValueError(
            f"'direction' must be one of {sorted(_VALID_DIRECTIONS)}, "
            f"got {direction!r}."
        )

    # Empty panel: return an empty, contract-shaped frame (no rows/cols).
    if prices.shape[1] == 0 or prices.shape[0] == 0:
        return prices.astype("float64").copy()

    # Work on float prices so integer-typed inputs still divide cleanly.
    px = prices.astype("float64")

    # Running sum of the per-horizon signs and a count of *defined* horizons.
    # A horizon L is defined at (t, asset) iff P_{t-L} exists and neither
    # P_t nor P_{t-L} is NaN. ``sign`` of a NaN ratio is NaN, which we detect.
    sign_sum = pd.DataFrame(
        0.0, index=px.index, columns=px.columns, dtype="float64"
    )
    defined_count = pd.DataFrame(
        0, index=px.index, columns=px.columns, dtype="int64"
    )

    n_lookbacks = len(lookbacks)
    for lb in lookbacks:
        lb = int(lb)
        # P_{t-L}: shift introduces NaN for the first L rows (no past price)
        # and never references any future price -> no look-ahead.
        ratio = px / px.shift(lb) - 1.0
        # sign(x) in {-1, 0, +1}; NaN where the ratio is undefined.
        s_l = np.sign(ratio)
        is_defined = s_l.notna()
        # Accumulate only the defined contributions (treat NaN as "absent").
        sign_sum = sign_sum.add(s_l.where(is_defined, 0.0))
        defined_count = defined_count.add(is_defined.astype("int64"))

    # Require ALL horizons to be defined at (t, asset); otherwise the mean
    # would be taken over a biased subset of horizons. Where fully defined,
    # the final signal is the simple mean of the per-horizon signs.
    fully_defined = defined_count == n_lookbacks
    signal = (sign_sum / float(n_lookbacks)).where(fully_defined, np.nan)

    if direction == "long_only":
        # Truncate negatives to 0, but preserve NaN (insufficient history).
        signal = signal.clip(lower=0.0)

    # Defensive clamp: the mean of signs is mathematically within [-1, 1],
    # but guard against any floating-point drift at the bounds.
    signal = signal.clip(lower=-1.0, upper=1.0)

    return signal
