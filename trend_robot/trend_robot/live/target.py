"""Compute today's target book (portfolio weights) for the live runner.

This is the live-side mirror of the research pipeline's weight computation, but
it returns only the *single most recent* row of target weights -- "today's
book". It reuses the exact same pure transforms (:func:`tsmom_signal` ->
:func:`target_weights`) so the live target is bit-identical to what the backtest
would target on the same date.

Anti-look-ahead
---------------
:func:`compute_target_book` selects the last row whose index is ``<= asof`` and
the underlying signal/sizing transforms are causal (EWMA, ``shift``). The book
is therefore computed from closes up to and including ``asof`` only; the
resulting orders are what you place *after* ``asof`` (next session). This
function is PURE -- it performs no I/O and never mutates its inputs.
"""

from __future__ import annotations

import pandas as pd

from trend_robot.config import Config
from trend_robot.portfolio.sizing import target_weights
from trend_robot.signals.tsmom import tsmom_signal

__all__ = ["compute_target_book"]


def compute_target_book(
    cfg: Config,
    prices: pd.DataFrame,
    asof: str | None = None,
) -> pd.Series:
    """Return today's target weights (one row) as a Series indexed by symbol.

    Steps (identical to the research pipeline, restricted to one date):

    1. ``returns = prices.pct_change(fill_method=None)`` (explicit NaN gaps);
    2. ``signals = tsmom_signal(prices, cfg.lookbacks, direction=cfg.direction)``;
    3. ``weights = target_weights(signals, returns, cfg)``;
    4. take the LAST row with index ``<= asof`` (or the very last row if
       ``asof`` is ``None``) and return it as a Series.

    Parameters
    ----------
    cfg:
        Typed configuration (lookbacks, direction, sizing parameters).
    prices:
        Adjusted-close panel (rows=dates, cols=symbols). Only rows up to ``asof``
        influence the returned book.
    asof:
        Inclusive "as of" date (``"YYYY-MM-DD"``). When ``None``, the last row of
        ``prices`` is used.

    Returns
    -------
    pandas.Series
        Today's target weights indexed by symbol. Empty Series if ``prices`` has
        no rows at/before ``asof``.

    Raises
    ------
    ValueError
        If ``prices`` has rows but none fall at/before ``asof``.
    """
    if prices.shape[0] == 0 or prices.shape[1] == 0:
        return pd.Series(dtype="float64", index=prices.columns)

    returns = prices.pct_change(fill_method=None)
    signals = tsmom_signal(prices, cfg.lookbacks, direction=cfg.direction)
    weights = target_weights(signals, returns, cfg)

    if asof is None:
        row = weights.iloc[-1]
        return row.astype("float64")

    asof_ts = pd.Timestamp(asof)
    eligible = weights.index[weights.index <= asof_ts]
    if len(eligible) == 0:
        raise ValueError(
            f"No price rows at or before asof={asof!r}; cannot compute target "
            f"book (earliest available row is {weights.index[0]!r})."
        )
    row = weights.loc[eligible[-1]]
    return row.astype("float64")
