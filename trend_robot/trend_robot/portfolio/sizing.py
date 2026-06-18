"""Volatility-targeted position sizing (spec 3.3).

Pure, side-effect-free transformation of TSMOM signals into portfolio target
weights. The public entry point :func:`target_weights` implements the five-step
sizing recipe from the spec:

1. **Ex-ante per-asset volatility** -- an EWMA estimate (centre-of-mass
   ``cfg.vol_window``) of daily-return volatility, annualized by
   ``sqrt(cfg.periods_per_year)``.
2. **Raw vol-targeted weight** -- ``w_i = s_i * (asset_vol_target / sigma_i)``,
   so each asset contributes (in isolation) roughly ``asset_vol_target`` of
   annualized risk, scaled by its momentum conviction ``s_i in [-1, 1]``.
3. **Portfolio-level vol scaling** -- estimate the portfolio's ex-ante
   annualized volatility ``sigma_p`` from the raw weights and an EWMA return
   covariance, then scale every weight by ``k = portfolio_vol_target / sigma_p``
   so the *aggregate* book targets ``portfolio_vol_target``.
4. **Fractional Kelly** -- multiply by ``cfg.kelly_fraction``.
5. **Gross-leverage cap** -- if ``sum_i |w_i| > cfg.max_gross_leverage`` on a
   date, renormalize that date's weights so the gross exposure equals the cap;
   otherwise leave them untouched.

Zero weight is assigned wherever the signal is ``0``/``NaN`` or the ex-ante
volatility is undefined (insufficient history / a zero-variance series), so an
asset with no usable estimate never takes risk.

No look-ahead
-------------
The estimate at date ``t`` uses only returns/signals observed up to and
including ``t`` (EWMA is causal; no future-looking centering or smoothing).
Execution-side lagging of the resulting weights before trading is the backtest
engine's responsibility, not this function's.

This is a **pure function**: it never mutates its inputs and performs no I/O.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trend_robot.config import Config

__all__ = ["target_weights"]


def _ewma_asset_vol(returns: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Annualized ex-ante per-asset volatility via causal EWMA.

    Uses :meth:`pandas.DataFrame.ewm` with ``com=cfg.vol_window`` on daily
    returns and annualizes the resulting daily standard deviation by
    ``sqrt(cfg.periods_per_year)``. The EWMA is strictly backward-looking, so no
    future information enters the estimate at any date (look-ahead-safe).

    Parameters
    ----------
    returns:
        Daily simple returns, indexed by trading day, one column per asset.
    cfg:
        Typed configuration supplying ``vol_window`` and ``periods_per_year``.

    Returns
    -------
    pandas.DataFrame
        Same shape as ``returns``. Annualized volatility per (date, asset);
        ``NaN`` where the estimate is undefined (insufficient history).
    """
    # min_periods=2 so a single observation does not yield a (degenerate) 0 vol.
    daily_vol = returns.ewm(com=float(cfg.vol_window), min_periods=2).std()
    return daily_vol * np.sqrt(float(cfg.periods_per_year))


def _ewma_cov_at(
    returns: pd.DataFrame, cfg: Config, valid_cols: pd.Index, asof_idx: int
) -> pd.DataFrame | None:
    """Annualized EWMA return covariance as-of one date, for ``valid_cols``.

    Computes the exponentially-weighted covariance of the (causally truncated)
    return history up to and including ``asof_idx`` and annualizes it by
    ``cfg.periods_per_year``. Only the assets in ``valid_cols`` are retained
    (those with a defined ex-ante vol *and* a non-zero raw weight on this date).

    Parameters
    ----------
    returns:
        Daily simple returns (full panel).
    cfg:
        Typed configuration (``vol_window``, ``periods_per_year``).
    valid_cols:
        Columns to include in the covariance.
    asof_idx:
        Positional index (row) of the as-of date; only rows ``[0, asof_idx]``
        are used, guaranteeing no look-ahead.

    Returns
    -------
    pandas.DataFrame | None
        Annualized covariance matrix over ``valid_cols``, or ``None`` if it
        cannot be estimated (too little history / all-NaN window).
    """
    if len(valid_cols) == 0:
        return None
    hist = returns.iloc[: asof_idx + 1].loc[:, valid_cols]
    if len(hist) < 2:
        return None
    # Pairwise EWMA covariance as-of the last available row.
    ewm_cov = hist.ewm(com=float(cfg.vol_window), min_periods=2).cov(pairwise=True)
    last_date = hist.index[-1]
    try:
        cov = ewm_cov.loc[last_date]
    except KeyError:  # pragma: no cover - defensive
        return None
    cov = cov.reindex(index=valid_cols, columns=valid_cols)
    if cov.isna().to_numpy().any():
        return None
    return cov * float(cfg.periods_per_year)


def _portfolio_vol(raw_w: pd.Series, cov: pd.DataFrame) -> float:
    """Annualized ex-ante portfolio vol ``sqrt(w' Sigma w)`` for one date.

    Parameters
    ----------
    raw_w:
        Raw (pre-scaling) weights for the date, aligned to ``cov``'s columns.
    cov:
        Annualized return covariance matrix over the same assets.

    Returns
    -------
    float
        Non-negative annualized portfolio volatility (``0.0`` if degenerate).
    """
    w = raw_w.reindex(cov.columns).to_numpy(dtype="float64")
    variance = float(w @ cov.to_numpy(dtype="float64") @ w)
    if not np.isfinite(variance) or variance <= 0.0:
        return 0.0
    return float(np.sqrt(variance))


def target_weights(
    signals: pd.DataFrame,
    returns: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """Convert TSMOM signals into vol-targeted portfolio weights (spec 3.3).

    This is a **pure function**: ``signals`` and ``returns`` are never mutated
    and no I/O is performed; identical inputs always produce identical output.

    The five sizing steps (see module docstring) are applied date-by-date:

    1. ex-ante per-asset vol ``sigma_i`` (EWMA, annualized);
    2. raw weight ``w_i = s_i * (asset_vol_target / sigma_i)``;
    3. portfolio-vol scaling ``k = portfolio_vol_target / sigma_p`` with
       ``sigma_p = sqrt(w' Sigma w)`` from an EWMA covariance;
    4. fractional Kelly (``* kelly_fraction``);
    5. gross-leverage cap (renormalize a date iff ``sum|w_i| > max_gross``).

    Parameters
    ----------
    signals:
        Per-asset/date momentum signal in ``[-1, 1]`` (from
        :func:`trend_robot.signals.tsmom.tsmom_signal`). ``NaN``/``0`` -> no
        position. Indexed by trading day, one column per asset.
    returns:
        Daily simple returns of the same assets used for ex-ante risk
        estimation. Need not share ``signals``' index exactly; it is reindexed
        onto the union and aligned by column. Explicit ``NaN`` gaps are honored.
    cfg:
        Typed configuration (``vol_window``, ``periods_per_year``,
        ``asset_vol_target``, ``portfolio_vol_target``, ``kelly_fraction``,
        ``max_gross_leverage``).

    Returns
    -------
    pandas.DataFrame
        Target weights indexed exactly like ``signals`` with the same columns.
        Each row satisfies ``sum_i |w_i| <= cfg.max_gross_leverage``. Assets
        with a zero/NaN signal or undefined vol receive a weight of ``0.0``.

    Raises
    ------
    TypeError
        If ``signals`` or ``returns`` is not a :class:`pandas.DataFrame`.
    """
    if not isinstance(signals, pd.DataFrame):
        raise TypeError(
            f"'signals' must be a pandas DataFrame, got {type(signals).__name__}."
        )
    if not isinstance(returns, pd.DataFrame):
        raise TypeError(
            f"'returns' must be a pandas DataFrame, got {type(returns).__name__}."
        )

    out_index = signals.index
    out_columns = signals.columns

    # Empty panel -> contract-shaped empty (float) frame.
    if signals.shape[0] == 0 or signals.shape[1] == 0:
        return signals.astype("float64").copy()

    # --- Align returns onto the signal grid (columns + index). -------------
    # Reindexing onto signals' columns guards against column-order mismatches;
    # reindexing onto signals' index keeps the output index intact. We do NOT
    # forward-fill: explicit gaps stay NaN (honoring the data contract).
    ret = returns.reindex(columns=out_columns)
    ret = ret.reindex(index=out_index)
    ret = ret.astype("float64")

    sig = signals.astype("float64")

    # --- Step 1: ex-ante per-asset annualized vol (EWMA). ------------------
    sigma = _ewma_asset_vol(ret, cfg)

    # --- Step 2: raw vol-targeted weight w_i = s_i * (asset_target/sigma_i).
    # A position is only taken where the signal is finite & non-zero AND the
    # vol estimate is finite & strictly positive; everything else -> 0.
    sig_ok = sig.notna() & (sig != 0.0)
    vol_ok = sigma.notna() & (sigma > 0.0)
    usable = sig_ok & vol_ok

    safe_sigma = sigma.where(vol_ok)  # NaN where vol unusable -> 0 below
    raw = sig * (float(cfg.asset_vol_target) / safe_sigma)
    raw = raw.where(usable, 0.0)

    # --- Steps 3-5 are per-date (covariance + cap depend on the cross-section).
    weights = pd.DataFrame(
        0.0, index=out_index, columns=out_columns, dtype="float64"
    )
    kelly = float(cfg.kelly_fraction)
    port_target = float(cfg.portfolio_vol_target)
    max_gross = float(cfg.max_gross_leverage)

    for pos, date in enumerate(out_index):
        raw_row = raw.loc[date]
        active = raw_row.index[raw_row.to_numpy() != 0.0]
        if len(active) == 0:
            continue  # row already all-zero

        # --- Step 3: portfolio-vol scaling factor k. -----------------------
        cov = _ewma_cov_at(ret, cfg, active, pos)
        if cov is None:
            # No reliable covariance yet -> cannot scale to portfolio target.
            # Fall back to the unit factor so per-asset targeting still holds;
            # the gross cap below remains the binding risk control.
            k = 1.0
        else:
            sigma_p = _portfolio_vol(raw_row.loc[active], cov)
            k = (port_target / sigma_p) if sigma_p > 0.0 else 1.0

        # --- Steps 3 (apply) + 4: scale by k and fractional Kelly. ---------
        w_row = raw_row * (k * kelly)

        # --- Step 5: gross-leverage cap (renormalize iff exceeded). --------
        gross = float(w_row.abs().sum())
        if gross > max_gross and gross > 0.0:
            w_row = w_row * (max_gross / gross)

        weights.loc[date] = w_row

    return weights
