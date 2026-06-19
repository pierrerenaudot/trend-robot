"""Deflated Sharpe Ratio -- Bailey & Lopez de Prado (2014) (spec 3.5 / 7).

The observed (in-sample) Sharpe ratio of a *selected* strategy is upward
biased: it is the maximum over many trials, and it is inflated further when the
return distribution is non-normal (negative skew / fat tails make a high Sharpe
less trustworthy). The **Deflated Sharpe Ratio (DSR)** corrects for both:

1. **Multiple testing.** Out of ``n_trials`` independent attempts, the *expected
   maximum* Sharpe under the null (no skill) is strictly greater than zero and
   grows with ``n_trials``. We subtract that expected maximum as the benchmark
   the observed Sharpe must beat.
2. **Non-normality.** The sampling variance of the Sharpe estimator depends on
   the skew and kurtosis of returns (Mertens / Lo). DSR uses this corrected
   standard error in the denominator.

The DSR is the probability (a value in ``[0, 1]``) that the *true* Sharpe is
positive once both corrections are applied:

    DSR = Phi( (SR_obs - SR0) * sqrt(T - 1)
               / sqrt(1 - gamma3 * SR_obs + (gamma4 - 1)/4 * SR_obs^2) )

where ``Phi`` is the standard-normal CDF, ``T`` the number of return
observations, ``gamma3`` the skew, ``gamma4`` the kurtosis (non-excess), and
``SR0`` the multiple-testing benchmark

    SR0 = sqrt(Var_trials) * [ (1 - euler_gamma) * Phi^{-1}(1 - 1/N)
                               + euler_gamma * Phi^{-1}(1 - 1/(N*e)) ].

Here ``N = n_trials``, ``Var_trials`` is the cross-trial variance of the Sharpe
estimates and ``e`` is Euler's number. When the caller does not supply
``Var_trials``, :func:`deflated_sharpe_ratio` estimates it as the per-observation
sampling variance of the Sharpe estimator, ``var_term / (T - 1)`` (Mertens/Lo),
so ``SR0`` lands on the same per-bar scale as the observed Sharpe. ``SR0``
increases with ``N``, so for a fixed observed Sharpe the DSR is monotonically
*decreasing* in ``n_trials``.

NOTE: ``SR0`` must be compared to a *per-observation* Sharpe. Scaling it by
``var_trials = 1.0`` (a plausible-looking default) makes ``SR0`` ~1.2 -- orders
of magnitude larger than a typical per-bar Sharpe (~0.05) -- which floors the DSR
at exactly 0 for any ``n_trials > 1``. The estimated per-bar variance above
avoids this.

All Sharpe inputs/benchmarks are in *per-observation* (non-annualized) units,
which is the convention of the paper; the caller is responsible for passing a
non-annualized Sharpe consistent with ``len(returns)``.

This module is pure: no I/O, no hard-coded market values.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm

__all__ = ["deflated_sharpe_ratio", "expected_max_sharpe", "observed_sharpe"]

# Euler-Mascheroni constant, used in the expected-maximum-of-Gaussians formula.
_EULER_GAMMA: float = 0.5772156649015329


def observed_sharpe(returns: pd.Series) -> float:
    """Per-observation (non-annualized) Sharpe ratio of a return series.

    Computed as ``mean(returns) / std(returns)`` with the sample standard
    deviation (``ddof=1``) and a zero risk-free rate. This is the unit in which
    :func:`deflated_sharpe_ratio` expects ``SR_obs`` to be expressed.

    Parameters
    ----------
    returns:
        Per-bar strategy returns.

    Returns
    -------
    float
        Non-annualized Sharpe, or ``nan`` if undefined (too few points or zero
        dispersion).
    """
    r = pd.Series(returns, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    if r.size < 2:
        return float("nan")
    sd = r.std(ddof=1)
    if sd == 0.0 or not np.isfinite(sd):
        return float("nan")
    return float(r.mean() / sd)


def expected_max_sharpe(n_trials: int, var_trials: float = 1.0) -> float:
    """Expected maximum Sharpe under the null across ``n_trials`` trials.

    Implements the Bailey & Lopez de Prado approximation to the expected value
    of the maximum of ``n_trials`` independent standard-normal Sharpe estimates,
    scaled by the cross-trial Sharpe standard deviation:

        SR0 = sqrt(var_trials) * [ (1 - g) * z(1 - 1/N)
                                   + g * z(1 - 1/(N*e)) ]

    with ``g`` the Euler-Mascheroni constant, ``z`` the standard-normal inverse
    CDF and ``N = n_trials``. The bracketed term is strictly increasing in
    ``N``, so ``SR0`` (the hurdle the observed Sharpe must clear) grows with the
    number of trials.

    Parameters
    ----------
    n_trials:
        Number of independent strategy configurations tested (``>= 1``). With
        ``n_trials == 1`` there is no multiple-testing inflation and ``SR0`` is
        ``0``.
    var_trials:
        Variance of the Sharpe estimates *across* trials. Defaults to ``1.0``
        (the standardized convention used when only one configuration's returns
        are available). Must be non-negative.

    Returns
    -------
    float
        The expected-maximum Sharpe benchmark ``SR0`` (per-observation units).

    Raises
    ------
    ValueError
        If ``n_trials < 1`` or ``var_trials < 0``.
    """
    if n_trials < 1:
        raise ValueError(f"'n_trials' must be >= 1, got {n_trials}.")
    if var_trials < 0.0:
        raise ValueError(f"'var_trials' must be non-negative, got {var_trials}.")

    if n_trials == 1 or var_trials == 0.0:
        return 0.0

    n = float(n_trials)
    sqrt_var = math.sqrt(var_trials)
    z1 = norm.ppf(1.0 - 1.0 / n)
    z2 = norm.ppf(1.0 - 1.0 / (n * math.e))
    return float(sqrt_var * ((1.0 - _EULER_GAMMA) * z1 + _EULER_GAMMA * z2))


def deflated_sharpe_ratio(
    returns: pd.Series,
    n_trials: int,
    skew: float,
    kurtosis: float,
    var_trials: float | None = None,
) -> float:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

    Returns the probability that the strategy's *true* Sharpe ratio is positive
    after deflating the observed Sharpe for (a) the number of trials and (b)
    non-normality of returns. A value near ``1`` is strong evidence of genuine
    skill; values near ``0.5`` or below mean the observed Sharpe is consistent
    with luck under multiple testing.

    Formula
    -------
    With ``SR`` the observed (per-observation) Sharpe, ``SR0`` the expected
    maximum Sharpe across ``n_trials`` (see :func:`expected_max_sharpe`), ``T``
    the number of return observations, ``g3`` the skew and ``g4`` the
    (non-excess) kurtosis:

        DSR = Phi( (SR - SR0) * sqrt(T - 1)
                   / sqrt(1 - g3 * SR + (g4 - 1)/4 * SR^2) )

    The numerator's hurdle ``SR0`` grows with ``n_trials``, so for fixed inputs
    **DSR decreases monotonically as ``n_trials`` increases**.

    Parameters
    ----------
    returns:
        Per-bar strategy returns. Used to derive the observed Sharpe and the
        sample length ``T``. Non-finite entries are dropped.
    n_trials:
        Number of strategy configurations tested (the multiple-testing count,
        ``>= 1``). Larger values impose a higher hurdle and a lower DSR.
    skew:
        Skewness of the returns (third standardized moment). Negative skew
        (frequent small gains, rare large losses) lowers the DSR.
    kurtosis:
        Kurtosis of the returns (non-excess; a normal distribution has ``3``).
        Heavier tails (``> 3``) lower the DSR.
    var_trials:
        Cross-trial variance of the (per-observation) Sharpe estimates. When
        ``None`` (default) it is estimated as ``var_term / (T - 1)`` so the
        multiple-testing hurdle ``SR0`` is on the same per-bar scale as the
        observed Sharpe. Pass an explicit value only if you measured it across
        real trials.

    Returns
    -------
    float
        Deflated Sharpe Ratio in ``[0, 1]``, or ``nan`` if the observed Sharpe
        is undefined or the corrected variance term is non-positive.

    Raises
    ------
    ValueError
        If ``n_trials < 1`` or ``var_trials < 0``.

    Notes
    -----
    Pure function: no I/O, no mutation of inputs. The Sharpe is taken in
    per-observation units to match the paper; annualization must not be applied
    before calling.
    """
    if n_trials < 1:
        raise ValueError(f"'n_trials' must be >= 1, got {n_trials}.")
    if var_trials is not None and var_trials < 0.0:
        raise ValueError(f"'var_trials' must be non-negative, got {var_trials}.")

    r = pd.Series(returns, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    n_obs = int(r.size)
    if n_obs < 2:
        return float("nan")

    sr = observed_sharpe(r)
    if not np.isfinite(sr):
        return float("nan")

    # Mertens/Lo corrected variance of the Sharpe estimator (the term under the
    # square root). A non-positive term means the correction is ill-defined for
    # the given moments and Sharpe; report nan rather than a complex result.
    var_term = 1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * (sr**2)
    if var_term <= 0.0 or not np.isfinite(var_term):
        return float("nan")

    # The expected-maximum hurdle SR0 must be on the SAME (per-observation) scale
    # as ``sr``. When the caller does not supply the cross-trial Sharpe variance,
    # estimate it by the per-observation sampling variance of the Sharpe
    # estimator, ``var_term / (T - 1)`` (Mertens/Lo) -- which under the
    # multiple-testing null approximates that cross-trial variance. Using the old
    # ``1.0`` default would put SR0 (~1.2) on a wildly different scale from a
    # per-bar Sharpe (~0.05) and floor the DSR at 0 for any n_trials > 1.
    if var_trials is None:
        var_trials = var_term / (n_obs - 1)

    sr0 = expected_max_sharpe(n_trials, var_trials=var_trials)

    denom = math.sqrt(var_term)
    z = (sr - sr0) * math.sqrt(n_obs - 1) / denom
    return float(norm.cdf(z))
