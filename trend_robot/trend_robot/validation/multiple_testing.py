"""Advanced multiple-testing tests (spec section 6.4).

When a researcher screens *many* candidate strategies against a benchmark and
keeps the best one, the best in-sample performance is upward biased: with enough
candidates *something* will look good by luck alone (data snooping). This module
implements two formal tests that control for that bias and return a single
honest p-value for the null hypothesis

    H0:  none of the K candidates beats the benchmark
         (i.e. max_k E[d_k] <= 0)

against the alternative that *at least one* genuinely does.

Tests implemented
-----------------
* :func:`whites_reality_check` -- White's Reality Check (White, 2000).
  Statistic ``V = max_k sqrt(T) * mean_k`` over the K candidates; the null
  distribution is obtained by a stationary bootstrap of the *recentred*
  performance series.
* :func:`hansens_spa` -- Hansen's Superior Predictive Ability test
  (Hansen, 2005), the studentized refinement of the Reality Check. The
  statistic ``T_SPA = max_k sqrt(T) * mean_k / std_k`` standardizes each
  candidate, and the *consistent* recentring drops candidates whose mean is too
  far below zero (relative to their own sampling noise) from the null
  re-centring -- which sharpens power versus White's conservative test.

Both rely on the **stationary bootstrap** of Politis & Romano (1994)
(:func:`stationary_bootstrap_indices`): blocks of geometrically distributed
random length are resampled with replacement and wrapped around the end of the
series, preserving short-range time dependence (autocorrelation /
volatility clustering) that an i.i.d. bootstrap would destroy.

Conventions
-----------
``perf`` is a ``(T, K)`` matrix of per-period performance statistics of K
candidates *relative to the benchmark*: entry ``perf[t, k]`` is the period-``t``
out-performance of candidate ``k`` (e.g. an excess return, or a loss
differential ``d_{k,t} = L(benchmark) - L(model_k)``) where **positive means the
candidate beat the benchmark**. The tests look for evidence that the column with
the largest mean has a *positive* population mean.

References
----------
White, H. (2000). "A Reality Check for Data Snooping." *Econometrica*, 68(5),
1097-1126.
Hansen, P. R. (2005). "A Test for Superior Predictive Ability." *Journal of
Business & Economic Statistics*, 23(4), 365-380.
Politis, D. N., & Romano, J. P. (1994). "The Stationary Bootstrap." *Journal of
the American Statistical Association*, 89(428), 1303-1313.

This module is pure: no I/O, no network, no hard-coded market values. Every
function is deterministic given its seed / RNG.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "hansens_spa",
    "stationary_bootstrap_indices",
    "whites_reality_check",
]


# ---------------------------------------------------------------------------
# Stationary bootstrap (Politis & Romano, 1994)
# ---------------------------------------------------------------------------
def stationary_bootstrap_indices(
    n: int,
    avg_block: float,
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Resampled time indices for the stationary bootstrap (Politis & Romano, 1994).

    Builds ``n_boot`` bootstrap samples, each of length ``n``, by laying down
    consecutive blocks of the original ``0..n-1`` index. Each block starts at a
    uniformly random position and has a *geometrically distributed* random
    length with mean ``avg_block`` (success probability ``p = 1 / avg_block``);
    indices wrap around modulo ``n`` so every block is well defined. The
    expected block length controls how much serial dependence is preserved: a
    longer ``avg_block`` keeps more autocorrelation, ``avg_block = 1`` reduces to
    the i.i.d. bootstrap.

    The construction is equivalent to the standard per-step formulation: at each
    step, with probability ``p`` jump to a fresh uniform start, otherwise
    advance one position from the previous index (wrapping). That per-step view
    gives exactly geometric block lengths with mean ``1 / p = avg_block``.

    Parameters
    ----------
    n:
        Length of the original series (number of time periods). Must be ``>= 1``.
    avg_block:
        Mean block length (``>= 1``). The geometric success probability is
        ``p = 1 / avg_block`` (clamped to ``(0, 1]``).
    n_boot:
        Number of bootstrap replications to draw. Must be ``>= 1``.
    rng:
        A seeded :class:`numpy.random.Generator`. Determinism is the caller's
        responsibility: the same ``rng`` state yields the same indices.

    Returns
    -------
    numpy.ndarray
        Integer array of shape ``(n_boot, n)`` whose rows are resampled time
        indices in ``range(n)``.

    Raises
    ------
    ValueError
        If ``n < 1``, ``n_boot < 1`` or ``avg_block < 1``.

    Notes
    -----
    Vectorized per-step implementation: a Bernoulli ``restart`` mask and a fresh
    uniform start for each position are drawn up front, then a single Python
    loop over the ``n`` columns fills the matrix (advance-or-restart). This is
    O(n_boot * n) and fully deterministic given ``rng``.
    """
    if n < 1:
        raise ValueError(f"'n' must be >= 1, got {n}.")
    if n_boot < 1:
        raise ValueError(f"'n_boot' must be >= 1, got {n_boot}.")
    if avg_block < 1.0:
        raise ValueError(f"'avg_block' must be >= 1, got {avg_block}.")

    p = min(1.0, 1.0 / float(avg_block))

    indices = np.empty((n_boot, n), dtype=np.int64)
    # Fresh uniform starts for every (replication, step): used whenever a new
    # block begins. Drawing the full matrix keeps the draw count -- and hence
    # the RNG stream consumption -- independent of the data, so determinism is
    # purely a function of (n, n_boot, p, seed).
    starts = rng.integers(0, n, size=(n_boot, n), dtype=np.int64)
    # Bernoulli(p) restart decisions for every step after the first.
    restart = rng.random((n_boot, n)) < p

    # First column always starts a fresh block.
    indices[:, 0] = starts[:, 0]
    for t in range(1, n):
        advanced = (indices[:, t - 1] + 1) % n
        indices[:, t] = np.where(restart[:, t], starts[:, t], advanced)
    return indices


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _as_matrix(perf: np.ndarray | pd.DataFrame) -> np.ndarray:
    """Coerce ``perf`` to a finite ``(T, K)`` float64 matrix.

    Accepts a 2-D array / DataFrame, or a 1-D array (treated as a single
    candidate column). Raises on empty input or non-finite entries.
    """
    if isinstance(perf, pd.DataFrame):
        arr = perf.to_numpy(dtype="float64")
    else:
        arr = np.asarray(perf, dtype="float64")
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"'perf' must be 1-D or 2-D, got {arr.ndim} dims.")
    if arr.shape[0] < 2 or arr.shape[1] < 1:
        raise ValueError(
            f"'perf' must be at least (2, 1), got shape {arr.shape}."
        )
    if not np.isfinite(arr).all():
        raise ValueError("'perf' contains non-finite values (NaN/inf).")
    return arr


def _column_std(perf: np.ndarray) -> np.ndarray:
    """Per-column standard error scale ``std_k`` for the studentized statistic.

    Uses the simple per-column sample standard deviation of ``sqrt(T)*mean``,
    i.e. ``std(perf_k, ddof=1)`` (so ``sqrt(T)*mean_k / std_k`` is the usual
    one-sample t-style statistic). Degenerate (zero / non-finite) scales are
    floored to a tiny positive number so the studentized statistic stays finite;
    such a column contributes no real signal.
    """
    sd = perf.std(axis=0, ddof=1)
    sd = np.where(np.isfinite(sd) & (sd > 0.0), sd, np.inf)
    return sd


# ---------------------------------------------------------------------------
# White's Reality Check (2000)
# ---------------------------------------------------------------------------
def whites_reality_check(
    perf: np.ndarray | pd.DataFrame,
    *,
    avg_block: float = 10.0,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, float | int]:
    """White's Reality Check for data snooping (White, 2000).

    Tests ``H0: max_k E[d_k] <= 0`` (no candidate beats the benchmark) against
    the alternative that the best candidate genuinely out-performs, controlling
    for the fact that the maximum was *selected* over K candidates.

    Statistic
    ---------
    With ``d_k`` the period performance of candidate ``k`` (positive = beats
    benchmark) and ``T`` periods,

        V = max_k  sqrt(T) * mean_k .

    Bootstrap null
    --------------
    The stationary bootstrap (:func:`stationary_bootstrap_indices`) resamples
    the *recentred* series ``d_k - mean_k`` so each column has mean zero (White's
    re-centring imposes the least-favourable point of the null). For each
    bootstrap replication ``b`` the maximum studentless statistic

        V*_b = max_k sqrt(T) * mean( d_k[idx_b] - mean_k )

    is recorded. The p-value is the right-tail mass:

        p = ( #{ b : V*_b >= V } ) / n_boot .

    A small p-value means the observed best performance is too large to be
    explained by luck across the K candidates.

    Parameters
    ----------
    perf:
        ``(T, K)`` matrix (array or DataFrame) of per-period performance of K
        candidates relative to the benchmark; positive entries mean the
        candidate beat the benchmark. A 1-D input is treated as a single column.
    avg_block:
        Mean block length for the stationary bootstrap (default ``10``). Should
        reflect the serial dependence of ``perf``.
    n_boot:
        Number of bootstrap replications (default ``2000``).
    seed:
        Seed for the internal :class:`numpy.random.Generator`; makes the result
        deterministic.

    Returns
    -------
    dict
        ``{"statistic": V, "p_value": p, "k_best": argmax_k mean_k}``.
        ``k_best`` is the 0-based index of the best candidate.

    References
    ----------
    White, H. (2000). "A Reality Check for Data Snooping." *Econometrica*,
    68(5), 1097-1126.
    """
    arr = _as_matrix(perf)
    t_obs, _ = arr.shape
    sqrt_t = np.sqrt(t_obs)

    col_mean = arr.mean(axis=0)
    stat_per_k = sqrt_t * col_mean
    v_obs = float(stat_per_k.max())
    k_best = int(np.argmax(col_mean))

    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(t_obs, avg_block, n_boot, rng)

    # Recentred series (column means removed): the null re-centring of White.
    recentred = arr - col_mean  # broadcast over rows
    # Bootstrap means: (n_boot, K) = mean over resampled rows of recentred cols.
    boot_means = recentred[idx].mean(axis=1)  # idx -> (n_boot, T, K) -> mean ax1
    boot_max = (sqrt_t * boot_means).max(axis=1)  # (n_boot,)

    p_value = float(np.mean(boot_max >= v_obs))
    return {"statistic": v_obs, "p_value": p_value, "k_best": k_best}


# ---------------------------------------------------------------------------
# Hansen's SPA test (2005)
# ---------------------------------------------------------------------------
def hansens_spa(
    perf: np.ndarray | pd.DataFrame,
    *,
    avg_block: float = 10.0,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, float | int]:
    """Hansen's test for Superior Predictive Ability (Hansen, 2005).

    The studentized refinement of White's Reality Check. Each candidate is
    standardized by its own sampling-noise scale, and the null re-centring is
    *consistent*: candidates that perform far worse than the benchmark (relative
    to their noise) are excluded from the re-centring rather than pulled up to
    zero. This removes the "irrelevant bad models inflate the threshold" defect
    of White's test and yields a less conservative (higher-power) p-value.

    Statistic
    ---------
    With per-column scale ``omega_k`` (here the sample std of ``d_k``) and ``T``
    periods,

        T_SPA = max( 0,  max_k sqrt(T) * mean_k / omega_k ) .

    Consistent re-centring
    ----------------------
    Each bootstrapped column mean is recentred by

        g_k = mean_k * 1{ sqrt(T) * mean_k / omega_k  >=  -sqrt(2 log log T) } ,

    i.e. a candidate keeps its sample mean as the re-centring point only if it is
    not *significantly* worse than the benchmark (the ``A_n = sqrt(2 log log T)``
    threshold of Hansen, 2005, eq. for the consistent estimator ``\\hat{mu}^c``).
    Columns failing the threshold are recentred to ``0`` and so cannot lower the
    bootstrap maximum -- they are simply dropped from the data-snooping pool. The
    studentized bootstrap statistic is

        T*_b = max( 0, max_k sqrt(T) * mean( d_k[idx_b] - g_k ) / omega_k ) ,

    and the consistent p-value is ``p_c = #{ b : T*_b >= T_SPA } / n_boot``.

    Parameters
    ----------
    perf:
        ``(T, K)`` matrix (array or DataFrame); positive entries mean the
        candidate beat the benchmark. A 1-D input is treated as one column.
    avg_block:
        Mean block length for the stationary bootstrap (default ``10``).
    n_boot:
        Number of bootstrap replications (default ``2000``).
    seed:
        Seed for the internal :class:`numpy.random.Generator`.

    Returns
    -------
    dict
        ``{"statistic": T_SPA, "p_value": p_c, "p_value_lower": p_l,
        "p_value_upper": p_u, "k_best": argmax_k (sqrt(T)*mean_k/omega_k)}``.
        ``p_value`` is the *consistent* p-value (the headline number);
        ``p_value_lower`` / ``p_value_upper`` are Hansen's bracketing variants.
        They differ only in *which* columns keep their sample mean as the
        re-centring point: the more columns recentred (pulled to zero mean), the
        higher the bootstrap threshold and the larger the p-value.

        * ``p_value_upper`` recentres **every** column (White's full
          re-centring) -- the most conservative, largest p-value;
        * ``p_value`` (consistent) recentres only columns with
          ``sqrt(T)*mean_k/omega_k >= -A_n`` (not significantly inferior);
        * ``p_value_lower`` recentres only columns with a strictly positive mean
          -- the most liberal, smallest p-value.

        They bracket the consistent value:
        ``p_lower <= p_consistent <= p_upper``.

    References
    ----------
    Hansen, P. R. (2005). "A Test for Superior Predictive Ability." *Journal of
    Business & Economic Statistics*, 23(4), 365-380.
    """
    arr = _as_matrix(perf)
    t_obs, _ = arr.shape
    sqrt_t = np.sqrt(t_obs)

    col_mean = arr.mean(axis=0)
    omega = _column_std(arr)  # per-column scale; +inf for degenerate columns
    t_per_k = sqrt_t * col_mean / omega
    t_spa = float(max(0.0, t_per_k.max()))
    k_best = int(np.argmax(t_per_k))

    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(t_obs, avg_block, n_boot, rng)
    boot_raw_means = arr[idx].mean(axis=1)  # (n_boot, K) means of resampled cols

    # Hansen's consistent threshold A_n = sqrt(2 log log T) (T >= 3 so positive).
    a_n = np.sqrt(2.0 * np.log(np.log(max(t_obs, 3))))
    keep_consistent = t_per_k >= -a_n  # candidates not significantly inferior

    def _p_value(recenter: np.ndarray) -> float:
        # boot statistic per replication for given per-column recentring vector.
        centered = boot_raw_means - recenter  # (n_boot, K)
        boot_t = (sqrt_t * centered / omega).max(axis=1)
        boot_t = np.maximum(0.0, boot_t)
        return float(np.mean(boot_t >= t_spa))

    # Each variant only ever recentres a column *by its own sample mean* (or
    # leaves it untouched, i.e. recentre by 0). Recentring a column removes its
    # mean -> raises the bootstrap maximum -> raises the p-value. Order of how
    # many columns are recentred: lower (positive-mean only) subset of consistent
    # (>= -A_n) subset of upper (all). Hence p_lower <= p_consistent <= p_upper.
    #
    # Consistent: recentre columns that are not significantly inferior.
    recenter_c = np.where(keep_consistent, col_mean, 0.0)
    # Lower (most liberal, smallest p): recentre only strictly-positive-mean cols.
    recenter_l = np.where(t_per_k > 0.0, col_mean, 0.0)
    # Upper (most conservative = White, largest p): recentre every column.
    recenter_u = col_mean

    return {
        "statistic": t_spa,
        "p_value": _p_value(recenter_c),
        "p_value_lower": _p_value(recenter_l),
        "p_value_upper": _p_value(recenter_u),
        "k_best": k_best,
    }
