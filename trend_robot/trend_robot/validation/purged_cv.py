"""Purged K-fold cross-validation with embargo (spec 6.3).

Naive K-fold CV is **invalid** for financial time series: an observation's
feature/label information window spans multiple bars (a TSMOM signal at ``t``
looks back over a lookback horizon; a forward label looks ahead). When a train
sample's information window overlaps the test fold, information leaks across the
split and the out-of-sample estimate is optimistically biased.

Lopez de Prado (2018, *Advances in Financial Machine Learning*, ch. 7) fixes
this with two operations applied to every fold:

* **Purging** -- drop from the *train* set any sample whose information window
  ``[i - horizon, i + horizon]`` overlaps the contiguous test fold
  ``[test_start, test_end]``. ``horizon`` captures the worst-case span over
  which a sample's features (lookback) and/or label (forward return) reach, so
  no leaked bar survives in the training data.
* **Embargo** -- additionally drop a block of ``round(cv_embargo * n)`` train
  samples *immediately after* each test fold. Serial correlation can leak
  information from the just-ended test fold into the bars right after it; the
  embargo blocks them out entirely.

The folds themselves are contiguous, time-ordered partitions of the index (no
shuffling), so each test fold is a single calendar block -- the only sensible
choice for a time series.

This module is pure: it derives index positions from the supplied length/index
and the configuration, performs no I/O and never mutates its inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trend_robot.config import Config

__all__ = ["PurgedKFold", "PurgedSplit", "purged_cv_splits"]


@dataclass(frozen=True)
class PurgedSplit:
    """A single purged/embargoed CV fold, in integer positions.

    Attributes
    ----------
    fold:
        Zero-based fold number.
    train_positions:
        Integer positions of the training samples *after* purging and embargo.
    test_positions:
        Integer positions of the contiguous test block.
    purged_positions:
        Train positions removed by purging (information-window overlap).
    embargoed_positions:
        Train positions removed by the embargo block after the test fold.
    """

    fold: int
    train_positions: np.ndarray
    test_positions: np.ndarray
    purged_positions: np.ndarray
    embargoed_positions: np.ndarray


def _contiguous_folds(n: int, n_splits: int) -> list[np.ndarray]:
    """Split ``range(n)`` into ``n_splits`` contiguous, near-equal blocks."""
    # np.array_split handles the non-divisible case (sizes differ by <= 1).
    return [block for block in np.array_split(np.arange(n), n_splits) if block.size]


class PurgedKFold:
    """PurgedKFold splitter with embargo (Lopez de Prado, 2018).

    Produces ``n_splits`` contiguous, time-ordered test folds. For each fold the
    training set is everything *outside* the fold, minus (a) samples whose
    information window overlaps the test block (**purging**) and (b) a block of
    samples immediately after the test block (**embargo**).

    Parameters
    ----------
    n_splits:
        Number of CV folds (``>= 2``).
    embargo:
        Embargo fraction in ``[0, 1)`` (typically ``cfg.cv_embargo``). The
        embargo block length is ``round(embargo * n_samples)``.
    horizon:
        Half-width (in bars) of each sample's information window. A sample at
        position ``i`` is considered to carry information over
        ``[i - horizon, i + horizon]``; if that interval intersects the test
        block the sample is purged. Set this to the largest relevant
        feature/label span (e.g. the maximum TSMOM lookback). Defaults to ``0``
        (purge only the bars directly adjacent through overlap, i.e. none beyond
        the fold itself).

    Raises
    ------
    ValueError
        If ``n_splits < 2``, ``embargo`` is outside ``[0, 1)`` or
        ``horizon < 0``.
    """

    def __init__(
        self,
        n_splits: int,
        embargo: float,
        horizon: int = 0,
    ) -> None:
        if n_splits < 2:
            raise ValueError(f"'n_splits' must be >= 2, got {n_splits}.")
        if not (0.0 <= embargo < 1.0):
            raise ValueError(
                f"'embargo' must be in the half-open interval [0, 1), "
                f"got {embargo}."
            )
        if horizon < 0:
            raise ValueError(f"'horizon' must be non-negative, got {horizon}.")
        self.n_splits = int(n_splits)
        self.embargo = float(embargo)
        self.horizon = int(horizon)

    def split(self, n_samples: int) -> list[PurgedSplit]:
        """Generate purged/embargoed folds over ``n_samples`` positions.

        Parameters
        ----------
        n_samples:
            Number of observations (the length of the time index).

        Returns
        -------
        list[PurgedSplit]
            One :class:`PurgedSplit` per fold, in chronological fold order.

        Raises
        ------
        ValueError
            If ``n_samples`` is too small to form ``n_splits`` folds.
        """
        if n_samples < self.n_splits:
            raise ValueError(
                f"need at least n_splits={self.n_splits} samples, "
                f"got {n_samples}."
            )

        all_positions = np.arange(n_samples)
        folds = _contiguous_folds(n_samples, self.n_splits)
        embargo_len = int(round(self.embargo * n_samples))

        splits: list[PurgedSplit] = []
        for fold, test_positions in enumerate(folds):
            test_start = int(test_positions[0])
            test_end = int(test_positions[-1])

            # Candidate train = everything not in the test fold.
            in_test = np.zeros(n_samples, dtype=bool)
            in_test[test_positions] = True
            candidate = all_positions[~in_test]

            # --- Purging: drop candidates whose information window
            #     [i - horizon, i + horizon] overlaps [test_start, test_end]. ---
            lo = candidate - self.horizon
            hi = candidate + self.horizon
            overlaps_test = (hi >= test_start) & (lo <= test_end)
            purged_positions = candidate[overlaps_test]

            # --- Embargo: drop a block of length embargo_len immediately
            #     after the test fold (only meaningful for samples after it). ---
            embargo_start = test_end + 1
            embargo_stop = test_end + embargo_len  # inclusive upper bound
            in_embargo = (candidate >= embargo_start) & (candidate <= embargo_stop)
            embargoed_positions = candidate[in_embargo]

            drop = overlaps_test | in_embargo
            train_positions = candidate[~drop]

            splits.append(
                PurgedSplit(
                    fold=fold,
                    train_positions=train_positions,
                    test_positions=np.asarray(test_positions),
                    purged_positions=purged_positions,
                    embargoed_positions=embargoed_positions,
                )
            )
        return splits


def purged_cv_splits(
    index_or_n: pd.Index | pd.Series | pd.DataFrame | int,
    cfg: Config,
    n_splits: int = 5,
    horizon: int | None = None,
) -> list[PurgedSplit]:
    """Purged K-fold CV folds with embargo, driven by :class:`Config`.

    Convenience wrapper around :class:`PurgedKFold` that reads the embargo from
    ``cfg.cv_embargo`` and, by default, sets the purge ``horizon`` to the
    largest configured TSMOM lookback (``max(cfg.lookbacks)``) -- the worst-case
    span over which a sample's signal information reaches, so every leaked bar
    is purged.

    Parameters
    ----------
    index_or_n:
        Either a time index / DataFrame / Series (its length is used) or an
        integer sample count.
    cfg:
        Typed configuration; ``cfg.cv_embargo`` sets the embargo and
        ``cfg.lookbacks`` sets the default horizon.
    n_splits:
        Number of folds (``>= 2``; default ``5``).
    horizon:
        Override for the purge horizon in bars. ``None`` (default) uses
        ``max(cfg.lookbacks)``.

    Returns
    -------
    list[PurgedSplit]
        Purged/embargoed folds in chronological order.
    """
    if isinstance(index_or_n, int):
        n_samples = index_or_n
    elif isinstance(index_or_n, (pd.DataFrame, pd.Series)):
        n_samples = len(index_or_n.index)
    elif isinstance(index_or_n, pd.Index):
        n_samples = len(index_or_n)
    else:
        raise TypeError(
            "expected an int or a pandas Index/Series/DataFrame, got "
            f"{type(index_or_n).__name__}."
        )

    h = max(cfg.lookbacks) if horizon is None else int(horizon)
    splitter = PurgedKFold(n_splits=n_splits, embargo=cfg.cv_embargo, horizon=h)
    return splitter.split(n_samples)
