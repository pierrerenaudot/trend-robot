"""Locked train/test and walk-forward splits (spec section 6.1 / 6.2).

This module implements the *time-ordered* split machinery that underpins the
validation protocol. There is no shuffling and no leakage of future bars into
the past: every split respects the arrow of time.

Two protocols are provided:

* :func:`train_test_split` -- the **locked** development/test split (spec 6.1).
  The *last* ``1 - cfg.train_test_ratio`` fraction of the history is carved off
  as the untouched out-of-sample (OOS) test set; the leading
  ``cfg.train_test_ratio`` fraction is the development set on which all tuning
  happens. The test set must remain untouched until the very end.

* :func:`walk_forward_splits` -- the **rolling** walk-forward generator (spec
  6.2). Fixed-length training (``wf_train_years``) and testing
  (``wf_test_years``) windows are rolled forward by ``wf_step_years`` at a time;
  the per-window test segments are non-overlapping and, when concatenated (see
  :func:`concat_test_segments`), form one contiguous "as-in-production" OOS
  track.

Years are converted to a number of bars via ``cfg.periods_per_year`` so the
windows are expressed in the same trading-bar units as the price index. No
market values are hard-coded: every length flows from the typed :class:`Config`.

This module is pure: it derives everything from the supplied time index and the
configuration, performs no I/O and never mutates its inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trend_robot.config import Config

__all__ = [
    "WalkForwardWindow",
    "concat_test_segments",
    "train_test_split",
    "walk_forward_splits",
]


def _as_index(index_or_df: pd.Index | pd.Series | pd.DataFrame) -> pd.Index:
    """Coerce a frame/series/index into a plain :class:`pandas.Index`.

    Accepts a :class:`pandas.DataFrame` or :class:`pandas.Series` (its ``.index``
    is used) or an :class:`pandas.Index` (returned as-is). This lets callers pass
    either a price panel or its date index interchangeably.

    Parameters
    ----------
    index_or_df:
        A DataFrame/Series (whose index is taken) or an Index.

    Returns
    -------
    pandas.Index
        The time index to split on.

    Raises
    ------
    TypeError
        If ``index_or_df`` is none of the accepted types.
    """
    if isinstance(index_or_df, (pd.DataFrame, pd.Series)):
        return index_or_df.index
    if isinstance(index_or_df, pd.Index):
        return index_or_df
    raise TypeError(
        "expected a pandas Index/Series/DataFrame, got "
        f"{type(index_or_df).__name__}."
    )


def train_test_split(
    index_or_df: pd.Index | pd.Series | pd.DataFrame,
    cfg: Config,
) -> tuple[pd.Index, pd.Index]:
    """Locked, time-ordered train/test split (spec 6.1).

    The *last* ``1 - cfg.train_test_ratio`` fraction of the (time-ordered) index
    is returned as the out-of-sample **test** set; the leading
    ``cfg.train_test_ratio`` fraction is the **train** (development) set. The
    boundary bar count is ``floor(n * train_test_ratio)`` so that, by
    construction, ``len(train) / n`` never exceeds ``train_test_ratio`` and the
    two slices are contiguous, disjoint and exhaustive.

    Parameters
    ----------
    index_or_df:
        The time index to split (or a DataFrame/Series whose index is used).
        Assumed already in ascending chronological order.
    cfg:
        Typed configuration; ``cfg.train_test_ratio`` governs the boundary.

    Returns
    -------
    tuple[pandas.Index, pandas.Index]
        ``(train_index, test_index)`` -- contiguous, non-overlapping slices that
        together cover the whole input index.

    Notes
    -----
    With ``n == 0`` both slices are empty. The split is purely positional, so it
    works for any monotonic index (dates, ints, ...).
    """
    index = _as_index(index_or_df)
    n = len(index)
    if n == 0:
        return index[:0], index[:0]

    boundary = int(n * cfg.train_test_ratio)
    # Guard the degenerate extremes so each side is well-defined when feasible.
    boundary = max(0, min(boundary, n))
    train_index = index[:boundary]
    test_index = index[boundary:]
    return train_index, test_index


@dataclass(frozen=True)
class WalkForwardWindow:
    """A single rolling walk-forward window (spec 6.2).

    Attributes
    ----------
    fold:
        Zero-based window number in the rolling sequence.
    train_index:
        Contiguous training bars for the window.
    test_index:
        Contiguous testing bars immediately following the train window. These
        are non-overlapping across folds and concatenate into the production
        OOS track (see :func:`concat_test_segments`).
    """

    fold: int
    train_index: pd.Index
    test_index: pd.Index


def _years_to_bars(years: int, periods_per_year: int) -> int:
    """Convert a span in years to a whole number of trading bars."""
    return int(round(years * periods_per_year))


def walk_forward_splits(
    index_or_df: pd.Index | pd.Series | pd.DataFrame,
    cfg: Config,
    *,
    train_bars: int | None = None,
    test_bars: int | None = None,
    step_bars: int | None = None,
) -> list[WalkForwardWindow]:
    """Generate rolling walk-forward train/test windows (spec 6.2).

    Train windows are ``wf_train_years`` long, test windows ``wf_test_years``
    long, and the pair is rolled forward by ``wf_step_years`` each iteration --
    all converted to bars via ``cfg.periods_per_year``. Each test window starts
    exactly where its train window ends (no gap, no overlap with the train), and
    successive test windows are stepped by ``step`` bars. With
    ``wf_step_years == wf_test_years`` (the default 1/1) the test windows tile
    the post-warmup history contiguously and without overlap.

    Only *complete* windows are emitted: a partial trailing test window (shorter
    than ``test`` bars) at the end of the history is dropped, so every returned
    window has full train/test lengths.

    The window lengths may be overridden **explicitly in bars** via the keyword
    arguments ``train_bars`` / ``test_bars`` / ``step_bars``. Each override that
    is not ``None`` replaces the corresponding ``cfg.wf_*_years *
    periods_per_year`` length; overrides left as ``None`` keep the config-derived
    length. With all three ``None`` (the default) the behaviour is exactly the
    config-driven one above, so existing callers are unaffected. This lets a
    short forward / hold-out slice be rolled with short windows (e.g. 6-month
    train / 3-month test) without changing the locked-test ``cfg`` defaults.

    Parameters
    ----------
    index_or_df:
        Time index to roll over (or a DataFrame/Series whose index is used),
        assumed ascending.
    cfg:
        Typed configuration providing ``wf_train_years`` / ``wf_test_years`` /
        ``wf_step_years`` and ``periods_per_year``.
    train_bars, test_bars, step_bars:
        Optional explicit window lengths in **bars**. When provided (not
        ``None``) they override the config-derived length for that window;
        otherwise the ``cfg.wf_*_years * periods_per_year`` length is used.

    Returns
    -------
    list[WalkForwardWindow]
        Rolling windows in chronological order (possibly empty if the history is
        shorter than one ``train + test`` block).
    """
    index = _as_index(index_or_df)
    n = len(index)

    train = (
        int(train_bars)
        if train_bars is not None
        else _years_to_bars(cfg.wf_train_years, cfg.periods_per_year)
    )
    test = (
        int(test_bars)
        if test_bars is not None
        else _years_to_bars(cfg.wf_test_years, cfg.periods_per_year)
    )
    step = (
        int(step_bars)
        if step_bars is not None
        else _years_to_bars(cfg.wf_step_years, cfg.periods_per_year)
    )
    step = max(step, 1)  # never stall

    windows: list[WalkForwardWindow] = []
    if n < train + test or train <= 0 or test <= 0:
        return windows

    fold = 0
    train_start = 0
    while True:
        train_end = train_start + train
        test_end = train_end + test
        if test_end > n:
            break
        windows.append(
            WalkForwardWindow(
                fold=fold,
                train_index=index[train_start:train_end],
                test_index=index[train_end:test_end],
            )
        )
        fold += 1
        train_start += step

    return windows


def concat_test_segments(
    windows: list[WalkForwardWindow],
) -> pd.Index:
    """Concatenate per-window test segments into one OOS track (spec 6.2).

    Stitches the test slices of a walk-forward run into a single contiguous,
    chronologically ordered, **non-overlapping** out-of-sample index -- the
    "as-in-production" track on which aggregate walk-forward performance is
    measured. If consecutive test windows overlap (e.g. ``step < test``), the
    overlap is de-duplicated so each bar appears at most once.

    Parameters
    ----------
    windows:
        Walk-forward windows (typically from :func:`walk_forward_splits`).

    Returns
    -------
    pandas.Index
        The union of all test segments, sorted and de-duplicated. Empty if
        ``windows`` is empty.
    """
    if not windows:
        return pd.Index([])

    combined = windows[0].test_index
    for window in windows[1:]:
        combined = combined.union(window.test_index)
    return combined.sort_values()
