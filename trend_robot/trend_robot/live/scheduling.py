"""Cadence + idempotence helpers for safe unattended scheduling.

A scheduled job (e.g. a daily launchd/cron task) may fire many times within a
single rebalance period. For a *monthly* strategy we must trade AT MOST ONCE per
calendar month no matter how often the scheduler wakes the runner. This module
provides the one pure primitive the runner needs to enforce that:

* :func:`period_key` collapses an ``asof`` date into a stable key identifying the
  rebalance *period* for the configured cadence (one key per day / ISO week /
  month). The runner records the period it last submitted for; if the saved
  state already shows a successful live submission for the current
  :func:`period_key`, it skips -- giving cadence gating and idempotence in a
  single check.

The cadence vocabulary (``daily`` / ``weekly`` / ``monthly``) mirrors
:data:`trend_robot.backtest.engine._CADENCE_FREQ` so the live cadence and the
backtested cadence stay in lock-step. Nothing here is hard-coded to a market
value; the period derives purely from the calendar date and the cadence string.
"""

from __future__ import annotations

import datetime as _dt

__all__ = ["period_key", "VALID_CADENCES"]

# Recognized cadences, kept in sync with the backtest engine's bucket map.
VALID_CADENCES: frozenset[str] = frozenset({"daily", "weekly", "monthly"})


def period_key(asof: str, cadence: str) -> str:
    """Return a stable key identifying the rebalance period of ``asof``.

    The key is the coarsest calendar bucket the cadence rebalances on, so two
    dates in the *same* period share a key (and the runner trades once per key):

    * ``"daily"``   -> ``"YYYY-MM-DD"`` (the date itself);
    * ``"weekly"``  -> ``"YYYY-Www"`` ISO week (e.g. ``"2026-W25"``), so any day
      Mon-Sun of the same ISO week maps to one key;
    * ``"monthly"`` -> ``"YYYY-MM"`` (the calendar month).

    Parameters
    ----------
    asof:
        As-of date in ISO ``"YYYY-MM-DD"`` form. A leading date component of a
        fuller ISO timestamp is also accepted (only the date is used).
    cadence:
        Rebalance cadence: ``"daily"``, ``"weekly"`` or ``"monthly"``.

    Returns
    -------
    str
        The period key for ``asof`` under ``cadence``.

    Raises
    ------
    ValueError
        If ``cadence`` is not a recognized cadence, or ``asof`` is not a valid
        ISO ``YYYY-MM-DD`` date.
    """
    if cadence not in VALID_CADENCES:
        raise ValueError(
            f"'cadence' must be one of {sorted(VALID_CADENCES)}, got {cadence!r}."
        )

    date_part = str(asof).strip()[:10]
    try:
        day = _dt.date.fromisoformat(date_part)
    except ValueError as exc:
        raise ValueError(
            f"'asof' must be an ISO 'YYYY-MM-DD' date, got {asof!r}."
        ) from exc

    if cadence == "daily":
        return day.isoformat()
    if cadence == "weekly":
        iso_year, iso_week, _ = day.isocalendar()
        return f"{iso_year:04d}-W{iso_week:02d}"
    # monthly
    return f"{day.year:04d}-{day.month:02d}"
