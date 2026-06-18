"""Multiple-testing trials counter (spec 6.4).

Every distinct strategy *configuration* evaluated during research is a separate
"trial". The more configurations you test, the more likely one looks good by
luck alone -- so the significance bar must rise with the number of trials. The
Deflated Sharpe Ratio (:func:`trend_robot.metrics.deflated_sharpe.deflated_sharpe_ratio`)
encodes exactly this via its ``n_trials`` argument.

:class:`TrialCounter` is the bookkeeper: it counts the configurations evaluated
(optionally de-duplicating identical configs) and exposes :attr:`n_trials`,
which is wire-compatible with ``deflated_sharpe_ratio(..., n_trials=...)``.

This module is pure Python: no I/O, no market values.
"""

from __future__ import annotations

from typing import Any, Hashable

__all__ = ["TrialCounter"]


class TrialCounter:
    """Count strategy configurations evaluated for multiple-testing correction.

    Use one counter per research campaign. Call :meth:`record` (or
    :meth:`__call__`) once per configuration evaluated; read :attr:`n_trials`
    to feed the Deflated Sharpe Ratio. ``n_trials`` is clamped to a minimum of
    ``1`` so it is always a valid input to ``deflated_sharpe_ratio`` even before
    any trial is recorded (one trial implies no multiple-testing inflation).

    Parameters
    ----------
    deduplicate:
        If ``True`` (default), recording the *same* configuration key twice
        counts once -- only genuinely distinct configurations inflate the
        trials count. If ``False``, every :meth:`record` call increments.
    """

    def __init__(self, deduplicate: bool = True) -> None:
        self._deduplicate = bool(deduplicate)
        self._count = 0
        self._seen: set[Hashable] = set()

    @staticmethod
    def _key(config: Any) -> Hashable:
        """Return a stable, hashable key for a configuration object.

        Hashable configs (tuples, frozen dataclasses, strings, ...) are used
        directly; otherwise a sorted, repr-based key derived from a ``dict`` or
        the object's ``__dict__`` is used so that logically equal configs
        de-duplicate.
        """
        if isinstance(config, Hashable):
            try:
                hash(config)
                return config
            except TypeError:  # pragma: no cover - defensive
                pass
        if isinstance(config, dict):
            items = config
        elif hasattr(config, "__dict__"):
            items = vars(config)
        else:  # pragma: no cover - defensive fallback
            return repr(config)
        return tuple(sorted((str(k), repr(v)) for k, v in items.items()))

    def record(self, config: Any = None) -> int:
        """Record one evaluated configuration; return the running trials count.

        Parameters
        ----------
        config:
            The configuration evaluated. When ``deduplicate`` is enabled this is
            keyed (see :meth:`_key`) so repeats are not double-counted. May be
            ``None`` (always counts as a fresh trial when deduplicating, since
            ``None`` keys to itself and is recorded at most once).

        Returns
        -------
        int
            The updated :attr:`n_trials`.
        """
        if self._deduplicate:
            key = self._key(config)
            if key not in self._seen:
                self._seen.add(key)
                self._count += 1
        else:
            self._count += 1
        return self.n_trials

    def __call__(self, config: Any = None) -> int:
        """Alias for :meth:`record` so the counter is callable."""
        return self.record(config)

    def record_many(self, configs: object) -> int:
        """Record an iterable of configurations; return the trials count."""
        for config in configs:  # type: ignore[union-attr]
            self.record(config)
        return self.n_trials

    @property
    def n_trials(self) -> int:
        """Number of distinct trials, clamped to ``>= 1`` for DSR validity."""
        return max(1, self._count)

    @property
    def raw_count(self) -> int:
        """Unclamped count of recorded configurations (``>= 0``)."""
        return self._count

    def reset(self) -> None:
        """Reset the counter to zero (start a fresh campaign)."""
        self._count = 0
        self._seen.clear()

    def __int__(self) -> int:
        """Integer conversion yields :attr:`n_trials`."""
        return self.n_trials

    def __repr__(self) -> str:
        return (
            f"TrialCounter(n_trials={self.n_trials}, "
            f"raw_count={self._count}, deduplicate={self._deduplicate})"
        )
