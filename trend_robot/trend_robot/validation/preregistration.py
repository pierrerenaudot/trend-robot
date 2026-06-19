"""Pre-registration of the frozen strategy decision (spec 6.5 / 11).

The section-6.5 ``RETAIN`` verdict produced on the locked test set is *not* a
clean out-of-sample read once a variant (here ``long_only`` / ``monthly``) has
been *selected* using exploration that peeked at all historical data -- the
locked test set included. After that peek, no past window is genuinely pristine:
the only truly untouched evidence is the FUTURE.

This module implements the **pre-registration** half of the forward-hold-out
protocol. It freezes, in a tamper-evident JSON record:

* the strategy-relevant subset of the typed :class:`~trend_robot.config.Config`
  that *defines* the chosen variant (the :func:`strategy_fingerprint`),
* a stable :func:`config_hash` of that fingerprint (drift detection),
* the **decision date** after which fresh bars count as out-of-sample, and
* the honest number of configurations already explored (``n_trials_spent``),
  which feeds the conservative multiple-testing hurdle in the forward read.

The forward hold-out (see :mod:`trend_robot.validation.holdout`) is defined as
bars whose index date is **strictly after** ``decision_date``.

This module hard-codes no market values: every fingerprinted field flows from
the typed :class:`Config`. It performs the minimal I/O required to persist and
reload the pre-registration record (pretty JSON on disk).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from trend_robot.config import Config

__all__ = [
    "DecisionRecord",
    "config_hash",
    "freeze_decision",
    "load_decision",
    "strategy_fingerprint",
    "verify_config_matches",
]

# The strategy-relevant fields of ``Config`` that *define* a variant. Two
# configs that agree on every one of these fields are, for the purposes of the
# forward hold-out, the *same* strategy -- operational knobs (initial_capital,
# split ratios, walk-forward window lengths, stress levels) are deliberately
# excluded because they do not change the traded book. The order here fixes the
# deterministic iteration order of :func:`strategy_fingerprint`.
_FINGERPRINT_FIELDS: tuple[str, ...] = (
    "universe",
    "direction",
    "rebalance",
    "lookbacks",
    "vol_window",
    "asset_vol_target",
    "portfolio_vol_target",
    "max_gross_leverage",
    "kelly_fraction",
    "cost_bps_per_side",
    "periods_per_year",
    "seed",
)


def strategy_fingerprint(cfg: Config) -> dict:
    """Return the strategy-relevant subset of ``cfg`` as an ordered dict.

    Only the fields that genuinely change the traded book are included (see
    :data:`_FINGERPRINT_FIELDS`); operational parameters such as
    ``initial_capital`` or the walk-forward window lengths are excluded so that
    re-running the *same* strategy over a different operational window does not
    look like a different variant.

    List-valued fields (``universe``, ``lookbacks``) are normalized to plain
    Python lists so the fingerprint is JSON-serializable and order-stable.

    Parameters
    ----------
    cfg:
        The validated, typed configuration to fingerprint.

    Returns
    -------
    dict
        Insertion-ordered mapping ``field -> value`` over the fingerprint
        fields, in the canonical order of :data:`_FINGERPRINT_FIELDS`.
    """
    fingerprint: dict = {}
    for name in _FINGERPRINT_FIELDS:
        value = getattr(cfg, name)
        if isinstance(value, (list, tuple)):
            value = list(value)
        fingerprint[name] = value
    return fingerprint


def config_hash(fingerprint: dict) -> str:
    """Stable SHA-256 hex digest of a strategy fingerprint.

    The fingerprint is canonicalized to JSON with ``sort_keys=True`` (so key
    ordering never affects the hash) and a compact separator, then hashed. The
    same fingerprint always yields the same digest; any change to a fingerprinted
    field changes the digest.

    Parameters
    ----------
    fingerprint:
        A mapping as returned by :func:`strategy_fingerprint` (must be
        JSON-serializable).

    Returns
    -------
    str
        The 64-character lowercase hexadecimal SHA-256 digest.
    """
    canonical = json.dumps(
        fingerprint, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DecisionRecord:
    """A tamper-evident, pre-registered strategy decision (spec 6.5 / 11).

    Freezing this record *before* any forward bars arrive is what makes the
    subsequent forward hold-out genuinely pristine: the chosen variant, the
    decision date and the honest exploration count are all committed up front,
    so the forward read cannot be retro-fitted to the data it later sees.

    Attributes
    ----------
    decision_date:
        ISO date (``"YYYY-MM-DD"``) of the freeze. The forward hold-out consists
        of bars whose index date is **strictly after** this date.
    config_fingerprint:
        The strategy-relevant subset of the chosen ``Config`` (see
        :func:`strategy_fingerprint`).
    config_hash:
        Stable SHA-256 of ``config_fingerprint`` (see :func:`config_hash`),
        used to detect config drift since freezing.
    n_trials_spent:
        Honest number of distinct configurations explored before this decision.
        Feeds the conservative ("carried") multiple-testing hurdle in the
        forward read; ``1`` means a single pre-registered test on fresh data.
    created_at:
        ISO-8601 UTC timestamp recording when the freeze was written.
    notes:
        Free-text human note (the rationale for the decision, caveats, etc.).
    """

    decision_date: str
    config_fingerprint: dict
    config_hash: str
    n_trials_spent: int
    created_at: str
    notes: str


class PreRegistrationError(RuntimeError):
    """Raised when a pre-registration record cannot be written or trusted."""


def _utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string (second precision)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def freeze_decision(
    cfg: Config,
    *,
    decision_date: str,
    n_trials_spent: int,
    notes: str,
    path: str | Path,
    overwrite: bool = False,
) -> DecisionRecord:
    """Build and persist a :class:`DecisionRecord` as pretty JSON.

    The record fingerprints the *current* ``cfg`` and stamps the decision date
    and honest exploration count. Pre-registration must be tamper-evident: if a
    record already exists at ``path`` whose ``config_hash`` **differs** from the
    one being written, this refuses to overwrite it and raises
    :class:`PreRegistrationError` unless ``overwrite=True`` is passed
    explicitly. Re-freezing an identical decision (same hash) is always allowed
    (it merely refreshes ``created_at``/``notes``).

    Parameters
    ----------
    cfg:
        The chosen, validated configuration to freeze.
    decision_date:
        ISO date (``"YYYY-MM-DD"``) after which bars count as the forward
        hold-out. Validated by attempting to parse it.
    n_trials_spent:
        Honest count of distinct configurations explored before this decision
        (``>= 1``).
    notes:
        Free-text rationale stored verbatim in the record.
    path:
        Destination file for the JSON record (parent dirs are created).
    overwrite:
        When ``True``, allow replacing an existing record whose hash differs.
        Defaults to ``False`` (tamper-evident).

    Returns
    -------
    DecisionRecord
        The record that was written.

    Raises
    ------
    PreRegistrationError
        If ``decision_date`` is not a valid ISO date, ``n_trials_spent < 1``,
        or an existing record with a different hash would be silently
        overwritten (and ``overwrite`` is ``False``).
    """
    # Validate the decision date eagerly so a typo fails at freeze time.
    try:
        datetime.strptime(decision_date, "%Y-%m-%d")
    except (TypeError, ValueError) as exc:
        raise PreRegistrationError(
            f"'decision_date' must be an ISO 'YYYY-MM-DD' date, "
            f"got {decision_date!r}."
        ) from exc

    if int(n_trials_spent) < 1:
        raise PreRegistrationError(
            f"'n_trials_spent' must be >= 1, got {n_trials_spent!r}."
        )

    fingerprint = strategy_fingerprint(cfg)
    digest = config_hash(fingerprint)

    out_path = Path(path)
    if out_path.is_file() and not overwrite:
        existing = load_decision(out_path)
        if existing.config_hash != digest:
            raise PreRegistrationError(
                "Refusing to overwrite an existing pre-registration record with "
                f"a DIFFERENT strategy hash at {out_path}.\n"
                f"  existing config_hash : {existing.config_hash}\n"
                f"  new config_hash      : {digest}\n"
                "Pre-registration is tamper-evident: pass overwrite=True only if "
                "you intend to deliberately re-register a new decision (which "
                "resets the pristine forward window)."
            )

    record = DecisionRecord(
        decision_date=decision_date,
        config_fingerprint=fingerprint,
        config_hash=digest,
        n_trials_spent=int(n_trials_spent),
        created_at=_utc_now_iso(),
        notes=str(notes),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(asdict(record), fh, indent=2, sort_keys=True)
        fh.write("\n")
    return record


def load_decision(path: str | Path) -> DecisionRecord:
    """Load a :class:`DecisionRecord` previously written by :func:`freeze_decision`.

    Parameters
    ----------
    path:
        Path to the JSON record on disk.

    Returns
    -------
    DecisionRecord
        The reconstructed record.

    Raises
    ------
    PreRegistrationError
        If the file is missing, not valid JSON, or missing required fields.
    """
    in_path = Path(path)
    if not in_path.is_file():
        raise PreRegistrationError(f"Decision record not found: {in_path}")

    try:
        with in_path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except json.JSONDecodeError as exc:
        raise PreRegistrationError(
            f"Decision record at {in_path} is not valid JSON: {exc}"
        ) from exc

    if not isinstance(raw, dict):
        raise PreRegistrationError(
            f"Decision record root must be a JSON object, got "
            f"{type(raw).__name__} at {in_path}."
        )

    required = {
        "decision_date",
        "config_fingerprint",
        "config_hash",
        "n_trials_spent",
        "created_at",
        "notes",
    }
    missing = required - set(raw)
    if missing:
        raise PreRegistrationError(
            f"Decision record at {in_path} is missing field(s): {sorted(missing)}."
        )

    return DecisionRecord(
        decision_date=str(raw["decision_date"]),
        config_fingerprint=dict(raw["config_fingerprint"]),
        config_hash=str(raw["config_hash"]),
        n_trials_spent=int(raw["n_trials_spent"]),
        created_at=str(raw["created_at"]),
        notes=str(raw["notes"]),
    )


def verify_config_matches(cfg: Config, record: DecisionRecord) -> bool:
    """Whether ``cfg`` still matches the frozen pre-registration.

    Recomputes the strategy hash of the *current* ``cfg`` and compares it to the
    hash stored in ``record``. A mismatch means the configuration has drifted
    since the decision was frozen (someone changed a fingerprinted field), which
    invalidates the pristine forward read.

    Parameters
    ----------
    cfg:
        The current configuration.
    record:
        The frozen :class:`DecisionRecord`.

    Returns
    -------
    bool
        ``True`` iff ``config_hash(strategy_fingerprint(cfg)) ==
        record.config_hash``.
    """
    return config_hash(strategy_fingerprint(cfg)) == record.config_hash
