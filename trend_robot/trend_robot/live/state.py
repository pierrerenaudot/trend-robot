"""Persist per-run live state to disk as JSON.

Each dry-run (or live run) writes a JSON record keyed by its ``asof`` date plus
a ``latest.json`` pointer, so a run is auditable and idempotency can be checked
(``has_run_for``). Records are made JSON-serializable up front (numpy scalars,
pandas Series and ``datetime`` are converted), so callers may hand in raw
computation outputs.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["save_run_state", "load_last_state", "has_run_for"]

_LATEST_NAME = "latest.json"


def _jsonable(obj: Any) -> Any:
    """Recursively convert ``obj`` into JSON-serializable primitives.

    Handles numpy scalars/arrays, pandas Series/Index/Timestamp, ``datetime``/
    ``date`` and ``Path``; recurses through dicts/lists/tuples. Anything else is
    returned as-is (and may raise later in ``json.dumps`` if truly unsupported).
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, pd.Series):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, pd.Index):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (pd.Timestamp, _dt.datetime, _dt.date)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def _state_filename(asof: str) -> str:
    """Return the per-run filename for an ``asof`` date."""
    return f"live_state_{asof}.json"


def save_run_state(state_dir: str | Path, record: dict) -> Path:
    """Persist ``record`` as JSON keyed by its ``asof`` and update ``latest``.

    The record is converted to JSON-serializable form, a ``generated_at`` UTC
    timestamp is added (if absent), and the file is written to
    ``<state_dir>/live_state_<asof>.json``. ``<state_dir>/latest.json`` is also
    (over)written with the same payload so the most recent run is easy to find.

    Parameters
    ----------
    state_dir:
        Directory under which state files are written (created if needed).
    record:
        The run record. Must contain an ``"asof"`` key used for the filename.

    Returns
    -------
    pathlib.Path
        Path to the per-run state file that was written.

    Raises
    ------
    ValueError
        If ``record`` has no ``"asof"`` key.
    """
    if "asof" not in record:
        raise ValueError("save_run_state requires record to contain an 'asof' key.")

    out_dir = Path(state_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = _jsonable(dict(record))
    payload.setdefault(
        "generated_at",
        _dt.datetime.now(_dt.timezone.utc).isoformat(),
    )

    asof = str(record["asof"])
    path = out_dir / _state_filename(asof)
    text = json.dumps(payload, indent=2, sort_keys=True)
    path.write_text(text, encoding="utf-8")
    (out_dir / _LATEST_NAME).write_text(text, encoding="utf-8")
    return path


def load_last_state(state_dir: str | Path) -> dict | None:
    """Load the most recent run record, or ``None`` if there is none.

    Prefers ``latest.json``; if that is missing, falls back to the
    lexicographically last ``live_state_*.json`` (ISO dates sort chronologically).

    Parameters
    ----------
    state_dir:
        Directory holding the state files.

    Returns
    -------
    dict | None
        The decoded record, or ``None`` when no readable state exists.
    """
    in_dir = Path(state_dir)
    if not in_dir.is_dir():
        return None

    latest = in_dir / _LATEST_NAME
    if latest.is_file():
        try:
            return json.loads(latest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass  # fall through to scanning per-run files

    candidates = sorted(in_dir.glob("live_state_*.json"))
    if not candidates:
        return None
    try:
        return json.loads(candidates[-1].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def has_run_for(state_dir: str | Path, asof: str) -> bool:
    """Return whether a per-run state file already exists for ``asof``.

    Parameters
    ----------
    state_dir:
        Directory holding the state files.
    asof:
        The ``"YYYY-MM-DD"`` date to check.

    Returns
    -------
    bool
        ``True`` if ``<state_dir>/live_state_<asof>.json`` exists.
    """
    return (Path(state_dir) / _state_filename(str(asof))).is_file()
