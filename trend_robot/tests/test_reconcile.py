"""Tests for reconciliation + alerting + the data-integrity gate.

Systematizes the July-2026 incident: the paper account was reset after the
monthly submission was recorded, leaving a flat book that the idempotence guard
then (correctly) refused to re-trade -- silently. These tests pin down:

* the pure reconciliation math (flat-vs-invested, drift tolerance);
* the escalation path in ``run_live`` (skip + off-target book => alert fired,
  ``exit_code`` 2, anomaly persisted in state);
* the data-integrity gate (live submits refuse synthetic/stale data; dry-runs
  only warn; the explicit test-only bypass works);
* ``send_alert`` never raises and honors the env-var configuration.

Deterministic / offline ONLY (synthetic data, mock broker, no network).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

import run_live
from tests.test_live_submit import (  # reuse the existing mock plumbing
    MockTradingClient,
    _DECISION_RECORD,
    _live_argv,
    _patch_alpaca_broker,
)
from trend_robot.live import alerts
from trend_robot.live.broker import Position
from trend_robot.live.reconcile import (
    format_reconcile_report,
    reconcile_book,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _pos(symbol: str, qty: float, price: float) -> Position:
    return Position(
        symbol=symbol, qty=qty, avg_price=price, market_value=qty * price
    )


# ---------------------------------------------------------------------------
# reconcile_book -- pure math
# ---------------------------------------------------------------------------
def test_reconcile_flat_book_vs_invested_targets_is_anomaly() -> None:
    """The July signature: invested targets, empty book => anomaly flags."""
    target = pd.Series({"SPY": 0.30, "TLT": 0.30, "GLD": 0.10})
    last_px = pd.Series({"SPY": 500.0, "TLT": 90.0, "GLD": 180.0})
    report = reconcile_book(target, {}, last_px, 100_000.0)

    assert report.current_gross == 0.0
    assert report.target_gross == pytest.approx(0.70)
    assert report.l1_deviation == pytest.approx(0.70)
    assert report.book_flat_but_target_invested is True
    assert report.materially_off_target is True
    assert report.anomaly is True


def test_reconcile_drifted_book_within_tolerance_is_ok() -> None:
    """A normally-drifted book (few % L1) reconciles clean."""
    equity = 100_000.0
    target = pd.Series({"SPY": 0.30, "TLT": 0.30})
    last_px = pd.Series({"SPY": 500.0, "TLT": 90.0})
    positions = {
        # ~0.315 and ~0.27 realized weights: 4.5% L1 drift total.
        "SPY": _pos("SPY", 63.0, 500.0),
        "TLT": _pos("TLT", 300.0, 90.0),
    }
    report = reconcile_book(target, positions, last_px, equity)

    assert report.l1_deviation == pytest.approx(0.045, abs=1e-9)
    assert report.anomaly is False
    assert report.book_flat_but_target_invested is False


def test_reconcile_untargeted_holding_counts_toward_gap() -> None:
    """A symbol held but absent from the target contributes |gap| fully."""
    equity = 10_000.0
    target = pd.Series({"SPY": 0.50})
    last_px = pd.Series({"SPY": 100.0, "XYZ": 10.0})
    positions = {
        "SPY": _pos("SPY", 50.0, 100.0),   # exactly on target (0.50)
        "XYZ": _pos("XYZ", 400.0, 10.0),   # 0.40 rogue position
    }
    report = reconcile_book(target, positions, last_px, equity)

    assert report.deviations["XYZ"] == pytest.approx(-0.40)
    assert report.l1_deviation == pytest.approx(0.40)
    assert report.materially_off_target is True  # > 0.25 default tolerance


def test_reconcile_empty_target_and_flat_book_is_clean() -> None:
    """Nothing targeted, nothing held: trivially reconciled (all-cash month)."""
    report = reconcile_book(pd.Series(dtype="float64"), {}, pd.Series(dtype="float64"), 5_000.0)
    assert report.anomaly is False
    assert report.l1_deviation == 0.0


def test_reconcile_bad_tolerance_raises() -> None:
    with pytest.raises(ValueError, match="tolerance"):
        reconcile_book(pd.Series({"SPY": 0.5}), {}, pd.Series({"SPY": 1.0}), 1.0, tolerance=0.0)


def test_format_reconcile_report_mentions_flat_book() -> None:
    target = pd.Series({"SPY": 0.5})
    report = reconcile_book(target, {}, pd.Series({"SPY": 100.0}), 1_000.0)
    text = format_reconcile_report(report)
    assert "ANOMALY" in text
    assert "FLAT" in text
    assert "SPY" in text


# ---------------------------------------------------------------------------
# send_alert -- never raises, env-configured
# ---------------------------------------------------------------------------
def test_send_alert_no_url_is_silent_noop(monkeypatch) -> None:
    monkeypatch.delenv(alerts.ALERT_WEBHOOK_ENV, raising=False)
    assert alerts.send_alert("hello") is False


def test_send_alert_posts_json_payload(monkeypatch) -> None:
    captured: dict = {}

    class _Resp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def _fake_urlopen(request, timeout=0):
        captured["url"] = request.full_url
        captured["body"] = request.data.decode("utf-8")
        captured["timeout"] = timeout
        return _Resp()

    monkeypatch.setattr(alerts.urllib.request, "urlopen", _fake_urlopen)
    ok = alerts.send_alert("boom", webhook_url="https://hooks.example/x")
    assert ok is True
    assert captured["url"] == "https://hooks.example/x"
    assert '"text": "boom"' in captured["body"]
    assert '"content": "boom"' in captured["body"]


def test_send_alert_network_failure_never_raises(monkeypatch) -> None:
    def _boom(request, timeout=0):
        raise OSError("network down")

    monkeypatch.setattr(alerts.urllib.request, "urlopen", _boom)
    assert alerts.send_alert("x", webhook_url="https://hooks.example/x") is False


def test_send_alert_reads_env_var(monkeypatch) -> None:
    calls: list[str] = []

    class _Resp:
        status = 204
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setenv(alerts.ALERT_WEBHOOK_ENV, "https://hooks.example/env")
    monkeypatch.setattr(
        alerts.urllib.request, "urlopen",
        lambda req, timeout=0: calls.append(req.full_url) or _Resp(),
    )
    assert alerts.send_alert("via env") is True
    assert calls == ["https://hooks.example/env"]


# ---------------------------------------------------------------------------
# run_live escalation: skip + off-target book => alert + exit_code 2
# ---------------------------------------------------------------------------
def test_skip_run_with_vanished_book_escalates(tmp_path, monkeypatch) -> None:
    """July-incident replay: submitted marker + flat book => alert + exit 2."""
    state_dir = tmp_path / "state"
    mock = MockTradingClient(equity=100_000.0, is_open=True)
    _patch_alpaca_broker(monkeypatch, mock)

    alerted: list[str] = []
    monkeypatch.setattr(run_live, "send_alert", lambda msg, **kw: alerted.append(msg) or True)

    # First run establishes the submitted-period marker (orders "fill" at the
    # mock, but the mock's positions stay empty -- exactly a vanished book).
    first = run_live.main(_live_argv(state_dir, asof="2021-06-15"))
    assert first["submitted"] > 0
    assert first["exit_code"] == 0  # a submitting run never escalates

    # Second run, same period: skip path + flat book vs invested target.
    second = run_live.main(_live_argv(state_dir, asof="2021-06-28"))
    assert second["submitted"] == 0
    assert second["skipped_reason"] is not None
    assert second["reconcile"]["anomaly"] is True
    assert second["reconcile"]["book_flat_but_target_invested"] is True
    assert second["reconcile"]["escalated"] is True
    assert second["exit_code"] == 2
    assert len(alerted) == 1 and "RECONCILIATION ANOMALY" in alerted[0]


def test_skip_run_with_book_near_target_stays_green(tmp_path, monkeypatch) -> None:
    """Skip path with a book near target: no anomaly, exit 0, no alert."""
    state_dir = tmp_path / "state"
    mock = MockTradingClient(equity=100_000.0, is_open=True)
    _patch_alpaca_broker(monkeypatch, mock)

    alerted: list[str] = []
    monkeypatch.setattr(run_live, "send_alert", lambda msg, **kw: alerted.append(msg) or True)

    first = run_live.main(_live_argv(state_dir, asof="2021-06-15"))
    assert first["submitted"] > 0

    # Rebuild the mock book to match the just-submitted targets, using each
    # order's est_price so reconciliation (qty * last_price / equity) lands on
    # the target weights up to a few days of drift.
    weights = first["target_weights"]
    est_px = {o["symbol"]: o["est_price"] for o in first["orders"]}
    mock._positions = {
        sym: (w * 100_000.0) / est_px[sym]
        for sym, w in weights.items()
        if abs(w) > 1e-9 and sym in est_px
    }

    second = run_live.main(_live_argv(state_dir, asof="2021-06-28"))
    assert second["submitted"] == 0
    # Drift between the 15th and 28th stays well under the 0.25 tolerance.
    assert second["reconcile"]["escalated"] is False
    assert second["exit_code"] == 0
    assert alerted == []


# ---------------------------------------------------------------------------
# Data-integrity gate
# ---------------------------------------------------------------------------
def test_data_gate_refuses_synthetic_live_submit(tmp_path, monkeypatch) -> None:
    """A live submit on synthetic data (no bypass) is REFUSED + alerted."""
    state_dir = tmp_path / "state"
    mock = MockTradingClient(equity=100_000.0, is_open=True)
    _patch_alpaca_broker(monkeypatch, mock)

    alerted: list[str] = []
    monkeypatch.setattr(run_live, "send_alert", lambda msg, **kw: alerted.append(msg) or True)

    argv = [a for a in _live_argv(state_dir, asof="2021-06-15")
            if a != "--allow-synthetic-live"]
    with pytest.raises(RuntimeError, match="DATA-INTEGRITY"):
        run_live.main(argv)
    assert mock.submitted == []           # nothing reached the broker
    assert len(alerted) == 1 and "REFUSED" in alerted[0]


def test_data_gate_dry_run_only_warns(tmp_path, caplog) -> None:
    """Synthetic data on a DRY-RUN warns but completes (offline preview)."""
    state_dir = tmp_path / "state"
    with caplog.at_level(logging.WARNING, logger="trend_robot.run_live"):
        record = run_live.main([
            "--dry-run",
            "--asof", "2021-06-15",
            "--state-dir", str(state_dir),
            "--cache-dir", str(state_dir / "cache"),
            "--decision", str(_DECISION_RECORD),
            "--history-years", "6",
            "--log-level", "WARNING",
        ])
    assert record["submitted"] == 0
    assert record["exit_code"] == 0
    assert any("DATA-INTEGRITY" in r.message for r in caplog.records)


def test_data_gate_refuses_stale_prices(tmp_path, monkeypatch) -> None:
    """A live submit on a stale panel (last bar too old vs asof) is refused."""
    state_dir = tmp_path / "state"
    mock = MockTradingClient(equity=100_000.0, is_open=True)
    _patch_alpaca_broker(monkeypatch, mock)
    monkeypatch.setattr(run_live, "send_alert", lambda msg, **kw: True)

    # Freeze the panel 30 days before asof: stale beyond the 7-day default.
    real_prices_asof = run_live.prices_asof

    def _stale(cfg, asof, cache_dir, **kw):
        prices, _src = real_prices_asof(cfg, "2021-05-16", cache_dir, **kw)
        return prices, "yfinance"  # pretend real source; staleness must catch it

    monkeypatch.setattr(run_live, "prices_asof", _stale)

    argv = _live_argv(state_dir, asof="2021-06-15")
    with pytest.raises(RuntimeError, match="stale"):
        run_live.main(argv)
    assert mock.submitted == []
