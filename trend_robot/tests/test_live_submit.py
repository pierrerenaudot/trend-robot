"""Tests for the HARDENED Alpaca paper submit path (milestone 2).

Every test here is OFFLINE and uses a MOCK Alpaca client injected into
:class:`~trend_robot.live.broker.AlpacaBroker` -- NEVER the real
``TradingClient`` and NEVER the network or any credentials. The real
connectivity test is done by the user at home.

Coverage
--------
* MockTradingClient maps cleanly through AlpacaBroker
  (account/positions/submit_order) and ``is_market_open`` reads the clock.
* ``run_live.main`` on the LIVE path (mock broker injected via ``_build_broker``)
  submits the intents, records :class:`OrderResult`\\ s and a ``submitted_period``
  in state, with ``submitted > 0``.
* Idempotence/cadence: a second ``main`` for the SAME period does NOT resubmit;
  ``--force`` overrides.
* Config-drift guard: a mismatch REFUSES a live submit but only WARNS on a
  dry-run.
* ``--status`` submits nothing.
* :func:`period_key` keys for daily/weekly/monthly; unknown cadence raises.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import run_live
from trend_robot.live.broker import AlpacaBroker, OrderIntent
from trend_robot.live.scheduling import period_key
from trend_robot.live.state import load_last_state


# ---------------------------------------------------------------------------
# Mock Alpaca client (plain classes -- no network, no creds, no alpaca-py)
# ---------------------------------------------------------------------------
class _Acct:
    """Canned Alpaca account object (string fields, as the real API returns)."""

    def __init__(self, equity: str, cash: str, buying_power: str) -> None:
        self.equity = equity
        self.cash = cash
        self.buying_power = buying_power


class _Pos:
    """Canned Alpaca position object."""

    def __init__(self, symbol, qty, avg_entry_price, market_value) -> None:
        self.symbol = symbol
        self.qty = qty
        self.avg_entry_price = avg_entry_price
        self.market_value = market_value


class _Clock:
    """Canned Alpaca clock object."""

    def __init__(self, is_open: bool) -> None:
        self.is_open = is_open


class _Order:
    """Canned Alpaca order object returned by submit_order/get_orders."""

    def __init__(self, symbol, qty, side, status, id) -> None:  # noqa: A002
        self.symbol = symbol
        self.qty = qty
        self.side = side
        self.status = status
        self.id = id


class MockTradingClient:
    """A drop-in stand-in for ``alpaca.trading.client.TradingClient``.

    Returns canned account/positions/clock objects and records every submitted
    order (so tests can assert on them). No network, no credentials.
    """

    def __init__(
        self,
        *,
        equity: float = 100_000.0,
        positions: dict[str, float] | None = None,
        is_open: bool = True,
    ) -> None:
        self._equity = float(equity)
        self._positions = positions or {}
        self._is_open = bool(is_open)
        self.submitted: list = []
        self._next_id = 1

    def get_account(self) -> _Acct:
        return _Acct(
            equity=str(self._equity),
            cash=str(self._equity),
            buying_power=str(self._equity * 2.0),
        )

    def get_all_positions(self) -> list[_Pos]:
        return [
            _Pos(symbol=sym, qty=str(qty), avg_entry_price="100.0",
                 market_value=str(float(qty) * 100.0))
            for sym, qty in self._positions.items()
        ]

    def get_clock(self) -> _Clock:
        return _Clock(self._is_open)

    def submit_order(self, order_data) -> _Order:
        order = _Order(
            symbol=order_data.symbol,
            qty=str(order_data.qty),
            side=str(order_data.side),
            status="accepted",
            id=f"mock-{self._next_id}",
        )
        self._next_id += 1
        self.submitted.append(order)
        return order

    def get_orders(self, filter=None):  # noqa: A002 - mirror alpaca-py signature
        return list(self.submitted)


# ---------------------------------------------------------------------------
# period_key
# ---------------------------------------------------------------------------
def test_period_key_daily_weekly_monthly() -> None:
    """period_key returns the right bucket per cadence."""
    assert period_key("2026-06-19", "daily") == "2026-06-19"
    assert period_key("2026-06-19", "weekly") == "2026-W25"
    assert period_key("2026-06-19", "monthly") == "2026-06"


def test_period_key_same_bucket_collapses() -> None:
    """Two dates in the same month/week share a key (one trade per period)."""
    assert period_key("2026-06-01", "monthly") == period_key("2026-06-30", "monthly")
    # 2026-06-15 (Mon) .. 2026-06-21 (Sun) is one ISO week.
    assert period_key("2026-06-15", "weekly") == period_key("2026-06-21", "weekly")
    assert period_key("2026-06-19", "daily") != period_key("2026-06-20", "daily")


def test_period_key_unknown_cadence_raises() -> None:
    """An unrecognized cadence raises ValueError."""
    with pytest.raises(ValueError, match="cadence"):
        period_key("2026-06-19", "yearly")


def test_period_key_bad_date_raises() -> None:
    """A non-ISO date raises ValueError."""
    with pytest.raises(ValueError, match="ISO"):
        period_key("not-a-date", "monthly")


# ---------------------------------------------------------------------------
# AlpacaBroker with an injected MOCK client
# ---------------------------------------------------------------------------
def test_alpaca_broker_injected_client_maps_account_and_positions() -> None:
    """Injected client: account + positions map correctly; no creds needed."""
    mock = MockTradingClient(equity=50_000.0, positions={"SPY": 10.0})
    broker = AlpacaBroker(client=mock)  # NO creds, NO network

    acct = broker.get_account()
    # NB: assert on the type *name*, not identity -- another test in the suite
    # reloads trend_robot.live.broker, which rebinds AccountSnapshot to a new
    # class object (so an identity isinstance check would be order-dependent).
    assert type(acct).__name__ == "AccountSnapshot"
    assert acct.equity == pytest.approx(50_000.0)
    assert acct.buying_power == pytest.approx(100_000.0)

    pos = broker.get_positions()
    assert pos["SPY"].qty == pytest.approx(10.0)
    assert pos["SPY"].avg_price == pytest.approx(100.0)
    assert pos["SPY"].market_value == pytest.approx(1000.0)


def test_alpaca_broker_injected_submit_order_maps_result() -> None:
    """submit_order routes through the mock and maps the Alpaca order back."""
    mock = MockTradingClient()
    broker = AlpacaBroker(client=mock)
    intent = OrderIntent("SPY", "buy", 5.0, 100.0, 500.0, 0.5, 0.0, "rebalance")

    result = broker.submit_order(intent)
    assert result.symbol == "SPY"
    assert result.side == "buy"
    assert result.qty == pytest.approx(5.0)
    assert result.status == "accepted"
    assert result.broker_order_id == "mock-1"
    # The mock actually received one order.
    assert len(mock.submitted) == 1
    assert float(mock.submitted[0].qty) == pytest.approx(5.0)


def test_alpaca_broker_is_market_open_reads_clock() -> None:
    """is_market_open reflects the injected clock; missing clock -> True."""
    assert AlpacaBroker(client=MockTradingClient(is_open=True)).is_market_open() is True
    assert AlpacaBroker(client=MockTradingClient(is_open=False)).is_market_open() is False

    class _NoClock:
        def get_account(self):  # pragma: no cover - not exercised here
            return _Acct("1", "1", "1")

    # A client without get_clock is tolerated (assumes open).
    assert AlpacaBroker(client=_NoClock()).is_market_open() is True


def test_alpaca_broker_recent_orders_best_effort() -> None:
    """recent_orders maps the client's orders; missing support -> []."""
    mock = MockTradingClient()
    broker = AlpacaBroker(client=mock)
    broker.submit_order(OrderIntent("SPY", "buy", 5.0, 100.0, 500.0, 0.5, 0.0, "rebalance"))
    recent = broker.recent_orders(limit=10)
    assert len(recent) == 1
    assert recent[0].symbol == "SPY"
    assert recent[0].side == "buy"

    class _NoOrders:
        pass

    assert AlpacaBroker(client=_NoOrders()).recent_orders() == []


# ---------------------------------------------------------------------------
# Helpers for the run_live.main live-path tests
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DECISION_RECORD = _PROJECT_ROOT / "decision_record.json"


def _patch_alpaca_broker(monkeypatch, mock_client: MockTradingClient) -> None:
    """Make ``_build_broker`` return an AlpacaBroker wrapping ``mock_client``.

    We monkeypatch the AlpacaBroker symbol that ``run_live`` resolves so the live
    path constructs a broker around the injected mock (no creds, no network).
    """
    def _factory(*args, **kwargs):
        return AlpacaBroker(client=mock_client)

    monkeypatch.setattr(run_live, "AlpacaBroker", _factory)


def _live_argv(state_dir: Path, *, asof: str, force: bool = False) -> list[str]:
    base = [
        "--no-dry-run", "--broker", "alpaca",
        "--asof", asof,
        "--state-dir", str(state_dir),
        "--cache-dir", str(state_dir / "cache"),
        "--decision", str(_DECISION_RECORD),
        "--history-years", "6",
        # Tests run on the deterministic synthetic provider; explicitly bypass
        # the data-integrity gate that (correctly) refuses synthetic live data.
        "--allow-synthetic-live",
        "--log-level", "WARNING",
    ]
    if force:
        base.append("--force")
    return base


# ---------------------------------------------------------------------------
# run_live.main LIVE path: submits + records results in state
# ---------------------------------------------------------------------------
def test_live_submit_records_results_and_period(tmp_path, monkeypatch) -> None:
    """A live run submits intents and records OrderResults + period in state."""
    state_dir = tmp_path / "state"
    mock = MockTradingClient(equity=100_000.0, is_open=True)
    _patch_alpaca_broker(monkeypatch, mock)

    record = run_live.main(_live_argv(state_dir, asof="2021-06-15"))

    assert record["dry_run"] is False
    assert record["submitted"] > 0
    assert record["period_key"] == period_key("2021-06-15", record["rebalance"])
    assert record["submitted_period"] == record["period_key"]
    # OrderResults captured with broker order ids + statuses.
    assert len(record["order_results"]) == record["submitted"]
    first = record["order_results"][0]
    assert set(first) == {"symbol", "side", "qty", "status", "broker_order_id"}
    assert first["status"] == "accepted"
    assert first["broker_order_id"].startswith("mock-")
    # The mock actually received the orders.
    assert len(mock.submitted) == record["submitted"]
    # State persisted with the marker.
    loaded = load_last_state(state_dir)
    assert loaded["submitted_period"] == record["period_key"]


def test_live_submit_idempotent_same_period_and_force(tmp_path, monkeypatch) -> None:
    """Second run in the SAME period skips submission; --force overrides."""
    state_dir = tmp_path / "state"
    mock = MockTradingClient(equity=100_000.0, is_open=True)
    _patch_alpaca_broker(monkeypatch, mock)

    first = run_live.main(_live_argv(state_dir, asof="2021-06-15"))
    assert first["submitted"] > 0
    n_after_first = len(mock.submitted)

    # Same calendar month -> same monthly period_key -> SKIP (no resubmission).
    second = run_live.main(_live_argv(state_dir, asof="2021-06-28"))
    assert second["submitted"] == 0
    assert second["skipped_reason"] is not None
    assert "already submitted" in second["skipped_reason"]
    assert len(mock.submitted) == n_after_first  # nothing new sent
    # The marker is preserved across the skipped run.
    assert second["submitted_period"] == first["period_key"]

    # --force overrides the guard and submits again.
    forced = run_live.main(_live_argv(state_dir, asof="2021-06-29", force=True))
    assert forced["submitted"] > 0
    assert len(mock.submitted) > n_after_first


def test_live_submit_new_period_resubmits(tmp_path, monkeypatch) -> None:
    """A run in a DIFFERENT period submits again (cadence advanced)."""
    state_dir = tmp_path / "state"
    mock = MockTradingClient(equity=100_000.0, is_open=True)
    _patch_alpaca_broker(monkeypatch, mock)

    first = run_live.main(_live_argv(state_dir, asof="2021-06-15"))
    assert first["submitted"] > 0

    # Next month -> new monthly period -> submits again without --force.
    second = run_live.main(_live_argv(state_dir, asof="2021-07-15"))
    assert second["submitted"] > 0
    assert second["skipped_reason"] is None
    assert second["submitted_period"] == period_key("2021-07-15", second["rebalance"])


def test_live_submit_market_closed_warns_but_proceeds(tmp_path, monkeypatch, caplog) -> None:
    """A closed market warns about queuing but does NOT block submission."""
    import logging

    state_dir = tmp_path / "state"
    mock = MockTradingClient(equity=100_000.0, is_open=False)
    _patch_alpaca_broker(monkeypatch, mock)

    with caplog.at_level(logging.WARNING):
        record = run_live.main(_live_argv(state_dir, asof="2021-06-15"))

    assert record["submitted"] > 0  # still submitted
    assert any("CLOSED" in r.message or "queue" in r.message.lower()
               for r in caplog.records)


# ---------------------------------------------------------------------------
# Config-drift guard
# ---------------------------------------------------------------------------
def _write_mismatched_config(tmp_path: Path, base_config: Path) -> Path:
    """Write a config.yaml with a DRIFTED fingerprint field (vol_window)."""
    text = base_config.read_text(encoding="utf-8")
    # Flip vol_window (a fingerprinted field) to drift the strategy hash.
    out = []
    changed = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("vol_window:") and not changed:
            out.append("vol_window: 99")
            changed = True
        else:
            out.append(line)
    assert changed, "expected a vol_window line in config.yaml"
    drifted = tmp_path / "config_drifted.yaml"
    drifted.write_text("\n".join(out) + "\n", encoding="utf-8")
    return drifted


def test_config_drift_refuses_live_submit(tmp_path, monkeypatch) -> None:
    """A drifted config REFUSES a live submit (RuntimeError)."""
    state_dir = tmp_path / "state"
    drifted_cfg = _write_mismatched_config(tmp_path, _PROJECT_ROOT / "config.yaml")
    mock = MockTradingClient(equity=100_000.0, is_open=True)
    _patch_alpaca_broker(monkeypatch, mock)

    argv = [
        "--no-dry-run", "--broker", "alpaca",
        "--config", str(drifted_cfg),
        "--asof", "2021-06-15",
        "--state-dir", str(state_dir),
        "--cache-dir", str(state_dir / "cache"),
        "--decision", str(_DECISION_RECORD),
        "--history-years", "6",
        "--allow-synthetic-live",  # isolate the DRIFT guard from the data gate
        "--log-level", "WARNING",
    ]
    with pytest.raises(RuntimeError, match="CONFIG DRIFT"):
        run_live.main(argv)
    # Nothing was sent to the broker.
    assert mock.submitted == []


def test_config_drift_dry_run_only_warns(tmp_path, monkeypatch, caplog) -> None:
    """A drifted config on a DRY-RUN only warns; the run completes."""
    import logging

    state_dir = tmp_path / "state"
    drifted_cfg = _write_mismatched_config(tmp_path, _PROJECT_ROOT / "config.yaml")

    argv = [
        "--dry-run",
        "--config", str(drifted_cfg),
        "--asof", "2021-06-15",
        "--state-dir", str(state_dir),
        "--cache-dir", str(state_dir / "cache"),
        "--decision", str(_DECISION_RECORD),
        "--history-years", "6",
        "--log-level", "WARNING",
    ]
    with caplog.at_level(logging.WARNING):
        record = run_live.main(argv)  # does NOT raise
    assert record["dry_run"] is True
    assert any("CONFIG DRIFT" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# --status submits nothing
# ---------------------------------------------------------------------------
def test_status_mode_submits_nothing(tmp_path, monkeypatch, capsys) -> None:
    """--status reads account + positions and submits nothing."""
    state_dir = tmp_path / "state"
    mock = MockTradingClient(equity=42_000.0, positions={"SPY": 7.0}, is_open=True)
    _patch_alpaca_broker(monkeypatch, mock)

    out = run_live.main([
        "--status", "--broker", "alpaca",
        "--state-dir", str(state_dir),
        "--log-level", "WARNING",
    ])
    assert out["status"] is True
    assert out["submitted"] == 0
    assert mock.submitted == []  # NOTHING submitted

    printed = capsys.readouterr().out
    assert "BROKER STATUS" in printed
    assert "no orders were planned or submitted" in printed.lower()
    assert "SPY" in printed  # the position is listed
    # No run-state file was written for a status check.
    assert not (state_dir).exists() or load_last_state(state_dir) is None
