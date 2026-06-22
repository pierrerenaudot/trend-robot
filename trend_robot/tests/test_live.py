"""Tests for the paper-trading DRY-RUN live layer.

Deterministic / offline ONLY: every test uses synthetic data or hand-built
frames. Nothing here ever touches a live download (Yahoo is HTTP 429 here) or
the real Alpaca API/credentials.
"""

from __future__ import annotations

import importlib
import json

import pandas as pd
import pytest

import run_live
from tests.conftest import make_config
from trend_robot.config import Config
from trend_robot.data.synthetic_provider import SyntheticProvider
from trend_robot.live.broker import (
    AccountSnapshot,
    LocalPaperBroker,
    OrderIntent,
    Position,
)
from trend_robot.live.executor import plan_orders, summarize_plan
from trend_robot.live.live_data import last_prices, prices_asof
from trend_robot.live.state import (
    has_run_for,
    load_last_state,
    save_run_state,
)
from trend_robot.live.target import compute_target_book


# ---------------------------------------------------------------------------
# executor math
# ---------------------------------------------------------------------------
def test_prices_asof_trims_trailing_all_nan_bar(cfg: Config, monkeypatch) -> None:
    """A phantom trailing all-NaN bar (e.g. today's unpublished close) is trimmed
    while interior NaN gaps are preserved -- so the live target is not zeroed out
    into a spurious flat book. Regression for the trailing-NaN bug."""
    idx = pd.bdate_range("2024-01-01", periods=5)
    cols = ["SPY", "TLT"]
    df = pd.DataFrame(
        [
            [100.0, 50.0],
            [101.0, 50.5],
            [float("nan"), float("nan")],  # interior gap -> must be PRESERVED
            [102.0, 51.0],
            [float("nan"), float("nan")],  # trailing phantom bar -> must be TRIMMED
        ],
        index=idx,
        columns=cols,
    )
    import run_research

    monkeypatch.setattr(
        run_research, "_load_prices", lambda *a, **k: (df.copy(), "synthetic")
    )
    out, src = prices_asof(cfg, idx[-1].strftime("%Y-%m-%d"), "/tmp/nocache")
    assert src == "synthetic"
    # Trailing all-NaN row dropped -> panel ends at the last real bar.
    assert out.index[-1] == idx[3]
    assert not out.iloc[-1].isna().all()
    # Interior gap preserved (not trimmed).
    assert out.loc[idx[2]].isna().all()
    assert len(out) == 4


def test_plan_orders_basic_sides_qty_notional(cfg: Config) -> None:
    """From a flat book, target weights map to correct buy sides/qty/notional."""
    target_w = pd.Series({"AAA": 0.5, "BBB": -0.25})
    last_px = pd.Series({"AAA": 100.0, "BBB": 50.0})
    equity = 10_000.0
    intents = plan_orders(target_w, {}, last_px, equity, cfg)

    by_sym = {i.symbol: i for i in intents}
    # AAA: target notional 5000 -> 50 shares buy.
    assert by_sym["AAA"].side == "buy"
    assert by_sym["AAA"].qty == pytest.approx(50.0)
    assert by_sym["AAA"].notional == pytest.approx(5000.0)
    assert by_sym["AAA"].target_weight == pytest.approx(0.5)
    assert by_sym["AAA"].current_weight == pytest.approx(0.0)
    assert by_sym["AAA"].reason == "rebalance"
    # BBB: target notional -2500 @ 50 -> 50 shares sell.
    assert by_sym["BBB"].side == "sell"
    assert by_sym["BBB"].qty == pytest.approx(50.0)
    assert by_sym["BBB"].notional == pytest.approx(2500.0)
    # Deterministic order: sorted by symbol.
    assert [i.symbol for i in intents] == ["AAA", "BBB"]


def test_plan_orders_below_min_trade_skipped(cfg: Config) -> None:
    """A delta notional below min_trade_notional is skipped."""
    target_w = pd.Series({"AAA": 0.01})
    last_px = pd.Series({"AAA": 100.0})
    equity = 10_000.0  # target notional = 100
    intents = plan_orders(
        target_w, {}, last_px, equity, cfg, min_trade_notional=200.0
    )
    assert intents == []


def test_plan_orders_held_but_dropped_sells_to_flat(cfg: Config) -> None:
    """A held symbol absent from the target is sold to flat with reason 'close'."""
    target_w = pd.Series({"AAA": 0.0})  # AAA dropped (weight 0)
    positions = {"AAA": Position("AAA", qty=10.0, avg_price=90.0, market_value=1000.0)}
    last_px = pd.Series({"AAA": 100.0})
    equity = 10_000.0
    intents = plan_orders(target_w, positions, last_px, equity, cfg)
    assert len(intents) == 1
    it = intents[0]
    assert it.side == "sell"
    assert it.qty == pytest.approx(10.0)
    assert it.reason == "close"
    assert it.target_weight == pytest.approx(0.0)
    assert it.current_weight == pytest.approx(0.1)  # 1000 / 10000


def test_plan_orders_held_symbol_not_in_target_index(cfg: Config) -> None:
    """A held symbol entirely absent from the target index is also sold to flat."""
    target_w = pd.Series({"BBB": 0.2})
    positions = {"AAA": Position("AAA", qty=5.0, avg_price=100.0, market_value=500.0)}
    last_px = pd.Series({"AAA": 100.0, "BBB": 50.0})
    equity = 10_000.0
    intents = plan_orders(target_w, positions, last_px, equity, cfg)
    by_sym = {i.symbol: i for i in intents}
    assert by_sym["AAA"].side == "sell"
    assert by_sym["AAA"].reason == "close"
    assert by_sym["AAA"].qty == pytest.approx(5.0)


def test_plan_orders_integer_truncation_vs_fractional(cfg: Config) -> None:
    """Whole-share truncation rounds toward zero; allow_fractional keeps exact."""
    # target notional 333 @ 100 -> 3.33 shares.
    target_w = pd.Series({"AAA": 0.0333})
    last_px = pd.Series({"AAA": 100.0})
    equity = 10_000.0

    whole = plan_orders(target_w, {}, last_px, equity, cfg)
    assert whole[0].qty == pytest.approx(3.0)  # truncated

    frac = plan_orders(target_w, {}, last_px, equity, cfg, allow_fractional=True)
    assert frac[0].qty == pytest.approx(3.33)


def test_plan_orders_rounds_to_zero_skipped(cfg: Config) -> None:
    """A trade that truncates to zero whole shares is skipped."""
    # target notional 50 @ 100 -> 0.5 shares -> truncates to 0.
    target_w = pd.Series({"AAA": 0.005})
    last_px = pd.Series({"AAA": 100.0})
    equity = 10_000.0
    assert plan_orders(target_w, {}, last_px, equity, cfg) == []


def test_plan_orders_leverage_guard_raises(cfg: Config) -> None:
    """Gross exposure above max_gross_leverage raises a clear error."""
    over = float(cfg.max_gross_leverage) + 1.0
    target_w = pd.Series({"AAA": over})
    last_px = pd.Series({"AAA": 100.0})
    with pytest.raises(ValueError, match="exceeds max_gross_leverage"):
        plan_orders(target_w, {}, last_px, 10_000.0, cfg)


def test_summarize_plan_totals(cfg: Config) -> None:
    """summarize_plan reports counts, notionals, gross exposure and est cost."""
    target_w = pd.Series({"AAA": 0.5, "BBB": -0.25})
    last_px = pd.Series({"AAA": 100.0, "BBB": 50.0})
    equity = 10_000.0
    intents = plan_orders(target_w, {}, last_px, equity, cfg)
    summary = summarize_plan(intents, target_w, cfg)
    assert summary["n_orders"] == 2
    assert summary["total_buy_notional"] == pytest.approx(5000.0)
    assert summary["total_sell_notional"] == pytest.approx(2500.0)
    assert summary["gross_exposure"] == pytest.approx(0.75)
    expected_cost = (cfg.cost_bps_per_side / 1e4) * 7500.0
    assert summary["est_cost"] == pytest.approx(expected_cost)


# ---------------------------------------------------------------------------
# target book no-look-ahead
# ---------------------------------------------------------------------------
def test_compute_target_book_no_lookahead(cfg: Config) -> None:
    """Target at asof is identical whether or not future rows are present."""
    provider = SyntheticProvider(seed=cfg.seed)
    prices = provider.get_prices(cfg.universe, "2015-01-01", "2021-12-31")
    asof = "2020-06-15"

    full = compute_target_book(cfg, prices, asof=asof)
    asof_ts = pd.Timestamp(asof)
    truncated = prices.loc[prices.index <= asof_ts]
    trunc_book = compute_target_book(cfg, truncated, asof=asof)

    pd.testing.assert_series_equal(full, trunc_book)


def test_compute_target_book_returns_series_over_universe(cfg: Config) -> None:
    """The book is a Series indexed by the universe symbols."""
    provider = SyntheticProvider(seed=cfg.seed)
    prices = provider.get_prices(cfg.universe, "2015-01-01", "2021-12-31")
    book = compute_target_book(cfg, prices, asof="2021-12-31")
    assert isinstance(book, pd.Series)
    assert list(book.index) == list(cfg.universe)


# ---------------------------------------------------------------------------
# live_data
# ---------------------------------------------------------------------------
def test_prices_asof_no_rows_after_asof(cfg: Config, tmp_path) -> None:
    """prices_asof never returns rows strictly after asof (offline/synthetic)."""
    asof = "2020-03-31"
    prices, source = prices_asof(
        cfg, asof, tmp_path / "cache", prefer_yfinance=False, history_years=5
    )
    assert source == "synthetic"
    assert not prices.empty
    assert prices.index.max() <= pd.Timestamp(asof)


def test_last_prices_last_valid(cfg: Config) -> None:
    """last_prices returns the last non-NaN close per column."""
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    df = pd.DataFrame(
        {"AAA": [1.0, 2.0, 3.0, float("nan")], "BBB": [10.0, 11.0, 12.0, 13.0]},
        index=idx,
    )
    lp = last_prices(df)
    assert lp["AAA"] == pytest.approx(3.0)  # last valid, NaN skipped
    assert lp["BBB"] == pytest.approx(13.0)


# ---------------------------------------------------------------------------
# LocalPaperBroker
# ---------------------------------------------------------------------------
def test_local_paper_broker_account_and_positions() -> None:
    """LocalPaperBroker reflects seeded equity and positions; records submits."""
    broker = LocalPaperBroker(equity=5000.0, positions={"AAA": 3.0})
    acct = broker.get_account()
    assert isinstance(acct, AccountSnapshot)
    assert acct.equity == pytest.approx(5000.0)
    pos = broker.get_positions()
    assert pos["AAA"].qty == pytest.approx(3.0)

    intent = OrderIntent("AAA", "buy", 2.0, 100.0, 200.0, 0.5, 0.0, "rebalance")
    result = broker.submit_order(intent)
    assert result.status == "accepted_simulated"
    assert result.broker_order_id is None
    assert len(broker.submitted) == 1


# ---------------------------------------------------------------------------
# run_live.main end-to-end (dry-run, synthetic, no network/alpaca)
# ---------------------------------------------------------------------------
def test_run_live_dry_run_end_to_end(tmp_path, capsys) -> None:
    """A dry-run on synthetic data produces a non-empty plan and sends nothing."""
    cache_dir = tmp_path / "cache"
    state_dir = tmp_path / "state"
    asof = "2021-06-15"

    captured: dict = {}
    real_build = run_live._build_broker

    def _spy_build(**kwargs):
        broker = real_build(**kwargs)
        captured["broker"] = broker
        return broker

    run_live._build_broker = _spy_build
    try:
        record = run_live.main([
            "--dry-run",
            "--asof", asof,
            "--cache-dir", str(cache_dir),
            "--state-dir", str(state_dir),
            "--history-years", "6",
            "--log-level", "WARNING",
        ])
    finally:
        run_live._build_broker = real_build

    # Non-empty plan from a flat book.
    assert record["dry_run"] is True
    assert record["data_source"] == "synthetic"
    assert record["submitted"] == 0
    assert len(record["orders"]) > 0

    # The local broker received ZERO submissions.
    broker = captured["broker"]
    assert isinstance(broker, LocalPaperBroker)
    assert broker.submitted == []

    # State was written and the preview banner printed.
    assert has_run_for(state_dir, asof)
    out = capsys.readouterr().out
    assert "DRY-RUN" in out
    assert "NOT a deploy signal" in out


def test_run_live_refuses_live_submit_on_local() -> None:
    """--no-dry-run --broker local is refused (argparse SystemExit)."""
    with pytest.raises(SystemExit):
        run_live.main(["--no-dry-run", "--broker", "local"])


# ---------------------------------------------------------------------------
# state round-trip + idempotence
# ---------------------------------------------------------------------------
def test_state_roundtrip_and_has_run_for(tmp_path) -> None:
    """save -> load round-trips; has_run_for is idempotence-aware."""
    state_dir = tmp_path / "state"
    asof = "2021-01-04"
    assert has_run_for(state_dir, asof) is False

    record = {
        "asof": asof,
        "target_weights": pd.Series({"AAA": 0.5}),  # exercises Series conversion
        "n": 3,
    }
    path = save_run_state(state_dir, record)
    assert path.is_file()
    assert has_run_for(state_dir, asof) is True

    loaded = load_last_state(state_dir)
    assert loaded is not None
    assert loaded["asof"] == asof
    assert loaded["target_weights"] == {"AAA": 0.5}
    assert "generated_at" in loaded


def test_state_load_last_none_when_empty(tmp_path) -> None:
    """load_last_state returns None when there is no state."""
    assert load_last_state(tmp_path / "missing") is None


# ---------------------------------------------------------------------------
# AlpacaBroker: import-clean, constructs-with-error without creds
# ---------------------------------------------------------------------------
def test_alpaca_module_imports_without_creds(monkeypatch) -> None:
    """The broker module imports fine with no alpaca creds in the environment."""
    for var in (
        "APCA_API_KEY_ID", "APCA_API_SECRET_KEY",
        "ALPACA_API_KEY", "ALPACA_SECRET_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    module = importlib.import_module("trend_robot.live.broker")
    importlib.reload(module)
    assert hasattr(module, "AlpacaBroker")


def test_alpaca_broker_raises_clear_error_without_creds(monkeypatch) -> None:
    """Constructing AlpacaBroker without keys raises a clear, actionable error."""
    for var in (
        "APCA_API_KEY_ID", "APCA_API_SECRET_KEY",
        "ALPACA_API_KEY", "ALPACA_SECRET_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    from trend_robot.live.broker import AlpacaBroker

    with pytest.raises(RuntimeError, match="requires API credentials"):
        AlpacaBroker()
