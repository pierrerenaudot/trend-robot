"""Paper-trading DRY-RUN entry point for the TSMOM robot (milestone 1).

ENGINEERING DRESS-REHEARSAL -- NOT A DEPLOY SIGNAL. The strategy spec defers
live/paper trading (section 10); deploy only after a human-approved section-6.5
RETAIN (see run_validation.py). The dry-run is plumbing, independent of any verdict.
This script COMPUTES and DISPLAYS today's rebalance orders for the chosen broker
(Alpaca paper) WITHOUT sending anything. It is structured so that wiring the real
Alpaca API later is just a plug-in (swap :class:`LocalPaperBroker` for
:class:`AlpacaBroker`).

Flow
----
load config + seed -> resolve ``asof`` -> fetch prices as-of (synthetic by
default; cached Yahoo with ``--live``) -> compute today's target book (no
look-ahead) -> reconcile against current positions into an order plan -> PRINT a
clear preview table + summary + a DRY-RUN/PAPER banner -> persist run state.

Safety
------
* DEFAULT is ``--dry-run`` (true). In a dry-run NOTHING is submitted -- only
  previewed.
* Real submission happens ONLY with ``--no-dry-run`` AND ``--broker alpaca``;
  ``--no-dry-run --broker local`` is refused.
* The dry-run requires NO Alpaca credentials and NO network: the synthetic
  fallback keeps it working fully offline.

Unattended-running guards (LIVE submit path)
--------------------------------------------
These make a daily-scheduled job safe to leave running for the monthly cadence:

* **Config-drift guard.** A live submit is REFUSED if the running config no
  longer matches the frozen pre-registered ``decision_record.json`` (we only
  paper-trade the pre-registered variant the forward hold-out will judge). A
  dry-run merely warns.
* **Cadence + idempotence guard.** At most ONE live submission per rebalance
  *period* (see :func:`~trend_robot.live.scheduling.period_key`): a re-run in
  the same period skips submission (still previews + records state) unless
  ``--force``. A daily job therefore trades once per month for monthly cadence.
* **Market-open check.** When the clock reports the market closed, the runner
  warns that DAY orders will queue for the next session; it does not hard-block.
* **--status.** Connect, print the account + positions, submit nothing, exit --
  the home connectivity test.

No market values are hard-coded: equity defaults to ``cfg.initial_capital`` and
all strategy parameters flow from ``config.yaml`` through the typed
:class:`Config`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from trend_robot.config import Config, load_config, set_global_seed
from trend_robot.live.alerts import send_alert
from trend_robot.live.broker import (
    AlpacaBroker,
    Broker,
    LocalPaperBroker,
    OrderIntent,
    OrderResult,
    Position,
)
from trend_robot.live.executor import plan_orders, summarize_plan
from trend_robot.live.live_data import last_prices, prices_asof
from trend_robot.live.reconcile import format_reconcile_report, reconcile_book
from trend_robot.live.scheduling import period_key
from trend_robot.live.state import load_last_state, save_run_state
from trend_robot.live.target import compute_target_book
from trend_robot.validation.preregistration import (
    PreRegistrationError,
    load_decision,
    verify_config_matches,
)

_PROJECT_ROOT = Path(__file__).resolve().parent
_DEFAULT_CONFIG = _PROJECT_ROOT / "config.yaml"
_DEFAULT_CACHE_DIR = _PROJECT_ROOT / ".cache"
_DEFAULT_STATE_DIR = _PROJECT_ROOT / "live_state"
_DEFAULT_DECISION = _PROJECT_ROOT / "decision_record.json"
_DEFAULT_HISTORY_YEARS = 15

_LOGGER = logging.getLogger("trend_robot.run_live")


def _load_positions(path: str | None) -> dict[str, float]:
    """Load a ``{symbol: qty}`` positions map from a JSON file (or empty).

    Parameters
    ----------
    path:
        Path to a JSON object mapping symbol -> quantity, or ``None`` for a flat
        book.

    Returns
    -------
    dict[str, float]
        The parsed positions (empty when ``path`` is ``None``).

    Raises
    ------
    ValueError
        If the file is not a JSON object of symbol -> number.
    """
    if not path:
        return {}
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(
            f"--positions file must contain a JSON object {{symbol: qty}}, "
            f"got {type(raw).__name__}."
        )
    return {str(k): float(v) for k, v in raw.items()}


def _build_broker(
    *,
    broker_name: str,
    dry_run: bool,
    equity: float,
    positions: dict[str, float],
    need_alpaca: bool = False,
) -> Broker:
    """Pick and construct the broker for this run.

    The dry-run always uses :class:`LocalPaperBroker` (seeded with ``equity`` and
    ``positions``) regardless of ``broker_name``, so it never needs credentials
    or a network. A real :class:`AlpacaBroker` is constructed for a live run
    targeting Alpaca, or whenever ``need_alpaca`` is set (e.g. the ``--status``
    connectivity test, which must talk to the real broker even though it submits
    nothing).
    """
    if need_alpaca and broker_name == "alpaca":
        return AlpacaBroker(paper=True)
    if dry_run or broker_name == "local":
        return LocalPaperBroker(equity=equity, positions=positions)
    # broker_name == "alpaca" and not dry_run
    return AlpacaBroker(paper=True)


def _format_preview_table(
    intents: list[OrderIntent], target_w: pd.Series
) -> str:
    """Render the aligned order-preview table as a string.

    Columns: symbol, target_w, current_w, side, qty, est_price, notional,
    reason. Includes a trailing note for in-target symbols that produced no
    order (so the operator can see they were considered).
    """
    header = (
        f"{'SYMBOL':<8}{'TARGET_W':>10}{'CURRENT_W':>11}"
        f"{'SIDE':>6}{'QTY':>12}{'EST_PX':>11}{'NOTIONAL':>14}  REASON"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    if not intents:
        lines.append("(no orders -- book already at target / all skipped)")
    for it in intents:
        lines.append(
            f"{it.symbol:<8}"
            f"{it.target_weight:>10.4f}"
            f"{it.current_weight:>11.4f}"
            f"{it.side:>6}"
            f"{it.qty:>12.4f}"
            f"{it.est_price:>11.2f}"
            f"{it.notional:>14.2f}"
            f"  {it.reason}"
        )

    # Trailing note: symbols carrying a non-trivial target weight that produced
    # NO order (already at target, rounded to zero, or below the min trade), so
    # the operator can see they were considered rather than silently dropped.
    ordered = {it.symbol for it in intents}
    considered = [
        str(sym)
        for sym in target_w.index
        if abs(float(target_w[sym])) > 1e-9 and str(sym) not in ordered
    ]
    if considered:
        lines.append(f"(in target, no order: {', '.join(considered)})")

    return "\n".join(lines)


def _print_preview(
    *,
    intents: list[OrderIntent],
    target_w: pd.Series,
    summary: dict,
    asof: str,
    data_source: str,
    equity: float,
    broker_name: str,
    dry_run: bool,
) -> None:
    """Print the banner, preview table and summary line to stdout."""
    mode = "DRY-RUN (preview only, nothing sent)" if dry_run else "LIVE SUBMIT"
    print("=" * 72)
    print("TSMOM PAPER-TRADING -- ORDER PREVIEW")
    print(f"  MODE       : {mode}")
    print(f"  asof       : {asof}   (orders to place NEXT session)")
    print(f"  broker     : {broker_name} (paper)   data source: {data_source}")
    print(f"  equity     : {equity:,.2f}")
    print("-" * 72)
    print(_format_preview_table(intents, target_w))
    print("-" * 72)
    print(
        f"  orders={summary['n_orders']}  "
        f"gross_exposure={summary['gross_exposure']:.4f}  "
        f"buy_notional={summary['total_buy_notional']:,.2f}  "
        f"sell_notional={summary['total_sell_notional']:,.2f}  "
        f"est_cost={summary['est_cost']:,.2f}"
    )
    print("=" * 72)
    if dry_run:
        print(
            "BANNER: This is a PAPER-TRADING DRY-RUN dress-rehearsal -- NOT a "
            "deploy signal."
        )
        print(
            "        Spec section 10 defers live/paper trading; deploy only after "
            "a human-approved §6.5 RETAIN (run_validation.py)."
        )
        print("        No orders were submitted to any broker.")
    else:
        print(
            "BANNER: LIVE PAPER SUBMIT -- orders were sent to the (paper) broker. "
            "PAPER money only."
        )
        print(
            "        This accrues the pristine forward hold-out; it is NOT a "
            "validated deploy (§6.5 verdict pending -- run_holdout.py)."
        )
    print("=" * 72)


def _check_config_drift(
    cfg: Config, decision_path: str, *, live_submit: bool
) -> None:
    """Guard a LIVE submit against drift from the pre-registered decision.

    If a ``decision_record.json`` exists at ``decision_path``, the running
    ``cfg`` is checked against the frozen pre-registered variant via
    :func:`verify_config_matches`. On a LIVE submit a mismatch is a hard error
    (:class:`RuntimeError`): we paper-trade ONLY the pre-registered strategy
    that the forward hold-out will judge -- trading a drifted variant would
    contaminate that pristine read. For a dry-run (``live_submit=False``) a
    mismatch only WARNS. A missing record warns and is otherwise tolerated.

    Parameters
    ----------
    cfg:
        The running configuration.
    decision_path:
        Path to the pre-registration ``decision_record.json``.
    live_submit:
        ``True`` on a real submission (``--no-dry-run``), where drift is fatal.

    Raises
    ------
    RuntimeError
        On a live submit when the running config has drifted from the frozen
        pre-registered fingerprint.
    """
    path = Path(decision_path)
    if not path.is_file():
        _LOGGER.warning(
            "No pre-registration decision record at %s; cannot verify the "
            "running config matches the frozen pre-registered variant.", path,
        )
        return

    try:
        record = load_decision(path)
    except PreRegistrationError as exc:
        if live_submit:
            raise RuntimeError(
                f"Could not load the pre-registration record at {path} for the "
                f"config-drift guard, refusing to submit live: {exc}"
            ) from exc
        _LOGGER.warning("Could not load decision record at %s: %s", path, exc)
        return

    if verify_config_matches(cfg, record):
        _LOGGER.info(
            "Config-drift guard OK: running config matches the pre-registered "
            "variant (hash %s).", record.config_hash[:12],
        )
        return

    message = (
        "CONFIG DRIFT: the running config does NOT match the frozen "
        "pre-registered strategy (decision_date={date}, expected hash "
        "{hash}...). We only paper-trade the pre-registered variant that the "
        "forward hold-out will judge.".format(
            date=record.decision_date, hash=record.config_hash[:12],
        )
    )
    if live_submit:
        raise RuntimeError(
            message
            + " Refusing the live submit. Re-freeze the decision deliberately "
            "(resets the pristine forward window) or restore the config."
        )
    _LOGGER.warning("%s (dry-run: warning only, not blocking).", message)


def _check_data_integrity(
    prices: pd.DataFrame,
    data_source: str,
    asof: str,
    *,
    live_submit: bool,
    max_stale_days: int,
    allow_synthetic_live: bool,
) -> None:
    """Refuse to trade real (paper) orders on synthetic or stale prices.

    The research pipeline deliberately falls back to the seeded
    :class:`SyntheticProvider` when Yahoo is unusable (HTTP 429) so offline
    runs keep working. That fallback must NEVER reach a live submission: a
    rate-limited first-of-month run would otherwise trade a random walk into
    the pre-registered forward track with a green checkmark. The daily
    scheduled job is the natural retry loop -- skipping today costs nothing.

    Checks (fatal on a live submit, warning on a dry-run):

    * ``data_source`` must be ``"yfinance"`` (real market data), unless
      ``allow_synthetic_live`` explicitly opts out (tests/simulation only);
    * the panel's last bar must be recent relative to ``asof`` (at most
      ``max_stale_days`` calendar days old -- generous enough for weekends and
      holiday bridges, tight enough to catch a dead cache).

    Raises
    ------
    RuntimeError
        On a live submit with synthetic/stale data (after firing an alert).
    """
    problems: list[str] = []

    if data_source != "yfinance" and not allow_synthetic_live:
        problems.append(
            f"data source is '{data_source}' (not real market data); a "
            "Yahoo failure must skip the run, not trade a synthetic random walk"
        )

    if len(prices.index) == 0:
        problems.append("price panel is empty")
    else:
        last_bar = pd.Timestamp(prices.index[-1])
        age_days = (pd.Timestamp(asof) - last_bar).days
        if age_days > int(max_stale_days):
            problems.append(
                f"last price bar {last_bar.date()} is {age_days} calendar "
                f"days old vs asof {asof} (max {max_stale_days}); the panel "
                "is stale"
            )

    if not problems:
        return

    message = (
        "DATA-INTEGRITY GATE: " + "; ".join(problems) + "."
    )
    if live_submit:
        send_alert(
            f"[TSMOM] LIVE SUBMIT REFUSED -- {message} The scheduled job will "
            "retry on the next run."
        )
        raise RuntimeError(
            message + " Refusing the live submit; the daily schedule will "
            "retry with fresh data."
        )
    _LOGGER.warning("%s (dry-run: warning only, not blocking).", message)


def _last_submitted_period(state_dir: str) -> str | None:
    """Return the period key of the last successful live submission, if any.

    Scans the most recent persisted run record for a ``submitted_period``
    marker written by a prior LIVE submit. Used to enforce one-trade-per-period
    idempotence across separate scheduled invocations.

    Parameters
    ----------
    state_dir:
        Directory holding the run-state files.

    Returns
    -------
    str | None
        The recorded ``submitted_period``, or ``None`` when no live submission
        has been recorded yet.
    """
    last = load_last_state(state_dir)
    if not last:
        return None
    marker = last.get("submitted_period")
    return str(marker) if marker else None


def _print_status(broker: Broker, *, broker_name: str) -> None:
    """Print an account snapshot + current positions; submit NOTHING.

    The ``--status`` connectivity test: confirms credentials and reachability by
    reading the account and positions and rendering a small table. No orders are
    planned or submitted.
    """
    account = broker.get_account()
    positions = broker.get_positions()

    print("=" * 72)
    print("TSMOM PAPER-TRADING -- BROKER STATUS (connectivity check)")
    print(f"  broker        : {broker_name} (paper)")
    print("-" * 72)
    print(
        f"  equity        : {account.equity:,.2f}\n"
        f"  cash          : {account.cash:,.2f}\n"
        f"  buying_power  : {account.buying_power:,.2f}"
    )
    print("-" * 72)
    if positions:
        header = f"{'SYMBOL':<8}{'QTY':>14}{'AVG_PRICE':>14}{'MKT_VALUE':>16}"
        print(header)
        print("-" * len(header))
        for sym in sorted(positions):
            pos = positions[sym]
            print(
                f"{pos.symbol:<8}{pos.qty:>14.4f}"
                f"{pos.avg_price:>14.2f}{pos.market_value:>16.2f}"
            )
    else:
        print("  (no open positions)")
    print("-" * 72)
    print("STATUS check only -- no orders were planned or submitted.")
    print("=" * 72)


def main(argv: list[str] | None = None) -> dict:
    """CLI entry point: compute and preview (or submit) today's orders.

    Parameters
    ----------
    argv:
        Optional argument vector (defaults to ``sys.argv``).

    Returns
    -------
    dict
        The persisted run record (also written to the state directory).
    """
    parser = argparse.ArgumentParser(
        description=(
            "TSMOM paper-trading DRY-RUN -- preview today's rebalance orders "
            "without sending anything (engineering dress-rehearsal, NOT a "
            "deploy signal)."
        )
    )
    parser.add_argument("--config", default=str(_DEFAULT_CONFIG),
                        help="Path to config.yaml (default: project config.yaml).")
    parser.add_argument("--asof", default=None,
                        help="As-of date YYYY-MM-DD (default: today).")
    parser.add_argument("--equity", type=float, default=None,
                        help="Account equity (default: cfg.initial_capital).")
    parser.add_argument("--positions", default=None,
                        help="JSON file {symbol: qty} (default: flat book).")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true",
                        default=True, help="Preview only, send nothing (DEFAULT).")
    parser.add_argument("--no-dry-run", dest="dry_run", action="store_false",
                        help="Actually submit orders (requires --broker alpaca).")
    parser.add_argument("--live", action="store_true",
                        help="Prefer cached Yahoo prices (else synthetic).")
    parser.add_argument("--broker", choices=["local", "alpaca"], default="local",
                        help="Broker to use (default: local).")
    parser.add_argument("--status", action="store_true",
                        help="Connect to the broker, print account + positions, "
                             "submit NOTHING, and exit (home connectivity test).")
    parser.add_argument("--force", action="store_true",
                        help="Override the per-period idempotence/cadence guard "
                             "and submit even if already submitted this period.")
    parser.add_argument("--cache-dir", default=str(_DEFAULT_CACHE_DIR),
                        help="Parquet price cache directory (default: ./.cache).")
    parser.add_argument("--state-dir", default=str(_DEFAULT_STATE_DIR),
                        help="Run-state directory (default: ./live_state).")
    parser.add_argument("--decision", default=str(_DEFAULT_DECISION),
                        help="Pre-registration decision_record.json for the "
                             "config-drift guard (default: project root).")
    parser.add_argument("--min-trade-notional", type=float, default=0.0,
                        help="Skip trades below this notional (default: 0).")
    parser.add_argument("--allow-fractional", action="store_true",
                        help="Allow fractional-share orders (default: whole shares).")
    parser.add_argument("--history-years", type=int, default=_DEFAULT_HISTORY_YEARS,
                        help=f"Years of price history (default: {_DEFAULT_HISTORY_YEARS}).")
    parser.add_argument("--reconcile-tolerance", type=float, default=0.25,
                        help="L1 weight-gap tolerance before the broker book is "
                             "flagged materially off-target (default: 0.25).")
    parser.add_argument("--max-stale-days", type=int, default=7,
                        help="Max calendar-day age of the last price bar vs asof "
                             "before the data is considered stale (default: 7).")
    parser.add_argument("--allow-synthetic-live", action="store_true",
                        help="TESTS/SIMULATION ONLY: bypass the data-integrity "
                             "gate that refuses live submits on synthetic data.")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # --- Safety: refuse a live submit against the local simulator. ---------
    if not args.dry_run and args.broker == "local":
        parser.error(
            "--no-dry-run requires --broker alpaca; the local broker is a "
            "simulator and cannot submit real orders."
        )

    # --- Config + reproducibility. -----------------------------------------
    cfg: Config = load_config(args.config)
    set_global_seed(cfg.seed)

    # --- --status: connectivity check. Read account + positions, submit ----
    #     NOTHING, return. No prices, no planning, no state mutation.
    if args.status:
        equity_status = (
            float(args.equity) if args.equity is not None
            else float(cfg.initial_capital)
        )
        status_broker = _build_broker(
            broker_name=args.broker, dry_run=args.dry_run,
            equity=equity_status, positions=_load_positions(args.positions),
            need_alpaca=True,
        )
        _print_status(status_broker, broker_name=args.broker)
        return {"status": True, "broker": args.broker, "submitted": 0}

    asof = args.asof or pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
    equity = float(args.equity) if args.equity is not None else float(cfg.initial_capital)
    positions_in = _load_positions(args.positions)

    _LOGGER.info(
        "Live preview | asof=%s | broker=%s | dry_run=%s | equity=%.2f | "
        "universe=%s",
        asof, args.broker, args.dry_run, equity, cfg.universe,
    )

    # --- Prices as-of (no rows after asof) -> today's target book. ---------
    prices, data_source = prices_asof(
        cfg, asof, args.cache_dir,
        prefer_yfinance=args.live, history_years=args.history_years,
    )
    target_w = compute_target_book(cfg, prices, asof=asof)
    last_px = last_prices(prices)

    # --- Data-integrity gate. NEVER trade real (paper) orders on synthetic or
    #     stale prices; a dry-run only warns. (The daily schedule retries.)
    _check_data_integrity(
        prices, data_source, asof,
        live_submit=not args.dry_run,
        max_stale_days=args.max_stale_days,
        allow_synthetic_live=args.allow_synthetic_live,
    )

    # --- Config-drift guard. On a LIVE submit, REFUSE if the running config
    #     drifted from the frozen pre-registered variant; a dry-run only warns.
    _check_config_drift(cfg, args.decision, live_submit=not args.dry_run)

    # --- Cadence + idempotence: at most one LIVE submit per rebalance period.
    this_period = period_key(asof, cfg.rebalance)
    last_period = _last_submitted_period(args.state_dir)
    already_submitted_period = (last_period == this_period)

    # --- Broker: account + current positions. ------------------------------
    broker = _build_broker(
        broker_name=args.broker, dry_run=args.dry_run,
        equity=equity, positions=positions_in,
    )
    account = broker.get_account()
    positions: dict[str, Position] = broker.get_positions()

    # --- Plan the rebalance orders. ----------------------------------------
    intents = plan_orders(
        target_w, positions, last_px, account.equity, cfg,
        min_trade_notional=args.min_trade_notional,
        allow_fractional=args.allow_fractional,
    )
    summary = summarize_plan(intents, target_w, cfg)

    # --- Reconciliation: is the broker book where the strategy expects it? -
    # A gap is EXPECTED right before a rebalance; it is an ANOMALY on a run
    # that skips because the period was already traded (the book should then
    # sit near target -- a large gap means the fills are gone: account reset,
    # silent rejection, manual liquidation). July-2026 incident, systematized.
    reconcile = reconcile_book(
        target_w, positions, last_px, account.equity,
        tolerance=args.reconcile_tolerance,
    )
    reconcile_escalated = False

    # --- SAFETY: only submit on an explicit live run. ----------------------
    submitted = 0
    submitted_period: str | None = last_period  # carry forward prior marker
    skipped_reason: str | None = None
    order_results: list[OrderResult] = []

    if not args.dry_run:
        # Cadence/idempotence gate: skip if we already submitted this period
        # (unless --force). Daily-scheduled monthly cadence -> trades once/month.
        if already_submitted_period and not args.force:
            skipped_reason = (
                f"already submitted for period '{this_period}' "
                f"(cadence={cfg.rebalance}); use --force to override"
            )
            _LOGGER.warning(
                "SKIP live submit: %s. Run state still recorded.", skipped_reason,
            )
            if reconcile.anomaly:
                # Period marked traded but the book is NOT near target: the
                # earlier submission's positions are gone. Alert loudly and
                # fail the run (red CI -> GitHub e-mail) so a human decides
                # (typically: re-run with --force after checking the account).
                reconcile_escalated = True
                report_text = format_reconcile_report(reconcile)
                _LOGGER.error(
                    "RECONCILIATION ANOMALY on a skipped run:\n%s", report_text,
                )
                print(report_text)
                send_alert(
                    "[TSMOM] RECONCILIATION ANOMALY -- period "
                    f"'{this_period}' is marked submitted but the broker book "
                    f"is off-target (L1 gap {reconcile.l1_deviation:.3f}, "
                    f"gross {reconcile.current_gross:.3f} vs target "
                    f"{reconcile.target_gross:.3f}). Check the account "
                    "(reset? rejected fills?) and re-run with --force to "
                    "repair."
                )
        else:
            # Market-open check (warn only; DAY orders queue for next session).
            try:
                market_open = broker.is_market_open()
            except Exception as exc:  # noqa: BLE001 - never block on the clock
                _LOGGER.warning("Could not read market clock (%s); proceeding.", exc)
                market_open = True
            if not market_open:
                _LOGGER.warning(
                    "Market is CLOSED: DAY orders will QUEUE for the next "
                    "trading session (not rejected). Proceeding.",
                )
            for intent in intents:
                order_results.append(broker.submit_order(intent))
                submitted += 1
            submitted_period = this_period
            _LOGGER.info(
                "Submitted %d live orders via %s for period %s.",
                submitted, args.broker, this_period,
            )

    # --- Preview. ----------------------------------------------------------
    _print_preview(
        intents=intents, target_w=target_w, summary=summary, asof=asof,
        data_source=data_source, equity=account.equity,
        broker_name=args.broker, dry_run=args.dry_run,
    )
    if skipped_reason:
        print(f"NOTE: live submission SKIPPED -- {skipped_reason}.")

    # --- Persist run state (captures the broker's order results + period). -
    record = {
        "asof": asof,
        "dry_run": bool(args.dry_run),
        "broker": args.broker,
        "data_source": data_source,
        "equity": account.equity,
        "period_key": this_period,
        "rebalance": cfg.rebalance,
        "submitted_period": submitted_period,
        "skipped_reason": skipped_reason,
        "target_weights": {k: float(v) for k, v in target_w.items()},
        "orders": [
            {
                "symbol": it.symbol, "side": it.side, "qty": it.qty,
                "est_price": it.est_price, "notional": it.notional,
                "target_weight": it.target_weight,
                "current_weight": it.current_weight, "reason": it.reason,
            }
            for it in intents
        ],
        "order_results": [
            {
                "symbol": r.symbol, "side": r.side, "qty": r.qty,
                "status": r.status, "broker_order_id": r.broker_order_id,
            }
            for r in order_results
        ],
        "summary": summary,
        "submitted": submitted,
        "reconcile": {
            "l1_deviation": reconcile.l1_deviation,
            "max_deviation": reconcile.max_deviation,
            "current_gross": reconcile.current_gross,
            "target_gross": reconcile.target_gross,
            "tolerance": reconcile.tolerance,
            "book_flat_but_target_invested":
                reconcile.book_flat_but_target_invested,
            "materially_off_target": reconcile.materially_off_target,
            "anomaly": reconcile.anomaly,
            "escalated": reconcile_escalated,
        },
        # Non-zero when a human must look (reconciliation anomaly on a run
        # that skipped): the CLI exits with this, turning the CI run red.
        "exit_code": 2 if reconcile_escalated else 0,
    }
    state_path = save_run_state(args.state_dir, record)
    _LOGGER.info("Wrote run state to %s", state_path)

    return record


if __name__ == "__main__":
    # Exit non-zero on an escalated reconciliation anomaly so the scheduled CI
    # run turns red and GitHub e-mails the owner (the primary alert channel).
    sys.exit(int(main().get("exit_code", 0)))
