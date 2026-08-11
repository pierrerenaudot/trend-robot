"""Live / paper-trading layer for the TSMOM robot (DRY-RUN milestone).

ENGINEERING DRESS-REHEARSAL ONLY -- this layer is NOT a deploy signal. The
strategy spec defers live/paper trading (section 10) and the current section-6.5
verdict is REJECT. The first milestone is a ``run_live.py --dry-run`` that
*computes and displays* today's orders WITHOUT sending anything, structured so
that wiring the real broker API later is a drop-in.

Public surface
--------------
* Value objects: :class:`AccountSnapshot`, :class:`Position`,
  :class:`OrderIntent`, :class:`OrderResult`.
* Brokers: the :class:`Broker` protocol, :class:`LocalPaperBroker` (the dry-run
  broker -- no external deps), :class:`AlpacaBroker` (lazy alpaca-py adapter).
* Data: :func:`prices_asof`, :func:`last_prices`.
* Target & execution: :func:`compute_target_book`, :func:`plan_orders`,
  :func:`summarize_plan`.
* State: :func:`save_run_state`, :func:`load_last_state`, :func:`has_run_for`.
"""

from __future__ import annotations

from trend_robot.live.broker import (
    AccountSnapshot,
    AlpacaBroker,
    Broker,
    LocalPaperBroker,
    OrderIntent,
    OrderResult,
    Position,
)
from trend_robot.live.alerts import send_alert
from trend_robot.live.executor import plan_orders, summarize_plan
from trend_robot.live.reconcile import (
    ReconcileReport,
    format_reconcile_report,
    reconcile_book,
)
from trend_robot.live.live_data import last_prices, prices_asof
from trend_robot.live.state import has_run_for, load_last_state, save_run_state
from trend_robot.live.target import compute_target_book

__all__ = [
    "AccountSnapshot",
    "Position",
    "OrderIntent",
    "OrderResult",
    "Broker",
    "LocalPaperBroker",
    "AlpacaBroker",
    "prices_asof",
    "last_prices",
    "compute_target_book",
    "plan_orders",
    "summarize_plan",
    "reconcile_book",
    "format_reconcile_report",
    "ReconcileReport",
    "send_alert",
    "save_run_state",
    "load_last_state",
    "has_run_for",
]
