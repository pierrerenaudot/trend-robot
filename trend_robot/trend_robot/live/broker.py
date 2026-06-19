"""Broker abstraction for the paper-trading dry-run milestone.

This module defines the small, typed *value objects* exchanged with a broker
(account snapshot, positions, order intents/results), the :class:`Broker`
:class:`typing.Protocol` that every concrete broker implements, and two
concrete brokers:

* :class:`LocalPaperBroker` -- a pure, in-memory simulator with NO external
  dependencies. This is the broker the DRY-RUN uses: ``submit_order`` merely
  *records* the intent and returns a simulated acceptance, so a dry-run never
  reaches out to any network or exchange.
* :class:`AlpacaBroker` -- a thin adapter over the ``alpaca-py`` PAPER
  endpoint. The ``alpaca`` package is imported *lazily* (inside ``__init__`` and
  the methods that need it) so that importing this module never requires the
  dependency or any credentials. Constructing :class:`AlpacaBroker` without API
  keys raises a clear, actionable error -- but only at construction time, never
  at import. This path is NOT exercised by the dry-run; it is here so wiring the
  real API later is a drop-in.

Design note
-----------
All money/quantity values flow from the caller (ultimately from :class:`Config`
and the live account); nothing is hard-coded here.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

_LOGGER = logging.getLogger("trend_robot.live.broker")

__all__ = [
    "AccountSnapshot",
    "Position",
    "OrderIntent",
    "OrderResult",
    "Broker",
    "LocalPaperBroker",
    "AlpacaBroker",
]


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AccountSnapshot:
    """Immutable snapshot of a broker account at a point in time.

    Attributes
    ----------
    equity:
        Total account equity (cash + market value of positions).
    cash:
        Settled cash available in the account.
    buying_power:
        Notional the account may deploy (may exceed ``cash`` under margin).
    """

    equity: float
    cash: float
    buying_power: float


@dataclass(frozen=True)
class Position:
    """Immutable description of a single held position.

    Attributes
    ----------
    symbol:
        Ticker symbol of the instrument.
    qty:
        Signed quantity held (negative for shorts).
    avg_price:
        Average entry price of the position.
    market_value:
        Current market value of the position (``qty * last_price``).
    """

    symbol: str
    qty: float
    avg_price: float
    market_value: float


@dataclass(frozen=True)
class OrderIntent:
    """A planned order, fully priced and reconciled, ready to be previewed.

    An :class:`OrderIntent` is the output of the executor's rebalance planner.
    It is *intent*, not execution: in a dry-run it is only displayed; only a
    live submission turns it into an :class:`OrderResult`.

    Attributes
    ----------
    symbol:
        Ticker symbol to trade.
    side:
        ``"buy"`` or ``"sell"``.
    qty:
        Absolute (non-negative) order quantity in shares.
    est_price:
        Estimated execution price (the last known close).
    notional:
        Estimated traded notional (``qty * est_price``, non-negative).
    target_weight:
        Desired portfolio weight for ``symbol`` after this trade.
    current_weight:
        Current portfolio weight for ``symbol`` before this trade.
    reason:
        Human-readable rationale (e.g. ``"rebalance"``, ``"close"``).
    """

    symbol: str
    side: str
    qty: float
    est_price: float
    notional: float
    target_weight: float
    current_weight: float
    reason: str


@dataclass(frozen=True)
class OrderResult:
    """The broker's response to a submitted :class:`OrderIntent`.

    Attributes
    ----------
    symbol:
        Ticker symbol the order was for.
    side:
        ``"buy"`` or ``"sell"``.
    qty:
        Absolute quantity submitted.
    status:
        Broker-reported status (e.g. ``"accepted_simulated"`` for the local
        paper broker, or the live Alpaca order status string).
    broker_order_id:
        Broker-assigned order id, or ``None`` when not applicable (e.g. the
        local simulator).
    """

    symbol: str
    side: str
    qty: float
    status: str
    broker_order_id: str | None


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------
@runtime_checkable
class Broker(Protocol):
    """The minimal broker contract used by the live runner.

    A broker exposes the current account and positions and accepts order
    intents. Concrete implementations may simulate (paper) or route to a real
    venue; the live runner only depends on this surface.
    """

    def get_account(self) -> AccountSnapshot:
        """Return the current :class:`AccountSnapshot`."""
        ...

    def get_positions(self) -> dict[str, Position]:
        """Return current positions keyed by symbol."""
        ...

    def submit_order(self, intent: OrderIntent) -> OrderResult:
        """Submit an :class:`OrderIntent` and return its :class:`OrderResult`."""
        ...


# ---------------------------------------------------------------------------
# Local in-memory paper broker (used by the dry-run)
# ---------------------------------------------------------------------------
class LocalPaperBroker:
    """Pure, in-memory simulated broker -- the dry-run's broker.

    The account is seeded with a starting equity and (optionally) initial
    positions. ``get_account``/``get_positions`` read from in-memory state and
    ``submit_order`` merely *records* the intent and returns a simulated
    acceptance: it NEVER touches a network or exchange. The recorded intents are
    exposed via :attr:`submitted` so tests can assert that a dry-run submits
    nothing.

    Parameters
    ----------
    equity:
        Starting account equity (currency units). Also used as cash and buying
        power for the snapshot unless positions imply otherwise.
    positions:
        Optional initial positions. Either ``{symbol: qty}`` (a price of ``0``
        and a zero market value are assumed until reconciled by the planner) or
        ``{symbol: Position}``. ``None`` means a flat book.
    """

    def __init__(
        self,
        equity: float,
        positions: dict[str, float] | dict[str, Position] | None = None,
    ) -> None:
        self._equity = float(equity)
        self._positions: dict[str, Position] = self._coerce_positions(positions)
        # Recorded order intents (dry-run asserts this stays empty).
        self.submitted: list[OrderResult] = []

    @staticmethod
    def _coerce_positions(
        positions: dict[str, float] | dict[str, Position] | None,
    ) -> dict[str, Position]:
        """Normalize the initial-positions argument into ``{sym: Position}``."""
        out: dict[str, Position] = {}
        if not positions:
            return out
        for symbol, value in positions.items():
            if isinstance(value, Position):
                out[str(symbol)] = value
            else:
                qty = float(value)
                out[str(symbol)] = Position(
                    symbol=str(symbol),
                    qty=qty,
                    avg_price=0.0,
                    market_value=0.0,
                )
        return out

    def get_account(self) -> AccountSnapshot:
        """Return the in-memory account snapshot (equity = cash = buying power)."""
        return AccountSnapshot(
            equity=self._equity,
            cash=self._equity,
            buying_power=self._equity,
        )

    def get_positions(self) -> dict[str, Position]:
        """Return a shallow copy of the in-memory positions."""
        return dict(self._positions)

    def submit_order(self, intent: OrderIntent) -> OrderResult:
        """Record ``intent`` and return a simulated acceptance (no I/O).

        Parameters
        ----------
        intent:
            The order to "submit".

        Returns
        -------
        OrderResult
            ``status='accepted_simulated'`` with ``broker_order_id=None``.
        """
        result = OrderResult(
            symbol=intent.symbol,
            side=intent.side,
            qty=intent.qty,
            status="accepted_simulated",
            broker_order_id=None,
        )
        self.submitted.append(result)
        return result


# ---------------------------------------------------------------------------
# Alpaca paper broker (lazy-import; NOT used by the dry-run)
# ---------------------------------------------------------------------------
# Environment variable names tried, in order, for the key/secret pair.
_KEY_ENV_VARS: tuple[str, ...] = ("APCA_API_KEY_ID", "ALPACA_API_KEY")
_SECRET_ENV_VARS: tuple[str, ...] = ("APCA_API_SECRET_KEY", "ALPACA_SECRET_KEY")


def _first_env(names: tuple[str, ...]) -> str | None:
    """Return the first non-empty environment value among ``names``."""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


class AlpacaBroker:
    """Adapter over the ``alpaca-py`` PAPER trading endpoint.

    The ``alpaca`` dependency is imported lazily so that *importing* this module
    never requires the package or any credentials. Credentials are read from the
    environment (``APCA_API_KEY_ID``/``APCA_API_SECRET_KEY``, falling back to
    ``ALPACA_API_KEY``/``ALPACA_SECRET_KEY``). A missing key/secret raises a
    clear :class:`RuntimeError` at construction time -- NOT at import, and never
    during a dry-run (which uses :class:`LocalPaperBroker` instead).

    Parameters
    ----------
    api_key, secret_key:
        Optional explicit credentials. When omitted they are read from the
        environment; if still absent, construction raises -- UNLESS ``client``
        is supplied (see below).
    paper:
        Use the paper endpoint (default ``True``). The dry-run milestone only
        ever targets paper.
    client:
        Optional pre-built trading client to use *directly*. When provided, the
        env-credential check and the real ``TradingClient`` construction are
        BOTH skipped, so tests can inject a mock client with no network and no
        credentials. When ``None`` (the production default) the current
        env-key + paper ``TradingClient`` behaviour is unchanged.
    """

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        *,
        paper: bool = True,
        client: Any | None = None,
    ) -> None:
        self._paper = bool(paper)

        # Injected client (tests): use it directly, skip creds + real client.
        if client is not None:
            self._client = client
            return

        key = api_key or _first_env(_KEY_ENV_VARS)
        secret = secret_key or _first_env(_SECRET_ENV_VARS)
        if not key or not secret:
            raise RuntimeError(
                "AlpacaBroker requires API credentials. Set the environment "
                "variables APCA_API_KEY_ID and APCA_API_SECRET_KEY (or "
                "ALPACA_API_KEY and ALPACA_SECRET_KEY) to your Alpaca PAPER "
                "keys, or pass api_key/secret_key explicitly. (The dry-run "
                "does NOT need these -- it uses the local paper broker.)"
            )

        # Lazy import: keeps this module import-clean without alpaca installed.
        from alpaca.trading.client import TradingClient

        self._client = TradingClient(key, secret, paper=self._paper)

    def get_account(self) -> AccountSnapshot:
        """Map the live Alpaca account into an :class:`AccountSnapshot`."""
        acct = self._client.get_account()
        return AccountSnapshot(
            equity=float(acct.equity),
            cash=float(acct.cash),
            buying_power=float(acct.buying_power),
        )

    def get_positions(self) -> dict[str, Position]:
        """Map live Alpaca positions into ``{symbol: Position}``."""
        positions: dict[str, Position] = {}
        for pos in self._client.get_all_positions():
            symbol = str(pos.symbol)
            positions[symbol] = Position(
                symbol=symbol,
                qty=float(pos.qty),
                avg_price=float(pos.avg_entry_price),
                market_value=float(pos.market_value),
            )
        return positions

    def submit_order(self, intent: OrderIntent) -> OrderResult:
        """Submit ``intent`` as an Alpaca market order on the paper endpoint.

        Parameters
        ----------
        intent:
            The order to route. ``intent.qty`` is the absolute quantity and
            ``intent.side`` selects buy/sell.

        Returns
        -------
        OrderResult
            Carrying Alpaca's status string and assigned order id.
        """
        # Lazy imports: only needed on the live path.
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        side = OrderSide.BUY if intent.side == "buy" else OrderSide.SELL
        request = MarketOrderRequest(
            symbol=intent.symbol,
            qty=abs(float(intent.qty)),
            side=side,
            time_in_force=TimeInForce.DAY,
        )
        order = self._client.submit_order(order_data=request)
        return OrderResult(
            symbol=intent.symbol,
            side=intent.side,
            qty=intent.qty,
            status=str(getattr(order, "status", "submitted")),
            broker_order_id=str(getattr(order, "id", None)),
        )

    def is_market_open(self) -> bool:
        """Return whether the U.S. equity market is currently open.

        Queries the Alpaca clock (``client.get_clock().is_open``). Clients that
        do not expose ``get_clock`` (e.g. a minimal mock or local stand-in) are
        tolerated: this returns ``True`` with a logged note so neither tests nor
        the local path break. The runner uses this only to *warn* (DAY orders
        queue for the next session when closed); it never hard-blocks.

        Returns
        -------
        bool
            ``True`` if the market is open (or the clock is unavailable),
            ``False`` if the clock reports it closed.
        """
        get_clock = getattr(self._client, "get_clock", None)
        if not callable(get_clock):
            _LOGGER.info(
                "Broker client has no get_clock(); assuming market open "
                "(cannot verify the trading clock)."
            )
            return True
        clock = get_clock()
        return bool(getattr(clock, "is_open", True))

    def recent_orders(self, limit: int = 20) -> list[OrderResult]:
        """Return the most recent orders as :class:`OrderResult` (best-effort).

        Maps the client's recent orders for a quick audit trail. This is purely
        informational and must never break the run: any unsupported client, or
        any error while fetching/mapping, yields an empty list (logged).

        Parameters
        ----------
        limit:
            Maximum number of recent orders to request.

        Returns
        -------
        list[OrderResult]
            Recent orders mapped to :class:`OrderResult`, or ``[]`` when the
            client cannot provide them.
        """
        get_orders = getattr(self._client, "get_orders", None)
        if not callable(get_orders):
            return []
        try:
            try:
                # Preferred path: a typed GetOrdersRequest filter with a limit.
                from alpaca.trading.requests import GetOrdersRequest

                orders = get_orders(filter=GetOrdersRequest(limit=int(limit)))
            except Exception:  # noqa: BLE001 - fall back for mocks/old clients
                orders = get_orders()
        except Exception as exc:  # noqa: BLE001 - audit is best-effort only
            _LOGGER.warning("Could not fetch recent orders: %s", exc)
            return []

        results: list[OrderResult] = []
        for order in list(orders or [])[: int(limit)]:
            side = str(getattr(order, "side", "")).lower()
            side = "buy" if "buy" in side else ("sell" if "sell" in side else side)
            qty_raw = getattr(order, "qty", None)
            try:
                qty = float(qty_raw) if qty_raw is not None else 0.0
            except (TypeError, ValueError):
                qty = 0.0
            results.append(
                OrderResult(
                    symbol=str(getattr(order, "symbol", "")),
                    side=side,
                    qty=qty,
                    status=str(getattr(order, "status", "")),
                    broker_order_id=str(getattr(order, "id", None)),
                )
            )
        return results
