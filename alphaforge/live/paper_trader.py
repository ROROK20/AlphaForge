"""
A client for placing orders by hand on Alpaca's paper API.

Not a trading loop: nothing in AlphaForge connects a signal to an order. The
tests drive it against a fake client; it has never run against the real SDK
(see SECURITY.md).

Reads these settings from the environment or, for any not set there, from a
``.env`` file in the current directory or its nearest parent that has one:
  ALPACA_API_KEY    — from https://alpaca.markets
  ALPACA_SECRET_KEY
  ALPACA_BASE_URL   — optional, defaults to paper trading URL; used only when
                      no ``base_url`` is passed, and must be the paper
                      endpoint unless ``allow_live=True``

Usage:
    trader = PaperTrader()
    trader.submit_order("AAPL", qty=10, side="buy")
    print(trader.get_account())
    print(trader.get_positions())
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from dotenv import dotenv_values, find_dotenv

logger = logging.getLogger(__name__)

_PAPER_URL = "https://paper-api.alpaca.markets"


def _is_paper_endpoint(url: str) -> bool:
    """True only for https://paper-api.alpaca.markets (any path)."""
    parsed, paper = urlparse(url), urlparse(_PAPER_URL)
    return parsed.scheme == paper.scheme and parsed.hostname == paper.hostname


@dataclass
class Order:
    """Minimal order representation."""
    symbol: str
    qty: float
    side: str  # "buy" | "sell"
    order_type: str  # "market" | "limit"
    status: str
    order_id: str
    filled_price: float | None = None


@dataclass
class Position:
    symbol: str
    qty: float
    avg_entry_price: float
    market_value: float
    unrealized_pl: float
    unrealized_plpc: float


class PaperTrader:
    """
    A client for placing orders by hand on Alpaca's paper trading API.

    Parameters
    ----------
    api_key : str | None
        Alpaca API key. Falls back to ALPACA_API_KEY (environment, then ``.env``).
    secret_key : str | None
        Alpaca secret key. Falls back to ALPACA_SECRET_KEY (environment, then ``.env``).
    base_url : str | None
        API base URL. An explicit argument always wins; if omitted,
        ``ALPACA_BASE_URL`` (environment, then ``.env``) is used, then the paper
        endpoint.
    allow_live : bool
        A ``PaperTrader`` refuses any endpoint other than Alpaca's paper API
        (``https://paper-api.alpaca.markets``) unless this is True, because
        anything else can place real-money orders.

    An explicit ``base_url`` beats the environment variable, so
    ``PaperTrader(base_url=<paper URL>)`` stays on paper on a machine whose
    environment has ``ALPACA_BASE_URL`` set to the live endpoint.
    """

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        base_url: str | None = None,
        allow_live: bool = False,
    ) -> None:
        # Read, not loaded: a constructor should not export a file's other
        # variables into the process.
        dotenv = dotenv_values(find_dotenv(usecwd=True))

        def setting(name: str) -> str:
            return os.getenv(name) or dotenv.get(name) or ""

        self.api_key = api_key or setting("ALPACA_API_KEY")
        self.secret_key = secret_key or setting("ALPACA_SECRET_KEY")
        self.base_url = (base_url or setting("ALPACA_BASE_URL") or _PAPER_URL).rstrip("/")
        if not allow_live and not _is_paper_endpoint(self.base_url):
            raise ValueError(
                f"PaperTrader refuses non-paper endpoint {self.base_url!r}: it would "
                f"trade real money. Use {_PAPER_URL!r}, or pass allow_live=True to "
                "connect to another endpoint deliberately."
            )
        self._api = self._init_api()

    def _init_api(self):
        """Initialize Alpaca API client."""
        try:
            import alpaca_trade_api as tradeapi  # type: ignore
            return tradeapi.REST(
                key_id=self.api_key,
                secret_key=self.secret_key,
                base_url=self.base_url,
            )
        except ImportError:
            logger.warning(
                "alpaca-trade-api not installed. Run: pip install alpaca-trade-api. "
                "Paper trading disabled."
            )
            return None

    def is_connected(self) -> bool:
        """Return True if Alpaca API is reachable."""
        if self._api is None:
            return False
        try:
            self._api.get_account()
            return True
        except Exception as e:
            logger.error(f"Alpaca connection failed: {e}")
            return False

    def get_account(self) -> dict[str, Any]:
        """Return account details: cash, equity, buying_power."""
        if self._api is None:
            return {}
        account = self._api.get_account()
        return {
            "equity": float(account.equity),
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
            "portfolio_value": float(account.portfolio_value),
            "status": account.status,
        }

    def get_positions(self) -> list[Position]:
        """Return all open positions."""
        if self._api is None:
            return []
        raw_positions = self._api.list_positions()
        return [
            Position(
                symbol=p.symbol,
                qty=float(p.qty),
                avg_entry_price=float(p.avg_entry_price),
                market_value=float(p.market_value),
                unrealized_pl=float(p.unrealized_pl),
                unrealized_plpc=float(p.unrealized_plpc),
            )
            for p in raw_positions
        ]

    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        limit_price: float | None = None,
        time_in_force: str = "gtc",
    ) -> Order | None:
        """
        Submit a paper trading order.

        Parameters
        ----------
        symbol : str
            Ticker symbol (e.g. 'AAPL').
        qty : float
            Number of shares.
        side : str
            'buy' or 'sell'.
        order_type : str
            'market' or 'limit'.
        limit_price : float | None
            Required for limit orders.
        time_in_force : str
            'gtc' (good-till-cancelled), 'day', 'ioc', etc.

        Returns
        -------
        Order | None
        """
        if self._api is None:
            logger.warning("Alpaca API not initialized. Order not submitted.")
            return None
        try:
            kwargs: dict[str, Any] = {
                "symbol": symbol,
                "qty": qty,
                "side": side,
                "type": order_type,
                "time_in_force": time_in_force,
            }
            if order_type == "limit" and limit_price is not None:
                kwargs["limit_price"] = str(limit_price)

            raw = self._api.submit_order(**kwargs)
            logger.info(f"Order submitted: {side.upper()} {qty} {symbol} @ {order_type}")
            return Order(
                symbol=symbol,
                qty=qty,
                side=side,
                order_type=order_type,
                status=raw.status,
                order_id=raw.id,
            )
        except Exception as e:
            logger.error(f"Order submission failed for {symbol}: {e}")
            return None

    def close_position(self, symbol: str) -> bool:
        """Close all shares of a position."""
        if self._api is None:
            return False
        try:
            self._api.close_position(symbol)
            logger.info(f"Closed position: {symbol}")
            return True
        except Exception as e:
            logger.error(f"Failed to close {symbol}: {e}")
            return False

    def close_all_positions(self) -> int:
        """Close all open positions. Returns number of positions closed."""
        if self._api is None:
            return 0
        try:
            # Count BEFORE closing: afterwards there is usually nothing left to count.
            n = len(self.get_positions())
            self._api.close_all_positions()
            logger.info(f"Closed all {n} positions")
            return n
        except Exception as e:
            logger.error(f"Failed to close all positions: {e}")
            return 0

    def get_orders(self, status: str = "open") -> list[dict]:
        """List orders by status: 'open', 'closed', 'all'."""
        if self._api is None:
            return []
        try:
            orders = self._api.list_orders(status=status)
            return [
                {
                    "id": o.id,
                    "symbol": o.symbol,
                    "qty": float(o.qty),
                    "side": o.side,
                    "type": o.order_type,
                    "status": o.status,
                    "submitted_at": str(o.submitted_at),
                }
                for o in orders
            ]
        except Exception as e:
            logger.error(f"Failed to fetch orders: {e}")
            return []

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        if self._api is None:
            return False
        try:
            self._api.cancel_order(order_id)
            return True
        except Exception as e:
            logger.error(f"Cancel order {order_id} failed: {e}")
            return False

    def get_portfolio_history(self, period: str = "1M", timeframe: str = "1D") -> dict:
        """
        Fetch portfolio equity history.

        Parameters
        ----------
        period : str
            Time period: '1D', '1W', '1M', '3M', '6M', '1A'.
        timeframe : str
            Bar timeframe: '1Min', '5Min', '1H', '1D'.
        """
        if self._api is None:
            return {}
        try:
            history = self._api.get_portfolio_history(period=period, timeframe=timeframe)
            return {
                "timestamps": history.timestamp,
                "equity": history.equity,
                "profit_loss": history.profit_loss,
                "profit_loss_pct": history.profit_loss_pct,
            }
        except Exception as e:
            logger.error(f"Portfolio history fetch failed: {e}")
            return {}

    def __repr__(self) -> str:
        connected = "connected" if self.is_connected() else "disconnected"
        return f"PaperTrader(url={self.base_url!r}, status={connected!r})"
