"""
Portfolio state for event-driven simulations: cash, positions and NAV.

Not used by the vectorized :class:`~alphaforge.backtest.engine.BacktestEngine`
(which computes shares and equity in array arithmetic and keeps no cash or order
ledger); this is the bookkeeping for a caller that steps through bars and trades
share quantities itself.

Accounting
----------
* A position has a size (``quantity`` > 0) and a side (``direction`` +1 long,
  -1 short); ``signed_quantity = direction * quantity``.
* Opening a LONG pays ``quantity * price`` plus commission out of cash.
  Opening a SHORT is a sale: its proceeds ``quantity * price`` are CREDITED to
  cash (less commission) and the position is a liability -- the shares are
  owed back -- marked at the current price.
* NAV is ``cash + sum(signed_quantity * price)``: a short's value is
  negative, so NAV falls as the shorted price rises.
* Closing (all or part of) a position reverses its cash flow at the exit
  price. The realized P&L it returns is net of the commission on BOTH legs
  (the entry commission is carried on the position, pro rata on a partial
  close), so a flat-to-flat round trip's realized P&L equals the exact change
  in cash.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Position:
    """An open position in a single asset."""

    symbol: str
    quantity: float          # size, > 0
    avg_entry_price: float
    direction: int           # +1 long, -1 short
    entry_commission: float = 0.0  # commission paid to open what is still held

    @property
    def signed_quantity(self) -> float:
        return self.direction * self.quantity

    @property
    def market_value(self) -> float:
        """Signed value at the entry price (cost basis); negative for a short."""
        return self.signed_quantity * self.avg_entry_price

    def value_at(self, price: float) -> float:
        """Signed value marked at *price*; negative for a short (a liability)."""
        return self.signed_quantity * price

    def unrealized_pnl(self, current_price: float) -> float:
        return self.signed_quantity * (current_price - self.avg_entry_price)


class Portfolio:
    """
    Tracks cash, positions, and computes portfolio NAV (see module docstring).

    Parameters
    ----------
    initial_capital : float
        Starting cash balance.
    """

    def __init__(self, initial_capital: float = 100_000.0) -> None:
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: dict[str, Position] = {}
        self._history: list[dict] = []

    @property
    def equity(self) -> float:
        """Cash + every position's signed value at its entry price."""
        return self.cash + sum(p.market_value for p in self.positions.values())

    def nav(self, prices: dict[str, float]) -> float:
        """Cash + every position's signed value at *prices* (entry price if absent)."""
        return self.cash + sum(
            p.value_at(prices.get(sym, p.avg_entry_price)) for sym, p in self.positions.items()
        )

    def open_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
        direction: int,
        commission: float = 0.001,
    ) -> None:
        """Open or add to a position.

        A long pays ``quantity * price * (1 + commission)``; if cash is short
        of that, the order is cut to what cash can pay for, and with no cash
        at all it is refused. Any order is refused once equity is <= 0. A short credits
        the sale proceeds ``quantity * price * (1 - commission)``. Adding on
        the side already held averages the entry price; opening the OPPOSITE
        side of an existing position raises -- close it first.
        """
        if direction not in (1, -1):
            raise ValueError(f"direction must be +1 (long) or -1 (short), got {direction!r}")
        if quantity <= 0 or price <= 0:
            raise ValueError("quantity and price must be positive")
        if self.equity <= 0:
            raise ValueError(f"account equity is {self.equity:,.2f} (<= 0): nothing to trade with")
        existing = self.positions.get(symbol)
        if existing is not None and existing.direction != direction:
            side = "long" if existing.direction == 1 else "short"
            raise ValueError(f"{symbol} is held {side}; close it before opening the other side")

        if direction == 1:
            if self.cash <= 0:
                # No partial fill to cash / price shares: that is 0 shares at zero
                # cash, and a NEGATIVE quantity after a short loss has pushed cash
                # below zero.
                raise ValueError(f"no cash to buy {symbol} with (cash {self.cash:,.2f})")
            cost = quantity * price * (1 + commission)
            if cost > self.cash:
                # Partial fill: as much as cash can pay for, commission included.
                quantity = self.cash / (price * (1 + commission))
                cost = quantity * price * (1 + commission)
            self.cash -= cost
        else:
            self.cash += quantity * price * (1 - commission)
        fee = quantity * price * commission

        if existing is not None:
            total_qty = existing.quantity + quantity
            avg_price = (existing.quantity * existing.avg_entry_price + quantity * price) / total_qty
            self.positions[symbol] = Position(
                symbol, total_qty, avg_price, direction, existing.entry_commission + fee
            )
        else:
            self.positions[symbol] = Position(symbol, quantity, price, direction, fee)

    def close_position(
        self,
        symbol: str,
        price: float,
        commission: float = 0.001,
        quantity: float | None = None,
    ) -> float:
        """Close all (default) or *quantity* of a position; return realized P&L.

        The P&L is the price move on the closed size, net of the exit
        commission and of the closed share of the entry commission. A long's
        close credits ``q * price * (1 - commission)``; a short's buys the
        shares back for ``q * price * (1 + commission)``.
        """
        pos = self.positions.get(symbol)
        if pos is None:
            return 0.0
        q = pos.quantity if quantity is None else float(quantity)
        if not 0 < q <= pos.quantity:
            raise ValueError(f"close quantity must be in (0, {pos.quantity}], got {quantity!r}")

        exit_fee = q * price * commission
        entry_fee = pos.entry_commission * (q / pos.quantity)
        if pos.direction == 1:
            self.cash += q * price - exit_fee
        else:
            self.cash -= q * price + exit_fee
        pnl = pos.direction * q * (price - pos.avg_entry_price) - entry_fee - exit_fee

        if q == pos.quantity:
            del self.positions[symbol]
        else:
            self.positions[symbol] = Position(
                symbol, pos.quantity - q, pos.avg_entry_price, pos.direction,
                pos.entry_commission - entry_fee,
            )
        return float(pnl)

    def snapshot(self, date: pd.Timestamp, prices: dict[str, float]) -> dict:
        """Record portfolio state at a given date (positions marked at *prices*)."""
        market_value = sum(
            p.value_at(prices.get(sym, p.avg_entry_price)) for sym, p in self.positions.items()
        )
        nav = self.cash + market_value
        snap = {
            "date": date,
            "cash": self.cash,
            "market_value": market_value,
            "nav": nav,
            "n_positions": len(self.positions),
        }
        self._history.append(snap)
        return snap

    def equity_curve(self) -> pd.Series:
        """Return NAV history as a Series indexed by date."""
        if not self._history:
            return pd.Series(dtype=float)
        df = pd.DataFrame(self._history).set_index("date")
        return df["nav"]

    def reset(self) -> None:
        """Reset portfolio to initial state."""
        self.cash = self.initial_capital
        self.positions.clear()
        self._history.clear()

    def __repr__(self) -> str:
        return (
            f"Portfolio(cash={self.cash:,.2f}, "
            f"positions={list(self.positions.keys())}, "
            f"equity={self.equity:,.2f})"
        )
