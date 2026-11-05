"""
SMA Crossover Strategy.

Classic trend-following strategy: go long when fast SMA crosses above
slow SMA, go short (or flat) when it crosses below.
"""

from __future__ import annotations

import pandas as pd

from alphaforge.signals.technical import SMACrossoverSignal
from alphaforge.backtest.engine import BacktestEngine, BacktestResult
from alphaforge.data.fetcher import DataFetcher


class SMACrossoverStrategy:
    """
    Simple Moving Average Crossover Strategy.

    Parameters
    ----------
    fast : int
        Fast SMA window (default 20).
    slow : int
        Slow SMA window (default 50).
    initial_capital : float
        Starting capital (default 100_000).
    commission : float
        Commission fraction per trade (default 0.001).
    long_only : bool
        If True, only take long positions (default True).

    Examples
    --------
    >>> strategy = SMACrossoverStrategy(fast=20, slow=50)
    >>> result = strategy.run("AAPL", "2020-01-01", "2023-12-31")
    >>> print(result.metrics)
    """

    NAME = "SMA Crossover"
    DESCRIPTION = (
        "Trend-following strategy using dual SMA crossover. "
        "Generates buy signals when the fast SMA crosses above the slow SMA "
        "and sell signals when it crosses below."
    )

    def __init__(
        self,
        fast: int = 20,
        slow: int = 50,
        initial_capital: float = 100_000.0,
        commission: float = 0.001,
        long_only: bool = True,
    ) -> None:
        self.fast = fast
        self.slow = slow
        self.initial_capital = initial_capital
        self.commission = commission
        self.long_only = long_only
        self.signal = SMACrossoverSignal(fast=fast, slow=slow, long_only=long_only)
        self.engine = BacktestEngine(
            initial_capital=initial_capital,
            commission=commission,
            long_only=long_only,
        )

    def run(
        self,
        symbol: str,
        start: str,
        end: str | None = None,
    ) -> BacktestResult:
        """
        Fetch data and run backtest.

        Parameters
        ----------
        symbol : str
            Ticker symbol.
        start : str
            Start date (YYYY-MM-DD).
        end : str | None
            End date. Defaults to today.

        Returns
        -------
        BacktestResult
        """
        fetcher = DataFetcher()
        df = fetcher.fetch_ohlcv(symbol, start, end)
        return self.engine.run(self.signal, df)

    def run_on_data(self, df: pd.DataFrame) -> BacktestResult:
        """Run backtest on pre-loaded OHLCV DataFrame."""
        return self.engine.run(self.signal, df)

    def __repr__(self) -> str:
        return f"SMACrossoverStrategy(fast={self.fast}, slow={self.slow})"
