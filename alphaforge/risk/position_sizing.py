"""
Position sizing algorithms.

Each sizer returns a fraction of capital to deploy in a single trade.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import pandas as pd

from alphaforge.backtest.metrics import infer_periods_per_year


class PositionSizer(ABC):
    """Abstract base for all position sizers."""

    @abstractmethod
    def size(self, capital: float, **kwargs) -> float:
        """Return fraction of capital to risk on one trade (0.0 – 1.0)."""


class KellyCriterion(PositionSizer):
    """
    Kelly Criterion position sizing.

    Full Kelly fraction = win_rate - (1 - win_rate) / (avg_win / avg_loss)
    Fractional Kelly = fraction * full_kelly (typically 0.25 to reduce variance).

    Parameters
    ----------
    win_rate : float
        Historical probability of a winning trade.
    avg_win : float
        Average win return (e.g. 0.05 for 5%).
    avg_loss : float
        Average loss return (positive magnitude, e.g. 0.03 for 3%).
    fraction : float
        Fractional Kelly multiplier (default 0.25 = quarter-Kelly).
    max_position : float
        Cap on position size fraction (default 0.20 = 20% of capital).
    """

    def __init__(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        fraction: float = 0.25,
        max_position: float = 0.20,
    ) -> None:
        if not 0 < win_rate < 1:
            raise ValueError(f"win_rate must be in (0, 1), got {win_rate}")
        self.win_rate = win_rate
        self.avg_win = avg_win
        self.avg_loss = max(avg_loss, 1e-9)
        self.fraction = fraction
        self.max_position = max_position
        self._full_kelly = self._compute_kelly()

    def _compute_kelly(self) -> float:
        """f* = (b*p - q) / b, where b = avg_win/avg_loss, p = win_rate, q = 1-p."""
        b = self.avg_win / self.avg_loss
        p = self.win_rate
        q = 1 - p
        full_kelly = (b * p - q) / b
        return max(0.0, full_kelly)  # never negative

    def size(self, capital: float, **kwargs) -> float:
        """Return fractional Kelly position size."""
        f = self.fraction * self._full_kelly
        return float(min(f, self.max_position))

    @property
    def full_kelly(self) -> float:
        return self._full_kelly

    def __repr__(self) -> str:
        return (
            f"KellyCriterion(win_rate={self.win_rate:.2%}, "
            f"avg_win={self.avg_win:.2%}, avg_loss={self.avg_loss:.2%}, "
            f"fraction={self.fraction}, full_kelly={self._full_kelly:.2%})"
        )


class FixedFractional(PositionSizer):
    """
    Fixed fractional (risk-based) position sizing.

    Sizes each trade so that the risk (distance to stop-loss) equals
    a fixed percentage of total capital.

    Parameters
    ----------
    risk_per_trade : float
        Maximum fraction of capital to risk per trade (default 0.02 = 2%).
    """

    def __init__(self, risk_per_trade: float = 0.02) -> None:
        if not 0 < risk_per_trade < 1:
            raise ValueError(f"risk_per_trade must be in (0, 1), got {risk_per_trade}")
        self.risk_per_trade = risk_per_trade

    def size(self, capital: float, stop_distance: float = 0.05, **kwargs) -> float:
        """
        Return fraction of capital to allocate.

        Parameters
        ----------
        capital : float
            Current portfolio value.
        stop_distance : float
            Fraction of entry price to place stop-loss (e.g. 0.05 = 5% stop).
        """
        if stop_distance <= 0:
            return 0.0
        # shares * entry * stop_distance = capital * risk_per_trade
        # → position_fraction = risk_per_trade / stop_distance
        position_fraction = self.risk_per_trade / stop_distance
        return float(min(position_fraction, 1.0))

    def __repr__(self) -> str:
        return f"FixedFractional(risk_per_trade={self.risk_per_trade:.2%})"


class EqualWeight(PositionSizer):
    """
    Equal-weight position sizing across N assets.

    Parameters
    ----------
    n_positions : int
        Number of concurrent positions to size for.
    """

    def __init__(self, n_positions: int = 10) -> None:
        if n_positions < 1:
            raise ValueError("n_positions must be >= 1")
        self.n_positions = n_positions

    def size(self, capital: float, **kwargs) -> float:
        """Return 1/n_positions as position size."""
        return 1.0 / self.n_positions

    def __repr__(self) -> str:
        return f"EqualWeight(n_positions={self.n_positions})"


class VolatilityParity(PositionSizer):
    """
    Volatility-parity (inverse-volatility) position sizing.

    Each asset receives a position inversely proportional to its
    recent realized volatility, so each contributes equal risk.

    Parameters
    ----------
    target_volatility : float
        Target portfolio annualized volatility (default 0.15 = 15%).
    lookback : int
        Rolling window for volatility estimate in bars (default 20).
    periods_per_year : float | None
        Bars per year, to annualise that estimate. None reads it from the dates
        of the returns (see ``alphaforge.backtest.metrics``).
    """

    def __init__(
        self, target_volatility: float = 0.15, lookback: int = 20, periods_per_year: float | None = None
    ) -> None:
        self.target_volatility = target_volatility
        self.lookback = lookback
        self.periods_per_year = periods_per_year

    def size(self, capital: float, returns: pd.Series | None = None, **kwargs) -> float:
        """
        Compute position size from recent volatility.

        Parameters
        ----------
        capital : float
            Portfolio value (unused, sizing is in fraction form).
        returns : pd.Series | None
            Recent returns to estimate vol. If None, defaults to 1/10.
        """
        if returns is None or len(returns) < 2:
            return 0.10  # fallback
        recent = returns.iloc[-self.lookback :]
        periods = infer_periods_per_year(returns.index) if self.periods_per_year is None else self.periods_per_year
        realized_vol = float(recent.std() * math.sqrt(periods))
        if realized_vol < 1e-9:
            return 0.10
        return float(min(self.target_volatility / realized_vol, 1.0))

    def __repr__(self) -> str:
        return (
            f"VolatilityParity(target_vol={self.target_volatility:.2%}, "
            f"lookback={self.lookback})"
        )
