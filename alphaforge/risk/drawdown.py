"""
Drawdown monitoring: reports when a drawdown or daily-loss limit is breached.
"""

from __future__ import annotations

from collections import deque

import pandas as pd


class DrawdownMonitor:
    """
    Tracks rolling drawdown and reports when a limit is breached.

    Parameters
    ----------
    max_drawdown_pct : float
        Maximum allowed drawdown from peak before halting trading, as a
        NEGATIVE fraction (default -0.20 = a 20% drawdown).
    daily_loss_limit : float
        Maximum allowed single-day loss before halting, as a NEGATIVE
        fraction (default -0.05 = a 5% loss).
    lookback_window : int | None
        None (default): drawdown is measured from the all-time peak. An
        integer N measures it from the highest value among the last N
        updates (a rolling peak).

    Both thresholds must be negative: a positive one such as
    ``max_drawdown_pct=0.15`` would halt on the very first update (every drawdown
    is <= 0.15).

    It only reports: ``update`` returns False and sets ``halt_reason``, and acting
    on that is up to the caller. No backtest consults it.
    """

    def __init__(
        self,
        max_drawdown_pct: float = -0.20,
        daily_loss_limit: float = -0.05,
        lookback_window: int | None = None,
    ) -> None:
        for name, value in (("max_drawdown_pct", max_drawdown_pct), ("daily_loss_limit", daily_loss_limit)):
            if not value < 0:
                raise ValueError(
                    f"{name} must be negative (a loss), e.g. -0.15 for 15%; got {value!r}"
                )
        if lookback_window is not None and lookback_window < 1:
            raise ValueError(f"lookback_window must be None or >= 1, got {lookback_window!r}")
        self.max_drawdown_pct = max_drawdown_pct
        self.daily_loss_limit = daily_loss_limit
        self.lookback_window = lookback_window
        self._recent: deque[float] = deque(maxlen=lookback_window)
        self._peak = float("-inf")
        self._halted = False
        self._halt_reason: str = ""

    @property
    def halted(self) -> bool:
        return self._halted

    @property
    def halt_reason(self) -> str:
        return self._halt_reason

    def update(self, current_value: float, daily_return: float) -> bool:
        """
        Update monitor with latest portfolio value.

        Parameters
        ----------
        current_value : float
            Current portfolio NAV.
        daily_return : float
            Today's return (fractional).

        Returns
        -------
        bool
            True if trading should continue, False if a limit was breached.
        """
        if self._halted:
            return False

        # Update peak: all-time, or the highest of the last N values.
        if self.lookback_window is None:
            self._peak = max(self._peak, current_value)
        else:
            self._recent.append(current_value)
            self._peak = max(self._recent)

        # Check drawdown from peak
        drawdown = (current_value - self._peak) / self._peak if self._peak > 0 else 0.0
        if drawdown <= self.max_drawdown_pct:
            self._halted = True
            self._halt_reason = (
                f"Max drawdown exceeded: {drawdown:.2%} <= {self.max_drawdown_pct:.2%}"
            )
            return False

        # Check daily loss limit
        if daily_return <= self.daily_loss_limit:
            self._halted = True
            self._halt_reason = (
                f"Daily loss limit exceeded: {daily_return:.2%} <= {self.daily_loss_limit:.2%}"
            )
            return False

        return True

    def reset(self) -> None:
        """Reset the monitor (e.g. after a cool-down period)."""
        self._peak = float("-inf")
        self._recent.clear()
        self._halted = False
        self._halt_reason = ""

    def __repr__(self) -> str:
        return (
            f"DrawdownMonitor("
            f"max_dd={self.max_drawdown_pct:.2%}, "
            f"daily_limit={self.daily_loss_limit:.2%}, "
            f"halted={self._halted})"
        )


def compute_drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """Drawdown as fraction from rolling peak. Returns series of <= 0 values."""
    peak = equity_curve.cummax()
    return (equity_curve - peak) / peak


def underwater_periods(equity_curve: pd.Series) -> pd.DataFrame:
    """
    Return a DataFrame of all drawdown periods with start, end, depth.

    Parameters
    ----------
    equity_curve : pd.Series
        NAV or price series.

    Returns
    -------
    pd.DataFrame with columns: start, end, depth (most negative drawdown), duration_days.
    """
    dd = compute_drawdown_series(equity_curve)
    in_dd = dd < 0

    periods = []
    start = None
    for i, (date, val) in enumerate(in_dd.items()):
        if val and start is None:
            start = date
        elif not val and start is not None:
            window = dd.loc[start:date]
            depth = float(window.min())
            duration = (date - start).days if hasattr(date - start, "days") else i
            periods.append({"start": start, "end": date, "depth": depth, "duration_days": duration})
            start = None

    if start is not None:
        # Still in drawdown at end of series
        window = dd.loc[start:]
        depth = float(window.min())
        end = dd.index[-1]
        duration = (end - start).days if hasattr(end - start, "days") else len(window)
        periods.append({"start": start, "end": end, "depth": depth, "duration_days": duration})

    if not periods:
        return pd.DataFrame(columns=["start", "end", "depth", "duration_days"])

    return pd.DataFrame(periods).sort_values("depth").reset_index(drop=True)
