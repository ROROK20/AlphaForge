"""
Abstract base class for all AlphaForge signals.

Every signal must implement `generate(df) -> pd.Series` returning target
positions in [-1, 1]: -1 short, 0 flat, +1 long, and a fraction for part of that.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Signal(ABC):
    """
    Abstract base class for all trading signals.

    Subclasses must implement :meth:`generate`. All signals share a common
    contract: they accept a DataFrame of OHLCV (+ optional indicator) data
    and return a Series of target positions aligned to the same index.

    Signal conventions
    ------------------
    +1  →  long
     0  →  flat / no position
    -1  →  short
    A fraction such as 0.5 holds that fraction of equity (see BacktestEngine).
    """

    name: str = "Signal"

    @abstractmethod
    def generate(self, df: pd.DataFrame) -> pd.Series:
        """
        Produce trading signals from *df*.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame. May contain pre-computed indicator columns.
            Must have a DatetimeIndex.

        Returns
        -------
        pd.Series
            Series aligned to *df*'s index with values in [-1, 1].
        """

    def __repr__(self) -> str:
        params = ", ".join(
            f"{k}={v!r}"
            for k, v in self.__dict__.items()
            if not k.startswith("_")
        )
        return f"{self.__class__.__name__}({params})"

    def _validate_df(self, df: pd.DataFrame, required: list[str] | None = None) -> None:
        """Raise if *df* is missing required columns."""
        base = ["Open", "High", "Low", "Close", "Volume"]
        needed = required or base
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(
                f"{self.__class__.__name__}: DataFrame is missing columns: {missing}"
            )

    def _fill_signals(self, signals: pd.Series, index: pd.Index) -> pd.Series:
        """Reindex to *index*, forward-fill, then fill remaining NaN with 0.

        Whole-number signals come back as integers. A fraction stays a fraction.
        """
        filled = signals.reindex(index).ffill().fillna(0)
        return filled.astype(int) if (filled == filled.round()).all() else filled.astype(float)


class SignalMetadata:
    """
    Optional metadata attached to signals for ensemble weighting.

    Attributes
    ----------
    sharpe : float | None
        In-sample Sharpe ratio of the signal.
    ic : float | None
        Information coefficient (rank correlation with forward returns).
    weight : float
        Weight in ensemble (default 1.0).
    """

    def __init__(
        self,
        sharpe: float | None = None,
        ic: float | None = None,
        weight: float = 1.0,
    ) -> None:
        self.sharpe = sharpe
        self.ic = ic
        self.weight = weight

    def __repr__(self) -> str:
        s = f"{self.sharpe:.3f}" if self.sharpe is not None else "None"
        ic = f"{self.ic:.3f}" if self.ic is not None else "None"
        return f"SignalMetadata(sharpe={s}, ic={ic}, weight={self.weight:.2f})"
