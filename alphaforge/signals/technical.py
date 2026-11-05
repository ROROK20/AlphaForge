"""
Technical analysis signals.

All signals inherit from :class:`alphaforge.signals.base.Signal` and return
integer series in {-1, 0, +1}.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.data.preprocessor import _rsi
from alphaforge.signals.base import Signal


class SMACrossoverSignal(Signal):
    """
    Simple Moving Average regime signal (trend-following).

    Holds +1 for every bar the fast SMA is above the slow SMA and -1 for
    every bar it is below (0 only while either average is warming up or the
    two are exactly equal). This is a persistent state signal — the position
    is held for the whole regime, not just on the crossover bar.

    Parameters
    ----------
    fast : int
        Fast SMA window (default 20).
    slow : int
        Slow SMA window (default 50).
    long_only : bool
        If True, never generates -1 signals (default False).
    """

    name = "SMA Crossover"

    def __init__(self, fast: int = 20, slow: int = 50, long_only: bool = False) -> None:
        if fast >= slow:
            raise ValueError(f"fast ({fast}) must be less than slow ({slow}).")
        self.fast = fast
        self.slow = slow
        self.long_only = long_only

    def generate(self, df: pd.DataFrame) -> pd.Series:
        self._validate_df(df)

        fast_col = f"SMA_{self.fast}"
        slow_col = f"SMA_{self.slow}"

        # Compute on the fly if not already present
        close = df["Close"]
        fast_sma = close.rolling(self.fast).mean() if fast_col not in df.columns else df[fast_col]
        slow_sma = close.rolling(self.slow).mean() if slow_col not in df.columns else df[slow_col]

        # Position = +1 when fast > slow, -1 when fast < slow
        raw = np.where(fast_sma > slow_sma, 1, np.where(fast_sma < slow_sma, -1, 0))
        signals = pd.Series(raw, index=df.index, name=self.name)

        if self.long_only:
            signals = signals.clip(lower=0)

        return signals.fillna(0).astype(int)


class RSIMeanReversionSignal(Signal):
    """
    RSI-based mean reversion signal.

    Buys when RSI is oversold (< *oversold*), sells when overbought (> *overbought*),
    and exits the position when RSI returns to neutral territory [40, 60].

    Parameters
    ----------
    period : int
        RSI lookback period (default 14).
    overbought : float
        RSI level above which the asset is considered overbought (default 70).
    oversold : float
        RSI level below which the asset is considered oversold (default 30).
    """

    name = "RSI Mean Reversion"

    def __init__(
        self,
        period: int = 14,
        overbought: float = 70.0,
        oversold: float = 30.0,
    ) -> None:
        self.period = period
        self.overbought = overbought
        self.oversold = oversold

    def generate(self, df: pd.DataFrame) -> pd.Series:
        self._validate_df(df)

        # Always computed from Close with THIS instance's period (Wilder
        # smoothing, as add_technical_features uses for RSI_14, the only RSI
        # column that function writes).
        rsi = _rsi(df["Close"].astype(float), self.period)

        # Regime-based signal: once triggered, hold until exit zone
        signals = pd.Series(0, index=df.index, dtype=int, name=self.name)
        position = 0

        # Compute a stable mid-zone that does not overlap the entry thresholds
        mid_lo = max(40.0, self.oversold + 5.0)
        mid_hi = min(60.0, self.overbought - 5.0)

        for i, (idx, val) in enumerate(rsi.items()):
            if pd.isna(val):
                signals.iloc[i] = 0
                continue
            if val <= self.oversold:
                position = 1
            elif val >= self.overbought:
                position = -1
            elif mid_lo <= val <= mid_hi:
                position = 0
            signals.iloc[i] = position

        return signals


class MACDSignal(Signal):
    """
    MACD signal-line regime (stop-and-reverse on crosses).

    Goes long (+1) on a bullish cross -- the MACD line crossing above its
    signal line -- and HOLDS that long until the next bearish cross, where it
    reverses to short (-1) and holds until the next bullish cross. It is flat
    (0) only before the first cross after the warm-up.

    It holds between crosses because the engine treats a signal as a position: a
    signal that is +/-1 on the crossover bar alone would hold every trade for one
    bar.

    Warm-up: crosses in the first ``(slow - 1) + (signal - 1)`` bars are
    ignored -- the conventional MACD lookback (as in TA-Lib), during which the
    EMAs are still dominated by their seed value.

    The lines are always computed from ``Close`` with THIS instance's parameters,
    never from pre-computed ``MACD`` / ``MACD_signal`` columns, which
    ``add_technical_features`` writes at a fixed 12/26/9.

    Parameters
    ----------
    fast : int
        Fast EMA period (default 12).
    slow : int
        Slow EMA period (default 26).
    signal : int
        Signal line EMA period (default 9).
    """

    name = "MACD Crossover"

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        if fast >= slow:
            raise ValueError(f"fast ({fast}) must be less than slow ({slow}).")
        self.fast = fast
        self.slow = slow
        self.signal_period = signal

    @property
    def warmup(self) -> int:
        """Bars at the start of a series on which no cross is acted on."""
        return (self.slow - 1) + (self.signal_period - 1)

    def generate(self, df: pd.DataFrame) -> pd.Series:
        self._validate_df(df)

        close = df["Close"].astype(float)
        ema_fast = close.ewm(span=self.fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=self.signal_period, adjust=False).mean()

        # Histogram > 0 = MACD above its signal line (bullish).
        hist = macd - macd_signal
        prev_hist = hist.shift(1)
        bullish_cross = (hist > 0) & (prev_hist <= 0)
        bearish_cross = (hist < 0) & (prev_hist >= 0)

        # Crosses are entry/reversal EVENTS; the position is the most recent
        # event, carried forward until the next one (a regime, not a pulse).
        events = pd.Series(np.nan, index=df.index)
        events[bullish_cross] = 1.0
        events[bearish_cross] = -1.0
        events.iloc[: self.warmup] = np.nan

        signals = events.ffill().fillna(0).astype(int)
        signals.name = self.name
        return signals


class BollingerBandSignal(Signal):
    """
    Bollinger Band mean-reversion signal.

    Buys when price touches the lower band, sells when price touches the upper band.
    Exits when price returns to the midline (20-period SMA).

    Parameters
    ----------
    period : int
        Lookback period for SMA and standard deviation (default 20).
    n_std : float
        Number of standard deviations for bands (default 2.0).
    """

    name = "Bollinger Bands"

    def __init__(self, period: int = 20, n_std: float = 2.0) -> None:
        self.period = period
        self.n_std = n_std

    def generate(self, df: pd.DataFrame) -> pd.Series:
        self._validate_df(df)

        # Bands are always computed with THIS instance's period and width, never
        # from pre-computed ``BB_*`` columns, which add_technical_features writes at
        # a fixed 20-period / 2-sigma.
        close = df["Close"].astype(float)
        mid = close.rolling(self.period).mean()
        std = close.rolling(self.period).std(ddof=0)
        upper = mid + self.n_std * std
        lower = mid - self.n_std * std

        # Percent bandwidth: where price is within the bands
        pct_b = (close - lower) / (upper - lower)

        # Signal: below lower band → long; above upper band → short; mid → flat
        raw = np.where(
            pct_b < 0,  1,   # price below lower band → buy
            np.where(
                pct_b > 1, -1,   # price above upper band → sell
                np.where(
                    (pct_b > 0.4) & (pct_b < 0.6), 0,  # near mid → exit
                    np.nan,
                )
            )
        )

        signals = pd.Series(raw, index=df.index, name=self.name)
        # Forward fill between signals, default to 0
        signals = signals.ffill().fillna(0).astype(int)

        return signals
