"""
Data preprocessing and feature engineering.

All functions are pure transformations on a pandas DataFrame containing
at minimum columns: Open, High, Low, Close, Volume.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Technical feature creation
# ---------------------------------------------------------------------------


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a comprehensive set of technical indicators to *df*.

    Indicators added
    ----------------
    SMA_20, SMA_50          Simple moving averages
    EMA_12, EMA_26          Exponential moving averages
    RSI_14                  Relative Strength Index
    MACD, MACD_signal       MACD line and signal line
    MACD_hist               MACD histogram
    BB_upper, BB_mid, BB_lower  Bollinger Bands (20-period, 2-sigma)
    ATR_14                  Average True Range
    volume_sma              20-period SMA of Volume
    OBV                     On-Balance Volume
    momentum_10             10-day price momentum
    returns                 Daily log returns

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: Open, High, Low, Close, Volume.

    Returns
    -------
    pd.DataFrame
        Original DataFrame plus new indicator columns. First rows may be NaN
        due to look-back periods.
    """
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # Moving averages
    df["SMA_20"] = close.rolling(20).mean()
    df["SMA_50"] = close.rolling(50).mean()
    df["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA_26"] = close.ewm(span=26, adjust=False).mean()

    # RSI
    df["RSI_14"] = _rsi(close, 14)

    # MACD
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    # Bollinger Bands
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std(ddof=0)
    df["BB_mid"] = bb_mid
    df["BB_upper"] = bb_mid + 2 * bb_std
    df["BB_lower"] = bb_mid - 2 * bb_std
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["BB_mid"]
    df["BB_pct"] = (close - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"])

    # ATR
    df["ATR_14"] = _atr(high, low, close, 14)

    # Volume
    df["volume_sma"] = volume.rolling(20).mean()
    df["volume_ratio"] = volume / df["volume_sma"]

    # OBV
    df["OBV"] = _obv(close, volume)

    # Momentum
    df["momentum_10"] = close / close.shift(10) - 1

    # Returns
    df["returns"] = close.pct_change()
    df["log_returns"] = np.log(close / close.shift(1))

    return df


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder-smoothed RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    # Guard against divide-by-zero: when there are no losses (monotone rise)
    # RSI should saturate at 100; when there are no gains it should be 0.
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss > 0, other=100.0)
    rsi = rsi.where(avg_gain > 0, other=0.0)
    # Warm-up: RSI over `period` needs `period` price changes, which the first
    # `period` bars do not have; they are undefined (NaN), not 0 or 100. Bar 0's NaN
    # change would otherwise fall through both guards above as RSI = 0 on EVERY
    # series, a fake "oversold" reading that puts RSIMeanReversionSignal long from
    # the very first bar.
    rsi.iloc[:period] = np.nan
    return rsi


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range using Wilder smoothing."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def normalize(
    df: pd.DataFrame,
    method: str = "zscore",
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Normalise numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
    method : str
        "zscore" (zero mean, unit variance) or "minmax" (0-1 range).
    columns : list[str] | None
        Columns to normalise. If None, normalises all numeric columns
        except OHLCV fundamentals.

    Returns
    -------
    pd.DataFrame
        Normalised DataFrame (copy).

    Warning
    -------
    The statistics (mean / std, or min / max) are computed over the WHOLE
    frame, so every row is scaled with information from later rows. Fine for
    exploration; for a backtest or a train/test split, normalise the training
    rows with their own statistics and apply those to later rows, or the
    features leak the future.
    """
    df = df.copy()
    _exclude = {"Open", "High", "Low", "Close", "Volume"}

    if columns is None:
        columns = [c for c in df.select_dtypes(include=[np.number]).columns if c not in _exclude]

    if method == "zscore":
        for col in columns:
            mu = df[col].mean()
            sigma = df[col].std(ddof=1)
            if sigma > 0:
                df[col] = (df[col] - mu) / sigma
            else:
                df[col] = 0.0
    elif method == "minmax":
        for col in columns:
            col_min = df[col].min()
            col_max = df[col].max()
            span = col_max - col_min
            if span > 0:
                df[col] = (df[col] - col_min) / span
            else:
                df[col] = 0.0
    else:
        raise ValueError(f"Unknown normalisation method {method!r}. Use 'zscore' or 'minmax'.")

    return df


# ---------------------------------------------------------------------------
# Lagged features for ML
# ---------------------------------------------------------------------------


def create_lagged_features(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    n_lags: int = 5,
) -> pd.DataFrame:
    """
    Create lagged versions of *columns* for use as ML input features.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list[str] | None
        Columns to lag. If None, lags: Close, Volume, returns, RSI_14,
        MACD, ATR_14 (if present).
    n_lags : int
        Number of lags to create (default 5).

    Returns
    -------
    pd.DataFrame
        DataFrame with additional columns named ``{col}_lag_{k}``
        for k in 1..n_lags.
    """
    df = df.copy()

    if columns is None:
        _defaults = ["Close", "Volume", "returns", "RSI_14", "MACD", "ATR_14"]
        columns = [c for c in _defaults if c in df.columns]

    for col in columns:
        for k in range(1, n_lags + 1):
            df[f"{col}_lag_{k}"] = df[col].shift(k)

    return df


# ---------------------------------------------------------------------------
# Forward return target
# ---------------------------------------------------------------------------


def create_target(
    df: pd.DataFrame,
    horizon: int = 5,
    column: str = "Close",
    method: str = "returns",
) -> pd.DataFrame:
    """
    Add a forward-looking target column for supervised ML.

    Parameters
    ----------
    df : pd.DataFrame
    horizon : int
        Number of periods ahead to predict (default 5).
    column : str
        Price column to use (default "Close").
    method : str
        "returns" → forward return; "direction" → +1 or -1.

    Returns
    -------
    pd.DataFrame
        DataFrame with new column ``target``.
    """
    df = df.copy()
    fwd_return = df[column].shift(-horizon) / df[column] - 1

    if method == "returns":
        df["target"] = fwd_return
    elif method == "direction":
        # The final ``horizon`` rows have no forward return (NaN). Keep the sign
        # as a float so those NaNs are preserved for the caller's ``dropna()``;
        # casting straight to ``int`` here would raise IntCastingNaNError. Values
        # are -1.0 / 0.0 / +1.0 (down / flat / up).
        df["target"] = np.sign(fwd_return)
    else:
        raise ValueError(f"Unknown target method {method!r}. Use 'returns' or 'direction'.")

    return df
