"""
ML Momentum Strategy.

Uses an XGBoost classifier trained on lagged price/volume features
to predict the direction of 5-day forward returns.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from alphaforge.signals.ml_signals import XGBoostSignal
from alphaforge.backtest.engine import BacktestEngine, BacktestResult
from alphaforge.data.fetcher import DataFetcher
from alphaforge.data.preprocessor import add_technical_features, create_lagged_features


class MLMomentumStrategy:
    """
    Machine-learning momentum strategy powered by XGBoost.

    Trains on historical data, then generates directional signals
    (+1 / -1 / 0) from probability thresholds.

    Parameters
    ----------
    target_horizon : int
        Forward return horizon in days for the prediction target (default 5).
    train_ratio : float
        Fraction of data used for training (default 0.6 = in-sample).
    long_threshold : float
        Minimum predicted probability for long entry (default 0.60).
    short_threshold : float
        Maximum predicted probability for short entry (default 0.40).
    initial_capital : float
        Starting capital.

    Examples
    --------
    >>> strat = MLMomentumStrategy(target_horizon=5, train_ratio=0.6)
    >>> result = strat.run("AAPL", "2018-01-01", "2023-12-31")
    >>> print(result.metrics["sharpe"])
    """

    NAME = "ML Momentum (XGBoost)"
    DESCRIPTION = (
        "Momentum strategy using XGBoost to classify 5-day price direction. "
        "Features include lagged returns, volume, RSI, MACD, and Bollinger Bands. "
        "Models are scored only on bars after their training window, with the "
        "label horizon purged at every train/test boundary."
    )

    def __init__(
        self,
        target_horizon: int = 5,
        train_ratio: float = 0.6,
        long_threshold: float = 0.60,
        short_threshold: float = 0.40,
        initial_capital: float = 100_000.0,
        commission: float = 0.001,
        model_dir: str | None = None,
    ) -> None:
        self.target_horizon = target_horizon
        self.train_ratio = train_ratio
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.initial_capital = initial_capital
        self.commission = commission
        self.model_dir = Path(model_dir) if model_dir else None
        # With no model_dir the signal is ephemeral: it trains on the data given below.
        self._signal = XGBoostSignal(
            target_horizon=target_horizon,
            threshold=long_threshold,
            short_threshold=short_threshold,
            model_dir=self.model_dir,
        )

    def run(
        self,
        symbol: str,
        start: str,
        end: str | None = None,
    ) -> BacktestResult:
        """
        Train and backtest the ML momentum strategy.

        Uses strict train/test split to avoid look-ahead bias.
        Training happens only on the first `train_ratio` of data.

        Parameters
        ----------
        symbol : str
            Ticker symbol.
        start, end : str
            Date range.

        Returns
        -------
        BacktestResult (evaluated on test set only).
        """
        fetcher = DataFetcher()
        df = fetcher.fetch_ohlcv(symbol, start, end)

        # Add features
        df = add_technical_features(df)
        df = create_lagged_features(df, n_lags=5)
        df = df.dropna()

        # Train / test split
        split_idx = int(len(df) * self.train_ratio)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        # Train
        self._signal.fit(train_df)
        # fit() persists the model only when a model_dir/model_name was given;
        # either way the in-memory model below is the one just trained here.

        # Score the FULL history so the test bars' rolling features are warmed
        # up by the (past) training bars; generate() itself zeroes every bar at
        # or before the training window's end, so only test bars are scored.
        signals = self._signal.generate(df).loc[test_df.index]

        # Backtest on test set
        engine = BacktestEngine(
            initial_capital=self.initial_capital,
            commission=self.commission,
        )
        result = engine.run(signals, test_df)
        result.metrics["train_size"] = len(train_df)
        result.metrics["test_size"] = len(test_df)
        return result

    def run_walk_forward(
        self,
        symbol: str,
        start: str,
        end: str | None = None,
        window_size: int = 252,
        step_size: int = 63,
    ) -> BacktestResult:
        """
        Walk-forward validation backtest.

        Trains on a rolling window, tests on the next step_size bars.
        Concatenates all out-of-sample predictions for final evaluation.

        Parameters
        ----------
        window_size : int
            Training window in trading days (default 252 = 1 year).
        step_size : int
            Out-of-sample test window (default 63 = 1 quarter).

        Returns
        -------
        BacktestResult on combined out-of-sample data.
        """
        fetcher = DataFetcher()
        df = fetcher.fetch_ohlcv(symbol, start, end)
        df = add_technical_features(df)
        df = create_lagged_features(df, n_lags=5)
        df = df.dropna()

        blocks: list[pd.Series] = []
        for i in range(window_size, len(df), step_size):
            train_slice = df.iloc[i - window_size: i]
            self._signal.fit(train_slice)
            # Score the block on a frame that starts where training started, with
            # the training window as feature warm-up. Scoring the 63-bar slice on
            # its own would recompute rolling features from scratch, forcing ~26 of
            # every 63 bars flat. Scoring from the first bar of the data would give
            # the model a different OBV, a running sum from its frame's first bar,
            # than it trained on: on AAPL 2020-2023, one bar reads 21M in training
            # and 2,304M in scoring.
            # Bars up to the training window's end are zeroed by generate(); the
            # block's own bars are out of sample.
            scored = self._signal.generate(df.iloc[i - window_size: i + step_size])
            blocks.append(scored.iloc[window_size:])

        test_start = df.index[window_size]
        oos_df = df.loc[test_start:]
        oos_signals = pd.concat(blocks).reindex(oos_df.index).fillna(0).astype(int)

        engine = BacktestEngine(
            initial_capital=self.initial_capital,
            commission=self.commission,
        )
        result = engine.run(oos_signals, oos_df)
        result.metrics["n_walk_forward_windows"] = len(blocks)
        return result

    def __repr__(self) -> str:
        return (
            f"MLMomentumStrategy(horizon={self.target_horizon}, "
            f"train_ratio={self.train_ratio})"
        )
