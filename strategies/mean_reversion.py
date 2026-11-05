"""
Mean Reversion (Pairs Trading) Strategy.

Trades the spread between two assets: a rolling OLS hedge ratio, a rolling
z-score of the hedged log-price spread, and an entry/exit state machine.
Every quantity a signal uses at bar *t* is computed from bars up to and
including *t* only (the engine then shifts the position one bar).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint, adfuller

from alphaforge.backtest.engine import BacktestEngine, BacktestResult
from alphaforge.data.fetcher import DataFetcher


class MeanReversionStrategy:
    """
    Statistical arbitrage via pairs trading.

    At each bar *t* the hedge ratio is the OLS slope of ``log(A)`` on
    ``log(B)`` over the ``hedge_lookback`` bars ending at *t*; the spread
    ``log(A) - beta_t * log(B)`` is z-scored over the ``lookback`` bars ending
    at *t*, with ``beta_t`` applied to the whole window. Nothing after bar *t*
    enters bar *t*'s signal, so appending future bars never changes a past
    signal.

    Position rules (long = long A / short B, short = the reverse):

    - flat: enter long when ``z < -entry_z``, short when ``z > entry_z``;
    - long: exit to flat when ``z >= -exit_z`` (reverted toward the mean);
      if on that bar ``z > entry_z``, reverse straight into a short;
    - short: exit to flat when ``z <= exit_z``; reverse if ``z < -entry_z``;
    - while the hedge ratio or z-score is undefined (warm-up, zero variance)
      the position is flat.

    Costs and exposure: ``run`` backtests the spread as one instrument, so
    commission and slippage are charged on the spread position -- leg A's
    notional -- only; the hedge leg's costs, including re-hedging as the ratio
    moves, are not modelled. The book's GROSS exposure is ``1 + |hedge|`` times
    leg A's (both legs are traded), which the result reports as
    ``metrics["gross_exposure_mean"]`` / ``["gross_exposure_max"]`` over the
    bars a position was held.

    Parameters
    ----------
    entry_z : float
        Z-score threshold to enter a trade (default 2.0).
    exit_z : float
        Z-score threshold to exit (default 0.5). Must be below ``entry_z``.
    lookback : int
        Rolling window for the spread mean/std (default 60).
    hedge_lookback : int
        Rolling window for the OLS hedge ratio (default 120).
    initial_capital : float
        Starting capital.
    commission : float
        Per-side commission passed to the engine.

    Examples
    --------
    >>> strat = MeanReversionStrategy(entry_z=2.0, exit_z=0.5)
    >>> result = strat.run("AAPL", "MSFT", "2020-01-01", "2023-12-31")
    """

    NAME = "Mean Reversion (Pairs)"
    DESCRIPTION = (
        "Statistical arbitrage strategy trading the spread between two "
        "cointegrated assets. Uses a rolling hedge ratio and rolling z-score "
        "to identify entry/exit points."
    )

    def __init__(
        self,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        lookback: int = 60,
        hedge_lookback: int = 120,
        initial_capital: float = 100_000.0,
        commission: float = 0.001,
    ) -> None:
        if not 0 <= exit_z < entry_z:
            raise ValueError("need 0 <= exit_z < entry_z")
        if lookback < 2 or hedge_lookback < 2:
            raise ValueError("lookback and hedge_lookback must be at least 2 bars")
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.lookback = lookback
        self.hedge_lookback = hedge_lookback
        self.initial_capital = initial_capital
        self.commission = commission

    def test_cointegration(
        self, price_a: pd.Series, price_b: pd.Series
    ) -> dict[str, float]:
        """
        Test cointegration between two price series.

        IN-SAMPLE diagnostic: the Engle-Granger test and the OLS hedge ratio
        use the whole window. Selecting a pair with it and then backtesting
        that pair over the same window is itself look-ahead; the trading
        signals (:meth:`generate_signals`) never use this hedge ratio.

        Returns dict with: p_value, hedge_ratio, adf_statistic.
        """
        log_a = np.log(price_a.dropna())
        log_b = np.log(price_b.dropna())
        common = log_a.index.intersection(log_b.index)
        log_a, log_b = log_a.loc[common], log_b.loc[common]

        # Engle-Granger cointegration test
        _, p_value, _ = coint(log_a, log_b)

        # OLS hedge ratio: log_a = β * log_b + ε
        beta = float(np.cov(log_a, log_b)[0, 1] / np.var(log_b, ddof=1))

        # ADF test on spread
        spread = log_a - beta * log_b
        adf_stat, adf_p, _, _, _, _ = adfuller(spread)

        return {
            "p_value": float(p_value),
            "hedge_ratio": beta,
            "adf_statistic": float(adf_stat),
            "adf_p_value": float(adf_p),
            "is_cointegrated": p_value < 0.05,
        }

    def compute_spread(
        self,
        price_a: pd.Series,
        price_b: pd.Series,
        hedge_ratio: float | None = None,
    ) -> pd.Series:
        """Log price spread ``log(A) - hedge_ratio * log(B)``.

        With ``hedge_ratio=None`` the ratio is fitted on the WHOLE window (an
        in-sample, descriptive spread). For the causal spread the signals use,
        see :meth:`spread_zscore`.
        """
        log_a = np.log(price_a)
        log_b = np.log(price_b)
        if hedge_ratio is None:
            hedge_ratio = float(np.cov(log_a, log_b)[0, 1] / np.var(log_b, ddof=1))
        return log_a - hedge_ratio * log_b

    def spread_zscore(self, price_a: pd.Series, price_b: pd.Series) -> pd.DataFrame:
        """Causal hedge ratio, spread and z-score, one row per common bar.

        Columns: ``hedge_ratio`` (rolling OLS slope over ``hedge_lookback``
        bars ending at the row), ``spread`` (``log A - hedge_ratio * log B``)
        and ``zscore`` (the spread's deviation from its mean over the
        ``lookback`` bars ending at the row, in standard deviations, with this
        row's hedge ratio applied across that window). NaN during warm-up.
        Every rolling statistic is computed left to right over trailing
        windows, so a row never depends on a later row.
        """
        log_a = np.log(price_a)
        log_b = np.log(price_b)
        common = log_a.index.intersection(log_b.index)
        log_a, log_b = log_a.loc[common], log_b.loc[common]

        hedge = log_a.rolling(self.hedge_lookback).cov(log_b) / (
            log_b.rolling(self.hedge_lookback).var().replace(0.0, np.nan)
        )

        # Window moments of log A and log B; the hedged spread's window mean
        # and variance follow from them for THIS row's hedge ratio:
        #   mean(a - h b) = mean(a) - h mean(b)
        #   var(a - h b)  = var(a) + h^2 var(b) - 2 h cov(a, b)
        win_a, win_b = log_a.rolling(self.lookback), log_b.rolling(self.lookback)
        spread = log_a - hedge * log_b
        spread_mean = win_a.mean() - hedge * win_b.mean()
        spread_var = (
            win_a.var() + hedge ** 2 * win_b.var()
            - 2.0 * hedge * log_a.rolling(self.lookback).cov(log_b)
        )
        spread_std = np.sqrt(spread_var.clip(lower=0.0)).replace(0.0, np.nan)
        zscore = (spread - spread_mean) / spread_std
        return pd.DataFrame({"hedge_ratio": hedge, "spread": spread, "zscore": zscore})

    def _positions(self, zscore: pd.Series) -> pd.Series:
        """The entry/exit state machine (see the class docstring)."""
        out = np.zeros(len(zscore), dtype=int)
        pos = 0
        for i, z in enumerate(zscore.to_numpy()):
            if not np.isfinite(z):
                pos = 0
            else:
                if pos == 1 and z >= -self.exit_z:
                    pos = 0
                elif pos == -1 and z <= self.exit_z:
                    pos = 0
                if pos == 0:
                    if z < -self.entry_z:
                        pos = 1
                    elif z > self.entry_z:
                        pos = -1
            out[i] = pos
        return pd.Series(out, index=zscore.index)

    def generate_signals(
        self,
        price_a: pd.Series,
        price_b: pd.Series,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Generate trading signals for the spread (long = long A / short B).

        Returns ``(signals, hedge_ratio)``: integer +1/-1/0 signals and the
        causal hedge-ratio series they were computed with (NaN in warm-up).
        """
        frame = self.spread_zscore(price_a, price_b)
        return self._positions(frame["zscore"]), frame["hedge_ratio"]

    def run(
        self,
        symbol_a: str,
        symbol_b: str,
        start: str,
        end: str | None = None,
    ) -> BacktestResult:
        """
        Run pairs trading backtest between two symbols.

        Parameters
        ----------
        symbol_a, symbol_b : str
            Ticker symbols for the pair.
        start : str
            Start date (YYYY-MM-DD).
        end : str | None
            End date.

        Returns
        -------
        BacktestResult for the spread. ``metrics["hedge_ratio"]`` is the
        latest hedge-ratio estimate (the one in force on the final bar).
        """
        fetcher = DataFetcher()
        df_a = fetcher.fetch_ohlcv(symbol_a, start, end)
        df_b = fetcher.fetch_ohlcv(symbol_b, start, end)
        return self.backtest_pair(df_a["Close"], df_b["Close"])

    def backtest_pair(self, close_a: pd.Series, close_b: pd.Series) -> BacktestResult:
        """Backtest the spread on two close-price series already in hand.

        The spread is backtested as one market-neutral instrument whose return
        on bar *t* is ``ret_A(t) - hedge(t-1) * ret_B(t)``: the hedge ratio in
        force while the bar *t* position was held is the one known when that
        position was decided, at bar *t - 1* (the engine's one-bar shift).
        """
        signals, hedge = self.generate_signals(close_a, close_b)
        common = signals.index
        ret_a = close_a.loc[common].pct_change().fillna(0)
        ret_b = close_b.loc[common].pct_change().fillna(0)
        # Bars whose prior-bar hedge is undefined are flat (warm-up), so the
        # fill value never reaches the P&L.
        spread_ret = ret_a - hedge.shift(1).fillna(0.0) * ret_b
        spread_price = (1 + spread_ret).cumprod() * float(close_a.loc[common].iloc[0])
        spread_df = pd.DataFrame(
            {
                "Open": spread_price,
                "High": spread_price,
                "Low": spread_price,
                "Close": spread_price,
                "Volume": 1.0,
            },
            index=common,
        )

        engine = BacktestEngine(
            initial_capital=self.initial_capital,
            commission=self.commission,
        )
        result = engine.run(signals, spread_df)
        latest = hedge.dropna()
        result.metrics["hedge_ratio"] = float(latest.iloc[-1]) if not latest.empty else float("nan")
        # Gross exposure per held bar: |position| x (1 + |hedge in force|).
        held = result.positions["signal"].abs()
        gross = (held * (1.0 + hedge.shift(1).abs())).loc[held > 0]
        result.metrics["gross_exposure_mean"] = float(gross.mean()) if not gross.empty else float("nan")
        result.metrics["gross_exposure_max"] = float(gross.max()) if not gross.empty else float("nan")
        return result

    def __repr__(self) -> str:
        return (
            f"MeanReversionStrategy(entry_z={self.entry_z}, exit_z={self.exit_z}, "
            f"lookback={self.lookback}, hedge_lookback={self.hedge_lookback})"
        )
