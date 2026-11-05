"""Tests for signal generators."""

import numpy as np
import pandas as pd
import pytest

from alphaforge.signals.base import Signal
from alphaforge.signals.technical import (
    SMACrossoverSignal,
    RSIMeanReversionSignal,
    MACDSignal,
    BollingerBandSignal,
)
from alphaforge.signals.ensemble import EnsembleSignal


def make_ohlcv(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    high = close * (1 + rng.uniform(0, 0.02, n))
    low = close * (1 - rng.uniform(0, 0.02, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    volume = rng.integers(100_000, 1_000_000, n).astype(float)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


class TestSMACrossoverSignal:
    def test_output_shape(self):
        df = make_ohlcv()
        sig = SMACrossoverSignal(fast=10, slow=20)
        signals = sig.generate(df)
        assert len(signals) == len(df)

    def test_signal_values(self):
        df = make_ohlcv()
        sig = SMACrossoverSignal(fast=10, slow=20)
        signals = sig.generate(df)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_long_only_no_shorts(self):
        df = make_ohlcv()
        sig = SMACrossoverSignal(fast=10, slow=20, long_only=True)
        signals = sig.generate(df)
        assert (signals < 0).sum() == 0

    def test_fast_slow_validation(self):
        with pytest.raises(ValueError):
            SMACrossoverSignal(fast=50, slow=20)  # fast >= slow

    def test_nan_handling(self):
        df = make_ohlcv()
        sig = SMACrossoverSignal(fast=10, slow=20)
        signals = sig.generate(df)
        assert not signals.isna().any()

    def test_index_alignment(self):
        df = make_ohlcv()
        sig = SMACrossoverSignal()
        signals = sig.generate(df)
        pd.testing.assert_index_equal(signals.index, df.index)


def wilder_rsi(close: pd.Series, period: int) -> pd.Series:
    """Independent textbook RSI (Wilder smoothing), NaN for the warm-up bars."""
    delta = close.diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = (-delta).clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    rsi = 100 - 100 / (1 + avg_gain / avg_loss)
    rsi.iloc[:period] = np.nan
    return rsi


class TestRSIMeanReversionSignal:
    def test_any_period_uses_its_own_rsi(self):
        """The position must follow RSI(period) itself, whatever the period:
        add_technical_features writes only RSI_14."""
        df = make_ohlcv(400, seed=11)
        sig = RSIMeanReversionSignal(period=7)
        signals = sig.generate(df)
        rsi = wilder_rsi(df["Close"], 7)
        assert (signals[rsi <= sig.oversold] == 1).all()
        assert (signals[rsi >= sig.overbought] == -1).all()
        assert ((rsi <= sig.oversold) | (rsi >= sig.overbought)).sum() > 10
        assert not signals.equals(RSIMeanReversionSignal(period=14).generate(df))

    def test_flat_until_rsi_is_defined(self):
        """RSI(period) needs `period` price changes: no position on bar 0 from a fake
        "oversold" reading."""
        for seed in (1, 2, 3):
            df = make_ohlcv(120, seed=seed)
            for period in (5, 14):
                signals = RSIMeanReversionSignal(period=period).generate(df)
                assert (signals.iloc[:period] == 0).all(), (seed, period)

    def test_output_shape(self):
        df = make_ohlcv()
        sig = RSIMeanReversionSignal()
        signals = sig.generate(df)
        assert len(signals) == len(df)

    def test_signal_values(self):
        df = make_ohlcv()
        sig = RSIMeanReversionSignal()
        signals = sig.generate(df)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_extreme_prices_trigger_signals(self):
        """Monotonically rising prices should eventually trigger overbought."""
        n = 100
        close = np.linspace(100, 200, n)
        df = pd.DataFrame({
            "Open": close, "High": close * 1.01, "Low": close * 0.99,
            "Close": close, "Volume": np.ones(n) * 1e6,
        }, index=pd.date_range("2020-01-01", periods=n, freq="B"))
        sig = RSIMeanReversionSignal(overbought=60)
        signals = sig.generate(df)
        assert (signals == -1).any()

    def test_no_nan(self):
        df = make_ohlcv()
        sig = RSIMeanReversionSignal()
        assert not sig.generate(df).isna().any()


class TestMACDSignal:
    def test_output_shape(self):
        df = make_ohlcv(300)
        sig = MACDSignal()
        signals = sig.generate(df)
        assert len(signals) == len(df)

    def test_signal_values(self):
        df = make_ohlcv(300)
        sig = MACDSignal()
        signals = sig.generate(df)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_no_nan(self):
        df = make_ohlcv(300)
        assert not MACDSignal().generate(df).isna().any()

    @staticmethod
    def _hist(df, fast=12, slow=26, signal=9):
        close = df["Close"]
        macd = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
        return macd - macd.ewm(span=signal, adjust=False).mean()

    def test_position_is_held_between_crosses(self):
        """Regime, not pulse: after the first post-warm-up cross the position
        is +1 on every bar the MACD line is above its signal line and -1 on
        every bar it is below (bars exactly on the line keep the prior side).

        Not +/-1 on the crossover bar only: the engine would hold each MACD trade
        for one bar.
        """
        df = make_ohlcv(600, seed=4)
        sig = MACDSignal()
        signals = sig.generate(df)
        hist = self._hist(df)
        live = signals.iloc[sig.warmup:]
        first = live[live != 0].index[0]
        after = signals.loc[first:]
        h = hist.loc[first:]
        assert (after[h > 0] == 1).all()
        assert (after[h < 0] == -1).all()
        assert (after != 0).all(), "once in the market the regime never goes flat"

    def test_trades_are_held_for_more_than_one_bar(self):
        from alphaforge.backtest.engine import BacktestEngine

        df = make_ohlcv(1000, seed=6)
        trades = BacktestEngine().run(MACDSignal(), df).trades
        bars = [
            df.index.get_loc(x) - df.index.get_loc(e)
            for e, x in zip(trades["entry_date"], trades["exit_date"])
        ]
        assert len(trades) > 5
        assert np.median(bars) > 3, f"median hold {np.median(bars)} bars"

    def test_no_position_during_warmup(self):
        df = make_ohlcv(300, seed=9)
        sig = MACDSignal()
        assert sig.warmup == 25 + 8
        assert (sig.generate(df).iloc[: sig.warmup] == 0).all()

    def test_parameters_win_over_precomputed_columns(self):
        """add_technical_features() writes MACD/MACD_signal at fixed 12/26/9;
        a MACDSignal with other parameters must not silently use them."""
        from alphaforge.data.preprocessor import add_technical_features

        df = make_ohlcv(400, seed=2)
        sig = MACDSignal(fast=5, slow=35, signal=5)
        pd.testing.assert_series_equal(
            sig.generate(add_technical_features(df)), sig.generate(df)
        )

    def test_fast_must_be_below_slow(self):
        with pytest.raises(ValueError):
            MACDSignal(fast=26, slow=12)


class TestBollingerBandSignal:
    def test_output_shape(self):
        df = make_ohlcv()
        sig = BollingerBandSignal()
        signals = sig.generate(df)
        assert len(signals) == len(df)

    def test_signal_values(self):
        df = make_ohlcv()
        sig = BollingerBandSignal()
        assert set(sig.generate(df).unique()).issubset({-1, 0, 1})

    def test_parameters_win_over_precomputed_columns(self):
        """add_technical_features() writes BB_* at a fixed 20 periods / 2
        sigma; a BollingerBandSignal with other parameters must not use them."""
        from alphaforge.data.preprocessor import add_technical_features

        df = make_ohlcv(400, seed=12)
        sig = BollingerBandSignal(period=10, n_std=1.0)
        pd.testing.assert_series_equal(sig.generate(add_technical_features(df)), sig.generate(df))

    def test_entries_follow_its_own_bands(self):
        df = make_ohlcv(400, seed=13)
        sig = BollingerBandSignal(period=10, n_std=1.0)
        signals = sig.generate(df)
        close = df["Close"]
        mid = close.rolling(10).mean()
        band = close.rolling(10).std(ddof=0)
        below, above = close < mid - band, close > mid + band
        assert below.sum() > 10 and above.sum() > 10
        assert (signals[below] == 1).all()
        assert (signals[above] == -1).all()


class TestEnsembleSignal:
    def test_combination(self):
        df = make_ohlcv(300)
        signals_list = [
            SMACrossoverSignal(fast=10, slow=20),
            RSIMeanReversionSignal(),
        ]
        ensemble = EnsembleSignal(signals_list, weights=[0.6, 0.4])
        signals = ensemble.generate(df)
        assert len(signals) == len(df)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_equal_weights_default(self):
        df = make_ohlcv(300)
        sigs = [SMACrossoverSignal(fast=10, slow=20), MACDSignal()]
        ensemble = EnsembleSignal(sigs)
        signals = ensemble.generate(df)
        assert not signals.isna().any()


def test_readme_custom_signal_example_holds_a_position():
    """A signal is a position, so the README's MomentumBreakoutSignal must hold
    between its entry and exit events, not pulse for one bar."""
    import re
    from pathlib import Path

    from alphaforge.backtest.engine import BacktestEngine

    readme = (Path(__file__).resolve().parent.parent / "README.md").read_text()
    section = readme.split("## Writing a Custom Strategy", 1)[1]
    code = re.search(r"```python\n(.*?)```", section, re.S).group(1)
    code = code.split("# Use it")[0]  # the class only; df comes from below
    ns: dict = {}
    exec(code, ns)
    rng = np.random.default_rng(3)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, 1000)))
    df = pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1e6},
                      index=pd.date_range("2020-01-01", periods=1000, freq="B"))
    result = BacktestEngine().run(ns["MomentumBreakoutSignal"](window=20), df)
    t = result.trades
    holds = [df.index.get_loc(x) - df.index.get_loc(e) for e, x in zip(t["entry_date"], t["exit_date"])]
    assert len(t) > 3 and np.median(holds) > 5, (len(t), np.median(holds))


class _HalfFromBar3(Signal):
    name = "Half"

    def generate(self, df):
        raw = pd.Series(np.nan, index=df.index)
        raw.iloc[3] = 0.5
        return self._fill_signals(raw, df.index)


def test_fill_signals_keeps_a_fraction_and_the_engine_holds_it():
    """0.5 stays 0.5, not truncated to flat: zero trades, and no error."""
    from alphaforge.backtest.engine import BacktestEngine

    index = pd.date_range("2021-01-01", periods=10, freq="B")
    close = pd.Series(100.0, index=index)
    df = pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1.0})
    signal = _HalfFromBar3()
    held = signal.generate(df)
    assert held.iloc[2] == 0.0 and (held.iloc[3:] == 0.5).all()
    result = BacktestEngine(commission=0.0, slippage=0.0).run(signal, df)
    assert len(result.trades) == 1
    assert result.trades["shares"].iloc[0] == pytest.approx(0.5 * 100_000 / 100.0)


def test_fill_signals_still_gives_integers_for_whole_positions():
    index = pd.date_range("2021-01-01", periods=4, freq="B")
    raw = pd.Series([1.0, np.nan, -1.0, np.nan], index=index)
    filled = _HalfFromBar3()._fill_signals(raw, index)
    assert filled.dtype.kind == "i"
    assert filled.tolist() == [1, 1, -1, -1]


class _Fixed(Signal):
    name = "Fixed"

    def __init__(self, value: float) -> None:
        self.value = value

    def generate(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(self.value, index=df.index)


class TestEnsembleNormalisation:
    """The threshold is a share of the total weight, whatever the weights add up to."""

    @pytest.fixture
    def df(self):
        index = pd.date_range("2024-01-01", periods=5, freq="B")
        return pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 1.0}, index=index)

    @pytest.mark.parametrize("scale", [0.25, 1.0, 40.0])
    def test_the_vote_does_not_depend_on_the_scale_of_the_weights(self, df, scale):
        # Net share of the vote: (3 - 1) / 4 = 0.5, whatever the scale.
        ensemble = EnsembleSignal([_Fixed(1), _Fixed(-1)], weights=[3 * scale, 1 * scale], threshold=0.6)
        assert (ensemble.generate(df) == 0).all()
        ensemble = EnsembleSignal([_Fixed(1), _Fixed(-1)], weights=[3 * scale, 1 * scale], threshold=0.4)
        assert (ensemble.generate(df) == 1).all()

    def test_a_negative_weight_counts_by_its_size_toward_the_total(self, df):
        # (+2 x 1 - 1 x 1) / (2 + 1) = 1/3: under 0.5, over 0.3.
        assert (EnsembleSignal([_Fixed(1), _Fixed(1)], [2.0, -1.0], threshold=0.5).generate(df) == 0).all()
        assert (EnsembleSignal([_Fixed(1), _Fixed(1)], [2.0, -1.0], threshold=0.3).generate(df) == 1).all()

