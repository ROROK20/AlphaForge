"""Tests for performance metrics — numerical accuracy validation."""

import math

import numpy as np
import pandas as pd
import pytest

from alphaforge.backtest.metrics import (
    compute_metrics,
    compute_drawdown_series,
    annualized_return,
    rolling_sharpe,
    TRADING_DAYS,
    RISK_FREE_RATE,
)


class TestSharpRatio:
    def test_sharpe_known_value(self):
        """Manual calculation: Sharpe = (mean_excess / std_excess) * sqrt(252)."""
        rng = np.random.default_rng(7)
        daily_returns = pd.Series(0.001 + rng.normal(0, 0.005, 252))
        equity = 100_000 * (1 + daily_returns).cumprod()
        m = compute_metrics(equity, daily_returns)

        rfr_daily = RISK_FREE_RATE / TRADING_DAYS
        excess = daily_returns - rfr_daily
        expected_sharpe = excess.mean() / excess.std() * np.sqrt(TRADING_DAYS)
        assert m["sharpe"] == pytest.approx(expected_sharpe, rel=1e-4)

    def test_zero_volatility_sharpe_is_undefined(self):
        """A riskless series has no Sharpe ratio (zero denominator): NaN, because 0.0
        reads as "no edge" and is indistinguishable from a real 0.00."""
        returns = pd.Series([0.0] * 252)
        equity = pd.Series([100_000.0] * 252)
        assert math.isnan(compute_metrics(equity, returns)["sharpe"])
        rising = pd.Series([0.001] * 252)  # riskless AND profitable
        assert math.isnan(compute_metrics(100_000 * (1 + rising).cumprod(), rising)["sharpe"])

    def test_negative_returns_negative_sharpe(self):
        rng = np.random.default_rng(11)
        returns = pd.Series(-0.002 + rng.normal(0, 0.005, 252))
        equity = 100_000 * (1 + returns).cumprod()
        m = compute_metrics(equity, returns)
        assert m["sharpe"] < 0


class TestCAGR:
    def test_cagr_one_year_known(self):
        """10% growth over exactly 1 year → CAGR = 10%."""
        n = TRADING_DAYS
        daily = (1.10) ** (1 / n) - 1
        returns = pd.Series([daily] * n)
        equity = 100_000 * (1 + returns).cumprod()
        m = compute_metrics(equity, returns)
        assert m["cagr"] == pytest.approx(0.10, rel=0.01)

    def test_cagr_counts_intervals_not_points(self):
        """253 equity points = 252 bar-to-bar intervals = exactly one year.

        Exactly +10% over that year must be a CAGR of 10%; a year count of
        len(points)/252 would read 9.9586%.
        """
        equity = pd.Series(100_000 * 1.10 ** (np.arange(253) / TRADING_DAYS))
        returns = equity.pct_change().fillna(0)
        m = compute_metrics(equity, returns)
        assert m["total_return"] == pytest.approx(0.10, rel=1e-12)
        assert m["cagr"] == pytest.approx(0.10, rel=1e-12)

    def test_total_return(self):
        returns = pd.Series([0.01, -0.005, 0.02])
        equity = 100_000 * (1 + returns).cumprod()
        m = compute_metrics(equity, returns)
        expected = equity.iloc[-1] / equity.iloc[0] - 1
        assert m["total_return"] == pytest.approx(expected, rel=1e-6)


class TestMaxDrawdown:
    def test_max_drawdown_known(self):
        """Known equity sequence: peak=110, trough=88 → dd = (88-110)/110 ≈ -20%."""
        equity = pd.Series([100, 110, 105, 95, 88, 100, 115],
                           index=pd.date_range("2020-01-01", periods=7, freq="B"))
        returns = equity.pct_change().fillna(0)
        m = compute_metrics(equity, returns)
        assert m["max_drawdown"] == pytest.approx((88 - 110) / 110, rel=0.01)

    def test_monotone_increase_zero_drawdown(self):
        equity = pd.Series(np.linspace(100_000, 150_000, 100),
                           index=pd.date_range("2020-01-01", periods=100, freq="B"))
        returns = equity.pct_change().fillna(0)
        m = compute_metrics(equity, returns)
        assert m["max_drawdown"] == pytest.approx(0.0, abs=1e-6)


class TestDrawdownSeries:
    def test_at_peak_zero(self):
        equity = pd.Series([100, 105, 110])
        dd = compute_drawdown_series(equity)
        assert dd.iloc[-1] == pytest.approx(0.0, abs=1e-9)  # at peak

    def test_after_peak_negative(self):
        equity = pd.Series([100, 110, 99])
        dd = compute_drawdown_series(equity)
        expected = (99 - 110) / 110
        assert dd.iloc[-1] == pytest.approx(expected, rel=1e-6)

    def test_length_preserved(self):
        equity = pd.Series(np.random.default_rng(1).uniform(90, 110, 50))
        dd = compute_drawdown_series(equity)
        assert len(dd) == 50


class TestAnnualizedReturn:
    def test_flat_returns(self):
        returns = pd.Series([0.0] * 252)
        assert annualized_return(returns) == pytest.approx(0.0, abs=1e-6)

    def test_positive_returns(self):
        # Daily return such that 252-day compounding gives 20%
        daily = (1.20) ** (1 / 252) - 1
        returns = pd.Series([daily] * 252)
        assert annualized_return(returns) == pytest.approx(0.20, rel=0.01)


class TestRollingSharpe:
    def test_length_preserved(self):
        returns = pd.Series(np.random.default_rng(1).normal(0.001, 0.01, 200))
        rs = rolling_sharpe(returns, window=50)
        assert len(rs) == 200

    def test_no_nan_after_warmup(self):
        returns = pd.Series(np.random.default_rng(1).normal(0.001, 0.01, 200))
        rs = rolling_sharpe(returns, window=50)
        # After warmup period, should be non-NaN
        assert not rs.iloc[50:].isna().any()


class TestSortino:
    """Sortino = mean excess return / FULL-SAMPLE downside deviation.

    These pin the textbook definition, not the sample std of only the below-hurdle
    returns, which misstates it.
    """

    def test_known_value(self, monkeypatch):
        """Hand-computed with a zero hurdle.

        r = [+2%, -1%, +3%, -2%]: mean = 0.005; shortfalls squared =
        [0, 1e-4, 0, 4e-4], mean 1.25e-4, so downside deviation = 0.0111803;
        Sortino = 0.005 / 0.0111803 * sqrt(252) = 7.0993. (A subset-std
        formula gives std([-1%, -2%]) = 0.0070711 -> 11.2250.)
        """
        import alphaforge.backtest.metrics as metrics

        monkeypatch.setattr(metrics, "RISK_FREE_RATE", 0.0)
        returns = pd.Series([0.02, -0.01, 0.03, -0.02])
        equity = 100_000 * (1 + returns).cumprod()
        m = metrics.compute_metrics(equity, returns)
        assert m["sortino"] == pytest.approx(7.0992957, rel=1e-6)

    def test_matches_definition_with_the_default_hurdle(self):
        rng = np.random.default_rng(21)
        returns = pd.Series(0.0004 + rng.normal(0, 0.01, 500))
        equity = 100_000 * (1 + returns).cumprod()
        m = compute_metrics(equity, returns)
        rf = RISK_FREE_RATE / TRADING_DAYS
        downside_dev = np.sqrt(np.mean(np.minimum(returns - rf, 0.0) ** 2))
        expected = (returns - rf).mean() / downside_dev * np.sqrt(TRADING_DAYS)
        assert m["sortino"] == pytest.approx(expected, rel=1e-9)

    def test_no_downside_is_undefined(self):
        """No bar below the hurdle: downside deviation is 0, so the ratio is
        undefined -> NaN (null over the API).

        A strategy that never had a down day must not read as Sortino 0, the same
        as one with no edge at all.
        """
        returns = pd.Series([0.01] * 50)
        equity = 100_000 * (1 + returns).cumprod()
        assert math.isnan(compute_metrics(equity, returns)["sortino"])


class TestUndefinedRatios:
    """Zero denominators give NaN, never 0."""

    def test_no_drawdown_calmar_is_undefined(self):
        equity = pd.Series(np.linspace(100_000, 150_000, 100))
        m = compute_metrics(equity, equity.pct_change().fillna(0))
        assert m["max_drawdown"] == 0.0
        assert math.isnan(m["calmar"])

    def test_no_losing_trade_profit_factor_is_undefined(self):
        equity = pd.Series([100.0, 101.0, 102.0, 103.0])
        trades = pd.DataFrame({"pnl_net_abs": [1.0, 2.0], "pnl_pct": [0.01, 0.02]})
        m = compute_metrics(equity, equity.pct_change().fillna(0), trades)
        assert math.isnan(m["profit_factor"])
        assert math.isnan(m["avg_loss"])
        assert m["win_rate"] == 1.0


class TestWinStatisticsOnNetDollars:
    """Win rate / profit factor are computed on net dollar P&L.

    On each trade's unweighted ``pnl_pct``, gross of commission, profit factor could
    exceed 1 on a book that lost money.
    """

    def test_size_and_costs_decide_wins_and_profit_factor(self):
        equity = pd.Series([100_000.0, 99_400.0])
        trades = pd.DataFrame({
            # Two small-percentage winners, one big trade that lost; the
            # second "winner" is +0.05% before commission but -$20 after.
            "pnl_pct": [0.02, 0.0005, -0.01],
            "pnl_net_abs": [100.0, -20.0, -680.0],
        })
        m = compute_metrics(equity, equity.pct_change().fillna(0), trades)
        assert m["win_rate"] == pytest.approx(1 / 3)
        assert m["profit_factor"] == pytest.approx(100.0 / 700.0)
        assert m["avg_win"] == pytest.approx(100.0)
        assert m["avg_loss"] == pytest.approx(350.0)
        assert m["expectancy"] == pytest.approx(-600.0 / 3)

    def test_profit_factor_above_one_iff_the_book_made_money(self):
        """Net-dollar trade P&L sums to the equity curve's P&L (the reconciliation invariant), so on
        every engine run PF > 1 exactly when total return > 0."""
        from alphaforge.backtest.engine import BacktestEngine
        from alphaforge.signals.technical import (
            MACDSignal, RSIMeanReversionSignal, SMACrossoverSignal,
        )

        checked = 0
        for seed in range(12):
            rng = np.random.default_rng(seed)
            close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.013, 1000)))
            df = pd.DataFrame(
                {"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1e6},
                index=pd.date_range("2020-01-01", periods=1000, freq="B"),
            )
            for sig in (SMACrossoverSignal(20, 50), MACDSignal(), RSIMeanReversionSignal(14),
                        SMACrossoverSignal(10, 30)):
                m = BacktestEngine(commission=0.001).run(sig, df).metrics
                if math.isnan(m.get("profit_factor", float("nan"))):
                    continue
                checked += 1
                assert (m["profit_factor"] > 1) == (m["total_return"] > 0), (
                    seed, type(sig).__name__, m["profit_factor"], m["total_return"]
                )
        assert checked >= 40


class TestLowFixes20261102:
    def test_rolling_sharpe_is_nan_where_undefined_not_zero(self):
        """It used fillna(0): warm-up bars and zero-volatility windows read 0.0."""
        r = pd.Series([0.001] * 30 + list(np.random.default_rng(2).normal(0.001, 0.01, 70)))
        rs = rolling_sharpe(r, window=20)
        assert rs.iloc[:19].isna().all()            # warm-up
        assert rs.iloc[19:30].isna().all()          # constant returns: zero volatility
        assert rs.iloc[50:].notna().all()

    def test_drawdown_duration_counts_bars_and_says_so(self):
        equity = pd.Series([100, 110, 105, 104, 103, 111],
                           index=pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-15",
                                                 "2024-01-22", "2024-01-29", "2024-02-05"]))
        m = compute_metrics(equity, equity.pct_change().fillna(0))
        assert m["max_drawdown_duration_bars"] == 3.0   # 3 bars, 21 calendar days
        assert "max_drawdown_duration_days" not in m

    def test_annualized_return_of_interval_returns_equals_the_engine_cagr(self):
        """BacktestResult.returns starts with bar 0's structural zero; the
        per-interval returns are returns.iloc[1:] (documented contract)."""
        from alphaforge.backtest.engine import BacktestEngine

        rng = np.random.default_rng(4)
        close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, 253)))
        df = pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1e6},
                          index=pd.date_range("2020-01-01", periods=253, freq="B"))
        result = BacktestEngine(commission=0.0, slippage=0.0).run(pd.Series(1.0, index=df.index), df)
        assert result.returns.iloc[0] == 0.0
        assert annualized_return(result.returns.iloc[1:]) == pytest.approx(result.metrics["cagr"], rel=1e-12)



class TestStatisticsThroughTheEngine:
    """Ratio statistics skip the structural pre-position bar(s), and undefined
    ratios are NaN consistently -- tested through the engine."""

    @staticmethod
    def _run(close, signal_value, fill="close"):
        from alphaforge.backtest.engine import BacktestEngine

        df = pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1e6},
                          index=pd.date_range("2021-01-01", periods=len(close), freq="B"))
        return BacktestEngine(commission=0.0, slippage=0.0, fill=fill).run(
            pd.Series(signal_value, index=df.index), df).metrics

    @pytest.mark.parametrize("fill", ["close", "next_close"])
    def test_a_riskless_long_has_no_sharpe_or_sortino(self, fill):
        """A steady +0.1%/bar long: bar 0's structural zero must not make the
        series 'volatile' (Sharpe 231, Sortino 2916)."""
        m = self._run(100 * 1.001 ** np.arange(300), 1.0, fill)
        assert math.isnan(m["sharpe"]) and math.isnan(m["sortino"])
        assert m["total_return"] > 0

    def test_a_run_that_never_trades_has_no_ratios(self):
        """It read Sortino -15.87 next to an undefined Sharpe."""
        m = self._run(np.full(100, 50.0), 0.0)
        assert math.isnan(m["sharpe"]) and math.isnan(m["sortino"])
