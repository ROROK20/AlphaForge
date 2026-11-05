"""Tests for risk management modules."""

import numpy as np
import pandas as pd
import pytest

from alphaforge.risk.position_sizing import (
    KellyCriterion,
    FixedFractional,
    EqualWeight,
    VolatilityParity,
)
from alphaforge.risk.drawdown import DrawdownMonitor, compute_drawdown_series, underwater_periods
from alphaforge.risk.var import parametric_cvar, parametric_var, historical_var, monte_carlo_var


class TestKellyCriterion:
    def test_positive_kelly_for_edge(self):
        k = KellyCriterion(win_rate=0.6, avg_win=0.05, avg_loss=0.03)
        size = k.size(100_000)
        assert size > 0

    def test_zero_edge_zero_kelly(self):
        """If expected value is zero, Kelly = 0."""
        # For edge = 0: b*p - q = 0 → b = q/p
        # With win_rate=0.5, avg_win/avg_loss=1 → b*p-q = 0.5-0.5 = 0
        k = KellyCriterion(win_rate=0.5, avg_win=0.05, avg_loss=0.05)
        assert k.full_kelly == pytest.approx(0.0, abs=1e-6)

    def test_max_position_cap(self):
        k = KellyCriterion(win_rate=0.9, avg_win=0.10, avg_loss=0.01, max_position=0.10)
        assert k.size(100_000) <= 0.10

    def test_invalid_win_rate(self):
        with pytest.raises(ValueError):
            KellyCriterion(win_rate=1.5, avg_win=0.05, avg_loss=0.03)

    def test_fractional_kelly_less_than_full(self):
        k = KellyCriterion(win_rate=0.6, avg_win=0.05, avg_loss=0.03, fraction=0.25)
        assert k.size(100_000) <= k.full_kelly

    def test_repr(self):
        k = KellyCriterion(win_rate=0.6, avg_win=0.05, avg_loss=0.03)
        assert "KellyCriterion" in repr(k)


class TestFixedFractional:
    def test_size_formula(self):
        """size = risk_per_trade / stop_distance."""
        ff = FixedFractional(risk_per_trade=0.02)
        size = ff.size(100_000, stop_distance=0.05)
        assert size == pytest.approx(0.40, rel=1e-6)

    def test_zero_stop_distance(self):
        ff = FixedFractional()
        assert ff.size(100_000, stop_distance=0) == 0.0

    def test_size_capped_at_one(self):
        ff = FixedFractional(risk_per_trade=0.10)
        # risk 10%, stop 1% → would be 10x which we cap at 1.0
        size = ff.size(100_000, stop_distance=0.01)
        assert size <= 1.0

    def test_invalid_risk_per_trade(self):
        with pytest.raises(ValueError):
            FixedFractional(risk_per_trade=1.5)


class TestEqualWeight:
    def test_size_inverse_n(self):
        ew = EqualWeight(n_positions=4)
        assert ew.size(100_000) == pytest.approx(0.25)

    def test_single_position(self):
        ew = EqualWeight(n_positions=1)
        assert ew.size(100_000) == pytest.approx(1.0)

    def test_invalid_n(self):
        with pytest.raises(ValueError):
            EqualWeight(n_positions=0)


class TestVolatilityParity:
    def test_high_vol_small_size(self):
        vp = VolatilityParity(target_volatility=0.10)
        high_vol_returns = pd.Series(np.random.default_rng(1).normal(0, 0.05, 100))
        size_high = vp.size(100_000, returns=high_vol_returns)

        low_vol_returns = pd.Series(np.random.default_rng(1).normal(0, 0.005, 100))
        size_low = vp.size(100_000, returns=low_vol_returns)
        assert size_low > size_high

    def test_fallback_no_returns(self):
        vp = VolatilityParity()
        assert vp.size(100_000) == pytest.approx(0.10)


class TestDrawdownMonitor:
    def test_no_halt_normal_conditions(self):
        monitor = DrawdownMonitor(max_drawdown_pct=-0.20, daily_loss_limit=-0.05)
        assert monitor.update(100_000, 0.01)

    def test_halt_on_max_drawdown(self):
        monitor = DrawdownMonitor(max_drawdown_pct=-0.10)
        monitor.update(100_000, 0.0)  # set peak
        result = monitor.update(89_000, -0.11)  # 11% drawdown
        assert not result
        assert monitor.halted

    def test_halt_on_daily_loss(self):
        monitor = DrawdownMonitor(daily_loss_limit=-0.05)
        result = monitor.update(100_000, -0.06)
        assert not result
        assert monitor.halted

    def test_reset_clears_halt(self):
        monitor = DrawdownMonitor(max_drawdown_pct=-0.10)
        monitor.update(100_000, 0.0)
        monitor.update(89_000, -0.11)
        assert monitor.halted
        monitor.reset()
        assert not monitor.halted

    @pytest.mark.parametrize("kwargs", [{"max_drawdown_pct": 0.15}, {"daily_loss_limit": 0.03},
                                        {"max_drawdown_pct": 0.0}])
    def test_positive_thresholds_are_rejected_not_an_instant_halt(self, kwargs):
        """max_drawdown_pct=0.15 would halt on the first update (0 <= 0.15)."""
        with pytest.raises(ValueError, match="negative"):
            DrawdownMonitor(**kwargs)

    def test_lookback_window_measures_from_a_rolling_peak(self):
        """lookback_window measures from the rolling peak, not the all-time one."""
        values = [100.0, 120.0, 115.0, 112.0, 110.0, 108.0, 107.0]
        rolling = DrawdownMonitor(max_drawdown_pct=-0.10, daily_loss_limit=-0.5, lookback_window=3)
        all_time = DrawdownMonitor(max_drawdown_pct=-0.10, daily_loss_limit=-0.5)
        for v in values:
            rolling.update(v, 0.0)
            all_time.update(v, 0.0)
        # 108 vs the all-time 120 peak is -10.0% (halt); the rolling peak of
        # the last 3 values never sits more than 6.7% above the latest one.
        assert all_time.halted
        assert not rolling.halted

    def test_halt_reason_populated(self):
        monitor = DrawdownMonitor(max_drawdown_pct=-0.10)
        monitor.update(100_000, 0.0)
        monitor.update(85_000, -0.15)
        assert "Max drawdown" in monitor.halt_reason


class TestVaR:
    @pytest.fixture
    def returns(self):
        rng = np.random.default_rng(42)
        return pd.Series(rng.normal(0.0005, 0.01, 500))

    def test_parametric_var_positive(self, returns):
        var = parametric_var(returns, confidence=0.95)
        assert var > 0

    def test_historical_var_positive(self, returns):
        var = historical_var(returns, confidence=0.95)
        assert var > 0

    def test_higher_confidence_higher_var(self, returns):
        var_95 = parametric_var(returns, confidence=0.95)
        var_99 = parametric_var(returns, confidence=0.99)
        assert var_99 > var_95

    def test_monte_carlo_var_cvar(self, returns):
        var, cvar = monte_carlo_var(returns, confidence=0.95, n_simulations=10_000, seed=42)
        assert var > 0
        assert cvar >= var  # CVaR >= VaR by definition

    def test_var_scales_with_portfolio(self, returns):
        var_100k = parametric_var(returns, portfolio_value=100_000)
        var_200k = parametric_var(returns, portfolio_value=200_000)
        assert var_200k == pytest.approx(var_100k * 2, rel=1e-6)


class TestParametricCVaR:
    """CVaR is the mean loss beyond VaR: a tail average, so divided by the tail's size."""

    @pytest.mark.parametrize("confidence", [0.90, 0.95, 0.99])
    def test_it_is_the_tail_mean_of_the_normal_it_assumes(self, confidence):
        rng = np.random.default_rng(7)
        returns = pd.Series(rng.normal(0.0005, 0.01, 500))
        draws = rng.normal(returns.mean(), returns.std(), 4_000_000)
        cut = np.quantile(draws, 1 - confidence)
        expected = -draws[draws <= cut].mean()
        assert parametric_cvar(returns, confidence) == pytest.approx(expected, rel=0.01)

    @pytest.mark.parametrize("confidence", [0.90, 0.95, 0.99])
    def test_it_is_never_smaller_than_var_at_the_same_confidence(self, confidence):
        returns = pd.Series(np.random.default_rng(8).normal(0.0005, 0.01, 500))
        assert parametric_cvar(returns, confidence) > parametric_var(returns, confidence)


class TestDrawdownFunctions:
    def test_compute_drawdown_series(self):
        equity = pd.Series([100, 110, 105, 90, 100, 115])
        dd = compute_drawdown_series(equity)
        assert dd.iloc[0] == 0  # at peak
        assert dd.min() < 0  # has a drawdown

    def test_underwater_periods_returns_dataframe(self):
        equity = pd.Series(
            [100, 105, 102, 95, 98, 110, 108, 107, 115],
            index=pd.date_range("2020-01-01", periods=9, freq="B")
        )
        df = underwater_periods(equity)
        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            assert "depth" in df.columns
            assert (df["depth"] <= 0).all()


class TestReadmeSizingExample:
    """The README's position-sizing block must print what its comments say.

    Full Kelly p - q/b = 0.58 - 0.42/1.5 = 30%, quartered, is 7.5%. This executes the
    block verbatim and checks
    every ``print(...)  # <expected>`` line against real output.
    """

    def test_every_print_matches_its_comment(self, capsys):
        import re
        from pathlib import Path

        readme = (Path(__file__).resolve().parent.parent / "README.md").read_text()
        section = readme.split("### Position Sizing", 1)[1]
        block = re.search(r"```python\n(.*?)```", section, re.S).group(1)
        expected = [
            m.group(1).strip()
            for m in re.finditer(r"^print\(.*\)\s*#\s*(.+)$", block, re.M)
        ]
        assert len(expected) >= 2, "README sizing block lost its annotated prints"
        exec(compile(block, "README.md:Position Sizing", "exec"), {})
        printed = capsys.readouterr().out.strip().splitlines()
        assert len(printed) == len(expected)
        for line, want in zip(printed, expected):
            assert line.endswith(want), f"README says {want!r}, the code prints {line!r}"


def _returns_with(mean: float, std: float, n: int = 1000, seed: int = 9) -> pd.Series:
    """A return series with exactly this sample mean and standard deviation."""
    raw = np.random.default_rng(seed).normal(size=n)
    return pd.Series(mean + std * (raw - raw.mean()) / raw.std(ddof=1))


@pytest.mark.parametrize("mean,std", [(0.0, 0.05), (0.01, 0.02), (-0.01, 0.03), (0.02, 0.05)])
def test_monte_carlo_var_is_the_lognormal_quantile_of_its_drift_and_volatility(mean, std):
    """GBM has a closed form: terminal = exp((mu - sigma^2 / 2) h + sigma sqrt(h) Z).
    This checks the value itself, not only VaR > 0 and CVaR >= VaR."""
    from scipy import stats

    horizon, confidence = 10, 0.95
    returns = _returns_with(mean, std)
    var, cvar = monte_carlo_var(returns, confidence=confidence, horizon=horizon, n_simulations=200_000, seed=42)
    log_drift = (mean - 0.5 * std**2) * horizon
    log_vol = std * np.sqrt(horizon)
    z = stats.norm.ppf(1 - confidence)
    assert var == pytest.approx(1 - np.exp(log_drift + z * log_vol), abs=0.002)
    # E[exp(X) | X <= q] for normal X, the partial expectation of a lognormal.
    tail_mean = np.exp(log_drift + 0.5 * log_vol**2) * stats.norm.cdf(z - log_vol) / (1 - confidence)
    assert cvar == pytest.approx(1 - tail_mean, abs=0.002)


def test_a_stronger_drift_lowers_monte_carlo_var():
    low, _ = monte_carlo_var(_returns_with(0.0, 0.02), n_simulations=50_000, seed=1)
    high, _ = monte_carlo_var(_returns_with(0.01, 0.02), n_simulations=50_000, seed=1)
    assert low - high > 0.05

