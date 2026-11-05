"""Annualised figures follow the spacing of the bars.

Counting every bar as a 252nd of a year would misreport the same prices held
buy-and-hold: CAGR 17.67% / Sharpe 0.85 daily but 125.03% / 2.12 weekly.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from alphaforge.api.app import app, synthetic_ohlcv
from alphaforge.backtest import metrics as metrics_module
from alphaforge.backtest.engine import BacktestEngine
from alphaforge.backtest.metrics import TRADING_DAYS, annualized_return, compute_metrics, rolling_sharpe
from alphaforge.optimization.markowitz import MarkowitzOptimizer
from alphaforge.risk.position_sizing import VolatilityParity
from alphaforge.risk.var import var_report

client = TestClient(app)


def infer_periods_per_year(index: pd.Index) -> float:
    return metrics_module.infer_periods_per_year(index)


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    return df.resample(rule).agg(agg).dropna()


@pytest.fixture(scope="module")
def daily() -> pd.DataFrame:
    return synthetic_ohlcv("AAPL", "2020-01-01", "2024-01-01")


def hold(df: pd.DataFrame, **engine_kw) -> dict:
    signal = pd.Series(1.0, index=df.index)
    return BacktestEngine(commission=0.0, slippage=0.0, **engine_kw).run(signal, df).metrics


@pytest.mark.parametrize(
    "rule, periods",
    [("B", 252), ("D", 252), ("W-FRI", 52), ("W-MON", 52), ("ME", 12), ("MS", 12), ("QE", 4), ("YE", 1)],
)
def test_the_spacing_names_the_bars_per_year(rule, periods):
    index = pd.date_range("2015-01-01", "2024-12-31", freq=rule)
    assert infer_periods_per_year(index) == periods


def test_holes_in_daily_bars_are_still_daily(daily):
    assert infer_periods_per_year(daily.index[np.arange(len(daily)) % 40 != 0]) == TRADING_DAYS


def test_timezone_aware_daily_bars_are_daily():
    index = pd.date_range("2021-01-01", periods=400, freq="B", tz="America/New_York")
    assert infer_periods_per_year(index) == TRADING_DAYS


def test_two_bars_across_a_weekend_are_daily():
    assert infer_periods_per_year(pd.DatetimeIndex(["2024-01-05", "2024-01-08"])) == TRADING_DAYS


@pytest.mark.parametrize("freq", ["h", "5min", "2B", "14D", "180D"])
def test_a_spacing_it_cannot_name_is_refused(freq):
    index = pd.date_range("2020-01-01", periods=200, freq=freq)
    with pytest.raises(ValueError, match="periods_per_year"):
        infer_periods_per_year(index)


def test_an_index_without_dates_is_daily_with_a_warning():
    with pytest.warns(UserWarning, match="not dates"):
        assert infer_periods_per_year(pd.RangeIndex(10)) == TRADING_DAYS


def test_one_bar_needs_no_spacing():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert infer_periods_per_year(pd.DatetimeIndex(["2024-01-05"])) == TRADING_DAYS


@pytest.mark.parametrize("rule, periods", [("W-FRI", 52), ("ME", 12)])
def test_the_same_prices_read_alike_on_any_bars(daily, rule, periods):
    """Buy-and-hold is one path: its CAGR, vol and Sharpe cannot depend much on the bars."""
    coarse = resample(daily, rule)
    m = hold(coarse)
    d = hold(daily)
    total = coarse["Close"].iloc[-1] / coarse["Close"].iloc[0] - 1
    assert m["cagr"] == pytest.approx((1 + total) ** (periods / (len(coarse) - 1)) - 1, rel=1e-12)
    assert m["cagr"] == pytest.approx(d["cagr"], abs=0.05)
    assert m["volatility"] == pytest.approx(d["volatility"], rel=0.35)
    assert m["sharpe"] == pytest.approx(d["sharpe"], abs=0.5)
    assert m["periods_per_year"] == periods


def test_daily_figures_stay_on_252(daily):
    m = hold(daily)
    n = len(daily)
    total = daily["Close"].iloc[-1] / daily["Close"].iloc[0] - 1
    assert m["cagr"] == pytest.approx((1 + total) ** (TRADING_DAYS / (n - 1)) - 1, rel=1e-12)
    assert m["periods_per_year"] == TRADING_DAYS


def test_the_engine_takes_an_explicit_year(daily):
    total = daily["Close"].iloc[-1] / daily["Close"].iloc[0] - 1
    m = hold(daily, periods_per_year=365)
    assert m["cagr"] == pytest.approx((1 + total) ** (365 / (len(daily) - 1)) - 1, rel=1e-12)


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_the_engine_refuses_a_nonsense_year(bad):
    with pytest.raises(ValueError, match="periods_per_year"):
        BacktestEngine(periods_per_year=bad)


def test_the_engine_refuses_hourly_bars_unless_told(daily):
    hourly = daily.iloc[:300].copy()
    hourly.index = pd.date_range("2024-01-02 09:30", periods=300, freq="h")
    signal = pd.Series(1.0, index=hourly.index)
    with pytest.raises(ValueError, match="periods_per_year"):
        BacktestEngine().run(signal, hourly)
    assert BacktestEngine(periods_per_year=1638).run(signal, hourly).metrics["periods_per_year"] == 1638


def test_run_multi_reads_the_same_spacing(daily):
    weekly = resample(daily, "W-FRI")
    signal = pd.Series(1.0, index=weekly.index)
    result = BacktestEngine().run_multi({"A": signal, "B": signal}, {"A": weekly, "B": weekly})
    assert result.metrics["periods_per_year"] == 52


def test_a_ruined_run_keeps_the_spacing_of_its_bars():
    """Ruined on bar 1: only two points reach the metrics, spaced as the whole run."""
    index = pd.date_range("2020-01-03", periods=60, freq="W-FRI")
    close = np.full(60, 100.0)
    close[1:] = 500.0
    df = pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1.0}, index=index)
    result = BacktestEngine(commission=0.0, slippage=0.0).run(pd.Series(-1.0, index=index), df)
    assert result.ruined
    assert result.metrics["periods_per_year"] == 52


def test_compute_metrics_and_the_helpers_read_the_dates(daily):
    weekly = resample(daily, "W-FRI")["Close"]
    rets = weekly.pct_change().dropna()
    expected = (1 + rets).prod() ** (52 / len(rets)) - 1
    assert annualized_return(rets) == pytest.approx(expected, rel=1e-12)
    assert annualized_return(rets, periods_per_year=52) == pytest.approx(expected, rel=1e-12)
    assert compute_metrics(weekly, rets)["periods_per_year"] == 52
    last = (rets - 0.02 / 52).iloc[-20:]
    assert rolling_sharpe(rets, window=20).iloc[-1] == pytest.approx(last.mean() / last.std() * np.sqrt(52), rel=1e-12)


def test_var_horizons_are_days_whatever_the_bars(daily):
    """A 1-day VaR estimated from weekly returns is near the one from daily returns, not sqrt(5) above."""
    long_daily = synthetic_ohlcv("MSFT", "2005-01-01", "2024-01-01")["Close"]
    from_daily = var_report(long_daily.pct_change().dropna(), confidence_levels=[0.95])
    weekly = long_daily.resample("W-FRI").last().dropna()
    from_weekly = var_report(weekly.pct_change().dropna(), confidence_levels=[0.95])
    for column in ("param_var_1d", "param_var_10d", "hist_var_1d", "hist_var_10d", "param_cvar_1d"):
        assert from_weekly[column].iloc[0] == pytest.approx(from_daily[column].iloc[0], rel=0.2), column


def test_the_optimizer_annualises_by_the_bars():
    rng = np.random.default_rng(0)
    index = pd.date_range("2015-01-02", periods=300, freq="W-FRI")
    returns = pd.DataFrame(rng.normal(0.002, 0.03, (300, 2)), index=index, columns=["A", "B"])
    opt = MarkowitzOptimizer(returns)
    pd.testing.assert_series_equal(opt.mu, returns.mean() * 52)
    pd.testing.assert_frame_equal(opt.cov, returns.cov() * 52)


def test_volatility_parity_annualises_by_the_bars():
    rng = np.random.default_rng(0)
    index = pd.date_range("2015-01-02", periods=100, freq="W-FRI")
    returns = pd.Series(rng.normal(0.0, 0.02, 100), index=index)
    expected = 0.15 / (returns.iloc[-20:].std() * np.sqrt(52))
    assert VolatilityParity().size(100_000, returns=returns) == pytest.approx(min(expected, 1.0), rel=1e-12)


def bars(df: pd.DataFrame) -> list[dict]:
    return [
        {"date": d.isoformat(), "open": r.Open, "high": r.High, "low": r.Low, "close": r.Close, "volume": r.Volume}
        for d, r in zip(df.index, df.itertuples())
    ]


def test_the_api_reads_supplied_weekly_bars_as_weekly(daily):
    weekly = resample(daily, "W-FRI")
    body = {
        "symbol": "AAPL", "strategy": "macd", "startDate": "2020-01-01", "endDate": "2024-01-01",
        "ohlcv": bars(weekly), "commission": 0.0, "slippage": 0.0,
    }
    got = client.post("/backtest", json=body)
    assert got.status_code == 200, got.text
    result = got.json()
    m = result["metrics"]
    expected = (1 + m["totalReturn"]) ** (52 / (len(result["equity"]) - 1)) - 1
    assert m["cagr"] == pytest.approx(expected, rel=1e-9)


def test_the_api_refuses_hourly_bars_unless_told(daily):
    hourly = daily.iloc[:300].copy()
    hourly.index = pd.date_range("2023-01-02 09:30", periods=300, freq="h")
    backtest = {
        "symbol": "AAPL", "strategy": "macd", "startDate": "2023-01-01", "endDate": "2023-03-01",
        "ohlcv": bars(hourly),
    }
    var = {"symbol": "AAPL", "startDate": "2023-01-01", "endDate": "2023-03-01", "ohlcv": bars(hourly)}
    for endpoint, body in (("/backtest", backtest), ("/var", var)):
        refused = client.post(endpoint, json=body)
        assert refused.status_code == 422, refused.text[:200]
        assert "periodsPerYear" in refused.json()["detail"]
        told = client.post(endpoint, json={**body, "periodsPerYear": 1638})
        assert told.status_code == 200, told.text[:200]


@pytest.mark.parametrize("bad", [0, -1, 1e9])
def test_the_api_refuses_a_nonsense_year(bad):
    got = client.post("/var", json={"symbol": "SPY", "periodsPerYear": bad})
    assert got.status_code == 422


def test_the_api_var_uses_days_on_weekly_bars():
    long_daily = synthetic_ohlcv("MSFT", "2005-01-01", "2024-01-01")
    weekly = resample(long_daily, "W-FRI")
    window = {"symbol": "MSFT", "startDate": "2005-01-01", "endDate": "2024-01-01"}
    row_daily = client.post("/var", json=window).json()["rows"][1]
    row_weekly = client.post("/var", json={**window, "ohlcv": bars(weekly)}).json()["rows"][1]
    assert row_weekly["paramVar1d"] == pytest.approx(row_daily["paramVar1d"], rel=0.2)
