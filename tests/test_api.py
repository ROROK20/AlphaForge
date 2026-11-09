"""Tests for the FastAPI layer (alphaforge/api/app.py) using TestClient.

These exercise the REAL engine end-to-end through HTTP: synthetic OHLCV →
BacktestEngine / var_report / MarkowitzOptimizer → JSON the frontend consumes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")
from fastapi.testclient import TestClient  # noqa: E402

from alphaforge.api.app import app  # noqa: E402

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["engine"] == "alphaforge"


def test_root_lists_endpoints():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "/backtest" in resp.json()["endpoints"]


@pytest.mark.parametrize(
    "strategy", ["sma_crossover", "rsi_meanrev", "macd", "ml_momentum"]
)
def test_backtest_returns_real_shaped_payload(strategy):
    resp = client.post(
        "/backtest",
        json={
            "symbol": "AAPL",
            "strategy": strategy,
            "startDate": "2020-01-01",
            "endDate": "2023-01-01",
            "initialCapital": 100_000,
            "commission": 0.001,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["source"] == "engine"

    # Equity curve present, non-empty, correctly shaped.
    equity = body["equity"]
    assert isinstance(equity, list) and len(equity) > 50
    first = equity[0]
    for key in ("date", "equity", "benchmark", "drawdown"):
        assert key in first
    assert all(p["drawdown"] <= 1e-9 for p in equity)  # drawdown is <= 0

    # Metrics present with the camelCase keys the frontend reads.
    metrics = body["metrics"]
    for key in (
        "totalReturn",
        "cagr",
        "sharpe",
        "sortino",
        "calmar",
        "maxDrawdown",
        "volatility",
        "winRate",
        "profitFactor",
        "nTrades",
    ):
        assert key in metrics, key

    # Trades, when present, carry the camelCase fields TradesTable renders.
    if body["trades"]:
        t = body["trades"][0]
        for key in (
            "id",
            "entryDate",
            "exitDate",
            "direction",
            "entryPrice",
            "exitPrice",
            "pnlPct",
            "pnlAbs",
        ):
            assert key in t, key
        assert t["direction"] in ("long", "short")

    # Truth-in-labeling: the response always discloses what actually ran.
    assert body["strategyExecuted"]
    if strategy == "ml_momentum":
        assert body["strategyExecuted"] == "sma_crossover(10,30)"
        assert "NOT ML-driven" in body["note"]
        assert "too slow" not in body["note"]
    else:
        assert body["note"] is None
        assert body["strategyExecuted"].startswith(strategy.split("_")[0])


def test_backtest_is_deterministic():
    payload = {
        "symbol": "MSFT",
        "strategy": "macd",
        "startDate": "2021-01-01",
        "endDate": "2023-06-01",
        "initialCapital": 50_000,
        "commission": 0.0005,
    }
    a = client.post("/backtest", json=payload).json()
    b = client.post("/backtest", json=payload).json()
    assert a["equity"][-1]["equity"] == b["equity"][-1]["equity"]


def test_backtest_accepts_supplied_ohlcv():
    bars = [
        {"date": f"2022-01-{d:02d}", "open": 100 + d, "high": 101 + d,
         "low": 99 + d, "close": 100 + d, "volume": 1_000_000}
        for d in range(1, 29)
    ]
    resp = client.post(
        "/backtest",
        json={
            "symbol": "X",
            "strategy": "sma_crossover",
            "startDate": "2022-01-01",
            "endDate": "2022-02-01",
            "ohlcv": bars,
        },
    )
    assert resp.status_code == 200, resp.text
    assert len(resp.json()["equity"]) == len(bars)


def test_backtest_rejects_empty_body():
    """A garbage/empty body must be rejected with 422, not silently 200."""
    resp = client.post("/backtest", json={})
    assert resp.status_code == 422, resp.text


def test_backtest_rejects_unknown_keys():
    """Unknown keys are forbidden -> 422 (extra='forbid')."""
    resp = client.post(
        "/backtest",
        json={
            "symbol": "AAPL",
            "strategy": "sma_crossover",
            "startDate": "2020-01-01",
            "endDate": "2023-01-01",
            "bogusField": True,
        },
    )
    assert resp.status_code == 422, resp.text


def test_backtest_rejects_reversed_date_range():
    """startDate >= endDate must be rejected with 422."""
    resp = client.post(
        "/backtest",
        json={
            "symbol": "AAPL",
            "strategy": "sma_crossover",
            "startDate": "2024-01-01",
            "endDate": "2020-01-01",
        },
    )
    assert resp.status_code == 422, resp.text


def test_var_report_shape():
    resp = client.post(
        "/var",
        json={
            "symbol": "SPY",
            "startDate": "2022-01-01",
            "endDate": "2024-01-01",
            "portfolioValue": 100_000,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["source"] == "engine"
    rows = body["rows"]
    assert len(rows) == 3  # default 90/95/99
    row = rows[0]
    for key in (
        "confidence",
        "paramVar1d",
        "histVar1d",
        "paramVar10d",
        "histVar10d",
        "paramCvar1d",
    ):
        assert key in row, key
    # 10-day VaR should exceed 1-day VaR (vol scales with sqrt(horizon)).
    assert row["paramVar10d"] > row["paramVar1d"]


def test_optimize_weights_sum_to_one():
    resp = client.post(
        "/optimize",
        json={
            "symbols": ["AAPL", "MSFT", "GOOGL", "SPY"],
            "objective": "max_sharpe",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    weights = body["weights"]
    assert len(weights) == 4
    assert abs(sum(weights.values()) - 1.0) < 0.02
    assert "sharpe" in body


# ---------------------------------------------------------------------------
# DoS-hardening: unauthenticated inputs are bounded -> clean 422, no hang/500.
# ---------------------------------------------------------------------------


def test_optimize_rejects_oversized_symbols():
    """A symbol list above the cap (50) must be rejected with 422, not hang."""
    resp = client.post(
        "/optimize",
        json={
            "symbols": [f"SYM{i}" for i in range(800)],
            "objective": "max_sharpe",
        },
    )
    assert resp.status_code == 422, resp.text


def test_optimize_accepts_max_symbols_boundary():
    """Exactly at the cap (50 symbols) still passes validation (no 422)."""
    resp = client.post(
        "/optimize",
        json={
            "symbols": [f"SYM{i}" for i in range(50)],
            "objective": "max_sharpe",
        },
    )
    assert resp.status_code != 422, resp.text


def test_var_rejects_oversized_confidence_levels():
    """confidenceLevels above the cap (20) must be rejected with 422."""
    resp = client.post(
        "/var",
        json={
            "symbol": "SPY",
            "startDate": "2022-01-01",
            "endDate": "2024-01-01",
            "confidenceLevels": [0.99] * 100_000,
        },
    )
    assert resp.status_code == 422, resp.text


def test_var_rejects_out_of_range_confidence_level():
    """A confidence level outside (0, 1) must be rejected with 422."""
    resp = client.post(
        "/var",
        json={
            "symbol": "SPY",
            "startDate": "2022-01-01",
            "endDate": "2024-01-01",
            "confidenceLevels": [0.95, 1.5],
        },
    )
    assert resp.status_code == 422, resp.text


def test_backtest_rejects_overlong_date_range():
    """A >40-year span must be rejected with 422 before generating bars."""
    resp = client.post(
        "/backtest",
        json={
            "symbol": "AAPL",
            "strategy": "sma_crossover",
            "startDate": "1900-01-01",
            "endDate": "2024-01-01",
        },
    )
    assert resp.status_code == 422, resp.text


def test_backtest_rejects_oversized_ohlcv():
    """An OHLCV list above the cap (50k bars) must be rejected with 422."""
    bar = {"date": "2022-01-01", "open": 100, "high": 101, "low": 99,
           "close": 100, "volume": 1_000_000}
    resp = client.post(
        "/backtest",
        json={
            "symbol": "X",
            "strategy": "sma_crossover",
            "startDate": "2022-01-01",
            "endDate": "2022-02-01",
            "ohlcv": [bar] * 50_001,
        },
    )
    assert resp.status_code == 422, resp.text


def test_var_rejects_overlong_date_range():
    """A >40-year span on /var must be rejected with 422."""
    resp = client.post(
        "/var",
        json={
            "symbol": "SPY",
            "startDate": "1900-01-01",
            "endDate": "2024-01-01",
        },
    )
    assert resp.status_code == 422, resp.text


# ---------------------------------------------------------------------------
# Money / price / symbol validation: nonsense money inputs, duplicate OHLCV dates and
# /optimize's ["AAPL", "aapl"] as two assets are all client errors.
# ---------------------------------------------------------------------------

BT = {"symbol": "AAPL", "strategy": "sma_crossover",
      "startDate": "2020-01-01", "endDate": "2023-01-01"}


def _bars(n=30, start="2022-01-03"):
    return [
        {"date": d.strftime("%Y-%m-%d"), "open": 100 + i, "high": 101 + i,
         "low": 99 + i, "close": 100 + i, "volume": 1_000_000}
        for i, d in enumerate(pd.bdate_range(start, periods=n))
    ]


@pytest.mark.parametrize(
    "field,value",
    [
        ("commission", -0.05),
        ("commission", 0.5),
        ("initialCapital", 0),
        ("initialCapital", -100_000),
        ("symbol", "A" * 33),
        ("symbol", "   "),
    ],
)
def test_backtest_rejects_nonsense_inputs(field, value):
    resp = client.post("/backtest", json={**BT, field: value})
    assert resp.status_code == 422, resp.text


def test_backtest_rejects_non_finite_money():
    # Python's json accepts NaN/Infinity tokens, so they can arrive on the wire.
    body = '{"symbol":"AAPL","strategy":"sma_crossover","startDate":"2020-01-01",' \
           '"endDate":"2023-01-01","commission":NaN,"initialCapital":Infinity}'
    resp = client.post("/backtest", content=body, headers={"content-type": "application/json"})
    assert resp.status_code == 422, resp.text


def test_backtest_accepts_the_commission_bounds():
    for c in (0.0, 0.1):
        assert client.post("/backtest", json={**BT, "commission": c}).status_code == 200


def test_backtest_rejects_duplicate_ohlcv_dates():
    bars = _bars()
    bars[5]["date"] = bars[4]["date"]
    resp = client.post("/backtest", json={**BT, "ohlcv": bars})
    assert resp.status_code == 422, resp.text
    assert "duplicate dates" in resp.text


@pytest.mark.parametrize(
    "mutate",
    [
        lambda b: b.update(date="not-a-date"),
        lambda b: b.update(close=0),
        lambda b: b.update(low=-1),
        lambda b: b.update(volume=-5),
    ],
    ids=["bad-date", "zero-close", "negative-low", "negative-volume"],
)
def test_backtest_rejects_malformed_bars(mutate):
    bars = _bars()
    mutate(bars[3])
    resp = client.post("/backtest", json={**BT, "ohlcv": bars})
    assert resp.status_code == 422, resp.text


def test_var_rejects_non_positive_portfolio_value():
    for v in (0, -5):
        assert client.post("/var", json={"portfolioValue": v}).status_code == 422


def test_var_and_optimize_reject_reversed_dates():
    assert client.post("/var", json={"startDate": "2024-01-01", "endDate": "2022-01-01"}).status_code == 422
    assert client.post("/optimize", json={"startDate": "2024-01-01", "endDate": "2021-01-01"}).status_code == 422


def test_var_rejects_duplicate_ohlcv_dates():
    bars = _bars()
    bars[2]["date"] = bars[1]["date"]
    assert client.post("/var", json={"ohlcv": bars}).status_code == 422


@pytest.mark.parametrize(
    "symbols",
    [["AAPL", "aapl"], ["AAPL", " AAPL "], ["MSFT"], ["AAPL", ""]],
    ids=["case-duplicate", "whitespace-duplicate", "single", "empty-symbol"],
)
def test_optimize_requires_two_distinct_symbols(symbols):
    resp = client.post("/optimize", json={"symbols": symbols})
    assert resp.status_code == 422, resp.text


def test_optimize_normalises_symbols():
    resp = client.post("/optimize", json={"symbols": [" aapl", "msft "]})
    assert resp.status_code == 200, resp.text
    assert set(resp.json()["weights"]) == {"AAPL", "MSFT"}


def test_optimize_rejects_unknown_objective_and_keys():
    assert client.post("/optimize", json={"objective": "moon"}).status_code == 422
    assert client.post("/optimize", json={"bogus": 1}).status_code == 422


def test_responses_say_their_prices_are_synthetic():
    """Synthetic GBM bars under a real ticker name must be labelled as such."""
    assert client.post("/backtest", json=BT).json()["dataSource"] == "synthetic"
    assert client.post("/var", json={}).json()["dataSource"] == "synthetic"
    assert client.post("/optimize", json={}).json()["dataSource"] == "synthetic"


def test_supplied_bars_are_labelled_supplied():
    bars = _bars(120)  # /var needs >= 60 returns
    bt = client.post("/backtest", json={**BT, "ohlcv": bars}).json()
    assert bt["dataSource"] == "supplied"
    assert client.post("/var", json={"ohlcv": bars}).json()["dataSource"] == "supplied"


# --- metrics: undefined -> null; win statistics on net dollars ---


def test_undefined_metrics_are_null_not_zero():
    """A flat price path: no volatility, no trades. Sharpe, Calmar, win rate
    and profit factor are undefined and must come back as JSON null
    (0.0 would be indistinguishable from "no edge")."""
    bars = [
        {"date": d.strftime("%Y-%m-%d"), "open": 50.0, "high": 50.0, "low": 50.0, "close": 50.0}
        for d in pd.date_range("2021-01-01", periods=120, freq="B")
    ]
    r = client.post("/backtest", json={**BT, "ohlcv": bars})
    assert r.status_code == 200, r.text
    m = r.json()["metrics"]
    assert m["nTrades"] == 0
    # Sortino too: a riskless series has no downside risk, whatever its hurdle shortfall.
    for key in ("sharpe", "sortino", "calmar", "winRate", "profitFactor"):
        assert m[key] is None, (key, m[key])
    assert m["totalReturn"] == 0.0


@pytest.mark.parametrize("symbol", ["AAPL", "QQQ", "TSLA", "SPY", "NVDA", "JPM"])
def test_profit_factor_agrees_with_the_sign_of_the_return(symbol):
    for strategy in ("sma_crossover", "rsi_meanrev", "macd", "ml_momentum"):
        m = client.post("/backtest", json={**BT, "symbol": symbol, "strategy": strategy}).json()["metrics"]
        if m["profitFactor"] is None:
            continue
        assert (m["profitFactor"] > 1) == (m["totalReturn"] > 0), (symbol, strategy, m)


# --- one synthetic price path per symbol ---


def test_every_strategy_runs_on_the_same_price_path_for_a_symbol():
    """Strategies are comparable only if they all run on the same AAPL: buy-and-hold
    must not depend on the strategy picked."""
    req = {"symbol": "AAPL", "startDate": "2020-01-01", "endDate": "2024-01-01"}
    paths = {
        s: [p["benchmark"] for p in client.post("/backtest", json={**req, "strategy": s}).json()["equity"]]
        for s in ("sma_crossover", "rsi_meanrev", "macd", "ml_momentum")
    }
    assert len({tuple(p) for p in paths.values()}) == 1, "strategies ran on different prices"
    # /var prices the same path too.
    from alphaforge.api.app import synthetic_ohlcv
    close = synthetic_ohlcv("AAPL", "2020-01-01", "2024-01-01")["Close"]
    assert paths["macd"][-1] == pytest.approx(100_000 * close.iloc[-1] / close.iloc[0], abs=0.01)


def test_the_feed_depends_on_the_symbol_only():
    from alphaforge.api.app import synthetic_ohlcv
    a = synthetic_ohlcv("AAPL", "2021-01-01", "2022-01-01")
    pd.testing.assert_frame_equal(a, synthetic_ohlcv("aapl", "2021-01-01", "2022-01-01"))
    assert not a["Close"].equals(synthetic_ohlcv("MSFT", "2021-01-01", "2022-01-01")["Close"])


# --- ruin: equity floors at zero ---


def _squeeze_bars():
    import numpy as np

    rng = np.random.default_rng(0)
    rets = list(rng.normal(0.003, 0.02, 80)) + [0.574, 0.271, -0.091, 0.065, -0.020, -0.003, 0.511,
            0.181, 0.927, 1.348, -0.443, 0.679, -0.308, -0.600, 0.192, -0.420] + list(rng.normal(0, 0.03, 40))
    px = 10 * np.cumprod(1 + np.array(rets))
    return [{"date": str(d.date()), "open": p, "high": p, "low": p, "close": p}
            for d, p in zip(pd.bdate_range("2020-09-01", periods=len(px)), px)]


def test_a_short_squeeze_ruins_the_account_instead_of_going_negative():
    """A GME-like path: rsi_meanrev shorts into a +134.8% day. Equity must stop at
    zero, with no negative bars and no later move changing a dead account."""
    r = client.post("/backtest", json={**BT, "symbol": "SQZ", "strategy": "rsi_meanrev",
                                       "startDate": "2020-09-01", "endDate": "2021-06-01",
                                       "ohlcv": _squeeze_bars()})
    assert r.status_code == 200, r.text
    body = r.json()
    eq = [p["equity"] for p in body["equity"]]
    assert min(eq) == 0.0, min(eq)
    assert body["ruined"] is True and body["ruinDate"]
    dates = [p["date"] for p in body["equity"]]
    assert all(e == 0.0 for e in eq[dates.index(body["ruinDate"]):])
    assert body["metrics"]["totalReturn"] == pytest.approx(-1.0)
    assert body["metrics"]["maxDrawdown"] == pytest.approx(-1.0)
    assert all(t["entryDate"] <= body["ruinDate"] for t in body["trades"])
    gap = abs(sum(t["pnlNetAbs"] for t in body["trades"]) + 100_000)
    assert gap <= 0.005 * (len(body["trades"]) + 1)


def test_a_solvent_run_says_so():
    body = client.post("/backtest", json=BT).json()
    assert body["ruined"] is False and body["ruinDate"] is None


# --- dates: wall-clock, exact windows, no future ---


def test_timezone_aware_dates_are_their_wall_clock_date_not_a_500():
    """A tz-aware startDate is read as its wall-clock date on /backtest and /var."""
    naive = client.post("/backtest", json={**BT, "startDate": "2020-01-01"}).json()
    aware = client.post("/backtest", json={**BT, "startDate": "2020-01-01T00:00:00+05:00"})
    assert aware.status_code == 200, aware.text
    assert aware.json()["equity"] == naive["equity"]
    var = client.post("/var", json={"startDate": "2022-01-01T00:00:00Z", "endDate": "2024-01-01"})
    assert var.status_code == 200, var.text
    assert var.json() == client.post("/var", json={"startDate": "2022-01-01", "endDate": "2024-01-01"}).json()


def test_tz_aware_and_mixed_bar_dates_name_their_trading_day():
    bars = _bars(40)
    aware = [{**b, "date": b["date"] + "T00:00:00-05:00"} for b in bars]
    mixed = [aware[i] if i % 2 else bars[i] for i in range(len(bars))]
    want = client.post("/backtest", json={**BT, "ohlcv": bars}).json()["equity"]
    for variant in (aware, mixed):
        r = client.post("/backtest", json={**BT, "ohlcv": variant})
        assert r.status_code == 200, r.text
        assert r.json()["equity"] == want


def test_a_short_span_is_not_padded_past_its_end_date():
    """endDate 2024-01-10 means bars through 2024-01-10, not a span padded past it."""
    r = client.post("/backtest", json={**BT, "startDate": "2024-01-01", "endDate": "2024-01-10"})
    assert r.status_code == 200, r.text
    dates = [p["date"] for p in r.json()["equity"]]
    assert dates[0] == "2024-01-01" and dates[-1] == "2024-01-10" and len(dates) == 8


def test_an_end_date_in_the_future_is_refused():
    future = (pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    for path, body in (("/backtest", {**BT, "endDate": future}),
                       ("/var", {"endDate": future}),
                       ("/optimize", {"endDate": future})):
        r = client.post(path, json=body)
        assert r.status_code == 422 and "future" in r.text, (path, r.status_code)


def test_supplied_bars_honour_the_requested_window():
    """Supplied bars outside [startDate, endDate] are not backtested."""
    bars = _bars(60)  # 2022-01-03 .. 2022-03-25
    r = client.post("/backtest", json={**BT, "startDate": "2022-02-01", "endDate": "2022-03-01", "ohlcv": bars})
    assert r.status_code == 200, r.text
    dates = [p["date"] for p in r.json()["equity"]]
    assert dates[0] == "2022-02-01" and dates[-1] == "2022-03-01"
    empty = client.post("/backtest", json={**BT, "startDate": "2021-01-01", "endDate": "2021-06-01", "ohlcv": bars})
    assert empty.status_code == 422
    # /var's dates have defaults: without an explicit window every bar is used...
    long_bars = _bars(200)  # 2022-01-03 .. 2022-10-07; /var needs >= 60 returns
    full = client.post("/var", json={"ohlcv": long_bars})
    # ...and with one, only the window.
    cut = client.post("/var", json={"ohlcv": long_bars, "startDate": "2022-02-01", "endDate": "2022-07-29"})
    assert full.status_code == 200 and cut.status_code == 200, (full.text, cut.text)
    assert full.json() != cut.json()


def test_fill_timing_is_a_request_option_and_is_echoed():
    close = client.post("/backtest", json=BT).json()
    late = client.post("/backtest", json={**BT, "fill": "next_close"}).json()
    assert close["fill"] == "close" and late["fill"] == "next_close"
    assert close["metrics"]["totalReturn"] != late["metrics"]["totalReturn"]
    assert client.post("/backtest", json={**BT, "fill": "next_open"}).status_code == 422


def test_both_cost_legs_are_request_options_and_are_echoed():
    """Slippage (5 bps per side) is a request field and echoed in the response, so a
    commission=0 run shows the cost it still pays."""
    body = client.post("/backtest", json=BT).json()
    assert body["commission"] == 0.001 and body["slippage"] == 0.0005
    free = client.post("/backtest", json={**BT, "commission": 0.0, "slippage": 0.0}).json()
    assert free["slippage"] == 0.0
    assert all(t["commissionAbs"] == 0.0 for t in free["trades"])
    # Zero costs: the equity curve is the position's gross P&L alone, so it
    # beats the default-cost run.
    assert free["equity"][-1]["equity"] > body["equity"][-1]["equity"]
    for bad in (-0.001, 0.5):
        assert client.post("/backtest", json={**BT, "slippage": bad}).status_code == 422


def test_money_inputs_are_bounded_so_nothing_overflows():
    """initialCapital=1e308 would overflow the equity curve, so it is refused."""
    for value in (1e13, 1e308):
        assert client.post("/backtest", json={**BT, "initialCapital": value}).status_code == 422
        assert client.post("/var", json={"portfolioValue": value}).status_code == 422
    ok = client.post("/backtest", json={**BT, "initialCapital": 1e12})
    assert ok.status_code == 200 and ok.json()["equity"][0]["equity"] == 1e12


# --- minimum observations; empty windows ---

_lenient = TestClient(app, raise_server_exceptions=False)  # so a 500 is a status, not a raise


def test_a_window_with_no_business_days_is_a_422_not_a_500():
    """A Sat-Sun window has no bars: a client error on all three endpoints."""
    weekend = {"startDate": "2021-01-02", "endDate": "2021-01-03"}
    for path, body in (("/backtest", {"symbol": "X", "strategy": "macd", **weekend}),
                       ("/var", weekend), ("/optimize", weekend)):
        r = _lenient.post(path, json=body)
        assert r.status_code == 422, (path, r.status_code, r.text[:120])


def test_var_needs_sixty_returns():
    """2 bars would give a parametric null and a historical 'VaR' of -1000 (a gain)."""
    assert client.post("/var", json={"ohlcv": _bars(2)}).status_code == 422
    assert client.post("/var", json={"ohlcv": _bars(60)}).status_code == 422   # 59 returns
    ok = client.post("/var", json={"ohlcv": _bars(61)})                        # 60 returns
    assert ok.status_code == 200, ok.text
    assert "at least 60 daily returns" in client.post("/var", json={"ohlcv": _bars(10)}).text


def test_optimize_needs_enough_returns_for_its_covariance():
    """2023-12-27 -> 12-29 would put 100% in one asset at Sharpe 49.26."""
    short = client.post("/optimize", json={"startDate": "2023-12-27", "endDate": "2023-12-29"})
    assert short.status_code == 422 and "need at least 60" in short.text
    # 50 symbols need 2 x 50 = 100 returns: 2023-07-01 .. 2023-11-01 has ~88.
    many = client.post("/optimize", json={"symbols": [f"S{i}" for i in range(50)],
                                          "startDate": "2023-07-01", "endDate": "2023-11-01"})
    assert many.status_code == 422 and "need at least 100" in many.text


def test_optimizer_sharpe_is_undefined_not_zero_at_zero_volatility():
    import math

    from alphaforge.optimization.markowitz import MarkowitzOptimizer

    flat = pd.DataFrame({"A": [0.0] * 100, "B": [0.0] * 100})
    assert math.isnan(MarkowitzOptimizer(flat).optimize("equal_weight").sharpe)


# --- request echoes, absurd dates, window edges ---


def test_a_422_does_not_echo_a_huge_request_back():
    bar = {"date": "2022-01-03", "open": 1, "high": 1, "low": 1, "close": 1}
    r = client.post("/backtest", json={**BT, "ohlcv": [bar] * 50_001})
    assert r.status_code == 422
    assert len(r.content) < 20_000, len(r.content)   # the whole body would be ~4.8 MB


@pytest.mark.parametrize("bad", ["2020-01-01\u0000", "0001-01-01", "2150-01-01"])
def test_control_characters_and_absurd_years_are_rejected(bad):
    assert client.post("/backtest", json={**BT, "startDate": bad}).status_code == 422
    bars = [{**b, "date": bad} if i == 0 else b for i, b in enumerate(_bars(30))]
    assert client.post("/backtest", json={**BT, "ohlcv": bars}).status_code == 422


@pytest.mark.parametrize("fill", ["close", "next_close"])
def test_trade_dates_name_the_bar_whose_close_was_filled(fill):
    """Over HTTP, on known prices: entryPrice is close(entryDate) x (1 +/- slippage),
    likewise exitPrice (the still-open last row's mark is the bare close)."""
    import numpy as np

    dates = pd.date_range("2021-01-04", periods=200, freq="B").strftime("%Y-%m-%d").tolist()
    closes = (100 + 15 * np.sin(np.arange(200) / 6) + 0.05 * np.arange(200)).round(4).tolist()
    bars = [{"date": d, "open": c, "high": c, "low": c, "close": c} for d, c in zip(dates, closes)]
    slip = 0.01
    resp = client.post("/backtest", json={
        "symbol": "TEST", "strategy": "macd", "startDate": dates[0], "endDate": dates[-1],
        "commission": 0.001, "slippage": slip, "fill": fill, "ohlcv": bars,
    })
    assert resp.status_code == 200, resp.text
    trades = resp.json()["trades"]
    assert len(trades) >= 3
    close_on = dict(zip(dates, closes))
    for t in trades:
        d = 1 if t["direction"] == "long" else -1
        assert t["entryPrice"] == pytest.approx(close_on[t["entryDate"]] * (1 + d * slip), abs=0.0051)
        factor = 1.0 if t["exitDate"] == dates[-1] else 1 - d * slip
        assert t["exitPrice"] == pytest.approx(close_on[t["exitDate"]] * factor, abs=0.0051)


def test_var_refuses_an_empty_list_of_confidence_levels():
    """A client error: var_report cannot build a table with no rows."""
    resp = client.post("/var", json={"symbol": "SPY", "confidenceLevels": []})
    assert resp.status_code == 422


def test_var_labels_keep_the_precision_of_the_level():
    """0.975 reads "97.5%", not "98%"; 0.999 and 0.9995 do not both read "100%"."""
    resp = client.post("/var", json={"symbol": "SPY", "confidenceLevels": [0.9, 0.95, 0.975, 0.99, 0.999, 0.9995]})
    assert resp.status_code == 200
    assert [r["confidence"] for r in resp.json()["rows"]] == ["90%", "95%", "97.5%", "99%", "99.9%", "99.95%"]


# --- a window is calendar dates, whatever time of day a bar is stamped ---


def test_a_bar_stamped_during_endDate_is_inside_the_window():
    """A bar stamped 16:00 on endDate is inside the window, whose end is that date,
    not that date's midnight."""
    bars = [{**b, "date": b["date"] + "T16:00:00"} for b in _bars(30)]
    start, end = bars[0]["date"][:10], bars[-1]["date"][:10]
    body = client.post("/backtest", json={**BT, "startDate": start, "endDate": end, "ohlcv": bars}).json()
    assert [p["date"] for p in body["equity"]][-1] == end
    assert len(body["equity"]) == 30


def test_a_bar_stamped_before_a_timed_startDate_is_inside_the_window():
    bars = [{**b, "date": b["date"] + "T09:30:00"} for b in _bars(30)]
    start, end = bars[0]["date"][:10], bars[-1]["date"][:10]
    body = client.post("/backtest", json={**BT, "startDate": start + "T16:00:00", "endDate": end, "ohlcv": bars}).json()
    assert len(body["equity"]) == 30


# --- every POST answers 422 for JSON it cannot use, never a 500 -------------

POST_ENDPOINTS = ["/backtest", "/var", "/optimize"]
_nested_list = '{"symbol":' + "[" * 2000 + "]" * 2000 + "}"
_nested_dict = '{"symbol":' + '{"a":' * 2000 + "1" + "}" * 2000 + "}"

UNUSABLE_JSON = {
    "surrogate as a value": r'{"symbol":"\ud800"}',
    "low surrogate": r'{"symbol":"\udfff"}',
    "surrogate as a key": r'{"\ud800":1}',
    "surrogates in a nested object": r'{"symbol":{"\ud800":"\udc00"}}',
    "surrogate in a date": r'{"startDate":"\ud800"}',
    "surrogate in a bar": r'{"ohlcv":[{"date":"\ud800","open":1,"high":1,"low":1,"close":1}]}',
    "surrogate as a bar key": r'{"ohlcv":[{"\ud800":1}]}',
    "surrogates in a list": r'{"symbols":["\ud800","\ud801"]}',
    "a long surrogate run": '{"symbol":"' + r"\ud800" * 500 + '"}',
    "a bare surrogate": r'"\ud800"',
    "nested 2,000 levels": _nested_list,
    "objects nested 2,000 levels": _nested_dict,
}


@pytest.mark.parametrize("endpoint", POST_ENDPOINTS)
@pytest.mark.parametrize("case", sorted(UNUSABLE_JSON))
def test_unusable_json_is_a_422_with_a_json_body(endpoint, case):
    tolerant = TestClient(app, raise_server_exceptions=False)
    resp = tolerant.post(
        endpoint, content=UNUSABLE_JSON[case].encode(), headers={"content-type": "application/json"}
    )
    # Python 3.11's parser gives up on deep nesting before FastAPI validates (its
    # 400); 3.12 parses it (422). Both are clean refusals; a 500 would be the failure.
    assert resp.status_code in ((400, 422) if "nested" in case else (422,)), resp.text[:200]
    assert resp.json()["detail"]


def test_a_rejected_surrogate_is_echoed_as_its_escape():
    resp = TestClient(app, raise_server_exceptions=False).post(
        "/var", content=rb'{"symbol":"\ud800"}', headers={"content-type": "application/json"}
    )
    assert resp.status_code == 422
    assert resp.json()["detail"][0]["input"] == "\\ud800"


def test_var_is_in_dollars_of_the_portfolio_it_was_asked_about():
    """Ten times the portfolio is ten times the loss, and the figures are var_report's."""
    from alphaforge.api.app import synthetic_ohlcv
    from alphaforge.risk.var import var_report

    window = {"symbol": "SPY", "startDate": "2022-01-01", "endDate": "2024-01-01"}
    small = client.post("/var", json={**window, "portfolioValue": 100_000}).json()["rows"]
    large = client.post("/var", json={**window, "portfolioValue": 1_000_000}).json()["rows"]
    fields = ["paramVar1d", "histVar1d", "paramVar10d", "histVar10d", "paramCvar1d"]
    for little, big in zip(small, large):
        for field in fields:
            assert little[field] > 100.0, field
            assert big[field] == pytest.approx(10 * little[field], abs=0.11), field
    returns = synthetic_ohlcv("SPY", "2022-01-01", "2024-01-01")["Close"].pct_change().dropna()
    report = var_report(returns, portfolio_value=1_000_000)
    assert large[1]["paramVar1d"] == pytest.approx(report.loc["95%", "param_var_1d"], abs=0.01)


def test_the_feed_replays_from_the_start_so_dates_select_a_length_not_a_period():
    """What the README says: same bar count, same path, whatever the dates."""
    runs = [
        client.post("/backtest", json={"symbol": "AAPL", "strategy": "sma_crossover",
                                       "startDate": start, "endDate": end}).json()
        for start, end in (("2016-01-01", "2020-01-01"), ("2020-01-01", "2024-01-01"))
    ]
    assert len(runs[0]["equity"]) == len(runs[1]["equity"]) == 1044
    assert runs[0]["metrics"] == runs[1]["metrics"]
    assert runs[0]["metrics"]["nTrades"] == 23
    assert f"{runs[0]['metrics']['totalReturn']:+.2%}" == "+6.67%"
    readme = (Path(__file__).resolve().parent.parent / "README.md").read_text()
    assert "dates select a window\nlength, not a market period" in readme
    assert "same 1,044 bars" in readme and "+6.67% in the same 23 trades" in readme


@pytest.mark.parametrize("symbol", ["AAPL", "MSFT", "TSLA"])
@pytest.mark.parametrize("end", ["2018-06-01", "2022-01-01", "2030-01-01", "2055-01-01"])
def test_a_later_end_date_never_changes_an_earlier_bar(symbol, end):
    """Moving endDate leaves every earlier bar's values alone. 2055 is a window
    longer than the column offsets, which must hold too."""
    from alphaforge.api.app import synthetic_ohlcv

    short = synthetic_ohlcv(symbol, "2015-01-01", "2018-01-01")
    longer = synthetic_ohlcv(symbol, "2015-01-01", end)
    assert len(longer) > len(short)
    pd.testing.assert_frame_equal(short, longer.iloc[: len(short)])


def test_the_documented_xgboost_window_keeps_its_bars():
    """The README's XGBoost figure reads this window's High, Low and Volume: they are
    one running generator's draws, rebuilt here independently of the feed's code."""
    from alphaforge.api.app import _FEED_DRIFT, _FEED_VOL, _seed, synthetic_ohlcv

    df = synthetic_ohlcv("AAPL", "2015-01-01", "2024-01-01")
    n = len(df)
    assert n == 2348
    rng = np.random.default_rng(_seed("AAPL"))
    close = 100.0 * np.exp(np.cumsum(rng.normal(_FEED_DRIFT, _FEED_VOL, n)))
    intraday = np.abs(rng.normal(0.0, _FEED_VOL * 0.5, n))
    volume = rng.uniform(5e5, 5e6, n)
    open_ = np.concatenate([[close[0]], close[:-1]])
    assert (df["Close"].to_numpy() == close).all()
    assert (df["Open"].to_numpy() == open_).all()
    assert (df["High"].to_numpy() == np.maximum.reduce([close * (1 + intraday), open_, close])).all()
    assert (df["Low"].to_numpy() == np.minimum.reduce([close * (1 - intraday), open_, close])).all()
    assert (df["Volume"].to_numpy() == volume).all()


def test_close_is_the_same_in_every_window_including_ones_past_the_offsets():
    """Every price-based figure in the docs reads Close alone: the running
    generator's first draws, whatever the window's length."""
    from alphaforge.api.app import _FEED_DRIFT, _FEED_VOL, _seed, synthetic_ohlcv

    for end in ("2020-01-01", "2024-01-01"):
        close = synthetic_ohlcv("SPY", "1990-01-01", end)["Close"].to_numpy()
        rng = np.random.default_rng(_seed("SPY"))
        assert (close == 100.0 * np.exp(np.cumsum(rng.normal(_FEED_DRIFT, _FEED_VOL, len(close))))).all()


def test_optimize_says_its_weights_are_in_sample():
    """A window's own means and covariance, reported back as "expectedReturn"."""
    body = client.post("/optimize", json={"objective": "equal_weight"}).json()
    assert "In-sample" in body["note"] and "forecast nothing" in body["note"]
    assert body["warning"] is None
    readme = (Path(__file__).resolve().parent.parent / "README.md").read_text()
    assert "**The weights are in-sample.**" in readme


def test_optimize_warns_when_one_asset_takes_everything():
    """100% GOOGL at Sharpe 4.41, fitted to one quarter, must come with something to
    say about it. The figures are the README's."""
    resp = client.post("/optimize", json={"symbols": ["GOOGL", "XOM"], "startDate": "2024-01-01", "endDate": "2024-04-01"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["weights"] == {"GOOGL": 1.0, "XOM": 0.0}
    assert (body["expectedReturn"], round(body["sharpe"], 2)) == (0.8405, 4.41)
    assert "GOOGL takes 100.0%" in body["warning"]
    assert "In-sample" in body["note"]


def test_max_sharpe_is_refused_when_every_asset_loses_to_the_risk_free_rate():
    """TSLA and XOM both lose money over this quarter, so every Sharpe ratio is
    negative and the largest was the least diversified book: max_sharpe answered
    100% XOM. It is a 400 now, with the reason; min_volatility still answers."""
    window = {"symbols": ["TSLA", "XOM"], "startDate": "2024-01-01", "endDate": "2024-04-01"}
    resp = client.post("/optimize", json=window)
    assert resp.status_code == 400, resp.text
    detail = resp.json()["detail"]
    assert "2.00% risk-free rate" in detail and "-49.21% a year" in detail
    assert "min_volatility" in detail
    calm = client.post("/optimize", json={**window, "objective": "min_volatility"})
    assert calm.status_code == 200, calm.text
    assert calm.json()["weights"] == {"TSLA": 0.5209, "XOM": 0.4791}


def test_a_diversified_optimum_carries_no_concentration_warning():
    from alphaforge.optimization.markowitz import concentration_warning

    assert concentration_warning(pd.Series({"A": 0.5, "B": 0.5})) is None
    assert concentration_warning(pd.Series({"A": 0.989, "B": 0.011})) is None
    assert "A takes 99.0%" in concentration_warning(pd.Series({"A": 0.99, "B": 0.01}))


@pytest.mark.parametrize("level", [0.01, 0.05, 0.49])
def test_var_refuses_a_confidence_level_under_a_half(level):
    """confidenceLevels [0.01] returned a "1% VaR" of -$2,919, a gain, HTTP 200."""
    resp = client.post("/var", json={"confidenceLevels": [0.95, level]})
    assert resp.status_code == 422, resp.text
    assert "0.99, not 0.01" in resp.text


def test_var_still_takes_a_half_and_above():
    resp = client.post("/var", json={"confidenceLevels": [0.5, 0.999]})
    assert resp.status_code == 200, resp.text
    assert [row["confidence"] for row in resp.json()["rows"]] == ["50%", "99.9%"]


@pytest.mark.parametrize(
    "path,body",
    [
        ("/backtest", {"symbol": "AAPL", "strategy": "macd", "startDate": "2020-01-01", "endDate": "2021-01-01"}),
        ("/var", {"symbol": "SPY"}),
    ],
)
def test_an_empty_ohlcv_list_is_refused_not_replaced_with_synthetic_bars(path, body):
    """ohlcv: [] is a caller who meant to send bars: it ran on GBM instead."""
    resp = client.post(path, json={**body, "ohlcv": []})
    assert resp.status_code == 422, resp.text
    assert "ohlcv" in resp.text


@pytest.mark.parametrize("scale", [0.002, 0.5, 40.0])
def test_a_trade_log_price_keeps_its_precision_below_a_dollar(scale):
    """A $0.002 close must not log as entryPrice 0.0. Prices from $1 stay at the cent."""
    n = 200
    days = pd.date_range("2021-01-04", periods=n, freq="B")
    closes = [scale * (1 + 0.15 * np.sin(i / 6)) for i in range(n)]
    bars = [
        {"date": d.strftime("%Y-%m-%d"), "open": c, "high": c, "low": c, "close": c}
        for d, c in zip(days, closes)
    ]
    resp = client.post("/backtest", json={
        "symbol": "TEST", "strategy": "macd", "startDate": bars[0]["date"], "endDate": bars[-1]["date"],
        "slippage": 0.0, "ohlcv": bars,
    })
    assert resp.status_code == 200, resp.text
    close = {b["date"]: b["close"] for b in bars}
    trades = resp.json()["trades"]
    assert len(trades) >= 3
    for t in trades:
        for price, date in ((t["entryPrice"], t["entryDate"]), (t["exitPrice"], t["exitDate"])):
            if close[date] >= 1:
                assert price == round(close[date], 2)
            else:
                assert price == pytest.approx(close[date], rel=1e-5)

