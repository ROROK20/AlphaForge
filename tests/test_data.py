"""Tests for data fetching and preprocessing."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from alphaforge.data.preprocessor import (
    add_technical_features,
    normalize,
    create_lagged_features,
    create_target,
)
from alphaforge.data.universe import Universe


def make_ohlcv(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    high = close * (1 + rng.uniform(0, 0.01, n))
    low = close * (1 - rng.uniform(0, 0.01, n))
    return pd.DataFrame({
        "Open": close, "High": high, "Low": low,
        "Close": close, "Volume": rng.integers(100_000, 1_000_000, n).astype(float),
    }, index=pd.date_range("2020-01-01", periods=n, freq="B"))


class TestPreprocessor:
    def test_add_technical_features_adds_columns(self):
        df = make_ohlcv(100)
        result = add_technical_features(df)
        expected_cols = ["SMA_20", "SMA_50", "RSI_14", "MACD", "BB_upper", "BB_lower"]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_add_technical_features_preserves_index(self):
        df = make_ohlcv(100)
        result = add_technical_features(df)
        pd.testing.assert_index_equal(result.index, df.index)

    def test_add_technical_features_no_extra_rows(self):
        df = make_ohlcv(100)
        result = add_technical_features(df)
        assert len(result) == len(df)

    def test_normalize_zscore(self):
        df = make_ohlcv(100)
        df = add_technical_features(df).dropna()
        normalized = normalize(df, method="zscore", columns=["Close"])
        # Z-score columns should have mean ≈ 0, std ≈ 1
        close_norm = normalized["Close"]
        assert abs(close_norm.mean()) < 0.5
        assert abs(close_norm.std() - 1.0) < 0.5

    def test_normalize_minmax(self):
        df = make_ohlcv(100)
        normalized = normalize(df, method="minmax", columns=["Close"])
        assert normalized["Close"].min() >= -0.01
        assert normalized["Close"].max() <= 1.01

    def test_normalize_invalid_method(self):
        df = make_ohlcv()
        with pytest.raises(ValueError):
            normalize(df, method="magic_norm")

    def test_create_lagged_features(self):
        df = make_ohlcv(100)
        result = create_lagged_features(df, n_lags=3)
        # Should add Close_lag_1, Close_lag_2, Close_lag_3
        for lag in range(1, 4):
            assert f"Close_lag_{lag}" in result.columns

    def test_lagged_features_values(self):
        df = make_ohlcv(50)
        result = create_lagged_features(df, n_lags=2)
        # Close_lag_1 at row i should equal Close at row i-1
        assert result["Close_lag_1"].iloc[5] == pytest.approx(df["Close"].iloc[4])

    def test_technical_features_rsi_bounds(self):
        """RSI should be between 0 and 100."""
        df = make_ohlcv(100)
        result = add_technical_features(df).dropna()
        rsi = result["RSI_14"]
        assert (rsi >= 0).all()
        assert (rsi <= 100).all()

    def test_rsi_is_undefined_during_its_warmup(self):
        """RSI(14) needs 14 price changes: bar 0 must not read 0.0 (a NaN change
        coerced by the divide-by-zero guards)."""
        rsi = add_technical_features(make_ohlcv(100))["RSI_14"]
        assert rsi.iloc[:14].isna().all()
        assert rsi.iloc[14:].notna().all()


class TestCreateTarget:
    def test_returns_method(self):
        df = make_ohlcv(100)
        result = create_target(df, horizon=5, method="returns")
        assert "target" in result.columns
        # forward return = Close.shift(-5)/Close - 1; last 5 rows are NaN
        assert result["target"].iloc[:-5].notna().all()
        assert result["target"].iloc[-5:].isna().all()

    def test_direction_method_values(self):
        df = make_ohlcv(100)
        result = create_target(df, horizon=5, method="direction")
        valid = result["target"].dropna().unique()
        assert set(valid).issubset({-1.0, 0.0, 1.0})

    def test_direction_does_not_crash_on_trailing_nan(self):
        # np.sign(fwd_return) leaves NaN in the final `horizon` rows, and casting
        # those to int raises IntCastingNaNError. The sign must stay NaN-safe so
        # the caller's dropna() can clean those rows.
        df = make_ohlcv(60)
        result = create_target(df, horizon=5, method="direction")
        assert result["target"].iloc[-5:].isna().all()
        # The documented downstream transform must succeed after dropna().
        clean = result.dropna()
        mapped = ((clean["target"] + 1) // 2).astype(int)
        assert set(mapped.unique()).issubset({0, 1})

    def test_invalid_method_raises(self):
        df = make_ohlcv(50)
        with pytest.raises(ValueError, match="Unknown target method"):
            create_target(df, horizon=5, method="bogus")


class TestUniverse:
    def test_sp500_subset(self):
        u = Universe.sp500_subset()
        assert len(u.symbols) > 0
        assert all(isinstance(s, str) for s in u.symbols)

    def test_tech_sector(self):
        u = Universe.tech_sector()
        assert "AAPL" in u.symbols or "MSFT" in u.symbols

    def test_custom_universe(self):
        u = Universe(["AAPL", "MSFT", "GOOGL"])
        assert len(u.symbols) == 3

    def test_filter_by_sector(self):
        u = Universe.sp500_subset()
        tech = u.filter_sector("Technology")
        assert len(tech.symbols) > 0

    def test_len(self):
        u = Universe(["AAPL", "MSFT"])
        assert len(u) == 2

    @pytest.mark.parametrize("sector", ["Healthcare", "Energy", "Industrials", "Biotech", "fintech-ish"])
    def test_unknown_sector_is_an_error_not_a_silent_no_op(self, sector):
        """Not the universe unfiltered, and "Biotech" does not match "tech"."""
        with pytest.raises(ValueError, match="Unknown sector"):
            Universe.sp500_subset().filter_sector(sector)

    def test_filter_only_keeps_this_universes_symbols(self):
        """An empty intersection is empty, not the WHOLE sector list."""
        assert Universe.finance_sector().filter_sector("Technology").symbols == []
        assert set(Universe(["AAPL", "JPM", "XOM"]).filter_sector("tech").symbols) == {"AAPL"}
        assert Universe(["AAPL", "JPM"]).filter_sector("Financial Services").symbols == ["JPM"]

    def test_survivorship_bias_is_disclosed(self):
        import alphaforge.data.universe as universe

        assert "survivorship" in universe.__doc__.lower()
        assert "survivorship" in Universe.filter_sector.__doc__.lower()


class TestDataFetcher:
    @patch("yfinance.download")
    def test_fetch_returns_dataframe(self, mock_download):
        """Test that DataFetcher returns a DataFrame when yfinance succeeds."""
        from alphaforge.data.fetcher import DataFetcher
        # Mock yfinance response
        n = 50
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        mock_df = pd.DataFrame({
            "Open": np.ones(n) * 100,
            "High": np.ones(n) * 105,
            "Low": np.ones(n) * 95,
            "Close": np.ones(n) * 102,
            "Volume": np.ones(n) * 1_000_000,
        }, index=dates)
        mock_download.return_value = mock_df

        fetcher = DataFetcher(cache=False)
        result = fetcher.fetch_ohlcv("AAPL", "2023-01-01", "2023-03-31")

        assert isinstance(result, pd.DataFrame)
        assert "Close" in result.columns

    @patch("yfinance.download")
    def test_empty_response_returns_empty_df(self, mock_download):
        from alphaforge.data.fetcher import DataFetcher
        mock_download.return_value = pd.DataFrame()

        fetcher = DataFetcher(cache=False)
        result = fetcher.fetch_ohlcv("FAKESYMBOL", "2023-01-01", "2023-12-31")
        assert isinstance(result, pd.DataFrame)
        assert result.empty


class TestAlphaVantageFallback:
    """Alpha Vantage is the fallback when yfinance returns nothing.

    Mocked at the HTTP transport (``HTTPAdapter.send``), so ``requests`` builds
    the real prepared URL and its real exception messages: the exact text that
    would leak the key into the log.
    """

    KEY = "DUMMY-av_key-0123456789"

    @staticmethod
    def _fetch(key, send):
        import logging
        from alphaforge.data.fetcher import DataFetcher

        fetcher = DataFetcher(use_cache=False, alpha_vantage_key=key)
        with patch.object(DataFetcher, "_fetch_yfinance", return_value=None), \
                patch("requests.adapters.HTTPAdapter.send", autospec=True, side_effect=send):
            logging.getLogger("alphaforge.data.fetcher").setLevel(logging.DEBUG)
            return fetcher.fetch_ohlcv("AAPL", "2024-01-01", "2024-01-31")

    @staticmethod
    def _http_500(adapter, request, **kwargs):
        import requests

        resp = requests.Response()
        resp.status_code = 500
        resp.reason = "Internal Server Error"
        resp.url = request.url  # what raise_for_status() quotes back
        resp.request = request
        return resp

    @staticmethod
    def _connection_error(adapter, request, **kwargs):
        import requests

        # urllib3's wording, as requests re-raises it: the path WITH its query.
        raise requests.ConnectionError(
            "HTTPSConnectionPool(host='www.alphavantage.co', port=443): Max retries "
            f"exceeded with url: {request.path_url} (Caused by NameResolutionError)",
            request=request,
        )

    @pytest.mark.parametrize("key", [KEY, "dummy+key/with=special&chars"])
    @pytest.mark.parametrize("failure", ["_http_500", "_connection_error"])
    def test_api_key_never_reaches_the_log(self, caplog, key, failure):
        from urllib.parse import quote, quote_plus

        caplog.set_level("DEBUG")
        df = self._fetch(key, getattr(self, failure))
        assert df.empty
        logged = "\n".join(r.getMessage() for r in caplog.records)
        assert "Alpha Vantage error" in logged, "the failure must still be logged"
        for form in (key, quote_plus(key), quote(key, safe="")):
            assert form not in logged, f"API key leaked into the log as {form!r}"
        assert "apikey=***" in logged, "the URL stays in the log, minus the key"

    def test_adjusted_close_adjusts_open_high_low_too(self):
        """yfinance (auto_adjust=True) adjusts all four prices; the fallback
        must too. Adjusting Close alone would return this bar as O=190 H=210 L=180
        C=100: Close below its own Low."""
        import json
        import requests

        payload = {"Time Series (Daily)": {
            "2024-01-02": {"1. open": "190", "2. high": "210", "3. low": "180",
                           "4. close": "200", "5. adjusted close": "100",
                           "6. volume": "5000", "7. dividend amount": "0",
                           "8. split coefficient": "1.0"},
            "2024-01-03": {"1. open": "50", "2. high": "60", "3. low": "40",
                           "4. close": "55", "5. adjusted close": "55",
                           "6. volume": "7000", "7. dividend amount": "0",
                           "8. split coefficient": "1.0"},
        }}

        def ok(adapter, request, **kwargs):
            resp = requests.Response()
            resp.status_code = 200
            resp._content = json.dumps(payload).encode()
            resp.url = request.url
            resp.request = request
            return resp

        df = self._fetch(self.KEY, ok)
        first = df.loc["2024-01-02"]
        assert (first["Open"], first["High"], first["Low"], first["Close"]) == (95, 105, 90, 100)
        assert first["Volume"] == 5000, "volume stays as traded"
        assert (df["Low"] <= df["Close"]).all() and (df["Close"] <= df["High"]).all()
        unadjusted = df.loc["2024-01-03"]
        assert (unadjusted["Open"], unadjusted["Close"]) == (50, 55)


class TestCacheFreshness:
    """A window ending today is still changing: it is never served from, or
    written to, the cache."""

    def test_a_window_ending_today_is_not_cached(self, tmp_path, monkeypatch):
        import alphaforge.data.fetcher as fetcher

        monkeypatch.setattr(fetcher, "_CACHE_DIR", tmp_path)
        frame = pd.DataFrame({"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0], "Volume": [1.0]},
                             index=pd.DatetimeIndex([pd.Timestamp("2024-01-02")]))
        calls = []
        monkeypatch.setattr(fetcher.DataFetcher, "_fetch_yfinance",
                            lambda self, *a, **k: calls.append(a) or frame.copy())
        today = pd.Timestamp.now().strftime("%Y-%m-%d")
        f = fetcher.DataFetcher(use_cache=True)
        f.fetch_ohlcv("AAPL", "2024-01-01", today)
        f.fetch_ohlcv("AAPL", "2024-01-01", today)
        assert len(calls) == 2                          # fetched twice, not served stale
        assert not list(tmp_path.glob("*.parquet"))
        f.fetch_ohlcv("AAPL", "2024-01-01", "2024-01-31")
        f.fetch_ohlcv("AAPL", "2024-01-01", "2024-01-31")
        assert len(calls) == 3                          # a closed window caches as before
        assert list(tmp_path.glob("*.parquet"))


def test_yfinance_prices_are_the_split_and_dividend_adjusted_ones(monkeypatch):
    """The docs promise adjusted prices. A fake Ticker answers as Yahoo does: raw
    prices (a 2:1 split between the bars) unless auto_adjust is on."""
    from alphaforge.data import fetcher as fetcher_module
    from alphaforge.data.fetcher import DataFetcher

    index = pd.date_range("2024-01-01", periods=3, freq="B")
    raw = pd.DataFrame(
        {"Open": [200.0, 100.0, 101.0], "High": [201.0, 101.0, 102.0], "Low": [199.0, 99.0, 100.0],
         "Close": [200.0, 100.0, 101.0], "Volume": 1e6, "Adj Close": [100.0, 100.0, 101.0]},
        index=index,
    )

    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, start, end, interval, auto_adjust):
            if not auto_adjust:
                return raw.copy()
            factor = raw["Adj Close"] / raw["Close"]
            adjusted = raw.drop(columns="Adj Close").copy()
            for col in ("Open", "High", "Low", "Close"):
                adjusted[col] = raw[col] * factor
            return adjusted

    monkeypatch.setattr(fetcher_module.yf, "Ticker", FakeTicker)
    df = DataFetcher(use_cache=False).fetch_ohlcv("SPLIT", "2024-01-01", "2024-01-04")
    assert df["Close"].tolist() == [100.0, 100.0, 101.0]
    assert df["Open"].iloc[0] == 100.0 and df["High"].iloc[0] == 100.5
    assert df["Close"].pct_change().iloc[1] == 0.0, "a split is not a -50% return"

