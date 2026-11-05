"""
DataFetcher — OHLCV data acquisition with caching.

Primary source: yfinance.
Optional fallback: Alpha Vantage (set ALPHA_VANTAGE_KEY env var).
Cache: Parquet files in ~/.alphaforge/cache/ keyed by symbol+interval+date range.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Optional
from urllib.parse import quote, quote_plus

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_CACHE_DIR = Path.home() / ".alphaforge" / "cache"

# Matches the value of an ``apikey`` query parameter in any URL-bearing text.
_APIKEY_PARAM = re.compile(r"(apikey=)[^&\s'\"]+", re.IGNORECASE)


def _redact(text: str, secret: Optional[str]) -> str:
    """Strip an API key from text that is about to be logged.

    ``requests`` puts the full request URL -- query string included -- into its
    exception messages (``HTTPError`` from ``raise_for_status()``, and the
    ``ConnectionError`` / timeout family via urllib3's "Max retries exceeded
    with url: ..."). Logging such an exception verbatim writes the Alpha
    Vantage key into the log. The key's raw and
    URL-encoded forms are masked, and so is any ``apikey=`` value.
    """
    if secret:
        for form in {secret, quote_plus(secret), quote(secret, safe="")}:
            text = text.replace(form, "***")
    return _APIKEY_PARAM.sub(r"\1***", text)


def _cache_path(symbol: str, start: str, end: str, interval: str) -> Path:
    key = f"{symbol}_{start}_{end}_{interval}"
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    return _CACHE_DIR / f"{symbol}_{interval}_{h}.parquet"


class DataFetcher:
    """
    Fetch OHLCV data from yfinance with optional Alpha Vantage fallback.

    Parameters
    ----------
    use_cache : bool
        If True (default), read/write Parquet cache in ~/.alphaforge/cache/.
    alpha_vantage_key : str | None
        Explicit AV key; falls back to ``ALPHA_VANTAGE_KEY`` env var.
    """

    def __init__(
        self,
        use_cache: bool = True,
        alpha_vantage_key: Optional[str] = None,
        cache: Optional[bool] = None,
    ) -> None:
        # `cache` is a friendlier alias for `use_cache`
        if cache is not None:
            use_cache = cache
        self.use_cache = use_cache
        self._av_key = alpha_vantage_key or os.getenv("ALPHA_VANTAGE_KEY")
        if use_cache:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_ohlcv(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for *symbol* between *start* and *end*.

        Parameters
        ----------
        symbol : str
            Ticker symbol, e.g. "AAPL".
        start : str
            Start date in ISO format "YYYY-MM-DD".
        end : str
            End date in ISO format "YYYY-MM-DD".
        interval : str
            Data frequency: "1d", "1wk", "1mo", or intraday "1h", "5m", etc.

        Returns
        -------
        pd.DataFrame
            Columns: Open, High, Low, Close, Volume.
            Index: DatetimeIndex (UTC for intraday, date-only for daily).
        """
        symbol = symbol.upper().strip()
        # A window that ends today (or later, or open-ended) is still changing:
        # never read or write it from the cache.
        cacheable = self.use_cache and end is not None and \
            pd.Timestamp(end).normalize() < pd.Timestamp.now().normalize()

        if cacheable:
            cached = self._load_cache(symbol, start, end, interval)
            if cached is not None:
                logger.debug("Cache hit: %s %s %s %s", symbol, start, end, interval)
                return cached

        df = self._fetch_yfinance(symbol, start, end, interval)

        if (df is None or df.empty) and self._av_key:
            logger.warning("yfinance returned no data for %s; trying Alpha Vantage.", symbol)
            df = self._fetch_alpha_vantage(symbol, start, end, interval)

        if df is None or df.empty:
            logger.warning(
                "No OHLCV data found for %r between %s and %s; returning empty frame.",
                symbol, start, end,
            )
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

        df = self._clean(df)

        if cacheable:
            self._save_cache(df, symbol, start, end, interval)

        return df

    def fetch_multiple(
        self,
        symbols: list[str],
        start: str,
        end: str,
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV for a list of symbols. Returns {symbol: df} dict."""
        return {sym: self.fetch_ohlcv(sym, start, end, interval) for sym in symbols}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_yfinance(
        self, symbol: str, start: str, end: str, interval: str
    ) -> Optional[pd.DataFrame]:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end, interval=interval, auto_adjust=True)
            if df.empty:
                return None
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            return df
        except Exception as exc:
            logger.error("yfinance error for %s: %s", symbol, exc)
            return None

    def _fetch_alpha_vantage(
        self, symbol: str, start: str, end: str, interval: str
    ) -> Optional[pd.DataFrame]:
        if not self._av_key:
            logger.debug("No Alpha Vantage key; skipping AV fallback.")
            return None
        try:
            import requests

            # Map yfinance interval → AV function
            if interval == "1d":
                func = "TIME_SERIES_DAILY_ADJUSTED"
                ts_key = "Time Series (Daily)"
            elif interval == "1wk":
                func = "TIME_SERIES_WEEKLY_ADJUSTED"
                ts_key = "Weekly Adjusted Time Series"
            else:
                func = "TIME_SERIES_INTRADAY"
                ts_key = f"Time Series ({interval.upper()})"

            params: dict = {
                "function": func,
                "symbol": symbol,
                "apikey": self._av_key,
                "outputsize": "full",
                "datatype": "json",
            }
            if func == "TIME_SERIES_INTRADAY":
                params["interval"] = interval

            resp = requests.get(
                "https://www.alphavantage.co/query", params=params, timeout=30
            )
            resp.raise_for_status()
            data = resp.json()

            if ts_key not in data:
                logger.warning("Alpha Vantage: unexpected response keys: %s", list(data.keys()))
                return None

            # Adjust ALL four prices, as yfinance's auto_adjust=True does: the
            # adjusted endpoints return as-traded open/high/low/close plus an
            # adjusted close, and scaling O/H/L by adjusted/raw close keeps the
            # bar internally consistent: adjusting Close alone could put it below
            # its own Low before any split or dividend. Volume stays as traded.
            # Intraday series carry no adjusted close and are used as returned.
            records = []
            for date_str, vals in data[ts_key].items():
                raw_close = float(vals.get("4. close", 0))
                adj_close = vals.get("5. adjusted close")
                factor = float(adj_close) / raw_close if adj_close is not None and raw_close else 1.0
                records.append(
                    {
                        "Date": pd.Timestamp(date_str),
                        "Open": float(vals.get("1. open", 0)) * factor,
                        "High": float(vals.get("2. high", 0)) * factor,
                        "Low": float(vals.get("3. low", 0)) * factor,
                        "Close": raw_close * factor,
                        "Volume": float(vals.get("6. volume", vals.get("5. volume", 0))),
                    }
                )

            df = pd.DataFrame(records).set_index("Date").sort_index()
            # Filter to requested date range
            df = df.loc[start:end]
            return df

        except Exception as exc:
            logger.error(
                "Alpha Vantage error for %s: %s", symbol, _redact(str(exc), self._av_key)
            )
            return None

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardise column names, drop NaNs, ensure float64."""
        df = df.copy()
        df.columns = [c.strip() for c in df.columns]
        # Ensure expected columns exist
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in df.columns:
                raise ValueError(f"Missing column {col!r} in fetched data.")
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df = df.dropna(subset=["Close"])
        df = df.astype({"Open": float, "High": float, "Low": float, "Close": float, "Volume": float})
        df.index.name = "Date"
        return df

    def _load_cache(
        self, symbol: str, start: str, end: str, interval: str
    ) -> Optional[pd.DataFrame]:
        path = _cache_path(symbol, start, end, interval)
        if path.exists():
            try:
                return pd.read_parquet(path)
            except Exception as exc:
                logger.warning("Cache read failed for %s: %s", path, exc)
        return None

    def _save_cache(
        self,
        df: pd.DataFrame,
        symbol: str,
        start: str,
        end: str,
        interval: str,
    ) -> None:
        path = _cache_path(symbol, start, end, interval)
        try:
            df.to_parquet(path)
        except Exception as exc:
            logger.warning("Cache write failed for %s: %s", path, exc)

    def clear_cache(self, symbol: Optional[str] = None) -> int:
        """
        Remove cached files.

        Parameters
        ----------
        symbol : str | None
            If given, only remove files for that symbol. Otherwise clears all.

        Returns
        -------
        int
            Number of files deleted.
        """
        pattern = f"{symbol.upper()}_*.parquet" if symbol else "*.parquet"
        deleted = 0
        for p in _CACHE_DIR.glob(pattern):
            p.unlink()
            deleted += 1
        return deleted
