"""
Universe — stock universe management and filtering.

Provides curated universes (S&P 500 subset, sector ETFs, custom) and
liquidity/filter methods to whittle them down before strategy runs.

Survivorship bias -- read before backtesting a built-in universe
----------------------------------------------------------------
The built-in lists are TODAY's large caps and sector leaders (curated
2026). Backtesting them over past windows only ever includes companies that
survived and grew into these lists; names that shrank, were acquired or were
delisted are missing, which biases historical returns upward. They are
convenient demo baskets, not point-in-time index membership. For an unbiased
historical test, supply the constituents as of each rebalance date.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


# Pre-defined universes -- current (2026) names; see "Survivorship bias" above.
_UNIVERSES: dict[str, list[str]] = {
    "mega_cap": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
        "META", "TSLA", "BRK-B", "JPM", "JNJ",
    ],
    "tech": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META",
        "TSLA", "AMD", "INTC", "CRM", "ORCL", "ADBE",
    ],
    "finance": [
        "JPM", "BAC", "WFC", "GS", "MS",
        "C", "BLK", "AXP", "USB", "PNC",
    ],
    "etf_sectors": [
        "SPY", "QQQ", "XLK", "XLF", "XLE",
        "XLV", "XLY", "XLP", "XLI", "XLU",
        "XLB", "XLRE", "XLC",
    ],
    "sp500_sample": [
        "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL",
        "META", "TSLA", "BRK-B", "UNH", "JNJ",
        "JPM", "V", "PG", "MA", "HD",
        "CVX", "MRK", "ABBV", "PFE", "KO",
        "PEP", "WMT", "BAC", "DIS", "CSCO",
    ],
}


class Universe:
    """
    Manages a collection of ticker symbols for strategy use.

    Parameters
    ----------
    symbols : list[str] | str
        Explicit list of tickers, or the name of a built-in universe.
    """

    BUILTIN_NAMES = list(_UNIVERSES.keys())
    # Recognised sector names (case-insensitive) -> built-in list. Exact names
    # only: a substring match would file "Biotech" under technology.
    SECTORS: dict[str, str] = {
        "technology": "tech",
        "tech": "tech",
        "information technology": "tech",
        "financials": "finance",
        "financial services": "finance",
        "finance": "finance",
    }

    @classmethod
    def sp500_subset(cls) -> "Universe":
        """Curated S&P 500 sample (25 tickers across sectors)."""
        return cls("sp500_sample")

    @classmethod
    def tech_sector(cls) -> "Universe":
        """Tech sector tickers."""
        return cls("tech")

    @classmethod
    def finance_sector(cls) -> "Universe":
        """Finance sector tickers."""
        return cls("finance")

    @classmethod
    def mega_cap(cls) -> "Universe":
        """Mega-cap US stocks."""
        return cls("mega_cap")

    def filter_sector(self, sector: str) -> "Universe":
        """
        Return a new Universe with THIS universe's symbols in *sector*.

        Recognised sectors (case-insensitive): Technology (also "tech",
        "information technology") and Financials (also "finance", "financial
        services"). Membership is the built-in ``tech`` / ``finance`` list --
        a curated 2026 basket (survivorship-biased, see the module docstring),
        not a live classification. Any other name raises ``ValueError``. The
        result can be empty.
        """
        key = self.SECTORS.get(sector.strip().lower())
        if key is None:
            raise ValueError(
                f"Unknown sector {sector!r}. Recognised: {sorted(self.SECTORS)}"
            )
        base = set(_UNIVERSES[key])
        return Universe([s for s in self._symbols if s in base])

    def __init__(self, symbols: list[str] | str) -> None:
        if isinstance(symbols, str):
            if symbols not in _UNIVERSES:
                raise ValueError(
                    f"Unknown built-in universe {symbols!r}. "
                    f"Available: {list(_UNIVERSES.keys())}"
                )
            self._symbols: list[str] = list(_UNIVERSES[symbols])
        else:
            self._symbols = [s.upper().strip() for s in symbols]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def symbols(self) -> list[str]:
        return list(self._symbols)

    def __len__(self) -> int:
        return len(self._symbols)

    def __repr__(self) -> str:
        n = len(self._symbols)
        preview = ", ".join(self._symbols[:5])
        suffix = f", ... +{n - 5} more" if n > 5 else ""
        return f"Universe([{preview}{suffix}], n={n})"

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter_by_liquidity(
        self,
        price_data: dict[str, pd.DataFrame],
        min_avg_volume: float = 1_000_000,
        min_price: float = 5.0,
    ) -> "Universe":
        """
        Remove illiquid tickers.

        Parameters
        ----------
        price_data : dict[str, pd.DataFrame]
            {symbol: OHLCV DataFrame} already fetched.
        min_avg_volume : float
            Minimum average daily dollar volume.
        min_price : float
            Minimum average closing price.

        Returns
        -------
        Universe
            New universe containing only liquid tickers.
        """
        valid: list[str] = []
        for sym in self._symbols:
            if sym not in price_data:
                logger.warning("Symbol %s not in price_data; skipping liquidity filter.", sym)
                continue
            df = price_data[sym]
            avg_vol = (df["Close"] * df["Volume"]).mean()
            avg_price = df["Close"].mean()
            if avg_vol >= min_avg_volume and avg_price >= min_price:
                valid.append(sym)
            else:
                logger.debug(
                    "Filtered out %s: avg_dollar_vol=%.0f avg_price=%.2f",
                    sym, avg_vol, avg_price,
                )
        return Universe(valid)

    def filter_by_returns(
        self,
        price_data: dict[str, pd.DataFrame],
        min_history_days: int = 252,
    ) -> "Universe":
        """
        Remove tickers with insufficient return history.

        Parameters
        ----------
        price_data : dict[str, pd.DataFrame]
        min_history_days : int
            Minimum number of trading days required (default 252 = ~1 year).

        Returns
        -------
        Universe
        """
        valid = [
            sym
            for sym in self._symbols
            if sym in price_data and len(price_data[sym]) >= min_history_days
        ]
        return Universe(valid)

    def add(self, symbol: str | list[str]) -> "Universe":
        """Return a new Universe with *symbol*(s) added."""
        extras = [symbol] if isinstance(symbol, str) else symbol
        combined = list(dict.fromkeys(self._symbols + [s.upper() for s in extras]))
        return Universe(combined)

    def remove(self, symbol: str | list[str]) -> "Universe":
        """Return a new Universe with *symbol*(s) removed."""
        drops = {symbol.upper()} if isinstance(symbol, str) else {s.upper() for s in symbol}
        return Universe([s for s in self._symbols if s not in drops])

    def intersection(self, other: "Universe") -> "Universe":
        other_set = set(other.symbols)
        return Universe([s for s in self._symbols if s in other_set])

    def to_list(self) -> list[str]:
        return list(self._symbols)


def get_builtin(name: str) -> Universe:
    """Shortcut to retrieve a named built-in universe."""
    return Universe(name)
