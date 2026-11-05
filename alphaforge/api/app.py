"""
AlphaForge REST API — FastAPI layer over the REAL engine.

This module connects the Next.js frontend to the AlphaForge engine. Every
endpoint runs actual computation:

* ``POST /backtest`` → real :class:`BacktestEngine` driven by a real technical
  :class:`Signal`, over synthetic-or-supplied OHLCV. Returns the equity curve,
  drawdown series, metrics and trade log in exactly the camelCase shape the
  frontend charts/components consume (see ``frontend/lib/types.ts``).
* ``POST /var``      → real :func:`alphaforge.risk.var.var_report`.
* ``POST /optimize`` → real :class:`alphaforge.optimization.markowitz.MarkowitzOptimizer`.

Synthetic OHLCV is generated deterministically per SYMBOL (one price path per
ticker, shared by every strategy and endpoint, so strategies are comparable)
using a geometric-Brownian-motion process, so the demo is reproducible and
never requires network access or API keys. Synthetic prices are NOT market data for
the ticker named: every response says which it used in ``dataSource``
(``"synthetic"`` or ``"supplied"``). If a caller supplies their own OHLCV bars
they are used verbatim — the engine does not care where the prices come from.

Run locally::

    uvicorn alphaforge.api.app:app --reload --port 8003
    # or
    python -m alphaforge.api.app
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Annotated, Any, Literal, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

# ---------------------------------------------------------------------------
# Input bounds (DoS guards for unauthenticated endpoints)
# ---------------------------------------------------------------------------
# These cap untrusted inputs so a single request cannot hang the server or
# materialise a huge in-memory frame. They are deliberately generous: legitimate
# callers stay well under them.
MAX_OHLCV_BARS = 50_000          # ~190 years of business days; real data is far smaller
MAX_DATE_SPAN_DAYS = 40 * 366    # ~40 years; bars are generated business-daily
MAX_OPTIMIZE_SYMBOLS = 50        # a portfolio of >50 tickers is not a demo workload
MAX_CONFIDENCE_LEVELS = 20       # 90/95/99 is typical; 20 is already excessive
MAX_SYMBOL_LENGTH = 32           # real tickers are <= ~10 chars ("BRK-B", "^GSPC")
MAX_COMMISSION = 0.1             # 10% per side: anything above is not a cost model
# Minimum observations per endpoint. Below these the outputs are not statistics:
# a 2-bar /var would return a "VaR" that is a gain, and a 3-day /optimize would put
# 100% in one asset at Sharpe 49. /var and /optimize share their minimums with the
# library (risk.var, optimization.markowitz), so the CLI refuses the same inputs.
MIN_BACKTEST_BARS = 2
MAX_PERIODS_PER_YEAR = 31_536_000  # one bar a second, all year
MAX_MONEY = 1e12                 # $1 trillion: far above any real account, and
                                 # 1e308 would overflow the equity curve

# Money and prices must be finite and positive (commission: in [0, 10%]). Unbounded,
# commission=-0.05 would *pay* the strategy on every trade, for a total return in
# the hundreds of percent.
Symbol = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_SYMBOL_LENGTH),
]
PositiveMoney = Annotated[float, Field(gt=0, le=MAX_MONEY, allow_inf_nan=False)]
# Bars per year. Null reads it from the bars' spacing, and a spacing it cannot
# name (intraday) is a 422: see alphaforge.backtest.metrics.
PeriodsPerYear = Annotated[float, Field(gt=0, le=MAX_PERIODS_PER_YEAR, allow_inf_nan=False)]
Price = Annotated[float, Field(gt=0, allow_inf_nan=False)]
DataSource = Literal["synthetic", "supplied"]
Fill = Literal["close", "next_close"]
Objective = Literal[
    "max_sharpe", "min_volatility", "risk_parity", "max_diversification", "equal_weight"
]

from alphaforge.backtest.engine import BacktestEngine
from alphaforge.backtest.metrics import compute_drawdown_series, infer_periods_per_year
from alphaforge.optimization.markowitz import (
    IN_SAMPLE_NOTE,
    MarkowitzOptimizer,
    concentration_warning,
    min_observations,
)
from alphaforge.risk.var import MIN_VAR_RETURNS, var_report
from alphaforge.signals.technical import (
    MACDSignal,
    RSIMeanReversionSignal,
    SMACrossoverSignal,
)

app = FastAPI(
    title="AlphaForge API",
    version="0.1.0",
    description="HTTP interface to the real AlphaForge quant engine.",
)

# Wide-open CORS: this is a public demo backend with no auth and no secrets.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _defined(value: Any) -> Optional[float]:
    """A metric as JSON: finite float, or None (null) when undefined/absent."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _json_safe(obj: Any) -> Any:
    """What a 422 can send back: non-finite floats become strings (JSON has no
    NaN), and a lone surrogate, which a JSON ``\\ud800`` escape can carry in but
    UTF-8 cannot carry out, becomes the text of that escape."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return str(obj)
    if isinstance(obj, str):
        return obj.encode("utf-8", "backslashreplace").decode("utf-8")
    if isinstance(obj, dict):
        return {_json_safe(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _echo(rejected: Any) -> Any:
    """The rejected input as a 422 may repeat it: at most 200 characters, so a
    50,000-bar body does not come back in full, 4.8 MB, inside its own 422."""
    try:
        safe = _json_safe(jsonable_encoder(rejected))
        shown = json.dumps(safe, default=str)
    except RecursionError:
        return "(not echoed: nested too deeply)"
    return safe if len(shown) <= 200 else shown[:200] + f"... ({len(shown):,} characters, truncated)"


@app.exception_handler(RequestValidationError)
async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """FastAPI's default 422 body echoes the rejected input, and some of what
    Python's json accepts on the way in cannot go back out: a NaN or Infinity, a
    lone surrogate (``{"symbol": "\\ud800"}``) or input nested about 2,000 levels
    deep. Echoed as is, each would turn a correctly rejected request into an HTTP
    500, so the body is made safe here."""
    errors = []
    for err in exc.errors():
        err = dict(err)
        echoed = {"input": _echo(err.pop("input"))} if "input" in err else {}
        errors.append({**_json_safe(jsonable_encoder(err)), **echoed})
    return JSONResponse(status_code=422, content={"detail": errors})


# ---------------------------------------------------------------------------
# Request / response models  (camelCase to match the frontend exactly)
# ---------------------------------------------------------------------------

# Strategy ids exactly as the frontend sends them (frontend/lib/types.ts)
Strategy = Literal["sma_crossover", "rsi_meanrev", "macd", "ml_momentum"]


class OHLCVBar(BaseModel):
    date: str
    open: Price
    high: Price
    low: Price
    close: Price
    volume: float = Field(default=0.0, ge=0, allow_inf_nan=False)

    @field_validator("date")
    @classmethod
    def _check_date(cls, v: str) -> str:
        try:
            _calendar_ts(v)
        except Exception as exc:
            raise ValueError(f"invalid bar date {v!r}: {exc}") from exc
        return v


def _calendar_ts(value: str) -> pd.Timestamp:
    """A request or bar date as a tz-naive Timestamp: its WALL-CLOCK value.

    A timezone offset, if one is given, is dropped rather than converted:
    "2020-01-01T00:00:00+05:00" is 1 January (as written), not 31 December
    UTC. That is also how exchange-local bar stamps (e.g. yfinance's
    America/New_York midnights) name their trading day, and it lets tz-aware and
    tz-naive dates be compared and mixed.
    """
    if any(ord(ch) < 32 for ch in value):
        raise ValueError(f"date {value!r} contains a control character")
    ts = pd.Timestamp(value)
    ts = ts.tz_localize(None) if ts.tzinfo is not None else ts
    if not 1900 <= ts.year <= 2100:
        raise ValueError(f"date {value!r} is outside 1900-2100")
    return ts


def _check_bars(bars: Optional[list[OHLCVBar]]) -> None:
    """Reject supplied bars the engine cannot index: duplicate dates.

    A duplicated date makes the engine's per-date lookups ambiguous; it is a
    client error, not a server one.
    """
    if not bars:
        return
    stamps = pd.DatetimeIndex([_calendar_ts(b.date) for b in bars])
    dupes = stamps[stamps.duplicated()].unique()
    if len(dupes):
        shown = ", ".join(d.strftime("%Y-%m-%d") for d in dupes[:5])
        raise ValueError(f"ohlcv has duplicate dates: {shown}")


def _check_date_order(start: str, end: str) -> None:
    try:
        start_ts = _calendar_ts(start)
        end_ts = _calendar_ts(end)
    except Exception as exc:
        raise ValueError(f"Invalid date: {exc}") from exc
    if start_ts >= end_ts:
        raise ValueError("startDate must be strictly before endDate")
    # A backtest cannot end after today: the synthetic feed would otherwise
    # produce "prices" for days that have not happened.
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    if end_ts.normalize() > today:
        raise ValueError(f"endDate {end_ts:%Y-%m-%d} is in the future (today is {today:%Y-%m-%d} UTC)")
    # Reject absurd spans BEFORE any bar generation: a 1900-2024 range would
    # materialise tens of thousands of synthetic bars and hang the engine.
    if (end_ts - start_ts).days > MAX_DATE_SPAN_DAYS:
        raise ValueError(
            f"Date range too large: spans must be at most "
            f"{MAX_DATE_SPAN_DAYS} days (~40 years)"
        )


class BacktestRequest(BaseModel):
    """Mirrors the ``BacktestParams`` interface in the frontend."""

    model_config = ConfigDict(extra="forbid")

    symbol: Symbol
    strategy: Strategy
    startDate: str = Field(...)
    endDate: str = Field(...)
    initialCapital: PositiveMoney = 100_000.0
    # Per-side fraction of traded notional (0.001 = 10 bps); a reversal pays
    # it twice. The frontend sends commission/100.
    commission: float = Field(default=0.001, ge=0, le=MAX_COMMISSION, allow_inf_nan=False)
    # Per-side slippage as a fraction of price (0.0005 = 5 bps), charged on the
    # same traded notional. A request field and echoed in the response, so a run
    # with commission=0 shows the cost it still pays.
    slippage: float = Field(default=0.0005, ge=0, le=MAX_COMMISSION, allow_inf_nan=False)
    # When a signal's position fills: "close" = at the close of the bar the
    # signal was computed on (market-on-close, the default); "next_close" = a
    # bar later. See BacktestEngine, "Execution model".
    fill: Fill = "close"
    # Optional caller-supplied OHLCV; if omitted, synthetic bars are generated.
    # Length is capped to prevent a multi-MB body from exhausting memory. An empty
    # list is refused rather than read as "no bars supplied".
    ohlcv: Optional[list[OHLCVBar]] = Field(default=None, min_length=1, max_length=MAX_OHLCV_BARS)
    periodsPerYear: Optional[PeriodsPerYear] = None

    @model_validator(mode="after")
    def _check(self) -> "BacktestRequest":
        _check_date_order(self.startDate, self.endDate)
        _check_bars(self.ohlcv)
        return self


class EquityPoint(BaseModel):
    date: str
    equity: float
    benchmark: float
    drawdown: float


class TradeOut(BaseModel):
    id: int
    # The bars whose closes are entryPrice and exitPrice (see BacktestEngine,
    # "Trade-log dates are fill dates"). A position still open on the last bar,
    # or ruined, exits at that bar's bare close, with no slippage.
    entryDate: str
    exitDate: str
    direction: Literal["long", "short"]
    entryPrice: float
    exitPrice: float
    # pnlPct is the trade's price-based return between the slippage-adjusted
    # entry and exit fills. pnlAbs is its dollars: its shares x the price moves
    # it held, less slippage. Do not derive one from the other: a resized trade's
    # dollars reflect every size it held, and pnlPct only its first and last
    # fills (see the accounting notes in BacktestEngine._build_trades).
    pnlPct: float
    pnlAbs: float
    # Commission attributed per trade; pnlAbs is net-of-slippage but gross of
    # commission (see BacktestEngine._build_trades accounting notes).
    commissionAbs: float
    pnlNetAbs: float


class MetricsOut(BaseModel):
    totalReturn: float
    cagr: float
    # null = undefined (zero denominator: no volatility / no downside / no
    # drawdown / no losing trade / no trades). Never 0, which reads as "no
    # edge" -- see alphaforge.backtest.metrics.
    sharpe: Optional[float]
    sortino: Optional[float]
    calmar: Optional[float]
    maxDrawdown: float
    volatility: float
    # Win rate and profit factor are on net dollar P&L, after costs.
    winRate: Optional[float]
    profitFactor: Optional[float]
    nTrades: int


class BacktestResponse(BaseModel):
    source: Literal["engine"] = "engine"
    # "synthetic" = seeded GBM bars, NOT market data for `symbol`.
    dataSource: DataSource
    symbol: str
    strategy: str
    # The signal actually executed (differs from `strategy` when a demo proxy
    # runs in place of a model-backed strategy) + human-readable disclosure.
    strategyExecuted: str
    fill: Fill
    # The per-side costs actually charged (fractions of traded notional).
    commission: float
    slippage: float
    note: Optional[str] = None
    # True when equity reached zero: the account was liquidated on ruinDate
    # and nothing traded afterwards (equity stays 0; metrics cover the path
    # through ruinDate). See BacktestEngine, "Ruin".
    ruined: bool = False
    ruinDate: Optional[str] = None
    equity: list[EquityPoint]
    trades: list[TradeOut]
    metrics: MetricsOut


class VarRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: Symbol = "SPY"
    startDate: str = "2022-01-01"
    endDate: str = "2024-01-01"
    portfolioValue: PositiveMoney = 100_000.0
    # At least one, capped length; each level in [0.5, 1); null means the defaults.
    # Below 0.5 is a typo for the tail, not a confidence: confidenceLevels [0.01],
    # meaning 99%, would answer with a "1% VaR" of -$2,919, a gain.
    confidenceLevels: Optional[list[float]] = Field(
        default=None, min_length=1, max_length=MAX_CONFIDENCE_LEVELS
    )
    ohlcv: Optional[list[OHLCVBar]] = Field(default=None, min_length=1, max_length=MAX_OHLCV_BARS)
    periodsPerYear: Optional[PeriodsPerYear] = None

    @field_validator("confidenceLevels")
    @classmethod
    def _check_confidence_levels(
        cls, v: Optional[list[float]]
    ) -> Optional[list[float]]:
        if v is not None:
            for level in v:
                if not (0.5 <= level < 1.0):
                    raise ValueError(
                        f"confidence level {level} is outside [0.5, 1): 99% is 0.99, not 0.01"
                    )
        return v

    @model_validator(mode="after")
    def _check(self) -> "VarRequest":
        _check_date_order(self.startDate, self.endDate)
        _check_bars(self.ohlcv)
        return self


class VarRow(BaseModel):
    confidence: str
    paramVar1d: float
    histVar1d: float
    paramVar10d: float
    histVar10d: float
    paramCvar1d: float


class VarResponse(BaseModel):
    source: Literal["engine"] = "engine"
    # "synthetic" = seeded GBM bars, NOT market data for `symbol`.
    dataSource: DataSource
    symbol: str
    portfolioValue: float
    rows: list[VarRow]


class OptimizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Symbol list is capped: each symbol triggers a synthetic-OHLCV generation,
    # so an unbounded list (e.g. 800 symbols) hangs the request.
    symbols: list[Symbol] = Field(
        default_factory=lambda: ["AAPL", "MSFT", "GOOGL", "SPY"],
        max_length=MAX_OPTIMIZE_SYMBOLS,
    )
    objective: Objective = "max_sharpe"
    startDate: str = "2021-01-01"
    endDate: str = "2024-01-01"

    @field_validator("symbols")
    @classmethod
    def _distinct_symbols(cls, v: list[str]) -> list[str]:
        # Normalise BEFORE counting: ["AAPL", "aapl"] would pass a raw len() >= 2
        # check, collapse to one key, and return a 100%-AAPL "portfolio".
        normalised = [s.upper() for s in v]
        dupes = sorted({s for s in normalised if normalised.count(s) > 1})
        if dupes:
            raise ValueError(f"duplicate symbols after upper-casing: {dupes}")
        if len(normalised) < 2:
            raise ValueError("need at least 2 distinct symbols to optimize a portfolio")
        return normalised

    @model_validator(mode="after")
    def _check(self) -> "OptimizeRequest":
        _check_date_order(self.startDate, self.endDate)
        return self


class OptimizeResponse(BaseModel):
    source: Literal["engine"] = "engine"
    # Always "synthetic" today: /optimize takes no price input.
    dataSource: DataSource
    objective: str
    weights: dict[str, float]
    expectedReturn: float
    volatility: Optional[float]
    # null = undefined (zero volatility), never 0.
    sharpe: Optional[float]
    # Every figure above is in-sample over [startDate, endDate]; says so.
    note: str = IN_SAMPLE_NOTE
    # Set when one asset takes ~100% of the portfolio.
    warning: Optional[str] = None


# ---------------------------------------------------------------------------
# Synthetic OHLCV generation (deterministic, no network, no keys)
# ---------------------------------------------------------------------------

# ONE price process for every symbol and strategy: daily log-return drift and
# volatility of the GBM. Strategies compare only if they run on the same "AAPL",
# so nothing about the strategy reaches the seed or these two values.
_FEED_DRIFT = 0.0005
_FEED_VOL = 0.012
# Where High/Low and Volume start reading the symbol's generator, in normal draws
# from its start (Close reads from draw 0). Fixed, so a bar's values do not depend
# on how many bars follow it: with a running generator, moving endDate would change
# every earlier bar's High, Low and Volume, and a model on volume features would
# disagree with itself about the past. 2,348 is the bar count of the README's
# XGBoost window. The cost of a fixed offset: in a window over 2,348 bars (9 years),
# a bar's High/Low range is half the SIZE, never the sign, of the Close shock 2,348
# bars later.
_FEED_RANGE_SKIP = 2_348
_FEED_VOLUME_SKIP = 2 * _FEED_RANGE_SKIP


def _seed(symbol: str) -> int:
    """Stable 32-bit FNV-1a seed from the (upper-cased) symbol alone."""
    h = 2166136261
    for ch in symbol.upper():
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def _feed_draws(symbol: str, skip: int) -> np.random.Generator:
    """The symbol's generator, *skip* normal draws in."""
    rng = np.random.default_rng(_seed(symbol))
    rng.normal(size=skip)
    return rng


def synthetic_ohlcv(symbol: str, start: str, end: str) -> pd.DataFrame:
    """The demo feed: a reproducible GBM OHLCV frame for *symbol*.

    Business-daily bars from *start* to *end*; the path depends only on the
    symbol (case-insensitive) and the dates, never on the strategy or the
    endpoint asking. NOT market data for the ticker named.

    One fixed path per symbol REPLAYS from *start*: bar k is the same bar in
    every window, whatever its date, so dates select a window length rather than
    a market period. A later *end* only appends bars; every earlier one, in all
    five columns, stays as it was.
    """
    try:
        start_ts = _calendar_ts(start)
        end_ts = _calendar_ts(end)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(400, f"Invalid date range: {exc}") from exc

    # Exactly the business days in [start, end]: a short span is not padded.
    dates = pd.date_range(start_ts, end_ts, freq="B")

    n = len(dates)
    if n == 0:
        # A window with no business days (e.g. a weekend): an empty frame, which
        # callers answer with a 422.
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"],
                            index=pd.DatetimeIndex([]), dtype=float)
    daily = _feed_draws(symbol, 0).normal(_FEED_DRIFT, _FEED_VOL, n)
    close = 100.0 * np.exp(np.cumsum(daily))
    intraday = np.abs(_feed_draws(symbol, _FEED_RANGE_SKIP).normal(0.0, _FEED_VOL * 0.5, n))
    high = close * (1 + intraday)
    low = close * (1 - intraday)
    open_ = np.concatenate([[close[0]], close[:-1]])
    volume = _feed_draws(symbol, _FEED_VOLUME_SKIP).uniform(5e5, 5e6, n)

    return pd.DataFrame(
        {
            "Open": open_,
            "High": np.maximum.reduce([high, open_, close]),
            "Low": np.minimum.reduce([low, open_, close]),
            "Close": close,
            "Volume": volume,
        },
        index=dates,
    )


def _trade_price(price: float) -> float:
    """A trade-log price for the wire: to the cent from $1 up, six significant
    digits below it, so a $0.002 close does not log as 0.0."""
    return round(price, 2) if abs(price) >= 1 else float(f"{price:.6g}")


def _bars_to_df(bars: list[OHLCVBar]) -> pd.DataFrame:
    idx = pd.DatetimeIndex([_calendar_ts(b.date) for b in bars])
    return pd.DataFrame(
        {
            "Open": [b.open for b in bars],
            "High": [b.high for b in bars],
            "Low": [b.low for b in bars],
            "Close": [b.close for b in bars],
            "Volume": [b.volume for b in bars],
        },
        index=idx,
    ).sort_index()


def _periods_per_year(requested: Optional[float], index: pd.Index) -> float:
    """Bars per year for this request; a spacing that cannot name it is a 422."""
    if requested is not None:
        return requested
    try:
        return infer_periods_per_year(index)
    except ValueError as exc:
        raise HTTPException(422, str(exc).replace("periods_per_year", "periodsPerYear")) from exc


def _window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """The rows of *df* dated within [start, end], inclusive, as calendar dates.

    A bar's time of day does not matter: a bar stamped 16:00 on ``end`` is inside
    the window, though it falls after that date's midnight.
    """
    days = df.index.normalize()
    return df[(days >= _calendar_ts(start).normalize()) & (days <= _calendar_ts(end).normalize())]


ML_MOMENTUM_NOTE = (
    "ml_momentum's XGBoost model retrains in a purged walk-forward on every run: "
    "seconds of CPU for a 4-year window and minutes at this API's 40-year limit, "
    "which this unauthenticated endpoint does not spend per request. The run below "
    "used an SMA(10/30) crossover proxy instead — the results are real backtest "
    "output but are NOT ML-driven."
)


def _build_signal(strategy: str) -> tuple[Any, str, Optional[str]]:
    """Map a frontend strategy id onto a real technical Signal.

    Returns ``(signal, executed_id, note)``. ``ml_momentum``'s real model
    retrains an XGBoost classifier per walk-forward block on every run. Measured
    2026-11-04 on one thread: 11 s for the demo's
    4-year window and 310 s at the 40-year limit, against 10 ms for an SMA
    run. The input bounds above exist so that one unauthenticated request
    cannot hang the server, so the demo proxies it with an SMA(10/30) crossover
    and DISCLOSES that in the response (executed_id + note), as the CLI does.
    """
    if strategy == "sma_crossover":
        return SMACrossoverSignal(fast=20, slow=50), "sma_crossover(20,50)", None
    if strategy == "rsi_meanrev":
        return RSIMeanReversionSignal(period=14), "rsi_meanrev(14)", None
    if strategy == "macd":
        return MACDSignal(), "macd(12,26,9)", None
    if strategy == "ml_momentum":
        return SMACrossoverSignal(fast=10, slow=30), "sma_crossover(10,30)", ML_MOMENTUM_NOTE
    raise HTTPException(400, f"Unknown strategy: {strategy!r}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "name": "AlphaForge API",
        "version": "0.1.0",
        "engine": "real",
        "endpoints": ["/health", "/backtest", "/var", "/optimize"],
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "engine": "alphaforge",
        "time": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/backtest", response_model=BacktestResponse)
def backtest(req: BacktestRequest) -> BacktestResponse:
    """Run a REAL backtest and return charts-ready, camelCase JSON."""
    if req.ohlcv:
        # The window is honoured exactly: only supplied bars inside
        # [startDate, endDate] are used.
        df = _window(_bars_to_df(req.ohlcv), req.startDate, req.endDate)
    else:
        df = synthetic_ohlcv(req.symbol, req.startDate, req.endDate)

    if len(df) < MIN_BACKTEST_BARS:
        raise HTTPException(
            422,
            f"Not enough data to run a backtest: need at least {MIN_BACKTEST_BARS} bars in "
            f"[startDate, endDate], got {len(df)}.",
        )

    signal, executed_id, note = _build_signal(req.strategy)
    engine = BacktestEngine(
        initial_capital=float(req.initialCapital),
        commission=float(req.commission),
        slippage=float(req.slippage),
        fill=req.fill,
        periods_per_year=_periods_per_year(req.periodsPerYear, df.index),
    )
    result = engine.run(signal, df)

    equity_curve = result.equity_curve
    drawdown = compute_drawdown_series(equity_curve)

    # Buy-and-hold benchmark on the same prices, same starting capital.
    close = df["Close"].astype(float)
    bench = float(req.initialCapital) * (close / close.iloc[0])
    bench = bench.reindex(equity_curve.index).ffill()

    equity_points = [
        EquityPoint(
            date=idx.strftime("%Y-%m-%d"),
            equity=round(float(equity_curve.loc[idx]), 2),
            benchmark=round(float(bench.loc[idx]), 2),
            drawdown=round(float(drawdown.loc[idx]), 4),
        )
        for idx in equity_curve.index
    ]

    trades_out: list[TradeOut] = []
    trades_df = result.trades
    if trades_df is not None and not trades_df.empty:
        for i, row in enumerate(trades_df.itertuples(index=False), start=1):
            direction = "long" if int(getattr(row, "direction", 1)) >= 0 else "short"
            entry_price = float(getattr(row, "entry_price"))
            exit_price = float(getattr(row, "exit_price"))
            pnl_pct = float(getattr(row, "pnl_pct"))
            pnl_abs = float(getattr(row, "pnl_abs"))
            commission_abs = float(getattr(row, "commission_abs"))
            pnl_net_abs = float(getattr(row, "pnl_net_abs"))
            trades_out.append(
                TradeOut(
                    id=i,
                    entryDate=pd.Timestamp(getattr(row, "entry_date")).strftime("%Y-%m-%d"),
                    exitDate=pd.Timestamp(getattr(row, "exit_date")).strftime("%Y-%m-%d"),
                    direction=direction,
                    entryPrice=_trade_price(entry_price),
                    exitPrice=_trade_price(exit_price),
                    pnlPct=round(pnl_pct, 4),
                    pnlAbs=round(pnl_abs, 2),
                    commissionAbs=round(commission_abs, 2),
                    pnlNetAbs=round(pnl_net_abs, 2),
                )
            )

    m = result.metrics
    metrics = MetricsOut(
        totalReturn=float(m.get("total_return", 0.0)),
        cagr=float(m.get("cagr", 0.0)),
        sharpe=_defined(m.get("sharpe")),
        sortino=_defined(m.get("sortino")),
        calmar=_defined(m.get("calmar")),
        maxDrawdown=float(m.get("max_drawdown", 0.0)),
        volatility=float(m.get("volatility", 0.0)),
        winRate=_defined(m.get("win_rate")),
        profitFactor=_defined(m.get("profit_factor")),
        nTrades=int(m.get("n_trades", len(trades_out))),
    )

    return BacktestResponse(
        dataSource="supplied" if req.ohlcv else "synthetic",
        symbol=req.symbol,
        strategy=req.strategy,
        strategyExecuted=executed_id,
        fill=req.fill,
        commission=float(req.commission),
        slippage=float(req.slippage),
        note=note,
        ruined=result.ruined,
        ruinDate=None if result.ruin_date is None else result.ruin_date.strftime("%Y-%m-%d"),
        equity=equity_points,
        trades=trades_out,
        metrics=metrics,
    )


@app.post("/var", response_model=VarResponse)
def var(req: VarRequest) -> VarResponse:
    """Compute a REAL VaR/CVaR report from the var_report engine function."""
    if req.ohlcv:
        df = _bars_to_df(req.ohlcv)
        # /var's dates have defaults: cut supplied bars to them only when the
        # caller actually sent a window.
        if {"startDate", "endDate"} & req.model_fields_set:
            df = _window(df, req.startDate, req.endDate)
    else:
        df = synthetic_ohlcv(req.symbol, req.startDate, req.endDate)

    returns = df["Close"].astype(float).pct_change().dropna()
    if len(returns) < MIN_VAR_RETURNS:
        raise HTTPException(
            422,
            f"Not enough data for VaR: need at least {MIN_VAR_RETURNS} daily returns "
            f"({MIN_VAR_RETURNS + 1} bars), got {len(returns)}.",
        )

    report = var_report(
        returns,
        portfolio_value=float(req.portfolioValue),
        confidence_levels=req.confidenceLevels,
        periods_per_year=_periods_per_year(req.periodsPerYear, returns.index),
    )

    rows = [
        VarRow(
            confidence=str(idx),
            paramVar1d=round(float(row["param_var_1d"]), 2),
            histVar1d=round(float(row["hist_var_1d"]), 2),
            paramVar10d=round(float(row["param_var_10d"]), 2),
            histVar10d=round(float(row["hist_var_10d"]), 2),
            paramCvar1d=round(float(row["param_cvar_1d"]), 2),
        )
        for idx, row in report.iterrows()
    ]

    return VarResponse(
        dataSource="supplied" if req.ohlcv else "synthetic",
        symbol=req.symbol,
        portfolioValue=float(req.portfolioValue),
        rows=rows,
    )


@app.post("/optimize", response_model=OptimizeResponse)
def optimize(req: OptimizeRequest) -> OptimizeResponse:
    """Run a REAL Markowitz optimization over synthetic prices.

    Each symbol gets its own independently seeded GBM series, so the assets
    are mutually UNCORRELATED (sample correlations are noise around zero) and
    none of it is market data -- the response says ``dataSource: "synthetic"``.
    It demonstrates the optimizer, not an allocation for these tickers.
    """
    symbols = req.symbols  # validated: stripped, upper-cased, distinct, >= 2

    price_data: dict[str, pd.Series] = {}
    for sym in symbols:
        df = synthetic_ohlcv(sym, req.startDate, req.endDate)
        price_data[sym] = df["Close"].astype(float)

    prices = pd.DataFrame(price_data).dropna()
    returns = prices.pct_change().dropna()
    need = min_observations(len(symbols))
    if len(returns) < need:
        raise HTTPException(
            422,
            f"Not enough data to optimize {len(symbols)} assets: need at least {need} daily "
            f"returns, got {len(returns)}.",
        )

    opt = MarkowitzOptimizer(returns)
    try:
        result = opt.optimize(req.objective)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    return OptimizeResponse(
        dataSource="synthetic",
        objective=result.objective,
        weights={k: round(float(v), 4) for k, v in result.weights.items()},
        expectedReturn=round(float(result.expected_return), 4),
        volatility=None if _defined(result.volatility) is None else round(float(result.volatility), 4),
        sharpe=None if _defined(result.sharpe) is None else round(float(result.sharpe), 4),
        warning=concentration_warning(result.weights),
    )


def main() -> None:
    """Console entry point: ``python -m alphaforge.api.app`` or ``alphaforge-api``."""
    import uvicorn

    uvicorn.run("alphaforge.api.app:app", host="0.0.0.0", port=8003, reload=False)


if __name__ == "__main__":
    main()
