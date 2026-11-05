# AlphaForge Architecture

## Overview

AlphaForge is a modular, vectorized backtesting framework. Each layer has a single responsibility and communicates through well-defined interfaces.

```
┌─────────────────────────────────────────────────────────────┐
│                        CLI / API                            │
│              alphaforge backtest / optimize / var           │
└──────┬──────────────────┬──────────────────┬────────────────┘
       │ backtest         │ var              │ optimize
┌──────▼──────────┐ ┌─────▼────────┐ ┌───────▼─────────────┐
│ BacktestEngine  │ │     var      │ │    Optimization     │
│ Vectorized P&L  │ │ Parametric   │ │  Markowitz          │
│ (shares + cash) │ │ Historical   │ │  max_sharpe         │
│ Trade log,      │ │ Monte Carlo  │ │  min_vol            │
│ result packaging│ └──────────────┘ │  risk_parity        │
└──────▲──────────┘                  └─────────────────────┘
       │ target positions
┌──────┴──────────┐   Standalone, called by your own code:
│  Signals        │   position_sizing, drawdown
│  (base.py)      │
│  technical      │
│  ml_signals     │
│  ensemble       │
└──────▲──────────┘
       │ OHLCV
┌──────┴────────────────────────────────────────────────────┐
│                      Data Layer                           │
│  DataFetcher (yfinance + Alpha Vantage)                   │
│  Preprocessor (technical indicators, normalization, lags) │
│  Universe (stock selection, sector filtering)             │
│  (the API uses seeded synthetic or caller-supplied bars)  │
└───────────────────────────────────────────────────────────┘
```

## Design Principles

### 1. Vectorization First
A position is a number of shares, fixed when it is entered and held until the signal's target changes. The engine steps from one target change to the next; within each stretch, equity and per-bar P&L are numpy array arithmetic over its bars (`cash + shares × price`), not a Python loop over bars. Four places do loop in Python, on purpose: the simulation iterates once per *position change*, the trade log once per *trade* (to attribute each cost leg exactly), the RSI signal's regime state machine once per bar, and the max-drawdown-duration metric scans the drawdown series once. Measured costs are under [Performance](#performance).

### 2. Signal Contract
Every built-in signal returns a `pd.Series` of `{-1, 0, +1}` integers aligned to the data's DatetimeIndex; the engine also accepts fractional target positions in `[-1, 1]`. This uniform interface allows signals to be mixed, ensembled, and swapped without changing the engine.

### 3. No Look-Ahead Bias
The engine applies a 1-bar shift to all signals before computing returns:
```python
signals = raw_signals.shift(1).fillna(0)  # position on bar t = signal from bar t-1
```
A signal computed on bar *t* is filled at bar *t*'s close (market-on-close,
`fill="close"`, the default) and earns from bar *t + 1*. That is optimistic for
a signal that needs bar *t*'s close; `fill="next_close"` fills a bar later
(the README's "Fill timing" table quantifies the difference). The engine never
reads `Open`, so there is no next-open fill. See `BacktestEngine`'s
"Execution model" docstring.

ML signals never score a bar their model trained on: an unfitted
`generate()` runs a purged expanding walk-forward (retrain each block on all
earlier history, label horizon purged at every boundary, warm-up flat), and a
fitted one scores only bars after its training window. This holds wherever
the signal is used, including inside an `EnsembleSignal`.

### 4. Composability
A strategy in `strategies/` pairs a signal with a `BacktestEngine`. You can mix technical and ML signals in an `EnsembleSignal` and run the result through the same engine.

Position sizers and `DrawdownMonitor` are separate. The engine has no sizer argument, and nothing in the engine, the strategies, the CLI or the API calls a sizer or the monitor. To backtest a sizer's output, multiply a signal's series by `sizer.size(capital)` yourself and pass that: the engine accepts fractional target positions in `[-1, 1]`. `DrawdownMonitor` reports a breach to the code that calls it; no backtest consults it.

## Module Reference

### `alphaforge.data`
| Module | Purpose |
|--------|---------|
| `fetcher.py` | Download OHLCV from yfinance/Alpha Vantage with disk caching |
| `preprocessor.py` | Compute 15+ technical indicators, normalization, lag features |
| `universe.py` | Stock universe management, sector filtering (built-in lists are 2026 names: survivorship-biased for historical backtests) |

### `alphaforge.signals`
| Module | Purpose |
|--------|---------|
| `base.py` | Abstract `Signal` class, validation helpers |
| `technical.py` | SMA crossover, RSI mean reversion, MACD, Bollinger Bands |
| `ml_signals.py` | XGBoost classifier, 2-layer LSTM predictor |
| `ensemble.py` | Weighted / voting combination of multiple signals |

### `alphaforge.backtest`
| Module | Purpose |
|--------|---------|
| `engine.py` | Vectorized single/multi-asset backtesting, trade log |
| `metrics.py` | Sharpe, Sortino, Calmar, CAGR, max drawdown, win rate |
| `portfolio.py` | Cash / position / NAV bookkeeping for event-driven simulations (long and short; not used by the vectorized engine) |

### `alphaforge.risk`
| Module | Purpose |
|--------|---------|
| `position_sizing.py` | Kelly Criterion, Fixed Fractional, Equal Weight, Volatility Parity. Standalone: nothing here calls them |
| `drawdown.py` | `DrawdownMonitor` (reports a limit breach to its caller; no backtest consults it), underwater period analysis |
| `var.py` | Parametric VaR, Historical Simulation VaR, Monte Carlo VaR/CVaR |

### `alphaforge.optimization`
| Module | Purpose |
|--------|---------|
| `markowitz.py` | Mean-variance optimizer: max Sharpe, min vol, risk parity, max diversification |

### `alphaforge.live`
| Module | Purpose |
|--------|---------|
| `paper_trader.py` | Alpaca paper-order client: submits, lists and closes orders by hand. No signal drives it, and it has only run against a fake client |

### `alphaforge.cli`
The `alphaforge` console script (Rich + Click) exposes three commands —
`backtest`, `optimize`, and `var` — that wire the modules above into formatted
terminal reports with ASCII equity curves.

### `frontend/` (web demo)
A **Next.js 15 + TypeScript + Tailwind** single-page app (`frontend/app/`,
port 3001) that renders equity/drawdown charts and a trade log via Recharts.
It computes nothing itself: every result is a `POST /backtest` response from
the FastAPI layer over the engine, or — when that API is unreachable — the
recorded snapshot of exactly such a response for the default configuration
(`frontend/public/engine-snapshot/`, written by
`scripts/record_engine_snapshot.py`, kept current by
`tests/test_engine_snapshot.py`). Any other configuration offline shows no
results.

## Data Flow

```
Symbol + Date Range
       │
       ▼
DataFetcher.fetch_ohlcv()
       │ DataFrame(OHLCV, DatetimeIndex)
       ▼
add_technical_features() / create_lagged_features()
       │ Enriched DataFrame
       ▼
Signal.generate()
       │ pd.Series({-1, 0, +1})
       ▼
BacktestEngine.run()
       │ signals.shift(1)  ← look-ahead prevention (fill at signal bar's close)
       │ on each target change: shares = target × equity / fill price
       │ costs = |Δshares| × fill price × (commission + slippage)
       │ equity = cash + shares × price   (floored at 0: ruin stops trading)
       ▼
compute_metrics()
       │ dict[str, float]
       ▼
BacktestResult
```

## Performance

Measured 2026-11-03 with `OMP_NUM_THREADS=1 python scripts/benchmark.py`
(median of 5 runs after a warm-up; synthetic data; macOS arm64, Python 3.12.5,
a lightly loaded shared machine). Timings move with hardware and load — re-run
the script rather than quoting these as guarantees.

| Operation | Time |
|---|---|
| BacktestEngine.run, precomputed SMA signal, 1,304 bars | 2.0 ms |
| BacktestEngine.run, precomputed SMA signal, 5,217 bars | 4.1 ms |
| SMACrossoverSignal.generate + run, 1,304 bars | 2.1 ms |
| RSIMeanReversionSignal.generate + run, 1,304 bars | 11.1 ms |
| MACDSignal.generate + run, 1,304 bars | 3.7 ms |
| BollingerBandSignal.generate + run, 1,304 bars | 3.0 ms |
| Monte Carlo VaR, 100,000 paths, 10-day horizon | 6.2 ms |
| Markowitz max_sharpe, 50 assets x 1,260 days | 53.4 ms |
| XGBoostSignal.fit, 1,000 bars (default 300 trees) | 446.9 ms |

1,304 bars is 5 years of business days, 5,217 is 20; the RSI row includes its
per-bar regime loop.

## See Also

- [usage.md](usage.md) — concrete Python API and CLI examples
- [../README.md](../README.md) — project overview and quick start
- [../CONTRIBUTING.md](../CONTRIBUTING.md) — development setup
