# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **`alphaforge backtest` plots the whole equity curve.** The ASCII curve kept every
  n-th bar and stopped at 60 columns, so it could end before the run did: on the
  README's 1,006-bar AAPL command it stopped 61 bars early, at $94,322 against a
  final $88,388. It now samples evenly spaced bars from the first to the last.

## [0.1.0] — first public release

2026-11-04

### What ships

- **Backtesting engine.** Single- and multi-asset, on Python 3.11+. Trade log, ruin
  handling, per-side costs, and a reconciliation of the log to the equity curve.
- **Signals.** SMA crossover, RSI and Bollinger reversion, MACD regime, weighted
  ensembles, an XGBoost classifier and an optional PyTorch LSTM.
- **Risk and portfolio tools.** Kelly, Fixed Fractional, Equal Weight and Volatility
  Parity sizers, a `DrawdownMonitor`, parametric, historical and Monte Carlo VaR/CVaR,
  and a Markowitz optimizer (max Sharpe, min volatility, risk parity, max diversification).
- **Interfaces.** The `alphaforge` CLI, a FastAPI service, a Next.js dashboard, and
  `e2e/smoke.mjs`, which drives the real API and dashboard together.
- **Alpaca paper-order client.** `PaperTrader` submits, lists and closes orders by hand.

### Behaviour to know

- **A signal is a target position in `[-1, 1]`**, held as shares fixed at entry, never
  rebalanced until the target changes.
- **Fills are closes.** A signal from bar *t* is filled at bar *t*'s close by default,
  which is optimistic for a signal that needs that close. `fill="next_close"` fills a
  bar later. No signal earns the bar it was computed on.
- **Costs are per side on traded notional:** 10 bps commission and 5 bps slippage by
  default, so a long-to-short reversal pays on both legs.
- **ML signals never score a bar they trained on.** Unfitted, `generate()` runs a purged
  expanding walk-forward; after `fit(train)` it scores only later bars.
- **Annualisation follows the spacing of the bars** (daily 252, weekly 52, and so on).
  Any other spacing, intraday included, is refused unless you pass the bars per year.
- **Undefined ratios are NaN** (`null` in the API, `n/a` in the CLI), never 0.
- **Equity floors at $0.** A ruined account stops trading.
- **Bad input is refused, not guessed at:** negative costs, bars out of order, duplicate
  dates, non-positive prices, a `price_col` that is not a close, and too little data.

### The demo API

- **Its prices are seeded synthetic data**, not market data for the ticker named. Each
  ticker has one path that replays from the window's start, so dates select a window
  length, not a market period. Supply your own `ohlcv` bars to use real prices.
- **`ml_momentum` runs an SMA 10/30 proxy** in the API and CLI, and says so in the
  response. The real XGBoost strategy is `strategies.ml_momentum`.
- **With the API unreachable** the dashboard shows only a dated recording of the
  engine's default runs, and no numbers for anything else.

### Known limits

- Shorts pay no borrow cost, cash earns 0%, and negative cash is financed free.
- There are no margin calls and no market impact.
- Sharpe and Sortino subtract a fixed 2% a year, whatever rates were over the window.
- Sizers and `DrawdownMonitor` are standalone: no backtest, strategy, CLI command or API
  endpoint calls them.
- Optimizer weights are in-sample. They describe the window and forecast nothing.
- Built-in stock universes are today's names, so they are survivorship-biased.
- Exact XGBoost figures depend on the xgboost release.
- `PaperTrader` has only run against a fake client, never the real Alpaca SDK.
- This is not financial advice, and it is not intended for trading real capital.
