# AlphaForge Usage Guide

Concrete, copy-paste examples for the Python API, the command-line interface,
and the web demo. All snippets are grounded in the actual package API. For a
component-level overview, see [architecture.md](architecture.md).

> **Note on data.** `DataFetcher` downloads OHLCV from yfinance / Alpha Vantage,
> so the examples below require network access. Returned DataFrames use a
> `DatetimeIndex` with `Open, High, Low, Close, Volume` columns. Daily and
> weekly prices are split- and dividend-adjusted on all four price columns
> from either source (yfinance `auto_adjust=True`; Alpha Vantage's adjusted
> close applied as a ratio to open/high/low; its `Volume` stays as traded).
> The Alpha Vantage fallback reads `ALPHA_VANTAGE_KEY` and never writes the key
> to logs.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Command-Line Interface](#command-line-interface)
- [Signals](#signals)
- [Backtesting](#backtesting)
- [Risk Management](#risk-management)
- [Portfolio Optimization](#portfolio-optimization)
- [Machine-Learning Signals](#machine-learning-signals)
- [Alpaca Paper-Order Client](#alpaca-paper-order-client)
- [Web Demo](#web-demo)

---

## Installation

```bash
git clone https://github.com/ROROK20/AlphaForge.git
cd AlphaForge
pip install -e .

pip install -e ".[torch]"   # optional: PyTorch LSTM signal
pip install -e ".[live]"    # optional: Alpaca paper-order client
pip install -e ".[dev]"     # optional: tests + jupyter
```

On macOS, run `brew install libomp` first: XGBoost's macOS wheels load the
OpenMP runtime from Homebrew and do not ship it.

---

## Quick Start

```python
from alphaforge import DataFetcher, BacktestEngine
from alphaforge.signals.technical import SMACrossoverSignal

df = DataFetcher().fetch_ohlcv("AAPL", "2020-01-01", "2023-12-31")
signal = SMACrossoverSignal(fast=20, slow=50)
result = BacktestEngine(initial_capital=100_000).run(signal, df)

print(result)
# Output of a real run, re-derived 2026-11-03 on yfinance data fetched
# 2026-07-29 (read back from the local cache, ~/.alphaforge/cache):
# BacktestResult(cagr=-3.05%, sharpe=-0.02, max_drawdown=-56.06%, trades=22)
# (Long/short 20/50 crossover, default costs. It lost money on AAPL over this
#  window — the example shows the API, not a recommended strategy. yfinance
#  back-adjusts history on every dividend, so re-runs drift slightly.)

print(result.metrics["sharpe"], result.metrics["max_drawdown"])
print(result.trades.head())       # per-trade log
print(result.equity_curve.tail()) # portfolio value over time
```

`BacktestEngine.run()` returns a `BacktestResult` dataclass with:
`equity_curve`, `returns`, `positions`, `trades`, `metrics`, and `signal_series`.

---

## Command-Line Interface

Installing the package exposes the `alphaforge` console script (three commands).

```bash
# Backtest a built-in strategy
alphaforge backtest --strategy sma_crossover --symbol AAPL \
  --start 2020-01-01 --end 2023-12-31 --capital 100000 --commission 0.001

# Strategies: sma_crossover (SMA 20/50) | mean_reversion (Bollinger 20, 2σ)
#   | rsi (RSI 14, 30/70) | macd (12/26/9 regime)
#   | ml_momentum (SMA 10/30 proxy, disclosed — not ML-driven)
alphaforge backtest -s macd -t TSLA --start 2021-01-01 --long-only

# Markowitz portfolio optimization
alphaforge optimize --symbols AAPL,MSFT,GOOGL,AMZN --objective max_sharpe
# Objectives: max_sharpe | min_volatility | risk_parity | max_diversification | equal_weight
# Only dates on which every ticker has a price are used: the command prints that
# window and its number of returns, and names a ticker that cut it short.

# Value-at-Risk report
alphaforge var --symbol SPY --capital 1000000
```

Every command renders rich tables in your terminal; `backtest` also draws an
ASCII equity curve.

---

## Signals

Every signal subclasses `alphaforge.signals.base.Signal` and implements
`generate(df) -> pd.Series` aligned to the input index. The built-in signals
return integers in `{-1, 0, +1}` (short / flat / long); the engine also accepts
fractional target positions in `[-1, 1]`.

```python
from alphaforge.signals.technical import (
    SMACrossoverSignal,
    RSIMeanReversionSignal,
    MACDSignal,
    BollingerBandSignal,
)

df = DataFetcher().fetch_ohlcv("MSFT", "2020-01-01", "2023-12-31")

sma = SMACrossoverSignal(fast=20, slow=50)
positions = sma.generate(df)   # pd.Series of {-1, 0, 1}
```

### Ensembles

Combine multiple signals with weights:

```python
from alphaforge.signals.ensemble import EnsembleSignal

ensemble = EnsembleSignal(
    signals=[RSIMeanReversionSignal(), MACDSignal()],
    weights=[0.5, 0.5],
)
result = BacktestEngine().run(ensemble, df)
```

### Custom signal

```python
import numpy as np
import pandas as pd
from alphaforge.signals.base import Signal


class MomentumBreakoutSignal(Signal):
    """Long from a close above the prior N-day high until a close below the
    prior N-day low (a Donchian-channel breakout that HOLDS the position)."""

    name = "Momentum Breakout"

    def __init__(self, window: int = 20) -> None:
        self.window = window

    def generate(self, df: pd.DataFrame) -> pd.Series:
        self._validate_df(df)
        prior = df["Close"].shift(1)
        signals = pd.Series(np.nan, index=df.index)          # NaN = no change
        signals[df["Close"] > prior.rolling(self.window).max()] = 1
        signals[df["Close"] < prior.rolling(self.window).min()] = 0
        return self._fill_signals(signals, df.index)         # hold between events
```

---

## Backtesting

```python
from alphaforge.backtest.engine import BacktestEngine

engine = BacktestEngine(
    initial_capital=100_000,
    commission=0.001,   # per side, fraction of traded notional (10 bps)
    slippage=0.0005,    # per side, fraction of price (5 bps)
    long_only=True,     # ignore -1 (short) signals
)
result = engine.run(signal, df)

print(f"Sharpe:       {result.metrics['sharpe']:.2f}")
print(f"CAGR:         {result.metrics['cagr']:.2%}")
print(f"Max Drawdown: {result.metrics['max_drawdown']:.2%}")
```

**Execution model.** The engine shifts every signal one bar: a signal
computed on bar *t* (it may use bar *t*'s close) is filled at bar *t*'s close
and earns `close(t) → close(t+1)` from bar *t + 1* on, so no signal ever earns
the bar it was computed on. That is the usual market-on-close assumption of
vectorized backtests — optimistic for a signal that needs the close to be
computed. `BacktestEngine(fill="next_close")` fills one bar later instead (the
README's "Fill timing" table shows how much that moves real results). Only `Close` is
read; `Open`/`High`/`Low` never affect fills. Costs are charged per side on
traded notional: entries and exits pay `commission + slippage` once, a −1 ↔ +1
reversal pays them twice.

**Positions are shares.** When the target changes the engine trades to
`target × equity / fill price` shares and holds them until it changes again —
equity is cash plus shares × price, and a trade's P&L is its shares × the
price change, less costs (no hidden daily rebalancing).

**Ruin.** Equity never goes below zero. The first bar on which it would
reach zero or less (a short through a 100% rise over any number of bars, a
long through a collapse) ruins the account:
equity is floored at $0, the position is liquidated at that bar's close, and
nothing trades afterwards. `result.ruined` / `result.ruin_date` say so (the
API returns `ruined` / `ruinDate`, the CLI and the web demo print it), metrics
cover the path through the ruin bar, and the loss beyond the account is
written back on that trade (`ruin_writeoff_abs`) so the trade log still sums
to the equity curve.

**What is not modelled.** Shorts pay no borrow cost, cash and short-sale
proceeds earn 0%, Sharpe and Sortino subtract a fixed 2% a year, and there are
no margin calls and no market impact. See the README's "What the model leaves out".

**Trade-log dates are fill dates.** `entry_date` and `exit_date` name the bar
whose close is the logged price: `entry_price = close(entry_date) x (1 +/-
slippage)`, and the same for the exit, under either `fill`. The bars between
the two dates are the bars the trade held. A position still open on the last
bar, or ruined, has no exit order: its `exit_date` is that bar and its
`exit_price` is that bar's bare close, no slippage, matching the curve, which
charged no exit cost. A fill's cost shows on the equity curve one bar after
the fill's date, the first bar the new position is held (so a closing fill's
cost can ruin an account on the bar after its `exit_date`).

### Multi-asset

```python
from alphaforge.signals.technical import SMACrossoverSignal

symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
fetcher = DataFetcher()

data_map = {s: fetcher.fetch_ohlcv(s, "2020-01-01", "2023-12-31") for s in symbols}
# run_multi takes a dict of symbol -> signal *series* (call .generate())
signals_map = {s: SMACrossoverSignal(fast=20, slow=50).generate(data_map[s]) for s in symbols}

result = BacktestEngine().run_multi(
    signals_map, data_map,
    weights={"AAPL": 0.3, "MSFT": 0.3, "GOOGL": 0.2, "AMZN": 0.2},
)
```

---

## Risk Management

### Position sizing

Sizers are standalone: the engine has no sizer argument, and nothing in the engine,
the strategies, the CLI or the API calls one. To backtest a sizer's output, multiply a
signal's series by it and pass that, since the engine takes fractional target positions.

```python
from alphaforge.risk.position_sizing import (
    KellyCriterion, FixedFractional, VolatilityParity,
)

# Quarter-Kelly from historical win statistics
# Every sizer returns a FRACTION of capital to deploy, not a dollar amount.
kelly = KellyCriterion(win_rate=0.58, avg_win=0.045, avg_loss=0.030, fraction=0.25)
print(f"{kelly.size(100_000):.1%}")   # 7.5%  (full Kelly 30%, quartered)

# Risk 2% of capital per trade given a 5% stop
ff = FixedFractional(risk_per_trade=0.02)
print(f"{ff.size(100_000, stop_distance=0.05):.1%}")   # 40.0%  (0.02 / 0.05)

# Target 15% annualized volatility
vp = VolatilityParity(target_volatility=0.15, lookback=20)
print(vp.size(100_000, returns=df["Close"].pct_change()))
```

### Drawdown monitor

`DrawdownMonitor` reports a breach to the code that calls it; no backtest consults it.

```python
from alphaforge.risk.drawdown import DrawdownMonitor

monitor = DrawdownMonitor(max_drawdown_pct=-0.15, daily_loss_limit=-0.03)

# Call each trading day with the latest portfolio value and daily return
if not monitor.update(portfolio_value, daily_return):
    print(f"Trading halted: {monitor.halt_reason}")
```

### Value-at-Risk

```python
from alphaforge.risk.var import var_report, parametric_var, historical_var, monte_carlo_var

returns = df["Close"].pct_change().dropna()
report = var_report(returns, portfolio_value=1_000_000)
print(report)   # by confidence: parametric + historical VaR (1- and 10-day), parametric CVaR (1-day)
```

---

## Portfolio Optimization

```python
import pandas as pd
from alphaforge.optimization.markowitz import MarkowitzOptimizer

prices = pd.DataFrame({sym: d["Close"] for sym, d in data_map.items()})
returns = prices.pct_change().dropna()

opt = MarkowitzOptimizer(returns, risk_free_rate=0.02, allow_short=False)

best = opt.optimize("max_sharpe")
print(best.weights)            # pd.Series of weights
print(best.expected_return, best.volatility, best.sharpe)

frontier = opt.efficient_frontier(n_points=100)
print(frontier[["expected_return", "volatility", "sharpe"]])
```

The weights are in-sample: fitted to `returns`, and the figures are measured on
the same `returns`. They describe that window and forecast nothing.

`max_sharpe` raises `ValueError` when no portfolio within the weight limits has
an expected return above `risk_free_rate`. Every Sharpe ratio is then negative,
and maximising it would pick the most volatile portfolio; use `min_volatility`
for the least risky one. With `allow_short=True` a long/short book that beats
the rate counts, even when every asset on its own does not.

---

## Machine-Learning Signals

### XGBoost direction classifier

`generate()` never scores a bar the model trained on. Without `fit`, it runs a
purged expanding walk-forward (`min_train_size=252` labelled rows before the
first scored bar, retrained every `retrain_every=63` bars, the
`target_horizon` label rows purged at each boundary;
`signal.walk_forward_folds_` lists the folds). After `fit(train)` it scores
only bars after `train`'s last bar.

```python
from alphaforge.signals.ml_signals import XGBoostSignal
from alphaforge.data.preprocessor import add_technical_features, create_lagged_features

df = DataFetcher().fetch_ohlcv("AAPL", "2018-01-01", "2023-12-31")
df = add_technical_features(df)
df = create_lagged_features(df, n_lags=5)
df = df.dropna()

split = int(len(df) * 0.7)
train, test = df.iloc[:split], df.iloc[split:]

signal = XGBoostSignal(target_horizon=5)
signal.fit(train)
preds = signal.generate(df).loc[test.index]   # history warms up test features

result = BacktestEngine().run(preds, test)
```

### LSTM predictor (requires `pip install -e ".[torch]"`)

```python
from alphaforge.signals.ml_signals import LSTMSignal

signal = LSTMSignal(hidden_size=64, num_layers=2, seq_len=20, epochs=50, random_state=0)
signal.fit(train)
preds = signal.generate(df).loc[test.index]
```

### Walk-forward strategy

```python
from strategies.ml_momentum import MLMomentumStrategy

strategy = MLMomentumStrategy(target_horizon=5, train_ratio=0.6)
result = strategy.run_walk_forward(
    "AAPL", "2018-01-01", "2023-12-31",
    window_size=252,  # 1 year training window
    step_size=63,     # retrain quarterly
)
print(f"OOS Sharpe: {result.metrics['sharpe']:.2f}")
```

---

## Alpaca Paper-Order Client

`PaperTrader` is a small client for Alpaca's paper API: it submits, lists and
closes orders that you place by hand. It is not a trading loop, and nothing
connects a signal to an order. The tests drive it against a fake client; it has
never run against the real SDK ([SECURITY.md](../SECURITY.md)).

Set credentials in a local `.env` (see [`.env.example`](../.env.example)) — never
commit real keys. `PaperTrader` reads it from the current directory or its nearest
parent that has one; a real environment variable or a constructor argument wins.

```bash
pip install -e ".[live]"
cp .env.example .env   # then fill in ALPACA_API_KEY / ALPACA_SECRET_KEY
```

```python
from alphaforge.live.paper_trader import PaperTrader

trader = PaperTrader()                       # ALPACA_*: environment, then .env
trader.submit_order("AAPL", qty=10, side="buy")
print(trader.get_account())                  # equity, cash, buying power, status
print(trader.get_positions())                # open positions
```

The endpoint is the `base_url` argument if you pass one, else
`ALPACA_BASE_URL` (environment, then `.env`), else the paper-trading endpoint
(`https://paper-api.alpaca.markets`). `PaperTrader` raises `ValueError` for any
other endpoint unless you pass `allow_live=True` — a live URL places
real-money orders. AlphaForge is intended for research and
paper trading, not live capital.

---

## Web Demo

An interactive Next.js dashboard lets you explore backtests in the browser.
Every result is computed by the engine API on its synthetic price feed (no API
keys or market data needed), so start both:

```bash
make demo-local      # API on :8003 + dashboard on :3001
```

Without the API there is no in-browser engine: the default configuration shows
a dated, labelled **recorded snapshot** of the engine's own output
(`scripts/record_engine_snapshot.py`), and any other configuration shows no
results.

Select a symbol, strategy, date range, and capital to see equity curves,
drawdown charts, a metrics grid, and a trade log update live.
