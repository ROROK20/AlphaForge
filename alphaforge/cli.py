"""
AlphaForge CLI — Rich terminal interface for backtesting and optimization.

Usage:
    alphaforge backtest --strategy sma_crossover --symbol AAPL --start 2020-01-01 --end 2023-12-31
    alphaforge optimize --symbols AAPL,MSFT,GOOGL --start 2020-01-01
    alphaforge var --symbol SPY --start 2022-01-01
"""

from __future__ import annotations

import sys
from datetime import datetime

import click
import numpy as np
import pandas as pd
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

console = Console()

BANNER = r"""
[bold cyan]
    _   _      _            _____
   / \ | |_ __| |__   __ _|  ___|__  _ __ __ _  ___
  / _ \| | '_ \| '_ \ / _` | |_ / _ \| '__/ _` |/ _ \
 / ___ \ | |_) | | | | (_| |  _| (_) | | | (_| |  __/
/_/   \_\_| .__/|_| |_|\__,_|_|  \___/|_|  \__, |\___|
          |_|                               |___/
[/bold cyan]
[dim]Python backtesting framework that keeps itself honest[/dim]
"""


def _date_option(ctx, param, value: str | None) -> str | None:
    """--start / --end: a calendar date, or a usage error instead of a traceback."""
    if value is not None:
        try:
            parsed = pd.Timestamp(value)
        except ValueError as exc:
            raise click.BadParameter(f"{value!r} is not a date (YYYY-MM-DD).") from exc
        if pd.isna(parsed):
            raise click.BadParameter(f"{value!r} is not a date (YYYY-MM-DD).")
    return value


@click.group()
def cli():
    """AlphaForge — Quantitative trading framework."""
    console.print(BANNER)


@cli.command()
@click.option("--strategy", "-s", default="sma_crossover",
              type=click.Choice(["sma_crossover", "mean_reversion", "ml_momentum", "macd", "rsi"]),
              help="Strategy to backtest: sma_crossover = SMA 20/50 regime; "
                   "mean_reversion = Bollinger Band (20, 2 sigma) reversion; "
                   "rsi = RSI(14) 30/70 reversion; macd = MACD 12/26/9 regime; "
                   "ml_momentum = SMA 10/30 proxy (disclosed; not ML-driven).")
@click.option("--symbol", "-t", default="AAPL", help="Ticker symbol.")
@click.option("--start", default="2020-01-01", callback=_date_option, help="Start date (YYYY-MM-DD).")
@click.option("--end", default=None, callback=_date_option, help="End date (YYYY-MM-DD). Defaults to today.")
@click.option("--capital", default=100_000, type=float, help="Initial capital.")
@click.option("--commission", default=0.001, type=float, help="Commission per side, fraction of traded notional (default 0.001 = 10 bps).")
@click.option("--slippage", default=0.0005, type=float, help="Slippage per side, fraction of price (default 0.0005 = 5 bps).")
@click.option("--long-only", is_flag=True, default=False, help="Long-only mode.")
@click.option("--fill", default="close", type=click.Choice(["close", "next_close"]),
              help="close = fill at the close of the signal's own bar (market-on-close, "
                   "default); next_close = fill one bar later.")
def backtest(strategy: str, symbol: str, start: str, end: str | None,
             capital: float, commission: float, slippage: float, long_only: bool, fill: str):
    """Run a backtest and print results."""
    from alphaforge.data.fetcher import DataFetcher
    from alphaforge.backtest.engine import BacktestEngine

    end = end or datetime.today().strftime("%Y-%m-%d")

    with console.status(f"[bold green]Fetching {symbol} data ({start} → {end})…"):
        fetcher = DataFetcher()
        df = fetcher.fetch_ohlcv(symbol, start, end)

    if df.empty:
        console.print(f"[red]No data returned for {symbol}. Check ticker and date range.[/red]")
        sys.exit(1)

    signal = _build_signal(strategy)

    with console.status("[bold green]Running backtest…"):
        try:
            engine = BacktestEngine(
                initial_capital=capital,
                commission=commission,
                slippage=slippage,
                long_only=long_only,
                fill=fill,
            )
            result = engine.run(signal, df)
        except ValueError as exc:
            _refuse(exc)

    _print_backtest_results(result, symbol, strategy, start, end, capital, fill,
                            commission, slippage)


@cli.command()
@click.option("--symbols", "-s", required=True, help="Comma-separated tickers (e.g. AAPL,MSFT,GOOGL).")
@click.option("--start", default="2020-01-01", callback=_date_option, help="Start date.")
@click.option("--end", default=None, callback=_date_option, help="End date.")
@click.option("--objective", "-o", default="max_sharpe",
              type=click.Choice(["max_sharpe", "min_volatility", "risk_parity",
                                  "max_diversification", "equal_weight"]),
              help="Optimization objective.")
def optimize(symbols: str, start: str, end: str | None, objective: str):
    """Run Markowitz portfolio optimization."""
    from alphaforge.data.fetcher import DataFetcher
    from alphaforge.optimization.markowitz import (
        IN_SAMPLE_NOTE, MarkowitzOptimizer, concentration_warning,
    )

    end = end or datetime.today().strftime("%Y-%m-%d")
    symbol_list = [s.strip().upper() for s in symbols.split(",")]

    with console.status(f"[bold green]Fetching data for {symbol_list}…"):
        fetcher = DataFetcher()
        price_data = {}
        for sym in symbol_list:
            df = fetcher.fetch_ohlcv(sym, start, end)
            if not df.empty:
                price_data[sym] = df["Close"]

    # Refused, not dropped: a portfolio of the others is not what was asked for.
    missing = [sym for sym in symbol_list if sym not in price_data]
    if missing:
        _refuse(f"No data returned for {', '.join(missing)}. Check the tickers and date range.")

    if len(price_data) < 2:
        console.print("[red]Need at least 2 symbols with valid data.[/red]")
        sys.exit(1)

    # Only the dates every symbol has a price on are used, so a young ticker cuts
    # every other asset to its history. Say which window that left.
    prices = pd.DataFrame(price_data).dropna()
    returns = prices.pct_change().dropna()
    _print_window(price_data, prices, len(returns))

    with console.status(f"[bold green]Optimizing ({objective})…"):
        try:
            result = MarkowitzOptimizer(returns).optimize(objective)
        except ValueError as exc:
            _refuse(exc)

    _print_optimization_results(result, symbol_list)
    console.print(f"[dim]{IN_SAMPLE_NOTE}[/dim]")
    warning = concentration_warning(result.weights)
    if warning:
        console.print(f"[yellow]{escape(warning)}[/yellow]")


@cli.command()
@click.option("--symbol", "-s", default="SPY", help="Ticker symbol.")
@click.option("--start", default="2022-01-01", callback=_date_option, help="Start date.")
@click.option("--end", default=None, callback=_date_option, help="End date.")
@click.option("--capital", default=100_000, type=click.FloatRange(min=0, min_open=True), help="Portfolio value.")
def var(symbol: str, start: str, end: str | None, capital: float):
    """Compute Value-at-Risk report."""
    from alphaforge.data.fetcher import DataFetcher
    from alphaforge.risk.var import MIN_VAR_RETURNS, var_report

    end = end or datetime.today().strftime("%Y-%m-%d")

    with console.status(f"[bold green]Fetching {symbol}…"):
        fetcher = DataFetcher()
        df = fetcher.fetch_ohlcv(symbol, start, end)
    if df.empty:
        console.print(f"[red]No data returned for {escape(symbol)}. Check ticker and date range.[/red]")
        sys.exit(1)
    returns = df["Close"].pct_change().dropna()
    if len(returns) < MIN_VAR_RETURNS:
        _refuse(f"Not enough data for VaR: need at least {MIN_VAR_RETURNS} daily returns, got {len(returns)}.")

    report = var_report(returns, portfolio_value=capital)

    console.print(Panel(f"[bold]VaR Report — {symbol}[/bold]", style="cyan"))

    table = Table(box=box.ROUNDED, show_header=True)
    table.add_column("Confidence", style="bold yellow")
    for col in report.columns:
        table.add_column(col.replace("_", " ").title())

    for idx, row in report.iterrows():
        table.add_row(
            str(idx),
            *[f"${v:,.0f}" for v in row.values],
        )

    console.print(table)


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------

ML_MOMENTUM_PROXY_NOTE = (
    "ml_momentum: the CLI runs an SMA(10/30) crossover proxy, as the web demo "
    "does, so these results are NOT ML-driven. For the XGBoost model (purged "
    "walk-forward, scored only on out-of-sample bars) use XGBoostSignal or "
    "strategies.ml_momentum.MLMomentumStrategy from Python."
)


def _build_signal(strategy: str):
    """One distinct signal per ``--strategy`` choice; ``ml_momentum``'s proxy is disclosed."""
    from alphaforge.signals.technical import (
        BollingerBandSignal, MACDSignal, RSIMeanReversionSignal, SMACrossoverSignal,
    )
    if strategy == "ml_momentum":
        console.print(f"[yellow]{ML_MOMENTUM_PROXY_NOTE}[/yellow]")
        return SMACrossoverSignal(fast=10, slow=30)
    builders = {
        "sma_crossover": lambda: SMACrossoverSignal(fast=20, slow=50),
        "mean_reversion": lambda: BollingerBandSignal(period=20, n_std=2.0),
        "rsi": lambda: RSIMeanReversionSignal(period=14),
        "macd": lambda: MACDSignal(),
    }
    return builders[strategy]()


def _print_window(price_data: dict[str, pd.Series], shared: pd.DataFrame, n_returns: int) -> None:
    """Print the dates ``optimize`` used: those on which every symbol has a price.

    Names the symbol whose history starts latest or ends earliest when it cut the
    others short, and counts dates inside the window dropped for a missing price.
    """
    if shared.empty:
        console.print("[yellow]The symbols have no date with a price for all of them.[/yellow]")
        return
    first, last = shared.index[0], shared.index[-1]
    notes = [f"Window: {first:%Y-%m-%d} → {last:%Y-%m-%d}, {n_returns} daily returns, "
             "on the dates every symbol has a price for."]
    starts = {sym: s.dropna().index[0] for sym, s in price_data.items()}
    ends = {sym: s.dropna().index[-1] for sym, s in price_data.items()}
    latest_start, earliest_end = max(starts.values()), min(ends.values())
    if min(starts.values()) < latest_start:
        late = ", ".join(f"{escape(sym)}'s prices start {latest_start:%Y-%m-%d}"
                         for sym, start in starts.items() if start == latest_start)
        notes.append(f"{late}: earlier dates of the other symbols are not used.")
    if max(ends.values()) > earliest_end:
        early = ", ".join(f"{escape(sym)}'s prices end {earliest_end:%Y-%m-%d}"
                          for sym, end in ends.items() if end == earliest_end)
        notes.append(f"{early}: later dates of the other symbols are not used.")
    gaps = len(pd.DataFrame(price_data).loc[first:last]) - len(shared)
    if gaps:
        notes.append(f"{gaps} dates inside the window are left out because a symbol "
                     "has no price on them.")
    console.print(f"[dim]{' '.join(notes)}[/dim]")


def _refuse(reason: object) -> None:
    """Print why an input was refused and exit 1: an error, not a traceback."""
    console.print(f"[red]{escape(str(reason))}[/red]")
    sys.exit(1)


FILL_LABEL = {"close": "signal-bar close (market-on-close)", "next_close": "next bar's close"}


def _print_backtest_results(result, symbol, strategy, start, end, capital, fill="close",
                            commission=0.001, slippage=0.0005):
    m = result.metrics

    console.print()
    console.print(Panel(
        f"[bold]Backtest Results — {symbol} | {strategy.replace('_', ' ').title()}[/bold]\n"
        f"[dim]{start} → {end} | Capital: ${capital:,.0f}[/dim]\n"
        f"[dim]Costs per side: {commission:.2%} commission + {slippage:.2%} slippage"
        f" | Fill: {FILL_LABEL[fill]}[/dim]",
        style="cyan"
    ))

    # Metrics table
    table = Table(box=box.ROUNDED, show_header=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    # Undefined ratios (zero denominator) are NaN in the metrics dict; they
    # print as "n/a", never as a misleading 0.00.
    def undefined(v): return v is None or v != v
    def pct(v): return "n/a" if undefined(v) else f"{v:.2%}"
    def f2(v): return "n/a" if undefined(v) else f"{v:.2f}"

    rows = [
        ("Total Return", pct(m.get("total_return"))),
        ("CAGR", pct(m.get("cagr"))),
        ("Sharpe Ratio", f2(m.get("sharpe"))),
        ("Sortino Ratio", f2(m.get("sortino"))),
        ("Calmar Ratio", f2(m.get("calmar"))),
        ("Max Drawdown", pct(m.get("max_drawdown"))),
        ("Max DD Duration", "n/a" if undefined(m.get("max_drawdown_duration_bars"))
         else f"{m['max_drawdown_duration_bars']:.0f} bars"),
        ("Annualized Vol", pct(m.get("volatility"))),
        ("Win Rate", pct(m.get("win_rate"))),
        ("Profit Factor", f2(m.get("profit_factor"))),
        ("# Trades", f"{m.get('n_trades', 0):.0f}"),
    ]

    for metric, value in rows:
        # Color code returns and drawdowns
        style = ""
        # Green/red against each metric's break-even: 0 for returns and
        # Sharpe, 1 for profit factor (gross win / gross loss). Win rate has
        # no break-even on its own, so it is not coloured. ("n/a" stays plain.)
        breakeven = {"Total Return": 0.0, "CAGR": 0.0, "Sharpe Ratio": 0.0, "Profit Factor": 1.0}
        if metric in breakeven:
            try:
                num = float(value.replace("%", "").replace("$", "").replace(",", ""))
                style = "green" if num > breakeven[metric] else "red"
            except ValueError:
                pass
        if "Drawdown" in metric and "Duration" not in metric:
            style = "red"
        table.add_row(metric, Text(value, style=style))

    console.print(table)
    if result.ruined:
        console.print(
            f"[bold red]RUINED on {result.ruin_date:%Y-%m-%d}: equity reached $0, the "
            f"position was liquidated and nothing traded afterwards. Metrics cover the "
            f"path through that bar.[/bold red]"
        )

    # ASCII equity curve
    if not result.equity_curve.empty:
        _print_ascii_curve(result.equity_curve, capital)


def _print_ascii_curve(equity: pd.Series, baseline: float, width: int = 60, height: int = 12):
    """Print a simple ASCII equity curve."""
    console.print()
    console.print("[bold cyan]── Equity Curve ──────────────────────────────[/bold cyan]")

    values = equity.values
    mn, mx = values.min(), values.max()
    if mx == mn:
        return

    def scale(v):
        return int((v - mn) / (mx - mn) * (height - 1))

    # A series wider than the plot is sampled at evenly spaced bars that always
    # include the first and the last, so the curve ends where the run ended.
    sampled = values
    if len(values) > width:
        sampled = values[np.arange(width) * (len(values) - 1) // (width - 1)]

    canvas = [[" "] * len(sampled) for _ in range(height)]
    for col, val in enumerate(sampled):
        row = height - 1 - scale(val)
        canvas[row][col] = "█" if val >= baseline else "▄"

    console.print(f"  ${mx:>10,.0f} ┐")
    for row in canvas:
        console.print("             │" + "".join(row))
    console.print(f"  ${mn:>10,.0f} └" + "─" * len(sampled))
    console.print()


def _print_optimization_results(result, symbols):
    console.print()
    console.print(Panel(
        f"[bold]Portfolio Optimization — {result.objective.replace('_', ' ').title()}[/bold]",
        style="cyan"
    ))

    table = Table(box=box.ROUNDED)
    table.add_column("Asset", style="bold")
    table.add_column("Weight", justify="right")
    table.add_column("Allocation", justify="right")

    for sym, w in result.weights.sort_values(ascending=False).items():
        bar = "█" * int(w * 30)
        table.add_row(sym, f"{w:.2%}", f"[green]{bar}[/green]")

    console.print(table)

    stats_table = Table(box=box.SIMPLE, show_header=False)
    stats_table.add_column("Metric", style="dim")
    stats_table.add_column("Value")
    stats_table.add_row("Expected Return", f"{result.expected_return:.2%}")
    stats_table.add_row("Volatility", f"{result.volatility:.2%}")
    stats_table.add_row("Sharpe Ratio", f"{result.sharpe:.2f}")
    console.print(stats_table)


def main():
    cli()


if __name__ == "__main__":
    main()
