"""Tests for the ``alphaforge`` CLI.

Data access is mocked (``DataFetcher`` is replaced), so nothing touches the
network or the on-disk cache.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from alphaforge.cli import _build_signal, backtest, optimize, var

STRATEGIES = backtest.params[0].type.choices  # the --strategy click.Choice


def make_ohlcv(n: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, n)))
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": 1e6,
        },
        index=pd.date_range("2021-01-01", periods=n, freq="B"),
    )


def _identity(signal) -> tuple:
    return type(signal).__name__, tuple(sorted(vars(signal).items()))


def test_strategy_option_is_the_first_param():
    assert backtest.params[0].name == "strategy"
    assert set(STRATEGIES) == {"sma_crossover", "mean_reversion", "ml_momentum", "macd", "rsi"}


def test_every_strategy_choice_runs_a_distinct_signal():
    """Five ``--strategy`` names, five different signals: none may quietly run
    another's."""
    built = {name: _identity(_build_signal(name)) for name in STRATEGIES}
    assert len(set(built.values())) == len(STRATEGIES), built


@pytest.mark.parametrize(
    "name,cls,params",
    [
        ("sma_crossover", "SMACrossoverSignal", {"fast": 20, "slow": 50}),
        ("mean_reversion", "BollingerBandSignal", {"period": 20, "n_std": 2.0}),
        ("rsi", "RSIMeanReversionSignal", {"period": 14}),
        ("macd", "MACDSignal", {"fast": 12, "slow": 26, "signal_period": 9}),
        ("ml_momentum", "SMACrossoverSignal", {"fast": 10, "slow": 30}),
    ],
)
def test_strategy_runs_what_the_help_text_says(name, cls, params):
    sig = _build_signal(name)
    assert type(sig).__name__ == cls
    for key, value in params.items():
        assert getattr(sig, key) == value
    assert name.split("_")[0] in backtest.params[0].help


def _run_backtest(strategy: str):
    with patch("alphaforge.data.fetcher.DataFetcher") as fetcher_cls:
        fetcher_cls.return_value.fetch_ohlcv.return_value = make_ohlcv()
        return CliRunner().invoke(
            backtest, ["--strategy", strategy, "--start", "2021-01-01", "--end", "2022-07-01"]
        )


@pytest.mark.parametrize("strategy", ["sma_crossover", "mean_reversion", "rsi", "macd"])
def test_backtest_command_runs_end_to_end(strategy):
    result = _run_backtest(strategy)
    assert result.exit_code == 0, result.output
    assert "Backtest Results" in result.output
    assert "NOT ML-driven" not in result.output


def test_ml_momentum_discloses_its_proxy_truthfully():
    """The substitution must be disclosed, and the stated reason must be true:
    the XGBoost signal does not "require pre-training" (it runs its own
    out-of-sample walk-forward), so the CLI must not claim it does."""
    result = _run_backtest("ml_momentum")
    assert result.exit_code == 0, result.output
    text = " ".join(result.output.split())  # undo Rich's line wrapping
    assert "SMA(10/30)" in text
    assert "NOT ML-driven" in text
    assert "pre-training" not in text



def test_undefined_ratios_print_as_na_not_zero():
    """Flat prices: no volatility and no trades. The table must say n/a for
    the undefined ratios, not 0.00 / 0.00%."""
    flat = make_ohlcv()
    flat[["Open", "High", "Low", "Close"]] = 50.0
    with patch("alphaforge.data.fetcher.DataFetcher") as fetcher_cls:
        fetcher_cls.return_value.fetch_ohlcv.return_value = flat
        result = CliRunner().invoke(backtest, ["--strategy", "sma_crossover"])
    assert result.exit_code == 0, result.output
    rows = {line.split("│")[1].strip(): line.split("│")[2].strip()
            for line in result.output.splitlines() if line.count("│") >= 3}
    # Sortino is undefined on a riskless book too.
    for metric in ("Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Win Rate", "Profit Factor"):
        assert rows.get(metric) == "n/a", (metric, rows.get(metric))


def test_fill_option_is_applied_and_printed():
    with patch("alphaforge.data.fetcher.DataFetcher") as fetcher_cls:
        fetcher_cls.return_value.fetch_ohlcv.return_value = make_ohlcv()
        moc = CliRunner().invoke(backtest, ["--strategy", "macd"])
        late = CliRunner().invoke(backtest, ["--strategy", "macd", "--fill", "next_close"])
    assert moc.exit_code == 0 and late.exit_code == 0, late.output
    def flat(out):  # undo Rich's wrapping, including the panel's border glyphs
        return " ".join(out.replace("│", " ").split())

    assert "Fill: signal-bar close (market-on-close)" in flat(moc.output)
    assert "Fill: next bar's close" in flat(late.output)
    assert moc.output.split("Total Return")[1][:30] != late.output.split("Total Return")[1][:30]


def test_both_cost_legs_are_printed():
    with patch("alphaforge.data.fetcher.DataFetcher") as fetcher_cls:
        fetcher_cls.return_value.fetch_ohlcv.return_value = make_ohlcv()
        out = CliRunner().invoke(backtest, ["--strategy", "macd", "--slippage", "0.002"]).output
    flat = " ".join(out.replace("│", " ").split())
    assert "Costs per side: 0.10% commission + 0.20% slippage" in flat


def test_drawdown_duration_is_labelled_in_bars():
    with patch("alphaforge.data.fetcher.DataFetcher") as fetcher_cls:
        fetcher_cls.return_value.fetch_ohlcv.return_value = make_ohlcv()
        out = CliRunner().invoke(backtest, ["--strategy", "macd"]).output
    row = next(line for line in out.splitlines() if "Max DD Duration" in line)
    assert row.split("│")[2].strip().endswith(" bars"), row


def _invoke(command, args, frame=None):
    with patch("alphaforge.data.fetcher.DataFetcher") as fetcher_cls:
        fetcher_cls.return_value.fetch_ohlcv.return_value = make_ohlcv() if frame is None else frame
        return CliRunner().invoke(command, args)


@pytest.mark.parametrize("args,word", [
    (["--commission", "-0.05"], "commission"),
    (["--slippage", "-0.01"], "slippage"),
    (["--capital", "0"], "initial_capital"),
])
def test_backtest_refuses_nonsense_costs_and_capital(args, word):
    """--commission -0.05 on the README's AAPL run would print Total Return +604.84%,
    Sharpe 1.52, exit 0."""
    result = _invoke(backtest, args)
    assert result.exit_code == 1 and word in result.output
    assert "Total Return" not in result.output and "Traceback" not in result.output


def test_backtest_refuses_a_newest_first_frame():
    result = _invoke(backtest, [], make_ohlcv().iloc[::-1])
    assert result.exit_code == 1 and "time order" in result.output
    assert "Total Return" not in result.output


def test_optimize_refuses_too_few_days():
    """Three days of data must not print a Sharpe ratio and exit 0."""
    result = _invoke(optimize, ["--symbols", "AAPL,MSFT"], make_ohlcv(4))
    assert result.exit_code == 1 and "need at least 60" in result.output
    assert "Sharpe" not in result.output


def test_optimize_still_runs_on_enough_days():
    with patch("alphaforge.data.fetcher.DataFetcher") as fetcher_cls:
        fetcher_cls.return_value.fetch_ohlcv.side_effect = lambda sym, s, e: make_ohlcv(300, seed=len(sym))
        result = CliRunner().invoke(optimize, ["--symbols", "AAPL,GOOGL"])
    assert result.exit_code == 0 and "Sharpe" in result.output


def test_var_on_a_ticker_with_no_data_says_so():
    """A message, not an IndexError traceback."""
    empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    result = _invoke(var, ["--symbol", "NOSUCHTICKER"], empty)
    assert result.exit_code == 1 and "No data returned for NOSUCHTICKER" in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_var_refuses_too_few_days_and_a_non_positive_portfolio():
    short = _invoke(var, [], make_ohlcv(10))
    assert short.exit_code == 1 and "need at least 60" in short.output
    assert _invoke(var, ["--capital", "-5"]).exit_code != 0


def test_var_still_reports_on_enough_days():
    result = _invoke(var, [])
    assert result.exit_code == 0 and "95%" in result.output


def test_the_results_table_says_na_for_a_run_with_no_figures(capsys):
    """One bar has no metrics: the table must not show them as zeros."""
    from alphaforge import cli
    from alphaforge.backtest.engine import BacktestEngine

    df = make_ohlcv(1)
    result = BacktestEngine().run(pd.Series(1.0, index=df.index), df)
    with cli.console.capture() as captured:
        cli._print_backtest_results(result, "AAPL", "sma_crossover", "2021-01-01", "2021-01-01", 100_000)
    table = captured.get()
    assert "0.00%" not in table
    for metric in ("Total Return", "CAGR", "Max Drawdown", "Max DD Duration", "Annualized Vol"):
        row = next(line for line in table.splitlines() if metric in line)
        assert "n/a" in row, row


def _ascii_curve(equity: pd.Series) -> tuple[list[str], int]:
    """The printed equity curve's canvas rows (top first) and its column count."""
    from alphaforge import cli

    with cli.console.capture() as captured:
        cli._print_ascii_curve(equity, baseline=100_000)
    lines = captured.get().splitlines()
    rows = [line.split("│", 1)[1] for line in lines if line.startswith("             │")]
    axis = next(line for line in lines if "└" in line)
    return rows, axis.count("─")


@pytest.mark.parametrize("n", [30, 61, 120, 1006, 2348])
def test_the_ascii_equity_curve_runs_from_the_first_bar_to_the_last(n):
    """The README's backtest command has 1,006 bars, and the plot stopped 61 bars
    short of the end, at an equity the run did not finish on. Here the first bar is
    the series' low and the last its high: the first column must sit on the bottom
    row and the last column on the top row, whatever the length."""
    equity = pd.Series(np.linspace(100_000, 90_000, n),
                       index=pd.date_range("2020-01-01", periods=n, freq="B"))
    equity.iloc[0] = 80_000
    equity.iloc[-1] = 120_000
    rows, columns = _ascii_curve(equity)
    assert columns == min(n, 60)
    assert len(rows[0]) == columns and rows[0][-1] != " ", "the final bar is not plotted"
    assert rows[-1][0] != " ", "the first bar is not plotted"


def test_optimize_refuses_a_ticker_with_no_data_rather_than_dropping_it():
    """Optimizing the others and exiting 0 is not what was asked for."""
    empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    def fetch(sym, start, end):
        return empty if sym == "NOSUCH" else make_ohlcv(300, seed=len(sym))

    with patch("alphaforge.data.fetcher.DataFetcher") as fetcher_cls:
        fetcher_cls.return_value.fetch_ohlcv.side_effect = fetch
        result = CliRunner().invoke(optimize, ["--symbols", "AAPL,GOOGL,NOSUCH"])
    assert result.exit_code == 1, result.output
    assert "No data returned for NOSUCH" in " ".join(result.output.split())
    assert "Sharpe" not in result.output


def test_optimize_prints_that_its_result_is_in_sample_and_warns_on_a_corner():
    returns_up = make_ohlcv(300, seed=1)
    returns_up["Close"] = 100 * np.exp(np.cumsum(np.random.default_rng(3).normal(0.004, 0.01, 300)))
    down = make_ohlcv(300, seed=2)
    down["Close"] = 100 * np.exp(np.cumsum(np.random.default_rng(4).normal(-0.004, 0.01, 300)))
    with patch("alphaforge.data.fetcher.DataFetcher") as fetcher_cls:
        fetcher_cls.return_value.fetch_ohlcv.side_effect = lambda sym, s, e: returns_up if sym == "UP" else down
        result = CliRunner().invoke(optimize, ["--symbols", "UP,DOWN"])
    assert result.exit_code == 0, result.output
    text = " ".join(result.output.split())
    assert "In-sample" in text and "forecast nothing" in text
    assert "UP takes 100.0%" in text


def test_optimize_says_which_window_it_used_when_a_ticker_is_younger():
    """Aligning the tickers keeps only the dates every one of them has, so a young
    ticker cuts every other asset to its own history. The command must print the
    window and the number of returns it actually used, and name the ticker that cut
    it, instead of reporting statistics for a window nobody asked for."""
    old = make_ohlcv(300, seed=1)
    young = make_ohlcv(300, seed=2).iloc[200:]
    with patch("alphaforge.data.fetcher.DataFetcher") as fetcher_cls:
        fetcher_cls.return_value.fetch_ohlcv.side_effect = lambda sym, s, e: old if sym == "OLD" else young
        result = CliRunner().invoke(optimize, ["--symbols", "OLD,YOUNG", "--objective", "min_volatility"])
    assert result.exit_code == 0, result.output
    text = " ".join(result.output.split())
    first, last = young.index[0], young.index[-1]
    assert f"{first:%Y-%m-%d} → {last:%Y-%m-%d}" in text
    assert "99 daily returns" in text
    assert f"YOUNG's prices start {first:%Y-%m-%d}" in text


def test_optimize_says_when_a_ticker_ends_early_or_misses_dates():
    full = make_ohlcv(300, seed=1)
    short = make_ohlcv(300, seed=2).iloc[:250].drop(make_ohlcv(300).index[[100, 101, 102]])
    with patch("alphaforge.data.fetcher.DataFetcher") as fetcher_cls:
        fetcher_cls.return_value.fetch_ohlcv.side_effect = lambda sym, s, e: full if sym == "FULL" else short
        result = CliRunner().invoke(optimize, ["--symbols", "FULL,SHORT", "--objective", "min_volatility"])
    assert result.exit_code == 0, result.output
    text = " ".join(result.output.split())
    assert f"SHORT's prices end {short.index[-1]:%Y-%m-%d}" in text
    assert "3 dates inside the window are left out" in text
    assert "246 daily returns" in text


def test_optimize_names_no_ticker_when_none_cut_the_window():
    with patch("alphaforge.data.fetcher.DataFetcher") as fetcher_cls:
        fetcher_cls.return_value.fetch_ohlcv.side_effect = lambda sym, s, e: make_ohlcv(300, seed=len(sym))
        result = CliRunner().invoke(optimize, ["--symbols", "AAPL,GOOGL", "--objective", "min_volatility"])
    assert result.exit_code == 0, result.output
    text = " ".join(result.output.split())
    assert "299 daily returns" in text
    assert "prices start" not in text and "prices end" not in text


def test_optimize_refuses_max_sharpe_when_nothing_beats_the_risk_free_rate():
    """Two losing assets: the default objective has no meaningful answer, so the
    command says why and exits 1 instead of printing a Sharpe ratio."""
    def losing(seed):
        frame = make_ohlcv(300, seed=seed)
        frame["Close"] = 100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(-0.002, 0.01, 300)))
        return frame

    with patch("alphaforge.data.fetcher.DataFetcher") as fetcher_cls:
        fetcher_cls.return_value.fetch_ohlcv.side_effect = lambda sym, s, e: losing(len(sym))
        result = CliRunner().invoke(optimize, ["--symbols", "AB,CDE"])
        calm = CliRunner().invoke(optimize, ["--symbols", "AB,CDE", "--objective", "min_volatility"])
    assert result.exit_code == 1, result.output
    assert "risk-free rate" in " ".join(result.output.split())
    assert "Sharpe Ratio" not in result.output
    assert calm.exit_code == 0 and "Sharpe Ratio" in calm.output


@pytest.mark.parametrize("command,args", [
    (backtest, ["--end", "garbage"]),
    (optimize, ["--symbols", "AAPL,MSFT", "--end", "garbage"]),
    (var, ["--end", "garbage"]),
])
def test_a_garbage_end_date_is_a_usage_error_not_a_traceback(command, args):
    """The real DataFetcher, unmocked: it reached pd.Timestamp("garbage") and the
    ValueError came out as a traceback. Refused before anything is fetched."""
    result = CliRunner().invoke(command, args)
    assert result.exit_code == 2, result.output
    assert isinstance(result.exception, SystemExit)
    assert "'garbage' is not a date" in result.output


@pytest.mark.parametrize("command,args", [
    (backtest, ["--start", "2020-13-45"]),
    (optimize, ["--symbols", "AAPL,MSFT", "--start", "soon"]),
    (var, ["--start", "not-a-date"]),
    (backtest, ["--end", "NaT"]),
])
def test_any_unparseable_date_option_is_refused(command, args):
    with patch("alphaforge.data.fetcher.DataFetcher") as fetcher_cls:
        fetcher_cls.return_value.fetch_ohlcv.return_value = make_ohlcv()
        result = CliRunner().invoke(command, args)
    assert result.exit_code == 2, result.output
    fetcher_cls.return_value.fetch_ohlcv.assert_not_called()

