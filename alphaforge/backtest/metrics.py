"""
Portfolio performance metrics.

All metrics are computed from an equity curve (pd.Series) and/or
a returns series (pd.Series of per-bar fractional returns).

Annualisation: bars per year, read from the bars
------------------------------------------------
Every annualised figure (CAGR, volatility, Sharpe, Sortino, Calmar) needs the
number of bars in a year. :func:`infer_periods_per_year` reads it from the median
spacing of the dates: daily bars are ``TRADING_DAYS = 252``, the
exchange-trading-day convention, then weekly 52, monthly 12, quarterly 4, yearly
1. Any other spacing, intraday included, is refused unless the caller passes
``periods_per_year``; an index without dates is taken as daily, with a warning.
Counting every bar as a 252nd of a year would misreport the same prices held
buy-and-hold: CAGR 17.67% / Sharpe 0.85 on daily bars but 125.03% / 2.12 on weekly
ones.

Daily bars count bars, not calendar time. The demo API's synthetic feed is
business-daily (about 261 bars a year, holidays included), so its annualised
figures are on the 252-bar convention, not its calendar: four "years" of that
feed is ~4.14 years by this count. Calendar-daily data such as crypto is 252 too
unless the caller passes ``periods_per_year=365``.

Risk-free rate: a fixed 2% a year
---------------------------------
Sharpe, Sortino and ``rolling_sharpe`` subtract ``RISK_FREE_RATE /
periods_per_year`` per bar, whatever rates were over the window; it is a
constant, not a parameter. The engine's cash earns 0%, so this is a hurdle that
cash does not clear.

Undefined ratios are NaN, never 0
---------------------------------
A ratio whose denominator is zero has no value: Sharpe with zero volatility,
Sortino with zero volatility or no return below the hurdle (a riskless series
has no downside RISK, whatever its distance from the hurdle), Calmar with no
drawdown, profit factor with no losing trade, and the win statistics of a run
with no trades. These are reported as ``float("nan")`` (the REST API sends
``null``, the CLI prints ``n/a``), because 0.0 is indistinguishable from "no edge
at all".
A run of fewer than two bars has no metrics at all, and prints ``n/a`` likewise.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


TRADING_DAYS = 252
RISK_FREE_RATE = 0.02  # annual
UNDEFINED = float("nan")

# (fewest, most) median days between bars -> bars per year: the spacings the data
# layer offers ("1d", "5d", "1wk", "1mo", "3mo"), with room for month lengths and
# holidays.
_BAR_SPACINGS = (
    (0.9, 1.5, TRADING_DAYS),
    (5.0, 9.0, 52),
    (27.0, 33.0, 12),
    (85.0, 98.0, 4),
    (350.0, 380.0, 1),
)
# So few daily bars can straddle a weekend or a holiday: Friday and Monday alone
# are three days apart.
_FEW_INTERVALS = 5
_LONGEST_DAILY_GAP = 4.5


def infer_periods_per_year(index: pd.Index) -> float:
    """Bars per year, from the median spacing of *index* (see the module docstring).

    Raises ``ValueError`` on a spacing it cannot name, intraday included: an
    annualised figure on a guessed calendar would look like a result.
    """
    if len(index) < 2:
        return TRADING_DAYS  # no interval, so nothing gets annualised
    if not isinstance(index, pd.DatetimeIndex):
        warnings.warn(
            f"alphaforge: the data's index is a {type(index).__name__}, not dates, so its "
            f"annualised figures assume daily bars ({TRADING_DAYS} a year). Pass "
            f"periods_per_year, or index it by date.",
            stacklevel=2,
        )
        return TRADING_DAYS
    stamps = np.sort(index.dropna().to_numpy())
    if len(stamps) < 2:
        return TRADING_DAYS
    gaps = np.diff(stamps) / np.timedelta64(1, "D")
    days = float(np.median(gaps))
    for fewest, most, periods in _BAR_SPACINGS:
        if fewest <= days <= most:
            return periods
    if 1.0 < days <= _LONGEST_DAILY_GAP and len(gaps) < _FEW_INTERVALS:
        return TRADING_DAYS
    raise ValueError(
        f"alphaforge: bars a median {days:.4g} days apart are not daily, weekly, monthly, "
        f"quarterly or yearly, so the number of bars in a year cannot be inferred and "
        f"annualised figures would be wrong. Pass periods_per_year (for hourly US-equity "
        f"bars, {TRADING_DAYS} x 6.5 = {TRADING_DAYS * 6.5:g})."
    )


def compute_metrics(
    equity_curve: pd.Series,
    returns: pd.Series,
    trades: pd.DataFrame | None = None,
    periods_per_year: float | None = None,
) -> dict[str, float]:
    """
    Compute a comprehensive set of performance metrics.

    Parameters
    ----------
    equity_curve : pd.Series
        Portfolio value over time.
    returns : pd.Series
        Per-bar net returns (fractional).
    trades : pd.DataFrame | None
        Trade log from BacktestEngine. Used for the win statistics
        (``win_rate``, ``profit_factor``, ``avg_win``, ``avg_loss``,
        ``expectancy``), which are computed on each trade's NET dollar P&L,
        ``pnl_net_abs`` -- after commission and slippage, sized by the
        capital actually at work. That is the column whose sum IS the equity
        curve's P&L (the reconciliation invariant), so ``profit_factor > 1`` exactly
        when the run made money. ``avg_win`` / ``avg_loss`` / ``expectancy``
        are therefore in dollars. A trade log without ``pnl_net_abs`` (one
        built outside the engine) falls back to ``pnl_pct``, in its units.
    periods_per_year : float | None
        Bars per year; None infers it from ``equity_curve``'s dates (see the
        module docstring). Reported back as ``metrics["periods_per_year"]``.

    Returns
    -------
    dict mapping metric name → float value.
    """
    metrics: dict[str, float] = {}

    if equity_curve.empty or len(equity_curve) < 2:
        return metrics

    periods = infer_periods_per_year(equity_curve.index) if periods_per_year is None else periods_per_year
    metrics["periods_per_year"] = float(periods)

    # --- CAGR ---
    # Years ELAPSED: N points span N - 1 bar-to-bar intervals. Counting points
    # instead would read exactly +10% over 252 intervals (253 points) as 9.96%.
    n_years = (len(equity_curve) - 1) / periods
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    metrics["total_return"] = float(total_return)
    metrics["cagr"] = float((1 + total_return) ** (1 / max(n_years, 1e-9)) - 1)

    # --- Volatility ---
    bar_vol = float(returns.std())
    metrics["volatility"] = float(bar_vol * np.sqrt(periods))

    # --- Sharpe Ratio ---
    rfr_per_bar = RISK_FREE_RATE / periods
    excess = returns - rfr_per_bar
    metrics["sharpe"] = float(
        excess.mean() / excess.std() * np.sqrt(periods)
        if excess.std() > 1e-10 else UNDEFINED
    )

    # --- Sortino Ratio ---
    # Downside deviation over the FULL sample, against the same hurdle as the
    # numerator: sqrt(mean(min(r - rf_per_bar, 0)^2)), every bar counted (bars
    # above the hurdle contribute zero). Not the sample std of only the below-hurdle
    # returns, demeaned around their own mean and divided by their own count, which
    # misstates downside risk.
    shortfall = (returns - rfr_per_bar).clip(upper=0.0)
    downside_dev = float(np.sqrt((shortfall ** 2).mean()))
    metrics["sortino"] = float(
        excess.mean() / downside_dev * np.sqrt(periods)
        if downside_dev > 1e-10 and excess.std() > 1e-10 else UNDEFINED
    )

    # --- Max Drawdown ---
    rolling_max = equity_curve.cummax()
    drawdown_series = (equity_curve - rolling_max) / rolling_max
    metrics["max_drawdown"] = float(drawdown_series.min())

    # --- Max Drawdown Duration ---
    # Longest run of consecutive BARS below the running peak. Bars, not days: they
    # differ on any calendar with weekends.
    in_dd = drawdown_series < 0
    dd_duration = _max_consecutive_true(in_dd)
    metrics["max_drawdown_duration_bars"] = float(dd_duration)

    # --- Calmar Ratio ---
    abs_mdd = abs(metrics["max_drawdown"])
    metrics["calmar"] = float(metrics["cagr"] / abs_mdd if abs_mdd > 1e-10 else UNDEFINED)

    # --- Win Rate / Profit Factor ---
    # On NET dollar P&L, so profit factor is below 1 exactly when the book lost
    # money. An unweighted, gross pnl_pct would give equal weight to a trade held at
    # 1% of the book and one at 100%, and count a +0.05% trade that lost money after
    # costs as a win.
    pnl_col = next((c for c in ("pnl_net_abs", "pnl_pct") if trades is not None and c in trades.columns), None)
    if trades is not None and not trades.empty and pnl_col is not None:
        pnl = trades[pnl_col].astype(float)
        wins = pnl > 0
        losses = pnl < 0
        gross_win = float(pnl[wins].sum())
        gross_loss = float(-pnl[losses].sum())
        metrics["win_rate"] = float(wins.sum() / len(pnl))
        metrics["avg_win"] = float(pnl[wins].mean()) if wins.any() else UNDEFINED
        metrics["avg_loss"] = float(-pnl[losses].mean()) if losses.any() else UNDEFINED
        metrics["profit_factor"] = gross_win / gross_loss if gross_loss > 0 else UNDEFINED
        metrics["n_trades"] = float(len(pnl))
        metrics["expectancy"] = float(pnl.mean())

    # --- Exposure ---
    # BARS on the curve, whatever their spacing, not days: a weekly run's n_days
    # is its number of weeks. The key is public in BacktestResult.metrics, so it
    # keeps its name.
    metrics["n_days"] = float(len(equity_curve))

    return metrics


def compute_drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """Return the drawdown series (negative values, 0 = at peak)."""
    rolling_max = equity_curve.cummax()
    return (equity_curve - rolling_max) / rolling_max


def annualized_return(returns: pd.Series, periods_per_year: float | None = None) -> float:
    """Compound annual growth rate from a series of PER-INTERVAL returns.

    Each element is one bar-to-bar return (``prices.pct_change().dropna()``),
    so ``len(returns)`` is the number of elapsed intervals. A
    ``BacktestResult.returns`` series is one element LONGER than that: its
    first element is bar 0's structural zero (no position, no interval before
    the first bar). Pass ``result.returns.iloc[1:]`` -- or read
    ``result.metrics["cagr"]`` -- to get the same figure as
    :func:`compute_metrics`; the whole series counts one interval too many.
    ``periods_per_year`` is as in :func:`compute_metrics`, read from these dates.
    """
    periods = infer_periods_per_year(returns.index) if periods_per_year is None else periods_per_year
    total = (1 + returns).prod()
    n_years = len(returns) / periods
    return float(total ** (1 / max(n_years, 1e-9)) - 1)


def rolling_sharpe(returns: pd.Series, window: int = 63, periods_per_year: float | None = None) -> pd.Series:
    """Rolling Sharpe ratio over *window* bars.

    NaN where it is undefined -- the first ``window - 1`` bars (warm-up) and
    any window with zero volatility -- never 0, which would read as "no edge".
    ``periods_per_year`` is as in
    :func:`compute_metrics`, read from these dates.
    """
    periods = infer_periods_per_year(returns.index) if periods_per_year is None else periods_per_year
    excess = returns - RISK_FREE_RATE / periods
    roll = excess.rolling(window)
    std = roll.std()
    return roll.mean() / std.where(std > 1e-10) * np.sqrt(periods)


def _max_consecutive_true(series: pd.Series) -> int:
    """Return the length of the longest consecutive True run."""
    max_len = 0
    current = 0
    for val in series:
        if val:
            current += 1
            max_len = max(max_len, current)
        else:
            current = 0
    return max_len
