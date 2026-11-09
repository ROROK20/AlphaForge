"""
Value-at-Risk (VaR) and Conditional Value-at-Risk (CVaR) implementations.

Three methods: parametric (normal), historical simulation, Monte Carlo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from alphaforge.backtest.metrics import TRADING_DAYS, infer_periods_per_year


def parametric_var(
    returns: pd.Series,
    confidence: float = 0.95,
    horizon: int = 1,
    portfolio_value: float = 1.0,
) -> float:
    """
    Parametric (normal distribution) VaR.

    VaR = μ - z * σ  (at given confidence, over given horizon)

    Parameters
    ----------
    returns : pd.Series
        Per-bar portfolio returns.
    confidence : float
        Confidence level (e.g. 0.95 for 95% VaR).
    horizon : float
        Horizon in bars of *returns* (scales volatility by sqrt(horizon)).
    portfolio_value : float
        Portfolio NAV to express VaR in dollar terms (default 1.0 = fraction).

    Returns
    -------
    float
        VaR as a positive number (i.e. loss magnitude).
    """
    mu = float(returns.mean()) * horizon
    sigma = float(returns.std()) * np.sqrt(horizon)
    z = stats.norm.ppf(1 - confidence)
    var = -(mu + z * sigma)
    return float(var * portfolio_value)


def historical_var(
    returns: pd.Series,
    confidence: float = 0.95,
    horizon: int = 1,
    portfolio_value: float = 1.0,
) -> float:
    """
    Historical simulation VaR.

    Parameters
    ----------
    returns : pd.Series
        Per-bar portfolio returns.
    confidence : float
        Confidence level.
    horizon : float
        Horizon in bars of *returns*. A longer horizon scales the 1-bar VaR by
        ``sqrt(horizon)`` (square-root-of-time; the drift is scaled by
        ``sqrt(horizon)`` too, not by ``horizon``).
    portfolio_value : float
        Portfolio NAV for dollar-denominated output.

    Returns
    -------
    float
        VaR as a positive loss magnitude.
    """
    # Square-root-of-time rule: the horizon's VaR is the 1-bar loss quantile
    # times sqrt(horizon), which assumes i.i.d. returns. The quantile includes
    # the drift, so the drift is scaled by sqrt(horizon) too, not by horizon.
    # (Scaling the returns before taking the percentile gives the same number:
    # a percentile scales with its data.)
    var_1d = -float(np.percentile(returns.dropna(), (1 - confidence) * 100))
    var = var_1d * np.sqrt(horizon)
    return float(var * portfolio_value)


def monte_carlo_var(
    returns: pd.Series,
    confidence: float = 0.95,
    horizon: int = 10,
    n_simulations: int = 100_000,
    portfolio_value: float = 1.0,
    seed: int | None = 42,
) -> tuple[float, float]:
    """
    Monte Carlo VaR and CVaR via GBM simulation.

    Parameters
    ----------
    returns : pd.Series
        Per-bar portfolio returns.
    confidence : float
        Confidence level.
    horizon : int
        Simulation horizon in bars of *returns*.
    n_simulations : int
        Number of Monte Carlo paths.
    portfolio_value : float
        Starting portfolio value.
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    tuple[float, float]
        (VaR, CVaR) both as positive loss magnitudes.
    """
    rng = np.random.default_rng(seed)
    mu = float(returns.mean())
    sigma = float(returns.std())

    # Simulate log returns
    log_returns = rng.normal(
        loc=(mu - 0.5 * sigma ** 2),
        scale=sigma,
        size=(n_simulations, horizon),
    )
    # Terminal portfolio values
    terminal = portfolio_value * np.exp(log_returns.sum(axis=1))
    final_pnl = terminal - portfolio_value  # profit/loss

    # VaR: negative of the (1-confidence) percentile of PnL
    var_threshold = np.percentile(final_pnl, (1 - confidence) * 100)
    var = float(-var_threshold)

    # CVaR (Expected Shortfall): mean of losses beyond VaR
    tail_losses = final_pnl[final_pnl <= var_threshold]
    cvar = float(-tail_losses.mean()) if len(tail_losses) > 0 else var

    return var, cvar


def parametric_cvar(
    returns: pd.Series,
    confidence: float = 0.95,
    horizon: int = 1,
    portfolio_value: float = 1.0,
) -> float:
    """
    Parametric CVaR (Expected Shortfall) assuming normal distribution.

    CVaR = -μ + σ * φ(z) / (1 - confidence)
    where φ is the normal PDF, z is the confidence quantile.

    Returns
    -------
    float
        CVaR as positive loss magnitude.
    """
    mu = float(returns.mean()) * horizon
    sigma = float(returns.std()) * np.sqrt(horizon)
    alpha = 1 - confidence
    z = stats.norm.ppf(alpha)
    cvar = -(mu - sigma * stats.norm.pdf(z) / alpha)
    return float(cvar * portfolio_value)


# Fewest daily returns the API and CLI treat as a VaR estimate: about a quarter
# of a year. A 2-bar "VaR" came back as a gain before the API refused it.
MIN_VAR_RETURNS = 60


def var_report(
    returns: pd.Series,
    portfolio_value: float = 100_000.0,
    confidence_levels: list[float] | None = None,
    periods_per_year: float | None = None,
) -> pd.DataFrame:
    """
    Generate a comprehensive VaR report at multiple confidence levels.

    Returns a DataFrame with rows = confidence levels, labelled to the precision
    they were given (97.5%, not 98%), cols = method × horizon.

    The 1-day and 10-day horizons are trading days whatever the bars are: on
    weekly returns one day is 52 / 252 of a bar. ``periods_per_year`` is read from
    the dates of *returns* unless given (see ``alphaforge.backtest.metrics``).
    """
    if confidence_levels is None:
        confidence_levels = [0.90, 0.95, 0.99]
    if len(confidence_levels) == 0:
        raise ValueError("var_report: confidence_levels must not be empty.")

    periods = infer_periods_per_year(returns.index) if periods_per_year is None else periods_per_year
    one_day = periods / TRADING_DAYS  # in bars of *returns*

    rows = []
    for conf in confidence_levels:
        p_var_1d = parametric_var(returns, conf, one_day, portfolio_value)
        h_var_1d = historical_var(returns, conf, one_day, portfolio_value)
        p_var_10d = parametric_var(returns, conf, 10 * one_day, portfolio_value)
        h_var_10d = historical_var(returns, conf, 10 * one_day, portfolio_value)
        p_cvar = parametric_cvar(returns, conf, one_day, portfolio_value)
        rows.append({
            "confidence": f"{conf * 100:.10g}%",
            "param_var_1d": p_var_1d,
            "hist_var_1d": h_var_1d,
            "param_var_10d": p_var_10d,
            "hist_var_10d": h_var_10d,
            "param_cvar_1d": p_cvar,
        })

    return pd.DataFrame(rows).set_index("confidence")
