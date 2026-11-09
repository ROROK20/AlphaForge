"""
Markowitz mean-variance portfolio optimization.

Uses scipy.optimize for constrained quadratic programming.
Supports max-Sharpe, min-volatility, and max-diversification objectives.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from alphaforge.backtest.metrics import infer_periods_per_year


RISK_FREE_RATE = 0.02  # annual
# Below this a covariance matrix and mean returns are not statistics: 60 daily
# returns is about a quarter of a year, and there must be more observations
# than assets. The library, the CLI and the API all refuse fewer.
MIN_OPTIMIZE_RETURNS = 60

# What the API and the CLI print beside a result: the optimizer fits sample means
# and a sample covariance, and reports that same sample back to you.
IN_SAMPLE_NOTE = (
    "In-sample: the weights are fitted to this window, and the expected return, "
    "volatility and Sharpe are measured on the same window. They describe it; "
    "they forecast nothing."
)
# "One asset takes everything", allowing for the solver stopping just short of 1.
CONCENTRATED_WEIGHT = 0.99


def concentration_warning(weights: pd.Series) -> str | None:
    """The warning for a corner solution, or None for a portfolio that holds several assets."""
    top = weights.idxmax()
    if weights[top] < CONCENTRATED_WEIGHT:
        return None
    return (
        f"{top} takes {weights[top]:.1%} of this portfolio: a corner solution on this "
        "window's sample, not a diversified allocation."
    )


def min_observations(n_assets: int) -> int:
    """Fewest returns :class:`MarkowitzOptimizer` accepts for *n_assets*."""
    return max(MIN_OPTIMIZE_RETURNS, 2 * n_assets)


@dataclass
class OptimizationResult:
    """Output from portfolio optimization."""

    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe: float
    objective: str

    def __repr__(self) -> str:
        top = self.weights.nlargest(5)
        return (
            f"OptimizationResult(objective={self.objective!r}, "
            f"sharpe={self.sharpe:.2f}, ret={self.expected_return:.2%}, "
            f"vol={self.volatility:.2%})\n"
            f"  Top 5 weights:\n{top.to_string()}"
        )


def _solved(res, objective: str):
    """Return a scipy result only if the solver converged.

    A failed solve's ``res.x`` is the initial equal-weight guess or wherever the
    solver stopped, not an optimum.
    """
    if not res.success:
        raise ValueError(f"{objective}: optimizer did not converge ({res.message})")
    return res


class MarkowitzOptimizer:
    """
    Markowitz mean-variance optimizer.

    Parameters
    ----------
    returns : pd.DataFrame
        Per-bar returns, columns = asset symbols. After dropping rows with a
        NaN, at least :func:`min_observations` of them (``ValueError`` below).
    risk_free_rate : float
        Annualized risk-free rate (default 0.02).
    allow_short : bool
        Allow negative weights (short selling). Default False.
    max_weight : float
        Maximum weight per asset (default 1.0 = no limit).
    min_weight : float
        Minimum weight per asset (default 0.0 for long-only).
    periods_per_year : float | None
        Bars per year, to annualise means and covariances. None reads it from
        the dates of *returns* (see ``alphaforge.backtest.metrics``).

    Examples
    --------
    >>> opt = MarkowitzOptimizer(returns_df)
    >>> result = opt.optimize(objective='max_sharpe')
    >>> print(result.weights)
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = RISK_FREE_RATE,
        allow_short: bool = False,
        max_weight: float = 1.0,
        min_weight: float = 0.0,
        periods_per_year: float | None = None,
    ) -> None:
        self.returns = returns.dropna()
        self.symbols = list(returns.columns)
        self.n = len(self.symbols)
        need = min_observations(self.n)
        if len(self.returns) < need:
            raise ValueError(
                f"Not enough data to optimize {self.n} assets: need at least {need} "
                f"daily returns, got {len(self.returns)}."
            )
        self.risk_free_rate = risk_free_rate
        self.allow_short = allow_short
        self.max_weight = max_weight
        self.min_weight = -max_weight if allow_short else min_weight

        # Annualized stats
        periods = infer_periods_per_year(self.returns.index) if periods_per_year is None else periods_per_year
        self.mu = self.returns.mean() * periods  # expected annual returns
        self.cov = self.returns.cov() * periods  # annual covariance matrix

    def optimize(self, objective: str = "max_sharpe") -> OptimizationResult:
        """
        Run optimization.

        Parameters
        ----------
        objective : str
            One of: 'max_sharpe', 'min_volatility', 'max_diversification',
            'risk_parity', 'equal_weight'.

        Returns
        -------
        OptimizationResult

        Raises
        ------
        ValueError
            For an unknown objective, a solve that does not converge, and
            'max_sharpe' when no portfolio within the weight limits has an
            expected return above ``risk_free_rate``. Every Sharpe ratio is then
            negative, and the largest belongs to the most volatile portfolio, so
            the objective is refused rather than answered; 'min_volatility' still
            answers. With ``allow_short`` a long/short book counts: the test is on
            the best return the weight limits allow, not on each asset.
        """
        if objective == "equal_weight":
            return self._equal_weight()
        elif objective == "max_sharpe":
            return self._max_sharpe()
        elif objective == "min_volatility":
            return self._min_volatility()
        elif objective == "max_diversification":
            return self._max_diversification()
        elif objective == "risk_parity":
            return self._risk_parity()
        else:
            raise ValueError(f"Unknown objective: {objective!r}. "
                             "Use 'max_sharpe', 'min_volatility', 'max_diversification', "
                             "'risk_parity', or 'equal_weight'.")

    def efficient_frontier(self, n_points: int = 50) -> pd.DataFrame:
        """
        Compute the efficient frontier.

        Target returns run from the minimum-volatility portfolio's up to the best
        asset's. Below that portfolio's return lies the inefficient branch, where
        each point has a twin with the same volatility and a higher return.

        Returns
        -------
        pd.DataFrame with columns: expected_return, volatility, sharpe, weights_*.
        """
        min_ret = self._min_volatility().expected_return
        max_ret = max(float(self.mu.max()), min_ret)
        targets = np.linspace(min_ret, max_ret, n_points)

        rows = []
        for target_ret in targets:
            result = self._min_vol_for_return(target_ret)
            if result is not None:
                row = {
                    "expected_return": result.expected_return,
                    "volatility": result.volatility,
                    "sharpe": result.sharpe,
                }
                for sym, w in result.weights.items():
                    row[f"w_{sym}"] = w
                rows.append(row)

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # private optimizers
    # ------------------------------------------------------------------

    def _bounds_and_constraints(self) -> tuple:
        bounds = tuple((self.min_weight, self.max_weight) for _ in range(self.n))
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        return bounds, constraints

    def _portfolio_stats(self, weights: np.ndarray) -> tuple[float, float, float]:
        """Return (expected_return, volatility, sharpe) for given weights."""
        w = np.array(weights)
        ret = float(w @ self.mu.values)
        vol = float(np.sqrt(w @ self.cov.values @ w))
        # Undefined with zero volatility: NaN, never 0 (which reads as "no
        # edge"); the API sends null.
        sharpe = (ret - self.risk_free_rate) / vol if vol > 1e-10 else float("nan")
        return ret, vol, sharpe

    def _make_result(self, weights: np.ndarray, objective: str) -> OptimizationResult:
        w_series = pd.Series(weights, index=self.symbols)
        ret, vol, sharpe = self._portfolio_stats(weights)
        return OptimizationResult(
            weights=w_series,
            expected_return=ret,
            volatility=vol,
            sharpe=sharpe,
            objective=objective,
        )

    def _best_attainable_return(self) -> float | None:
        """The highest expected return of any portfolio within the weight limits.

        Maximising ``w @ mu`` subject to ``min_weight <= w_i <= max_weight`` and
        ``sum(w) == 1`` is solved exactly by starting every weight at its floor and
        handing the rest of the budget to the highest-return assets first. None when
        no weights satisfy the limits (the solver then reports the failure).
        """
        lo, hi = self.min_weight, self.max_weight
        if self.n * lo > 1 or self.n * hi < 1:
            return None
        weights = np.full(self.n, lo, dtype=float)
        budget = 1.0 - weights.sum()
        for i in np.argsort(-self.mu.values):
            step = min(hi - lo, budget)
            weights[i] += step
            budget -= step
        return float(weights @ self.mu.values)

    def _max_sharpe(self) -> OptimizationResult:
        best = self._best_attainable_return()
        if best is not None and best <= self.risk_free_rate:
            raise ValueError(
                f"max_sharpe has no answer for these returns: no portfolio within the "
                f"weight limits earns more than the {self.risk_free_rate:.2%} risk-free "
                f"rate (the best expected return they allow is {best:.2%} a year). Every "
                "Sharpe ratio is negative, and the largest belongs to the most volatile "
                "portfolio. Use min_volatility for the least risky one."
            )
        bounds, constraints = self._bounds_and_constraints()
        x0 = np.ones(self.n) / self.n

        def neg_sharpe(w):
            ret, vol, sharpe = self._portfolio_stats(w)
            return -sharpe

        res = _solved(minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints,
                               options={"ftol": 1e-9, "maxiter": 1000}), "max_sharpe")
        weights = res.x / res.x.sum()  # normalize
        return self._make_result(weights, "max_sharpe")

    def _min_volatility(self) -> OptimizationResult:
        bounds, constraints = self._bounds_and_constraints()
        x0 = np.ones(self.n) / self.n

        def portfolio_vol(w):
            return float(np.sqrt(np.array(w) @ self.cov.values @ np.array(w)))

        res = _solved(minimize(portfolio_vol, x0, method="SLSQP", bounds=bounds, constraints=constraints,
                               options={"ftol": 1e-9, "maxiter": 1000}), "min_volatility")
        weights = res.x / res.x.sum()
        return self._make_result(weights, "min_volatility")

    def _max_diversification(self) -> OptimizationResult:
        """Maximize diversification ratio = weighted avg vol / portfolio vol."""
        bounds, constraints = self._bounds_and_constraints()
        x0 = np.ones(self.n) / self.n
        individual_vols = np.sqrt(np.diag(self.cov.values))

        def neg_diversification(w):
            w = np.array(w)
            weighted_vols = float(w @ individual_vols)
            port_vol = float(np.sqrt(w @ self.cov.values @ w))
            return -(weighted_vols / port_vol) if port_vol > 1e-10 else 0.0

        res = _solved(minimize(neg_diversification, x0, method="SLSQP", bounds=bounds,
                               constraints=constraints, options={"ftol": 1e-9, "maxiter": 1000}),
                      "max_diversification")
        weights = res.x / res.x.sum()
        return self._make_result(weights, "max_diversification")

    def _risk_parity(self) -> OptimizationResult:
        """Equal risk contribution (risk parity) portfolio."""
        bounds, constraints = self._bounds_and_constraints()
        x0 = np.ones(self.n) / self.n
        cov_mat = self.cov.values

        def risk_parity_objective(w):
            w = np.array(w)
            port_var = float(w @ cov_mat @ w)
            marginal_risk = cov_mat @ w
            risk_contrib = w * marginal_risk / port_var if port_var > 1e-10 else w
            target = np.ones(self.n) / self.n
            return float(np.sum((risk_contrib - target) ** 2))

        res = _solved(minimize(risk_parity_objective, x0, method="SLSQP", bounds=bounds,
                               constraints=constraints, options={"ftol": 1e-12, "maxiter": 2000}),
                      "risk_parity")
        weights = np.abs(res.x) / np.abs(res.x).sum()
        return self._make_result(weights, "risk_parity")

    def _equal_weight(self) -> OptimizationResult:
        weights = np.ones(self.n) / self.n
        return self._make_result(weights, "equal_weight")

    def _min_vol_for_return(self, target_return: float) -> OptimizationResult | None:
        bounds, base_constraints = self._bounds_and_constraints()
        constraints = base_constraints + [
            {"type": "eq", "fun": lambda w: float(np.array(w) @ self.mu.values) - target_return}
        ]
        x0 = np.ones(self.n) / self.n

        def portfolio_vol(w):
            return float(np.sqrt(np.array(w) @ self.cov.values @ np.array(w)))

        res = minimize(portfolio_vol, x0, method="SLSQP", bounds=bounds, constraints=constraints,
                       options={"ftol": 1e-9, "maxiter": 500})
        if res.success:
            weights = res.x / res.x.sum()
            return self._make_result(weights, "min_vol_for_return")
        return None
