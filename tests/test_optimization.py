"""Tests for portfolio optimization."""

import numpy as np
import pandas as pd
import pytest

from alphaforge.optimization.markowitz import MarkowitzOptimizer, OptimizationResult


def make_returns(n_assets: int = 4, n_days: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate correlated return matrix."""
    rng = np.random.default_rng(seed)
    # Random correlation structure
    L = rng.normal(0, 0.05, (n_assets, n_assets))
    cov = L @ L.T / n_assets
    mus = rng.uniform(0.0001, 0.001, n_assets)
    raw = rng.multivariate_normal(mus, cov, size=n_days)
    symbols = [f"ASSET{i+1}" for i in range(n_assets)]
    return pd.DataFrame(raw, columns=symbols)


class TestMarkowitzOptimizer:
    @pytest.fixture
    def optimizer(self):
        returns = make_returns()
        return MarkowitzOptimizer(returns)

    def test_weights_sum_to_one(self, optimizer):
        result = optimizer.optimize("max_sharpe")
        assert result.weights.sum() == pytest.approx(1.0, abs=1e-4)

    def test_weights_nonnegative_long_only(self, optimizer):
        result = optimizer.optimize("max_sharpe")
        assert (result.weights >= -1e-6).all()

    def test_min_vol_less_volatile_than_max_sharpe(self, optimizer):
        max_sharpe = optimizer.optimize("max_sharpe")
        min_vol = optimizer.optimize("min_volatility")
        assert min_vol.volatility <= max_sharpe.volatility + 1e-4

    def test_equal_weight_exact(self, optimizer):
        result = optimizer.optimize("equal_weight")
        n = len(result.weights)
        expected = 1.0 / n
        assert (result.weights - expected).abs().max() < 1e-9

    def test_risk_parity_weights_sum_to_one(self, optimizer):
        result = optimizer.optimize("risk_parity")
        assert result.weights.sum() == pytest.approx(1.0, abs=1e-3)

    def test_max_diversification(self, optimizer):
        result = optimizer.optimize("max_diversification")
        assert result.weights.sum() == pytest.approx(1.0, abs=1e-4)

    def test_invalid_objective(self, optimizer):
        with pytest.raises(ValueError):
            optimizer.optimize("magic_objective")

    def test_result_has_expected_fields(self, optimizer):
        result = optimizer.optimize("max_sharpe")
        assert isinstance(result, OptimizationResult)
        assert result.expected_return is not None
        assert result.volatility > 0
        assert result.sharpe is not None
        assert result.objective == "max_sharpe"

    def test_efficient_frontier_returns_dataframe(self, optimizer):
        frontier = optimizer.efficient_frontier(n_points=10)
        assert isinstance(frontier, pd.DataFrame)
        assert len(frontier) > 0
        assert "expected_return" in frontier.columns
        assert "volatility" in frontier.columns

    def test_allows_short_selling(self):
        returns = make_returns()
        opt = MarkowitzOptimizer(returns, allow_short=True)
        result = opt.optimize("max_sharpe")
        # With short selling, some weights can be negative
        assert result.weights.sum() == pytest.approx(1.0, abs=1e-4)

    def test_two_asset_portfolio(self):
        """Simple 2-asset sanity check."""
        dates = pd.date_range("2020-01-01", periods=300, freq="B")
        returns = pd.DataFrame({
            "A": np.random.default_rng(1).normal(0.001, 0.01, 300),
            "B": np.random.default_rng(2).normal(0.0005, 0.015, 300),
        }, index=dates)
        opt = MarkowitzOptimizer(returns)
        result = opt.optimize("min_volatility")
        assert result.weights.sum() == pytest.approx(1.0, abs=1e-4)
        assert result.volatility > 0


@pytest.mark.parametrize("objective", ["max_sharpe", "min_volatility", "risk_parity", "max_diversification"])
def test_a_failed_solve_is_an_error_not_an_answer(objective, monkeypatch):
    """No objective may use res.x when res.success says the solve failed."""
    from scipy.optimize import OptimizeResult

    import alphaforge.optimization.markowitz as mk

    def failed(fun, x0, **kwargs):
        return OptimizeResult(x=np.asarray(x0), success=False, message="Iteration limit reached")

    monkeypatch.setattr(mk, "minimize", failed)
    rng = np.random.default_rng(1)
    returns = pd.DataFrame(rng.normal(0.0005, 0.01, (300, 3)), columns=["A", "B", "C"])
    with pytest.raises(ValueError, match="did not converge"):
        mk.MarkowitzOptimizer(returns).optimize(objective)


def test_api_reports_a_failed_solve_as_a_client_visible_error(monkeypatch):
    from fastapi.testclient import TestClient
    from scipy.optimize import OptimizeResult

    import alphaforge.optimization.markowitz as mk
    from alphaforge.api.app import app

    monkeypatch.setattr(mk, "minimize", lambda fun, x0, **kw: OptimizeResult(x=x0, success=False, message="stuck"))
    r = TestClient(app).post("/optimize", json={"objective": "max_sharpe"})
    assert r.status_code == 400 and "did not converge" in r.text


@pytest.mark.parametrize("n_assets,n_days,need", [(2, 3, 60), (2, 59, 60), (50, 99, 100)])
def test_too_few_observations_are_an_error_not_an_optimum(n_assets, n_days, need):
    """The library refuses these as the API does, so a three-day window never prints
    a Sharpe."""
    returns = pd.DataFrame(np.random.default_rng(1).normal(0.001, 0.01, (n_days, n_assets)))
    with pytest.raises(ValueError, match=f"need at least {need} daily returns, got {n_days}"):
        MarkowitzOptimizer(returns)


def test_the_minimum_itself_is_accepted():
    returns = pd.DataFrame(np.random.default_rng(1).normal(0.001, 0.01, (60, 3)))
    assert MarkowitzOptimizer(returns).optimize("equal_weight").weights.sum() == pytest.approx(1.0)


def test_reported_figures_are_the_portfolios_own_annualised_statistics():
    """Computed from the portfolio's return series, not through the optimizer's matrices."""
    index = pd.date_range("2020-01-01", periods=500, freq="B")
    returns = make_returns().set_index(index)
    result = MarkowitzOptimizer(returns).optimize("equal_weight")
    portfolio = returns.mean(axis=1)
    assert result.expected_return == pytest.approx(portfolio.mean() * 252, rel=1e-12)
    assert result.volatility == pytest.approx(portfolio.std() * np.sqrt(252), rel=1e-12)
    assert result.sharpe == pytest.approx((result.expected_return - 0.02) / result.volatility, rel=1e-12)


@pytest.mark.parametrize("seed", [42, 7])
def test_the_frontier_holds_no_dominated_portfolio(seed):
    """More return never comes with less volatility: such a point has a better twin."""
    index = pd.date_range("2020-01-01", periods=500, freq="B")
    opt = MarkowitzOptimizer(make_returns(seed=seed).set_index(index))
    frontier = opt.efficient_frontier(n_points=25).sort_values("expected_return")
    assert len(frontier) >= 20
    assert (frontier["volatility"].diff().dropna() > -1e-6).all(), frontier[["expected_return", "volatility"]]
    assert frontier["volatility"].iloc[0] == pytest.approx(opt.optimize("min_volatility").volatility, rel=1e-3)
    assert frontier["expected_return"].iloc[-1] == pytest.approx(opt.mu.max(), rel=1e-6)


def _uncorrelated_returns(vols, n_days: int = 500, seed: int = 3) -> pd.DataFrame:
    """Returns with exactly these sample vols and exactly zero sample correlation."""
    raw = np.random.default_rng(seed).normal(size=(n_days, len(vols)))
    raw -= raw.mean(axis=0)
    q, _ = np.linalg.qr(raw)
    scaled = q * np.sqrt(n_days - 1) * np.asarray(vols)
    index = pd.date_range("2020-01-01", periods=n_days, freq="B")
    return pd.DataFrame(scaled + 0.0005, columns=[f"V{i}" for i in range(len(vols))], index=index)


def _risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.Series:
    w = weights.to_numpy()
    return pd.Series(w * (cov.to_numpy() @ w) / (w @ cov.to_numpy() @ w), index=weights.index)


def _diversification_ratio(w: np.ndarray, cov: pd.DataFrame) -> float:
    return float(w @ np.sqrt(np.diag(cov.to_numpy())) / np.sqrt(w @ cov.to_numpy() @ w))


@pytest.mark.parametrize("objective", ["risk_parity", "max_diversification"])
def test_uncorrelated_assets_are_held_in_inverse_proportion_to_their_volatility(objective):
    """The closed form for both objectives at zero correlation: vols 1/2/4% hold
    4/7, 2/7, 1/7. Equal weight, which also sums to one, must fail this."""
    returns = _uncorrelated_returns([0.01, 0.02, 0.04])
    weights = MarkowitzOptimizer(returns).optimize(objective).weights
    assert weights.to_numpy() == pytest.approx([4 / 7, 2 / 7, 1 / 7], abs=1e-3)


def test_risk_parity_equalises_risk_contributions_on_a_correlated_book():
    opt = MarkowitzOptimizer(make_returns().set_index(pd.date_range("2020-01-01", periods=500, freq="B")))
    result = opt.optimize("risk_parity")
    assert result.objective == "risk_parity"
    contributions = _risk_contributions(result.weights, opt.cov)
    assert contributions.to_numpy() == pytest.approx([0.25] * 4, abs=1e-3)
    equal = pd.Series(0.25, index=result.weights.index)
    assert _risk_contributions(equal, opt.cov).sub(0.25).abs().max() > 0.02


def test_max_diversification_beats_every_other_portfolio_on_its_own_ratio():
    opt = MarkowitzOptimizer(make_returns().set_index(pd.date_range("2020-01-01", periods=500, freq="B")))
    result = opt.optimize("max_diversification")
    assert result.objective == "max_diversification"
    best = _diversification_ratio(result.weights.to_numpy(), opt.cov)
    assert best > _diversification_ratio(np.full(4, 0.25), opt.cov) + 0.01
    for w in np.random.default_rng(5).dirichlet(np.ones(4), size=2000):
        assert _diversification_ratio(w, opt.cov) <= best + 1e-6

