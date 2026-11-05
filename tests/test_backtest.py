"""Tests for the backtesting engine and metrics."""

import numpy as np
import pandas as pd
import pytest

from alphaforge.backtest.engine import FILLS, BacktestEngine, BacktestResult
from alphaforge.backtest.metrics import (
    compute_metrics,
    annualized_return,
    rolling_sharpe,
)
from alphaforge.signals.ensemble import EnsembleSignal
from alphaforge.signals.technical import (
    SMACrossoverSignal,
    RSIMeanReversionSignal,
    MACDSignal,
    BollingerBandSignal,
)


def make_ohlcv(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "Open": close, "High": close * 1.005, "Low": close * 0.995,
        "Close": close, "Volume": np.ones(n) * 1e6,
    }, index=dates)


def make_signals(n: int, pattern: str = "all_long") -> pd.Series:
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    if pattern == "all_long":
        return pd.Series(1, index=dates)
    elif pattern == "all_short":
        return pd.Series(-1, index=dates)
    elif pattern == "alternating":
        vals = np.where(np.arange(n) % 10 < 5, 1, -1)
        return pd.Series(vals, index=dates)
    return pd.Series(0, index=dates)


class TestBacktestEngine:
    def test_returns_backtest_result(self):
        df = make_ohlcv()
        signals = make_signals(len(df))
        engine = BacktestEngine()
        result = engine.run(signals, df)
        assert isinstance(result, BacktestResult)

    def test_equity_curve_length(self):
        df = make_ohlcv(100)
        signals = make_signals(100)
        engine = BacktestEngine()
        result = engine.run(signals, df)
        assert len(result.equity_curve) == 100

    def test_no_nan_in_equity_curve(self):
        df = make_ohlcv()
        signals = make_signals(len(df))
        result = BacktestEngine().run(signals, df)
        assert not result.equity_curve.isna().any()

    def test_flat_signal_constant_equity(self):
        """Zero signal = no trades = equity stays at initial capital."""
        df = make_ohlcv(50)
        signals = make_signals(50, "all_long") * 0
        engine = BacktestEngine(initial_capital=100_000, commission=0)
        result = engine.run(signals, df)
        # Equity should start and end at initial capital
        assert abs(result.equity_curve.iloc[0] - 100_000) < 1e-6

    def test_commission_reduces_returns(self):
        df = make_ohlcv(200)
        signals = make_signals(200, "alternating")
        no_cost = BacktestEngine(commission=0).run(signals, df)
        with_cost = BacktestEngine(commission=0.01).run(signals, df)
        # With high commission, total return must be worse
        assert with_cost.equity_curve.iloc[-1] < no_cost.equity_curve.iloc[-1]

    def test_long_only_no_short_returns(self):
        df = make_ohlcv()
        signals = make_signals(len(df), "all_short")
        engine = BacktestEngine(long_only=True)
        result = engine.run(signals, df)
        # Long-only: short signals are clipped → no positions → equity flat
        assert result.equity_curve.iloc[-1] == pytest.approx(100_000, rel=0.01)

    def test_metrics_populated(self):
        df = make_ohlcv()
        signals = make_signals(len(df))
        result = BacktestEngine().run(signals, df)
        for key in ["sharpe", "cagr", "max_drawdown", "volatility"]:
            assert key in result.metrics

    def test_initial_capital_respected(self):
        df = make_ohlcv(50)
        signals = make_signals(50)
        result = BacktestEngine(initial_capital=50_000).run(signals, df)
        # First value should be close to 50_000
        assert abs(result.equity_curve.iloc[0] - 50_000) < 500


class TestExecutionModel:
    """Pins the fill model the engine documents.

    The engine never reads Open: the position on bar t is the signal from bar
    t-1, filled at close(t-1). These tests pin the documented behaviour so the
    prose and the code cannot diverge silently.
    """

    def test_open_high_low_never_affect_results(self):
        df = make_ohlcv(120)
        raw = SMACrossoverSignal(fast=5, slow=20).generate(df)
        scrambled = df.copy()
        rng = np.random.default_rng(3)
        for col in ("Open", "High", "Low"):
            scrambled[col] = rng.uniform(1, 1000, len(df))
        a = BacktestEngine().run(raw, df)
        b = BacktestEngine().run(raw, scrambled)
        pd.testing.assert_series_equal(a.equity_curve, b.equity_curve)
        pd.testing.assert_frame_equal(a.trades, b.trades)

    def test_position_is_previous_bars_signal_filled_at_its_close(self):
        """Shares held on bar t were bought at close(t-1) for the signal computed
        on bar t-1, worth target x equity(t-1) there; returns[t] ==
        shares(t) x (close(t) - close(t-1)) / equity(t-1), exactly.
        """
        df = make_ohlcv(80)
        rng = np.random.default_rng(5)
        raw = pd.Series(rng.choice([-1, 0, 1], len(df)), index=df.index)
        result = BacktestEngine(commission=0.0, slippage=0.0).run(raw, df)
        close, eq, shares = df["Close"], result.equity_curve, result.positions["shares"]
        prev_eq = eq.shift(1).fillna(100_000)
        expected = shares * close.diff().fillna(0) / prev_eq
        assert np.allclose(result.returns.to_numpy(), expected.to_numpy(), atol=1e-12)
        held = raw.shift(1).fillna(0)
        entries = held.ne(held.shift(1)) & held.ne(0)
        for t in np.flatnonzero(entries.to_numpy()):
            assert shares.iloc[t] * close.iloc[t - 1] == pytest.approx(held.iloc[t] * eq.iloc[t - 1])
        # Trade-log entry price: the close of the bar the signal was computed on.
        for _, row in result.trades.iterrows():
            assert row["entry_price"] == pytest.approx(close.loc[row["entry_date"]])

    def test_a_signal_cannot_earn_the_bar_it_was_computed_on(self):
        """No look-ahead: a signal equal to the sign of its OWN bar's return
        (legitimately computable at that bar's close) must not collect that
        return -- shares bought for it only ever earn the NEXT bar.
        """
        df = make_ohlcv(400, seed=8)
        ret = df["Close"].pct_change().fillna(0)
        same_bar_oracle = np.sign(ret).astype(int)
        result = BacktestEngine(commission=0.0, slippage=0.0).run(same_bar_oracle, df)
        # Had the engine applied signal[t] to bar t, every bar would earn |ret|.
        assert result.returns.sum() < 0.5 * ret.abs().sum()
        held = same_bar_oracle.shift(1).fillna(0)
        assert (np.sign(result.positions["shares"]) == np.sign(held)).all()

class TestTradeLogReconciliation:
    """The trade log's accounting must reconcile with the equity curve.

    Historically the trade log baked slippage into execution prices while the
    equity curve charged costs separately, ignored commissions entirely, and
    documented none of it — the two views of P&L could not be compared. These
    tests pin the accounting contract described in ``_build_trades``.
    """

    def _run(self, commission=0.001, slippage=0.0005, n=200):
        df = make_ohlcv(n)
        signals = make_signals(n, "alternating")
        engine = BacktestEngine(
            initial_capital=100_000, commission=commission, slippage=slippage
        )
        return engine, engine.run(signals, df)

    def test_trade_log_has_commission_and_net_columns(self):
        _, result = self._run()
        for col in ["commission_abs", "pnl_net_abs"]:
            assert col in result.trades.columns

    def test_net_pnl_identity_per_row(self):
        _, result = self._run()
        t = result.trades
        assert np.allclose(t["pnl_net_abs"], t["pnl_abs"] - t["commission_abs"])

    @staticmethod
    def _capital_base(result, initial_capital=100_000):
        """The equity the curve actually compounded on, per bar.

        Bar d's P&L accrues against bar d-1's equity, with initial capital
        standing in before the first bar. This is the accrual base the trade
        log must use; it is NOT recoverable as ``pnl_abs / pnl_pct`` (for a
        multi-bar hold no single such base exists).
        """
        base = result.equity_curve.shift(1)
        base.iloc[0] = initial_capital
        return base

    def test_reversal_bars_split_two_commission_legs(self):
        """Alternating signals: every close is a reversal (or the final MTM).

        A reversal trades both positions' notional, so it pays commission on
        both legs, one owned by each side: every closed trade owns its entry
        leg (|shares| x entry close) AND its exit leg (|shares| x exit close);
        the final, still-open trade owns its entry leg only.
        """
        engine, result = self._run()
        close = make_ohlcv(200)["Close"]
        t = result.trades
        entry = close.reindex(t["entry_date"]).to_numpy()
        exit_ = close.reindex(t["exit_date"]).to_numpy()
        closed = np.ones(len(t), dtype=bool)
        closed[-1] = False  # the last trade is the mark-to-market close
        expected = engine.commission * t["shares"].abs().to_numpy() * (entry + np.where(closed, exit_, 0.0))
        assert np.allclose(t["commission_abs"], expected)

    def test_reversal_is_charged_two_commission_legs_in_the_equity_curve(self):
        """The equity side of the same contract: a reversal bar's equity change
        is the new shares' move minus commission on BOTH legs' notional.
        """
        n = 30
        df = make_ohlcv(n)
        raw = pd.Series(-1, index=df.index)
        raw.iloc[10:] = 1  # short, then reverse to long
        engine = BacktestEngine(initial_capital=100_000, commission=0.01, slippage=0.0)
        result = engine.run(raw, df)
        close, eq, sh = df["Close"], result.equity_curve, result.positions["shares"]
        k = 11  # the reversal bar (signals shift one bar)
        legs = (abs(sh.iloc[k - 1]) + abs(sh.iloc[k])) * close.iloc[k - 1]
        assert eq.iloc[k] - eq.iloc[k - 1] == pytest.approx(
            sh.iloc[k] * (close.iloc[k] - close.iloc[k - 1]) - 0.01 * legs
        )
        assert sh.iloc[k] * close.iloc[k - 1] == pytest.approx(eq.iloc[k - 1])  # long sized at full equity
        # ...and a plain entry (bar 1: 0 -> short) pays one leg, on the full capital.
        assert eq.iloc[1] - 100_000 == pytest.approx(
            sh.iloc[1] * (close.iloc[1] - close.iloc[0]) - 0.01 * 100_000
        )

    def test_flat_close_charges_two_commission_legs(self):
        """A long stretch closed to flat owns entry AND exit commissions, each on
        |shares| x that fill's close.
        """
        n = 60
        df = make_ohlcv(n)
        raw = pd.Series(0, index=df.index)
        raw.iloc[5:40] = 1  # one long trade, closed back to flat
        engine = BacktestEngine(initial_capital=100_000, commission=0.002, slippage=0.0)
        t = engine.run(raw, df).trades
        assert len(t) == 1
        close, q = df["Close"], t["shares"].iloc[0]
        assert q * close.iloc[5] == pytest.approx(100_000)       # bought at close(5) with all capital
        assert t["commission_abs"].iloc[0] == pytest.approx(0.002 * abs(q) * (close.iloc[5] + close.iloc[40]))

    def test_mtm_open_trade_charges_entry_leg_only(self):
        """A position still open at the end pays no exit commission — the
        equity curve charged none either (no closing trade bar exists)."""
        n = 50
        df = make_ohlcv(n)
        signals = make_signals(n, "all_long")
        engine = BacktestEngine(
            initial_capital=100_000, commission=0.002, slippage=0.0
        )
        result = engine.run(signals, df)
        t = result.trades
        assert len(t) == 1
        entry_base = result.equity_curve.loc[t["entry_date"].iloc[0]]
        assert t["commission_abs"].iloc[0] == pytest.approx(
            engine.commission * entry_base
        )
        # The trade fills at bar 0's close (first held on bar 1), so the base is
        # the untouched starting capital, not equity AFTER the first held bar's move.
        assert entry_base == pytest.approx(100_000)
        assert t["commission_abs"].iloc[0] == pytest.approx(200.0)

    def test_sum_of_net_trade_pnl_tracks_equity_curve(self):
        """Headline reconciliation: total net trade P&L == equity-curve P&L.

        The trade log accrues against marked-to-market equity
        (``engine._build_trades``), which makes this an EXACT decomposition:
        every bar's gross return falls in exactly one trade's held range and
        every cost leg is attributed exactly once, so the sum telescopes onto
        the curve's own per-bar identity. Only float rounding remains — worst
        measured residual across 80 (signal x bar-count x cost) cells was
        $1.6e-9 on $100k, so $1e-6 keeps ~600x margin. Do NOT loosen this: a loose
        bound lets a real reconciliation break pass unnoticed.
        """
        _, result = self._run()
        equity_pnl = result.equity_curve.iloc[-1] - result.equity_curve.iloc[0]
        trade_pnl = result.trades["pnl_net_abs"].sum()
        assert abs(trade_pnl - equity_pnl) < 1e-6


class TestReconciliationInvariant:
    """The reconciliation invariant, pinned on the SHIPPED signals.

    "equity curve <-> trade-log P&L reconcile within 0.1% on every shipped
    strategy" (README, "Testing").

    Parametrised over cost regimes including a zero-cost control (commission=0,
    slippage=0), so a break cannot hide behind a single fixture or be put down to
    costs.
    """

    INITIAL = 100_000.0

    @staticmethod
    def _gbm(n: int, seed: int) -> pd.DataFrame:
        """Deterministic GBM bars, same shape as the API's synthetic feed."""
        rng = np.random.default_rng(seed)
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.012, n)))
        intraday = np.abs(rng.normal(0.0, 0.006, n))
        open_ = np.concatenate([[close[0]], close[:-1]])
        return pd.DataFrame(
            {
                "Open": open_,
                "High": np.maximum.reduce([close * (1 + intraday), open_, close]),
                "Low": np.minimum.reduce([close * (1 - intraday), open_, close]),
                "Close": close,
                "Volume": np.full(n, 1e6),
            },
            index=pd.date_range("2018-01-01", periods=n, freq="B"),
        )

    @pytest.mark.parametrize("n_bars", [250, 1000])
    @pytest.mark.parametrize(
        "commission,slippage",
        [(0.001, 0.0005), (0.0, 0.0), (0.005, 0.002)],
        ids=["defaults", "zero-cost-control", "heavy-cost"],
    )
    @pytest.mark.parametrize(
        "factory",
        [
            lambda: SMACrossoverSignal(fast=20, slow=50),
            lambda: RSIMeanReversionSignal(period=14),
            lambda: MACDSignal(),
            lambda: BollingerBandSignal(period=20, n_std=2.0),
        ],
        ids=["sma_crossover", "rsi_meanrev", "macd", "bollinger"],
    )
    def test_trade_log_reconciles_with_equity_curve(
        self, factory, commission, slippage, n_bars
    ):
        df = self._gbm(n_bars, seed=n_bars + 7)
        engine = BacktestEngine(
            initial_capital=self.INITIAL, commission=commission, slippage=slippage
        )
        result = engine.run(factory(), df)
        assert not result.trades.empty, "fixture must actually trade"

        equity_pnl = float(result.equity_curve.iloc[-1]) - self.INITIAL
        trade_pnl = float(result.trades["pnl_net_abs"].sum())
        gap = abs(trade_pnl - equity_pnl)

        # The invariant's hard bound: within 0.1% of initial capital.
        assert gap < 0.001 * self.INITIAL, (
            f"reconciliation violated: gap ${gap:,.2f} = "
            f"{gap / self.INITIAL * 100:.4f}% of initial capital"
        )
        # The real tolerance: exact to float rounding.
        assert gap < 1e-6, f"reconciliation not exact: gap ${gap:.3e}"

    def test_ensemble_signal_reconciles(self):
        """The reconciliation is signal-agnostic: it holds for a composed signal too.

        ``EnsembleSignal`` is a shipped signal that produces a different
        trade-pattern shape (many short holds) than any single technical rule.
        """
        df = self._gbm(800, seed=555)
        ens = EnsembleSignal(
            [SMACrossoverSignal(20, 50), MACDSignal(), RSIMeanReversionSignal(14)]
        )
        result = BacktestEngine(self.INITIAL, 0.001, 0.0005).run(ens, df)
        assert not result.trades.empty
        gap = abs(
            float(result.trades["pnl_net_abs"].sum())
            - (float(result.equity_curve.iloc[-1]) - self.INITIAL)
        )
        assert gap < 1e-6, f"gap ${gap:.3e}"

    def test_position_opened_on_the_final_bar_is_still_logged(self):
        """A trade opened on the LAST bar is held for exactly that bar.

        The equity curve books that bar's return and charges its commission, so
        the trade log must carry the row, or the bar's P&L goes unattributed.
        """
        n = 40
        df = self._gbm(n, seed=99)
        raw = pd.Series(0, index=df.index)
        raw.iloc[-2] = 1  # filled at bar -2's close -> held only on bar -1
        engine = BacktestEngine(
            initial_capital=self.INITIAL, commission=0.001, slippage=0.0005
        )
        result = engine.run(raw, df)

        assert len(result.trades) == 1, "final-bar entry must still be logged"
        row = result.trades.iloc[0]
        assert row["entry_date"] == df.index[-2]
        assert row["exit_date"] == df.index[-1]
        # Entry commission leg only, against untouched capital (nothing traded
        # before this bar, so the curve was flat at initial capital).
        assert row["commission_abs"] == pytest.approx(0.001 * self.INITIAL)

        equity_pnl = float(result.equity_curve.iloc[-1]) - self.INITIAL
        assert equity_pnl != 0.0, "fixture must move the curve"
        assert abs(float(result.trades["pnl_net_abs"].sum()) - equity_pnl) < 1e-6



class TestFractionalPositions:
    """A signal is a target position in [-1, 1], not an integer: a fraction earns
    that fraction, out-of-domain values raise, and the trade log still reconciles
    exactly when positions are resized inside a trade.
    """

    INITIAL = 100_000.0

    def test_a_fraction_earns_that_fraction_of_the_move(self):
        """A fractional target buys that fraction of equity in shares, and holds
        them (no rebalancing): its P&L is shares x the price change.
        """
        df = make_ohlcv(120, seed=3)
        close = df["Close"]
        for size in (0.99, 0.5, -0.25):
            raw = pd.Series(size, index=df.index)
            result = BacktestEngine(commission=0.0, slippage=0.0).run(raw, df)
            shares = result.positions["shares"]
            assert shares.iloc[1] * close.iloc[0] == pytest.approx(size * 100_000), size
            assert (shares.iloc[1:] == shares.iloc[1]).all(), "held, not rebalanced"
            pnl = shares.iloc[1] * (close.iloc[-1] - close.iloc[0])
            assert result.equity_curve.iloc[-1] - 100_000 == pytest.approx(pnl)

    @pytest.mark.parametrize("bad", [5.0, 1.5, -1.0001, -3.0, np.inf, -np.inf])
    def test_positions_outside_minus_one_to_one_are_rejected(self, bad):
        df = make_ohlcv(60)
        raw = pd.Series(0.0, index=df.index)
        raw.iloc[10:20] = bad
        with pytest.raises(ValueError, match=r"\[-1, 1\]"):
            BacktestEngine().run(raw, df)

    def test_nan_and_missing_bars_mean_flat(self):
        df = make_ohlcv(60)
        raw = pd.Series(1.0, index=df.index[::2])  # every other bar missing
        raw.iloc[3] = np.nan
        result = BacktestEngine().run(raw, df)
        assert set(np.unique(result.positions["signal"])) <= {0.0, 1.0}

    @staticmethod
    def _sticky_fractions(index: pd.Index, seed: int) -> pd.Series:
        """Positions that persist, resize on the same side, flip and go flat."""
        rng = np.random.default_rng(seed)
        levels = np.array([-1.0, -0.6, -0.25, 0.0, 0.0, 0.3, 0.75, 1.0])
        vals = np.empty(len(index))
        cur = 0.0
        for i in range(len(index)):
            if rng.random() < 0.15:
                cur = float(rng.choice(levels))
            vals[i] = cur
        return pd.Series(vals, index=index)

    @pytest.mark.parametrize("commission,slippage", [(0.001, 0.0005), (0.0, 0.0), (0.005, 0.002)])
    def test_resized_trades_reconcile_exactly(self, commission, slippage):
        """Resizing inside a trade leaves the trade log reconciled with the curve."""
        df = make_ohlcv(600, seed=17)
        raw = self._sticky_fractions(df.index, seed=4)
        engine = BacktestEngine(self.INITIAL, commission=commission, slippage=slippage)
        result = engine.run(raw, df)
        close, eq, sh = df["Close"], result.equity_curve, result.positions["shares"]
        # The curve: shares x price move, minus costs on |shares change| x fill close.
        traded = sh.diff().fillna(sh).abs() * close.shift(1).fillna(close.iloc[0])
        d_eq = eq.diff().fillna(eq.iloc[0] - self.INITIAL)
        expected = sh * close.diff().fillna(0) - (commission + slippage) * traded
        assert np.allclose(d_eq.to_numpy(), expected.to_numpy(), atol=1e-6)
        # Resizes really happen inside trades in this fixture.
        pos = raw.shift(1).fillna(0)
        same_side_resize = (np.sign(pos) == np.sign(pos.shift(1))) & (pos != pos.shift(1)) & (pos != 0)
        assert same_side_resize.sum() > 5
        # Exact reconciliation, and every cost leg attributed exactly once.
        t = result.trades
        assert abs(float(t["pnl_net_abs"].sum()) - (float(eq.iloc[-1]) - self.INITIAL)) < 1e-6
        assert float(t["commission_abs"].sum()) == pytest.approx(commission * float(traded.sum()), abs=1e-6)
        assert set(t["direction"]) <= {-1, 1}

    def test_resized_multi_asset_book_reconciles_exactly(self):
        """Resizing the legs of a multi-asset book leaves its trade log reconciled."""
        data_map = {s: make_ohlcv(400, seed=k) for k, s in enumerate(["A", "B", "C"], start=30)}
        signals_map = {
            s: self._sticky_fractions(d.index, seed=k) for k, (s, d) in enumerate(data_map.items())
        }
        weights = {"A": 0.5, "B": 0.3, "C": 0.2}
        result = BacktestEngine(self.INITIAL, 0.001, 0.0005).run_multi(signals_map, data_map, weights=weights)
        sh = result.positions
        close = pd.DataFrame({s: d["Close"] for s, d in data_map.items()})
        traded = (sh.diff().fillna(sh).abs() * close.shift(1).fillna(close.iloc[0])).sum(axis=1)
        d_eq = result.equity_curve.diff().fillna(result.equity_curve.iloc[0] - self.INITIAL)
        expected = (sh * close.diff().fillna(0)).sum(axis=1) - 0.0015 * traded
        assert np.allclose(d_eq.to_numpy(), expected.to_numpy(), atol=1e-6)
        gap = abs(float(result.trades["pnl_net_abs"].sum()) - (float(result.equity_curve.iloc[-1]) - self.INITIAL))
        assert gap < 1e-6

    def test_a_sizer_output_can_be_backtested(self):
        """``signal * sizer.size(capital)`` is how a sizer's output reaches the engine."""
        from alphaforge.risk.position_sizing import KellyCriterion

        df = make_ohlcv(300, seed=9)
        unit = SMACrossoverSignal(fast=10, slow=30).generate(df)
        f = KellyCriterion(win_rate=0.58, avg_win=0.045, avg_loss=0.030, fraction=0.25).size(self.INITIAL)
        assert f == pytest.approx(0.075)
        sized = BacktestEngine(commission=0.0, slippage=0.0).run(unit * f, df)
        full = BacktestEngine(commission=0.0, slippage=0.0).run(unit, df)
        first = sized.positions["shares"].ne(0).idxmax()
        assert sized.positions["shares"][first] == pytest.approx(f * full.positions["shares"][first])
        # The first trade starts from the same capital: f x the shares, f x the P&L.
        assert sized.trades["pnl_net_abs"].iloc[0] == pytest.approx(f * full.trades["pnl_net_abs"].iloc[0])
        assert sized.returns.abs().sum() > 0

class TestRuin:
    """Equity cannot go below zero.

    The first bar with a net return <= -100% ruins the account: equity floors at exactly 0,
    the position is liquidated, nothing trades afterwards, metrics cover the
    path through the ruin bar, and the trade log still reconciles exactly.
    """

    INITIAL = 100_000.0

    @staticmethod
    def _squeeze(n: int = 60, jump_at: int = 30, jump: float = 1.5) -> pd.DataFrame:
        close = np.full(n, 100.0)
        close[jump_at:] = 100.0 * (1 + jump)          # +150% in one bar
        close[jump_at + 5:] *= np.linspace(1.0, 0.5, n - jump_at - 5)  # then it falls
        return pd.DataFrame(
            {"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1e6},
            index=pd.date_range("2021-01-01", periods=n, freq="B"),
        )

    def _run(self, commission=0.001, slippage=0.0005):
        df = self._squeeze()
        raw = pd.Series(-1.0, index=df.index)  # short throughout
        return df, BacktestEngine(self.INITIAL, commission, slippage).run(raw, df)

    def test_equity_never_goes_negative_and_stays_zero(self):
        df, result = self._run()
        eq = result.equity_curve
        assert (eq >= 0).all(), f"min equity {eq.min():.2f}"
        assert result.ruined and result.ruin_date == df.index[30]
        assert (eq.loc[result.ruin_date:] == 0.0).all()
        assert (eq.loc[: df.index[29]] > 0).all()

    def test_nothing_trades_after_ruin(self):
        df, result = self._run()
        # Flat from the bar after ruin; the short's later "profits" (price falls
        # 50%) must not reach a dead account -- before, they moved equity.
        assert (result.positions["signal"].loc[df.index[31]:] == 0).all()
        assert (pd.to_datetime(result.trades["entry_date"]) <= result.ruin_date).all()
        assert result.returns.loc[df.index[32]:].abs().sum() == 0.0

    @pytest.mark.parametrize("commission,slippage", [(0.001, 0.0005), (0.0, 0.0), (0.01, 0.005)])
    def test_trade_log_reconciles_exactly_with_the_floor(self, commission, slippage):
        _, result = self._run(commission, slippage)
        t = result.trades
        assert float(t["pnl_net_abs"].sum()) == pytest.approx(-self.INITIAL, abs=1e-6)
        # The write-off is exactly the loss beyond the account on the ruin bar.
        bar_loss = -float(result.returns.iloc[30]) * self.INITIAL  # capped: -100%
        assert bar_loss == pytest.approx(self.INITIAL)
        assert float(t["ruin_writeoff_abs"].sum()) > 0
        assert (t["ruin_writeoff_abs"] >= 0).all()

    def test_metrics_cover_the_path_through_ruin(self):
        _, result = self._run()
        m = result.metrics
        assert m["total_return"] == pytest.approx(-1.0)
        assert m["max_drawdown"] == pytest.approx(-1.0)
        assert m["cagr"] == pytest.approx(-1.0)
        assert m["n_days"] == 31.0  # bars 0..30, not the dead tail
        assert "RUINED 2021-02-12" in repr(result)

    def test_a_solvent_run_is_untouched(self):
        df = make_ohlcv(300)
        result = BacktestEngine().run(SMACrossoverSignal(10, 30), df)
        assert not result.ruined and result.ruin_date is None
        assert (result.trades["ruin_writeoff_abs"] == 0).all()

    def test_a_ruined_book_in_run_multi_floors_and_reconciles(self):
        squeeze = self._squeeze()
        calm = make_ohlcv(60, seed=3)
        calm.index = squeeze.index
        data_map = {"SQZ": squeeze, "CALM": calm}
        signals_map = {s: pd.Series(-1.0, index=squeeze.index) for s in data_map}
        result = BacktestEngine(self.INITIAL, 0.001, 0.0005).run_multi(
            signals_map, data_map, weights={"SQZ": 0.8, "CALM": 0.2}
        )
        assert result.ruined and (result.equity_curve >= 0).all()
        assert (result.equity_curve.loc[result.ruin_date:] == 0).all()
        assert float(result.trades["pnl_net_abs"].sum()) == pytest.approx(-self.INITIAL, abs=1e-6)

    def test_costs_alone_can_ruin_and_the_closing_trade_owns_the_writeoff(self):
        """Flat on the ruin bar: the closing leg's cost (110% of a full
        position) is what exceeds the account. The position is built in
        resizes so no earlier bar's cost reaches -100%."""
        close = 100 * 1.01 ** np.arange(30)  # steady +1% bars
        df = pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1e6},
                          index=pd.date_range("2021-01-01", periods=30, freq="B"))
        raw = pd.Series(0.0, index=df.index)
        raw.iloc[3], raw.iloc[4], raw.iloc[5:10] = 0.4, 0.8, 1.0  # then flat from bar 10
        result = BacktestEngine(self.INITIAL, commission=0.7, slippage=0.4).run(raw, df)
        ruin_bar = df.index[11]  # the flat-to bar (signals shift one bar)
        assert result.ruined and result.ruin_date == ruin_bar
        assert result.positions["signal"].loc[ruin_bar] == 0.0  # flat ON the ruin bar
        assert (result.equity_curve >= 0).all()
        t = result.trades
        # Filled at the close of the bar before: the cost reaches the curve, and ruins it, on ruin_bar.
        assert len(t) == 1 and pd.Timestamp(t["exit_date"].iloc[0]) == df.index[10]
        assert t["ruin_writeoff_abs"].iloc[0] > 0
        assert float(t["pnl_net_abs"].sum()) == pytest.approx(-self.INITIAL, abs=1e-6)



def _assert_every_logged_price_is_the_close_of_its_date(trades, close, slippage, ruin_date=None):
    """entry_price and exit_price are close(their date) x (1 +/- slippage).

    Only an open position's final mark and a ruined position's close are not
    fills: their exit price is the bare close of that bar.
    """
    assert len(trades), "the fixture must trade"
    for _, row in trades.iterrows():
        d = row["direction"]
        assert row["entry_price"] == pytest.approx(close.loc[row["entry_date"]] * (1 + d * slippage), rel=1e-12)
        not_a_fill = row["exit_date"] == close.index[-1] or row["exit_date"] == ruin_date
        want = close.loc[row["exit_date"]] * (1.0 if not_a_fill else 1 - d * slippage)
        assert row["exit_price"] == pytest.approx(want, rel=1e-12)


class TestTradeDatesAreFillDates:
    """entry_date / exit_date name the bar whose close was filled, on every row:
    closed by a fill, marked to market, or ruined.
    """

    @pytest.mark.parametrize("fill", sorted(FILLS))
    @pytest.mark.parametrize("slippage", [0.0, 0.0005, 0.01])
    @pytest.mark.parametrize("side", [1.0, -1.0])
    def test_every_row_of_every_fill_mode(self, fill, slippage, side):
        df = make_ohlcv(400, seed=31)
        rng = np.random.default_rng(32)
        # Runs of long / flat / short, including resizes inside a run.
        raw = pd.Series(side * rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0], 400), index=df.index)
        raw = raw.where(rng.random(400) < 0.15).ffill().fillna(0.0)
        result = BacktestEngine(commission=0.001, slippage=slippage, fill=fill).run(raw, df)
        _assert_every_logged_price_is_the_close_of_its_date(result.trades, df["Close"], slippage)

    @pytest.mark.parametrize("fill,entry,exit_,entry_px,exit_px", [
        ("close", "2021-01-05", "2021-01-08", 102.0, 105.0),
        ("next_close", "2021-01-06", "2021-01-11", 103.0, 106.0),
    ])
    def test_a_trade_worked_by_hand(self, fill, entry, exit_, entry_px, exit_px):
        """Signal long on 01-05, flat from 01-08: market-on-close buys 01-05's
        close and sells 01-08's; a bar later buys 01-06's and sells 01-11's."""
        idx = pd.DatetimeIndex(["2021-01-04", "2021-01-05", "2021-01-06", "2021-01-07", "2021-01-08",
                                "2021-01-11", "2021-01-12"])
        close = [100.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0]
        df = pd.DataFrame({"Close": close}, index=idx)
        raw = pd.Series([0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], index=idx)
        row = BacktestEngine(commission=0.0, slippage=0.0, fill=fill).run(raw, df).trades.iloc[0]
        assert (row["entry_date"], row["exit_date"]) == (pd.Timestamp(entry), pd.Timestamp(exit_))
        assert (row["entry_price"], row["exit_price"]) == (entry_px, exit_px)

    @pytest.mark.parametrize("fill", sorted(FILLS))
    def test_the_bars_between_the_dates_are_the_bars_held(self, fill):
        """Every row, the still-open last one included."""
        df = make_ohlcv(300, seed=33)
        rng = np.random.default_rng(34)
        raw = pd.Series(rng.choice([-1.0, 0.0, 1.0], 300), index=df.index)
        raw = raw.where(rng.random(300) < 0.1).ffill().fillna(0.0)
        raw.iloc[-20:] = 1.0  # still open on the last bar
        result = BacktestEngine(fill=fill).run(raw, df)
        t = result.trades
        assert t["exit_date"].iloc[-1] == df.index[-1], "the fixture must end in an open position"
        spans = [df.index.get_loc(x) - df.index.get_loc(e) for e, x in zip(t["entry_date"], t["exit_date"])]
        assert sum(spans) == int((result.positions["shares"] != 0).sum())
        assert min(spans) >= 1

    def test_a_position_open_at_the_end_is_marked_at_the_last_close_with_no_slippage(self):
        df = make_ohlcv(50, seed=35)
        raw = pd.Series(1.0, index=df.index)
        row = BacktestEngine(slippage=0.01).run(raw, df).trades.iloc[0]
        assert row["entry_date"] == df.index[0]
        assert row["entry_price"] == pytest.approx(df["Close"].iloc[0] * 1.01)
        assert row["exit_date"] == df.index[-1]
        assert row["exit_price"] == df["Close"].iloc[-1]

    def test_a_ruined_position_exits_on_the_ruin_bar_with_no_slippage(self):
        px = np.r_[np.full(5, 100.0), np.linspace(100.0, 200.0, 11)[1:], np.full(5, 200.0)]
        df = pd.DataFrame({"Close": px}, index=pd.date_range("2024-01-01", periods=len(px), freq="B"))
        result = BacktestEngine(commission=0.0, slippage=0.01).run(pd.Series(-1.0, index=df.index), df)
        assert result.ruined
        _assert_every_logged_price_is_the_close_of_its_date(result.trades, df["Close"], 0.01, result.ruin_date)
        row = result.trades.iloc[0]
        assert row["entry_date"] == df.index[0]
        assert row["exit_date"] == result.ruin_date
        assert row["exit_price"] == df["Close"].loc[result.ruin_date]

    @pytest.mark.parametrize("fill", sorted(FILLS))
    def test_every_asset_of_a_multi_asset_book(self, fill):
        data = {s: make_ohlcv(300, seed=40 + i) for i, s in enumerate("ABC")}
        rng = np.random.default_rng(44)
        sigs = {}
        for s, df in data.items():
            raw = pd.Series(rng.choice([-1.0, 0.0, 1.0], 300), index=df.index)
            sigs[s] = raw.where(rng.random(300) < 0.1).ffill().fillna(0.0)
        t = BacktestEngine(slippage=0.002, fill=fill).run_multi(sigs, data).trades
        assert set(t["symbol"]) == set(data)
        for s, df in data.items():
            _assert_every_logged_price_is_the_close_of_its_date(t[t["symbol"] == s], df["Close"], 0.002)

    def test_the_dates_moved_and_the_money_did_not(self):
        """Reconciliation is untouched: the log still sums to the curve's P&L."""
        df = make_ohlcv(300, seed=45)
        result = BacktestEngine().run(SMACrossoverSignal(5, 20), df)
        assert float(result.trades["pnl_net_abs"].sum()) == pytest.approx(
            result.equity_curve.iloc[-1] - 100_000, abs=1e-6
        )


class TestSharesHeldAtEntry:
    """A position is shares bought at entry and held, not a constant fraction of
    equity re-applied every bar: an uncharged daily rebalance under which a short
    through a steady doubling loses 52.5% instead of everything.
    """

    @staticmethod
    def _ramp(start=100.0, end=200.0, n_up=10):
        px = np.r_[np.full(5, start), np.linspace(start, end, n_up + 1)[1:], np.full(5, end)]
        return pd.DataFrame({"Open": px, "High": px, "Low": px, "Close": px, "Volume": 1e6},
                            index=pd.date_range("2024-01-01", periods=len(px), freq="B"))

    def test_a_short_through_a_doubling_is_ruined(self):
        """Short from 100 to 200 over ten bars, no costs: ruined. A constant-fraction
        model would end at $47,368.42 (-52.6%) on this path."""
        df = self._ramp()
        result = BacktestEngine(commission=0.0, slippage=0.0).run(pd.Series(-1.0, index=df.index), df)
        assert result.ruined
        assert result.ruin_date == df.index[df["Close"].to_numpy().argmax()]  # the bar price reached 200
        assert result.equity_curve.iloc[-1] == 0.0 and (result.equity_curve >= 0).all()
        t = result.trades.iloc[0]
        assert t["pnl_pct"] == pytest.approx(-1.0)
        assert t["pnl_net_abs"] == pytest.approx(-100_000.0)   # logged return == booked P&L

    @pytest.mark.parametrize("direction", [1, -1])
    def test_logged_trade_return_matches_booked_pnl_for_longs_and_shorts(self, direction):
        """Zero costs: every trade's P&L is shares x (exit close - entry close),
        and pnl_pct x entry notional is that same number."""
        df = make_ohlcv(300, seed=21)
        raw = SMACrossoverSignal(fast=5, slow=20).generate(df) * direction
        result = BacktestEngine(commission=0.0, slippage=0.0).run(raw, df)
        close = df["Close"]
        for _, row in result.trades.iterrows():
            entry, exit_ = close.loc[row["entry_date"]], close.loc[row["exit_date"]]
            assert row["pnl_net_abs"] == pytest.approx(row["shares"] * (exit_ - entry), abs=1e-6)
            assert row["pnl_net_abs"] == pytest.approx(abs(row["shares"]) * entry * row["pnl_pct"], abs=1e-6)

    def test_shares_are_not_rebalanced_while_the_target_is_unchanged(self):
        df = self._ramp(100.0, 150.0)
        result = BacktestEngine(commission=0.0, slippage=0.0).run(pd.Series(1.0, index=df.index), df)
        shares = result.positions["shares"].iloc[1:]
        assert shares.nunique() == 1
        assert result.equity_curve.iloc[-1] == pytest.approx(100_000 * 1.5)  # +50%, not compounded weights


class TestRunMultiGross:
    def test_weights_are_scaled_to_one_times_gross(self):
        """{A: 3, B: -2} scales by sum(|w|) = 5, not sum(w) = 1, which would run 5x gross."""
        data_map = {s: make_ohlcv(100, seed=k) for k, s in enumerate(["A", "B"], start=60)}
        signals_map = {s: pd.Series(1.0, index=d.index) for s, d in data_map.items()}
        result = BacktestEngine(100_000, 0.0, 0.0).run_multi(signals_map, data_map, weights={"A": 3, "B": -2})
        r = pd.DataFrame({s: d["Close"].pct_change() for s, d in data_map.items()}).iloc[1]
        # First held bar: 0.6 x A's move - 0.4 x B's (a 5x book: 3 x A - 2 x B).
        assert result.equity_curve.iloc[1] - 100_000 == pytest.approx(100_000 * (0.6 * r["A"] - 0.4 * r["B"]))
        first = result.positions.iloc[1]
        value = first * pd.DataFrame({s: d["Close"] for s, d in data_map.items()}).iloc[0]
        assert value["A"] == pytest.approx(60_000) and value["B"] == pytest.approx(-40_000)
        assert value.abs().sum() == pytest.approx(100_000)    # 1x gross at entry



class TestSignalAlignment:
    """A signal that does not line up with the prices is an error.

    An ndarray, a reset_index()-ed Series, or tz-naive signal dates against tz-aware
    prices, reindexed, would turn to NaN -> flat: zero trades, no word.
    """

    def test_an_array_of_one_value_per_bar_is_accepted(self):
        df = make_ohlcv(200)
        sig = SMACrossoverSignal(10, 30).generate(df)
        a = BacktestEngine().run(sig.to_numpy(), df)
        b = BacktestEngine().run(sig, df)
        pd.testing.assert_series_equal(a.equity_curve, b.equity_curve)
        assert len(a.trades) > 0

    def test_an_array_of_the_wrong_length_is_refused(self):
        df = make_ohlcv(200)
        with pytest.raises(ValueError, match="one value per bar"):
            BacktestEngine().run(np.ones(150), df)

    def test_a_reset_index_series_is_refused(self):
        df = make_ohlcv(200)
        sig = SMACrossoverSignal(10, 30).generate(df).reset_index(drop=True)
        with pytest.raises(ValueError, match="not dates"):
            BacktestEngine().run(sig, df)

    def test_tz_naive_signal_on_tz_aware_prices_is_refused(self):
        df = make_ohlcv(200)
        sig = SMACrossoverSignal(10, 30).generate(df)
        aware = df.tz_localize("America/New_York")
        with pytest.raises(ValueError, match="timezone-aware"):
            BacktestEngine().run(sig, aware)

    def test_a_signal_covering_under_half_the_bars_is_refused(self):
        df = make_ohlcv(200)
        sig = pd.Series(1.0, index=df.index[:60])
        with pytest.raises(ValueError, match="30%"):
            BacktestEngine().run(sig, df)
        # A sub-window that covers most bars is still fine (gaps are flat).
        assert BacktestEngine().run(pd.Series(1.0, index=df.index[20:]), df).trades.shape[0] == 1


class TestInputsTheEngineRefuses:
    """The API's guards hold in the library and the CLI too: a negative commission
    would pay the strategy, a newest-first frame would run backwards."""

    @pytest.mark.parametrize("kwargs,word", [
        ({"commission": -0.05}, "commission"),
        ({"slippage": -0.0001}, "slippage"),
        ({"commission": float("nan")}, "commission"),
        ({"slippage": float("inf")}, "slippage"),
        ({"initial_capital": 0.0}, "initial_capital"),
        ({"initial_capital": -5.0}, "initial_capital"),
        ({"initial_capital": float("inf")}, "initial_capital"),
    ])
    def test_nonsense_costs_and_capital(self, kwargs, word):
        with pytest.raises(ValueError, match=word):
            BacktestEngine(**kwargs)

    def test_a_cost_of_zero_and_a_large_cost_are_still_allowed(self):
        df = make_ohlcv(60)
        assert BacktestEngine(commission=0.0, slippage=0.0).run(make_signals(60), df).metrics
        assert BacktestEngine(commission=0.7, slippage=0.4).run(make_signals(60), df).metrics

    def test_a_newest_first_frame_does_not_run_backwards(self):
        df = make_ohlcv(100)
        sig = SMACrossoverSignal(5, 20).generate(df)
        with pytest.raises(ValueError, match="sort_index"):
            BacktestEngine().run(sig, df.iloc[::-1])
        with pytest.raises(ValueError, match="sort_index"):
            BacktestEngine().run(SMACrossoverSignal(5, 20), df.iloc[::-1])

    def test_duplicate_dates(self):
        df = make_ohlcv(100)
        doubled = pd.concat([df.iloc[:50], df.iloc[49:]])
        with pytest.raises(ValueError, match="one bar per date"):
            BacktestEngine().run(pd.Series(1.0, index=doubled.index), doubled)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0, -1.0])
    def test_a_price_a_fill_could_not_use(self, bad):
        df = make_ohlcv(100)
        df.iloc[40, df.columns.get_loc("Close")] = bad
        with pytest.raises(ValueError, match=str(df.index[40].date())):
            BacktestEngine().run(pd.Series(1.0, index=df.index), df)

    def test_run_multi_checks_every_frame(self):
        a, b = make_ohlcv(100, seed=1), make_ohlcv(100, seed=2)
        sigs = {"A": pd.Series(1.0, index=a.index), "B": pd.Series(1.0, index=a.index)}
        with pytest.raises(ValueError, match="the data for B"):
            BacktestEngine().run_multi(sigs, {"A": a, "B": b.iloc[::-1]})
        b_nan = b.copy()
        b_nan.iloc[40, b_nan.columns.get_loc("Close")] = float("nan")
        with pytest.raises(ValueError, match="the data for B"):
            BacktestEngine().run_multi(sigs, {"A": a, "B": b_nan})

    def test_run_multi_with_no_common_dates(self):
        a, b = make_ohlcv(50, seed=1), make_ohlcv(50, seed=2)
        b.index = b.index + pd.Timedelta(days=3650)
        sigs = {"A": pd.Series(1.0, index=a.index), "B": pd.Series(1.0, index=b.index)}
        with pytest.raises(ValueError, match="share no dates"):
            BacktestEngine().run_multi(sigs, {"A": a, "B": b})


class TestRunMultiGuards:
    """run_multi's signals pass the alignment check before any reindex, and long_only
    stops a negative weight."""

    @staticmethod
    def _book():
        a, b = make_ohlcv(120, seed=1), make_ohlcv(120, seed=2)
        sig = SMACrossoverSignal(5, 20).generate(a)
        return {"A": a, "B": b}, sig

    def test_a_reset_index_signal_is_refused(self):
        data, sig = self._book()
        with pytest.raises(ValueError, match="not dates"):
            BacktestEngine().run_multi({"A": sig.reset_index(drop=True), "B": sig}, data)

    def test_a_tz_mismatched_signal_is_refused(self):
        data, sig = self._book()
        with pytest.raises(ValueError, match="timezone-aware"):
            BacktestEngine().run_multi({"A": sig.tz_localize("America/New_York"), "B": sig}, data)

    def test_a_signal_covering_under_half_the_bars_is_refused(self):
        data, sig = self._book()
        with pytest.raises(ValueError, match="matches only"):
            BacktestEngine().run_multi({"A": sig.iloc[:30], "B": sig}, data)

    def test_a_signal_on_a_wider_calendar_is_still_fine(self):
        """Matched by date: a signal covering more than the common bars is cut to them."""
        data, sig = self._book()
        narrow = {"A": data["A"], "B": data["B"].iloc[10:]}
        result = BacktestEngine().run_multi({"A": sig, "B": sig}, narrow)
        assert len(result.equity_curve) == 110 and len(result.trades) > 0

    def test_long_only_refuses_a_negative_weight_that_would_go_short(self):
        data, sig = self._book()
        with pytest.raises(ValueError, match="long_only"):
            BacktestEngine(long_only=True).run_multi({"A": sig, "B": sig}, data, weights={"A": 1.0, "B": -1.0})

    def test_long_only_with_positive_weights_stays_long(self):
        data, sig = self._book()
        result = BacktestEngine(long_only=True).run_multi({"A": sig, "B": sig}, data, weights={"A": 0.6, "B": 0.4})
        assert (result.positions.to_numpy() >= 0).all() and len(result.trades) > 0


class TestMetrics:
    def _make_returns(self, n: int = 252, annual_ret: float = 0.30) -> pd.Series:
        """Generate returns with approximate annual return."""
        daily = (1 + annual_ret) ** (1 / 252) - 1
        # Small noise relative to drift so sign of mean is deterministic
        returns = pd.Series(daily + np.random.default_rng(42).normal(0, 0.003, n))
        equity = 100_000 * (1 + returns).cumprod()
        return returns, equity

    def test_sharpe_positive_for_positive_returns(self):
        returns, equity = self._make_returns(annual_ret=0.15)
        m = compute_metrics(equity, returns)
        assert m["sharpe"] > 0

    def test_cagr_reasonable(self):
        returns, equity = self._make_returns(annual_ret=0.10)
        m = compute_metrics(equity, returns)
        # With smaller drift the realised CAGR can drift further from the
        # target — assert a wider, still-meaningful band.
        assert 0.02 < m["cagr"] < 0.30

    def test_max_drawdown_nonpositive(self):
        returns, equity = self._make_returns()
        m = compute_metrics(equity, returns)
        assert m["max_drawdown"] <= 0

    def test_empty_returns(self):
        m = compute_metrics(pd.Series(dtype=float), pd.Series(dtype=float))
        assert m == {}

    def test_annualized_return(self):
        # 252 days of 0.1% daily ≈ 28% annualized
        returns = pd.Series([0.001] * 252)
        annual = annualized_return(returns)
        assert 0.20 < annual < 0.35

    def test_rolling_sharpe_length(self):
        returns = pd.Series(np.random.normal(0.001, 0.01, 252))
        rs = rolling_sharpe(returns, window=63)
        assert len(rs) == 252


class TestRunMulti:
    """Multi-asset backtests: the trade log is sized by each asset's weight in the
    combined book, so that it sums to the weighted equity curve.
    """

    INITIAL = 100_000.0
    SYMBOLS = ["A", "B", "C", "D"]

    def _book(self, n=600, seed0=700):
        data = {
            s: TestReconciliationInvariant._gbm(n, seed=seed0 + i)
            for i, s in enumerate(self.SYMBOLS)
        }
        signals = {
            s: SMACrossoverSignal(fast=20, slow=50).generate(data[s]) for s in self.SYMBOLS
        }
        return signals, data

    def _engine(self):
        return BacktestEngine(initial_capital=self.INITIAL, commission=0.001, slippage=0.0005)

    @pytest.mark.parametrize(
        "weights",
        [
            None,
            {"A": 1, "B": 1, "C": 2, "D": 4},
            {"A": 0.7, "B": 0.1, "C": 0.1, "D": 0.1},
            {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25},
        ],
        ids=["equal-default", "integer-weights", "lopsided", "explicit-equal"],
    )
    def test_n5_reconciles_for_every_weighting(self, weights):
        """The reconciliation on the combined book: Σ pnl_net_abs == combined equity P&L."""
        signals, data = self._book()
        result = self._engine().run_multi(signals, data, weights)
        assert not result.trades.empty, "fixture must trade"

        equity_pnl = float(result.equity_curve.iloc[-1]) - self.INITIAL
        trade_pnl = float(result.trades["pnl_net_abs"].sum())
        gap = abs(trade_pnl - equity_pnl)
        assert gap < 0.001 * self.INITIAL, (
            f"reconciliation violated: gap ${gap:,.2f} = {gap / self.INITIAL * 100:.4f}% of initial"
        )
        assert gap < 1e-6, f"not exact: gap ${gap:.3e}"

    @pytest.mark.parametrize("n_symbols", [1, 2, 4])
    def test_n5_reconciles_for_any_book_size(self, n_symbols):
        signals, data = self._book()
        keep = self.SYMBOLS[:n_symbols]
        result = self._engine().run_multi(
            {k: signals[k] for k in keep}, {k: data[k] for k in keep}
        )
        gap = abs(
            float(result.trades["pnl_net_abs"].sum())
            - (float(result.equity_curve.iloc[-1]) - self.INITIAL)
        )
        assert gap < 1e-6, f"gap ${gap:.3e}"

    def test_n5_reconciles_across_ragged_calendars(self):
        """Assets are aligned to the common calendar before accruing.

        Bars outside the intersection never reach the combined curve, so
        accruing trades over them could not reconcile.
        """
        signals, data = self._book()
        data["B"] = data["B"].iloc[10:]
        data["C"] = data["C"].iloc[:-15]
        result = self._engine().run_multi(signals, data)
        expected_bars = len(data["B"].index.intersection(data["C"].index))
        assert len(result.equity_curve) == expected_bars
        gap = abs(
            float(result.trades["pnl_net_abs"].sum())
            - (float(result.equity_curve.iloc[-1]) - self.INITIAL)
        )
        assert gap < 1e-6, f"gap ${gap:.3e}"

    def test_weight_actually_reaches_the_trade_log(self):
        """Dollar P&L must scale with weight, or the trade log cannot reconcile."""
        signals, data = self._book()
        eng = self._engine()
        light = eng.run_multi(signals, data, {"A": 0.1, "B": 0.3, "C": 0.3, "D": 0.3})
        heavy = eng.run_multi(signals, data, {"A": 0.7, "B": 0.1, "C": 0.1, "D": 0.1})

        a_light = light.trades[light.trades["symbol"] == "A"]
        a_heavy = heavy.trades[heavy.trades["symbol"] == "A"]
        assert len(a_light) == len(a_heavy) > 0
        # Same trades, same percentage returns -- only the dollar sizing moves.
        assert np.allclose(a_light["pnl_pct"].to_numpy(), a_heavy["pnl_pct"].to_numpy())
        assert abs(a_heavy["pnl_abs"].abs().sum()) > 3 * abs(a_light["pnl_abs"].abs().sum())
        # Commission scales with the weighted base too.
        assert a_heavy["commission_abs"].sum() > 3 * a_light["commission_abs"].sum()

    def test_every_symbol_is_labelled_and_present(self):
        signals, data = self._book()
        result = self._engine().run_multi(signals, data)
        assert "symbol" in result.trades.columns
        assert set(result.trades["symbol"].unique()) == set(self.SYMBOLS)

    def test_weights_are_normalised_not_levered(self):
        """{1,1,1,1} must behave as {0.25,...}, i.e. a 1x gross book."""
        signals, data = self._book()
        eng = self._engine()
        unit = eng.run_multi(signals, data, {s: 1 for s in self.SYMBOLS})
        frac = eng.run_multi(signals, data, {s: 0.25 for s in self.SYMBOLS})
        assert np.allclose(unit.equity_curve.to_numpy(), frac.equity_curve.to_numpy())
        assert np.allclose(
            unit.trades["pnl_net_abs"].to_numpy(), frac.trades["pnl_net_abs"].to_numpy()
        )

    def test_non_positive_weights_rejected(self):
        signals, data = self._book()
        with pytest.raises(ValueError, match="must not all be zero"):
            self._engine().run_multi(signals, data, {s: 0 for s in self.SYMBOLS})

    def test_single_asset_book_matches_run(self):
        """A 1-asset, weight-1 book is the single-asset backtest."""
        signals, data = self._book()
        eng = self._engine()
        multi = eng.run_multi({"A": signals["A"]}, {"A": data["A"]})
        single = eng.run(signals["A"], data["A"])
        assert np.allclose(multi.equity_curve.to_numpy(), single.equity_curve.to_numpy())
        assert np.allclose(
            multi.trades["pnl_net_abs"].to_numpy(), single.trades["pnl_net_abs"].to_numpy()
        )


class TestFillTiming:
    """fill="next_close" fills a signal one bar after its own close.

    The default market-on-close fill is optimistic for a signal that needs the
    close to be computed; the stricter fill is an engine option.
    """

    def test_next_close_is_one_more_bar_of_delay(self):
        """next_close equals the default fill with the signal shifted one bar by hand."""
        df = make_ohlcv(150, seed=12)
        raw = SMACrossoverSignal(fast=5, slow=20).generate(df)
        late = BacktestEngine(commission=0.0, slippage=0.0, fill="next_close").run(raw, df)
        by_hand = BacktestEngine(commission=0.0, slippage=0.0).run(raw.shift(1).fillna(0), df)
        pd.testing.assert_series_equal(late.equity_curve, by_hand.equity_curve)
        held = raw.shift(2).fillna(0)
        assert (np.sign(late.positions["shares"]) == np.sign(held)).all()

    @pytest.mark.parametrize("fill", ["close", "next_close"])
    def test_both_fills_reconcile_exactly(self, fill):
        df = make_ohlcv(400, seed=13)
        result = BacktestEngine(100_000, 0.001, 0.0005, fill=fill).run(MACDSignal(), df)
        gap = abs(float(result.trades["pnl_net_abs"].sum()) - (float(result.equity_curve.iloc[-1]) - 100_000))
        assert gap < 1e-6

    def test_ruin_stops_trading_under_a_delayed_fill_too(self):
        df = TestRuin._squeeze()
        result = BacktestEngine(fill="next_close").run(pd.Series(-1.0, index=df.index), df)
        assert result.ruined and (result.equity_curve >= 0).all()
        assert (result.positions["signal"].loc[result.ruin_date:].iloc[1:] == 0).all()

    def test_unknown_fill_is_rejected(self):
        with pytest.raises(ValueError, match="fill"):
            BacktestEngine(fill="next_open")


class TestFillsNeverPrecedeTheSignal:
    """No argument can fill a signal at a price from before the data it read.

    ``price_col="Open"`` would fill a signal computed at bar t's close at bar t's
    open: on a random walk, sign(close change) would read Sharpe 17.03.
    """

    @staticmethod
    def walk(seed: int, n: int = 500) -> pd.DataFrame:
        """A driftless random walk whose Open is the previous Close, as on the demo feed."""
        rng = np.random.default_rng(seed)
        close = 100 * np.exp(np.cumsum(rng.normal(0.0, 0.012, n)))
        open_ = np.r_[close[0], close[:-1]]
        spread = np.abs(rng.normal(0.0, 0.006, n))
        high = np.maximum(open_, close) * (1 + spread)
        low = np.minimum(open_, close) * (1 - spread)
        return pd.DataFrame({
            "Open": open_, "High": high, "Low": low, "Close": close,
            "Adj Close": close * 0.9, "VWAP": (open_ + high + low + close) / 4.0,
            "Volume": np.full(n, 1e6),
        }, index=pd.date_range("2020-01-01", periods=n, freq="B"))

    SIGNALS = {
        "sign(close change)": lambda d: np.sign(d["Close"].pct_change()).fillna(0.0),
        "sign(close - open)": lambda d: np.sign(d["Close"] - d["Open"]),
    }

    @pytest.mark.parametrize("column", ["Open", "High", "Low", "VWAP", "open", "Typical"])
    @pytest.mark.parametrize("fill", sorted(FILLS))
    def test_a_price_from_before_the_close_is_refused(self, column, fill):
        df = self.walk(0)
        df["Typical"] = df["VWAP"]
        with pytest.raises(ValueError, match="not a close"):
            BacktestEngine(fill=fill).run(self.SIGNALS["sign(close change)"](df), df, price_col=column)

    @pytest.mark.parametrize("column", ["Close", "Adj Close", "adj_close", "AdjClose", "close"])
    def test_a_close_is_accepted_and_earns_the_same_returns(self, column):
        df = self.walk(1)
        df["adj_close"] = df["AdjClose"] = df["Adj Close"]
        df["close"] = df["Close"]
        signal = pd.Series(1.0, index=df.index)
        got = BacktestEngine(commission=0.0, slippage=0.0).run(signal, df, price_col=column)
        want = BacktestEngine(commission=0.0, slippage=0.0).run(signal, df)
        assert got.metrics["total_return"] == pytest.approx(want.metrics["total_return"], abs=1e-12)

    @pytest.mark.parametrize("name", sorted(SIGNALS))
    @pytest.mark.parametrize("fill", sorted(FILLS))
    @pytest.mark.parametrize("long_only", [False, True])
    def test_a_random_walk_has_no_edge_under_any_accepted_argument(self, name, fill, long_only):
        """Every column the engine accepts, frictionless, averaged over 20 walks.

        Each run's Sharpe has a standard error near 0.7 (two years of bars), so
        the mean of 20 sits within about 0.2 of the hurdle's drag. The leak this
        guards against read 17.
        """
        engine = BacktestEngine(commission=0.0, slippage=0.0, fill=fill, long_only=long_only)
        accepted: dict[str, list[float]] = {}
        for seed in range(20):
            df = self.walk(seed)
            for column in df.columns.drop("Volume"):
                try:
                    result = engine.run(self.SIGNALS[name](df), df, price_col=column)
                except ValueError:
                    continue
                accepted.setdefault(column, []).append(result.metrics["sharpe"])
        assert {"Close", "Adj Close"} <= set(accepted)
        for column, sharpes in accepted.items():
            assert abs(float(np.mean(sharpes))) < 0.75, (column, float(np.mean(sharpes)))

    def test_run_multi_fills_at_the_close_whatever_the_frame_holds(self):
        """run_multi takes no price column: changing Open/High/Low changes nothing."""
        df = self.walk(2)
        signal = self.SIGNALS["sign(close change)"](df)
        moved = df.assign(Open=df["Open"] * 1.5, High=df["High"] * 2.0, Low=df["Low"] * 0.5)
        one = BacktestEngine().run_multi({"A": signal}, {"A": df})
        two = BacktestEngine().run_multi({"A": signal}, {"A": moved})
        pd.testing.assert_series_equal(one.equity_curve, two.equity_curve)
        assert abs(one.metrics["sharpe"]) < 5.0


class TestWhatTheWeightsBound:
    """Weights bound a leg at entry, not the book."""

    @staticmethod
    def book(result: BacktestResult, prices: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """(gross exposure, cash), each from the shares actually held."""
        value = result.positions * prices
        return value.abs().sum(axis=1) / result.equity_curve, result.equity_curve - value.sum(axis=1)

    def test_a_leg_sized_on_a_grown_book_passes_1x_and_borrows_for_free(self):
        index = pd.date_range("2024-01-01", periods=6, freq="B")
        prices = pd.DataFrame({"A": [100.0, 100, 400, 400, 400, 400], "B": [100.0] * 6}, index=index)
        frames = {s: pd.DataFrame({"Close": prices[s]}) for s in prices}
        signals = {"A": pd.Series(1.0, index=index), "B": pd.Series([0.0, 0, 0, 1, 1, 1], index=index)}
        result = BacktestEngine(commission=0.0, slippage=0.0).run_multi(signals, frames)
        gross, cash = self.book(result, prices)
        assert gross.iloc[-1] == pytest.approx(1.30)
        assert cash.iloc[-1] == pytest.approx(-75_000.0)
        # Flat prices, cash -$75,000 for two bars: no interest is charged.
        assert result.equity_curve.iloc[-3:].tolist() == pytest.approx([250_000.0] * 3)

    def test_the_readme_figures_reproduce_and_the_docstring_makes_no_such_claim(self):
        from pathlib import Path

        from alphaforge.api.app import synthetic_ohlcv

        assert "at most" not in BacktestEngine.run_multi.__doc__
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
        data = {s: synthetic_ohlcv(s, "2015-01-01", "2024-01-01") for s in symbols}
        signals = {s: SMACrossoverSignal(20, 50).generate(data[s]) for s in symbols}
        result = BacktestEngine(long_only=True).run_multi(
            signals, data, weights={"AAPL": 0.3, "MSFT": 0.3, "GOOGL": 0.2, "AMZN": 0.2}
        )
        gross, cash = self.book(result, pd.DataFrame({s: data[s]["Close"] for s in symbols}))
        readme = " ".join((Path(__file__).parent.parent / "README.md").read_text().split())
        assert "Negative cash is financed free" in readme
        assert f"reaches {gross.max():.2f}x gross" in readme
        assert f"negative on {int((cash < 0).sum())} of {len(cash):,} bars" in readme
        assert f"as low as −${-cash.min():,.0f}" in readme


class TestARunWithNoFiguresReadsNA:
    """Fewer than two bars have no metrics; 0.00% would read as a result."""

    @pytest.mark.parametrize("bars", [0, 1])
    def test_repr_says_na(self, bars):
        df = make_ohlcv(5).iloc[:bars]
        text = repr(BacktestEngine().run(pd.Series(1.0, index=df.index), df))
        assert text == "BacktestResult(cagr=n/a, sharpe=n/a, max_drawdown=n/a, trades=0)"

    def test_an_undefined_sharpe_says_na(self):
        """A run that never trades has zero volatility, so no Sharpe, but it has a CAGR."""
        df = make_ohlcv(50)
        text = repr(BacktestEngine().run(pd.Series(0.0, index=df.index), df))
        assert text == "BacktestResult(cagr=0.00%, sharpe=n/a, max_drawdown=0.00%, trades=0)"

    def test_a_real_run_keeps_its_figures(self):
        df = make_ohlcv()
        result = BacktestEngine().run(make_signals(len(df)), df)
        assert f"cagr={result.metrics['cagr']:.2%}, sharpe={result.metrics['sharpe']:.2f}," in repr(result)


def test_no_margin_call_a_short_runs_on_at_1999x_gross():
    """The figure in the README's "What the model leaves out"."""
    from pathlib import Path

    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    closes = [100.0, 199.9, 199.9]
    df = pd.DataFrame(
        {"Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": 1e6}, index=dates
    )
    signal = pd.Series(-1.0, index=dates)
    result = BacktestEngine(initial_capital=100_000.0, commission=0.0, slippage=0.0).run(signal, df)
    assert not result.ruined
    equity = float(result.equity_curve.iloc[1])
    assert equity == pytest.approx(100.0)
    assert abs(result.positions["shares"].iloc[1]) * 199.9 / equity == pytest.approx(1999.0)
    readme = " ".join((Path(__file__).resolve().parent.parent / "README.md").read_text().split())
    assert "at 1,999x gross, not ruined" in readme

