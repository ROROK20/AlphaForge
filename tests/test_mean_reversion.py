"""Tests for the pairs (mean-reversion) strategy: causality (appending future bars
must not change past signals) and the entry/exit state machine, on behaviour.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from strategies.mean_reversion import MeanReversionStrategy


def cointegrated_pair(n: int = 1000, seed: int = 0) -> tuple[pd.Series, pd.Series]:
    """log A = 0.8 log B + 1 + AR(1) noise (phi 0.9): a mean-reverting spread."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    b = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.9 * noise[i - 1] + rng.normal(0, 0.01)
    a = np.exp(0.8 * np.log(b) + 1.0 + noise)
    return pd.Series(a, idx), pd.Series(b, idx)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_appending_future_bars_changes_no_past_signal(seed):
    """Signals for bars 0-499 must not depend on bars 500-999: a full-sample hedge
    ratio changes 14 of the first 500 signals on seed 0."""
    a, b = cointegrated_pair(seed=seed)
    strat = MeanReversionStrategy()
    past, _ = strat.generate_signals(a.iloc[:500], b.iloc[:500])
    full, _ = strat.generate_signals(a, b)
    differences = int((past != full.iloc[:500]).sum())
    assert differences == 0, f"{differences} of the first 500 signals changed"
    assert (full != 0).sum() > 20, "fixture must actually trade"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_positions_exit_back_to_flat(seed):
    """A mean-reverting spread must be traded in round trips: open, revert,
    close to flat, not held on every bar inside the exit band."""
    a, b = cointegrated_pair(seed=seed)
    signals, _ = MeanReversionStrategy().generate_signals(a, b)
    exits_to_flat = int(((signals.shift(1).fillna(0) != 0) & (signals == 0)).sum())
    entries = int(((signals.shift(1).fillna(0) == 0) & (signals != 0)).sum())
    assert exits_to_flat >= 5, exits_to_flat
    assert entries - exits_to_flat in (0, 1)  # at most the final trade is still open
    assert (signals == 0).mean() > 0.3, "flat most of the time between trades"


def test_state_machine_obeys_its_thresholds():
    a, b = cointegrated_pair(seed=4)
    strat = MeanReversionStrategy(entry_z=2.0, exit_z=0.5)
    z = strat.spread_zscore(a, b)["zscore"]
    pos, _ = strat.generate_signals(a, b)
    prev = pos.shift(1).fillna(0)
    # Held long only while still below -exit_z; held short only above +exit_z.
    assert (z[pos == 1] < -0.5).all()
    assert (z[pos == -1] > 0.5).all()
    # Every new position is opened beyond the entry band, on the right side.
    opened_long = (pos == 1) & (prev != 1)
    opened_short = (pos == -1) & (prev != -1)
    assert opened_long.any() and opened_short.any()
    assert (z[opened_long] < -2.0).all()
    assert (z[opened_short] > 2.0).all()
    # Undefined z (warm-up) means flat.
    assert (pos[z.isna()] == 0).all()


def test_hedge_ratio_is_estimated_from_trailing_bars_only():
    a, b = cointegrated_pair(seed=5)
    strat = MeanReversionStrategy(hedge_lookback=120)
    frame = strat.spread_zscore(a, b)
    assert frame["hedge_ratio"].iloc[:119].isna().all()  # needs 120 bars
    t = 400
    window = slice(t - 119, t + 1)
    la, lb = np.log(a.iloc[window]), np.log(b.iloc[window])
    ols = np.cov(la, lb)[0, 1] / np.var(lb, ddof=1)
    assert frame["hedge_ratio"].iloc[t] == pytest.approx(ols, rel=1e-9)
    # ...and it tracks the true 0.8 on this fixture.
    assert frame["hedge_ratio"].iloc[200:].median() == pytest.approx(0.8, abs=0.15)


def test_backtest_is_causal_and_reconciles_exactly():
    """The equity curve over bars 0-499 must not change when bars 500-999
    exist, and the trade log must reconcile with it to float precision."""
    a, b = cointegrated_pair(seed=0)
    strat = MeanReversionStrategy()
    short = strat.backtest_pair(a.iloc[:500], b.iloc[:500])
    full = strat.backtest_pair(a, b)
    pd.testing.assert_series_equal(short.equity_curve, full.equity_curve.iloc[:500], check_freq=False)
    for result in (short, full):
        gap = abs(
            float(result.trades["pnl_net_abs"].sum())
            - (float(result.equity_curve.iloc[-1]) - strat.initial_capital)
        )
        assert gap < 1e-6
    assert np.isfinite(full.metrics["hedge_ratio"])


def test_run_fetches_both_legs_and_backtests_the_pair():
    a, b = cointegrated_pair(seed=1)
    frames = {s: pd.DataFrame({"Close": p}) for s, p in (("A", a), ("B", b))}
    with patch("strategies.mean_reversion.DataFetcher") as fetcher_cls:
        fetcher_cls.return_value.fetch_ohlcv.side_effect = lambda sym, *_: frames[sym]
        result = MeanReversionStrategy().run("A", "B", "2018-01-01")
    expected = MeanReversionStrategy().backtest_pair(a, b)
    pd.testing.assert_series_equal(result.equity_curve, expected.equity_curve)


def test_exit_must_sit_inside_entry():
    with pytest.raises(ValueError):
        MeanReversionStrategy(entry_z=1.0, exit_z=1.5)


def test_gross_exposure_is_reported():
    """The spread is traded at (1 + |hedge|)x leg A's notional; that leverage
    is reported next to the (already disclosed) leg-B cost gap."""
    a, b = cointegrated_pair(seed=0)
    strat = MeanReversionStrategy()
    result = strat.backtest_pair(a, b)
    _, hedge = strat.generate_signals(a, b)
    held = result.positions["signal"] != 0
    expected = (1 + hedge.shift(1).abs())[held]
    assert result.metrics["gross_exposure_mean"] == pytest.approx(expected.mean())
    assert result.metrics["gross_exposure_max"] == pytest.approx(expected.max())
    assert result.metrics["gross_exposure_mean"] > 1.5  # hedge ~0.8 on this fixture


def test_each_bar_earns_the_hedge_known_a_bar_earlier():
    """Bar t's spread return is ret_A(t) - hedge(t-1) x ret_B(t).

    hedge(t) is fitted on bar t's own prices, so using it for bar t's return
    would let the spread see the move it is being paid for. The prefix check
    above cannot catch that: hedge(t) depends only on bars up to t.
    """
    a, b = cointegrated_pair(seed=0)
    strat = MeanReversionStrategy()
    _, hedge = strat.generate_signals(a, b)
    spread = strat.backtest_pair(a, b).positions["price"].pct_change().iloc[1:]
    ret_a, ret_b = a.pct_change().iloc[1:], b.pct_change().iloc[1:]
    pd.testing.assert_series_equal(
        spread, ret_a - hedge.shift(1).fillna(0.0).iloc[1:] * ret_b, check_names=False, check_freq=False
    )
    same_bar = ret_a - hedge.fillna(0.0).iloc[1:] * ret_b
    assert not np.allclose(spread, same_bar, rtol=0.0, atol=1e-9), "fixture cannot tell the two hedges apart"
