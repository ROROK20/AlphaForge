"""Tests for the ML momentum strategy: the full `run()` / `run_walk_forward()` paths
end to end, with the network fetch mocked.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from strategies.ml_momentum import MLMomentumStrategy
from alphaforge.backtest.engine import BacktestResult


def _synth_ohlcv(n: int = 600, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.013, n)))
    dates = pd.date_range("2021-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Open": close * (1 + rng.normal(0, 0.003, n)),
            "High": close * (1 + rng.uniform(0, 0.02, n)),
            "Low": close * (1 - rng.uniform(0, 0.02, n)),
            "Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        },
        index=dates,
    )


@pytest.fixture
def mock_fetch():
    with patch("strategies.ml_momentum.DataFetcher") as MockFetcher:
        MockFetcher.return_value.fetch_ohlcv.return_value = _synth_ohlcv()
        yield MockFetcher


class TestMLMomentumStrategy:
    def test_run_executes_end_to_end(self, mock_fetch):
        pytest.importorskip("xgboost")
        result = MLMomentumStrategy().run("TEST", "2021-01-01", "2023-01-01")
        assert isinstance(result, BacktestResult)
        assert len(result.equity_curve) > 0

    def test_run_walk_forward_executes_end_to_end(self, mock_fetch):
        pytest.importorskip("xgboost")
        result = MLMomentumStrategy().run_walk_forward(
            "TEST", "2021-01-01", "2023-01-01", window_size=200, step_size=100
        )
        assert isinstance(result, BacktestResult)
        assert len(result.equity_curve) > 0

    def test_thresholds_reach_the_signal(self):
        """long_/short_threshold reach the signal, which otherwise runs at its own
        0.55/0.45."""
        strat = MLMomentumStrategy(long_threshold=0.62, short_threshold=0.35)
        assert strat._signal.threshold == 0.62
        assert strat._signal.short_threshold == 0.35

    def test_walk_forward_blocks_are_not_forced_flat_by_feature_warmup(self, mock_fetch):
        """A block featurised on its own 63-bar slice would have NaN rolling features
        (SMA_50, lags) for its first ~26 bars, forcing them flat. With past bars as
        warm-up and a 0.5/0.5 threshold (every scored bar is +/-1), almost
        nothing should be flat."""
        pytest.importorskip("xgboost")
        strat = MLMomentumStrategy(long_threshold=0.5, short_threshold=0.5)
        result = strat.run_walk_forward(
            "TEST", "2021-01-01", "2023-01-01", window_size=200, step_size=63
        )
        flat = float((result.signal_series == 0).mean())
        assert flat < 0.05, f"{flat:.0%} of out-of-sample bars flat"
        # Every block is scored, including the final partial one.
        assert result.metrics["n_walk_forward_windows"] == len(range(200, 551, 63))

    def test_run_scores_every_test_bar_out_of_sample(self, mock_fetch):
        pytest.importorskip("xgboost")
        strat = MLMomentumStrategy(long_threshold=0.5, short_threshold=0.5)
        result = strat.run("TEST", "2021-01-01", "2023-01-01")
        assert float((result.signal_series == 0).mean()) < 0.05
        # The test window starts strictly after the model's training window.
        assert result.equity_curve.index[0] > strat._signal.trained_through_


def test_walk_forward_scores_a_bar_with_the_features_it_trained_on(mock_fetch, monkeypatch):
    """OBV is a running sum from its frame's first bar, so training and scoring
    frames must start together. Every feature must agree on every bar the two
    frames share."""
    pytest.importorskip("xgboost")
    strat = MLMomentumStrategy()
    signal = strat._signal
    features = signal._features
    frames = []

    def recording(df):
        out = features(df)
        frames.append(out)
        return out

    monkeypatch.setattr(signal, "_features", recording)
    strat.run_walk_forward("TEST", "2021-01-01", "2023-01-01", window_size=200, step_size=100)
    assert len(frames) >= 6
    for train, score in zip(frames[::2], frames[1::2]):
        shared = train.index.intersection(score.index)
        assert len(shared) == 200
        pd.testing.assert_frame_equal(train.loc[shared], score.loc[shared])

