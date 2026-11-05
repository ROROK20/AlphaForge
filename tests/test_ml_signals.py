"""Tests for ML signal generators (XGBoost, LSTM).

These exercise the full ``generate()`` path — including the walk-forward's training
and the ``create_target`` forward-target construction — on a plain OHLCV DataFrame,
exactly as a user follows the README.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alphaforge.signals.ml_signals import XGBoostSignal, LSTMSignal


def make_ohlcv(n: int = 400, seed: int = 7) -> pd.DataFrame:
    """Synthetic OHLCV with enough rows to train an ML model."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n)))
    high = close * (1 + rng.uniform(0, 0.02, n))
    low = close * (1 - rng.uniform(0, 0.02, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    dates = pd.date_range("2021-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


class TestXGBoostSignal:
    def test_generate_runs_end_to_end(self, tmp_path):
        pytest.importorskip("xgboost")
        df = make_ohlcv()
        # model_dir is a real, honoured kwarg: this keeps the test off ~/.alphaforge.
        sig = XGBoostSignal(model_dir=str(tmp_path))
        signals = sig.generate(df)
        assert len(signals) == len(df)

    def test_model_dir_is_honoured(self, tmp_path):
        """The cache lands where the caller asked, not in ~/.alphaforge."""
        pytest.importorskip("xgboost")
        sig = XGBoostSignal(model_dir=str(tmp_path), n_estimators=10)
        sig.fit(make_ohlcv())
        assert (tmp_path / "xgb_signal.json").exists()
        assert (tmp_path / "xgb_signal_features.txt").exists()

    def test_default_instance_cannot_load_a_previous_runs_model(
        self, tmp_path, monkeypatch
    ):
        """Reproducibility pin: a default-constructed signal trains on ITS data.

        A fresh instance's ``generate()`` must not load whatever a previous run
        trained under a shared stem in ~/.alphaforge/models: the result would
        depend on machine state.
        """
        pytest.importorskip("xgboost")
        import alphaforge.signals.ml_signals as ml

        monkeypatch.setattr(ml, "_MODEL_DIR", tmp_path)
        df = make_ohlcv()

        # A caller who opts in populates the shared cache under a shared stem.
        ml.XGBoostSignal(model_name="xgb_signal", n_estimators=10).fit(df)
        assert (tmp_path / "xgb_signal.json").exists()

        # A default-constructed instance must ignore it entirely.
        fresh = ml.XGBoostSignal(n_estimators=10)
        assert fresh._load() is False
        fresh.fit(df)
        assert fresh._model is not None
        # ...and must not write into the shared cache either.
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "xgb_signal.json",
            "xgb_signal_features.txt",
        ]

    def test_import_does_not_create_a_home_directory(self):
        """Importing a library must not write to the user's home.

        Asserted on the PARSED module body, not by grepping raw source: a raw
        grep for "_MODEL_DIR.mkdir" goes red for the wrong reason the moment a
        comment or docstring legitimately names the removed call — and this
        module's own comment discusses exactly that. ``ast`` cannot see comments,
        and the positive control below proves the scan is not vacuous.
        """
        import ast

        import alphaforge.signals.ml_signals as ml

        def mkdir_call_lines(body):
            found = []
            for node in body:
                for sub in ast.walk(node):
                    if (
                        isinstance(sub, ast.Call)
                        and isinstance(sub.func, ast.Attribute)
                        and sub.func.attr == "mkdir"
                    ):
                        found.append(sub.lineno)
            return found

        tree = ast.parse(Path(ml.__file__).read_text())
        # Module level = statements NOT nested inside a def/class, i.e. exactly
        # what executes on import.
        top_level = [
            n
            for n in tree.body
            if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        assert mkdir_call_lines(top_level) == [], (
            "a directory is created at import time; move it into the code that "
            "is actually about to write"
        )

        # POSITIVE CONTROL: the scan must be able to see a real mkdir, or the
        # assertion above would pass vacuously for any module.
        control = ast.parse("from pathlib import Path\nPath('x').mkdir(parents=True)\n")
        assert mkdir_call_lines(control.body), "scan is broken — cannot detect a real mkdir"

    def test_save_creates_its_own_directory(self, tmp_path):
        """Opt-in persistence creates its own directory.

        `_save` creates whatever directory it is about to write to, including a
        path several levels deep that does not exist.
        """
        pytest.importorskip("xgboost")
        nested = tmp_path / "does" / "not" / "exist" / "yet"
        assert not nested.exists()
        sig = XGBoostSignal(model_dir=str(nested), n_estimators=10)
        sig.fit(make_ohlcv())
        assert (nested / "xgb_signal.json").exists()
        assert (nested / "xgb_signal_features.txt").exists()

    def test_signal_values_in_range(self):
        pytest.importorskip("xgboost")
        df = make_ohlcv()
        signals = XGBoostSignal().generate(df)
        assert set(np.unique(signals.dropna())).issubset({-1, 0, 1})

    def test_no_nan_in_core_region(self):
        pytest.importorskip("xgboost")
        df = make_ohlcv()
        signals = XGBoostSignal().generate(df)
        # The warm-up region may be 0/NaN, but the bulk must be populated.
        assert signals.notna().sum() > len(df) * 0.5


class TestLSTMSignal:
    def test_generate_runs_end_to_end(self):
        pytest.importorskip("torch")
        df = make_ohlcv()
        signals = LSTMSignal(seq_len=20, epochs=2).generate(df)
        assert len(signals) == len(df)

    def test_signal_values_in_range(self):
        pytest.importorskip("torch")
        df = make_ohlcv()
        signals = LSTMSignal(seq_len=20, epochs=2).generate(df)
        assert set(np.unique(signals.dropna())).issubset({-1, 0, 1})


# ---------------------------------------------------------------------------
# Out-of-sample contract
# ---------------------------------------------------------------------------
#
# These tests pin "never score a bar the model was trained on": fit(df) and predict
# on the same df, with 5-bar forward labels, scores the bars it just learned.


def random_walk(n: int, seed: int, start: str = "2010-01-01") -> pd.DataFrame:
    """Zero-drift GBM: no strategy has a real edge on it."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0, 0.012, n)))
    return pd.DataFrame(
        {
            "Open": np.r_[close[0], close[:-1]],
            "High": close * (1 + np.abs(rng.normal(0, 0.006, n))),
            "Low": close * (1 - np.abs(rng.normal(0, 0.006, n))),
            "Close": close,
            "Volume": rng.uniform(5e5, 5e6, n),
        },
        index=pd.date_range(start, periods=n, freq="B"),
    )


def with_rewritten_future(df: pd.DataFrame, cut: int, seed: int) -> pd.DataFrame:
    """Same bars before ``cut``; an unrelated random path from ``cut`` on."""
    other = random_walk(len(df) - cut, seed)
    scale = df["Close"].iloc[cut - 1] / other["Close"].iloc[0]
    out = df.copy()
    for col in ("Open", "High", "Low", "Close"):
        out.iloc[cut:, out.columns.get_loc(col)] = other[col].to_numpy() * scale
    out.iloc[cut:, out.columns.get_loc("Volume")] = other["Volume"].to_numpy()
    return out


class TestXGBoostOutOfSample:
    def test_future_data_cannot_change_past_signals(self):
        """The defining property of no look-ahead, checked exactly: rewriting
        every bar from ``cut`` on must leave every signal before ``cut``
        unchanged. A fit-on-everything path fails this, because the model
        scoring early bars has learned from the future."""
        pytest.importorskip("xgboost")
        cut = 520
        df = random_walk(700, seed=31)
        a = XGBoostSignal(n_estimators=30).generate(df)
        b = XGBoostSignal(n_estimators=30).generate(with_rewritten_future(df, cut, seed=99))
        assert (a.iloc[:cut] != 0).sum() > 30, "fixture must actually trade before the cut"
        pd.testing.assert_series_equal(a.iloc[:cut], b.iloc[:cut])

    def test_ensemble_inherits_the_contract(self):
        """EnsembleSignal is how ML reaches mixed strategies (architecture.md)."""
        pytest.importorskip("xgboost")
        from alphaforge.signals.ensemble import EnsembleSignal
        from alphaforge.signals.technical import SMACrossoverSignal

        cut = 520
        df = random_walk(700, seed=32)

        def ens():
            return EnsembleSignal([SMACrossoverSignal(10, 30), XGBoostSignal(n_estimators=30)])

        a = ens().generate(df)
        b = ens().generate(with_rewritten_future(df, cut, seed=98))
        pd.testing.assert_series_equal(a.iloc[:cut], b.iloc[:cut])

    def test_walk_forward_folds_are_purged_and_warmup_is_flat(self):
        pytest.importorskip("xgboost")
        df = random_walk(700, seed=33)
        sig = XGBoostSignal(n_estimators=20, min_train_size=252, retrain_every=63)
        signals = sig.generate(df)
        folds = sig.walk_forward_folds_
        assert len(folds) >= 5
        pos = {d: i for i, d in enumerate(df.index)}
        for trained_through, oos_start, oos_end in folds:
            # Training data ends on the bar before the block: the prefix's last
            # target_horizon rows carry no label, so no label reaches oos_start.
            assert pos[trained_through] == pos[oos_start] - 1
            assert oos_start <= oos_end
        # Consecutive, non-overlapping blocks that run to the last bar.
        for (_, _, end), (_, nxt, _) in zip(folds, folds[1:]):
            assert pos[nxt] == pos[end] + 1
        assert folds[-1][2] == df.index[-1]
        first = pos[folds[0][1]]
        # min_train_size labelled rows (features complete, label purged) exist
        # only after the 49-bar SMA_50 warm-up + 252 rows + the 5-bar purge.
        assert first >= 252 + 5
        assert (signals.iloc[:first] == 0).all(), "bars before the first fold must be flat"

    def test_explicit_fit_never_scores_its_training_bars(self):
        pytest.importorskip("xgboost")
        df = random_walk(700, seed=34)
        sig = XGBoostSignal(n_estimators=20).fit(df.iloc[:400])
        signals = sig.generate(df)
        assert (signals.iloc[:400] == 0).all(), "scored bars the model was fitted on"
        assert (signals.iloc[400:] != 0).sum() > 30
        assert sig.trained_through_ == df.index[399]

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_random_walk_has_no_edge(self, seed):
        """No spurious edge on data that has none.

        Measured 2026-11-01 on these exact fixtures (1500 bars, 50 trees,
        depth 3, zero costs): scoring in-sample reads gross Sharpe
        +3.16 / +2.95 / +3.47; the purged walk-forward at its default cadence
        scores -1.69 / +0.41 / -0.44. The 1.0 bound sits well clear of both.
        (Seed 1's -1.69 is a tail draw, not a bias: over seeds 1-12 the
        timing component of the return averaged -2.5%/yr, SE 3.0%.)
        """
        pytest.importorskip("xgboost")
        from alphaforge.backtest.engine import BacktestEngine

        df = random_walk(1500, seed)
        signals = XGBoostSignal(n_estimators=50, max_depth=3).generate(df)
        sharpe = BacktestEngine(commission=0, slippage=0).run(signals, df).metrics["sharpe"]
        assert sharpe < 1.0, f"seed {seed}: Sharpe {sharpe:.2f} on a random walk"

    def test_cache_without_a_training_window_is_ignored(self, tmp_path):
        """A cache file that cannot say which bars it learned from must not be
        trusted to score bars."""
        xgboost = pytest.importorskip("xgboost")
        X = np.random.default_rng(0).normal(size=(100, 3))
        legacy = xgboost.XGBClassifier(n_estimators=2).fit(X, (X[:, 0] > 0).astype(int))
        legacy.save_model(str(tmp_path / "xgb_signal.json"))
        (tmp_path / "xgb_signal_features.txt").write_text("a\nb\nc")

        sig = XGBoostSignal(model_dir=str(tmp_path), n_estimators=20)
        assert sig._load() is False
        sig.generate(random_walk(450, seed=35))
        assert sig.walk_forward_folds_, "fell back to the walk-forward"

    def test_cached_model_keeps_its_out_of_sample_boundary(self, tmp_path):
        pytest.importorskip("xgboost")
        df = random_walk(600, seed=36)
        XGBoostSignal(model_dir=str(tmp_path), n_estimators=20).fit(df.iloc[:400])
        fresh = XGBoostSignal(model_dir=str(tmp_path))
        signals = fresh.generate(df)
        assert (signals.iloc[:400] == 0).all(), "cached model scored its training bars"
        assert (signals.iloc[400:] != 0).any()
        assert fresh.trained_through_ == df.index[399]


class TestLSTMOutOfSample:
    # threshold=0.5: a 2-epoch net sits near p=0.5, so at the default 0.55
    # almost every bar would be flat and the comparisons below would be vacuous.
    KW = dict(seq_len=10, hidden_size=8, epochs=2, retrain_every=100, random_state=0,
              threshold=0.5)
    # No seed: at threshold=0.5 nearly every scored bar is +/-1 with or without one.
    KW_UNSEEDED = dict(seq_len=10, hidden_size=8, epochs=2, threshold=0.5)

    def test_future_data_cannot_change_past_signals(self):
        pytest.importorskip("torch")
        cut = 420
        df = random_walk(460, seed=41)
        a = LSTMSignal(**self.KW).generate(df)
        b = LSTMSignal(**self.KW).generate(with_rewritten_future(df, cut, seed=97))
        assert (a.iloc[:cut] != 0).sum() > 30, "fixture must actually trade before the cut"
        pd.testing.assert_series_equal(a.iloc[:cut], b.iloc[:cut])

    def test_each_block_trains_only_on_bars_before_it(self, monkeypatch):
        """Read from the frames the network is trained on, because its signals cannot
        show this: a two-epoch net sits near p=0.5 and often points one way across a
        whole block, so training on the block itself can leave every sign unchanged.
        """
        pytest.importorskip("torch")
        df = random_walk(460, seed=41)
        last_trained_on = []
        train = LSTMSignal._train

        def recording(signal, frame):
            last_trained_on.append(frame.index[-1])
            return train(signal, frame)

        monkeypatch.setattr(LSTMSignal, "_train", recording)
        sig = LSTMSignal(**self.KW)
        sig.generate(df)
        folds = sig.walk_forward_folds_
        assert len(folds) >= 2
        assert len(last_trained_on) == len(folds)
        for trained_on, (trained_through, oos_start, _) in zip(last_trained_on, folds):
            assert trained_on == trained_through
            assert trained_on < oos_start, "a block trained on its own bars"

    def test_explicit_fit_never_scores_its_training_bars(self):
        pytest.importorskip("torch")
        df = random_walk(460, seed=42)
        sig = LSTMSignal(**self.KW_UNSEEDED).fit(df.iloc[:350])
        signals = sig.generate(df)
        assert (signals.iloc[:350] == 0).all(), "scored bars the network was fitted on"
        assert (signals.iloc[350:] != 0).sum() > 30
        assert sig.trained_through_ == df.index[349]

    def test_cache_round_trips_under_safe_loading(self, tmp_path):
        """torch.load defaults to weights_only=True since torch 2.6, which refuses
        numpy arrays. The cache must actually load, and keep its boundary."""
        pytest.importorskip("torch")
        df = random_walk(460, seed=43)
        LSTMSignal(model_dir=str(tmp_path), **self.KW_UNSEEDED).fit(df.iloc[:350])
        fresh = LSTMSignal(model_dir=str(tmp_path), **self.KW_UNSEEDED)
        assert fresh._load() is True
        assert fresh.trained_through_ == df.index[349]
        assert (fresh.generate(df).iloc[:350] == 0).all()


# The README's exact digits were measured with this xgboost on this platform.
# The trees depend on the xgboost RELEASE (numpy and scipy versions change nothing),
# so CI's Python 3.12 leg, which resolves the newest xgboost, cannot reproduce the
# digits -- only the claim they support.
README_FIGURE_MEASURED_WITH = {"xgboost": "3.2.0", "system": "Darwin", "machine": "arm64"}


def test_readme_random_walk_figure_reproduces_as_written():
    """The README's XGBoost random-walk figure must come from the exact call it
    prints, feed window included.

    Exact digits are checked in the environment the README names; anywhere else
    the version-independent claim is checked (no edge on a random walk), and
    the README must keep naming the version it was measured with."""
    import platform
    import re

    import xgboost

    from alphaforge.api.app import synthetic_ohlcv
    from alphaforge.backtest.engine import BacktestEngine

    readme = (Path(__file__).resolve().parent.parent / "README.md").read_text()
    m = re.search(r"# CAGR (-?[\d.]+)% / Sharpe (-?[\d.]+)\s+\(purged walk-forward", readme)
    assert m, "README must print the walk-forward figure"
    want_cagr, want_sharpe = float(m.group(1)), float(m.group(2))
    assert 'synthetic_ohlcv("AAPL", "2015-01-01", "2024-01-01")' in readme
    assert "XGBoostSignal(n_estimators=200)" in readme

    measured = README_FIGURE_MEASURED_WITH
    assert f"xgboost\n{measured['xgboost']}" in readme or f"xgboost {measured['xgboost']}" in readme, (
        "README must name the xgboost version its figure was measured with"
    )

    df = synthetic_ohlcv("AAPL", "2015-01-01", "2024-01-01")
    got = BacktestEngine().run(XGBoostSignal(n_estimators=200), df).metrics
    assert len(df) == 2348
    same_env = (
        xgboost.__version__ == measured["xgboost"]
        and platform.system() == measured["system"]
        and platform.machine() == measured["machine"]
    )
    if same_env:
        assert f"{got['cagr'] * 100:.2f}" == f"{want_cagr:.2f}"
        assert f"{got['sharpe']:.2f}" == f"{want_sharpe:.2f}"
    # The claim the digits support, on any xgboost / platform: out of sample on
    # a random walk there is no edge (scoring in-sample reads a Sharpe above 4).
    assert got["sharpe"] < 0.5, got["sharpe"]
    assert got["cagr"] < 0.0, got["cagr"]

    # The hero quotes the same digits as the block they come from.
    hero = readme.split('<p align="center">')[0]
    assert f"CAGR {want_cagr:.2f}% / Sharpe {want_sharpe:.2f}" in hero

    # The leak the contract rules out, with the guard switched off by hand.
    leaky = XGBoostSignal(n_estimators=200).fit(df)
    leaky.trained_through_ = df.index[0] - pd.Timedelta(days=1)
    leak = BacktestEngine().run(leaky, df).metrics
    if same_env:
        assert f"**CAGR {leak['cagr'] * 100:.2f}% / Sharpe {leak['sharpe']:.2f}**" in readme
    assert leak["sharpe"] > 2.0, leak["sharpe"]


def test_readme_random_walk_snippet_defines_every_name_it_uses():
    """Pasted as written the snippet must run. The figure test above runs the same
    call; this checks the snippet itself."""
    import ast
    import builtins
    import re

    readme = (Path(__file__).resolve().parent.parent / "README.md").read_text()
    block = next(b for b in re.findall(r"```python\n(.*?)```", readme, re.S) if "synthetic_ohlcv(" in b)
    tree = ast.parse(block)
    defined = set(dir(builtins))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            defined |= {a.asname or a.name for a in node.names}
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            defined.add(node.id)
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    assert used <= defined, sorted(used - defined)


def test_a_later_end_date_on_the_demo_feed_does_not_change_past_signals():
    """The obvious look-ahead falsification: more bars at the end must not change a
    signal on any earlier bar."""
    from alphaforge.api.app import synthetic_ohlcv

    short = synthetic_ohlcv("AAPL", "2015-01-01", "2018-06-01")
    longer = synthetic_ohlcv("AAPL", "2015-01-01", "2019-01-01")
    a = XGBoostSignal(n_estimators=30).generate(short)
    b = XGBoostSignal(n_estimators=30).generate(longer).iloc[: len(short)]
    assert a.ne(0).any()
    assert int((a != b).sum()) == 0


class TestLSTMWindow:
    """A prediction for bar t reads the seq_len feature rows before t: not t itself,
    which would be look-ahead for a same-bar fill, and not one row fewer."""

    SEQ = 7

    def _one_feature_frame(self, n: int = 40) -> pd.DataFrame:
        return pd.DataFrame({"f": np.arange(n, dtype=float)}, index=pd.date_range("2022-01-03", periods=n, freq="B"))

    def test_scoring_reads_exactly_the_rows_before_each_bar(self):
        torch = pytest.importorskip("torch")

        class Recording:
            def __call__(self, x):
                self.seen = x.numpy().copy()
                return torch.full((x.shape[0],), 0.5)

        net, feat = Recording(), self._one_feature_frame()
        signal = LSTMSignal(seq_len=self.SEQ)
        out = signal._predict(net, ["f"], np.zeros(1, dtype=np.float32), np.ones(1, dtype=np.float32), feat, start=20, stop=30)
        assert out.tolist() == [0] * 10
        assert net.seen.shape == (10, self.SEQ, 1)
        for k, bar in enumerate(range(20, 30)):
            assert net.seen[k, :, 0].tolist() == list(range(bar - self.SEQ, bar))

    def test_the_first_scorable_bar_has_a_full_window_behind_it(self):
        torch = pytest.importorskip("torch")

        class Recording:
            def __call__(self, x):
                self.seen = x.numpy().copy()
                return torch.full((x.shape[0],), 0.5)

        net, feat = Recording(), self._one_feature_frame()
        LSTMSignal(seq_len=self.SEQ)._predict(net, ["f"], np.zeros(1, dtype=np.float32), np.ones(1, dtype=np.float32), feat)
        assert net.seen.shape[0] == len(feat) - self.SEQ
        assert net.seen[0, :, 0].tolist() == list(range(self.SEQ))

    def test_training_pairs_each_window_with_the_label_of_the_bar_after_it(self, monkeypatch):
        torch = pytest.importorskip("torch")
        import torch.utils.data as tud

        from alphaforge.data.preprocessor import add_technical_features, create_target

        captured = {}
        real = tud.TensorDataset

        def recording(x, y):
            captured["x"], captured["y"] = x.numpy().copy(), y.numpy().copy()
            return real(x, y)

        monkeypatch.setattr(tud, "TensorDataset", recording)
        df = make_ohlcv(150)
        signal = LSTMSignal(seq_len=self.SEQ, epochs=1, random_state=0)
        _, cols, mu, sigma = signal._train(df)

        frame = create_target(add_technical_features(df), horizon=signal.target_horizon, method="direction").dropna()
        rows = (frame[cols].to_numpy(dtype=np.float32) - mu) / sigma
        labels = np.clip((frame["target"].to_numpy(dtype=np.float32) + 1) / 2, 0, 1)
        assert captured["x"].shape == (len(rows) - self.SEQ, self.SEQ, len(cols))
        assert labels[self.SEQ :].tolist() != labels[self.SEQ - 1 : -1].tolist()
        assert captured["y"].tolist() == labels[self.SEQ :].tolist()
        for k in (0, 1, len(rows) - self.SEQ - 1):
            assert captured["x"][k] == pytest.approx(rows[k : k + self.SEQ])

