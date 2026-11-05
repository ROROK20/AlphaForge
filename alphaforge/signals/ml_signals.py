"""
Machine-learning-based signals.

XGBoostSignal  — gradient-boosted direction predictor.
LSTMSignal     — PyTorch LSTM sequence model for price direction.

Out-of-sample contract (both classes)
-------------------------------------
``generate(df)`` never scores a bar the model was trained on.

* If the signal has NOT been fitted (and loads no cache), ``generate`` runs a
  **purged, expanding walk-forward** over *df*: the series is cut into
  consecutive blocks of ``retrain_every`` bars; the model for the block that
  starts at bar *s* is trained only on rows whose ``target_horizon``-bar
  forward label is fully known before *s* (row *i* qualifies iff
  ``i + target_horizon <= s - 1``: the label horizon is purged at every
  boundary). Bars before the first block -- until ``min_train_size`` labelled
  rows exist -- are flat (0). ``walk_forward_folds_`` records every fold.
* If the signal HAS been fitted (``fit``) or loaded from its cache, bars at or
  before the last bar of the training frame (``trained_through_``) are flat;
  only later bars are scored.

Fitting on *df* and predicting on that same *df*, with 5-bar forward labels, scores
the very bars the model just learned. On a pure random walk (the API's synthetic
feed) that reads a Sharpe above 4 for a model whose honest out-of-sample Sharpe is
negative; the README prints the figures.

xgboost and torch in one process
--------------------------------
On macOS each brings its own OpenMP runtime: xgboost's wheel links Homebrew's
libomp, torch ships another. At their default thread counts they cannot share a
process. Measured 2026-11-04 on macOS arm64 (xgboost 3.2.0, torch 2.13.0), a fit
of each with nothing else set: XGBoost then LSTM deadlocks in the LSTM's first
tensor, and LSTM then XGBoost, or importing torch first, ends in a segfault. So
where both are installed on macOS, these classes fit and score on ONE thread
each (``xgboost.config_context(nthread=1)``; ``n_jobs=1`` alone still segfaults,
because xgboost loads labels on its global thread count) and
``torch.set_num_threads(1)``), which starts no worker pool. Both are restored on
return, nothing is limited anywhere else, and nothing touches the environment.
Your own xgboost or torch calls in that process are not covered: for those,
start Python with ``OMP_NUM_THREADS=1``.
"""

from __future__ import annotations

import contextlib
import functools
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import pandas as pd

from alphaforge.data.preprocessor import add_technical_features, create_lagged_features, create_target
from alphaforge.signals.base import Signal

logger = logging.getLogger(__name__)

# Default cache location for models a caller explicitly asks to persist.
# NOT created at import time: importing a library should not write to the
# user's home, and an instance that never opts into the cache never needs it.
# ``_save`` creates whatever directory it is actually about to write to.
_MODEL_DIR = Path.home() / ".alphaforge" / "models"

_PRICE_COLS = ["Open", "High", "Low", "Close", "Volume"]
_NON_FEATURES = {*_PRICE_COLS, "target"}
# Key under which the out-of-sample boundary is persisted with a cached model.
# A cache file without it cannot prove which bars it was trained on, so it is
# ignored rather than trusted.
_TRAINED_THROUGH_KEY = "alphaforge_trained_through"


@functools.lru_cache(maxsize=1)
def _openmp_runtimes_clash() -> bool:
    """True where xgboost and torch cannot share a process at their default
    thread counts (see the module docstring). Checks what is installed, not what
    is loaded: by the time both are loaded, a first fit has already started the
    worker pool that the second one trips over."""
    return sys.platform == "darwin" and all(
        importlib.util.find_spec(name) is not None for name in ("xgboost", "torch")
    )


@contextlib.contextmanager
def _xgboost_threads() -> Iterator[None]:
    """Run an xgboost block on one thread where the runtimes clash; restore on exit."""
    if not _openmp_runtimes_clash():
        yield
        return
    import xgboost

    with xgboost.config_context(nthread=1):
        yield


@contextlib.contextmanager
def _torch_threads() -> Iterator[None]:
    """Run a torch block on one thread where the runtimes clash; restore on exit."""
    if not _openmp_runtimes_clash():
        yield
        return
    import torch

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _walk_forward_starts(
    usable: np.ndarray,
    horizon: int,
    min_train_size: int,
    retrain_every: int,
) -> list[int]:
    """Bar positions at which each out-of-sample block of a purged walk-forward starts.

    ``usable[i]`` says whether row *i* has complete features. The model for a
    block starting at bar *s* may train only on usable rows *i* with
    ``i + horizon <= s - 1`` -- their forward label closes before *s*. The first
    block starts at the earliest *s* with at least ``min_train_size`` such rows;
    later blocks start every ``retrain_every`` bars.
    """
    n = len(usable)
    counts = np.cumsum(usable.astype(int))  # counts[j] = usable rows in [0, j]
    for s in range(horizon + 1, n):
        if counts[s - 1 - horizon] >= min_train_size:
            return list(range(s, n, retrain_every))
    return []


def _check_walk_forward_params(target_horizon: int, min_train_size: int, retrain_every: int) -> None:
    if target_horizon < 1:
        raise ValueError(f"target_horizon must be >= 1, got {target_horizon}")
    if min_train_size < 1:
        raise ValueError(f"min_train_size must be >= 1, got {min_train_size}")
    if retrain_every < 1:
        raise ValueError(f"retrain_every must be >= 1, got {retrain_every}")


def _mask_in_sample(
    signals: pd.Series, trained_through: Optional[pd.Timestamp], owner: str
) -> pd.Series:
    """Zero every bar at or before the last bar the model's training saw."""
    if trained_through is None:
        logger.warning(
            "%s: model has no recorded training window; cannot verify the scored "
            "bars are out of sample.", owner,
        )
        return signals
    in_sample = signals.index <= trained_through
    if in_sample.any():
        signals = signals.copy()
        signals[in_sample] = 0
        if in_sample.all():
            logger.warning(
                "%s: every bar is at or before the model's training window end "
                "(%s); nothing can be scored out of sample.", owner, trained_through,
            )
    return signals


# ---------------------------------------------------------------------------
# XGBoost Signal
# ---------------------------------------------------------------------------


class XGBoostSignal(Signal):
    """
    XGBoost-based directional signal.

    Trains on lagged technical features to predict whether the price
    will be higher or lower *target_horizon* periods forward. See the module
    docstring for the out-of-sample contract of :meth:`generate`.

    Parameters
    ----------
    target_horizon : int
        Prediction horizon in periods (default 5). Also the purge width at
        every walk-forward boundary.
    n_lags : int
        Number of lag periods for feature creation (default 10).
    n_estimators : int
        Number of boosting rounds (default 300).
    max_depth : int
        Maximum tree depth (default 4).
    threshold : float
        Predicted up-probability above which the signal is +1 (default 0.55).
    short_threshold : float | None
        Up-probability below which the signal is -1. Default None means the
        symmetric ``1 - threshold``.
    min_train_size : int
        Labelled rows required before the first walk-forward block is scored
        (default 252, one trading year). Earlier bars are flat.
    retrain_every : int
        Walk-forward block length in bars: the model is retrained on all
        purged history at the start of every block (default 63, a quarter).
    model_name : str | None
        File stem for saving/loading the model. **Default None = ephemeral:**
        the instance never reads or writes the on-disk cache, so results do not
        depend on what happens to be on disk. Pass a name to opt into the
        shared cache deliberately.
    model_dir : str | Path | None
        Directory for the model cache (default ``~/.alphaforge/models``).
        Passing one also opts the instance into persistence, under the default
        stem if ``model_name`` is omitted.
    """

    name = "XGBoost"

    def __init__(
        self,
        target_horizon: int = 5,
        n_lags: int = 10,
        n_estimators: int = 300,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        threshold: float = 0.55,
        short_threshold: float | None = None,
        min_train_size: int = 252,
        retrain_every: int = 63,
        model_name: str | None = None,
        model_dir: str | Path | None = None,
    ) -> None:
        _check_walk_forward_params(target_horizon, min_train_size, retrain_every)
        self.target_horizon = target_horizon
        self.n_lags = n_lags
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.short_threshold = 1 - threshold if short_threshold is None else short_threshold
        if not self.short_threshold <= self.threshold:
            raise ValueError("short_threshold must not exceed threshold")
        self.min_train_size = min_train_size
        self.retrain_every = retrain_every
        self.model_name = model_name
        self.model_dir = Path(model_dir) if model_dir is not None else None
        # Persistence is opt-in: naming the model or choosing a directory is
        # the caller asking for a cache. Anything else stays in memory.
        self._cache_enabled = model_name is not None or model_dir is not None
        self._model_stem = model_name or "xgb_signal"
        self._model = None
        self._feature_cols: list[str] = []
        self.trained_through_: Optional[pd.Timestamp] = None
        self.walk_forward_folds_: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []

    # ------------------------------------------------------------------

    def _features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Every feature here is causal (rolling / EWM / shift over past bars
        # only), so a row's value never depends on later rows.
        return create_lagged_features(add_technical_features(df), n_lags=self.n_lags)

    def _train(self, df: pd.DataFrame) -> tuple[Any, list[str]]:
        """Fit a fresh model on *df*; touches no instance state."""
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError("xgboost is required: pip install xgboost")

        frame = create_target(self._features(df), horizon=self.target_horizon, method="direction")
        frame = frame.dropna()  # drops the last target_horizon rows: unlabelled
        if frame.empty:
            raise ValueError("XGBoostSignal: no labelled rows to train on.")
        feature_cols = [c for c in frame.columns if c not in _NON_FEATURES]

        X = frame[feature_cols].values
        y = ((frame["target"] + 1) // 2).astype(int).values  # map {-1,0,+1} → {0,0,1}
        y = np.clip(y, 0, 1)

        model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            eval_metric="logloss",
            tree_method="hist",
            verbosity=0,
        )
        with _xgboost_threads():
            model.fit(X, y)
        logger.info("XGBoostSignal: trained on %d samples, %d features.", len(X), len(feature_cols))
        return model, feature_cols

    def _predict(self, model: Any, feature_cols: list[str], feat: pd.DataFrame) -> np.ndarray:
        """Integer signals for the rows of *feat*; rows with incomplete features are flat."""
        X = feat[feature_cols]
        complete = X.notna().all(axis=1).to_numpy()
        out = np.zeros(len(X), dtype=int)
        if complete.any():
            with _xgboost_threads():
                proba = model.predict_proba(X.to_numpy()[complete])[:, 1]
            out[complete] = np.where(
                proba > self.threshold, 1, np.where(proba < self.short_threshold, -1, 0)
            )
        return out

    def fit(self, df: pd.DataFrame) -> "XGBoostSignal":
        """
        Train the XGBoost model on *df*.

        After this, :meth:`generate` scores only bars strictly after
        ``df.index[-1]`` (recorded as ``trained_through_``): every earlier bar
        either was a training row or supplied a training label.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame. Technical features are computed automatically.

        Returns
        -------
        self
        """
        self._model, self._feature_cols = self._train(df)
        self.trained_through_ = pd.Timestamp(df.index[-1])
        self._save()
        return self

    def generate(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate out-of-sample signals for *df* (see the module docstring).

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame.

        Returns
        -------
        pd.Series
            Integer signals in {-1, 0, +1}. 0 on every bar no model could
            score out of sample.
        """
        if self._model is None and not self._load():
            return self._generate_walk_forward(df)

        feat = self._features(df)
        missing = [c for c in self._feature_cols if c not in feat.columns]
        if missing:
            logger.debug("XGBoost: %d feature(s) missing at inference; padding with 0.", len(missing))
            for c in missing:
                feat[c] = 0.0
        raw = self._predict(self._model, self._feature_cols, feat)
        signals = pd.Series(raw, index=df.index, name=self.name, dtype=int)
        return _mask_in_sample(signals, self.trained_through_, "XGBoostSignal")

    def _generate_walk_forward(self, df: pd.DataFrame) -> pd.Series:
        feat = self._features(df)
        feature_cols = [c for c in feat.columns if c not in _NON_FEATURES]
        usable = feat[feature_cols + _PRICE_COLS].notna().all(axis=1).to_numpy()
        starts = _walk_forward_starts(
            usable, self.target_horizon, self.min_train_size, self.retrain_every
        )
        out = np.zeros(len(df), dtype=int)
        folds = []
        for j, s in enumerate(starts):
            e = starts[j + 1] if j + 1 < len(starts) else len(df)
            # Training on the prefix df[:s] IS the purge: create_target leaves
            # the prefix's last target_horizon rows unlabelled, so no label
            # reaches bar s or beyond.
            model, cols = self._train(df.iloc[:s])
            out[s:e] = self._predict(model, cols, feat.iloc[s:e])
            folds.append((df.index[s - 1], df.index[s], df.index[e - 1]))
        if not starts:
            logger.warning(
                "XGBoostSignal: %d bars is not enough history for one out-of-sample "
                "fold (min_train_size=%d); all signals are flat.", len(df), self.min_train_size,
            )
        self.walk_forward_folds_ = folds
        return pd.Series(out, index=df.index, name=self.name, dtype=int)

    def feature_importance(self) -> pd.Series:
        """Return feature importances of the fitted model as a sorted Series."""
        if self._model is None:
            raise RuntimeError(
                "Model not trained. Call .fit() first (an unfitted generate() runs a "
                "walk-forward and keeps no single model)."
            )
        scores = self._model.feature_importances_
        return pd.Series(scores, index=self._feature_cols).sort_values(ascending=False)

    def _cache_dir(self) -> Path:
        return self.model_dir if self.model_dir is not None else _MODEL_DIR

    def _model_path(self) -> Path:
        return self._cache_dir() / f"{self._model_stem}.json"

    def _features_path(self) -> Path:
        return self._cache_dir() / f"{self._model_stem}_features.txt"

    def _save(self) -> None:
        if self._model is None or not self._cache_enabled:
            return
        path = self._model_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.trained_through_ is not None:
            self._model.get_booster().set_attr(
                **{_TRAINED_THROUGH_KEY: self.trained_through_.isoformat()}
            )
        self._model.save_model(str(path))
        # Save feature list
        self._features_path().write_text("\n".join(self._feature_cols))

    def _load(self) -> bool:
        if not self._cache_enabled:
            return False
        path = self._model_path()
        feat_path = self._features_path()
        if path.exists() and feat_path.exists():
            try:
                from xgboost import XGBClassifier
                model = XGBClassifier()
                model.load_model(str(path))
                through = model.get_booster().attr(_TRAINED_THROUGH_KEY)
                if through is None:
                    logger.warning(
                        "XGBoostSignal: ignoring cached model %s -- it records no "
                        "training window, so its out-of-sample bars are unknown.", path,
                    )
                    return False
                self._model = model
                self._feature_cols = feat_path.read_text().strip().splitlines()
                self.trained_through_ = pd.Timestamp(through)
                return True
            except Exception as exc:
                logger.warning("Failed to load XGBoost model: %s", exc)
        return False


# ---------------------------------------------------------------------------
# LSTM Signal
# ---------------------------------------------------------------------------


class LSTMSignal(Signal):
    """
    PyTorch LSTM signal for sequential price prediction.

    Architecture: 2-layer LSTM (hidden_size=64) → Linear(64 → 1) → sigmoid.
    Trained to predict whether next *target_horizon* return is positive. The
    prediction for bar *t* uses the ``seq_len`` feature rows before *t*. Same
    out-of-sample contract as :class:`XGBoostSignal` (see the module
    docstring); the walk-forward retrains the network once per block, so
    ``epochs * number_of_blocks`` is the training cost of an unfitted
    ``generate``.

    Parameters
    ----------
    seq_len : int
        Sequence length (lookback window) in periods (default 20).
    hidden_size : int
        LSTM hidden state dimension (default 64).
    num_layers : int
        Number of stacked LSTM layers (default 2).
    target_horizon : int
        Forward-return horizon for labels (default 5); the walk-forward purge.
    epochs : int
        Training epochs (default 50).
    lr : float
        Adam learning rate (default 1e-3).
    threshold : float
        Probability threshold for directional signal (default 0.55).
    min_train_size : int
        Labelled rows required before the first block is scored (default 252).
    retrain_every : int
        Walk-forward block length in bars (default 63).
    random_state : int | None
        Seed for weight initialisation, dropout and batch shuffling. None
        (default) leaves torch's global RNG in charge; an int makes training
        reproducible without disturbing the caller's global RNG state.
    model_name : str | None
        Stem for saving model weights. **Default None = ephemeral** — same
        contract as :class:`XGBoostSignal`: an unnamed instance never touches
        the on-disk cache, so it cannot load a previous run's weights.
    model_dir : str | Path | None
        Directory for the weight cache (default ``~/.alphaforge/models``).
        Passing one also opts the instance into persistence.
    """

    name = "LSTM"

    def __init__(
        self,
        seq_len: int = 20,
        hidden_size: int = 64,
        num_layers: int = 2,
        target_horizon: int = 5,
        epochs: int = 50,
        lr: float = 1e-3,
        threshold: float = 0.55,
        min_train_size: int = 252,
        retrain_every: int = 63,
        random_state: int | None = None,
        model_name: str | None = None,
        model_dir: str | Path | None = None,
    ) -> None:
        _check_walk_forward_params(target_horizon, min_train_size, retrain_every)
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.target_horizon = target_horizon
        self.epochs = epochs
        self.lr = lr
        self.threshold = threshold
        self.min_train_size = min_train_size
        self.retrain_every = retrain_every
        self.random_state = random_state
        self.model_name = model_name
        self.model_dir = Path(model_dir) if model_dir is not None else None
        self._cache_enabled = model_name is not None or model_dir is not None
        self._model_stem = model_name or "lstm_signal"
        self._net = None
        self._feature_cols: list[str] = []
        self._mu: Optional[np.ndarray] = None
        self._sigma: Optional[np.ndarray] = None
        self.trained_through_: Optional[pd.Timestamp] = None
        self.walk_forward_folds_: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []

    def _build_net(self, n_features: int):
        try:
            import torch.nn as nn
        except ImportError:
            raise ImportError("PyTorch is required: pip install torch")

        class _Net(nn.Module):
            def __init__(self, n_feat, hidden, layers):
                super().__init__()
                self.lstm = nn.LSTM(n_feat, hidden, num_layers=layers, batch_first=True, dropout=0.2)
                self.fc = nn.Linear(hidden, 1)
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                out, _ = self.lstm(x)
                last = out[:, -1, :]
                return self.sigmoid(self.fc(last)).squeeze(-1)

        return _Net(n_features, self.hidden_size, self.num_layers)

    def _train(self, df: pd.DataFrame) -> tuple[Any, list[str], np.ndarray, np.ndarray]:
        """Train a fresh network on *df*; touches no instance state."""
        try:
            import torch  # noqa: F401
        except ImportError:
            raise ImportError("PyTorch is required: pip install torch")
        with _torch_threads():
            return self._train_network(df)

    def _train_network(self, df: pd.DataFrame) -> tuple[Any, list[str], np.ndarray, np.ndarray]:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        frame = create_target(add_technical_features(df), horizon=self.target_horizon, method="direction")
        frame = frame.dropna()  # drops the last target_horizon rows: unlabelled
        feat_cols = [c for c in frame.columns if c not in _NON_FEATURES]

        X_raw = frame[feat_cols].values.astype(np.float32)
        y_raw = ((frame["target"] + 1) / 2).values.astype(np.float32)  # {-1,1} → {0,1}
        y_raw = np.clip(y_raw, 0, 1)

        # Normalise with TRAINING statistics only.
        mu = X_raw.mean(axis=0)
        sigma = X_raw.std(axis=0) + 1e-8
        X_norm = (X_raw - mu) / sigma

        # Sequence ending just before row i predicts row i's forward label.
        seqs, labels = [], []
        for i in range(self.seq_len, len(X_norm)):
            seqs.append(X_norm[i - self.seq_len : i])
            labels.append(y_raw[i])

        if len(seqs) < 10:
            raise ValueError("Insufficient data for LSTM training (need seq_len + 10 rows min).")

        X_t = torch.tensor(np.array(seqs), dtype=torch.float32)
        y_t = torch.tensor(np.array(labels), dtype=torch.float32)

        seeded = self.random_state is not None
        rng_scope = torch.random.fork_rng(devices=[]) if seeded else contextlib.nullcontext()
        with rng_scope:
            if seeded:
                torch.manual_seed(self.random_state)
            loader = DataLoader(TensorDataset(X_t, y_t), batch_size=32, shuffle=True)
            net = self._build_net(len(feat_cols))
            optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
            criterion = nn.BCELoss()

            net.train()
            for epoch in range(self.epochs):
                total_loss = 0.0
                for xb, yb in loader:
                    optimizer.zero_grad()
                    loss = criterion(net(xb), yb)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                if (epoch + 1) % 10 == 0:
                    logger.debug(
                        "LSTM epoch %d/%d — loss: %.4f", epoch + 1, self.epochs, total_loss / len(loader)
                    )
        net.eval()
        logger.info("LSTMSignal: trained %d epochs on %d sequences.", self.epochs, len(seqs))
        return net, feat_cols, mu, sigma

    def _predict(
        self,
        net: Any,
        feat_cols: list[str],
        mu: np.ndarray,
        sigma: np.ndarray,
        feat: pd.DataFrame,
        start: int = 0,
        stop: Optional[int] = None,
    ) -> np.ndarray:
        """Signals for bar positions [start, stop) of *feat* (earlier rows feed sequences only)."""
        import torch

        stop = len(feat) if stop is None else stop
        complete = feat[feat_cols].notna().all(axis=1).to_numpy()
        rows = np.flatnonzero(complete)  # positions of complete feature rows
        X_norm = ((feat[feat_cols].to_numpy(dtype=np.float32)[rows] - mu) / sigma).astype(np.float32)

        targets, seqs = [], []
        for j, pos in enumerate(rows):
            if start <= pos < stop and j >= self.seq_len:
                targets.append(pos)
                seqs.append(X_norm[j - self.seq_len : j])
        out = np.zeros(stop - start, dtype=int)
        if not seqs:
            return out
        with _torch_threads(), torch.no_grad():
            prob = net(torch.tensor(np.array(seqs), dtype=torch.float32)).numpy().reshape(-1)
        sig = np.where(prob > self.threshold, 1, np.where(prob < (1 - self.threshold), -1, 0))
        out[np.array(targets) - start] = sig
        return out

    def fit(self, df: pd.DataFrame) -> "LSTMSignal":
        """
        Train the LSTM on *df*. Afterwards :meth:`generate` scores only bars
        strictly after ``df.index[-1]`` (``trained_through_``).

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame.

        Returns
        -------
        self
        """
        self._net, self._feature_cols, self._mu, self._sigma = self._train(df)
        self.trained_through_ = pd.Timestamp(df.index[-1])
        self._save()
        return self

    def generate(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate out-of-sample signals for *df* (see the module docstring).

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame.

        Returns
        -------
        pd.Series
            Integer signals in {-1, 0, +1}.
        """
        try:
            import torch  # noqa: F401
        except ImportError:
            raise ImportError("PyTorch is required: pip install torch")

        if self._net is None and not self._load():
            return self._generate_walk_forward(df)

        feat = add_technical_features(df)
        for col in self._feature_cols:
            if col not in feat.columns:
                feat[col] = 0.0
        raw = self._predict(self._net, self._feature_cols, self._mu, self._sigma, feat)
        signals = pd.Series(raw, index=df.index, dtype=int, name=self.name)
        return _mask_in_sample(signals, self.trained_through_, "LSTMSignal")

    def _generate_walk_forward(self, df: pd.DataFrame) -> pd.Series:
        feat = add_technical_features(df)
        feature_cols = [c for c in feat.columns if c not in _NON_FEATURES]
        usable = feat[feature_cols + _PRICE_COLS].notna().all(axis=1).to_numpy()
        starts = _walk_forward_starts(
            usable, self.target_horizon, self.min_train_size, self.retrain_every
        )
        out = np.zeros(len(df), dtype=int)
        folds = []
        for j, s in enumerate(starts):
            e = starts[j + 1] if j + 1 < len(starts) else len(df)
            net, cols, mu, sigma = self._train(df.iloc[:s])  # prefix = purge
            out[s:e] = self._predict(net, cols, mu, sigma, feat, start=s, stop=e)
            folds.append((df.index[s - 1], df.index[s], df.index[e - 1]))
        if not starts:
            logger.warning(
                "LSTMSignal: %d bars is not enough history for one out-of-sample "
                "fold (min_train_size=%d); all signals are flat.", len(df), self.min_train_size,
            )
        self.walk_forward_folds_ = folds
        return pd.Series(out, index=df.index, dtype=int, name=self.name)

    def _model_path(self) -> Path:
        base = self.model_dir if self.model_dir is not None else _MODEL_DIR
        return base / f"{self._model_stem}.pt"

    def _save(self) -> None:
        if self._net is not None and self._cache_enabled:
            try:
                import torch
                # Plain lists, not numpy arrays: torch.load's safe
                # weights_only=True mode (torch's default since 2.6) refuses
                # numpy objects.
                state = {
                    "state_dict": self._net.state_dict(),
                    "feature_cols": self._feature_cols,
                    "mu": np.asarray(self._mu, dtype=float).tolist(),
                    "sigma": np.asarray(self._sigma, dtype=float).tolist(),
                    _TRAINED_THROUGH_KEY: (
                        self.trained_through_.isoformat() if self.trained_through_ is not None else None
                    ),
                    "params": {
                        "hidden_size": self.hidden_size,
                        "num_layers": self.num_layers,
                        "seq_len": self.seq_len,
                    },
                }
                path = self._model_path()
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(state, str(path))
            except Exception as exc:
                logger.warning("Failed to save LSTM model: %s", exc)

    def _load(self) -> bool:
        if not self._cache_enabled:
            return False
        path = self._model_path()
        if not path.exists():
            return False
        try:
            import torch
            # weights_only=True: never unpickle arbitrary objects from disk.
            state = torch.load(str(path), map_location="cpu", weights_only=True)
            through = state.get(_TRAINED_THROUGH_KEY)
            if through is None:
                logger.warning(
                    "LSTMSignal: ignoring cached weights %s -- they record no "
                    "training window, so their out-of-sample bars are unknown.", path,
                )
                return False
            self._feature_cols = state["feature_cols"]
            self._mu = np.asarray(state["mu"], dtype=np.float32)
            self._sigma = np.asarray(state["sigma"], dtype=np.float32)
            self._net = self._build_net(len(self._feature_cols))
            self._net.load_state_dict(state["state_dict"])
            self._net.eval()
            self.trained_through_ = pd.Timestamp(through)
            return True
        except Exception as exc:
            logger.warning("Failed to load LSTM model: %s", exc)
            return False
