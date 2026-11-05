"""
Ensemble signal: weighted combination of multiple Signal instances.

Supports equal weighting, manual weights, and Sharpe-weighted combination.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.signals.base import Signal


class EnsembleSignal(Signal):
    """
    Weighted ensemble of signals.

    Aggregates multiple :class:`Signal` instances by computing a weighted
    sum of their outputs. The result is discretised to {-1, 0, +1} using
    a configurable threshold.

    Parameters
    ----------
    signals : list[Signal]
        List of signal generators to combine.
    weights : list[float] | None
        Per-signal weights. If None, uses equal weighting.
    threshold : float
        Minimum absolute weighted score to generate a non-zero signal.
        Default is 0.3, meaning 30% of max weight must agree.
    vote_mode : str
        "weighted_sum" (default) or "majority_vote".
    """

    name = "Ensemble"

    def __init__(
        self,
        signals: list[Signal],
        weights: list[float] | None = None,
        threshold: float = 0.3,
        vote_mode: str = "weighted_sum",
    ) -> None:
        if not signals:
            raise ValueError("EnsembleSignal requires at least one signal.")
        if weights is not None and len(weights) != len(signals):
            raise ValueError(
                f"weights length ({len(weights)}) must match signals length ({len(signals)})."
            )
        self.signals = signals
        self.weights = weights if weights is not None else [1.0 / len(signals)] * len(signals)
        self.threshold = threshold
        self.vote_mode = vote_mode

    def generate(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate combined signal.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame (shared input for all component signals).

        Returns
        -------
        pd.Series
            Integer signals in {-1, 0, +1}.
        """
        all_signals: list[pd.Series] = []
        for sig in self.signals:
            s = sig.generate(df).reindex(df.index).fillna(0)
            all_signals.append(s)

        if self.vote_mode == "majority_vote":
            return self._majority_vote(all_signals, df.index)
        else:
            return self._weighted_sum(all_signals, df.index)

    def _weighted_sum(self, all_signals: list[pd.Series], index: pd.Index) -> pd.Series:
        total_weight = sum(abs(w) for w in self.weights)
        combined = pd.Series(0.0, index=index)
        for s, w in zip(all_signals, self.weights):
            combined += s.fillna(0) * w

        # Normalise to [-1, 1]
        if total_weight > 0:
            combined /= total_weight

        result = np.where(
            combined > self.threshold, 1,
            np.where(combined < -self.threshold, -1, 0),
        )
        return pd.Series(result, index=index, dtype=int, name=self.name)

    def _majority_vote(self, all_signals: list[pd.Series], index: pd.Index) -> pd.Series:
        weighted_votes = pd.Series(0.0, index=index)
        for s, w in zip(all_signals, self.weights):
            weighted_votes += s.fillna(0) * w

        result = np.where(weighted_votes > 0, 1, np.where(weighted_votes < 0, -1, 0))
        return pd.Series(result, index=index, dtype=int, name=self.name)

    def add_signal(self, signal: Signal, weight: float = 1.0) -> "EnsembleSignal":
        """Return a new EnsembleSignal with *signal* appended."""
        new_signals = self.signals + [signal]
        new_weights = self.weights + [weight]
        total = sum(new_weights)
        normalised = [w / total for w in new_weights]
        return EnsembleSignal(
            signals=new_signals,
            weights=normalised,
            threshold=self.threshold,
            vote_mode=self.vote_mode,
        )

    @classmethod
    def from_sharpe_weights(
        cls,
        signals: list[Signal],
        sharpe_scores: list[float],
        threshold: float = 0.3,
    ) -> "EnsembleSignal":
        """
        Build an ensemble weighted by Sharpe ratios.

        Parameters
        ----------
        signals : list[Signal]
        sharpe_scores : list[float]
            In-sample Sharpe ratio for each signal. Negative Sharpe → weight 0.
        threshold : float

        Returns
        -------
        EnsembleSignal
        """
        clipped = [max(s, 0) for s in sharpe_scores]
        total = sum(clipped)
        if total == 0:
            weights = [1.0 / len(signals)] * len(signals)
        else:
            weights = [s / total for s in clipped]
        return cls(signals=signals, weights=weights, threshold=threshold)
