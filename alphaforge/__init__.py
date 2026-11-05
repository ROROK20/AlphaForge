"""
AlphaForge is a Python backtesting framework that keeps itself honest, with ML
signals, portfolio optimization and a web dashboard.
"""

__version__ = "0.1.0"
__author__ = "Rohan Khiani"
__license__ = "MIT"

from alphaforge.data.fetcher import DataFetcher
from alphaforge.data.preprocessor import add_technical_features, normalize, create_lagged_features
from alphaforge.backtest.engine import BacktestEngine
from alphaforge.backtest.metrics import compute_metrics

__all__ = [
    "DataFetcher",
    "add_technical_features",
    "normalize",
    "create_lagged_features",
    "BacktestEngine",
    "compute_metrics",
]
