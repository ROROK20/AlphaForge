"""Data layer: fetching, preprocessing, and universe management."""

from alphaforge.data.fetcher import DataFetcher
from alphaforge.data.preprocessor import add_technical_features, normalize, create_lagged_features
from alphaforge.data.universe import Universe

__all__ = [
    "DataFetcher",
    "add_technical_features",
    "normalize",
    "create_lagged_features",
    "Universe",
]
