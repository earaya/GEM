"""
GEM (Global Equities Momentum) - Improved Implementation

A modern Python implementation of Gary Antonacci's Global Equities Momentum strategy
with enhanced code quality, error handling, and extensibility.

This package provides:
- Data fetching from multiple financial sources
- Momentum strategy calculation and backtesting
- Comprehensive analysis and visualization tools
- CLI interface for easy usage
"""

__version__ = "2.0.0"
__author__ = "Improved GEM Implementation"

from gem.strategy.momentum import GlobalEquitiesMomentum
from gem.data.fetchers import DataFetcher
from gem.analysis.metrics import PerformanceMetrics

__all__ = [
    "GlobalEquitiesMomentum",
    "DataFetcher", 
    "PerformanceMetrics",
]