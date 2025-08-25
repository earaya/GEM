"""
Data module for GEM strategy.

This module handles data fetching, processing, and caching for financial data
used in the Global Equities Momentum strategy.
"""

from gem.data.fetchers import DataFetcher
from gem.data.processors import DataProcessor

__all__ = ["DataFetcher", "DataProcessor"]