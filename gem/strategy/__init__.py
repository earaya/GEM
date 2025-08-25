"""
Strategy module for GEM implementation.

This module contains the core momentum strategy logic and backtesting functionality.
"""

from gem.strategy.momentum import GlobalEquitiesMomentum
from gem.strategy.backtest import Backtester

__all__ = ["GlobalEquitiesMomentum", "Backtester"]