"""
Pytest configuration and fixtures for GEM strategy tests.
"""

from typing import Dict, Generator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pytest

from gem.config import GemConfig, AssetConfig, StrategyConfig


@pytest.fixture
def sample_config() -> GemConfig:
    """Create a sample configuration for testing."""
    config = GemConfig()
    config.data.cache_enabled = False  # Disable cache for tests
    config.strategy.lookback_months = 12
    config.strategy.transaction_cost = 0.001
    config.backtest.initial_capital = 10000.0
    return config


@pytest.fixture
def sample_returns_data() -> Dict[str, pd.Series]:
    """Create sample return data for testing."""
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='ME')
    
    # Generate realistic return data
    np.random.seed(42)  # For reproducible tests
    
    returns_data = {}
    
    # US Equity - higher volatility, positive drift
    us_returns = np.random.normal(0.008, 0.04, len(dates))
    returns_data['us_equity'] = pd.Series(us_returns, index=dates)
    
    # International Equity - similar to US but slightly different
    intl_returns = np.random.normal(0.006, 0.035, len(dates))
    returns_data['intl_equity'] = pd.Series(intl_returns, index=dates)
    
    # Bonds - lower volatility, lower returns
    bond_returns = np.random.normal(0.003, 0.015, len(dates))
    returns_data['bonds'] = pd.Series(bond_returns, index=dates)
    
    # Cash - very low volatility, minimal returns
    cash_returns = np.random.normal(0.001, 0.002, len(dates))
    returns_data['cash'] = pd.Series(cash_returns, index=dates)
    
    return returns_data


@pytest.fixture
def sample_price_data() -> Dict[str, pd.DataFrame]:
    """Create sample price data for testing."""
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    price_data = {}
    
    for asset in ['us_equity', 'intl_equity', 'bonds', 'cash']:
        # Generate price series
        returns = np.random.normal(0.0003, 0.015, len(dates))  # Daily returns
        prices = 100 * np.cumprod(1 + returns)  # Starting at 100
        
        # Create OHLCV data
        df = pd.DataFrame(index=dates)
        df['Close'] = prices
        df['Open'] = df['Close'] * (1 + np.random.normal(0, 0.001, len(dates)))
        df['High'] = np.maximum(df['Open'], df['Close']) * (1 + np.abs(np.random.normal(0, 0.005, len(dates))))
        df['Low'] = np.minimum(df['Open'], df['Close']) * (1 - np.abs(np.random.normal(0, 0.005, len(dates))))
        df['Volume'] = np.random.randint(1000000, 10000000, len(dates))
        
        price_data[asset] = df
        
    return price_data


@pytest.fixture
def sample_allocation_data() -> pd.DataFrame:
    """Create sample allocation data for testing."""
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='ME')
    
    # Create alternating allocations (simple momentum-like pattern)
    data = []
    for i, date in enumerate(dates):
        if i % 6 < 3:  # First 3 months - equity allocation
            allocation = {
                'allocation_us_equity': 1.0,
                'allocation_intl_equity': 0.0,
                'allocation_bonds': 0.0,
                'allocation_cash': 0.0,
                'primary_asset': 'us_equity',
                'signal': 'bullish'
            }
        else:  # Next 3 months - bond allocation
            allocation = {
                'allocation_us_equity': 0.0,
                'allocation_intl_equity': 0.0,
                'allocation_bonds': 1.0,
                'allocation_cash': 0.0,
                'primary_asset': 'bonds',
                'signal': 'bearish'
            }
        
        allocation['date'] = date
        data.append(allocation)
    
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    return df


@pytest.fixture
def mock_yfinance_data() -> Generator[Dict[str, pd.DataFrame], None, None]:
    """Mock yfinance data for testing."""
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    mock_data = {}
    
    symbols = ['SPY', 'VEU', 'AGG', 'BIL']
    
    for symbol in symbols:
        returns = np.random.normal(0.0003, 0.015, len(dates))
        prices = 100 * np.cumprod(1 + returns)
        
        df = pd.DataFrame(index=dates)
        df['Close'] = prices
        df['Open'] = df['Close'] * (1 + np.random.normal(0, 0.001, len(dates)))
        df['High'] = np.maximum(df['Open'], df['Close']) * (1 + np.abs(np.random.normal(0, 0.005, len(dates))))
        df['Low'] = np.minimum(df['Open'], df['Close']) * (1 - np.abs(np.random.normal(0, 0.005, len(dates))))
        df['Volume'] = np.random.randint(1000000, 10000000, len(dates))
        
        mock_data[symbol] = df
    
    yield mock_data


@pytest.fixture
def temp_output_dir(tmp_path) -> Generator[str, None, None]:
    """Create temporary output directory for tests."""
    output_dir = tmp_path / "gem_test_output"
    output_dir.mkdir()
    yield str(output_dir)


# Test data constants
TEST_START_DATE = datetime(2020, 1, 1)
TEST_END_DATE = datetime(2023, 12, 31)
TEST_INITIAL_CAPITAL = 10000.0
TEST_TRANSACTION_COST = 0.001