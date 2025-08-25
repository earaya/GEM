"""
Tests for data fetching functionality.
"""

import pytest
from unittest.mock import Mock, patch
import pandas as pd
from datetime import datetime, timedelta

from gem.data.fetchers import DataFetcher, DataFetchError


class TestDataFetcher:
    """Test DataFetcher class."""
    
    def test_initialization(self, sample_config):
        """Test DataFetcher initialization."""
        fetcher = DataFetcher(sample_config)
        
        assert fetcher.config == sample_config
        assert fetcher.cache is None  # Cache disabled in test config
        assert fetcher._last_request_time == 0.0
    
    def test_cache_setup_enabled(self, sample_config, temp_output_dir):
        """Test cache setup when enabled."""
        sample_config.data.cache_enabled = True
        sample_config.output_directory = temp_output_dir
        
        fetcher = DataFetcher(sample_config)
        
        assert fetcher.cache is not None
    
    def test_get_cache_key(self, sample_config):
        """Test cache key generation."""
        fetcher = DataFetcher(sample_config)
        
        key = fetcher._get_cache_key("SPY", "2020-01-01", "2023-12-31")
        expected = "data_SPY_2020-01-01_2023-12-31"
        
        assert key == expected
    
    def test_validate_data_valid(self, sample_config):
        """Test data validation with valid data."""
        fetcher = DataFetcher(sample_config)
        
        # Create valid data
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        data = pd.DataFrame({
            'Open': [100] * len(dates),
            'High': [105] * len(dates),
            'Low': [95] * len(dates),
            'Close': [102] * len(dates),
            'Volume': [1000000] * len(dates)
        }, index=dates)
        
        validated_data = fetcher._validate_data(data, "SPY")
        
        assert not validated_data.empty
        assert all(col in validated_data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])
    
    def test_validate_data_empty(self, sample_config):
        """Test data validation with empty data."""
        fetcher = DataFetcher(sample_config)
        
        empty_data = pd.DataFrame()
        
        with pytest.raises(DataFetchError, match="No data returned"):
            fetcher._validate_data(empty_data, "SPY")
    
    def test_validate_data_missing_columns(self, sample_config):
        """Test data validation with missing columns."""
        fetcher = DataFetcher(sample_config)
        
        # Create data with missing columns
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        data = pd.DataFrame({
            'Close': [102] * len(dates),
        }, index=dates)
        
        with pytest.raises(DataFetchError, match="Missing required columns"):
            fetcher._validate_data(data, "SPY")
    
    @patch('gem.data.fetchers.yf.Ticker')
    def test_fetch_price_data_success(self, mock_ticker_class, sample_config, mock_yfinance_data):
        """Test successful price data fetching."""
        # Setup mock
        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_yfinance_data['SPY']
        mock_ticker_class.return_value = mock_ticker
        
        fetcher = DataFetcher(sample_config)
        
        # Fetch data
        data = fetcher.fetch_price_data("SPY", "2020-01-01", "2023-12-31")
        
        assert not data.empty
        assert all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])
        mock_ticker.history.assert_called_once()
    
    @patch('gem.data.fetchers.yf.Ticker')
    def test_fetch_price_data_failure(self, mock_ticker_class, sample_config):
        """Test price data fetching failure."""
        # Setup mock to raise exception
        mock_ticker = Mock()
        mock_ticker.history.side_effect = Exception("API Error")
        mock_ticker_class.return_value = mock_ticker
        
        fetcher = DataFetcher(sample_config)
        
        with pytest.raises(DataFetchError, match="Failed to fetch data for SPY"):
            fetcher.fetch_price_data("SPY", "2020-01-01", "2023-12-31")
    
    @patch('gem.data.fetchers.yf.Ticker')
    def test_fetch_multiple_assets(self, mock_ticker_class, sample_config, mock_yfinance_data):
        """Test fetching multiple assets."""
        # Setup mock
        def mock_ticker_side_effect(symbol):
            mock_ticker = Mock()
            mock_ticker.history.return_value = mock_yfinance_data.get(symbol, pd.DataFrame())
            return mock_ticker
        
        mock_ticker_class.side_effect = mock_ticker_side_effect
        
        fetcher = DataFetcher(sample_config)
        
        # Fetch data for multiple assets
        results = fetcher.fetch_multiple_assets(sample_config.assets, "2020-01-01", "2023-12-31")
        
        assert len(results) == len(sample_config.assets)
        for asset_key in sample_config.assets.keys():
            assert asset_key in results
            assert not results[asset_key].empty
    
    @patch('gem.data.fetchers.yf.Ticker')
    def test_fetch_multiple_assets_partial_failure(self, mock_ticker_class, sample_config):
        """Test fetching multiple assets with some failures."""
        # Setup mock - some succeed, some fail
        def mock_ticker_side_effect(symbol):
            mock_ticker = Mock()
            if symbol == "SPY":
                # Create valid data for SPY
                dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
                data = pd.DataFrame({
                    'Open': [100] * len(dates),
                    'High': [105] * len(dates),
                    'Low': [95] * len(dates),
                    'Close': [102] * len(dates),
                    'Volume': [1000000] * len(dates)
                }, index=dates)
                mock_ticker.history.return_value = data
            else:
                mock_ticker.history.side_effect = Exception("API Error")
            return mock_ticker
        
        mock_ticker_class.side_effect = mock_ticker_side_effect
        
        fetcher = DataFetcher(sample_config)
        
        # Should succeed for SPY but fail for others
        results = fetcher.fetch_multiple_assets(sample_config.assets, "2020-01-01", "2023-12-31")
        
        # Should have at least one successful result
        assert len(results) >= 1
        assert "us_equity" in results  # SPY should succeed
    
    def test_rate_limiting(self, sample_config):
        """Test rate limiting functionality."""
        sample_config.data.rate_limit_seconds = 0.1
        fetcher = DataFetcher(sample_config)
        
        start_time = fetcher._last_request_time = 0.0
        
        # First call should not be rate limited
        fetcher._rate_limit()
        
        # Second call should be rate limited
        import time
        call_time = time.time()
        fetcher._rate_limit()
        end_time = time.time()
        
        # Should have waited at least the rate limit duration
        assert end_time - call_time >= sample_config.data.rate_limit_seconds * 0.9  # Allow some tolerance
    
    @patch('gem.data.fetchers.yf.Ticker')
    def test_get_monthly_returns(self, mock_ticker_class, sample_config, mock_yfinance_data):
        """Test monthly returns calculation."""
        # Setup mock
        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_yfinance_data['SPY']
        mock_ticker_class.return_value = mock_ticker
        
        fetcher = DataFetcher(sample_config)
        
        monthly_returns = fetcher.get_monthly_returns("SPY", "2020-01-01", "2023-12-31")
        
        assert isinstance(monthly_returns, pd.Series)
        assert not monthly_returns.empty
        assert monthly_returns.index.freq is not None or len(monthly_returns) < len(mock_yfinance_data['SPY'])
    
    def test_clear_cache_no_cache(self, sample_config):
        """Test clearing cache when cache is disabled."""
        fetcher = DataFetcher(sample_config)
        
        # Should not raise error
        fetcher.clear_cache()
    
    def test_get_cache_stats_no_cache(self, sample_config):
        """Test getting cache stats when cache is disabled."""
        fetcher = DataFetcher(sample_config)
        
        stats = fetcher.get_cache_stats()
        
        assert stats == {"size": 0, "volume": 0}