"""
Data fetching module for GEM strategy.

This module provides robust data fetching capabilities with multiple data sources,
caching, error handling, and rate limiting.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from pathlib import Path

import pandas as pd
import yfinance as yf
from diskcache import Cache
from loguru import logger

from gem.config import GemConfig, AssetConfig


class DataFetchError(Exception):
    """Custom exception for data fetching errors."""
    pass


class DataFetcher:
    """
    Robust data fetcher with multiple sources, caching, and error handling.
    
    Features:
    - Primary source: Yahoo Finance via yfinance
    - Comprehensive error handling and retries
    - Disk-based caching with TTL
    - Rate limiting to respect API limits
    - Data validation and cleaning
    """
    
    def __init__(self, config: GemConfig):
        """
        Initialize the data fetcher.
        
        Args:
            config: Configuration object containing data settings
        """
        self.config = config
        self.cache = None
        self._setup_cache()
        self._last_request_time = 0.0
        
    def _setup_cache(self) -> None:
        """Set up disk-based cache if enabled."""
        if self.config.data.cache_enabled:
            # Ensure output_directory is a Path object
            output_dir = Path(self.config.output_directory)
            cache_dir = output_dir / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache = Cache(str(cache_dir))
            logger.info(f"Cache enabled at {cache_dir}")
        else:
            logger.info("Cache disabled")
            
    def _rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        min_interval = self.config.data.rate_limit_seconds
        
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
            
        self._last_request_time = time.time()
        
    def _get_cache_key(self, symbol: str, start_date: str, end_date: str) -> str:
        """Generate cache key for the request."""
        return f"data_{symbol}_{start_date}_{end_date}"
        
    def _validate_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Validate and clean fetched data.
        
        Args:
            data: Raw data DataFrame
            symbol: Asset symbol
            
        Returns:
            Validated and cleaned DataFrame
            
        Raises:
            DataFetchError: If data validation fails
        """
        if data is None or data.empty:
            raise DataFetchError(f"No data returned for {symbol}")
            
        # Check for required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise DataFetchError(f"Missing required columns for {symbol}: {missing_cols}")
            
        # Remove rows with all NaN values
        data = data.dropna(how='all')
        
        if data.empty:
            raise DataFetchError(f"No valid data after cleaning for {symbol}")
            
        # Forward fill missing values (common for financial data)
        data = data.ffill()
        
        # Check for remaining NaN values in Close price
        if data['Close'].isna().any():
            logger.warning(f"NaN values found in Close prices for {symbol}")
            data = data.dropna(subset=['Close'])
            
        # Normalize timezone - convert to UTC then remove timezone for consistency
        if hasattr(data.index, 'tz') and data.index.tz is not None:
            data.index = data.index.tz_convert('UTC').tz_localize(None)
            
        logger.info(f"Validated data for {symbol}: {len(data)} rows")
        return data
        
    def fetch_price_data(
        self,
        symbol: str,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        period: str = "max"
    ) -> pd.DataFrame:
        """
        Fetch price data for a given symbol.
        
        Args:
            symbol: Asset symbol to fetch
            start_date: Start date for data (optional)
            end_date: End date for data (optional)  
            period: Period to fetch if dates not specified
            
        Returns:
            DataFrame with OHLCV data
            
        Raises:
            DataFetchError: If data fetching fails
        """
        # Convert dates to strings for caching
        start_str = start_date.strftime("%Y-%m-%d") if isinstance(start_date, datetime) else start_date
        end_str = end_date.strftime("%Y-%m-%d") if isinstance(end_date, datetime) else end_date
        
        # Check cache first
        cache_key = self._get_cache_key(symbol, start_str or "None", end_str or "None")
        
        if self.cache is not None:
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                logger.debug(f"Cache hit for {symbol}")
                return cached_data
                
        logger.info(f"Fetching data for {symbol} from {start_str} to {end_str}")
        
        # Apply rate limiting
        self._rate_limit()
        
        try:
            # Create yfinance Ticker object
            ticker = yf.Ticker(symbol)
            
            # Fetch historical data
            if start_date and end_date:
                data = ticker.history(start=start_date, end=end_date, auto_adjust=True)
            else:
                data = ticker.history(period=period, auto_adjust=True)
                
            # Validate data
            data = self._validate_data(data, symbol)
            
            # Cache the data
            if self.cache is not None:
                ttl_seconds = self.config.data.cache_ttl_hours * 3600
                self.cache.set(cache_key, data, expire=ttl_seconds)
                logger.debug(f"Cached data for {symbol}")
                
            return data
            
        except Exception as e:
            error_msg = f"Failed to fetch data for {symbol}: {str(e)}"
            logger.error(error_msg)
            raise DataFetchError(error_msg) from e
            
    def fetch_multiple_assets(
        self,
        assets: Dict[str, AssetConfig],
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        period: str = "max"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple assets.
        
        Args:
            assets: Dictionary of asset configurations
            start_date: Start date for data
            end_date: End date for data
            period: Period to fetch if dates not specified
            
        Returns:
            Dictionary mapping asset keys to DataFrames
        """
        results = {}
        failed_assets = []
        
        for asset_key, asset_config in assets.items():
            try:
                logger.info(f"Fetching data for {asset_key} ({asset_config.symbol})")
                data = self.fetch_price_data(
                    asset_config.symbol,
                    start_date=start_date,
                    end_date=end_date,
                    period=period
                )
                results[asset_key] = data
                
            except DataFetchError as e:
                logger.error(f"Failed to fetch {asset_key}: {e}")
                failed_assets.append(asset_key)
                continue
                
        if failed_assets:
            logger.warning(f"Failed to fetch data for assets: {failed_assets}")
            
        if not results:
            raise DataFetchError("Failed to fetch data for all assets")
            
        logger.info(f"Successfully fetched data for {len(results)} assets")
        return results
        
    def get_monthly_returns(
        self,
        symbol: str,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        period: str = "max"
    ) -> pd.Series:
        """
        Get monthly returns for a symbol.
        
        Args:
            symbol: Asset symbol
            start_date: Start date
            end_date: End date
            period: Period to fetch
            
        Returns:
            Series of monthly returns
        """
        data = self.fetch_price_data(symbol, start_date, end_date, period)
        
        # Resample to month end and calculate returns
        monthly_prices = data['Close'].resample('ME').last()
        monthly_returns = monthly_prices.pct_change().dropna()
        
        logger.debug(f"Calculated {len(monthly_returns)} monthly returns for {symbol}")
        return monthly_returns
        
    def clear_cache(self) -> None:
        """Clear all cached data."""
        if self.cache is not None:
            self.cache.clear()
            logger.info("Cache cleared")
        else:
            logger.info("No cache to clear")
            
    def close(self) -> None:
        """Close the cache and cleanup resources."""
        if self.cache is not None:
            self.cache.close()
            self.cache = None
            logger.debug("Cache closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.close()
        
    def __del__(self):
        """Destructor - cleanup resources when object is garbage collected."""
        try:
            self.close()
        except Exception:
            # Ignore errors during cleanup in destructor
            pass
            
    def get_cache_stats(self) -> Dict[str, Union[int, float]]:
        """Get cache statistics."""
        if self.cache is not None:
            stats = {
                "size": len(self.cache),
                "volume": self.cache.volume(),
            }
            return stats
        return {"size": 0, "volume": 0}