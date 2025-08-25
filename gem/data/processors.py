"""
Data processing module for GEM strategy.

This module provides data processing utilities for financial time series,
including return calculations, data alignment, and momentum indicators.
"""

from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from loguru import logger

from gem.config import GemConfig


class DataProcessor:
    """
    Data processing utilities for financial time series analysis.
    
    Provides methods for:
    - Return calculations (simple, log, total)
    - Data alignment across multiple assets
    - Momentum indicator calculation
    - End-of-month filtering
    - Missing data handling
    """
    
    def __init__(self, config: GemConfig):
        """
        Initialize the data processor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
    @staticmethod
    def calculate_returns(
        prices: pd.Series,
        method: str = "simple",
        periods: int = 1
    ) -> pd.Series:
        """
        Calculate returns from price series.
        
        Args:
            prices: Price series
            method: Return calculation method ('simple', 'log', 'total')
            periods: Number of periods for return calculation
            
        Returns:
            Series of returns
        """
        if method == "simple":
            returns = prices.pct_change(periods=periods)
        elif method == "log":
            returns = np.log(prices / prices.shift(periods))
        elif method == "total":
            # Total return over the entire period
            returns = (prices / prices.iloc[0]) - 1
        else:
            raise ValueError(f"Unknown return method: {method}")
            
        return returns.dropna()
        
    @staticmethod
    def get_month_end_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        Filter data to month-end observations.
        
        Args:
            data: DataFrame with datetime index
            
        Returns:
            DataFrame with month-end data only
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex")
            
        # Convert timezone-aware index to UTC then to timezone-naive for consistency
        if data.index.tz is not None:
            data_copy = data.copy()
            data_copy.index = data_copy.index.tz_convert('UTC').tz_localize(None)
        else:
            data_copy = data
            
        # Use resample for proper month-end handling
        monthly_data = data_copy.resample('M').last().dropna(how='all')
        
        logger.debug(f"Filtered to {len(monthly_data)} month-end observations")
        return monthly_data
        
    def calculate_momentum(
        self,
        returns: pd.Series,
        lookback_months: Optional[int] = None
    ) -> pd.Series:
        """
        Calculate momentum indicator.
        
        Args:
            returns: Return series
            lookback_months: Lookback period (uses config default if None)
            
        Returns:
            Momentum series
        """
        if lookback_months is None:
            lookback_months = self.config.strategy.lookback_months
            
        # Calculate cumulative return over lookback period
        momentum = (1 + returns).rolling(window=lookback_months).apply(
            lambda x: x.prod() - 1, raw=True
        )
        
        logger.debug(f"Calculated {lookback_months}-month momentum")
        return momentum.dropna()
        
    @staticmethod
    def align_data(
        data_dict: Dict[str, pd.DataFrame],
        method: str = "inner"
    ) -> Dict[str, pd.DataFrame]:
        """
        Align multiple DataFrames to common date range.
        
        Args:
            data_dict: Dictionary of DataFrames to align
            method: Alignment method ('inner', 'outer', 'left', 'right')
            
        Returns:
            Dictionary of aligned DataFrames
        """
        if not data_dict:
            return {}
            
        # Get all unique dates
        all_dates = []
        for df in data_dict.values():
            if isinstance(df.index, pd.DatetimeIndex):
                all_dates.extend(df.index.tolist())
            else:
                raise ValueError("All DataFrames must have DatetimeIndex")
                
        unique_dates = sorted(set(all_dates))
        
        # Create common index based on method
        if method == "inner":
            # Find intersection of all date ranges
            common_dates = None
            for df in data_dict.values():
                if common_dates is None:
                    common_dates = set(df.index)
                else:
                    common_dates = common_dates.intersection(set(df.index))
            common_index = pd.DatetimeIndex(sorted(common_dates))
            
        elif method == "outer":
            # Use union of all dates
            common_index = pd.DatetimeIndex(unique_dates)
            
        else:
            raise ValueError(f"Alignment method '{method}' not supported")
            
        # Reindex all DataFrames
        aligned_data = {}
        for key, df in data_dict.items():
            aligned_df = df.reindex(common_index, method='ffill')
            aligned_data[key] = aligned_df
            
        logger.info(f"Aligned {len(data_dict)} DataFrames to {len(common_index)} dates")
        return aligned_data
        
    @staticmethod
    def calculate_total_return_index(
        prices: pd.Series,
        dividends: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Calculate total return index including dividends.
        
        Args:
            prices: Price series
            dividends: Dividend series (optional)
            
        Returns:
            Total return index
        """
        if dividends is None:
            # Assume price series already includes dividends (adjusted close)
            return prices / prices.iloc[0]
            
        # Calculate total return with explicit dividend adjustment
        combined_returns = (prices.pct_change() + 
                          dividends.reindex(prices.index, fill_value=0) / prices.shift(1))
        
        total_return_index = (1 + combined_returns).cumprod()
        return total_return_index.fillna(method='ffill')
        
    @staticmethod
    def detect_outliers(
        returns: pd.Series,
        method: str = "iqr",
        threshold: float = 3.0
    ) -> pd.Series:
        """
        Detect outliers in return series.
        
        Args:
            returns: Return series
            method: Outlier detection method ('iqr', 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            Boolean series indicating outliers
        """
        if method == "iqr":
            Q1 = returns.quantile(0.25)
            Q3 = returns.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = (returns < lower_bound) | (returns > upper_bound)
            
        elif method == "zscore":
            z_scores = np.abs((returns - returns.mean()) / returns.std())
            outliers = z_scores > threshold
            
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
            
        logger.info(f"Detected {outliers.sum()} outliers using {method} method")
        return outliers
        
    def prepare_strategy_data(
        self,
        data_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.Series]:
        """
        Prepare data for momentum strategy calculation.
        
        Args:
            data_dict: Dictionary of price DataFrames
            
        Returns:
            Dictionary of monthly return series
        """
        strategy_data = {}
        
        for asset_key, df in data_dict.items():
            try:
                # Convert to month-end data
                monthly_data = self.get_month_end_data(df)
                
                # Calculate monthly returns
                monthly_returns = self.calculate_returns(
                    monthly_data['Close'],
                    method="simple",
                    periods=1
                )
                
                strategy_data[asset_key] = monthly_returns
                logger.debug(f"Prepared strategy data for {asset_key}")
                
            except Exception as e:
                logger.error(f"Failed to prepare data for {asset_key}: {e}")
                continue
                
        if not strategy_data:
            raise ValueError("Failed to prepare strategy data for any assets")
            
        # Align all return series
        aligned_returns = {}
        common_dates = None
        
        for key, returns in strategy_data.items():
            if common_dates is None:
                common_dates = set(returns.index)
            else:
                common_dates = common_dates.intersection(set(returns.index))
                
        common_index = pd.DatetimeIndex(sorted(common_dates))
        
        for key, returns in strategy_data.items():
            aligned_returns[key] = returns.reindex(common_index)
            
        logger.info(f"Prepared strategy data for {len(aligned_returns)} assets "
                   f"over {len(common_index)} periods")
        
        return aligned_returns