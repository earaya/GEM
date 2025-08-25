"""
Global Equities Momentum strategy implementation.

This module implements Gary Antonacci's Global Equities Momentum (GEM) strategy
with modern Python practices, comprehensive error handling, and extensibility.
"""

from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
import calendar

import numpy as np
import pandas as pd
from loguru import logger

from gem.config import GemConfig
from gem.data.fetchers import DataFetcher
from gem.data.processors import DataProcessor


class MomentumSignal(Enum):
    """Enumeration for momentum signals."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class AssetAllocation:
    """Class representing asset allocation decision."""
    
    def __init__(
        self,
        date: datetime,
        primary_asset: str,
        allocation: Dict[str, float],
        signal: MomentumSignal,
        momentum_scores: Dict[str, float]
    ):
        """
        Initialize asset allocation.
        
        Args:
            date: Decision date
            primary_asset: Primary asset to allocate to
            allocation: Dictionary of asset allocations (must sum to 1.0)
            signal: Momentum signal
            momentum_scores: Momentum scores for each asset
        """
        self.date = date
        self.primary_asset = primary_asset
        self.allocation = allocation
        self.signal = signal
        self.momentum_scores = momentum_scores
        
        # Validate allocation sums to 1.0
        total_allocation = sum(allocation.values())
        if not np.isclose(total_allocation, 1.0, rtol=1e-5):
            raise ValueError(f"Allocations must sum to 1.0, got {total_allocation}")


class GlobalEquitiesMomentum:
    """
    Global Equities Momentum (GEM) strategy implementation.
    
    The GEM strategy follows a dual momentum approach:
    1. Absolute momentum: Compare equity returns to cash/bonds
    2. Relative momentum: Choose between domestic and international equities
    
    Strategy logic:
    - If equity momentum > cash momentum: invest in equities
    - If cash momentum > equity momentum: invest in bonds/cash
    - For equity allocation: choose higher momentum between domestic/international
    """
    
    def __init__(self, config: GemConfig):
        """
        Initialize the GEM strategy.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.data_fetcher = DataFetcher(config)
        self.data_processor = DataProcessor(config)
        
        # Strategy state
        self.returns_data: Optional[Dict[str, pd.Series]] = None
        self.momentum_data: Optional[Dict[str, pd.Series]] = None
        self.allocations: List[AssetAllocation] = []
        
        logger.info("Initialized Global Equities Momentum strategy")
        
    def _resolve_calculation_date(self, date: Optional[datetime] = None) -> datetime:
        """
        Resolve the calculation date based on the configured date mode.
        
        Args:
            date: Optional date to use. If None, uses current date.
            
        Returns:
            Resolved datetime for calculation
        """
        if date is None:
            date = datetime.now()
            
        date_mode = self.config.strategy.date_mode
        
        if date_mode == "current":
            logger.debug(f"Using current date mode: {date.date()}")
            return date
            
        elif date_mode == "end_of_month":
            # Calculate the last business day of the previous month
            if date.day == 1:
                # If it's the first day of the month, go to previous month
                prev_month = date.replace(day=1) - timedelta(days=1)
            else:
                # Go to the end of the previous month
                first_day_current_month = date.replace(day=1)
                prev_month = first_day_current_month - timedelta(days=1)
            
            # Get the last day of the previous month
            last_day = calendar.monthrange(prev_month.year, prev_month.month)[1]
            end_of_prev_month = prev_month.replace(day=last_day)
            
            # Adjust for weekends (move to previous Friday if weekend)
            while end_of_prev_month.weekday() > 4:  # 5=Saturday, 6=Sunday
                end_of_prev_month = end_of_prev_month - timedelta(days=1)
            
            logger.debug(f"Using end-of-month mode: {end_of_prev_month.date()} (original: {date.date()})")
            return end_of_prev_month
            
        else:
            raise ValueError(f"Unknown date mode: {date_mode}")
        
    def fetch_data(
        self,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None
    ) -> None:
        """
        Fetch and prepare data for strategy calculation.
        
        Args:
            start_date: Start date for data
            end_date: End date for data
        """
        logger.info("Fetching data for GEM strategy")
        
        # Fetch price data for all assets
        price_data = self.data_fetcher.fetch_multiple_assets(
            self.config.assets,
            start_date=start_date,
            end_date=end_date
        )
        
        # Process data for strategy use
        self.returns_data = self.data_processor.prepare_strategy_data(price_data)
        
        # Calculate momentum for each asset
        self.momentum_data = {}
        for asset_key, returns in self.returns_data.items():
            momentum = self.data_processor.calculate_momentum(
                returns,
                self.config.strategy.lookback_months
            )
            self.momentum_data[asset_key] = momentum
            
        logger.info(f"Prepared data for {len(self.returns_data)} assets")
        
    def calculate_momentum_scores(self, date: datetime) -> Dict[str, float]:
        """
        Calculate momentum scores for all assets at a given date.
        
        Args:
            date: Date for momentum calculation
            
        Returns:
            Dictionary of momentum scores
        """
        if self.momentum_data is None:
            raise ValueError("Must fetch data before calculating momentum scores")
            
        scores = {}
        for asset_key, momentum_series in self.momentum_data.items():
            try:
                # Get momentum score at the specified date
                score = momentum_series.asof(date)
                if pd.isna(score):
                    logger.warning(f"No momentum data for {asset_key} at {date}")
                    score = 0.0
                scores[asset_key] = score
            except (KeyError, IndexError):
                logger.warning(f"Missing momentum data for {asset_key} at {date}")
                scores[asset_key] = 0.0
                
        return scores
        
    def generate_allocation_signal(
        self,
        momentum_scores: Dict[str, float]
    ) -> Tuple[str, MomentumSignal, Dict[str, float]]:
        """
        Generate asset allocation based on momentum scores.
        
        Args:
            momentum_scores: Dictionary of momentum scores
            
        Returns:
            Tuple of (primary_asset, signal, allocation_dict)
        """
        # Get momentum scores for key assets
        us_equity_momentum = momentum_scores.get("us_equity", 0.0)
        intl_equity_momentum = momentum_scores.get("intl_equity", 0.0)
        cash_momentum = momentum_scores.get("cash", 0.0)
        
        # Apply minimum momentum threshold
        threshold = self.config.strategy.minimum_momentum_threshold
        
        # Step 1: Absolute momentum - compare best equity vs cash
        best_equity_momentum = max(us_equity_momentum, intl_equity_momentum)
        
        if best_equity_momentum > cash_momentum + threshold:
            # Equity momentum is strong - use relative momentum to choose
            signal = MomentumSignal.BULLISH
            
            if us_equity_momentum > intl_equity_momentum:
                primary_asset = "us_equity"
                allocation = {"us_equity": 1.0, "intl_equity": 0.0, "bonds": 0.0, "cash": 0.0}
            else:
                primary_asset = "intl_equity"
                allocation = {"us_equity": 0.0, "intl_equity": 1.0, "bonds": 0.0, "cash": 0.0}
                
        else:
            # Cash/bonds momentum is stronger - defensive allocation
            signal = MomentumSignal.BEARISH
            primary_asset = "bonds"
            allocation = {"us_equity": 0.0, "intl_equity": 0.0, "bonds": 1.0, "cash": 0.0}
            
        logger.debug(f"Generated allocation: {primary_asset} ({signal.value})")
        return primary_asset, signal, allocation
        
    def calculate_strategy_allocation(
        self,
        date: Optional[datetime] = None
    ) -> AssetAllocation:
        """
        Calculate strategy allocation for a specific date.
        
        Args:
            date: Date for allocation calculation. If None, uses current date
                 with date_mode resolution applied.
            
        Returns:
            AssetAllocation object
        """
        if self.momentum_data is None:
            raise ValueError("Must fetch data before calculating allocations")
            
        # Resolve the calculation date based on configuration
        resolved_date = self._resolve_calculation_date(date)
        
        # Calculate momentum scores
        momentum_scores = self.calculate_momentum_scores(resolved_date)
        
        # Generate allocation signal
        primary_asset, signal, allocation = self.generate_allocation_signal(momentum_scores)
        
        return AssetAllocation(
            date=resolved_date,
            primary_asset=primary_asset,
            allocation=allocation,
            signal=signal,
            momentum_scores=momentum_scores
        )
        
    def run_strategy(
        self,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        rebalance_frequency: str = "monthly"
    ) -> pd.DataFrame:
        """
        Run the complete GEM strategy over a date range.
        
        Args:
            start_date: Strategy start date
            end_date: Strategy end date
            rebalance_frequency: Rebalancing frequency
            
        Returns:
            DataFrame with allocation decisions over time
        """
        logger.info(f"Running GEM strategy from {start_date} to {end_date}")
        
        # Fetch data if not already done
        if self.returns_data is None:
            self.fetch_data(start_date, end_date)
            
        # Get rebalancing dates
        rebalance_dates = self._get_rebalance_dates(
            start_date, end_date, rebalance_frequency
        )
        
        # Calculate allocations for each rebalancing date
        self.allocations = []
        for date in rebalance_dates:
            try:
                allocation = self.calculate_strategy_allocation(date)
                self.allocations.append(allocation)
            except Exception as e:
                logger.error(f"Failed to calculate allocation for {date}: {e}")
                continue
                
        # Convert to DataFrame for easy analysis
        allocation_df = self._allocations_to_dataframe()
        
        logger.info(f"Generated {len(self.allocations)} allocation decisions")
        return allocation_df
        
    def _get_rebalance_dates(
        self,
        start_date: Optional[Union[str, datetime]],
        end_date: Optional[Union[str, datetime]],
        frequency: str
    ) -> List[datetime]:
        """Get list of rebalancing dates based on frequency."""
        if self.returns_data is None:
            raise ValueError("No data available for date range calculation")
            
        # Get available dates from data
        available_dates = None
        for returns_series in self.returns_data.values():
            if available_dates is None:
                available_dates = set(returns_series.index)
            else:
                available_dates = available_dates.intersection(set(returns_series.index))
                
        if not available_dates:
            raise ValueError("No common dates available across assets")
            
        available_dates = sorted(available_dates)
        
        # Filter by start/end dates if provided
        if start_date:
            start_dt = pd.to_datetime(start_date)
            available_dates = [d for d in available_dates if d >= start_dt]
            
        if end_date:
            end_dt = pd.to_datetime(end_date)
            available_dates = [d for d in available_dates if d <= end_dt]
            
        # Select dates based on frequency
        if frequency == "monthly":
            # Use all available dates (assuming monthly data)
            rebalance_dates = available_dates
        elif frequency == "quarterly":
            # Use every 3rd month
            rebalance_dates = available_dates[::3]
        elif frequency == "annually":
            # Use every 12th month
            rebalance_dates = available_dates[::12]
        else:
            raise ValueError(f"Unsupported rebalancing frequency: {frequency}")
            
        return rebalance_dates
        
    def _allocations_to_dataframe(self) -> pd.DataFrame:
        """Convert allocation list to DataFrame."""
        if not self.allocations:
            return pd.DataFrame()
            
        records = []
        for allocation in self.allocations:
            record = {
                "date": allocation.date,
                "primary_asset": allocation.primary_asset,
                "signal": allocation.signal.value,
            }
            
            # Add allocation percentages
            for asset, weight in allocation.allocation.items():
                record[f"allocation_{asset}"] = weight
                
            # Add momentum scores
            for asset, score in allocation.momentum_scores.items():
                record[f"momentum_{asset}"] = score
                
            records.append(record)
            
        df = pd.DataFrame(records)
        df.set_index("date", inplace=True)
        return df.sort_index()
        
    def get_current_allocation(self, date: Optional[datetime] = None) -> Optional[AssetAllocation]:
        """
        Get the current allocation decision.
        
        If date is provided, calculates a new allocation for that date.
        Otherwise, returns the most recent allocation from the allocations list.
        
        Args:
            date: Optional date to use for calculation. If None, returns most recent
                 allocation from the list.
                 
        Returns:
            AssetAllocation object or None if no allocations exist and no date provided
        """
        # If date is explicitly provided, calculate a new allocation
        if date is not None:
            if self.momentum_data is None:
                # If no data loaded, fetch recent data first
                resolved_date = self._resolve_calculation_date(date)
                start_date = resolved_date - timedelta(days=365 * 2)  # 2 years of data
                self.fetch_data(start_date, resolved_date)
            
            return self.calculate_strategy_allocation(date)
        
        # Otherwise, return the most recent allocation from the list (backward compatibility)
        if not self.allocations:
            return None
        return max(self.allocations, key=lambda x: x.date)
        
    def get_current_allocation_live(self, date: Optional[datetime] = None) -> AssetAllocation:
        """
        Get live allocation decision based on date mode configuration.
        
        This method always calculates a new allocation and is intended for live trading.
        
        Args:
            date: Optional date to use for calculation. If None, uses current date
                 with date_mode resolution applied.
                 
        Returns:
            AssetAllocation object for the resolved date
        """
        if self.momentum_data is None:
            # If no data loaded, fetch recent data first
            resolved_date = self._resolve_calculation_date(date)
            start_date = resolved_date - timedelta(days=365 * 2)  # 2 years of data
            self.fetch_data(start_date, resolved_date)
            
        return self.calculate_strategy_allocation(date)
        
    def get_performance_summary(self) -> Dict[str, Union[int, float]]:
        """Get summary statistics for the strategy."""
        if not self.allocations:
            return {}
            
        total_decisions = len(self.allocations)
        bullish_signals = sum(1 for a in self.allocations if a.signal == MomentumSignal.BULLISH)
        bearish_signals = sum(1 for a in self.allocations if a.signal == MomentumSignal.BEARISH)
        
        # Asset allocation statistics
        asset_counts = {}
        for allocation in self.allocations:
            primary_asset = allocation.primary_asset
            asset_counts[primary_asset] = asset_counts.get(primary_asset, 0) + 1
            
        return {
            "total_decisions": total_decisions,
            "bullish_signals": bullish_signals,
            "bearish_signals": bearish_signals,
            "bullish_percentage": (bullish_signals / total_decisions) * 100 if total_decisions > 0 else 0,
            "asset_counts": asset_counts,
        }