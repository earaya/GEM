"""
Tests for momentum strategy functionality.
"""

import pytest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import calendar

from gem.strategy.momentum import GlobalEquitiesMomentum, AssetAllocation, MomentumSignal


class TestAssetAllocation:
    """Test AssetAllocation class."""
    
    def test_allocation_creation(self):
        """Test creating asset allocation."""
        date = datetime(2023, 1, 31)
        allocation_dict = {"us_equity": 1.0, "intl_equity": 0.0, "bonds": 0.0, "cash": 0.0}
        momentum_scores = {"us_equity": 0.12, "intl_equity": 0.08, "bonds": 0.03, "cash": 0.01}
        
        allocation = AssetAllocation(
            date=date,
            primary_asset="us_equity",
            allocation=allocation_dict,
            signal=MomentumSignal.BULLISH,
            momentum_scores=momentum_scores
        )
        
        assert allocation.date == date
        assert allocation.primary_asset == "us_equity"
        assert allocation.allocation == allocation_dict
        assert allocation.signal == MomentumSignal.BULLISH
        assert allocation.momentum_scores == momentum_scores
    
    def test_allocation_validation_sum(self):
        """Test allocation validation - must sum to 1.0."""
        date = datetime(2023, 1, 31)
        invalid_allocation = {"us_equity": 0.5, "intl_equity": 0.3, "bonds": 0.1, "cash": 0.05}  # Sum = 0.95
        momentum_scores = {"us_equity": 0.12, "intl_equity": 0.08, "bonds": 0.03, "cash": 0.01}
        
        with pytest.raises(ValueError, match="Allocations must sum to 1.0"):
            AssetAllocation(
                date=date,
                primary_asset="us_equity",
                allocation=invalid_allocation,
                signal=MomentumSignal.BULLISH,
                momentum_scores=momentum_scores
            )


class TestGlobalEquitiesMomentum:
    """Test GlobalEquitiesMomentum strategy class."""
    
    def test_initialization(self, sample_config):
        """Test strategy initialization."""
        strategy = GlobalEquitiesMomentum(sample_config)
        
        assert strategy.config == sample_config
        assert strategy.returns_data is None
        assert strategy.momentum_data is None
        assert strategy.allocations == []
    
    @patch('gem.strategy.momentum.DataFetcher')
    def test_fetch_data(self, mock_data_fetcher_class, sample_config, sample_price_data):
        """Test data fetching."""
        # Setup mock
        mock_fetcher = Mock()
        mock_fetcher.fetch_multiple_assets.return_value = sample_price_data
        mock_data_fetcher_class.return_value = mock_fetcher
        
        strategy = GlobalEquitiesMomentum(sample_config)
        
        # Mock the data processor
        with patch.object(strategy.data_processor, 'prepare_strategy_data') as mock_prepare:
            mock_returns = {key: pd.Series([0.01, 0.02, -0.01], 
                                         index=pd.date_range('2020-01-01', periods=3, freq='ME'))
                           for key in sample_price_data.keys()}
            mock_prepare.return_value = mock_returns
            
            strategy.fetch_data("2020-01-01", "2023-12-31")
            
            assert strategy.returns_data is not None
            assert strategy.momentum_data is not None
            assert len(strategy.returns_data) == len(sample_config.assets)
    
    def test_calculate_momentum_scores(self, sample_config, sample_returns_data):
        """Test momentum score calculation."""
        strategy = GlobalEquitiesMomentum(sample_config)
        
        # Set up momentum data
        strategy.momentum_data = {}
        for asset_key, returns in sample_returns_data.items():
            # Calculate simple momentum (12-month cumulative return)
            momentum = (1 + returns).rolling(12).apply(lambda x: x.prod() - 1, raw=True)
            strategy.momentum_data[asset_key] = momentum
        
        # Test score calculation
        test_date = sample_returns_data['us_equity'].index[-1]  # Last available date
        scores = strategy.calculate_momentum_scores(test_date)
        
        assert len(scores) == len(sample_config.assets)
        for asset_key in sample_config.assets.keys():
            assert asset_key in scores
            assert isinstance(scores[asset_key], (int, float))
    
    def test_generate_allocation_signal_bullish(self, sample_config):
        """Test allocation signal generation - bullish scenario."""
        strategy = GlobalEquitiesMomentum(sample_config)
        
        # Setup momentum scores favoring equities
        momentum_scores = {
            "us_equity": 0.15,    # Strong positive momentum
            "intl_equity": 0.10,  # Positive but lower
            "bonds": 0.02,        # Low positive
            "cash": 0.01          # Very low
        }
        
        primary_asset, signal, allocation = strategy.generate_allocation_signal(momentum_scores)
        
        assert signal == MomentumSignal.BULLISH
        assert primary_asset == "us_equity"  # Highest momentum
        assert allocation["us_equity"] == 1.0
        assert sum(allocation.values()) == 1.0
    
    def test_generate_allocation_signal_bearish(self, sample_config):
        """Test allocation signal generation - bearish scenario."""
        strategy = GlobalEquitiesMomentum(sample_config)
        
        # Setup momentum scores favoring cash/bonds
        momentum_scores = {
            "us_equity": -0.05,   # Negative momentum
            "intl_equity": -0.08, # Negative momentum
            "bonds": 0.03,        # Low positive
            "cash": 0.04          # Higher than equities
        }
        
        primary_asset, signal, allocation = strategy.generate_allocation_signal(momentum_scores)
        
        assert signal == MomentumSignal.BEARISH
        assert primary_asset == "bonds"
        assert allocation["bonds"] == 1.0
        assert sum(allocation.values()) == 1.0
    
    def test_calculate_strategy_allocation(self, sample_config, sample_returns_data):
        """Test strategy allocation calculation."""
        # Use current mode to match the original test expectation
        sample_config.strategy.date_mode = "current"
        strategy = GlobalEquitiesMomentum(sample_config)
        
        # Setup test data
        strategy.returns_data = sample_returns_data
        strategy.momentum_data = {}
        
        for asset_key, returns in sample_returns_data.items():
            momentum = (1 + returns).rolling(12).apply(lambda x: x.prod() - 1, raw=True)
            strategy.momentum_data[asset_key] = momentum
        
        # Test allocation calculation
        test_date = sample_returns_data['us_equity'].index[-1]
        allocation = strategy.calculate_strategy_allocation(test_date)
        
        assert isinstance(allocation, AssetAllocation)
        assert allocation.date == test_date
        assert allocation.primary_asset in sample_config.assets.keys()
        assert sum(allocation.allocation.values()) == pytest.approx(1.0)
    
    def test_calculate_strategy_allocation_no_data(self, sample_config):
        """Test strategy allocation calculation without data."""
        strategy = GlobalEquitiesMomentum(sample_config)
        
        test_date = datetime(2023, 1, 31)
        
        with pytest.raises(ValueError, match="Must fetch data before calculating allocations"):
            strategy.calculate_strategy_allocation(test_date)
    
    def test_get_rebalance_dates_monthly(self, sample_config, sample_returns_data):
        """Test rebalancing date generation - monthly."""
        strategy = GlobalEquitiesMomentum(sample_config)
        strategy.returns_data = sample_returns_data
        
        dates = strategy._get_rebalance_dates("2020-01-01", "2020-12-31", "monthly")
        
        assert len(dates) > 0
        assert all(isinstance(date, datetime) for date in dates)
    
    def test_get_rebalance_dates_quarterly(self, sample_config, sample_returns_data):
        """Test rebalancing date generation - quarterly."""
        strategy = GlobalEquitiesMomentum(sample_config)
        strategy.returns_data = sample_returns_data
        
        dates = strategy._get_rebalance_dates("2020-01-01", "2020-12-31", "quarterly")
        monthly_dates = strategy._get_rebalance_dates("2020-01-01", "2020-12-31", "monthly")
        
        assert len(dates) <= len(monthly_dates)
        assert len(dates) > 0
    
    def test_allocations_to_dataframe(self, sample_config):
        """Test conversion of allocations to DataFrame."""
        strategy = GlobalEquitiesMomentum(sample_config)
        
        # Create sample allocations
        dates = [datetime(2023, 1, 31), datetime(2023, 2, 28)]
        allocations = []
        
        for date in dates:
            allocation = AssetAllocation(
                date=date,
                primary_asset="us_equity",
                allocation={"us_equity": 1.0, "intl_equity": 0.0, "bonds": 0.0, "cash": 0.0},
                signal=MomentumSignal.BULLISH,
                momentum_scores={"us_equity": 0.12, "intl_equity": 0.08, "bonds": 0.03, "cash": 0.01}
            )
            allocations.append(allocation)
        
        strategy.allocations = allocations
        
        df = strategy._allocations_to_dataframe()
        
        assert len(df) == len(allocations)
        assert all(date in df.index for date in dates)
        assert 'primary_asset' in df.columns
        assert 'signal' in df.columns
        
        # Check allocation columns
        for asset in sample_config.assets.keys():
            assert f'allocation_{asset}' in df.columns
            assert f'momentum_{asset}' in df.columns
    
    def test_get_current_allocation_empty(self, sample_config):
        """Test getting current allocation when no allocations exist."""
        strategy = GlobalEquitiesMomentum(sample_config)
        
        current = strategy.get_current_allocation()
        assert current is None
    
    def test_get_current_allocation(self, sample_config):
        """Test getting current allocation."""
        strategy = GlobalEquitiesMomentum(sample_config)
        
        # Create allocations with different dates
        early_allocation = AssetAllocation(
            date=datetime(2023, 1, 31),
            primary_asset="bonds",
            allocation={"us_equity": 0.0, "intl_equity": 0.0, "bonds": 1.0, "cash": 0.0},
            signal=MomentumSignal.BEARISH,
            momentum_scores={"us_equity": -0.05, "intl_equity": -0.03, "bonds": 0.02, "cash": 0.01}
        )
        
        latest_allocation = AssetAllocation(
            date=datetime(2023, 2, 28),
            primary_asset="us_equity",
            allocation={"us_equity": 1.0, "intl_equity": 0.0, "bonds": 0.0, "cash": 0.0},
            signal=MomentumSignal.BULLISH,
            momentum_scores={"us_equity": 0.12, "intl_equity": 0.08, "bonds": 0.03, "cash": 0.01}
        )
        
        strategy.allocations = [early_allocation, latest_allocation]
        
        current = strategy.get_current_allocation()
        assert current == latest_allocation
    
    def test_get_performance_summary(self, sample_config):
        """Test performance summary generation."""
        strategy = GlobalEquitiesMomentum(sample_config)
        
        # Create sample allocations
        allocations = [
            AssetAllocation(
                date=datetime(2023, 1, 31),
                primary_asset="us_equity",
                allocation={"us_equity": 1.0, "intl_equity": 0.0, "bonds": 0.0, "cash": 0.0},
                signal=MomentumSignal.BULLISH,
                momentum_scores={"us_equity": 0.12, "intl_equity": 0.08, "bonds": 0.03, "cash": 0.01}
            ),
            AssetAllocation(
                date=datetime(2023, 2, 28),
                primary_asset="bonds",
                allocation={"us_equity": 0.0, "intl_equity": 0.0, "bonds": 1.0, "cash": 0.0},
                signal=MomentumSignal.BEARISH,
                momentum_scores={"us_equity": -0.05, "intl_equity": -0.03, "bonds": 0.02, "cash": 0.01}
            )
        ]
        
        strategy.allocations = allocations
        
        summary = strategy.get_performance_summary()
        
        assert summary['total_decisions'] == 2
        assert summary['bullish_signals'] == 1
        assert summary['bearish_signals'] == 1
        assert summary['bullish_percentage'] == 50.0
        assert 'asset_counts' in summary
        assert summary['asset_counts']['us_equity'] == 1
        assert summary['asset_counts']['bonds'] == 1


class TestDateModeResolution:
    """Test date mode resolution functionality."""
    
    def test_resolve_calculation_date_current_mode(self, sample_config):
        """Test date resolution in current mode."""
        # Set date mode to current
        sample_config.strategy.date_mode = "current"
        strategy = GlobalEquitiesMomentum(sample_config)
        
        # Test with specific date
        test_date = datetime(2023, 8, 15)
        resolved_date = strategy._resolve_calculation_date(test_date)
        assert resolved_date == test_date
        
        # Test with no date (should use current time, but we can't test exact match)
        resolved_date = strategy._resolve_calculation_date()
        assert isinstance(resolved_date, datetime)
    
    def test_resolve_calculation_date_end_of_month_mode(self, sample_config):
        """Test date resolution in end_of_month mode."""
        # Set date mode to end_of_month
        sample_config.strategy.date_mode = "end_of_month"
        strategy = GlobalEquitiesMomentum(sample_config)
        
        # Test mid-month date - should resolve to end of previous month
        test_date = datetime(2023, 8, 15)  # August 15th
        resolved_date = strategy._resolve_calculation_date(test_date)
        
        # Should resolve to July 31st, 2023 (or July 28th if weekend)
        expected_month = 7  # July
        expected_year = 2023
        assert resolved_date.month == expected_month
        assert resolved_date.year == expected_year
        
        # Should be last business day of July
        last_day_july = calendar.monthrange(2023, 7)[1]  # July 31st
        july_31_2023 = datetime(2023, 7, 31)
        
        # If July 31st was a weekend, it should be moved to Friday
        while july_31_2023.weekday() > 4:  # Move to Friday if weekend
            july_31_2023 = july_31_2023 - timedelta(days=1)
        
        assert resolved_date.date() == july_31_2023.date()
    
    def test_resolve_calculation_date_first_day_of_month(self, sample_config):
        """Test date resolution when input is first day of month."""
        sample_config.strategy.date_mode = "end_of_month" 
        strategy = GlobalEquitiesMomentum(sample_config)
        
        # Test first day of month - should still go to end of previous month
        test_date = datetime(2023, 8, 1)  # August 1st
        resolved_date = strategy._resolve_calculation_date(test_date)
        
        # Should resolve to end of July
        assert resolved_date.month == 7
        assert resolved_date.year == 2023
    
    def test_resolve_calculation_date_weekend_adjustment(self, sample_config):
        """Test weekend adjustment in end_of_month mode."""
        sample_config.strategy.date_mode = "end_of_month"
        strategy = GlobalEquitiesMomentum(sample_config)
        
        # Test with a month ending on weekend (June 2024 ends on Sunday June 30)
        test_date = datetime(2024, 7, 15)  # Mid July
        resolved_date = strategy._resolve_calculation_date(test_date)
        
        # Should resolve to end of June, adjusted for weekends
        assert resolved_date.month == 6
        assert resolved_date.year == 2024
        # Should be a weekday (Monday=0 to Friday=4)
        assert resolved_date.weekday() <= 4
    
    def test_calculate_strategy_allocation_with_date_modes(self, sample_config, sample_returns_data):
        """Test strategy allocation calculation with different date modes."""
        # Mock the data fetcher and processor
        with patch.object(GlobalEquitiesMomentum, 'fetch_data'), \
             patch.object(GlobalEquitiesMomentum, 'calculate_momentum_scores') as mock_momentum:
            
            mock_momentum.return_value = {
                "us_equity": 0.12,
                "intl_equity": 0.08, 
                "bonds": 0.03,
                "cash": 0.01
            }
            
            # Test current mode
            sample_config.strategy.date_mode = "current"
            strategy = GlobalEquitiesMomentum(sample_config)
            strategy.momentum_data = sample_returns_data
            
            test_date = datetime(2023, 8, 15)
            allocation = strategy.calculate_strategy_allocation(test_date)
            assert allocation.date == test_date
            
            # Test end_of_month mode
            sample_config.strategy.date_mode = "end_of_month" 
            strategy = GlobalEquitiesMomentum(sample_config)
            strategy.momentum_data = sample_returns_data
            
            allocation = strategy.calculate_strategy_allocation(test_date)
            # Date should be resolved to end of previous month
            assert allocation.date.month == 7  # July
            assert allocation.date.year == 2023
    
    def test_get_current_allocation_live_with_date_modes(self, sample_config):
        """Test get_current_allocation_live with different date modes."""
        with patch.object(GlobalEquitiesMomentum, 'fetch_data'), \
             patch.object(GlobalEquitiesMomentum, 'calculate_strategy_allocation') as mock_calc:
            
            # Mock return value
            mock_allocation = AssetAllocation(
                date=datetime(2023, 7, 31),
                primary_asset="us_equity",
                allocation={"us_equity": 1.0, "intl_equity": 0.0, "bonds": 0.0, "cash": 0.0},
                signal=MomentumSignal.BULLISH,
                momentum_scores={"us_equity": 0.12, "intl_equity": 0.08, "bonds": 0.03, "cash": 0.01}
            )
            mock_calc.return_value = mock_allocation
            
            # Test that get_current_allocation_live calls calculate_strategy_allocation
            strategy = GlobalEquitiesMomentum(sample_config)
            result = strategy.get_current_allocation_live()
            
            mock_calc.assert_called_once_with(None)
            assert result == mock_allocation
    
    def test_invalid_date_mode_raises_error(self, sample_config):
        """Test that invalid date mode raises ValueError."""
        sample_config.strategy.date_mode = "invalid_mode"
        strategy = GlobalEquitiesMomentum(sample_config)
        
        with pytest.raises(ValueError, match="Unknown date mode: invalid_mode"):
            strategy._resolve_calculation_date(datetime(2023, 8, 15))