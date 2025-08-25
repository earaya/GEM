"""
Tests for performance metrics analysis.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from gem.analysis.metrics import PerformanceMetrics


class TestPerformanceMetrics:
    """Test PerformanceMetrics class."""
    
    def test_initialization(self):
        """Test metrics calculator initialization."""
        metrics = PerformanceMetrics()
        assert metrics.risk_free_rate == 0.02
        
        metrics_custom = PerformanceMetrics(risk_free_rate=0.03)
        assert metrics_custom.risk_free_rate == 0.03
    
    def test_calculate_returns_metrics_basic(self, sample_returns_data):
        """Test basic return metrics calculation."""
        metrics = PerformanceMetrics()
        returns = sample_returns_data['us_equity']
        
        result = metrics.calculate_returns_metrics(returns)
        
        # Check that all expected metrics are present
        expected_metrics = [
            'total_return', 'annualized_return', 'mean_return', 'median_return',
            'volatility', 'downside_deviation', 'sharpe_ratio', 'sortino_ratio'
        ]
        
        for metric in expected_metrics:
            assert metric in result
            assert isinstance(result[metric], (int, float))
        
        # Check logical relationships
        assert result['annualized_return'] != 0  # Should have some return
        assert result['volatility'] >= 0  # Volatility should be non-negative
        assert result['downside_deviation'] >= 0  # Downside deviation should be non-negative
    
    def test_calculate_returns_metrics_with_benchmark(self, sample_returns_data):
        """Test return metrics calculation with benchmark."""
        metrics = PerformanceMetrics()
        returns = sample_returns_data['us_equity']
        benchmark = sample_returns_data['bonds']
        
        result = metrics.calculate_returns_metrics(returns, benchmark)
        
        # Should have benchmark-specific metrics
        benchmark_metrics = ['beta', 'alpha', 'information_ratio', 'tracking_error']
        
        for metric in benchmark_metrics:
            assert metric in result
            assert isinstance(result[metric], (int, float))
        
        # Beta should be reasonable (not extreme)
        assert -5 <= result['beta'] <= 5
    
    def test_calculate_returns_metrics_empty(self):
        """Test return metrics calculation with empty data."""
        metrics = PerformanceMetrics()
        empty_returns = pd.Series(dtype=float)
        
        result = metrics.calculate_returns_metrics(empty_returns)
        
        assert result == {}
    
    def test_calculate_risk_metrics(self, sample_returns_data):
        """Test risk metrics calculation."""
        metrics = PerformanceMetrics()
        returns = sample_returns_data['us_equity']
        
        result = metrics.calculate_risk_metrics(returns)
        
        # Check expected risk metrics
        expected_metrics = [
            'max_drawdown', 'avg_drawdown', 'max_drawdown_duration', 'avg_drawdown_duration',
            'calmar_ratio', 'value_at_risk', 'conditional_var', 'skewness', 'kurtosis',
            'win_rate', 'avg_win', 'avg_loss', 'win_loss_ratio'
        ]
        
        for metric in expected_metrics:
            assert metric in result
        
        # Check logical constraints
        assert result['max_drawdown'] <= 0  # Max drawdown should be negative or zero
        assert 0 <= result['win_rate'] <= 1  # Win rate should be between 0 and 1
        assert result['avg_win'] >= 0 if result['avg_win'] != 0 else True  # Avg win should be positive
        assert result['avg_loss'] <= 0 if result['avg_loss'] != 0 else True  # Avg loss should be negative
        
        # VaR and CVaR should be dictionaries
        assert isinstance(result['value_at_risk'], dict)
        assert isinstance(result['conditional_var'], dict)
    
    def test_calculate_risk_metrics_confidence_levels(self, sample_returns_data):
        """Test risk metrics with custom confidence levels."""
        metrics = PerformanceMetrics()
        returns = sample_returns_data['us_equity']
        
        custom_levels = [0.90, 0.95, 0.99]
        result = metrics.calculate_risk_metrics(returns, confidence_levels=custom_levels)
        
        # Check that all confidence levels are included
        for level in custom_levels:
            level_str = f'{int(level*100)}%'
            assert level_str in result['value_at_risk']
            assert level_str in result['conditional_var']
    
    def test_calculate_rolling_metrics(self, sample_returns_data):
        """Test rolling metrics calculation."""
        metrics = PerformanceMetrics()
        returns = sample_returns_data['us_equity']
        
        # Use smaller window for test data
        window = min(12, len(returns) // 2)
        rolling_result = metrics.calculate_rolling_metrics(returns, window=window)
        
        assert isinstance(rolling_result, pd.DataFrame)
        assert len(rolling_result) > 0
        
        # Check default metrics are calculated
        expected_cols = ['rolling_return', 'rolling_volatility', 'rolling_sharpe']
        for col in expected_cols:
            if col in rolling_result.columns:  # Some might be NaN if window is too large
                assert not rolling_result[col].isna().all()
    
    def test_calculate_rolling_metrics_custom(self, sample_returns_data):
        """Test rolling metrics with custom metrics list."""
        metrics = PerformanceMetrics()
        returns = sample_returns_data['us_equity']
        
        custom_metrics = ['volatility', 'return']
        window = min(12, len(returns) // 2)
        rolling_result = metrics.calculate_rolling_metrics(
            returns, 
            window=window, 
            metrics=custom_metrics
        )
        
        assert isinstance(rolling_result, pd.DataFrame)
        # Should only have the requested metrics
        expected_cols = ['rolling_volatility', 'rolling_return']
        for col in expected_cols:
            assert col in rolling_result.columns
    
    def test_compare_strategies(self, sample_returns_data):
        """Test strategy comparison."""
        metrics = PerformanceMetrics()
        
        # Use two strategies for comparison
        strategies = {
            'Strategy A': sample_returns_data['us_equity'],
            'Strategy B': sample_returns_data['bonds']
        }
        
        comparison = metrics.compare_strategies(strategies)
        
        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 2
        assert 'Strategy A' in comparison.index
        assert 'Strategy B' in comparison.index
        
        # Should have key metrics
        expected_cols = ['total_return', 'annualized_return', 'volatility', 'sharpe_ratio']
        for col in expected_cols:
            assert col in comparison.columns
    
    def test_compare_strategies_with_benchmark(self, sample_returns_data):
        """Test strategy comparison with benchmark."""
        metrics = PerformanceMetrics()
        
        strategies = {
            'Strategy A': sample_returns_data['us_equity'],
            'Strategy B': sample_returns_data['intl_equity']
        }
        benchmark = sample_returns_data['bonds']
        
        comparison = metrics.compare_strategies(strategies, benchmark)
        
        # Should have benchmark-specific metrics
        benchmark_cols = ['beta', 'alpha', 'information_ratio']
        for col in benchmark_cols:
            assert col in comparison.columns
    
    def test_infer_frequency_daily(self):
        """Test frequency inference for daily data."""
        # Create daily data
        daily_returns = pd.Series(
            [0.01, -0.005, 0.003], 
            index=pd.date_range('2023-01-01', periods=3, freq='D')
        )
        
        freq = PerformanceMetrics._infer_frequency(daily_returns)
        assert freq == 252
    
    def test_infer_frequency_monthly(self):
        """Test frequency inference for monthly data."""
        # Create monthly data
        monthly_returns = pd.Series(
            [0.02, -0.01, 0.015], 
            index=pd.date_range('2023-01-01', periods=3, freq='ME')
        )
        
        freq = PerformanceMetrics._infer_frequency(monthly_returns)
        assert freq == 12
    
    def test_calculate_downside_deviation(self):
        """Test downside deviation calculation."""
        returns = pd.Series([0.05, -0.02, 0.03, -0.01, 0.04, -0.03])
        
        downside_dev = PerformanceMetrics._calculate_downside_deviation(returns)
        
        assert downside_dev >= 0
        assert isinstance(downside_dev, float)
    
    def test_calculate_downside_deviation_no_negative(self):
        """Test downside deviation with no negative returns."""
        returns = pd.Series([0.01, 0.02, 0.03, 0.015])
        
        downside_dev = PerformanceMetrics._calculate_downside_deviation(returns)
        
        assert downside_dev == 0.0
    
    def test_calculate_beta(self):
        """Test beta calculation."""
        # Create correlated series
        benchmark = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        returns = benchmark * 1.5 + pd.Series([0.001, -0.001, 0.002, 0.001, -0.001])  # Higher beta
        
        beta = PerformanceMetrics._calculate_beta(returns, benchmark)
        
        assert beta > 1.0  # Should be higher than 1 due to scaling
        assert isinstance(beta, float)
    
    def test_calculate_beta_uncorrelated(self):
        """Test beta calculation with uncorrelated data."""
        np.random.seed(42)
        benchmark = pd.Series(np.random.normal(0, 0.02, 100))
        returns = pd.Series(np.random.normal(0, 0.02, 100))
        
        beta = PerformanceMetrics._calculate_beta(returns, benchmark)
        
        # Should be close to 0 for uncorrelated data
        assert abs(beta) < 1.0  
        assert isinstance(beta, float)
    
    def test_align_series(self):
        """Test series alignment."""
        dates1 = pd.date_range('2023-01-01', periods=5, freq='D')
        dates2 = pd.date_range('2023-01-02', periods=4, freq='D')  # Overlapping but different
        
        series1 = pd.Series([1, 2, 3, 4, 5], index=dates1)
        series2 = pd.Series([10, 20, 30, 40], index=dates2)
        
        aligned1, aligned2 = PerformanceMetrics._align_series(series1, series2)
        
        # Should have same length and index
        assert len(aligned1) == len(aligned2)
        assert aligned1.index.equals(aligned2.index)
        
        # Should only have overlapping dates
        expected_dates = dates1.intersection(dates2)
        assert len(aligned1) == len(expected_dates)