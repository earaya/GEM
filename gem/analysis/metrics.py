"""
Performance metrics calculation module.

This module provides comprehensive performance analysis tools for investment
strategies, including risk metrics, ratios, and statistical measures.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from loguru import logger


class PerformanceMetrics:
    """
    Comprehensive performance metrics calculator for investment strategies.
    
    Provides methods for calculating:
    - Return metrics (total, annualized, excess returns)
    - Risk metrics (volatility, VaR, CVaR, maximum drawdown)
    - Risk-adjusted ratios (Sharpe, Sortino, Calmar, Information)
    - Statistical measures (skewness, kurtosis, beta)
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize performance metrics calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate (default 2%)
        """
        self.risk_free_rate = risk_free_rate
        
    def calculate_returns_metrics(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive return metrics.
        
        Args:
            returns: Return series
            benchmark_returns: Benchmark return series (optional)
            
        Returns:
            Dictionary of return metrics
        """
        if returns.empty:
            return {}
            
        returns_clean = returns.dropna()
        periods_per_year = self._infer_frequency(returns_clean)
        
        metrics = {}
        
        # Basic return metrics
        metrics['total_return'] = (1 + returns_clean).prod() - 1
        metrics['annualized_return'] = (1 + metrics['total_return']) ** (1 / (len(returns_clean) / periods_per_year)) - 1
        metrics['mean_return'] = returns_clean.mean()
        metrics['median_return'] = returns_clean.median()
        
        # Volatility metrics
        metrics['volatility'] = returns_clean.std() * np.sqrt(periods_per_year)
        metrics['downside_deviation'] = self._calculate_downside_deviation(returns_clean) * np.sqrt(periods_per_year)
        
        # Risk-adjusted metrics
        excess_return = metrics['annualized_return'] - self.risk_free_rate
        if metrics['volatility'] > 0:
            metrics['sharpe_ratio'] = excess_return / metrics['volatility']
        else:
            metrics['sharpe_ratio'] = 0.0
            
        if metrics['downside_deviation'] > 0:
            metrics['sortino_ratio'] = excess_return / metrics['downside_deviation']
        else:
            metrics['sortino_ratio'] = 0.0
            
        # Benchmark comparison
        if benchmark_returns is not None:
            benchmark_clean = benchmark_returns.dropna()
            aligned_returns, aligned_benchmark = self._align_series(returns_clean, benchmark_clean)
            
            if len(aligned_returns) > 1:
                # Alpha and Beta
                metrics['beta'] = self._calculate_beta(aligned_returns, aligned_benchmark)
                benchmark_annual_return = (1 + aligned_benchmark).prod() ** (periods_per_year / len(aligned_benchmark)) - 1
                metrics['alpha'] = metrics['annualized_return'] - (self.risk_free_rate + metrics['beta'] * (benchmark_annual_return - self.risk_free_rate))
                
                # Information ratio
                excess_returns = aligned_returns - aligned_benchmark
                tracking_error = excess_returns.std() * np.sqrt(periods_per_year)
                if tracking_error > 0:
                    metrics['information_ratio'] = excess_returns.mean() * periods_per_year / tracking_error
                else:
                    metrics['information_ratio'] = 0.0
                    
                metrics['tracking_error'] = tracking_error
                
        return metrics
        
    def calculate_risk_metrics(
        self,
        returns: pd.Series,
        confidence_levels: List[float] = [0.95, 0.99]
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Calculate comprehensive risk metrics.
        
        Args:
            returns: Return series
            confidence_levels: Confidence levels for VaR/CVaR calculation
            
        Returns:
            Dictionary of risk metrics
        """
        if returns.empty:
            return {}
            
        returns_clean = returns.dropna()
        periods_per_year = self._infer_frequency(returns_clean)
        
        metrics = {}
        
        # Maximum drawdown
        cumulative_returns = (1 + returns_clean).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        metrics['max_drawdown'] = drawdown.min()
        metrics['avg_drawdown'] = drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0.0
        
        # Drawdown duration
        drawdown_periods = self._calculate_drawdown_duration(drawdown)
        metrics['max_drawdown_duration'] = drawdown_periods.max() if len(drawdown_periods) > 0 else 0
        metrics['avg_drawdown_duration'] = drawdown_periods.mean() if len(drawdown_periods) > 0 else 0
        
        # Calmar ratio
        if abs(metrics['max_drawdown']) > 1e-6:
            annualized_return = (1 + returns_clean).prod() ** (periods_per_year / len(returns_clean)) - 1
            metrics['calmar_ratio'] = annualized_return / abs(metrics['max_drawdown'])
        else:
            metrics['calmar_ratio'] = 0.0
            
        # Value at Risk and Conditional Value at Risk
        var_metrics = {}
        cvar_metrics = {}
        
        for confidence_level in confidence_levels:
            var_value = np.percentile(returns_clean, (1 - confidence_level) * 100)
            cvar_value = returns_clean[returns_clean <= var_value].mean()
            
            var_metrics[f'{int(confidence_level*100)}%'] = var_value
            cvar_metrics[f'{int(confidence_level*100)}%'] = cvar_value
            
        metrics['value_at_risk'] = var_metrics
        metrics['conditional_var'] = cvar_metrics
        
        # Statistical measures
        metrics['skewness'] = returns_clean.skew()
        metrics['kurtosis'] = returns_clean.kurtosis()
        
        # Win/Loss metrics
        positive_returns = returns_clean[returns_clean > 0]
        negative_returns = returns_clean[returns_clean < 0]
        
        metrics['win_rate'] = len(positive_returns) / len(returns_clean) if len(returns_clean) > 0 else 0
        metrics['avg_win'] = positive_returns.mean() if len(positive_returns) > 0 else 0
        metrics['avg_loss'] = negative_returns.mean() if len(negative_returns) > 0 else 0
        
        if len(negative_returns) > 0 and metrics['avg_loss'] != 0:
            metrics['win_loss_ratio'] = abs(metrics['avg_win'] / metrics['avg_loss'])
        else:
            metrics['win_loss_ratio'] = 0.0
            
        return metrics
        
    def calculate_rolling_metrics(
        self,
        returns: pd.Series,
        window: int = 252,
        metrics: List[str] = None
    ) -> pd.DataFrame:
        """
        Calculate rolling performance metrics.
        
        Args:
            returns: Return series
            window: Rolling window size
            metrics: List of metrics to calculate
            
        Returns:
            DataFrame with rolling metrics
        """
        if metrics is None:
            metrics = ['return', 'volatility', 'sharpe', 'max_drawdown']
            
        returns_clean = returns.dropna()
        periods_per_year = self._infer_frequency(returns_clean)
        
        rolling_metrics = {}
        
        if 'return' in metrics:
            rolling_metrics['rolling_return'] = returns_clean.rolling(window).apply(
                lambda x: (1 + x).prod() - 1
            ) * (periods_per_year / window)
            
        if 'volatility' in metrics:
            rolling_metrics['rolling_volatility'] = returns_clean.rolling(window).std() * np.sqrt(periods_per_year)
            
        if 'sharpe' in metrics:
            rolling_returns = returns_clean.rolling(window).mean() * periods_per_year
            rolling_vol = returns_clean.rolling(window).std() * np.sqrt(periods_per_year)
            rolling_metrics['rolling_sharpe'] = (rolling_returns - self.risk_free_rate) / rolling_vol
            
        if 'max_drawdown' in metrics:
            rolling_metrics['rolling_max_drawdown'] = returns_clean.rolling(window).apply(
                self._rolling_max_drawdown
            )
            
        return pd.DataFrame(rolling_metrics, index=returns_clean.index)
        
    def compare_strategies(
        self,
        strategy_returns: Dict[str, pd.Series],
        benchmark_returns: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Compare multiple strategies.
        
        Args:
            strategy_returns: Dictionary of strategy return series
            benchmark_returns: Benchmark return series
            
        Returns:
            DataFrame comparing strategies
        """
        comparison_data = []
        
        for name, returns in strategy_returns.items():
            metrics = self.calculate_returns_metrics(returns, benchmark_returns)
            risk_metrics = self.calculate_risk_metrics(returns)
            
            # Combine metrics
            combined_metrics = {**metrics, **risk_metrics}
            combined_metrics['strategy'] = name
            comparison_data.append(combined_metrics)
            
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.set_index('strategy', inplace=True)
        
        return comparison_df
        
    @staticmethod
    def _infer_frequency(returns: pd.Series) -> int:
        """Infer the frequency of returns (periods per year)."""
        if len(returns) < 2:
            return 252  # Default to daily
            
        # Calculate average time difference
        time_diffs = returns.index.to_series().diff().dropna()
        avg_diff = time_diffs.mean()
        
        # Estimate periods per year based on average difference
        if avg_diff <= pd.Timedelta(days=1.5):
            return 252  # Daily
        elif avg_diff <= pd.Timedelta(days=8):
            return 52   # Weekly
        elif avg_diff <= pd.Timedelta(days=35):
            return 12   # Monthly
        else:
            return 4    # Quarterly
            
    @staticmethod
    def _calculate_downside_deviation(returns: pd.Series, target_return: float = 0.0) -> float:
        """Calculate downside deviation."""
        downside_returns = returns[returns < target_return]
        if len(downside_returns) == 0:
            return 0.0
        return ((downside_returns - target_return) ** 2).mean() ** 0.5
        
    @staticmethod
    def _calculate_beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate beta relative to benchmark."""
        if len(returns) != len(benchmark_returns) or len(returns) < 2:
            return 0.0
            
        covariance = np.cov(returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        
        return covariance / benchmark_variance if benchmark_variance > 0 else 0.0
        
    @staticmethod
    def _align_series(series1: pd.Series, series2: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Align two series to common dates."""
        common_dates = series1.index.intersection(series2.index)
        return series1.loc[common_dates], series2.loc[common_dates]
        
    @staticmethod
    def _calculate_drawdown_duration(drawdown: pd.Series) -> pd.Series:
        """Calculate drawdown duration periods."""
        in_drawdown = drawdown < 0
        drawdown_periods = []
        
        current_duration = 0
        for is_drawdown in in_drawdown:
            if is_drawdown:
                current_duration += 1
            else:
                if current_duration > 0:
                    drawdown_periods.append(current_duration)
                current_duration = 0
                
        # Handle case where series ends in drawdown
        if current_duration > 0:
            drawdown_periods.append(current_duration)
            
        return pd.Series(drawdown_periods)
        
    @staticmethod
    def _rolling_max_drawdown(returns: pd.Series) -> float:
        """Calculate maximum drawdown for a rolling window."""
        if len(returns) == 0:
            return 0.0
            
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        return drawdown.min()