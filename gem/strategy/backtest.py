"""
Backtesting module for GEM strategy.

This module provides comprehensive backtesting capabilities with portfolio
simulation, transaction costs, and detailed performance analysis.
"""

from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

from gem.config import GemConfig
from gem.strategy.momentum import GlobalEquitiesMomentum, AssetAllocation
from gem.data.fetchers import DataFetcher


@dataclass
class TradeRecord:
    """Record of a portfolio trade."""
    date: datetime
    asset_from: str
    asset_to: str
    amount: float
    transaction_cost: float
    portfolio_value_before: float
    portfolio_value_after: float


class Portfolio:
    """
    Portfolio simulation with transaction costs and rebalancing.
    """
    
    def __init__(self, initial_capital: float, transaction_cost_rate: float = 0.001):
        """
        Initialize portfolio.
        
        Args:
            initial_capital: Initial portfolio value
            transaction_cost_rate: Transaction cost as decimal (e.g., 0.001 = 0.1%)
        """
        self.initial_capital = initial_capital
        self.transaction_cost_rate = transaction_cost_rate
        
        # Portfolio state
        self.cash = initial_capital
        self.positions: Dict[str, float] = {}  # Asset -> number of shares
        self.portfolio_history: List[Dict] = []
        self.trade_history: List[TradeRecord] = []
        
    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        """
        Calculate current portfolio value.
        
        Args:
            prices: Dictionary of current asset prices
            
        Returns:
            Total portfolio value
        """
        value = self.cash
        for asset, shares in self.positions.items():
            if asset in prices:
                value += shares * prices[asset]
            else:
                logger.warning(f"No price available for {asset}")
                
        return value
        
    def rebalance(
        self,
        date: datetime,
        target_allocation: Dict[str, float],
        prices: Dict[str, float]
    ) -> float:
        """
        Rebalance portfolio to target allocation.
        
        Args:
            date: Rebalancing date
            target_allocation: Target allocation weights
            prices: Current asset prices
            
        Returns:
            Total transaction costs incurred
        """
        current_value = self.get_portfolio_value(prices)
        total_transaction_cost = 0.0
        
        # Calculate target dollar amounts
        target_amounts = {
            asset: weight * current_value
            for asset, weight in target_allocation.items()
        }
        
        # Calculate current dollar amounts
        current_amounts = {asset: 0.0 for asset in target_allocation.keys()}
        current_amounts["cash"] = self.cash
        
        for asset, shares in self.positions.items():
            if asset in prices:
                current_amounts[asset] = shares * prices[asset]
                
        # Execute trades
        trades_executed = []
        
        for asset, target_amount in target_amounts.items():
            current_amount = current_amounts.get(asset, 0.0)
            difference = target_amount - current_amount
            
            if abs(difference) > 1.0:  # Only trade if difference > $1
                if asset in prices:
                    # Calculate transaction cost
                    transaction_cost = abs(difference) * self.transaction_cost_rate
                    total_transaction_cost += transaction_cost
                    
                    if difference > 0:
                        # Buy asset
                        shares_to_buy = (difference - transaction_cost) / prices[asset]
                        self.positions[asset] = self.positions.get(asset, 0) + shares_to_buy
                        self.cash -= difference
                        
                        trade = TradeRecord(
                            date=date,
                            asset_from="cash",
                            asset_to=asset,
                            amount=difference,
                            transaction_cost=transaction_cost,
                            portfolio_value_before=current_value,
                            portfolio_value_after=self.get_portfolio_value(prices)
                        )
                        
                    else:
                        # Sell asset
                        shares_to_sell = abs(difference) / prices[asset]
                        if asset in self.positions:
                            self.positions[asset] = max(0, self.positions[asset] - shares_to_sell)
                            self.cash += abs(difference) - transaction_cost
                            
                            trade = TradeRecord(
                                date=date,
                                asset_from=asset,
                                asset_to="cash",
                                amount=abs(difference),
                                transaction_cost=transaction_cost,
                                portfolio_value_before=current_value,
                                portfolio_value_after=self.get_portfolio_value(prices)
                            )
                            
                    trades_executed.append(trade)
                    
        self.trade_history.extend(trades_executed)
        
        # Record portfolio state
        portfolio_state = {
            "date": date,
            "total_value": self.get_portfolio_value(prices),
            "cash": self.cash,
            "transaction_cost": total_transaction_cost,
        }
        
        # Add position values
        for asset in target_allocation.keys():
            shares = self.positions.get(asset, 0)
            price = prices.get(asset, 0)
            portfolio_state[f"{asset}_value"] = shares * price
            portfolio_state[f"{asset}_shares"] = shares
            
        self.portfolio_history.append(portfolio_state)
        
        return total_transaction_cost


class Backtester:
    """
    Comprehensive backtesting engine for GEM strategy.
    """
    
    def __init__(self, config: GemConfig):
        """
        Initialize backtester.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.strategy = GlobalEquitiesMomentum(config)
        self.data_fetcher = DataFetcher(config)
        
        # Backtest results
        self.portfolio: Optional[Portfolio] = None
        self.benchmark_data: Optional[pd.Series] = None
        self.results: Optional[pd.DataFrame] = None
        
    def run_backtest(
        self,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        initial_capital: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Run complete backtest of GEM strategy.
        
        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Initial capital (uses config default if None)
            
        Returns:
            DataFrame with backtest results
        """
        if initial_capital is None:
            initial_capital = self.config.backtest.initial_capital
            
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        logger.info(f"Initial capital: ${initial_capital:,.2f}")
        
        # Run strategy to get allocations
        allocation_df = self.strategy.run_strategy(start_date, end_date)
        
        if allocation_df.empty:
            raise ValueError("No allocation decisions generated")
            
        # Initialize portfolio
        self.portfolio = Portfolio(
            initial_capital=initial_capital,
            transaction_cost_rate=self.config.strategy.transaction_cost
        )
        
        # Fetch benchmark data
        self._fetch_benchmark_data(start_date, end_date)
        
        # Simulate portfolio performance
        self._simulate_portfolio(allocation_df)
        
        # Generate results DataFrame
        self.results = self._generate_results()
        
        logger.info("Backtest completed successfully")
        return self.results
        
    def _fetch_benchmark_data(
        self,
        start_date: Optional[Union[str, datetime]],
        end_date: Optional[Union[str, datetime]]
    ) -> None:
        """Fetch benchmark data for comparison."""
        benchmark_symbol = self.config.backtest.benchmark_symbol
        
        try:
            benchmark_df = self.data_fetcher.fetch_price_data(
                benchmark_symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            # Get monthly returns
            monthly_prices = benchmark_df['Close'].resample('M').last()
            self.benchmark_data = monthly_prices / monthly_prices.iloc[0] * self.config.backtest.initial_capital
            
            logger.info(f"Fetched benchmark data for {benchmark_symbol}")
            
        except Exception as e:
            logger.error(f"Failed to fetch benchmark data: {e}")
            self.benchmark_data = None
            
    def _simulate_portfolio(self, allocation_df: pd.DataFrame) -> None:
        """Simulate portfolio performance over time."""
        if self.portfolio is None:
            raise ValueError("Portfolio not initialized")
            
        # Get price data for all assets
        asset_symbols = {key: config.symbol for key, config in self.config.assets.items()}
        
        for date in allocation_df.index:
            try:
                # Get current prices
                prices = {}
                for asset_key, symbol in asset_symbols.items():
                    try:
                        price_data = self.strategy.returns_data[asset_key]
                        # Convert returns to price level (cumulative)
                        cumulative_return = (1 + price_data.loc[:date]).prod()
                        prices[asset_key] = cumulative_return * 100  # Base price of 100
                    except (KeyError, IndexError):
                        logger.warning(f"No price data for {asset_key} at {date}")
                        prices[asset_key] = 100.0  # Default price
                        
                # Get target allocation
                target_allocation = {}
                for asset_key in self.config.assets.keys():
                    allocation_col = f"allocation_{asset_key}"
                    if allocation_col in allocation_df.columns:
                        target_allocation[asset_key] = allocation_df.loc[date, allocation_col]
                    else:
                        target_allocation[asset_key] = 0.0
                        
                # Rebalance portfolio
                transaction_cost = self.portfolio.rebalance(date, target_allocation, prices)
                
                logger.debug(f"Rebalanced portfolio on {date}, cost: ${transaction_cost:.2f}")
                
            except Exception as e:
                logger.error(f"Failed to simulate portfolio on {date}: {e}")
                continue
                
    def _generate_results(self) -> pd.DataFrame:
        """Generate comprehensive results DataFrame."""
        if self.portfolio is None or not self.portfolio.portfolio_history:
            raise ValueError("No portfolio history available")
            
        # Convert portfolio history to DataFrame
        portfolio_df = pd.DataFrame(self.portfolio.portfolio_history)
        portfolio_df.set_index('date', inplace=True)
        
        # Calculate returns
        portfolio_df['portfolio_return'] = portfolio_df['total_value'].pct_change()
        portfolio_df['cumulative_return'] = (portfolio_df['total_value'] / 
                                           self.config.backtest.initial_capital) - 1
        
        # Add benchmark data if available
        if self.benchmark_data is not None:
            benchmark_aligned = self.benchmark_data.reindex(portfolio_df.index, method='ffill')
            portfolio_df['benchmark_value'] = benchmark_aligned
            portfolio_df['benchmark_return'] = benchmark_aligned.pct_change()
            portfolio_df['benchmark_cumulative_return'] = (benchmark_aligned / 
                                                         self.config.backtest.initial_capital) - 1
            portfolio_df['excess_return'] = portfolio_df['portfolio_return'] - portfolio_df['benchmark_return']
            
        # Calculate rolling statistics
        portfolio_df['rolling_volatility'] = portfolio_df['portfolio_return'].rolling(12).std() * np.sqrt(12)
        portfolio_df['rolling_sharpe'] = (
            (portfolio_df['portfolio_return'].rolling(12).mean() * 12 - self.config.backtest.risk_free_rate) /
            portfolio_df['rolling_volatility']
        )
        
        # Add drawdown calculations
        portfolio_df['running_max'] = portfolio_df['total_value'].expanding().max()
        portfolio_df['drawdown'] = (portfolio_df['total_value'] - portfolio_df['running_max']) / portfolio_df['running_max']
        
        return portfolio_df
        
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        if self.results is None:
            raise ValueError("Must run backtest before calculating metrics")
            
        returns = self.results['portfolio_return'].dropna()
        
        if len(returns) == 0:
            return {}
            
        # Basic metrics
        total_return = self.results['cumulative_return'].iloc[-1]
        annualized_return = (1 + total_return) ** (12 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(12)
        sharpe_ratio = (annualized_return - self.config.backtest.risk_free_rate) / volatility if volatility > 0 else 0
        
        # Drawdown metrics
        max_drawdown = self.results['drawdown'].min()
        avg_drawdown = self.results[self.results['drawdown'] < 0]['drawdown'].mean()
        
        # Win rate
        win_rate = (returns > 0).mean()
        
        # Transaction costs
        total_transaction_costs = sum(state.get('transaction_cost', 0) 
                                    for state in self.portfolio.portfolio_history)
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown if not pd.isna(avg_drawdown) else 0,
            'win_rate': win_rate,
            'total_transaction_costs': total_transaction_costs,
            'final_portfolio_value': self.results['total_value'].iloc[-1],
        }
        
        # Add benchmark comparison if available
        if 'benchmark_return' in self.results.columns:
            benchmark_returns = self.results['benchmark_return'].dropna()
            if len(benchmark_returns) > 0:
                benchmark_total_return = self.results['benchmark_cumulative_return'].iloc[-1]
                benchmark_volatility = benchmark_returns.std() * np.sqrt(12)
                alpha = annualized_return - (self.config.backtest.risk_free_rate + 
                                           (benchmark_total_return - self.config.backtest.risk_free_rate))
                
                # Information ratio
                excess_returns = self.results['excess_return'].dropna()
                if len(excess_returns) > 0 and excess_returns.std() > 0:
                    information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(12)
                else:
                    information_ratio = 0
                
                # Benchmark drawdown analysis
                benchmark_cumulative = (1 + benchmark_returns.fillna(0)).cumprod()
                benchmark_running_max = benchmark_cumulative.expanding().max()
                benchmark_drawdown = (benchmark_cumulative - benchmark_running_max) / benchmark_running_max
                
                benchmark_max_drawdown = benchmark_drawdown.min()
                benchmark_avg_drawdown = benchmark_drawdown[benchmark_drawdown < 0].mean()
                benchmark_avg_drawdown = benchmark_avg_drawdown if not pd.isna(benchmark_avg_drawdown) else 0
                    
                metrics.update({
                    'benchmark_total_return': benchmark_total_return,
                    'benchmark_volatility': benchmark_volatility,
                    'benchmark_max_drawdown': benchmark_max_drawdown,
                    'benchmark_avg_drawdown': benchmark_avg_drawdown,
                    'alpha': alpha,
                    'information_ratio': information_ratio,
                })
                
        return metrics
        
    def get_trade_analysis(self) -> pd.DataFrame:
        """Analyze trading activity."""
        if self.portfolio is None or not self.portfolio.trade_history:
            return pd.DataFrame()
            
        trades_data = []
        for trade in self.portfolio.trade_history:
            trades_data.append({
                'date': trade.date,
                'from_asset': trade.asset_from,
                'to_asset': trade.asset_to,
                'amount': trade.amount,
                'transaction_cost': trade.transaction_cost,
                'portfolio_value_before': trade.portfolio_value_before,
                'portfolio_value_after': trade.portfolio_value_after,
            })
            
        trade_df = pd.DataFrame(trades_data)
        if not trade_df.empty:
            trade_df.set_index('date', inplace=True)
            
        return trade_df