"""
Command-line interface for GEM strategy.

This module provides a comprehensive CLI for running the Global Equities Momentum
strategy, including backtesting, visualization, and configuration management.
"""

import sys
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta

import click
import pandas as pd
from loguru import logger

from gem.config import GemConfig
from gem.strategy.momentum import GlobalEquitiesMomentum
from gem.strategy.backtest import Backtester
from gem.analysis.metrics import PerformanceMetrics
from gem.analysis.visualization import Visualizer


def setup_logging(log_level: str, log_file: Optional[str] = None) -> None:
    """Set up logging configuration."""
    logger.remove()
    
    # Console logging
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level=log_level.upper(),
        colorize=True
    )
    
    # File logging if specified
    if log_file:
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level="DEBUG",
            rotation="10 MB",
            retention="1 week"
        )


@click.group()
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    help="Set the logging level"
)
@click.option(
    "--log-file",
    type=click.Path(),
    help="Log file path"
)
@click.option(
    "--config-file",
    type=click.Path(exists=True),
    help="Custom configuration file path"
)
@click.pass_context
def cli(ctx: click.Context, log_level: str, log_file: Optional[str], config_file: Optional[str]) -> None:
    """
    GEM (Global Equities Momentum) Strategy CLI.
    
    A comprehensive tool for running momentum-based investment strategies
    with backtesting, analysis, and visualization capabilities.
    """
    # Set up logging
    setup_logging(log_level, log_file)
    
    # Initialize configuration
    config = GemConfig()
    if config_file:
        # TODO: Load custom configuration from file
        logger.info(f"Loading configuration from {config_file}")
    
    # Store in context for subcommands
    ctx.ensure_object(dict)
    ctx.obj['config'] = config
    ctx.obj['log_level'] = log_level


@cli.command()
@click.option(
    "--start-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Start date for data fetching (YYYY-MM-DD)"
)
@click.option(
    "--end-date", 
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="End date for data fetching (YYYY-MM-DD)"
)
@click.option(
    "--clear-cache",
    is_flag=True,
    help="Clear cached data before fetching"
)
@click.pass_context
def fetch_data(
    ctx: click.Context,
    start_date: Optional[datetime],
    end_date: Optional[datetime],
    clear_cache: bool
) -> None:
    """Fetch financial data for strategy assets."""
    config = ctx.obj['config']
    
    try:
        strategy = GlobalEquitiesMomentum(config)
        
        if clear_cache:
            strategy.data_fetcher.clear_cache()
            logger.info("Cache cleared")
        
        logger.info(f"Fetching data from {start_date} to {end_date}")
        strategy.fetch_data(start_date, end_date)
        
        # Show cache statistics
        cache_stats = strategy.data_fetcher.get_cache_stats()
        logger.info(f"Cache statistics: {cache_stats}")
        
        click.echo("✅ Data fetching completed successfully")
        
    except Exception as e:
        logger.error(f"Data fetching failed: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--start-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Backtest start date (YYYY-MM-DD)"
)
@click.option(
    "--end-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Backtest end date (YYYY-MM-DD)"
)
@click.option(
    "--initial-capital",
    type=float,
    help="Initial capital amount"
)
@click.option(
    "--output-file",
    type=click.Path(),
    help="Output file path for results"
)
@click.option(
    "--show-metrics",
    is_flag=True,
    help="Display performance metrics"
)
@click.option(
    "--create-charts",
    is_flag=True,
    help="Create visualization charts"
)
@click.pass_context
def backtest(
    ctx: click.Context,
    start_date: Optional[datetime],
    end_date: Optional[datetime],
    initial_capital: Optional[float],
    output_file: Optional[str],
    show_metrics: bool,
    create_charts: bool
) -> None:
    """Run backtesting for the GEM strategy."""
    config = ctx.obj['config']
    
    try:
        # Initialize backtester
        backtester = Backtester(config)
        
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=365 * 5)  # 5 years default
        
        logger.info(f"Running backtest from {start_date.date()} to {end_date.date()}")
        
        if initial_capital:
            logger.info(f"Using initial capital: ${initial_capital:,.2f}")
        
        # Run backtest
        results_df = backtester.run_backtest(start_date, end_date, initial_capital)
        
        # Save results if requested
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(output_path)
            logger.info(f"Results saved to {output_path}")
        
        # Display metrics
        if show_metrics:
            metrics = backtester.get_performance_metrics()
            
            click.echo("\n📊 STRATEGY PERFORMANCE METRICS")
            click.echo("=" * 55)
            
            # Strategy metrics
            strategy_metrics = {
                'total_return': 'Total Return',
                'annualized_return': 'Annualized Return', 
                'volatility': 'Volatility',
                'sharpe_ratio': 'Sharpe Ratio',
                'max_drawdown': 'Maximum Drawdown',
                'avg_drawdown': 'Average Drawdown',
                'win_rate': 'Win Rate',
                'final_portfolio_value': 'Final Value'
            }
            
            for key, name in strategy_metrics.items():
                if key in metrics:
                    value = metrics[key]
                    if 'return' in key or 'drawdown' in key or 'rate' in key:
                        formatted_value = f"{value:.2%}"
                    elif 'ratio' in key:
                        formatted_value = f"{value:.2f}"
                    elif 'value' in key:
                        formatted_value = f"${value:,.2f}"
                    else:
                        formatted_value = f"{value:.4f}"
                    
                    click.echo(f"{name:.<35} {formatted_value}")
            
            # Benchmark comparison section
            if any(key.startswith('benchmark_') for key in metrics.keys()):
                click.echo("\n📈 BENCHMARK COMPARISON (Buy & Hold SPY)")
                click.echo("=" * 55)
                
                benchmark_metrics = {
                    'benchmark_total_return': 'Benchmark Total Return',
                    'benchmark_volatility': 'Benchmark Volatility',
                    'benchmark_max_drawdown': 'Benchmark Max Drawdown', 
                    'benchmark_avg_drawdown': 'Benchmark Avg Drawdown'
                }
                
                for key, name in benchmark_metrics.items():
                    if key in metrics:
                        value = metrics[key]
                        if 'return' in key or 'drawdown' in key:
                            formatted_value = f"{value:.2%}"
                        elif 'volatility' in key:
                            formatted_value = f"{value:.2%}"
                        else:
                            formatted_value = f"{value:.4f}"
                        
                        click.echo(f"{name:.<35} {formatted_value}")
                
                # Performance comparison
                if 'total_return' in metrics and 'benchmark_total_return' in metrics:
                    excess = metrics['total_return'] - metrics['benchmark_total_return']
                    click.echo(f"{'Excess Return':.<35} {excess:.2%}")
                
                if 'alpha' in metrics:
                    click.echo(f"{'Alpha':.<35} {metrics['alpha']:.2%}")
                
                if 'information_ratio' in metrics:
                    click.echo(f"{'Information Ratio':.<35} {metrics['information_ratio']:.2f}")
                    
                # Risk comparison
                if 'max_drawdown' in metrics and 'benchmark_max_drawdown' in metrics:
                    drawdown_diff = metrics['max_drawdown'] - metrics['benchmark_max_drawdown']
                    better_worse = "better" if drawdown_diff > 0 else "worse"
                    click.echo(f"{'Drawdown vs Benchmark':.<35} {abs(drawdown_diff):.2%} {better_worse}")
        
        # Create charts
        if create_charts:
            visualizer = Visualizer(config)
            
            # Get strategy returns
            strategy_returns = results_df['portfolio_return'].dropna()
            benchmark_returns = results_df.get('benchmark_return')
            
            # Create performance dashboard
            fig = visualizer.create_performance_dashboard(
                strategy_returns,
                benchmark_returns,
                backtester.strategy.allocations[0].__dict__ if backtester.strategy.allocations else None
            )
            
            # Save chart
            chart_path = config.output_directory / "backtest_results.html"
            visualizer.save_figure(fig, "backtest_results", "html")
            click.echo(f"📈 Performance dashboard saved to {chart_path}")
        
        click.echo("✅ Backtest completed successfully")
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Date for allocation calculation (default: uses date_mode setting)"
)
@click.option(
    "--date-mode",
    type=click.Choice(["current", "end_of_month"], case_sensitive=False),
    help="Date calculation mode (overrides config setting)"
)
@click.option(
    "--end-of-month",
    is_flag=True,
    help="Use end-of-month mode (shortcut for --date-mode=end_of_month)"
)
@click.option(
    "--output-format",
    type=click.Choice(["table", "json"], case_sensitive=False),
    default="table",
    help="Output format"
)
@click.pass_context
def allocate(
    ctx: click.Context,
    date: Optional[datetime],
    date_mode: Optional[str],
    end_of_month: bool,
    output_format: str
) -> None:
    """Get current asset allocation recommendation."""
    config = ctx.obj['config']
    
    try:
        # Override date mode if specified via CLI
        if end_of_month:
            date_mode = "end_of_month"
        
        if date_mode:
            # Create a copy of config with overridden date_mode
            config_dict = config.dict()
            config_dict['strategy']['date_mode'] = date_mode
            from gem.config import GemConfig
            config = GemConfig(**config_dict)
        
        strategy = GlobalEquitiesMomentum(config)
        
        # Calculate allocation (date resolution handled internally)
        allocation = strategy.get_current_allocation_live(date)
        
        if output_format == "json":
            import json
            output_data = {
                "date": allocation.date.isoformat(),
                "primary_asset": allocation.primary_asset,
                "signal": allocation.signal.value,
                "allocation": allocation.allocation,
                "momentum_scores": allocation.momentum_scores,
                "date_mode": config.strategy.date_mode
            }
            click.echo(json.dumps(output_data, indent=2))
        else:
            # Show which date mode is being used
            date_mode_info = f"({config.strategy.date_mode} mode)"
            
            click.echo(f"\n🎯 ALLOCATION RECOMMENDATION for {allocation.date.date()} {date_mode_info}")
            click.echo("=" * 55)
            click.echo(f"Signal: {allocation.signal.value.upper()}")
            click.echo(f"Primary Asset: {allocation.primary_asset}")
            
            # Show date resolution info if using end_of_month mode
            if config.strategy.date_mode == "end_of_month" and date:
                click.echo(f"📅 Calculation date: {allocation.date.date()} (resolved from {date.date()})")
            
            click.echo("\n📈 ASSET ALLOCATION:")
            for asset, weight in allocation.allocation.items():
                asset_name = config.assets[asset].name
                click.echo(f"{asset_name:.<40} {weight:.1%}")
            
            click.echo("\n⚡ MOMENTUM SCORES:")
            for asset, score in allocation.momentum_scores.items():
                asset_name = config.assets[asset].name
                click.echo(f"{asset_name:.<40} {score:.2%}")
        
        click.echo("✅ Allocation calculation completed")
        
    except Exception as e:
        logger.error(f"Allocation calculation failed: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--asset",
    multiple=True,
    help="Asset symbols to compare (can be used multiple times)"
)
@click.option(
    "--start-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Start date for comparison"
)
@click.option(
    "--end-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="End date for comparison"
)
@click.option(
    "--save-chart",
    is_flag=True,
    help="Save comparison chart"
)
@click.pass_context
def compare(
    ctx: click.Context,
    asset: tuple,
    start_date: Optional[datetime],
    end_date: Optional[datetime],
    save_chart: bool
) -> None:
    """Compare asset performance."""
    config = ctx.obj['config']
    
    try:
        from gem.data.fetchers import DataFetcher
        
        fetcher = DataFetcher(config)
        visualizer = Visualizer(config)
        
        # Use default assets if none specified
        assets_to_compare = list(asset) if asset else [cfg.symbol for cfg in config.assets.values()]
        
        # Set default dates
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=365 * 3)  # 3 years
        
        # Fetch data and calculate returns
        returns_data = {}
        for symbol in assets_to_compare:
            try:
                returns = fetcher.get_monthly_returns(symbol, start_date, end_date)
                returns_data[symbol] = returns
            except Exception as e:
                logger.warning(f"Failed to fetch data for {symbol}: {e}")
                continue
        
        if not returns_data:
            click.echo("❌ No data available for comparison", err=True)
            sys.exit(1)
        
        # Calculate performance metrics
        perf_metrics = PerformanceMetrics()
        comparison_df = perf_metrics.compare_strategies(returns_data)
        
        # Display comparison table
        click.echo("\n📊 ASSET PERFORMANCE COMPARISON")
        click.echo("=" * 60)
        
        # Select key metrics to display
        key_metrics = [
            'total_return', 'annualized_return', 'volatility', 
            'sharpe_ratio', 'max_drawdown', 'win_rate'
        ]
        
        display_df = comparison_df[key_metrics].copy()
        
        # Format for display
        for col in display_df.columns:
            if 'return' in col or 'drawdown' in col or 'rate' in col:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}")
            elif 'ratio' in col:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}")
            else:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
        
        click.echo(display_df.to_string())
        
        # Create chart if requested
        if save_chart:
            fig = visualizer.plot_cumulative_returns(
                returns_data,
                title="Asset Performance Comparison"
            )
            visualizer.save_figure(fig, "asset_comparison", "html")
            
            chart_path = config.output_directory / "asset_comparison.html"
            click.echo(f"\n📈 Comparison chart saved to {chart_path}")
        
        click.echo("\n✅ Asset comparison completed")
        
    except Exception as e:
        logger.error(f"Asset comparison failed: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def info(ctx: click.Context) -> None:
    """Display configuration and system information."""
    config = ctx.obj['config']
    
    click.echo("\n🔧 CONFIGURATION INFO")
    click.echo("=" * 40)
    
    click.echo(f"Output Directory: {config.output_directory}")
    click.echo(f"Log Level: {ctx.obj['log_level']}")
    
    click.echo(f"\n📈 STRATEGY CONFIG:")
    click.echo(f"Lookback Months: {config.strategy.lookback_months}")
    click.echo(f"Transaction Cost: {config.strategy.transaction_cost:.3%}")
    click.echo(f"Rebalancing: {config.strategy.rebalancing_frequency}")
    click.echo(f"Date Mode: {config.strategy.date_mode}")
    
    click.echo(f"\n💰 ASSET UNIVERSE:")
    for key, asset_config in config.assets.items():
        click.echo(f"{key:.<15} {asset_config.symbol} ({asset_config.name})")
    
    click.echo(f"\n💾 DATA CONFIG:")
    click.echo(f"Cache Enabled: {config.data.cache_enabled}")
    click.echo(f"Cache TTL: {config.data.cache_ttl_hours} hours")
    click.echo(f"Rate Limit: {config.data.rate_limit_seconds} seconds")
    
    click.echo(f"\n🧪 BACKTEST CONFIG:")
    click.echo(f"Initial Capital: ${config.backtest.initial_capital:,.2f}")
    click.echo(f"Benchmark: {config.backtest.benchmark_symbol}")
    click.echo(f"Risk-free Rate: {config.backtest.risk_free_rate:.2%}")


@cli.command()
@click.pass_context  
def version(ctx: click.Context) -> None:
    """Display version information."""
    from gem import __version__
    
    click.echo(f"GEM Strategy v{__version__}")
    click.echo("Global Equities Momentum - Improved Implementation")


def main() -> None:
    """Main entry point for the CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\n\n⚠️  Operation cancelled by user", err=True)
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error occurred")
        click.echo(f"\n❌ Unexpected error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()