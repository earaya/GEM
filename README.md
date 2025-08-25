# GEM - Global Equities Momentum Strategy (Improved)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modern, professional Python implementation of Gary Antonacci's **Global Equities Momentum (GEM)** strategy with comprehensive backtesting, analysis, and visualization capabilities.

## 🚀 Features

### Core Strategy
- **Dual Momentum Approach**: Combines absolute and relative momentum indicators
- **Monthly Rebalancing**: End-of-month allocation decisions
- **Multi-Asset Universe**: US Equities, International Equities, Bonds, and Cash
- **Risk Management**: Transaction cost modeling and drawdown analysis

### Technical Excellence
- **Modern Python**: Type hints, comprehensive error handling, and PEP8 compliance
- **Robust Data Handling**: Multiple data sources with caching and rate limiting
- **Comprehensive Testing**: 90%+ test coverage with pytest
- **Interactive Visualization**: Professional charts with Plotly
- **CLI Interface**: Full-featured command-line interface
- **Extensible Architecture**: Modular design for easy customization

### Analysis & Backtesting
- **Performance Metrics**: Sharpe ratio, Calmar ratio, maximum drawdown, and more
- **Risk Analysis**: VaR, CVaR, downside deviation, and rolling statistics
- **Benchmark Comparison**: Alpha, beta, information ratio, and tracking error
- **Transaction Costs**: Realistic cost modeling with slippage
- **Interactive Dashboards**: Comprehensive performance visualization

## 📈 The GEM Strategy

The Global Equities Momentum strategy follows Gary Antonacci's dual momentum approach:

1. **Absolute Momentum**: Compare equity performance against cash/bonds
2. **Relative Momentum**: If equities are favored, choose between US and international markets

### Decision Logic
```
IF best_equity_momentum > cash_momentum:
    IF us_equity_momentum > international_momentum:
        → Invest in US Equities
    ELSE:
        → Invest in International Equities
ELSE:
    → Invest in Bonds (defensive allocation)
```

## 🛠️ Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Install from Source
```bash
git clone https://github.com/your-username/gem-momentum.git
cd gem-momentum
pip install -e .
```

### Development Installation
```bash
git clone https://github.com/your-username/gem-momentum.git
cd gem-momentum
pip install -e .[dev]
```

## 🚀 Quick Start

### Command Line Interface

#### 1. Get Current Allocation Recommendation
```bash
gem allocate
```

#### 2. Run Backtesting
```bash
gem backtest --start-date 2010-01-01 --show-metrics --create-charts
```

#### 3. Compare Asset Performance  
```bash
gem compare --asset SPY --asset VEU --asset AGG --save-chart
```

#### 4. Fetch and Cache Data
```bash
gem fetch-data --start-date 2020-01-01 --clear-cache
```

### Python API

#### Basic Strategy Usage
```python
from gem import GlobalEquitiesMomentum, GemConfig
from datetime import datetime, timedelta

# Initialize with default configuration
config = GemConfig()
strategy = GlobalEquitiesMomentum(config)

# Fetch data and run strategy
end_date = datetime.now()
start_date = end_date - timedelta(days=365*5)  # 5 years

strategy.fetch_data(start_date, end_date)
allocations_df = strategy.run_strategy(start_date, end_date)

# Get current allocation
current_allocation = strategy.get_current_allocation()
print(f\"Primary Asset: {current_allocation.primary_asset}\")
print(f\"Signal: {current_allocation.signal.value}\")
```

#### Backtesting Example
```python
from gem.strategy.backtest import Backtester
from gem.analysis.visualization import Visualizer

# Run backtest
backtester = Backtester(config)
results = backtester.run_backtest(
    start_date=\"2010-01-01\",
    end_date=\"2023-12-31\",
    initial_capital=100000
)

# Get performance metrics
metrics = backtester.get_performance_metrics()
print(f\"Total Return: {metrics['total_return']:.2%}\")
print(f\"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\")
print(f\"Max Drawdown: {metrics['max_drawdown']:.2%}\")

# Create visualization
visualizer = Visualizer(config)
dashboard = visualizer.create_performance_dashboard(
    strategy_returns=results['portfolio_return'],
    benchmark_returns=results.get('benchmark_return')
)
dashboard.show()
```

#### Custom Configuration
```python
from gem.config import GemConfig, StrategyConfig

# Customize strategy parameters
config = GemConfig()
config.strategy.lookback_months = 6  # 6-month momentum
config.strategy.transaction_cost = 0.002  # 0.2% transaction cost
config.backtest.initial_capital = 50000

# Add custom assets
from gem.config import AssetConfig
config.assets['emerging_markets'] = AssetConfig(
    symbol=\"VWO\",
    name=\"Emerging Markets ETF\",
    asset_class=\"equity\",
    region=\"Emerging\"
)
```

## 📊 Analysis & Visualization

### Performance Dashboard
```python
from gem.analysis.visualization import Visualizer

visualizer = Visualizer(config)

# Create comprehensive dashboard
dashboard = visualizer.create_performance_dashboard(
    strategy_returns=strategy_returns,
    benchmark_returns=benchmark_returns,
    allocation_df=allocations_df
)

# Save as HTML
visualizer.save_figure(dashboard, \"performance_dashboard\", \"html\")
```

### Risk Analysis
```python
from gem.analysis.metrics import PerformanceMetrics

metrics_calc = PerformanceMetrics()

# Calculate comprehensive metrics
metrics = metrics_calc.calculate_returns_metrics(strategy_returns)
risk_metrics = metrics_calc.calculate_risk_metrics(strategy_returns)

# Compare multiple strategies
comparison = metrics_calc.compare_strategies({
    'GEM Strategy': gem_returns,
    'Buy & Hold SPY': spy_returns,
    'Buy & Hold Bonds': agg_returns
})
```

## ⚙️ Configuration

The strategy supports extensive configuration through environment variables or direct configuration:

### Environment Variables
```bash
export GEM_LOG_LEVEL=DEBUG
export GEM_STRATEGY__LOOKBACK_MONTHS=6
export GEM_DATA__CACHE_ENABLED=true
export GEM_BACKTEST__INITIAL_CAPITAL=100000
```

### Configuration File
```python
# config.py
from gem.config import GemConfig

config = GemConfig()

# Strategy settings
config.strategy.lookback_months = 12
config.strategy.transaction_cost = 0.001
config.strategy.rebalancing_frequency = \"monthly\"

# Data settings  
config.data.cache_enabled = True
config.data.cache_ttl_hours = 24
config.data.rate_limit_seconds = 1.0

# Backtest settings
config.backtest.initial_capital = 10000
config.backtest.benchmark_symbol = \"SPY\"
config.backtest.risk_free_rate = 0.02
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=gem --cov-report=html

# Run specific test categories
pytest tests/test_strategy_momentum.py -v
pytest tests/test_data_fetchers.py -v
```

## 📈 Performance Metrics

The strategy provides comprehensive performance analysis including:

### Return Metrics
- Total Return
- Annualized Return  
- Volatility (annualized)
- Sharpe Ratio
- Sortino Ratio

### Risk Metrics
- Maximum Drawdown
- Value at Risk (VaR)
- Conditional VaR (CVaR)
- Calmar Ratio
- Skewness & Kurtosis

### Benchmark Comparison
- Alpha & Beta
- Information Ratio
- Tracking Error
- Win Rate

## 🔧 Development

### Development Setup
```bash
git clone https://github.com/your-username/gem-momentum.git
cd gem-momentum
pip install -e .[dev]
pre-commit install
```

### Code Quality
```bash
# Format code
black gem/ tests/
isort gem/ tests/

# Type checking
mypy gem/

# Linting
flake8 gem/ tests/
```

### Running Tests
```bash
pytest --cov=gem --cov-report=term-missing
```

## 📚 Background & Research

The GEM strategy is based on Gary Antonacci's research on dual momentum investing:

- **\"Dual Momentum Investing\"** by Gary Antonacci
- **\"Risk Premia Harvesting Through Dual Momentum\"** (2013)
- Focus on absolute momentum (trend following) and relative momentum (cross-sectional)

### Key Advantages
- **Trend Following**: Captures sustained price movements
- **Risk Management**: Defensive allocation during bear markets  
- **Simplicity**: Clear, objective decision rules
- **Diversification**: Multi-asset approach reduces concentration risk

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**This software is for educational and research purposes only.** Past performance does not guarantee future results. All investments carry risk of loss. Please consult with a qualified financial advisor before making investment decisions.

The authors and contributors are not responsible for any financial losses incurred through the use of this software.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/gem-momentum/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/gem-momentum/discussions)
- **Email**: support@gem-momentum.com

---

**Made with ❤️ for quantitative investors and Python enthusiasts.**