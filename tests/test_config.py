"""
Tests for configuration management.
"""

import pytest
from pydantic import ValidationError

from gem.config import GemConfig, AssetConfig, StrategyConfig, DataConfig, BacktestConfig


class TestAssetConfig:
    """Test AssetConfig class."""
    
    def test_asset_config_creation(self):
        """Test creating asset configuration."""
        asset = AssetConfig(
            symbol="SPY",
            name="S&P 500 ETF",
            asset_class="equity",
            region="US"
        )
        
        assert asset.symbol == "SPY"
        assert asset.name == "S&P 500 ETF"
        assert asset.asset_class == "equity"
        assert asset.region == "US"
    
    def test_asset_config_validation(self):
        """Test asset configuration validation."""
        # Missing required fields should raise error
        with pytest.raises(ValidationError):
            AssetConfig()


class TestStrategyConfig:
    """Test StrategyConfig class."""
    
    def test_strategy_config_defaults(self):
        """Test strategy configuration defaults."""
        config = StrategyConfig()
        
        assert config.lookback_months == 12
        assert config.rebalancing_frequency == "monthly"
        assert config.transaction_cost == 0.001
        assert config.minimum_momentum_threshold == 0.0
    
    def test_lookback_months_validation(self):
        """Test lookback months validation."""
        # Valid range
        config = StrategyConfig(lookback_months=6)
        assert config.lookback_months == 6
        
        config = StrategyConfig(lookback_months=24)
        assert config.lookback_months == 24
        
        # Invalid range should raise error
        with pytest.raises(ValidationError):
            StrategyConfig(lookback_months=0)
        
        with pytest.raises(ValidationError):
            StrategyConfig(lookback_months=25)
    
    def test_rebalancing_frequency_validation(self):
        """Test rebalancing frequency validation."""
        # Valid frequencies
        for freq in ["monthly", "quarterly", "annually"]:
            config = StrategyConfig(rebalancing_frequency=freq)
            assert config.rebalancing_frequency == freq
        
        # Invalid frequency should raise error
        with pytest.raises(ValidationError):
            StrategyConfig(rebalancing_frequency="daily")
    
    def test_transaction_cost_validation(self):
        """Test transaction cost validation."""
        # Valid range
        config = StrategyConfig(transaction_cost=0.0)
        assert config.transaction_cost == 0.0
        
        config = StrategyConfig(transaction_cost=0.1)
        assert config.transaction_cost == 0.1
        
        # Invalid range should raise error
        with pytest.raises(ValidationError):
            StrategyConfig(transaction_cost=-0.1)
        
        with pytest.raises(ValidationError):
            StrategyConfig(transaction_cost=0.2)
    
    def test_date_mode_validation(self):
        """Test date mode validation."""
        # Valid modes
        config = StrategyConfig(date_mode="current")
        assert config.date_mode == "current"
        
        config = StrategyConfig(date_mode="end_of_month")
        assert config.date_mode == "end_of_month"
        
        # Invalid mode should raise error
        with pytest.raises(ValidationError, match="Date mode must be one of"):
            StrategyConfig(date_mode="invalid_mode")
    
    def test_date_mode_default(self):
        """Test that date_mode defaults to end_of_month."""
        config = StrategyConfig()
        assert config.date_mode == "end_of_month"


class TestDataConfig:
    """Test DataConfig class."""
    
    def test_data_config_defaults(self):
        """Test data configuration defaults."""
        config = DataConfig()
        
        assert config.primary_source == "yfinance"
        assert config.backup_sources == []
        assert config.cache_enabled is True
        assert config.cache_ttl_hours == 24
        assert config.rate_limit_seconds == 1.0


class TestBacktestConfig:
    """Test BacktestConfig class."""
    
    def test_backtest_config_defaults(self):
        """Test backtest configuration defaults."""
        config = BacktestConfig()
        
        assert config.start_date is None
        assert config.end_date is None
        assert config.initial_capital == 10000.0
        assert config.benchmark_symbol == "SPY"
        assert config.risk_free_rate == 0.02
    
    def test_initial_capital_validation(self):
        """Test initial capital validation."""
        # Valid value
        config = BacktestConfig(initial_capital=1000.0)
        assert config.initial_capital == 1000.0
        
        # Invalid value should raise error
        with pytest.raises(ValidationError):
            BacktestConfig(initial_capital=0.0)
        
        with pytest.raises(ValidationError):
            BacktestConfig(initial_capital=-1000.0)


class TestGemConfig:
    """Test main GemConfig class."""
    
    def test_gem_config_creation(self):
        """Test creating main configuration."""
        config = GemConfig()
        
        # Check asset configurations
        assert "us_equity" in config.assets
        assert "intl_equity" in config.assets
        assert "bonds" in config.assets
        assert "cash" in config.assets
        
        # Check nested configurations
        assert isinstance(config.strategy, StrategyConfig)
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.backtest, BacktestConfig)
        
        # Check defaults
        assert config.log_level == "INFO"
        assert config.output_directory.name == "output"
    
    def test_asset_symbols(self):
        """Test default asset symbols."""
        config = GemConfig()
        
        assert config.assets["us_equity"].symbol == "SPY"
        assert config.assets["intl_equity"].symbol == "VEU"
        assert config.assets["bonds"].symbol == "AGG"
        assert config.assets["cash"].symbol == "BIL"
    
    def test_output_directory_creation(self):
        """Test that output directory is created."""
        config = GemConfig()
        assert config.output_directory.exists()
        assert config.output_directory.is_dir()
    
    def test_configuration_override(self):
        """Test configuration override capabilities."""
        # Test direct configuration override
        config = GemConfig(
            log_level="DEBUG",
            strategy={"lookback_months": 6, "date_mode": "current"},
            data={"cache_enabled": False}
        )
        
        assert config.log_level == "DEBUG"
        assert config.strategy.lookback_months == 6
        assert config.strategy.date_mode == "current"
        assert config.data.cache_enabled is False