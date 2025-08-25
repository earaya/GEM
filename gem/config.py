"""
Configuration management for GEM strategy.

This module provides configuration classes using Pydantic for type safety
and validation of strategy parameters and data sources.
"""

from typing import Dict, List, Optional
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings


class AssetConfig(BaseModel):
    """Configuration for financial assets."""
    
    symbol: str = Field(..., description="Asset symbol/ticker")
    name: str = Field(..., description="Human-readable asset name")
    asset_class: str = Field(..., description="Asset class (equity, bond, cash)")
    region: Optional[str] = Field(None, description="Geographic region")


class StrategyConfig(BaseModel):
    """Configuration for momentum strategy parameters."""
    
    lookback_months: int = Field(12, ge=1, le=24, description="Momentum lookback period in months")
    rebalancing_frequency: str = Field("monthly", description="Rebalancing frequency")
    transaction_cost: float = Field(0.001, ge=0, le=0.1, description="Transaction cost as decimal")
    minimum_momentum_threshold: float = Field(0.0, description="Minimum momentum threshold")
    date_mode: str = Field("end_of_month", description="Date calculation mode: 'current' or 'end_of_month'")
    
    @field_validator("rebalancing_frequency")
    @classmethod
    def validate_rebalancing_frequency(cls, v: str) -> str:
        allowed_frequencies = ["monthly", "quarterly", "annually"]
        if v not in allowed_frequencies:
            raise ValueError(f"Rebalancing frequency must be one of {allowed_frequencies}")
        return v
    
    @field_validator("date_mode")
    @classmethod
    def validate_date_mode(cls, v: str) -> str:
        allowed_modes = ["current", "end_of_month"]
        if v not in allowed_modes:
            raise ValueError(f"Date mode must be one of {allowed_modes}")
        return v


class DataConfig(BaseModel):
    """Configuration for data sources and caching."""
    
    primary_source: str = Field("yfinance", description="Primary data source")
    backup_sources: List[str] = Field(default_factory=list, description="Backup data sources")
    cache_enabled: bool = Field(True, description="Enable data caching")
    cache_ttl_hours: int = Field(24, ge=1, description="Cache TTL in hours")
    rate_limit_seconds: float = Field(1.0, ge=0.1, description="Rate limit between requests")
    
    
class BacktestConfig(BaseModel):
    """Configuration for backtesting parameters."""
    
    start_date: Optional[str] = Field(None, description="Backtest start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="Backtest end date (YYYY-MM-DD)")
    initial_capital: float = Field(10000.0, gt=0, description="Initial capital")
    benchmark_symbol: str = Field("SPY", description="Benchmark symbol")
    risk_free_rate: float = Field(0.02, ge=0, description="Annual risk-free rate")


class GemConfig(BaseSettings):
    """Main configuration class for GEM strategy."""
    
    # Asset definitions
    assets: Dict[str, AssetConfig] = Field(
        default_factory=lambda: {
            "us_equity": AssetConfig(
                symbol="SPY",
                name="S&P 500 ETF",
                asset_class="equity",
                region="US"
            ),
            "intl_equity": AssetConfig(
                symbol="VEU", 
                name="FTSE Developed Markets ETF",
                asset_class="equity",
                region="International"
            ),
            "bonds": AssetConfig(
                symbol="AGG",
                name="Core U.S. Aggregate Bond ETF", 
                asset_class="bond",
                region="US"
            ),
            "cash": AssetConfig(
                symbol="BIL",
                name="1-3 Month Treasury Bill ETF",
                asset_class="cash",
                region="US"
            )
        }
    )
    
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    
    # Application settings
    log_level: str = Field("INFO", description="Logging level")
    output_directory: Path = Field(Path("output"), description="Output directory for results")
    
    model_config = ConfigDict(
        env_prefix="GEM_",
        case_sensitive=False
    )
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create output directory if it doesn't exist
        self.output_directory.mkdir(parents=True, exist_ok=True)


# Global configuration instance
config = GemConfig()