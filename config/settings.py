"""
Enhanced Multi-Agent Portfolio System Configuration
====================================================
Includes regime-based risk aversion, institutional flow tracking,
NLP sentiment analysis, and execution algorithms (TWAP/VWAP).
"""

import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class RegimeType(Enum):
    BULL_LOW_VOL = 0
    NORMAL = 1
    BEAR_HIGH_VOL = 2
    CRISIS = 3
    RECOVERY = 4


@dataclass
class RegimeConfig:
    """Regime-specific configuration parameters"""
    risk_aversion: float  # γ in mean-variance: higher = more conservative
    risk_multiplier: float  # Scales position sizes
    max_position: float  # Maximum single position size
    momentum_weight: float  # Factor weight for momentum
    low_vol_weight: float  # Factor weight for low volatility
    rebalance_threshold: float  # Min change to trigger rebalance
    stop_loss: float  # Stop loss level
    take_profit: float  # Take profit level


@dataclass 
class ExecutionConfig:
    """Execution algorithm configuration"""
    default_algo: str = "TWAP"  # TWAP, VWAP, or ADAPTIVE
    twap_intervals: int = 10  # Number of time slices for TWAP
    vwap_participation_rate: float = 0.10  # Max % of volume
    urgency_factor: float = 1.0  # 1.0 = normal, >1 = faster execution
    max_slippage_bps: float = 25  # Maximum allowed slippage
    min_order_size: float = 100  # Minimum order size in dollars


@dataclass
class InstitutionalFlowConfig:
    """Institutional flow tracking configuration"""
    accumulation_threshold: float = 0.15  # 15% volume spike = accumulation signal
    distribution_threshold: float = -0.15  # -15% = distribution signal
    piggyback_entry_threshold: float = 0.10  # Enter when 10% accumulation detected
    take_profit_pct: float = 0.10  # Exit at 10% profit
    stop_loss_pct: float = 0.05  # Exit at 5% loss
    short_tracking_enabled: bool = True  # Track short positions
    lookback_days: int = 20  # Days to analyze flow patterns
    min_institutional_ratio: float = 0.30  # Min institutional volume ratio


@dataclass
class SentimentConfig:
    """NLP sentiment analysis configuration"""
    enabled: bool = True
    sources: List[str] = field(default_factory=lambda: ["twitter", "reddit", "news", "stocktwits"])
    language: str = "en"  # English only
    lookback_hours: int = 24
    min_mentions: int = 10  # Minimum mentions for valid signal
    sentiment_weight: float = 0.15  # Weight in overall signal
    nuance_learning: bool = True  # Enable context-aware sentiment
    positive_threshold: float = 0.6
    negative_threshold: float = 0.4
    update_frequency_minutes: int = 15


@dataclass
class FuturesConfig:
    """Nasdaq futures positioning configuration"""
    enabled: bool = True
    ticker: str = "NQ=F"  # Nasdaq 100 E-mini futures
    cot_source: str = "CFTC"  # Commitment of Traders data
    positioning_weight: float = 0.20  # Weight in regime assessment
    extreme_long_threshold: float = 0.80  # 80th percentile = extreme long
    extreme_short_threshold: float = 0.20  # 20th percentile = extreme short
    lookback_weeks: int = 52  # 1 year for percentile calculation


@dataclass
class StatisticalConfig:
    """Statistical significance testing configuration"""
    bootstrap_iterations: int = 10000
    bootstrap_block_size: int = 21  # Days for block bootstrap
    confidence_level: float = 0.95
    newey_west_lags: int = 5  # Lags for HAC standard errors
    min_observations: int = 60  # Minimum for valid inference
    multiple_testing_correction: str = "bonferroni"  # or "fdr"


# Default regime configurations
DEFAULT_REGIME_CONFIGS = {
    RegimeType.BULL_LOW_VOL: RegimeConfig(
        risk_aversion=2.0,  # Lower risk aversion in bull markets
        risk_multiplier=1.0,
        max_position=0.30,
        momentum_weight=0.7,
        low_vol_weight=0.3,
        rebalance_threshold=0.03,
        stop_loss=-0.08,
        take_profit=0.15
    ),
    RegimeType.NORMAL: RegimeConfig(
        risk_aversion=3.0,
        risk_multiplier=0.90,
        max_position=0.25,
        momentum_weight=0.5,
        low_vol_weight=0.5,
        rebalance_threshold=0.04,
        stop_loss=-0.06,
        take_profit=0.12
    ),
    RegimeType.BEAR_HIGH_VOL: RegimeConfig(
        risk_aversion=5.0,  # Higher risk aversion in bear markets
        risk_multiplier=0.70,
        max_position=0.20,
        momentum_weight=0.3,
        low_vol_weight=0.7,
        rebalance_threshold=0.05,
        stop_loss=-0.05,
        take_profit=0.10
    ),
    RegimeType.CRISIS: RegimeConfig(
        risk_aversion=8.0,  # Very high risk aversion in crisis
        risk_multiplier=0.50,
        max_position=0.15,
        momentum_weight=0.2,
        low_vol_weight=0.8,
        rebalance_threshold=0.06,
        stop_loss=-0.04,
        take_profit=0.08
    ),
    RegimeType.RECOVERY: RegimeConfig(
        risk_aversion=2.5,  # Moderate risk aversion in recovery
        risk_multiplier=0.85,
        max_position=0.28,
        momentum_weight=0.6,
        low_vol_weight=0.4,
        rebalance_threshold=0.03,
        stop_loss=-0.07,
        take_profit=0.12
    )
}


@dataclass
class SystemConfig:
    """Master system configuration"""
    # Core settings
    universe: List[str] = field(default_factory=lambda: [
        # Technology
        "AAPL", "MSFT", "NVDA", "AMD", "AVGO",
        # Energy
        "XOM", "CVX", "SLB", "MPC", "PSX",
        # Financials
        "JPM", "BAC", "GS", "MS", "WFC",
        # Consumer
        "AMZN", "TSLA", "MCD", "NKE", "SBUX",
        # Healthcare
        "JNJ", "LLY", "MRK", "AMGN", "TMO"
    ])
    benchmark: str = "SPY"
    risk_free_rate: float = 0.05  # Current ~5% Fed funds rate
    initial_capital: float = 1_000_000
    
    # Regime settings
    n_regimes: int = 5
    regime_configs: Dict[RegimeType, RegimeConfig] = field(
        default_factory=lambda: DEFAULT_REGIME_CONFIGS.copy()
    )
    
    # Sub-configs
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    institutional: InstitutionalFlowConfig = field(default_factory=InstitutionalFlowConfig)
    sentiment: SentimentConfig = field(default_factory=SentimentConfig)
    futures: FuturesConfig = field(default_factory=FuturesConfig)
    statistical: StatisticalConfig = field(default_factory=StatisticalConfig)
    
    # Rebalancing
    rebalance_frequency: str = "weekly"  # daily, weekly, monthly
    max_turnover: float = 0.30
    
    # Risk limits
    max_sector_exposure: float = 0.35
    max_drawdown_limit: float = 0.20
    var_limit_95: float = 0.03  # 3% daily VaR limit


def load_config(config_path: Optional[str] = None) -> SystemConfig:
    """Load configuration from YAML file or return defaults"""
    if config_path:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        # Parse and create config object
        return SystemConfig(**config_dict)
    return SystemConfig()


def save_config(config: SystemConfig, config_path: str) -> None:
    """Save configuration to YAML file"""
    import dataclasses
    config_dict = dataclasses.asdict(config)
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
