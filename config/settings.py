"""
Agent Portfolio System Configuration
=====================================
Full universe with 5 sectors, bonds, commodities, and all parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import yaml
import os


class RegimeType(Enum):
    """Market regime types"""
    BULL = 0
    TRANSITION = 1
    BEAR = 2


@dataclass
class MarketDataConfig:
    """Market data configuration"""
    start_date: str = '2015-01-01'
    end_date: str = '2025-11-30'
    cache_dir: str = './data/cache'
    cache_ttl_hours: int = 24
    min_data_quality: float = 0.5
    
    # Full universe - 5 sectors with 8 stocks each (40 stocks)
    sectors: Dict[str, List[str]] = field(default_factory=lambda: {
        'Technology': ['AAPL', 'MSFT', 'NVDA', 'AVGO', 'ORCL', 'ADBE', 'CRM', 'AMD'],
        'Healthcare': ['LLY', 'UNH', 'JNJ', 'MRK', 'ABBV', 'TMO', 'AMGN', 'PFE'],
        'Financials': ['JPM', 'V', 'MA', 'BAC', 'WFC', 'MS', 'GS', 'AXP'],
        'Consumer': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'LOW', 'SBUX', 'TGT'],
        'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'PSX', 'OXY']
    })
    
    # Fixed income
    bonds: List[str] = field(default_factory=lambda: ['TLT', 'IEF', 'SHY', 'LQD', 'HYG'])
    
    # Commodities
    commodities: List[str] = field(default_factory=lambda: ['GLD', 'USO', 'SLV', 'DBC'])
    
    # Others
    others: List[str] = field(default_factory=lambda: ['VNQ', 'SPY'])
    
    @property
    def full_universe(self) -> List[str]:
        """Get complete list of all tickers"""
        all_tickers = []
        for sector_tickers in self.sectors.values():
            all_tickers.extend(sector_tickers)
        all_tickers.extend(self.bonds)
        all_tickers.extend(self.commodities)
        all_tickers.extend(self.others)
        return all_tickers


@dataclass
class MLConfig:
    """ML Agent (HMM) configuration"""
    n_components: int = 3  # Number of HMM states
    covariance_type: str = 'full'
    n_iter: int = 1000
    feature_window: int = 10
    min_observations: int = 100
    random_state: int = 42


@dataclass 
class QuantConfig:
    """Quant Agent (MS-VECM) configuration"""
    lookback_window: int = 252
    forecast_horizon: int = 5
    det_order: int = -1  # Johansen test deterministic order
    k_ar_diff: int = 1   # VECM lag order
    use_shrinkage: bool = True
    regime_adjustment: bool = True
    shrinkage_intensity: float = 0.2


@dataclass
class RegimeRiskParams:
    """Risk parameters for a specific regime"""
    risk_multiplier: float = 1.0
    max_position: float = 0.30
    max_leverage: float = 1.0
    slippage_bps: int = 5
    stop_loss: float = 0.05
    take_profit: float = 0.15
    risk_aversion: float = 3.0


@dataclass
class RiskConfig:
    """Risk Manager configuration"""
    max_drawdown_limit: float = 0.20
    var_confidence: float = 0.95
    var_limit: float = 0.05
    position_sizing_method: str = 'risk_parity'
    max_position: float = 0.30
    max_leverage: float = 1.0
    max_sector_exposure: float = 0.35
    
    # Regime-specific parameters
    regime_params: Dict[int, RegimeRiskParams] = field(default_factory=lambda: {
        0: RegimeRiskParams(  # Bull
            risk_multiplier=1.0,
            max_position=0.30,
            max_leverage=1.0,
            slippage_bps=5,
            stop_loss=0.08,
            take_profit=0.20,
            risk_aversion=2.0
        ),
        1: RegimeRiskParams(  # Transition
            risk_multiplier=0.90,
            max_position=0.25,
            max_leverage=1.0,
            slippage_bps=10,
            stop_loss=0.06,
            take_profit=0.15,
            risk_aversion=3.5
        ),
        2: RegimeRiskParams(  # Bear
            risk_multiplier=0.70,
            max_position=0.20,
            max_leverage=0.8,
            slippage_bps=20,
            stop_loss=0.04,
            take_profit=0.10,
            risk_aversion=6.0
        )
    })


@dataclass
class PortfolioConfig:
    """Portfolio Agent configuration"""
    use_black_litterman: bool = True
    rebalance_threshold: float = 0.03
    max_turnover: float = 0.40
    transaction_cost_bps: int = 10
    risk_free_rate: float = 0.05
    min_weight: float = 0.0
    max_weight: float = 0.30


@dataclass
class ExecutionConfig:
    """Execution configuration"""
    default_algo: str = 'TWAP'  # TWAP, VWAP, ADAPTIVE
    twap_intervals: int = 10
    vwap_participation_rate: float = 0.10
    max_slippage_bps: int = 25
    urgency_factor: float = 1.0


@dataclass
class InstitutionalFlowConfig:
    """Institutional flow tracking configuration"""
    enabled: bool = True
    accumulation_threshold: float = 0.15
    distribution_threshold: float = -0.15
    piggyback_entry_threshold: float = 0.10
    take_profit_pct: float = 0.10
    stop_loss_pct: float = 0.05
    lookback_days: int = 20


@dataclass
class SentimentConfig:
    """NLP Sentiment configuration"""
    enabled: bool = True
    sources: List[str] = field(default_factory=lambda: ['twitter', 'reddit', 'news'])
    language: str = 'en'
    lookback_hours: int = 24
    min_mentions: int = 10
    sentiment_weight: float = 0.15


@dataclass
class StatisticalConfig:
    """Statistical validation configuration"""
    bootstrap_iterations: int = 10000
    bootstrap_block_size: int = 21
    confidence_level: float = 0.95
    newey_west_lags: int = 5
    min_observations: int = 60


@dataclass
class BacktestConfig:
    """Backtest configuration"""
    initial_capital: float = 1_000_000
    start_date: str = '2020-01-01'
    end_date: str = '2025-11-30'
    rebalance_freq: str = 'M'  # D, W, M
    benchmark: str = 'SPY'
    include_costs: bool = True


@dataclass
class DashboardConfig:
    """Dashboard configuration"""
    port: int = 8501
    host: str = 'localhost'
    theme: str = 'light'
    auto_refresh_seconds: int = 60
    show_realtime: bool = True
    show_backtest: bool = True


@dataclass
class SystemConfig:
    """Complete system configuration"""
    market_data: MarketDataConfig = field(default_factory=MarketDataConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    institutional: InstitutionalFlowConfig = field(default_factory=InstitutionalFlowConfig)
    sentiment: SentimentConfig = field(default_factory=SentimentConfig)
    statistical: StatisticalConfig = field(default_factory=StatisticalConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    
    # Orchestrator
    timeout_seconds: int = 60
    retry_attempts: int = 3
    
    @property
    def universe(self) -> List[str]:
        return self.market_data.full_universe
    
    @property
    def n_assets(self) -> int:
        return len(self.universe)


def load_config(config_path: str = 'config/agent_config.yaml') -> SystemConfig:
    """Load configuration from YAML file"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        # Parse into SystemConfig
        config = SystemConfig()
        
        if 'market_data' in yaml_config:
            md = yaml_config['market_data']
            config.market_data = MarketDataConfig(
                start_date=md.get('start_date', config.market_data.start_date),
                end_date=md.get('end_date', config.market_data.end_date),
                sectors=md.get('sectors', config.market_data.sectors),
                bonds=md.get('bonds', config.market_data.bonds),
                commodities=md.get('commodities', config.market_data.commodities)
            )
        
        if 'ml' in yaml_config:
            ml = yaml_config['ml']
            config.ml = MLConfig(
                n_components=ml.get('n_components', config.ml.n_components),
                covariance_type=ml.get('covariance_type', config.ml.covariance_type),
                n_iter=ml.get('n_iter', config.ml.n_iter)
            )
        
        if 'backtest' in yaml_config:
            bt = yaml_config['backtest']
            config.backtest = BacktestConfig(
                initial_capital=bt.get('initial_capital', config.backtest.initial_capital),
                start_date=bt.get('start_date', config.backtest.start_date),
                rebalance_freq=bt.get('rebalance_freq', config.backtest.rebalance_freq)
            )
        
        return config
    else:
        return SystemConfig()


def save_config(config: SystemConfig, config_path: str = 'config/agent_config.yaml'):
    """Save configuration to YAML file"""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    yaml_config = {
        'market_data': {
            'start_date': config.market_data.start_date,
            'end_date': config.market_data.end_date,
            'sectors': config.market_data.sectors,
            'bonds': config.market_data.bonds,
            'commodities': config.market_data.commodities
        },
        'ml': {
            'n_components': config.ml.n_components,
            'covariance_type': config.ml.covariance_type,
            'n_iter': config.ml.n_iter
        },
        'quant': {
            'lookback_window': config.quant.lookback_window,
            'forecast_horizon': config.quant.forecast_horizon,
            'use_shrinkage': config.quant.use_shrinkage
        },
        'risk': {
            'max_drawdown_limit': config.risk.max_drawdown_limit,
            'var_confidence': config.risk.var_confidence,
            'max_position': config.risk.max_position
        },
        'portfolio': {
            'use_black_litterman': config.portfolio.use_black_litterman,
            'rebalance_threshold': config.portfolio.rebalance_threshold,
            'transaction_cost_bps': config.portfolio.transaction_cost_bps,
            'risk_free_rate': config.portfolio.risk_free_rate
        },
        'backtest': {
            'initial_capital': config.backtest.initial_capital,
            'start_date': config.backtest.start_date,
            'end_date': config.backtest.end_date,
            'rebalance_freq': config.backtest.rebalance_freq,
            'benchmark': config.backtest.benchmark
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(yaml_config, f, default_flow_style=False)


if __name__ == "__main__":
    config = SystemConfig()
    print(f"Total Universe: {config.n_assets} assets")
    print(f"Sectors: {list(config.market_data.sectors.keys())}")
    for sector, tickers in config.market_data.sectors.items():
        print(f"  {sector}: {len(tickers)} stocks - {tickers}")
    print(f"Bonds: {config.market_data.bonds}")
    print(f"Commodities: {config.market_data.commodities}")
