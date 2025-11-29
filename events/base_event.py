"""
Base Event Classes
==================
Dataclass definitions for all events in the multi-agent system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Literal, Any
import pandas as pd
import numpy as np
import uuid


@dataclass
class Event:
    """Base event class for all agent communications."""
    timestamp: datetime = field(default_factory=datetime.now)
    source_agent: str = ""
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize event for logging."""
        return {
            'event_type': self.__class__.__name__,
            'timestamp': self.timestamp.isoformat(),
            'source_agent': self.source_agent,
            'correlation_id': self.correlation_id
        }


@dataclass
class MarketDataEvent(Event):
    """
    Published by Market Data Agent.
    Contains preprocessed price data for the asset universe.
    """
    prices: Optional[pd.DataFrame] = None
    returns: Optional[pd.DataFrame] = None
    asset_universe: List[str] = field(default_factory=list)
    sector_mapping: Dict[str, List[str]] = field(default_factory=dict)
    data_quality_score: float = 1.0
    missing_tickers: List[str] = field(default_factory=list)
    date_range: tuple = ("", "")
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            'n_assets': len(self.asset_universe),
            'data_quality_score': self.data_quality_score,
            'date_range': self.date_range,
            'missing_tickers': self.missing_tickers
        })
        return base


@dataclass
class RegimeEvent(Event):
    """
    Published by ML Agent (HMM Regime Detector).
    Contains current market regime and transition probabilities.
    """
    current_regime: int = 0
    regime_name: str = "Unknown"
    regime_probabilities: Dict[int, float] = field(default_factory=dict)
    transition_matrix: Optional[np.ndarray] = None
    regime_duration: int = 0  # Days in current regime
    regime_history: Optional[pd.Series] = None
    model_confidence: float = 0.0
    features_used: List[str] = field(default_factory=list)
    
    # Regime interpretations
    REGIME_NAMES = {0: "Bull/LowVol", 1: "Transition", 2: "Bear/HighVol"}
    
    def get_regime_name(self) -> str:
        return self.REGIME_NAMES.get(self.current_regime, "Unknown")
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            'current_regime': self.current_regime,
            'regime_name': self.get_regime_name(),
            'regime_probabilities': self.regime_probabilities,
            'regime_duration': self.regime_duration,
            'model_confidence': self.model_confidence
        })
        return base


@dataclass
class QuantSignalEvent(Event):
    """
    Published by Quant Agent (MS-VECM).
    Contains expected returns and covariance estimates.
    """
    expected_returns: Optional[pd.Series] = None
    covariance_matrix: Optional[pd.DataFrame] = None
    cointegration_rank: int = 0
    is_cointegrated: bool = False
    forecast_horizon: int = 5
    forecast_confidence: float = 0.0
    asset_classes: List[str] = field(default_factory=list)
    methodology: str = "MS-VECM"
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            'cointegration_rank': self.cointegration_rank,
            'is_cointegrated': self.is_cointegrated,
            'forecast_horizon': self.forecast_horizon,
            'forecast_confidence': self.forecast_confidence,
            'methodology': self.methodology,
            'n_asset_classes': len(self.asset_classes)
        })
        return base


@dataclass
class RiskAdjustedSignalEvent(Event):
    """
    Published by Risk Manager Agent.
    Contains risk-adjusted signals and position constraints.
    """
    risk_adjusted_returns: Optional[pd.Series] = None
    position_limits: Dict[str, float] = field(default_factory=dict)
    risk_budget: float = 1.0
    max_leverage: float = 1.0
    
    # Risk Metrics
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    expected_shortfall: float = 0.0
    current_drawdown: float = 0.0
    max_drawdown_limit: float = 0.15
    
    # Correlation Regime
    avg_correlation: float = 0.0
    correlation_regime: str = "Normal"
    
    # Alerts and Constraints
    alerts: List[str] = field(default_factory=list)
    circuit_breaker_active: bool = False
    regime_risk_multiplier: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            'risk_budget': self.risk_budget,
            'max_leverage': self.max_leverage,
            'var_95': self.var_95,
            'cvar_95': self.cvar_95,
            'current_drawdown': self.current_drawdown,
            'alerts': self.alerts,
            'circuit_breaker_active': self.circuit_breaker_active
        })
        return base


@dataclass
class TradeOrder:
    """Individual trade order."""
    ticker: str
    action: Literal["BUY", "SELL", "HOLD"]
    current_weight: float
    target_weight: float
    delta_weight: float = 0.0
    urgency: Literal["HIGH", "MEDIUM", "LOW"] = "MEDIUM"
    reason: str = ""
    estimated_cost: float = 0.0
    
    def __post_init__(self):
        self.delta_weight = self.target_weight - self.current_weight


@dataclass
class TradeSignalEvent(Event):
    """
    Published by Portfolio Agent.
    Contains final trade signals (BUY/SELL/HOLD).
    """
    target_weights: Dict[str, float] = field(default_factory=dict)
    detailed_weights: Dict[str, float] = field(default_factory=dict)
    current_weights: Dict[str, float] = field(default_factory=dict)
    trades: List[TradeOrder] = field(default_factory=list)
    rebalance_reason: str = ""
    total_turnover: float = 0.0
    estimated_transaction_cost: float = 0.0
    
    # Optimization metadata
    optimization_method: str = "Black-Litterman"
    objective: str = "Max Sharpe"
    constraints_satisfied: bool = True
    
    def get_buy_orders(self) -> List[TradeOrder]:
        return [t for t in self.trades if t.action == "BUY"]
    
    def get_sell_orders(self) -> List[TradeOrder]:
        return [t for t in self.trades if t.action == "SELL"]
    
    def get_hold_orders(self) -> List[TradeOrder]:
        return [t for t in self.trades if t.action == "HOLD"]
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            'n_trades': len(self.trades),
            'n_buys': len(self.get_buy_orders()),
            'n_sells': len(self.get_sell_orders()),
            'total_turnover': self.total_turnover,
            'estimated_cost': self.estimated_transaction_cost,
            'rebalance_reason': self.rebalance_reason
        })
        return base


@dataclass
class AgentStatusEvent(Event):
    """
    Status update from any agent.
    Used for monitoring and orchestration.
    """
    agent_name: str = ""
    status: Literal["READY", "RUNNING", "COMPLETED", "ERROR", "PAUSED"] = "READY"
    message: str = ""
    progress: float = 0.0  # 0-100
    error_details: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            'agent_name': self.agent_name,
            'status': self.status,
            'message': self.message,
            'progress': self.progress
        })
        return base
