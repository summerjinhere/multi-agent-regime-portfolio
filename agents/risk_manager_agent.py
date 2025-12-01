"""
Risk Manager Agent (FIXED)
==========================
Central risk management and position sizing.

Key Fixes:
1. Higher minimum risk_budget (0.95 instead of 0.2)
2. Less aggressive drawdown penalties
3. Regime multipliers that don't over-reduce allocation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Type, Tuple
from scipy import stats
from dataclasses import dataclass

from agents.base_agent import BaseAgent
from events.message_bus import MessageBus
from events.base_event import (
    Event, MarketDataEvent, RegimeEvent, 
    QuantSignalEvent, RiskAdjustedSignalEvent
)


@dataclass
class RiskLimits:
    """Risk limits configuration."""
    max_position: float = 0.30          # Max weight per position (increased)
    max_sector_exposure: float = 0.50   # Max weight per sector (increased)
    max_drawdown: float = 0.20          # Circuit breaker trigger (relaxed)
    min_liquidity_days: int = 5         # Minimum liquidity window
    max_leverage: float = 1.0           # Maximum leverage
    var_limit: float = 0.05             # VaR limit (5%)


class RiskManagerAgent(BaseAgent):
    """
    Risk Manager Agent for position sizing and risk constraints.
    
    FIXED:
    - Higher minimum risk budget (95% invested)
    - Less aggressive regime penalties
    - More realistic position limits
    """
    
    # Regime-dependent risk parameters (FIXED: Higher allocations)
    DEFAULT_REGIME_PARAMS = {
        0: {  # Bull - Full exposure
            'risk_multiplier': 1.0,
            'max_position': 0.30,
            'max_leverage': 1.0,
            'slippage_bps': 5,
            'stop_loss': 0.05
        },
        1: {  # Transition - Slightly reduced
            'risk_multiplier': 0.95,
            'max_position': 0.25,
            'max_leverage': 1.0,
            'slippage_bps': 10,
            'stop_loss': 0.04
        },
        2: {  # Bear - Still substantial exposure
            'risk_multiplier': 0.85,
            'max_position': 0.20,
            'max_leverage': 0.9,
            'slippage_bps': 20,
            'stop_loss': 0.03
        }
    }
    
    def __init__(self, message_bus: MessageBus, config: Optional[Dict] = None):
        super().__init__("RiskManagerAgent", message_bus, config)
        
        # Configuration
        self.max_drawdown_limit = self.get_config('max_drawdown_limit', 0.20)
        self.var_confidence = self.get_config('var_confidence', 0.95)
        self.position_sizing_method = self.get_config('position_sizing_method', 'risk_parity')
        self.regime_params = self.get_config('regime_params', self.DEFAULT_REGIME_PARAMS)
        
        # Risk limits
        self.risk_limits = RiskLimits(
            max_position=self.get_config('max_position', 0.30),
            max_drawdown=self.max_drawdown_limit,
            max_leverage=self.get_config('max_leverage', 1.0),
            var_limit=self.get_config('var_limit', 0.05)
        )
        
        # State
        self.market_data: Optional[MarketDataEvent] = None
        self.regime_event: Optional[RegimeEvent] = None
        self.quant_signal: Optional[QuantSignalEvent] = None
        
        # Portfolio state
        self.current_portfolio_value = self.get_config('initial_capital', 1000000)
        self.peak_value = self.current_portfolio_value
        self.current_drawdown = 0.0
        
        # Risk metrics
        self.var_95 = 0.0
        self.var_99 = 0.0
        self.cvar_95 = 0.0
        self.avg_correlation = 0.0
        
        # Alerts
        self.alerts: List[str] = []
        self.circuit_breaker_active = False
    
    def _initialize(self) -> None:
        """Initialize Risk Manager."""
        self.logger.info(f"Initialized with max_drawdown={self.max_drawdown_limit}, "
                        f"var_confidence={self.var_confidence}")
    
    def get_subscribed_events(self) -> List[Type[Event]]:
        """Subscribe to required events."""
        return [MarketDataEvent, RegimeEvent, QuantSignalEvent]
    
    def on_event(self, event: Event) -> None:
        """Handle incoming events."""
        if isinstance(event, MarketDataEvent):
            self.market_data = event
        elif isinstance(event, RegimeEvent):
            self.regime_event = event
        elif isinstance(event, QuantSignalEvent):
            self.quant_signal = event
            self.logger.info("Received QuantSignalEvent, ready for risk assessment")
    
    def _process(self) -> None:
        """
        Main processing:
        1. Check circuit breaker conditions
        2. Calculate risk metrics (VaR, CVaR)
        3. Assess correlation regime
        4. Apply position limits based on regime
        5. Adjust expected returns for risk
        6. Publish RiskAdjustedSignalEvent
        """
        if self.quant_signal is None:
            raise ValueError("No quant signal available")
        
        self.alerts = []
        
        # Get current regime parameters
        current_regime = self.regime_event.current_regime if self.regime_event else 0
        regime_params = self.regime_params.get(current_regime, self.regime_params[0])
        
        # Check circuit breaker (with higher tolerance)
        self._check_circuit_breaker()
        
        if self.circuit_breaker_active:
            self.alerts.append("CIRCUIT BREAKER ACTIVE - Defensive mode")
            regime_params = self.regime_params[2]  # Use bear parameters
        
        # Calculate risk metrics
        self._calculate_risk_metrics()
        
        # Assess correlation regime
        self._assess_correlation_regime()
        
        # Calculate position limits
        position_limits = self._calculate_position_limits(regime_params)
        
        # Risk-adjust expected returns
        risk_adjusted_returns = self._risk_adjust_returns(regime_params)
        
        # Calculate risk budget (FIXED: Higher minimum)
        risk_budget = self._calculate_risk_budget(regime_params)
        
        # Publish event
        event = RiskAdjustedSignalEvent(
            source_agent=self.name,
            risk_adjusted_returns=risk_adjusted_returns,
            position_limits=position_limits,
            risk_budget=risk_budget,
            max_leverage=regime_params['max_leverage'],
            var_95=self.var_95,
            var_99=self.var_99,
            cvar_95=self.cvar_95,
            expected_shortfall=self.cvar_95,
            current_drawdown=self.current_drawdown,
            max_drawdown_limit=self.max_drawdown_limit,
            avg_correlation=self.avg_correlation,
            correlation_regime=self._get_correlation_regime_name(),
            alerts=self.alerts,
            circuit_breaker_active=self.circuit_breaker_active,
            regime_risk_multiplier=regime_params['risk_multiplier']
        )
        
        self.publish(event)
        self.logger.info(f"Published RiskAdjustedSignalEvent: risk_budget={risk_budget:.2%}, "
                        f"VaR95={self.var_95:.2%}, drawdown={self.current_drawdown:.2%}")
    
    def _check_circuit_breaker(self) -> None:
        """Check if circuit breaker should be triggered."""
        # Update drawdown
        if self.current_portfolio_value > self.peak_value:
            self.peak_value = self.current_portfolio_value
        
        self.current_drawdown = (self.peak_value - self.current_portfolio_value) / self.peak_value
        
        # Trigger circuit breaker if drawdown exceeds limit (20%)
        if self.current_drawdown > self.max_drawdown_limit:
            if not self.circuit_breaker_active:
                self.circuit_breaker_active = True
                self.alerts.append(f"Drawdown ({self.current_drawdown:.1%}) exceeds limit ({self.max_drawdown_limit:.1%})")
                self.logger.warning(f"Circuit breaker triggered at {self.current_drawdown:.1%} drawdown")
        
        # Reset circuit breaker when drawdown recovers to half the limit
        elif self.current_drawdown < self.max_drawdown_limit / 2:
            if self.circuit_breaker_active:
                self.circuit_breaker_active = False
                self.logger.info("Circuit breaker reset")
    
    def _calculate_risk_metrics(self) -> None:
        """Calculate VaR, CVaR, and other risk metrics."""
        if self.market_data is None or self.market_data.returns is None:
            return
        
        # Use portfolio returns or market proxy
        returns = self.market_data.returns.mean(axis=1)  # Equal-weighted proxy
        
        # Historical VaR
        self.var_95 = -np.percentile(returns, 5)
        self.var_99 = -np.percentile(returns, 1)
        
        # Conditional VaR (Expected Shortfall)
        var_threshold = np.percentile(returns, 5)
        tail_returns = returns[returns <= var_threshold]
        self.cvar_95 = -tail_returns.mean() if len(tail_returns) > 0 else self.var_95
        
        # Parametric VaR (assuming normal distribution)
        mu = returns.mean()
        sigma = returns.std()
        z_95 = stats.norm.ppf(0.05)
        parametric_var = -(mu + z_95 * sigma)
        
        self.logger.debug(f"Historical VaR95: {self.var_95:.4f}, Parametric VaR95: {parametric_var:.4f}")
    
    def _assess_correlation_regime(self) -> None:
        """
        Assess current correlation regime.
        High correlation indicates risk-on/risk-off dynamics.
        """
        if self.market_data is None or self.market_data.returns is None:
            return
        
        returns = self.market_data.returns
        
        # Calculate rolling correlation (6-month window)
        window = 126
        if len(returns) < window:
            return
        
        recent_returns = returns.tail(window)
        corr_matrix = recent_returns.corr()
        
        # Average off-diagonal correlation
        n = len(corr_matrix)
        if n > 1:
            mask = ~np.eye(n, dtype=bool)
            self.avg_correlation = corr_matrix.values[mask].mean()
        
        # Alert if correlation is unusually high (risk-off environment)
        if self.avg_correlation > 0.7:
            self.alerts.append(f"High correlation regime ({self.avg_correlation:.2f}) - diversification reduced")
    
    def _get_correlation_regime_name(self) -> str:
        """Get descriptive name for correlation regime."""
        if self.avg_correlation > 0.6:
            return "High (Risk-Off)"
        elif self.avg_correlation > 0.3:
            return "Normal"
        else:
            return "Low (Diversified)"
    
    def _calculate_position_limits(self, regime_params: Dict) -> Dict[str, float]:
        """
        Calculate position limits for each asset.
        """
        if self.quant_signal is None or self.quant_signal.expected_returns is None:
            return {}
        
        assets = self.quant_signal.expected_returns.index.tolist()
        base_limit = regime_params['max_position']
        
        # Adjust limits based on volatility (lower limit for higher vol assets)
        limits = {}
        if self.quant_signal.covariance_matrix is not None:
            cov = self.quant_signal.covariance_matrix
            vols = np.sqrt(np.diag(cov))
            avg_vol = vols.mean()
            
            for asset in assets:
                if asset in cov.index:
                    asset_vol = np.sqrt(cov.loc[asset, asset])
                    # Scale limit inversely with volatility
                    vol_ratio = avg_vol / asset_vol if asset_vol > 0 else 1.0
                    limits[asset] = min(base_limit * vol_ratio, base_limit * 1.5)
                else:
                    limits[asset] = base_limit
        else:
            limits = {asset: base_limit for asset in assets}
        
        return limits
    
    def _risk_adjust_returns(self, regime_params: Dict) -> pd.Series:
        """
        Apply risk adjustments to expected returns.
        
        FIXED: Less aggressive adjustments
        """
        if self.quant_signal is None or self.quant_signal.expected_returns is None:
            return pd.Series()
        
        returns = self.quant_signal.expected_returns.copy()
        
        # Apply regime multiplier (less aggressive)
        returns = returns * regime_params['risk_multiplier']
        
        # Apply drawdown penalty only for severe drawdowns (>10%)
        if self.current_drawdown > 0.10:
            drawdown_penalty = 1 - (self.current_drawdown / self.max_drawdown_limit) * 0.5
            returns = returns * max(drawdown_penalty, 0.7)
            self.alerts.append(f"Drawdown penalty applied: {drawdown_penalty:.2f}")
        
        return returns
    
    def _calculate_risk_budget(self, regime_params: Dict) -> float:
        """
        Calculate total risk budget (0-1).
        
        FIXED: Much higher minimum (95%) to ensure market exposure
        """
        base_budget = 1.0
        
        # Reduce based on regime (but only slightly)
        budget = base_budget * regime_params['risk_multiplier']
        
        # Reduce based on drawdown (only for severe drawdowns)
        if self.current_drawdown > 0.10:
            # Less aggressive reduction
            drawdown_factor = 1 - (self.current_drawdown / self.max_drawdown_limit) * 0.3
            budget *= max(drawdown_factor, 0.8)
        
        # Reduce if correlation is very high
        if self.avg_correlation > 0.7:
            budget *= 0.95
        
        # CRITICAL FIX: High minimum budget to ensure market participation
        budget = max(budget, 0.90)
        
        self.logger.info(f"Risk budget: {budget:.2%} (regime={regime_params['risk_multiplier']:.2f})")
        
        return budget
    
    # =========================================================================
    # Position Sizing Methods
    # =========================================================================
    
    def kelly_criterion(self, expected_returns: pd.Series, 
                       covariance: pd.DataFrame,
                       fraction: float = 0.25) -> pd.Series:
        """Calculate Kelly Criterion position sizes."""
        try:
            cov_inv = np.linalg.inv(covariance.values)
            weights = np.dot(cov_inv, expected_returns.values)
            weights = weights / np.sum(np.abs(weights))
            weights = weights * fraction
            return pd.Series(weights, index=expected_returns.index)
        except:
            return expected_returns / expected_returns.sum()
    
    def risk_parity(self, covariance: pd.DataFrame) -> pd.Series:
        """Calculate Risk Parity weights."""
        vols = np.sqrt(np.diag(covariance.values))
        inv_vols = 1 / vols
        weights = inv_vols / inv_vols.sum()
        return pd.Series(weights, index=covariance.index)
    
    # =========================================================================
    # Public API
    # =========================================================================
    
    def update_portfolio_value(self, value: float) -> None:
        """Update current portfolio value (called by execution)."""
        self.current_portfolio_value = value
        if value > self.peak_value:
            self.peak_value = value
    
    def get_risk_summary(self) -> Dict:
        """Get summary of current risk state."""
        return {
            'current_drawdown': self.current_drawdown,
            'var_95': self.var_95,
            'cvar_95': self.cvar_95,
            'avg_correlation': self.avg_correlation,
            'circuit_breaker': self.circuit_breaker_active,
            'alerts': self.alerts
        }
