"""
Risk Manager Agent
===================
Risk monitoring and position sizing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Type
from scipy import stats

from agents.base_agent import BaseAgent
from events.message_bus import (
    MessageBus, Event, MarketDataEvent, RegimeEvent, 
    QuantSignalEvent, RiskAssessmentEvent
)


class RiskManagerAgent(BaseAgent):
    """
    Risk Manager Agent for portfolio risk monitoring and constraints.
    
    Responsibilities:
    - VaR and CVaR calculation
    - Drawdown monitoring
    - Position sizing based on regime
    - Risk budget allocation
    
    Config Options:
        max_drawdown_limit: Maximum allowed drawdown
        var_confidence: VaR confidence level
        var_limit: Maximum VaR limit
        position_sizing_method: 'equal', 'risk_parity', 'inverse_vol'
        max_position: Maximum single position weight
        max_sector_exposure: Maximum sector exposure
    """
    
    def __init__(self, message_bus: MessageBus, config: Optional[Dict] = None):
        super().__init__("RiskManagerAgent", message_bus, config)
        
        # Configuration
        self.max_drawdown_limit = self.get_config('max_drawdown_limit', 0.20)
        self.var_confidence = self.get_config('var_confidence', 0.95)
        self.var_limit = self.get_config('var_limit', 0.05)
        self.position_sizing_method = self.get_config('position_sizing_method', 'risk_parity')
        self.max_position = self.get_config('max_position', 0.30)
        self.max_sector_exposure = self.get_config('max_sector_exposure', 0.35)
        self.max_leverage = self.get_config('max_leverage', 1.0)
        
        # Regime-specific risk parameters
        self.regime_params = self.get_config('regime_params', {
            0: {  # Bull
                'risk_multiplier': 1.0,
                'max_position': 0.30,
                'stop_loss': 0.08,
                'take_profit': 0.20,
                'risk_aversion': 2.0
            },
            1: {  # Transition
                'risk_multiplier': 0.90,
                'max_position': 0.25,
                'stop_loss': 0.06,
                'take_profit': 0.15,
                'risk_aversion': 3.5
            },
            2: {  # Bear
                'risk_multiplier': 0.70,
                'max_position': 0.20,
                'stop_loss': 0.04,
                'take_profit': 0.10,
                'risk_aversion': 6.0
            }
        })
        
        # State
        self.market_data: Optional[MarketDataEvent] = None
        self.regime_event: Optional[RegimeEvent] = None
        self.quant_signal: Optional[QuantSignalEvent] = None
        
        # Tracking
        self.portfolio_values: List[float] = []
        self.peak_value: float = 0.0
        self.current_drawdown: float = 0.0
        self.max_drawdown: float = 0.0
    
    def _initialize(self) -> None:
        """Initialize Risk Manager."""
        self.logger.info(f"Initialized with max_drawdown={self.max_drawdown_limit:.1%}, "
                        f"var_confidence={self.var_confidence:.1%}")
    
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
    
    def _process(self) -> None:
        """
        Main processing:
        1. Calculate risk metrics
        2. Determine position limits based on regime
        3. Generate risk budget
        4. Check for alerts
        5. Publish RiskAssessmentEvent
        """
        if self.quant_signal is None or self.quant_signal.covariance_matrix is None:
            raise ValueError("No quant signal available")
        
        cov_matrix = self.quant_signal.covariance_matrix
        
        # Get current regime
        current_regime = 0
        if self.regime_event is not None:
            current_regime = self.regime_event.current_regime
        
        # Get regime-specific parameters
        regime_params = self.regime_params.get(current_regime, self.regime_params[1])
        
        # Calculate VaR and CVaR (assuming equal weights for now)
        n_assets = len(cov_matrix)
        equal_weights = np.ones(n_assets) / n_assets
        
        portfolio_var, portfolio_cvar = self._calculate_var_cvar(
            equal_weights, 
            cov_matrix.values
        )
        
        # Calculate risk budget (risk parity)
        risk_budget = self._calculate_risk_budget(cov_matrix)
        
        # Determine position limits
        position_limits = self._determine_position_limits(cov_matrix, regime_params)
        
        # Check for alerts
        risk_alerts = self._check_risk_alerts(portfolio_var, regime_params)
        
        # Publish event
        event = RiskAssessmentEvent(
            source_agent=self.name,
            portfolio_var=portfolio_var,
            portfolio_cvar=portfolio_cvar,
            current_drawdown=self.current_drawdown,
            max_drawdown=self.max_drawdown,
            risk_budget=risk_budget,
            position_limits=position_limits,
            risk_alerts=risk_alerts,
            regime_risk_params=regime_params
        )
        
        self.publish(event)
        self.logger.info(f"Published RiskAssessmentEvent: VaR={portfolio_var:.2%}, "
                        f"alerts={len(risk_alerts)}")
    
    def _calculate_var_cvar(self, 
                           weights: np.ndarray, 
                           cov_matrix: np.ndarray) -> tuple:
        """Calculate parametric VaR and CVaR."""
        # Portfolio volatility (daily)
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        daily_vol = portfolio_vol / np.sqrt(252)
        
        # VaR (parametric, assuming normal distribution)
        z_score = stats.norm.ppf(self.var_confidence)
        var = -z_score * daily_vol
        
        # CVaR (Expected Shortfall)
        cvar = daily_vol * stats.norm.pdf(z_score) / (1 - self.var_confidence)
        
        return var, cvar
    
    def _calculate_risk_budget(self, cov_matrix: pd.DataFrame) -> Dict[str, float]:
        """Calculate risk parity weights."""
        n = len(cov_matrix)
        
        # Individual volatilities
        vols = np.sqrt(np.diag(cov_matrix.values))
        
        # Inverse volatility weights
        inv_vol = 1.0 / (vols + 1e-10)
        weights = inv_vol / inv_vol.sum()
        
        return dict(zip(cov_matrix.columns, weights))
    
    def _determine_position_limits(self, 
                                   cov_matrix: pd.DataFrame, 
                                   regime_params: Dict) -> Dict[str, float]:
        """Determine position limits based on volatility and regime."""
        limits = {}
        
        vols = np.sqrt(np.diag(cov_matrix.values)) / np.sqrt(252)  # Daily vol
        max_vol = vols.max()
        
        base_limit = regime_params.get('max_position', self.max_position)
        
        for i, asset in enumerate(cov_matrix.columns):
            # Lower limit for higher volatility assets
            vol_adjustment = 1.0 - 0.5 * (vols[i] / max_vol) if max_vol > 0 else 1.0
            limits[asset] = base_limit * vol_adjustment
        
        return limits
    
    def _check_risk_alerts(self, portfolio_var: float, regime_params: Dict) -> List[str]:
        """Check for risk limit breaches."""
        alerts = []
        
        # VaR limit check
        if portfolio_var > self.var_limit:
            alerts.append(f"VaR_BREACH: {portfolio_var:.2%} > {self.var_limit:.2%}")
        
        # Drawdown check
        if self.current_drawdown < -self.max_drawdown_limit:
            alerts.append(f"DRAWDOWN_BREACH: {self.current_drawdown:.2%}")
        
        # Regime-based alert
        regime = self.regime_event.current_regime if self.regime_event else 0
        if regime == 2:  # Bear regime
            alerts.append("REGIME_ALERT: Bear market regime detected")
        
        return alerts
    
    def update_portfolio_value(self, value: float) -> None:
        """Update portfolio value and drawdown tracking."""
        self.portfolio_values.append(value)
        
        if value > self.peak_value:
            self.peak_value = value
        
        self.current_drawdown = (value / self.peak_value) - 1
        self.max_drawdown = min(self.max_drawdown, self.current_drawdown)
    
    # =========================================================================
    # Public API
    # =========================================================================
    
    def get_regime_risk_multiplier(self) -> float:
        """Get current regime's risk multiplier."""
        if self.regime_event is None:
            return 1.0
        
        regime = self.regime_event.current_regime
        params = self.regime_params.get(regime, {})
        return params.get('risk_multiplier', 1.0)
    
    def get_risk_aversion(self) -> float:
        """Get current regime's risk aversion parameter."""
        if self.regime_event is None:
            return 3.0
        
        regime = self.regime_event.current_regime
        params = self.regime_params.get(regime, {})
        return params.get('risk_aversion', 3.0)
