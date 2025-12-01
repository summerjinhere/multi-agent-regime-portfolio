"""
Portfolio Agent
================
Portfolio optimization and rebalancing decisions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Type
from scipy.optimize import minimize
import warnings

from agents.base_agent import BaseAgent
from events.message_bus import (
    MessageBus, Event, MarketDataEvent, RegimeEvent,
    QuantSignalEvent, RiskAssessmentEvent, PortfolioSignalEvent
)


class PortfolioAgent(BaseAgent):
    """
    Portfolio Agent for asset allocation optimization.
    
    Responsibilities:
    - Mean-variance optimization
    - Black-Litterman model
    - Rebalancing decisions
    - Transaction cost awareness
    
    Config Options:
        use_black_litterman: Whether to use Black-Litterman
        rebalance_threshold: Minimum deviation to trigger rebalance
        max_turnover: Maximum portfolio turnover
        transaction_cost_bps: Transaction cost in basis points
        risk_free_rate: Risk-free rate for Sharpe calculation
        min_weight: Minimum position weight
        max_weight: Maximum position weight
    """
    
    def __init__(self, message_bus: MessageBus, config: Optional[Dict] = None):
        super().__init__("PortfolioAgent", message_bus, config)
        
        # Configuration
        self.use_black_litterman = self.get_config('use_black_litterman', True)
        self.rebalance_threshold = self.get_config('rebalance_threshold', 0.03)
        self.max_turnover = self.get_config('max_turnover', 0.40)
        self.transaction_cost_bps = self.get_config('transaction_cost_bps', 10)
        self.risk_free_rate = self.get_config('risk_free_rate', 0.05)
        self.min_weight = self.get_config('min_weight', 0.0)
        self.max_weight = self.get_config('max_weight', 0.30)
        
        # State
        self.market_data: Optional[MarketDataEvent] = None
        self.regime_event: Optional[RegimeEvent] = None
        self.quant_signal: Optional[QuantSignalEvent] = None
        self.risk_assessment: Optional[RiskAssessmentEvent] = None
        
        # Portfolio state
        self.current_weights: Optional[pd.Series] = None
        self.target_weights: Optional[pd.Series] = None
    
    def _initialize(self) -> None:
        """Initialize Portfolio Agent."""
        self.logger.info(f"Initialized with BL={self.use_black_litterman}, "
                        f"rebalance_threshold={self.rebalance_threshold:.1%}")
    
    def get_subscribed_events(self) -> List[Type[Event]]:
        """Subscribe to required events."""
        return [MarketDataEvent, RegimeEvent, QuantSignalEvent, RiskAssessmentEvent]
    
    def on_event(self, event: Event) -> None:
        """Handle incoming events."""
        if isinstance(event, MarketDataEvent):
            self.market_data = event
        elif isinstance(event, RegimeEvent):
            self.regime_event = event
        elif isinstance(event, QuantSignalEvent):
            self.quant_signal = event
        elif isinstance(event, RiskAssessmentEvent):
            self.risk_assessment = event
    
    def calculate_weights(self, 
                         expected_returns: pd.Series, 
                         cov_matrix: pd.DataFrame, 
                         risk_aversion: float = 3.0,
                         current_weights: Optional[Dict[str, float]] = None) -> pd.Series:
        """
        Calculate optimal portfolio weights using Black-Litterman and MVO.
        Includes turnover control if current_weights is provided.
        """
        # 1. Black-Litterman: Combine Market Views with Model Views
        # In this simplified version, we treat 'expected_returns' as the combined view
        # In a full BL implementation, we would blend market equilibrium returns with these views
        # Here we assume QuantAgent has already done the heavy lifting of view generation
        if self.use_black_litterman:
            expected_returns = self._apply_black_litterman(
                expected_returns, cov_matrix
            )
        
        # 2. Mean-Variance Optimization
        weights = self._optimize_portfolio(expected_returns, cov_matrix, risk_aversion, {})
        
        # 3. Turnover Control (Smooth Rebalancing)
        if current_weights is not None and not weights.empty:
            # Align indices
            curr_w_series = pd.Series(current_weights).reindex(weights.index).fillna(0)
            
            # Blend: 80% current, 20% new (Effective max turnover ~20% per rebalance)
            # This dampens the rebalancing and reduces kurtosis
            weights = 0.8 * curr_w_series + 0.2 * weights
            
            # Re-normalize to sum to 1.0 (or close to it, MVO usually sums to 1)
            if weights.sum() > 0:
                weights = weights / weights.sum()
        
        return weights

    def _process(self) -> None:
        """
        Main processing:
        1. Get expected returns and covariance
        2. Apply Black-Litterman if enabled
        3. Optimize portfolio
        4. Check rebalancing need
        5. Publish PortfolioSignalEvent
        """
        if self.quant_signal is None:
            raise ValueError("No quant signal available")
        
        expected_returns = self.quant_signal.expected_returns
        cov_matrix = self.quant_signal.covariance_matrix
        
        if expected_returns is None or cov_matrix is None:
            raise ValueError("Missing expected returns or covariance matrix")
        
        # Get risk parameters
        risk_aversion = 3.0
        if self.risk_assessment is not None:
            if self.risk_assessment.regime_risk_params:
                risk_aversion = self.risk_assessment.regime_risk_params.get('risk_aversion', 3.0)
        
        # Calculate weights
        self.target_weights = self.calculate_weights(
            expected_returns, cov_matrix, risk_aversion
        )
        
        # Initialize current weights if needed
        if self.current_weights is None:
            n_assets = len(self.target_weights)
            self.current_weights = pd.Series(
                1.0 / n_assets, 
                index=self.target_weights.index
            )
        
        # Check rebalancing need
        turnover = self._calculate_turnover(self.current_weights, self.target_weights)
        rebalance_required = turnover > self.rebalance_threshold
        
        # Calculate portfolio metrics
        exp_return = np.dot(self.target_weights, expected_returns)
        exp_vol = np.sqrt(np.dot(self.target_weights, np.dot(cov_matrix, self.target_weights)))
        sharpe = (exp_return - self.risk_free_rate) / exp_vol if exp_vol > 0 else 0
        
        # Optimization method used
        method = "Mean-Variance"
        if self.use_black_litterman:
            method = "Black-Litterman + MVO"
        
        # Publish event
        event = PortfolioSignalEvent(
            source_agent=self.name,
            target_weights=self.target_weights,
            current_weights=self.current_weights,
            rebalance_required=rebalance_required,
            turnover=turnover,
            expected_return=exp_return,
            expected_volatility=exp_vol,
            sharpe_ratio=sharpe,
            optimization_method=method
        )
        
        self.publish(event)
        self.logger.info(f"Published PortfolioSignalEvent: E[R]={exp_return:.2%}, "
                        f"Vol={exp_vol:.2%}, Sharpe={sharpe:.2f}, "
                        f"Rebalance={rebalance_required}")
    
    def _apply_black_litterman(self, 
                               expected_returns: pd.Series,
                               cov_matrix: pd.DataFrame) -> pd.Series:
        """
        Apply Black-Litterman model to combine prior with views.
        """
        try:
            n = len(expected_returns)
            tau = 0.05  # Scaling factor for uncertainty
            
            # Market capitalization weights (proxy with equal weights)
            market_weights = np.ones(n) / n
            
            # Implied equilibrium returns (reverse optimization)
            delta = 2.5  # Risk aversion
            pi = delta * np.dot(cov_matrix.values, market_weights)
            
            # Views matrix (use our expected returns as views)
            P = np.eye(n)  # Identity = absolute views
            Q = expected_returns.values  # Our forecasts
            
            # Omega (uncertainty in views) - diagonal
            omega = tau * np.diag(np.diag(cov_matrix.values))
            
            # Black-Litterman formula
            tau_sigma = tau * cov_matrix.values
            tau_sigma_inv = np.linalg.inv(tau_sigma)
            omega_inv = np.linalg.inv(omega)
            
            # Posterior expected returns
            M1 = np.linalg.inv(tau_sigma_inv + np.dot(P.T, np.dot(omega_inv, P)))
            M2 = np.dot(tau_sigma_inv, pi) + np.dot(P.T, np.dot(omega_inv, Q))
            
            bl_returns = np.dot(M1, M2)
            
            return pd.Series(bl_returns, index=expected_returns.index)
            
        except Exception as e:
            self.logger.warning(f"Black-Litterman failed: {e}")
            return expected_returns
    
    def _optimize_portfolio(self,
                           expected_returns: pd.Series,
                           cov_matrix: pd.DataFrame,
                           risk_aversion: float,
                           position_limits: Dict[str, float]) -> pd.Series:
        """
        Mean-variance portfolio optimization.
        """
        n = len(expected_returns)
        
        # Objective: maximize utility = E[R] - (gamma/2) * Var
        def objective(w):
            ret = np.dot(expected_returns.values, w)
            risk = np.dot(w, np.dot(cov_matrix.values, w))
            return -(ret - (risk_aversion / 2) * risk)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Weights sum to 1
        ]
        
        # Bounds
        bounds = []
        for i, asset in enumerate(expected_returns.index):
            max_w = position_limits.get(asset, self.max_weight)
            bounds.append((self.min_weight, max_w))
        
        # Initial guess (equal weight)
        x0 = np.ones(n) / n
        
        # Optimize
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-8}
            )
        
        if result.success:
            weights = result.x
        else:
            self.logger.warning(f"Optimization failed: {result.message}")
            weights = x0
        
        # Ensure weights are positive and sum to 1
        weights = np.maximum(weights, 0)
        weights = weights / weights.sum()
        
        return pd.Series(weights, index=expected_returns.index)
    
    def _calculate_turnover(self, 
                           current: pd.Series, 
                           target: pd.Series) -> float:
        """Calculate portfolio turnover."""
        # Align indices
        all_assets = current.index.union(target.index)
        current_aligned = current.reindex(all_assets, fill_value=0)
        target_aligned = target.reindex(all_assets, fill_value=0)
        
        return np.sum(np.abs(target_aligned - current_aligned))
    
    def update_weights(self, new_weights: pd.Series) -> None:
        """Update current portfolio weights after execution."""
        self.current_weights = new_weights
    
    # =========================================================================
    # Public API
    # =========================================================================
    
    def get_portfolio_metrics(self) -> Dict:
        """Get current portfolio metrics."""
        if self.target_weights is None or self.quant_signal is None:
            return {}
        
        exp_ret = self.quant_signal.expected_returns
        cov = self.quant_signal.covariance_matrix
        
        if exp_ret is None or cov is None:
            return {}
        
        w = self.target_weights
        
        return {
            'expected_return': np.dot(w, exp_ret),
            'volatility': np.sqrt(np.dot(w, np.dot(cov, w))),
            'n_assets': (w > 0.01).sum(),
            'max_weight': w.max(),
            'concentration': (w ** 2).sum()  # Herfindahl index
        }
    
    def get_top_positions(self, n: int = 10) -> pd.Series:
        """Get top N positions by weight."""
        if self.target_weights is None:
            return pd.Series()
        
        return self.target_weights.nlargest(n)
