"""
Portfolio Agent
===============
Final portfolio optimization and trade signal generation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Type, Tuple
from pypfopt import EfficientFrontier, black_litterman, BlackLittermanModel
from pypfopt import objective_functions

from agents.base_agent import BaseAgent
from events.message_bus import MessageBus
from events.base_event import (
    Event, MarketDataEvent, RegimeEvent, 
    RiskAdjustedSignalEvent, TradeSignalEvent, TradeOrder
)


class PortfolioAgent(BaseAgent):
    """
    Portfolio Agent for final optimization and trade execution signals.
    
    Responsibilities:
    - Black-Litterman portfolio optimization
    - Transaction cost modeling
    - Turnover optimization
    - Generate BUY/SELL/HOLD signals
    - Stock selection within asset classes
    
    Config Options:
        use_black_litterman: Use BL model (default: True)
        rebalance_threshold: Min weight change to trigger rebalance (default: 0.05)
        max_turnover: Maximum portfolio turnover per rebalance (default: 0.30)
        transaction_cost_bps: Base transaction cost in basis points
        risk_free_rate: Risk-free rate for Sharpe calculation
    """
    
    def __init__(self, message_bus: MessageBus, config: Optional[Dict] = None):
        super().__init__("PortfolioAgent", message_bus, config)
        
        # Configuration
        self.use_black_litterman = self.get_config('use_black_litterman', True)
        self.rebalance_threshold = self.get_config('rebalance_threshold', 0.05)
        self.max_turnover = self.get_config('max_turnover', 0.30)
        self.transaction_cost_bps = self.get_config('transaction_cost_bps', 10)
        self.risk_free_rate = self.get_config('risk_free_rate', 0.02)
        
        # Regime-dependent slippage (bps)
        self.regime_slippage = {
            0: 5,    # Bull: low slippage
            1: 10,   # Transition
            2: 25    # Bear: high slippage
        }
        
        # State
        self.market_data: Optional[MarketDataEvent] = None
        self.regime_event: Optional[RegimeEvent] = None
        self.risk_signal: Optional[RiskAdjustedSignalEvent] = None
        
        # Current portfolio state
        self.current_weights: Dict[str, float] = {}
        self.target_weights: Dict[str, float] = {}
    
    def _initialize(self) -> None:
        """Initialize Portfolio Agent."""
        self.logger.info(f"Initialized with BL={self.use_black_litterman}, "
                        f"threshold={self.rebalance_threshold}")
    
    def get_subscribed_events(self) -> List[Type[Event]]:
        """Subscribe to required events."""
        return [MarketDataEvent, RegimeEvent, RiskAdjustedSignalEvent]
    
    def on_event(self, event: Event) -> None:
        """Handle incoming events."""
        if isinstance(event, MarketDataEvent):
            self.market_data = event
        elif isinstance(event, RegimeEvent):
            self.regime_event = event
        elif isinstance(event, RiskAdjustedSignalEvent):
            self.risk_signal = event
            self.logger.info("Received RiskAdjustedSignalEvent, ready for optimization")
    
    def _process(self) -> None:
        """
        Main processing:
        1. Optimize portfolio (Mean-Variance or Black-Litterman)
        2. Apply position limits
        3. Calculate turnover and transaction costs
        4. Generate trade orders (BUY/SELL/HOLD)
        5. Publish TradeSignalEvent
        """
        if self.risk_signal is None:
            raise ValueError("No risk signal available")
        
        expected_returns = self.risk_signal.risk_adjusted_returns
        position_limits = self.risk_signal.position_limits
        
        if expected_returns is None or len(expected_returns) == 0:
            raise ValueError("No expected returns available")
        
        # Get covariance matrix (from cached quant signal)
        cov_matrix = self._get_covariance_matrix()
        
        if cov_matrix is None:
            raise ValueError("No covariance matrix available")
        
        # Check circuit breaker
        if self.risk_signal.circuit_breaker_active:
            self.logger.warning("Circuit breaker active - generating defensive allocation")
            self.target_weights = self._get_defensive_allocation(expected_returns.index)
        else:
            # Optimize
            try:
                if self.use_black_litterman:
                    self.target_weights = self._optimize_black_litterman(
                        expected_returns, cov_matrix, position_limits
                    )
                else:
                    self.target_weights = self._optimize_mean_variance(
                        expected_returns, cov_matrix, position_limits
                    )
            except Exception as e:
                self.logger.error(f"Optimization failed: {e}. Falling back to Equal Weight.")
                # Fallback to equal weight
                n = len(expected_returns)
                self.target_weights = {ticker: 1.0/n for ticker in expected_returns.index}
        
        # Apply risk budget
        risk_budget = self.risk_signal.risk_budget
        self.target_weights = {k: v * risk_budget for k, v in self.target_weights.items()}
        
        # Add cash for remaining allocation
        total_invested = sum(self.target_weights.values())
        if total_invested < 1.0:
            self.target_weights['Cash'] = 1.0 - total_invested
        
        # Generate trade orders
        trades = self._generate_trade_orders()
        
        # Calculate metrics
        total_turnover = self._calculate_turnover()
        transaction_cost = self._estimate_transaction_cost(total_turnover)
        
        # Determine rebalance reason
        rebalance_reason = self._determine_rebalance_reason()
        
        # Publish event
        event = TradeSignalEvent(
            source_agent=self.name,
            target_weights=self.target_weights,
            current_weights=self.current_weights,
            trades=trades,
            rebalance_reason=rebalance_reason,
            total_turnover=total_turnover,
            estimated_transaction_cost=transaction_cost,
            optimization_method="Black-Litterman" if self.use_black_litterman else "Mean-Variance",
            objective="Max Sharpe",
            constraints_satisfied=True,
            detailed_weights=self._generate_detailed_allocation()
        )
        
        self.publish(event)
        
        # Log summary
        n_buys = len([t for t in trades if t.action == "BUY"])
        n_sells = len([t for t in trades if t.action == "SELL"])
        n_holds = len([t for t in trades if t.action == "HOLD"])
        
        self.logger.info(f"Published TradeSignalEvent: {n_buys} BUY, {n_sells} SELL, "
                        f"{n_holds} HOLD, turnover={total_turnover:.2%}")
    
    def _get_covariance_matrix(self) -> Optional[pd.DataFrame]:
        """Get covariance matrix from message bus history."""
        from events.base_event import QuantSignalEvent
        
        quant_event = self.bus.get_last_event(QuantSignalEvent)
        if quant_event and quant_event.covariance_matrix is not None:
            return quant_event.covariance_matrix
        return None
    
    def _optimize_black_litterman(self, expected_returns: pd.Series,
                                  cov_matrix: pd.DataFrame,
                                  position_limits: Dict[str, float]) -> Dict[str, float]:
        """
        Black-Litterman portfolio optimization.
        
        Args:
            expected_returns: Risk-adjusted expected returns (our views)
            cov_matrix: Covariance matrix
            position_limits: Max weight per asset
            
        Returns:
            Optimal weights
        """
        try:
            # Align data
            common_assets = list(set(expected_returns.index) & set(cov_matrix.index))
            mu = expected_returns[common_assets]
            S = cov_matrix.loc[common_assets, common_assets]
            
            # Market equilibrium weights (equal-weighted as proxy)
            market_weights = pd.Series({a: 1.0/len(common_assets) for a in common_assets})
            
            # Calculate risk aversion from market
            delta = black_litterman.market_implied_risk_aversion(
                market_weights, 
                risk_free_rate=self.risk_free_rate
            )
            
            # Prior returns (market equilibrium)
            prior = black_litterman.market_implied_prior_returns(
                market_caps=market_weights,
                risk_aversion=delta,
                cov_matrix=S,
                risk_free_rate=self.risk_free_rate
            )
            
            # Create BL model with our views
            bl = BlackLittermanModel(
                S, 
                pi=prior, 
                absolute_views=mu.to_dict()
            )
            
            # Get posterior estimates
            bl_returns = bl.bl_returns()
            bl_cov = bl.bl_cov()
            
            # Optimize
            ef = EfficientFrontier(bl_returns, bl_cov)
            
            # Add position limits as constraints
            for asset in common_assets:
                if asset in position_limits:
                    ef.add_constraint(lambda w, a=asset: w[list(common_assets).index(a)] <= position_limits[a])
            
            # Maximize Sharpe
            weights = ef.max_sharpe(risk_free_rate=self.risk_free_rate)
            cleaned = ef.clean_weights()
            
            return cleaned
            
        except Exception as e:
            self.logger.warning(f"Black-Litterman optimization failed: {e}")
            return self._optimize_mean_variance(expected_returns, cov_matrix, position_limits)
    
    def _optimize_mean_variance(self, expected_returns: pd.Series,
                                cov_matrix: pd.DataFrame,
                                position_limits: Dict[str, float]) -> Dict[str, float]:
        """
        Standard Mean-Variance optimization (fallback).
        """
        try:
            # Align data
            common_assets = list(set(expected_returns.index) & set(cov_matrix.index))
            mu = expected_returns[common_assets]
            S = cov_matrix.loc[common_assets, common_assets]
            
            ef = EfficientFrontier(mu, S)
            
            # Add regularization
            ef.add_objective(objective_functions.L2_reg, gamma=0.1)
            
            weights = ef.max_sharpe(risk_free_rate=self.risk_free_rate)
            cleaned = ef.clean_weights()
            
            # Apply position limits
            for asset in cleaned:
                if asset in position_limits:
                    cleaned[asset] = min(cleaned[asset], position_limits[asset])
            
            # Renormalize
            total = sum(cleaned.values())
            if total > 0:
                cleaned = {k: v/total for k, v in cleaned.items()}
            
            return cleaned
            
        except Exception as e:
            self.logger.warning(f"Mean-Variance optimization failed: {e}")
            # Equal weight fallback
            n = len(expected_returns)
            return {k: 1.0/n for k in expected_returns.index}
    
    def _get_defensive_allocation(self, assets: pd.Index) -> Dict[str, float]:
        """
        Generate defensive allocation when circuit breaker is active.
        Heavy cash, some bonds, minimal equity.
        """
        weights = {}
        
        for asset in assets:
            if 'Bond' in asset or asset in ['TLT', 'IEF', 'SHY']:
                weights[asset] = 0.15
            elif 'Volatility' in asset or asset == '^VIX':
                weights[asset] = 0.05
            elif 'Commodity' in asset or asset in ['GLD', 'SLV']:
                weights[asset] = 0.05
            else:
                weights[asset] = 0.02
        
        # Normalize to leave room for cash
        total = sum(weights.values())
        if total > 0.5:
            weights = {k: v * 0.5 / total for k, v in weights.items()}
        
        return weights
    
    def _generate_trade_orders(self) -> List[TradeOrder]:
        """
        Generate trade orders based on weight differences.
        
        Returns:
            List of TradeOrder objects
        """
        trades = []
        
        all_assets = set(self.target_weights.keys()) | set(self.current_weights.keys())
        
        for asset in all_assets:
            current_w = self.current_weights.get(asset, 0.0)
            target_w = self.target_weights.get(asset, 0.0)
            delta = target_w - current_w
            
            # Determine action
            if abs(delta) < self.rebalance_threshold:
                action = "HOLD"
                urgency = "LOW"
            elif delta > 0:
                action = "BUY"
                urgency = "HIGH" if delta > 0.1 else "MEDIUM"
            else:
                action = "SELL"
                urgency = "HIGH" if abs(delta) > 0.1 else "MEDIUM"
            
            # Estimate cost
            current_regime = self.regime_event.current_regime if self.regime_event else 0
            slippage = self.regime_slippage.get(current_regime, 10) / 10000
            cost = abs(delta) * (self.transaction_cost_bps / 10000 + slippage)
            
            # Generate reason
            reason = self._generate_trade_reason(asset, action, delta)
            
            order = TradeOrder(
                ticker=asset,
                action=action,
                current_weight=current_w,
                target_weight=target_w,
                urgency=urgency,
                reason=reason,
                estimated_cost=cost
            )
            
            trades.append(order)
        
        # Sort by urgency and delta
        trades.sort(key=lambda x: (
            0 if x.urgency == "HIGH" else (1 if x.urgency == "MEDIUM" else 2),
            -abs(x.delta_weight)
        ))
        
        return trades
    
    def _generate_trade_reason(self, asset: str, action: str, delta: float) -> str:
        """Generate human-readable reason for trade."""
        if action == "HOLD":
            return f"Weight change ({delta:+.2%}) below threshold"
        
        reasons = []
        
        # Check if regime-driven
        if self.regime_event:
            regime_name = self.regime_event.get_regime_name()
            if action == "SELL" and self.regime_event.current_regime == 2:
                reasons.append(f"Bear regime ({regime_name})")
            elif action == "BUY" and self.regime_event.current_regime == 0:
                reasons.append(f"Bull regime ({regime_name})")
        
        # Check if risk-driven
        if self.risk_signal and self.risk_signal.circuit_breaker_active:
            reasons.append("Circuit breaker active")
        
        if not reasons:
            reasons.append("Optimization target")
        
        return f"{action} {abs(delta):.1%}: {', '.join(reasons)}"
    
    def _calculate_turnover(self) -> float:
        """Calculate total portfolio turnover."""
        all_assets = set(self.target_weights.keys()) | set(self.current_weights.keys())
        
        turnover = 0
        for asset in all_assets:
            current_w = self.current_weights.get(asset, 0.0)
            target_w = self.target_weights.get(asset, 0.0)
            turnover += abs(target_w - current_w)
        
        return turnover / 2  # Divide by 2 since buy and sell cancel out
    
    def _estimate_transaction_cost(self, turnover: float) -> float:
        """Estimate total transaction cost as fraction of portfolio."""
        current_regime = self.regime_event.current_regime if self.regime_event else 0
        slippage = self.regime_slippage.get(current_regime, 10) / 10000
        
        total_cost_bps = self.transaction_cost_bps / 10000 + slippage
        return turnover * total_cost_bps
    
    def _determine_rebalance_reason(self) -> str:
        """Determine primary reason for rebalance."""
        reasons = []
        
        if self.risk_signal and self.risk_signal.circuit_breaker_active:
            reasons.append("Circuit breaker triggered")
        
        if self.regime_event:
            reasons.append(f"Regime: {self.regime_event.get_regime_name()}")
        
        if self._calculate_turnover() > self.max_turnover:
            reasons.append("Turnover limit applied")
        
        return "; ".join(reasons) if reasons else "Scheduled rebalance"
    
    def _generate_detailed_allocation(self) -> Dict[str, float]:
        """
        Generate detailed stock-level allocation from sector weights.
        """
        detailed_weights = {}
        
        for asset, weight in self.target_weights.items():
            if weight <= 0.001:
                continue
                
            # Check if asset is a sector
            if self.market_data and asset in self.market_data.sector_mapping:
                # Select top stocks
                top_stocks = self.select_stocks_for_sector(asset, n_top=5)
                if top_stocks:
                    # Equal weight within sector for now
                    stock_weight = weight / len(top_stocks)
                    for stock in top_stocks:
                        detailed_weights[stock] = stock_weight
                else:
                    # Fallback if no stocks found (e.g. ETF)
                    detailed_weights[asset] = weight
            else:
                # Not a sector (e.g. Cash, Bonds, specific ticker)
                detailed_weights[asset] = weight
        
        self.logger.info(f"Generated detailed weights: {len(detailed_weights)} items")
        return detailed_weights
    
    # =========================================================================
    # Public API
    # =========================================================================
    
    def update_current_weights(self, weights: Dict[str, float]) -> None:
        """Update current portfolio weights."""
        self.current_weights = weights
    
    def get_allocation_summary(self) -> Dict:
        """Get allocation summary."""
        return {
            'target_weights': self.target_weights,
            'current_weights': self.current_weights,
            'turnover': self._calculate_turnover(),
            'estimated_cost': self._estimate_transaction_cost(self._calculate_turnover())
        }
    
    def select_stocks_for_sector(self, sector: str, n_top: int = 5) -> List[str]:
        """
        Select top stocks within a sector based on regime.
        
        Args:
            sector: Sector name
            n_top: Number of stocks to select
            
        Returns:
            List of selected ticker symbols
        """
        if self.market_data is None or self.market_data.prices is None:
            return []
        
        sector_mapping = self.market_data.sector_mapping
        if sector not in sector_mapping:
            return []
        
        tickers = sector_mapping[sector]
        valid_tickers = [t for t in tickers if t in self.market_data.prices.columns]
        
        if not valid_tickers:
            return []
        
        prices = self.market_data.prices[valid_tickers]
        returns = prices.pct_change()
        
        current_regime = self.regime_event.current_regime if self.regime_event else 0
        
        # Score based on regime
        if current_regime == 0:  # Bull: favor momentum
            scores = returns.mean()
        elif current_regime == 2:  # Bear: favor low vol
            scores = -returns.std()
        else:  # Transition: balanced
            scores = returns.mean() / returns.std()
        
        return scores.nlargest(n_top).index.tolist()
