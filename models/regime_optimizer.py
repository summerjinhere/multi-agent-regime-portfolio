"""
Regime-Adaptive Portfolio Optimizer
====================================

Features:
1. Regime-dependent risk aversion (γ)
2. Mean-variance optimization with regime-specific parameters
3. Risk parity and risk budgeting
4. Robust optimization under uncertainty
5. Transaction cost awareness

Author: Anonymous
Date: November 2025
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
from scipy import stats
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class OptimizationObjective(Enum):
    """Optimization objective types"""
    MEAN_VARIANCE = "mean_variance"
    MAX_SHARPE = "max_sharpe"
    MIN_VARIANCE = "min_variance"
    RISK_PARITY = "risk_parity"
    MAX_DIVERSIFICATION = "max_diversification"
    ROBUST_MVO = "robust_mvo"


@dataclass
class RegimeParameters:
    """Regime-specific optimization parameters"""
    risk_aversion: float  # γ in mean-variance utility
    expected_returns: np.ndarray
    covariance_matrix: np.ndarray
    factor_loadings: Optional[np.ndarray] = None
    return_uncertainty: float = 0.0  # For robust optimization
    cov_uncertainty: float = 0.0


@dataclass
class PortfolioConstraints:
    """Portfolio constraints"""
    min_weight: float = 0.0
    max_weight: float = 0.30
    max_sector_weight: Dict[str, float] = field(default_factory=dict)
    max_turnover: float = 0.30
    min_holdings: int = 5
    max_holdings: int = 25
    target_volatility: Optional[float] = None
    max_tracking_error: Optional[float] = None
    sector_mapping: Dict[str, str] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    """Result of portfolio optimization"""
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    risk_contribution: np.ndarray
    sector_weights: Dict[str, float]
    turnover: float
    objective_value: float
    success: bool
    message: str


class RegimeAdaptiveOptimizer:
    """
    Portfolio optimizer that adapts risk aversion and constraints
    based on detected market regime.
    """
    
    # Default risk aversion by regime
    DEFAULT_RISK_AVERSION = {
        0: 2.0,   # Bull/Low Vol - more aggressive
        1: 3.0,   # Normal
        2: 5.0,   # Bear/High Vol - conservative
        3: 8.0,   # Crisis - very conservative
        4: 2.5    # Recovery - moderately aggressive
    }
    
    def __init__(self,
                 tickers: List[str],
                 sector_mapping: Optional[Dict[str, str]] = None,
                 risk_free_rate: float = 0.05):
        self.tickers = tickers
        self.n_assets = len(tickers)
        self.sector_mapping = sector_mapping or {}
        self.risk_free_rate = risk_free_rate
        self.current_weights = np.ones(self.n_assets) / self.n_assets
    
    def optimize(self,
                regime: int,
                expected_returns: np.ndarray,
                covariance_matrix: np.ndarray,
                constraints: Optional[PortfolioConstraints] = None,
                objective: OptimizationObjective = OptimizationObjective.MEAN_VARIANCE,
                custom_risk_aversion: Optional[float] = None,
                return_uncertainty: float = 0.0
                ) -> OptimizationResult:
        """
        Optimize portfolio with regime-adaptive parameters.
        
        Parameters:
        -----------
        regime : int
            Current market regime (0-4)
        expected_returns : np.ndarray
            Expected returns vector
        covariance_matrix : np.ndarray
            Covariance matrix
        constraints : PortfolioConstraints
            Portfolio constraints
        objective : OptimizationObjective
            Optimization objective type
        custom_risk_aversion : float, optional
            Override default risk aversion
        return_uncertainty : float
            Uncertainty in return estimates (for robust optimization)
        
        Returns:
        --------
        OptimizationResult
        """
        if constraints is None:
            constraints = PortfolioConstraints()
        
        # Get regime-specific risk aversion
        risk_aversion = custom_risk_aversion or self.DEFAULT_RISK_AVERSION.get(regime, 3.0)
        
        logger.info(f"Optimizing with regime={regime}, γ={risk_aversion:.1f}")
        
        # Create regime parameters
        params = RegimeParameters(
            risk_aversion=risk_aversion,
            expected_returns=expected_returns,
            covariance_matrix=covariance_matrix,
            return_uncertainty=return_uncertainty
        )
        
        # Select optimization method
        if objective == OptimizationObjective.MEAN_VARIANCE:
            result = self._mean_variance_optimize(params, constraints)
        elif objective == OptimizationObjective.MAX_SHARPE:
            result = self._max_sharpe_optimize(params, constraints)
        elif objective == OptimizationObjective.MIN_VARIANCE:
            result = self._min_variance_optimize(params, constraints)
        elif objective == OptimizationObjective.RISK_PARITY:
            result = self._risk_parity_optimize(params, constraints)
        elif objective == OptimizationObjective.ROBUST_MVO:
            result = self._robust_mvo_optimize(params, constraints)
        else:
            result = self._mean_variance_optimize(params, constraints)
        
        # Update current weights
        if result.success:
            self.current_weights = result.weights
        
        return result
    
    def _mean_variance_optimize(self,
                               params: RegimeParameters,
                               constraints: PortfolioConstraints
                               ) -> OptimizationResult:
        """
        Mean-variance optimization with regime-specific risk aversion.
        
        Maximize: μ'w - (γ/2) * w'Σw
        """
        mu = params.expected_returns
        sigma = params.covariance_matrix
        gamma = params.risk_aversion
        
        def objective(w):
            ret = np.dot(mu, w)
            risk = np.dot(w, np.dot(sigma, w))
            utility = ret - (gamma / 2) * risk
            return -utility  # Minimize negative utility
        
        def objective_gradient(w):
            grad_ret = mu
            grad_risk = 2 * np.dot(sigma, w)
            return -(grad_ret - (gamma / 2) * grad_risk)
        
        # Constraints
        cons = self._build_constraints(constraints)
        bounds = [(constraints.min_weight, constraints.max_weight) 
                 for _ in range(self.n_assets)]
        
        # Initial guess
        x0 = self.current_weights.copy()
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            jac=objective_gradient,
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 500, 'ftol': 1e-8}
        )
        
        return self._create_result(result, params, constraints)
    
    def _max_sharpe_optimize(self,
                            params: RegimeParameters,
                            constraints: PortfolioConstraints
                            ) -> OptimizationResult:
        """
        Maximize Sharpe ratio.
        Uses transformation to convert to convex problem.
        """
        mu = params.expected_returns - self.risk_free_rate
        sigma = params.covariance_matrix
        
        def neg_sharpe(w):
            ret = np.dot(mu, w)
            vol = np.sqrt(np.dot(w, np.dot(sigma, w)))
            return -ret / (vol + 1e-10)
        
        cons = self._build_constraints(constraints)
        bounds = [(constraints.min_weight, constraints.max_weight) 
                 for _ in range(self.n_assets)]
        
        x0 = self.current_weights.copy()
        
        result = minimize(
            neg_sharpe,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 500}
        )
        
        return self._create_result(result, params, constraints)
    
    def _min_variance_optimize(self,
                              params: RegimeParameters,
                              constraints: PortfolioConstraints
                              ) -> OptimizationResult:
        """Minimize portfolio variance."""
        sigma = params.covariance_matrix
        
        def portfolio_variance(w):
            return np.dot(w, np.dot(sigma, w))
        
        cons = self._build_constraints(constraints)
        bounds = [(constraints.min_weight, constraints.max_weight) 
                 for _ in range(self.n_assets)]
        
        x0 = self.current_weights.copy()
        
        result = minimize(
            portfolio_variance,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )
        
        return self._create_result(result, params, constraints)
    
    def _risk_parity_optimize(self,
                             params: RegimeParameters,
                             constraints: PortfolioConstraints
                             ) -> OptimizationResult:
        """
        Risk parity optimization.
        Each asset contributes equally to portfolio risk.
        """
        sigma = params.covariance_matrix
        n = self.n_assets
        
        def risk_contribution(w):
            """Calculate marginal risk contribution of each asset"""
            portfolio_vol = np.sqrt(np.dot(w, np.dot(sigma, w)))
            marginal_contrib = np.dot(sigma, w) / (portfolio_vol + 1e-10)
            risk_contrib = w * marginal_contrib
            return risk_contrib
        
        def risk_parity_objective(w):
            """Minimize sum of squared differences from equal risk contribution"""
            rc = risk_contribution(w)
            target_rc = rc.sum() / n
            return np.sum((rc - target_rc) ** 2)
        
        # Constraints: long only, fully invested
        cons = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        bounds = [(0.01, constraints.max_weight) for _ in range(n)]
        
        # Start with equal weight
        x0 = np.ones(n) / n
        
        result = minimize(
            risk_parity_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )
        
        return self._create_result(result, params, constraints)
    
    def _robust_mvo_optimize(self,
                            params: RegimeParameters,
                            constraints: PortfolioConstraints
                            ) -> OptimizationResult:
        """
        Robust mean-variance optimization.
        Accounts for uncertainty in return estimates.
        
        Uses worst-case optimization within uncertainty set.
        """
        mu = params.expected_returns
        sigma = params.covariance_matrix
        gamma = params.risk_aversion
        kappa = params.return_uncertainty  # Uncertainty parameter
        
        def robust_objective(w):
            # Expected return with penalty for uncertainty
            ret = np.dot(mu, w)
            
            # Uncertainty penalty (worst-case adjustment)
            if kappa > 0:
                ret_uncertainty = kappa * np.sqrt(np.dot(w, np.dot(sigma, w)))
                ret = ret - ret_uncertainty  # Pessimistic return estimate
            
            # Risk term
            risk = np.dot(w, np.dot(sigma, w))
            
            utility = ret - (gamma / 2) * risk
            return -utility
        
        cons = self._build_constraints(constraints)
        bounds = [(constraints.min_weight, constraints.max_weight) 
                 for _ in range(self.n_assets)]
        
        x0 = self.current_weights.copy()
        
        result = minimize(
            robust_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )
        
        return self._create_result(result, params, constraints)
    
    def _build_constraints(self, constraints: PortfolioConstraints) -> List[Dict]:
        """Build scipy constraints from PortfolioConstraints"""
        cons = []
        
        # Fully invested constraint
        cons.append({
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1.0
        })
        
        # Turnover constraint (if we have current weights)
        if constraints.max_turnover < 1.0:
            current = self.current_weights
            cons.append({
                'type': 'ineq',
                'fun': lambda w: constraints.max_turnover - np.sum(np.abs(w - current))
            })
        
        # Sector constraints
        if constraints.max_sector_weight and self.sector_mapping:
            sectors = set(self.sector_mapping.values())
            for sector in sectors:
                sector_indices = [i for i, t in enumerate(self.tickers) 
                                if self.sector_mapping.get(t) == sector]
                if sector_indices:
                    max_weight = constraints.max_sector_weight.get(sector, 0.35)
                    cons.append({
                        'type': 'ineq',
                        'fun': lambda w, idx=sector_indices, m=max_weight: 
                               m - np.sum(w[idx])
                    })
        
        # Target volatility constraint
        if constraints.target_volatility:
            target = constraints.target_volatility
            cons.append({
                'type': 'eq',
                'fun': lambda w, t=target: 
                       np.sqrt(np.dot(w, np.dot(self._cov, w))) - t
            })
        
        return cons
    
    def _create_result(self,
                      opt_result,
                      params: RegimeParameters,
                      constraints: PortfolioConstraints) -> OptimizationResult:
        """Create OptimizationResult from scipy result"""
        
        weights = opt_result.x
        
        # Calculate metrics
        exp_ret = np.dot(params.expected_returns, weights)
        vol = np.sqrt(np.dot(weights, np.dot(params.covariance_matrix, weights)))
        sharpe = (exp_ret - self.risk_free_rate) / (vol + 1e-10)
        
        # Risk contribution
        portfolio_vol = vol
        marginal_contrib = np.dot(params.covariance_matrix, weights) / (portfolio_vol + 1e-10)
        risk_contrib = weights * marginal_contrib
        risk_contrib = risk_contrib / risk_contrib.sum()  # Normalize
        
        # Sector weights
        sector_weights = {}
        if self.sector_mapping:
            for ticker, sector in self.sector_mapping.items():
                idx = self.tickers.index(ticker) if ticker in self.tickers else -1
                if idx >= 0:
                    sector_weights[sector] = sector_weights.get(sector, 0) + weights[idx]
        
        # Turnover
        turnover = np.sum(np.abs(weights - self.current_weights))
        
        return OptimizationResult(
            weights=weights,
            expected_return=exp_ret,
            volatility=vol,
            sharpe_ratio=sharpe,
            risk_contribution=risk_contrib,
            sector_weights=sector_weights,
            turnover=turnover,
            objective_value=opt_result.fun,
            success=opt_result.success,
            message=opt_result.message if hasattr(opt_result, 'message') else ""
        )
    
    def get_regime_risk_aversion(self, regime: int) -> float:
        """Get risk aversion for a given regime"""
        return self.DEFAULT_RISK_AVERSION.get(regime, 3.0)
    
    def set_regime_risk_aversion(self, regime: int, risk_aversion: float) -> None:
        """Set custom risk aversion for a regime"""
        self.DEFAULT_RISK_AVERSION[regime] = risk_aversion


class BlackLittermanOptimizer:
    """
    Black-Litterman model for combining market equilibrium with views.
    Useful for incorporating regime-based views.
    """
    
    def __init__(self,
                 market_caps: np.ndarray,
                 covariance: np.ndarray,
                 risk_aversion: float = 2.5,
                 tau: float = 0.05):
        self.market_caps = market_caps
        self.covariance = covariance
        self.risk_aversion = risk_aversion
        self.tau = tau
        
        # Calculate market weights
        self.market_weights = market_caps / market_caps.sum()
        
        # Implied equilibrium returns (reverse optimization)
        self.equilibrium_returns = self._calculate_equilibrium_returns()
    
    def _calculate_equilibrium_returns(self) -> np.ndarray:
        """Calculate implied equilibrium returns from market weights"""
        return self.risk_aversion * np.dot(self.covariance, self.market_weights)
    
    def posterior_returns(self,
                         views: np.ndarray,
                         view_matrix: np.ndarray,
                         view_confidence: np.ndarray) -> np.ndarray:
        """
        Calculate posterior expected returns incorporating views.
        
        Parameters:
        -----------
        views : np.ndarray
            Vector of view returns (K views)
        view_matrix : np.ndarray
            K x N matrix mapping assets to views
        view_confidence : np.ndarray
            Diagonal elements of uncertainty matrix (1/confidence)
        
        Returns:
        --------
        np.ndarray : Posterior expected returns
        """
        P = view_matrix
        Q = views
        tau_sigma = self.tau * self.covariance
        
        # Omega - uncertainty of views
        omega = np.diag(view_confidence)
        
        # Black-Litterman formula
        # E[R] = [(τΣ)^(-1) + P'Ω^(-1)P]^(-1) [(τΣ)^(-1)π + P'Ω^(-1)Q]
        
        M1 = np.linalg.inv(tau_sigma)
        M2 = np.dot(P.T, np.dot(np.linalg.inv(omega), P))
        M3 = np.dot(P.T, np.dot(np.linalg.inv(omega), Q))
        
        posterior = np.linalg.solve(M1 + M2, 
                                   np.dot(M1, self.equilibrium_returns) + M3)
        
        return posterior


if __name__ == "__main__":
    # Test the optimizer
    np.random.seed(42)
    
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    n = len(tickers)
    
    # Generate sample data
    expected_returns = np.array([0.12, 0.10, 0.08, 0.15, 0.11])
    
    # Random correlation matrix
    A = np.random.randn(n, n)
    corr = np.dot(A, A.T)
    corr = corr / np.sqrt(np.outer(np.diag(corr), np.diag(corr)))
    
    vols = np.array([0.25, 0.22, 0.28, 0.30, 0.35])
    cov = corr * np.outer(vols, vols)
    
    # Test different regimes
    optimizer = RegimeAdaptiveOptimizer(tickers)
    
    print("Regime-Adaptive Portfolio Optimization Test")
    print("=" * 60)
    
    for regime in range(5):
        result = optimizer.optimize(
            regime=regime,
            expected_returns=expected_returns,
            covariance_matrix=cov,
            objective=OptimizationObjective.MEAN_VARIANCE
        )
        
        regime_names = {0: 'Bull', 1: 'Normal', 2: 'Bear', 3: 'Crisis', 4: 'Recovery'}
        gamma = optimizer.get_regime_risk_aversion(regime)
        
        print(f"\nRegime {regime} ({regime_names[regime]}), γ = {gamma:.1f}:")
        print(f"  Expected Return: {result.expected_return:.2%}")
        print(f"  Volatility: {result.volatility:.2%}")
        print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"  Weights: {dict(zip(tickers, np.round(result.weights, 3)))}")
