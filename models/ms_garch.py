"""
Markov-Switching GARCH Volatility Model
========================================
Implements regime-dependent GARCH volatility inspired by:
1. Factor Overnight GARCH-Itô Models - time-varying volatility
2. Wavelet-Multivariate MS-GARCH - regime-dependent variance

This replaces the simple HMM with a more sophisticated MS-GARCH approach.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MarkovSwitchingGARCH:
    """
    Markov Switching GARCH Model for regime-dependent volatility.
    
    Features:
    - Multiple volatility regimes
    - Time-varying conditional variance
    - Regime-dependent GARCH parameters
    """
    
    def __init__(self, n_regimes: int = 3):
        self.n_regimes = n_regimes
        self.transition_matrix = None
        self.regime_params = {}
        self.smoothed_probs = None
        self.current_regime = 0
        
    def fit(self, returns: pd.DataFrame, max_iter: int = 100) -> 'MarkovSwitchingGARCH':
        """
        Fit MS-GARCH model to return data.
        
        Args:
            returns: DataFrame of asset returns
            max_iter: Maximum EM iterations
            
        Returns:
            self
        """
        n_obs = len(returns)
        
        # Initialize parameters for each regime
        for regime in range(self.n_regimes):
            self.regime_params[regime] = {
                'omega': 0.00001 * (regime + 1),  # Base variance
                'alpha': 0.05 + (regime * 0.05),  # ARCH term
                'beta': 0.90 - (regime * 0.10),   # GARCH term
                'mean': returns.mean().mean() * (1 + regime * 0.1)
            }
        
        # Initialize transition matrix
        self.transition_matrix = self._initialize_transition_matrix()
        
        # EM algorithm
        for iteration in range(max_iter):
            # E-step: Calculate regime probabilities
            filtered_probs, log_likelihood = self._filter_regimes(returns)
            self.smoothed_probs = self._smooth_regimes(filtered_probs)
            
            # M-step: Update parameters
            self._update_parameters(returns, self.smoothed_probs)
            self._update_transition_matrix(self.smoothed_probs)
            
            if iteration % 10 == 0:
                logger.debug(f"MS-GARCH iteration {iteration}, log-likelihood: {log_likelihood:.2f}")
        
        # Set current regime as most probable
        self.current_regime = int(self.smoothed_probs[-1].argmax())
        
        logger.info(f"MS-GARCH fitted: {self.n_regimes} regimes, current regime: {self.current_regime}")
        return self
    
    def _initialize_transition_matrix(self) -> np.ndarray:
        """Initialize transition probability matrix."""
        # Start with persistent regimes
        P = np.ones((self.n_regimes, self.n_regimes)) * 0.05
        for i in range(self.n_regimes):
            P[i, i] = 0.85  # High persistence
        return P
    
    def _filter_regimes(self, returns: pd.DataFrame) -> Tuple[np.ndarray, float]:
        """
        Forward filtering to get regime probabilities.
        
        Returns:
            filtered_probs: (n_obs, n_regimes) array
            log_likelihood: Model fit
        """
        n_obs = len(returns)
        filtered_probs = np.zeros((n_obs, self.n_regimes))
        log_likelihood = 0
        
        # Initialize with stationary distribution
        filtered_probs[0] = np.ones(self.n_regimes) / self.n_regimes
        
        # Convert DataFrame to array for faster computation
        returns_array = returns.values
        
        for t in range(1, n_obs):
            # Predict
            predicted = filtered_probs[t-1] @ self.transition_matrix
            
            # Calculate likelihood for each regime
            likelihoods = np.zeros(self.n_regimes)
            for regime in range(self.n_regimes):
                # GARCH(1,1) conditional variance
                params = self.regime_params[regime]
                if t > 1:
                    prev_return = returns_array[t-1].mean()
                    prev_var = self._garch_variance(t-1, regime, returns_array)
                    variance = params['omega'] + params['alpha'] * prev_return**2 + params['beta'] * prev_var
                else:
                    variance = params['omega'] / (1 - params['alpha'] - params['beta'])
                
                variance = max(variance, 1e-6)  # Numerical stability
                
                # Gaussian likelihood
                curr_return = returns_array[t].mean()
                likelihoods[regime] = np.exp(-0.5 * (curr_return - params['mean'])**2 / variance) / np.sqrt(2 * np.pi * variance)
            
            # Update
            filtered_probs[t] = predicted * likelihoods
            filtered_probs[t] /= filtered_probs[t].sum() + 1e-10
            
            log_likelihood += np.log(filtered_probs[t].sum() + 1e-10)
        
        return filtered_probs, log_likelihood
    
    def _garch_variance(self, t: int, regime: int, returns: np.ndarray) -> float:
        """Calculate GARCH conditional variance at time t for given regime."""
        params = self.regime_params[regime]
        
        if t <= 0:
            return params['omega'] / (1 - params['alpha'] - params['beta'])
        
        prev_return = returns[t].mean()
        prev_var = self._garch_variance(t-1, regime, returns)
        
        variance = params['omega'] + params['alpha'] * prev_return**2 + params['beta'] * prev_var
        return max(variance, 1e-6)
    
    def _smooth_regimes(self, filtered_probs: np.ndarray) -> np.ndarray:
        """
        Backward smoothing for regime probabilities.
        
        Returns:
            smoothed_probs: (n_obs, n_regimes) array
        """
        n_obs = filtered_probs.shape[0]
        smoothed_probs = filtered_probs.copy()
        
        # Backward pass
        for t in range(n_obs - 2, -1, -1):
            smoothed_probs[t] = filtered_probs[t] * (
                self.transition_matrix @ (smoothed_probs[t+1] / (filtered_probs[t] @ self.transition_matrix + 1e-10))
            )
            smoothed_probs[t] /= smoothed_probs[t].sum() + 1e-10
        
        return smoothed_probs
    
    def _update_parameters(self, returns: pd.DataFrame, probs: np.ndarray):
        """Update GARCH parameters for each regime."""
        returns_array = returns.values
        
        for regime in range(self.n_regimes):
            # Weighted statistics
            weights = probs[:, regime]
            
            # Update mean
            self.regime_params[regime]['mean'] = np.average(
                returns_array.mean(axis=1), weights=weights
            )
            
            # Update GARCH params (simplified MLE)
            # In practice, would use numerical optimization
            residuals = returns_array.mean(axis=1) - self.regime_params[regime]['mean']
            weighted_var = np.average(residuals**2, weights=weights)
            
            # Simple update (could be improved with full MLE)
            self.regime_params[regime]['omega'] = weighted_var * 0.1
            self.regime_params[regime]['alpha'] = min(0.15, weighted_var * 0.3)
            self.regime_params[regime]['beta'] = 0.85 - self.regime_params[regime]['alpha']
    
    def _update_transition_matrix(self, probs: np.ndarray):
        """Update transition probability matrix."""
        n_obs = len(probs)
        
        # Count transitions (weighted)
        counts = np.zeros((self.n_regimes, self.n_regimes))
        
        for t in range(n_obs - 1):
            for i in range(self.n_regimes):
                for j in range(self.n_regimes):
                    counts[i, j] += probs[t, i] * probs[t+1, j]
        
        # Normalize to get probabilities
        self.transition_matrix = counts / (counts.sum(axis=1, keepdims=True) + 1e-10)
    
    def predict_volatility(self, horizon: int = 1) -> Dict[int, float]:
        """
        Predict volatility for each regime.
        
        Args:
            horizon: Forecast horizon (periods)
            
        Returns:
            Dict mapping regime to predicted volatility
        """
        volatilities = {}
        
        for regime in range(self.n_regimes):
            params = self.regime_params[regime]
            # Long-run variance
            long_run_var = params['omega'] / (1 - params['alpha'] - params['beta'])
            volatilities[regime] = np.sqrt(long_run_var)
        
        return volatilities
    
    def get_regime_features(self) -> pd.DataFrame:
        """Get regime characteristics as DataFrame."""
        features = []
        
        for regime in range(self.n_regimes):
            params = self.regime_params[regime]
            long_run_vol = np.sqrt(params['omega'] / (1 - params['alpha'] - params['beta']))
            
            features.append({
                'Regime': regime,
                'Mean': params['mean'],
                'Volatility': long_run_vol,
                'Persistence': params['beta'],
                'Shock_Response': params['alpha']
            })
        
        return pd.DataFrame(features)


def integrate_oil_shocks(returns: pd.DataFrame, oil_returns: pd.Series) -> pd.DataFrame:
    """
    Integrate oil shock transmission into return data.
    
    Based on "Oil shock transmission to stock market returns" paper.
    
    Args:
        returns: Stock returns DataFrame
        oil_returns: Oil price returns Series
        
    Returns:
        Adjusted returns with oil shock component
    """
    if len(oil_returns) == 0 or oil_returns.isna().all():
        logger.warning("Oil returns unavailable, skipping oil shock integration")
        return returns
    
    # Align indices
    common_idx = returns.index.intersection(oil_returns.index)
    if len(common_idx) < 10:
        logger.warning(f"Insufficient overlapping data ({len(common_idx)} points)")
        return returns
    
    returns_aligned = returns.loc[common_idx]
    oil_aligned = oil_returns.loc[common_idx]
    
    # Simple linear transmission (could be enhanced with wavelets)
    oil_beta = 0.1  # Simplified transmission coefficient
    adjusted_returns = returns_aligned + oil_beta * oil_aligned.values.reshape(-1, 1)
    
    return adjusted_returns
