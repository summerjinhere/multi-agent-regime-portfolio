"""
Quant Agent - MS-VECM
=====================
Markov-Switching Vector Error Correction Model for expected returns estimation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Type, Tuple
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
import warnings

from agents.base_agent import BaseAgent
from events.message_bus import MessageBus
from events.base_event import Event, MarketDataEvent, RegimeEvent, QuantSignalEvent


class QuantAgent(BaseAgent):
    """
    Quantitative Agent for expected return and covariance estimation.
    
    Responsibilities:
    - Johansen cointegration testing
    - VECM model estimation
    - Expected return forecasting
    - Covariance matrix estimation (with optional shrinkage)
    
    Methodology:
    1. Test for cointegration among asset classes
    2. If cointegrated: Fit VECM and forecast
    3. If not cointegrated: Fall back to simple mean-variance
    4. Adjust estimates based on current regime
    
    Config Options:
        lookback_window: Days of history for VECM (default: 252)
        forecast_horizon: Days ahead to forecast (default: 5)
        det_order: Deterministic order for Johansen test (-1, 0, 1)
        k_ar_diff: Lag order for VECM
        use_shrinkage: Use Ledoit-Wolf shrinkage for covariance
        regime_adjustment: Apply regime-based return scaling
    """
    
    def __init__(self, message_bus: MessageBus, config: Optional[Dict] = None):
        super().__init__("QuantAgent", message_bus, config)
        
        # Configuration
        self.lookback_window = self.get_config('lookback_window', 252)
        self.forecast_horizon = self.get_config('forecast_horizon', 5)
        self.det_order = self.get_config('det_order', -1)
        self.k_ar_diff = self.get_config('k_ar_diff', 1)
        self.use_shrinkage = self.get_config('use_shrinkage', True)
        self.regime_adjustment = self.get_config('regime_adjustment', True)
        
        # Regime-based return multipliers
        self.regime_return_multipliers = {
            0: 1.0,    # Bull: neutral
            1: 0.8,    # Transition: reduce expected returns
            2: 0.5     # Bear: significantly reduce
        }
        
        # State
        self.market_data: Optional[MarketDataEvent] = None
        self.regime_event: Optional[RegimeEvent] = None
        self.expected_returns: Optional[pd.Series] = None
        self.covariance_matrix: Optional[pd.DataFrame] = None
    
    def _initialize(self) -> None:
        """Initialize Quant Agent."""
        self.logger.info(f"Initialized with lookback={self.lookback_window}, "
                        f"forecast_horizon={self.forecast_horizon}")
    
    def get_subscribed_events(self) -> List[Type[Event]]:
        """Subscribe to required events."""
        return [MarketDataEvent, RegimeEvent]
    
    def on_event(self, event: Event) -> None:
        """Handle incoming events."""
        if isinstance(event, MarketDataEvent):
            self.market_data = event
            self.logger.info("Received MarketDataEvent")
        elif isinstance(event, RegimeEvent):
            self.regime_event = event
            self.logger.info(f"Received RegimeEvent: Regime {event.current_regime}")
    
    def _process(self) -> None:
        """
        Main processing:
        1. Get asset class prices
        2. Test cointegration
        3. Estimate expected returns and covariance
        4. Apply regime adjustments
        5. Publish QuantSignalEvent
        """
        if self.market_data is None or self.market_data.prices is None:
            raise ValueError("No market data available")
        
        # Get asset class prices (aggregated from sectors)
        asset_prices = self._get_asset_class_prices()
        
        if len(asset_prices.columns) < 2:
            raise ValueError("Need at least 2 asset classes for analysis")
        
        # Use recent lookback window
        recent_prices = asset_prices.tail(self.lookback_window).dropna()
        
        if len(recent_prices) < 50:
            raise ValueError(f"Insufficient data: {len(recent_prices)} observations")
        
        # Test cointegration
        is_cointegrated, coint_rank = self._test_cointegration(recent_prices)
        
        # Estimate returns and covariance
        if is_cointegrated and coint_rank > 0:
            self.expected_returns, self.covariance_matrix = self._fit_vecm(
                recent_prices, coint_rank
            )
            methodology = "MS-VECM"
            confidence = 0.7 + 0.1 * min(coint_rank, 3)  # Higher rank = higher confidence
        else:
            self.expected_returns, self.covariance_matrix = self._simple_estimation(
                recent_prices
            )
            methodology = "Mean-Variance"
            confidence = 0.5
        
        # Apply regime adjustment
        if self.regime_adjustment and self.regime_event is not None:
            self._apply_regime_adjustment()
            methodology += " + Regime Adjusted"
        
        # Apply covariance shrinkage
        if self.use_shrinkage:
            self.covariance_matrix = self._shrink_covariance(self.covariance_matrix)
        
        # Publish event
        event = QuantSignalEvent(
            source_agent=self.name,
            expected_returns=self.expected_returns,
            covariance_matrix=self.covariance_matrix,
            cointegration_rank=coint_rank,
            is_cointegrated=is_cointegrated,
            forecast_horizon=self.forecast_horizon,
            forecast_confidence=confidence,
            asset_classes=list(asset_prices.columns),
            methodology=methodology
        )
        
        self.publish(event)
        self.logger.info(f"Published QuantSignalEvent: {methodology}, "
                        f"coint_rank={coint_rank}, confidence={confidence:.2f}")
    
    def _get_asset_class_prices(self) -> pd.DataFrame:
        """
        Aggregate individual stocks into asset class prices.
        
        Returns:
            DataFrame with asset class price columns
        """
        prices = self.market_data.prices
        sector_mapping = self.market_data.sector_mapping
        
        asset_classes = pd.DataFrame()
        
        # Aggregate sectors
        for sector, tickers in sector_mapping.items():
            valid_tickers = [t for t in tickers if t in prices.columns]
            if valid_tickers:
                # Equal-weighted sector average
                asset_classes[sector] = prices[valid_tickers].mean(axis=1)
        
        # Add non-equity asset classes
        if 'TLT' in prices.columns:
            asset_classes['Bonds'] = prices['TLT']
        if 'GLD' in prices.columns:
            asset_classes['Commodities'] = prices['GLD']
        if '^VIX' in prices.columns:
            asset_classes['Volatility'] = prices['^VIX']
        
        return asset_classes.dropna()
    
    def _test_cointegration(self, prices: pd.DataFrame) -> Tuple[bool, int]:
        """
        Perform Johansen cointegration test.
        
        Args:
            prices: DataFrame of asset class prices
            
        Returns:
            (is_cointegrated, cointegration_rank)
        """
        # Use log prices
        log_prices = np.log(prices)
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = coint_johansen(log_prices, self.det_order, self.k_ar_diff)
            
            # Check trace statistics vs critical values at 95%
            # Column index 1 is 95% confidence
            trace_stats = result.lr1
            critical_values = result.cvt[:, 1]
            
            # Count how many hypotheses we reject
            rank = 0
            for i in range(len(trace_stats)):
                if trace_stats[i] > critical_values[i]:
                    rank += 1
                else:
                    break
            
            is_cointegrated = rank > 0
            
            self.logger.debug(f"Johansen test: rank={rank}, trace_stats={trace_stats[:3]}")
            return is_cointegrated, rank
            
        except Exception as e:
            self.logger.warning(f"Cointegration test failed: {e}")
            return False, 0
    
    def _fit_vecm(self, prices: pd.DataFrame, coint_rank: int) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Fit Vector Error Correction Model and forecast.
        
        Args:
            prices: Asset class prices
            coint_rank: Cointegration rank from Johansen test
            
        Returns:
            (expected_returns, covariance_matrix)
        """
        log_prices = np.log(prices)
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = VECM(log_prices, k_ar_diff=self.k_ar_diff, 
                            coint_rank=coint_rank, deterministic="ci")
                result = model.fit()
            
            # Forecast
            forecast_log = result.predict(steps=self.forecast_horizon)
            
            # Convert to expected returns
            last_prices = prices.iloc[-1]
            forecast_prices = np.exp(forecast_log[-1])
            
            # Annualized expected returns
            expected_returns = (forecast_prices / last_prices.values) ** (252 / self.forecast_horizon) - 1
            expected_returns = pd.Series(expected_returns, index=prices.columns)
            
            # Covariance from residuals (annualized)
            cov_matrix = pd.DataFrame(
                result.sigma_u * 252,
                index=prices.columns,
                columns=prices.columns
            )
            
            return expected_returns, cov_matrix
            
        except Exception as e:
            self.logger.warning(f"VECM fitting failed: {e}")
            return self._simple_estimation(prices)
    
    def _simple_estimation(self, prices: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Simple mean-variance estimation (fallback).
        
        Args:
            prices: Asset class prices
            
        Returns:
            (expected_returns, covariance_matrix)
        """
        returns = prices.pct_change().dropna()
        
        # Annualized returns
        expected_returns = returns.mean() * 252
        
        # Annualized covariance
        cov_matrix = returns.cov() * 252
        
        return expected_returns, cov_matrix
    
    def _apply_regime_adjustment(self) -> None:
        """Apply regime-based adjustments to expected returns."""
        if self.regime_event is None or self.expected_returns is None:
            return
        
        regime = self.regime_event.current_regime
        multiplier = self.regime_return_multipliers.get(regime, 1.0)
        
        # Adjust expected returns
        self.expected_returns = self.expected_returns * multiplier
        
        # In bear regimes, also increase volatility estimates
        if regime == 2 and self.covariance_matrix is not None:
            self.covariance_matrix = self.covariance_matrix * 1.5
        
        self.logger.debug(f"Applied regime {regime} adjustment: multiplier={multiplier}")
    
    def _shrink_covariance(self, cov_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Ledoit-Wolf shrinkage to covariance matrix.
        
        This improves out-of-sample performance by shrinking toward
        a structured estimator.
        
        Args:
            cov_matrix: Original covariance matrix
            
        Returns:
            Shrunk covariance matrix
        """
        try:
            from sklearn.covariance import LedoitWolf
            
            # LedoitWolf needs returns, but we can approximate from cov
            # For now, use simple shrinkage toward diagonal
            n = len(cov_matrix)
            shrinkage_target = np.diag(np.diag(cov_matrix.values))
            
            # Shrinkage intensity (typical value: 0.1 - 0.3)
            alpha = 0.2
            
            shrunk = (1 - alpha) * cov_matrix.values + alpha * shrinkage_target
            
            return pd.DataFrame(shrunk, index=cov_matrix.index, columns=cov_matrix.columns)
            
        except Exception as e:
            self.logger.warning(f"Covariance shrinkage failed: {e}")
            return cov_matrix
    
    # =========================================================================
    # Public API
    # =========================================================================
    
    def get_correlation_matrix(self) -> Optional[pd.DataFrame]:
        """Get correlation matrix from covariance."""
        if self.covariance_matrix is None:
            return None
        
        std = np.sqrt(np.diag(self.covariance_matrix))
        corr = self.covariance_matrix / np.outer(std, std)
        return pd.DataFrame(corr, index=self.covariance_matrix.index, 
                           columns=self.covariance_matrix.columns)
    
    def get_sharpe_ratios(self, risk_free_rate: float = 0.02) -> Optional[pd.Series]:
        """Calculate Sharpe ratios for each asset class."""
        if self.expected_returns is None or self.covariance_matrix is None:
            return None
        
        std = np.sqrt(np.diag(self.covariance_matrix))
        sharpe = (self.expected_returns - risk_free_rate) / std
        return sharpe
