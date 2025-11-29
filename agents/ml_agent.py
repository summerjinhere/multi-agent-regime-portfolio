"""
ML Agent - Regime Detection
============================
Hidden Markov Model based market regime detection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Type
from hmmlearn.hmm import GaussianHMM
import warnings

from agents.base_agent import BaseAgent
from events.message_bus import MessageBus
from events.base_event import Event, MarketDataEvent, RegimeEvent


class MLAgent(BaseAgent):
    """
    Machine Learning Agent for market regime detection using Hidden Markov Models.
    
    Responsibilities:
    - Feature engineering (returns, volatility, momentum)
    - HMM model fitting
    - Regime state prediction
    - Transition probability estimation
    
    Regimes:
    - 0: Bull / Low Volatility
    - 1: Transition / Medium Volatility
    - 2: Bear / High Volatility
    
    Config Options:
        n_components: Number of HMM states (default: 3)
        covariance_type: HMM covariance type ('full', 'diag', 'spherical')
        n_iter: Maximum EM iterations
        feature_window: Rolling window for volatility calculation
        min_observations: Minimum data points for fitting
    """
    
    REGIME_NAMES = {
        0: "Bull/LowVol",
        1: "Transition",
        2: "Bear/HighVol"
    }
    
    def __init__(self, message_bus: MessageBus, config: Optional[Dict] = None):
        super().__init__("MLAgent", message_bus, config)
        
        # Model configuration
        self.n_components = self.get_config('n_components', 3)
        self.covariance_type = self.get_config('covariance_type', 'full')
        self.n_iter = self.get_config('n_iter', 1000)
        self.feature_window = self.get_config('feature_window', 10)
        self.min_observations = self.get_config('min_observations', 100)
        
        # Model state
        self.model: Optional[GaussianHMM] = None
        self.features: Optional[pd.DataFrame] = None
        self.hidden_states: Optional[np.ndarray] = None
        self.state_mapping: Dict[int, int] = {}
        
        # Market data (received from MarketDataAgent)
        self.market_data: Optional[MarketDataEvent] = None
    
    def _initialize(self) -> None:
        """Initialize HMM model."""
        self.model = GaussianHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=42
        )
        self.logger.info(f"Initialized HMM with {self.n_components} states")
    
    def get_subscribed_events(self) -> List[Type[Event]]:
        """Subscribe to MarketDataEvent."""
        return [MarketDataEvent]
    
    def on_event(self, event: Event) -> None:
        """Handle incoming MarketDataEvent."""
        if isinstance(event, MarketDataEvent):
            self.market_data = event
            self.logger.info("Received MarketDataEvent, ready for processing")
    
    def _process(self) -> None:
        """
        Main processing:
        1. Engineer features from market data
        2. Fit HMM
        3. Detect current regime
        4. Publish RegimeEvent
        """
        if self.market_data is None or self.market_data.prices is None:
            raise ValueError("No market data available. Ensure MarketDataAgent ran first.")
        
        prices = self.market_data.prices
        
        # Create market proxy (equal-weighted average)
        market_proxy = prices.mean(axis=1)
        
        # Engineer features
        self.features = self._engineer_features(market_proxy)
        
        if len(self.features) < self.min_observations:
            raise ValueError(f"Insufficient data: {len(self.features)} < {self.min_observations}")
        
        # Fit HMM
        self._fit_model()
        
        # Get current regime and probabilities
        current_regime = self._get_current_regime()
        regime_probs = self._get_regime_probabilities()
        transition_matrix = self._get_transition_matrix()
        regime_duration = self._calculate_regime_duration()
        
        # Publish event
        event = RegimeEvent(
            source_agent=self.name,
            current_regime=current_regime,
            regime_name=self.REGIME_NAMES.get(current_regime, "Unknown"),
            regime_probabilities=regime_probs,
            transition_matrix=transition_matrix,
            regime_duration=regime_duration,
            regime_history=pd.Series(self.hidden_states, index=self.features.index),
            model_confidence=max(regime_probs.values()) if regime_probs else 0.0,
            features_used=['Returns', 'Volatility']
        )
        
        self.publish(event)
        self.logger.info(f"Published RegimeEvent: Regime {current_regime} ({self.REGIME_NAMES.get(current_regime)}) "
                        f"with confidence {event.model_confidence:.2%}")
    
    def _engineer_features(self, price_series: pd.Series) -> pd.DataFrame:
        """
        Create features for HMM.
        
        Features:
        - Returns: Daily log returns
        - Volatility: Rolling standard deviation of returns
        
        Args:
            price_series: Price series (typically market proxy)
            
        Returns:
            DataFrame with feature columns
        """
        # Log returns (more stable than simple returns)
        returns = np.log(price_series / price_series.shift(1))
        
        # Rolling volatility
        volatility = returns.rolling(window=self.feature_window).std()
        
        # Additional features (optional)
        # momentum = returns.rolling(window=20).mean()
        # skewness = returns.rolling(window=20).skew()
        
        features = pd.DataFrame({
            'Returns': returns,
            'Volatility': volatility
        }).dropna()
        
        self.logger.debug(f"Engineered {len(features)} observations with features: {list(features.columns)}")
        return features
    
    def _fit_model(self) -> None:
        """Fit the HMM model."""
        X = self.features.values
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X)
        
        # Predict hidden states
        self.hidden_states = self.model.predict(X)
        
        # Reorder states by volatility (0 = lowest vol, N = highest vol)
        self._reorder_states_by_volatility()
        
        self.logger.info(f"HMM fitted. Converged: {self.model.monitor_.converged}")
    
    def _reorder_states_by_volatility(self) -> None:
        """
        Reorder states so that:
        - State 0 = Lowest volatility (Bull)
        - State N = Highest volatility (Bear)
        """
        state_volatilities = []
        
        for state in range(self.n_components):
            mask = (self.hidden_states == state)
            if mask.any():
                vol = self.features['Returns'][mask].std()
                state_volatilities.append((state, vol))
        
        # Sort by volatility
        sorted_states = sorted(state_volatilities, key=lambda x: x[1])
        
        # Create mapping: old_state -> new_state
        self.state_mapping = {old: new for new, (old, _) in enumerate(sorted_states)}
        
        # Apply mapping
        self.hidden_states = np.array([self.state_mapping[s] for s in self.hidden_states])
    
    def _get_current_regime(self) -> int:
        """Get the current (most recent) regime."""
        if self.hidden_states is None or len(self.hidden_states) == 0:
            return 0
        return int(self.hidden_states[-1])
    
    def _get_regime_probabilities(self) -> Dict[int, float]:
        """Get probabilities of being in each regime for the current observation."""
        if self.model is None or self.features is None:
            return {i: 1.0/self.n_components for i in range(self.n_components)}
        
        try:
            probs = self.model.predict_proba(self.features.values)[-1]
            
            # Apply state mapping to probabilities
            mapped_probs = {}
            for old_state, new_state in self.state_mapping.items():
                mapped_probs[new_state] = float(probs[old_state])
            
            return mapped_probs
        except Exception as e:
            self.logger.warning(f"Failed to get regime probabilities: {e}")
            return {i: 1.0/self.n_components for i in range(self.n_components)}
    
    def _get_transition_matrix(self) -> np.ndarray:
        """Get the transition probability matrix."""
        if self.model is None:
            return np.eye(self.n_components)
        
        try:
            # Get original transition matrix
            trans = self.model.transmat_.copy()
            
            # Reorder according to state mapping
            n = self.n_components
            reordered = np.zeros((n, n))
            
            for old_i, new_i in self.state_mapping.items():
                for old_j, new_j in self.state_mapping.items():
                    reordered[new_i, new_j] = trans[old_i, old_j]
            
            return reordered
        except Exception as e:
            self.logger.warning(f"Failed to get transition matrix: {e}")
            return np.eye(self.n_components)
    
    def _calculate_regime_duration(self) -> int:
        """Calculate how many days we've been in the current regime."""
        if self.hidden_states is None or len(self.hidden_states) == 0:
            return 0
        
        current = self.hidden_states[-1]
        duration = 1
        
        for state in reversed(self.hidden_states[:-1]):
            if state == current:
                duration += 1
            else:
                break
        
        return duration
    
    # =========================================================================
    # Public API
    # =========================================================================
    
    def get_regime_statistics(self) -> Dict[int, Dict]:
        """Get statistics for each regime."""
        if self.features is None or self.hidden_states is None:
            return {}
        
        stats = {}
        for state in range(self.n_components):
            mask = (self.hidden_states == state)
            if mask.any():
                returns = self.features['Returns'][mask]
                stats[state] = {
                    'name': self.REGIME_NAMES.get(state, f"State {state}"),
                    'frequency': mask.mean(),
                    'mean_return': returns.mean(),
                    'volatility': returns.std(),
                    'sharpe': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
                    'count': mask.sum()
                }
        
        return stats
    
    def predict_regime(self, new_prices: pd.Series) -> int:
        """Predict regime for new data."""
        features = self._engineer_features(new_prices)
        if len(features) == 0 or self.model is None:
            return 0
        
        predicted = self.model.predict(features.values)
        return self.state_mapping.get(predicted[-1], 0)
    
    def get_regime_for_date(self, date: pd.Timestamp) -> Optional[int]:
        """Get regime for a specific date."""
        if self.features is None or self.hidden_states is None:
            return None
        
        if date in self.features.index:
            idx = self.features.index.get_loc(date)
            return int(self.hidden_states[idx])
        
        return None
