"""
Enhanced Regime Detection Module
================================

Features:
1. HMM-based regime detection
2. Nasdaq futures positioning analysis
3. Institutional/Retail/Foreign investor flow tracking
4. Combined regime scoring

Author: Anonymous
Date: November 2025
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from enum import Enum
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class RegimeState(Enum):
    """Market regime states"""
    BULL_LOW_VOL = 0
    NORMAL = 1
    BEAR_HIGH_VOL = 2
    CRISIS = 3
    RECOVERY = 4


@dataclass
class RegimeAssessment:
    """Complete regime assessment including all factors"""
    timestamp: datetime
    primary_regime: RegimeState
    regime_probabilities: Dict[RegimeState, float]
    hmm_regime: int
    hmm_confidence: float
    futures_signal: float  # -1 to +1
    investor_flow_signal: float  # -1 to +1
    sentiment_signal: float  # -1 to +1
    combined_score: float  # -1 (bearish) to +1 (bullish)
    risk_level: float  # 0 to 1
    recommended_exposure: float  # 0 to 1
    details: Dict = field(default_factory=dict)


@dataclass
class InvestorFlow:
    """Investor flow data by type"""
    date: datetime
    institutional_net: float  # Net buying/selling
    retail_net: float
    foreign_net: float
    total_volume: float
    institutional_ratio: float  # % of total volume
    retail_ratio: float
    foreign_ratio: float


class FuturesPositioningAnalyzer:
    """
    Analyzes Nasdaq futures positioning from COT data
    and derives regime signals.
    """
    
    def __init__(self, lookback_weeks: int = 52):
        self.lookback_weeks = lookback_weeks
        self.positioning_history = []
        self.extreme_long_pct = 80  # Percentile for extreme long
        self.extreme_short_pct = 20  # Percentile for extreme short
    
    def analyze_positioning(self,
                           commercial_net: float,
                           non_commercial_net: float,
                           small_spec_net: float,
                           open_interest: float,
                           historical_data: Optional[pd.DataFrame] = None
                           ) -> Dict:
        """
        Analyze futures positioning and generate signal.
        
        Parameters:
        -----------
        commercial_net : float
            Commercial hedgers net position
        non_commercial_net : float
            Large speculators (hedge funds, CTAs) net position
        small_spec_net : float
            Small speculators net position
        open_interest : float
            Total open interest
        historical_data : pd.DataFrame
            Historical positioning data for percentile calculation
        
        Returns:
        --------
        Dict with positioning analysis
        """
        # Calculate ratios
        spec_ratio = non_commercial_net / open_interest if open_interest > 0 else 0
        commercial_ratio = commercial_net / open_interest if open_interest > 0 else 0
        
        # Historical percentile (if data available)
        if historical_data is not None and len(historical_data) >= self.lookback_weeks:
            spec_percentile = stats.percentileofscore(
                historical_data['non_commercial_net'].values[-self.lookback_weeks:],
                non_commercial_net
            )
        else:
            spec_percentile = 50  # Neutral if no history
        
        # Generate signal based on speculator positioning
        # Contrarian approach: extreme positioning often precedes reversals
        if spec_percentile > self.extreme_long_pct:
            # Speculators extremely long -> bearish contrarian signal
            signal = -((spec_percentile - self.extreme_long_pct) / 
                      (100 - self.extreme_long_pct))
            positioning_state = "extreme_long"
        elif spec_percentile < self.extreme_short_pct:
            # Speculators extremely short -> bullish contrarian signal
            signal = (self.extreme_short_pct - spec_percentile) / self.extreme_short_pct
            positioning_state = "extreme_short"
        else:
            # Neutral zone
            signal = (spec_percentile - 50) / 50 * 0.3  # Weak trend-following
            positioning_state = "neutral"
        
        # Confidence based on open interest trend
        # Rising OI with positioning = stronger signal
        
        return {
            'signal': np.clip(signal, -1, 1),
            'spec_percentile': spec_percentile,
            'spec_ratio': spec_ratio,
            'commercial_ratio': commercial_ratio,
            'positioning_state': positioning_state,
            'open_interest': open_interest
        }
    
    def get_regime_adjustment(self, positioning_signal: float) -> float:
        """
        Convert positioning signal to regime adjustment factor.
        
        Returns factor to adjust regime probabilities.
        """
        # Strong contrarian signal suggests regime change more likely
        if abs(positioning_signal) > 0.5:
            return 0.2  # Increase probability of regime change
        else:
            return 0.0


class InvestorFlowAnalyzer:
    """
    Analyzes investor flow by type (institutional, retail, foreign).
    Generates regime signals based on flow patterns.
    """
    
    def __init__(self, lookback_days: int = 20):
        self.lookback_days = lookback_days
        self.flow_history: List[InvestorFlow] = []
    
    def analyze_flow(self,
                    institutional_buy: float,
                    institutional_sell: float,
                    retail_buy: float,
                    retail_sell: float,
                    foreign_buy: float,
                    foreign_sell: float,
                    total_volume: float) -> InvestorFlow:
        """
        Analyze investor flow data.
        """
        # Calculate net flows
        inst_net = institutional_buy - institutional_sell
        retail_net = retail_buy - retail_sell
        foreign_net = foreign_buy - foreign_sell
        
        # Calculate ratios
        inst_volume = institutional_buy + institutional_sell
        retail_volume = retail_buy + retail_sell
        foreign_volume = foreign_buy + foreign_sell
        
        flow = InvestorFlow(
            date=datetime.now(),
            institutional_net=inst_net,
            retail_net=retail_net,
            foreign_net=foreign_net,
            total_volume=total_volume,
            institutional_ratio=inst_volume / total_volume if total_volume > 0 else 0,
            retail_ratio=retail_volume / total_volume if total_volume > 0 else 0,
            foreign_ratio=foreign_volume / total_volume if total_volume > 0 else 0
        )
        
        self.flow_history.append(flow)
        
        return flow
    
    def get_flow_signal(self) -> Dict:
        """
        Generate trading signal from recent flow patterns.
        
        Smart money (institutional + foreign) vs. retail.
        """
        if len(self.flow_history) < 5:
            return {'signal': 0, 'confidence': 0, 'pattern': 'insufficient_data'}
        
        recent_flows = self.flow_history[-self.lookback_days:]
        
        # Aggregate flows
        inst_net_sum = sum(f.institutional_net for f in recent_flows)
        retail_net_sum = sum(f.retail_net for f in recent_flows)
        foreign_net_sum = sum(f.foreign_net for f in recent_flows)
        
        # Smart money = institutional + foreign
        smart_money_net = inst_net_sum + foreign_net_sum
        
        # Total absolute flow for normalization
        total_flow = sum(f.total_volume for f in recent_flows)
        
        if total_flow == 0:
            return {'signal': 0, 'confidence': 0, 'pattern': 'no_volume'}
        
        # Normalize signal
        smart_money_signal = smart_money_net / total_flow
        retail_signal = retail_net_sum / total_flow
        
        # Generate signal
        # Positive when smart money buying, negative when selling
        signal = np.tanh(smart_money_signal * 10)  # Normalize to [-1, 1]
        
        # Confidence based on agreement and volume
        agreement = 1 if np.sign(smart_money_signal) != np.sign(retail_signal) else 0.5
        volume_confidence = min(1, len(recent_flows) / self.lookback_days)
        confidence = agreement * volume_confidence
        
        # Identify pattern
        if smart_money_signal > 0.05 and retail_signal < -0.02:
            pattern = "smart_money_accumulation"
        elif smart_money_signal < -0.05 and retail_signal > 0.02:
            pattern = "smart_money_distribution"
        elif smart_money_signal > 0.05 and retail_signal > 0.02:
            pattern = "broad_buying"
        elif smart_money_signal < -0.05 and retail_signal < -0.02:
            pattern = "broad_selling"
        else:
            pattern = "mixed"
        
        return {
            'signal': signal,
            'confidence': confidence,
            'pattern': pattern,
            'smart_money_net': smart_money_net,
            'retail_net': retail_net_sum,
            'foreign_net': foreign_net_sum,
            'institutional_net': inst_net_sum
        }
    
    def get_regime_impact(self, flow_signal: Dict) -> Tuple[float, RegimeState]:
        """
        Determine regime impact based on flow pattern.
        
        Returns (impact_strength, suggested_regime)
        """
        signal = flow_signal['signal']
        pattern = flow_signal['pattern']
        
        if pattern == "smart_money_accumulation":
            return (0.3, RegimeState.RECOVERY)
        elif pattern == "smart_money_distribution":
            return (0.3, RegimeState.BEAR_HIGH_VOL)
        elif pattern == "broad_buying":
            return (0.2, RegimeState.BULL_LOW_VOL)
        elif pattern == "broad_selling":
            return (0.2, RegimeState.CRISIS)
        else:
            return (0.0, RegimeState.NORMAL)


class EnhancedRegimeDetector:
    """
    Enhanced regime detection combining multiple signals:
    1. HMM-based price/volatility regime
    2. Futures positioning
    3. Investor flow analysis
    4. Sentiment analysis
    """
    
    def __init__(self, n_regimes: int = 5, config: Optional[Dict] = None):
        self.n_regimes = n_regimes
        self.config = config or {}
        
        # Component weights
        self.weights = {
            'hmm': self.config.get('hmm_weight', 0.40),
            'futures': self.config.get('futures_weight', 0.20),
            'flow': self.config.get('flow_weight', 0.25),
            'sentiment': self.config.get('sentiment_weight', 0.15)
        }
        
        # Initialize analyzers
        self.futures_analyzer = FuturesPositioningAnalyzer(
            lookback_weeks=self.config.get('futures_lookback_weeks', 52)
        )
        self.flow_analyzer = InvestorFlowAnalyzer(
            lookback_days=self.config.get('flow_lookback_days', 20)
        )
        
        # HMM model (to be fit)
        self.hmm_model = None
        self.regime_history: List[RegimeAssessment] = []
    
    def fit_hmm(self, returns: pd.Series, volatility: pd.Series) -> None:
        """
        Fit HMM model on historical data.
        """
        try:
            from hmmlearn import hmm
            
            # Prepare features
            features = np.column_stack([
                returns.values.reshape(-1, 1),
                volatility.values.reshape(-1, 1)
            ])
            
            # Standardize
            self.feature_mean = features.mean(axis=0)
            self.feature_std = features.std(axis=0) + 1e-8
            features_scaled = (features - self.feature_mean) / self.feature_std
            
            # Fit HMM
            self.hmm_model = hmm.GaussianHMM(
                n_components=self.n_regimes,
                covariance_type='diag',
                n_iter=100,
                random_state=42
            )
            self.hmm_model.fit(features_scaled)
            
            logger.info(f"HMM fitted with {self.n_regimes} regimes")
            
        except ImportError:
            logger.warning("hmmlearn not available, using rule-based regime detection")
            self.hmm_model = None
    
    def detect_regime(self,
                     current_return: float,
                     current_volatility: float,
                     futures_data: Optional[Dict] = None,
                     flow_data: Optional[Dict] = None,
                     sentiment_score: float = 0.0) -> RegimeAssessment:
        """
        Detect current market regime using all available signals.
        
        Parameters:
        -----------
        current_return : float
            Current period return
        current_volatility : float
            Current volatility estimate
        futures_data : Dict
            Futures positioning data
        flow_data : Dict
            Investor flow data
        sentiment_score : float
            Sentiment analysis score (-1 to 1)
        
        Returns:
        --------
        RegimeAssessment with complete analysis
        """
        # 1. HMM-based regime
        hmm_regime, hmm_probs, hmm_confidence = self._get_hmm_regime(
            current_return, current_volatility
        )
        
        # 2. Futures positioning signal
        if futures_data:
            futures_analysis = self.futures_analyzer.analyze_positioning(**futures_data)
            futures_signal = futures_analysis['signal']
        else:
            futures_signal = 0.0
            futures_analysis = {}
        
        # 3. Investor flow signal
        if flow_data:
            self.flow_analyzer.analyze_flow(**flow_data)
        flow_signal_data = self.flow_analyzer.get_flow_signal()
        flow_signal = flow_signal_data['signal']
        
        # 4. Combine signals
        combined_score = (
            self.weights['hmm'] * self._hmm_to_signal(hmm_regime) +
            self.weights['futures'] * futures_signal +
            self.weights['flow'] * flow_signal +
            self.weights['sentiment'] * sentiment_score
        )
        
        # 5. Determine primary regime
        primary_regime = self._score_to_regime(combined_score, current_volatility)
        
        # 6. Calculate regime probabilities
        regime_probs = self._calculate_regime_probabilities(
            hmm_probs, futures_signal, flow_signal, sentiment_score
        )
        
        # 7. Calculate risk level and recommended exposure
        risk_level = self._calculate_risk_level(
            primary_regime, current_volatility, combined_score
        )
        recommended_exposure = self._calculate_recommended_exposure(
            primary_regime, risk_level, combined_score
        )
        
        assessment = RegimeAssessment(
            timestamp=datetime.now(),
            primary_regime=primary_regime,
            regime_probabilities=regime_probs,
            hmm_regime=hmm_regime,
            hmm_confidence=hmm_confidence,
            futures_signal=futures_signal,
            investor_flow_signal=flow_signal,
            sentiment_signal=sentiment_score,
            combined_score=combined_score,
            risk_level=risk_level,
            recommended_exposure=recommended_exposure,
            details={
                'hmm_probs': hmm_probs,
                'futures_analysis': futures_analysis,
                'flow_analysis': flow_signal_data
            }
        )
        
        self.regime_history.append(assessment)
        
        return assessment
    
    def _get_hmm_regime(self,
                       current_return: float,
                       current_volatility: float
                       ) -> Tuple[int, Dict, float]:
        """Get regime from HMM model"""
        if self.hmm_model is None:
            # Rule-based fallback
            regime = self._rule_based_regime(current_return, current_volatility)
            probs = {r: 0.0 for r in range(self.n_regimes)}
            probs[regime] = 1.0
            return regime, probs, 0.5
        
        # Prepare features
        features = np.array([[current_return, current_volatility]])
        features_scaled = (features - self.feature_mean) / self.feature_std
        
        # Predict
        regime = self.hmm_model.predict(features_scaled)[0]
        
        # Get probabilities
        try:
            log_probs = self.hmm_model.score_samples(features_scaled)[1]
            probs = np.exp(log_probs[0])
            probs = {i: p for i, p in enumerate(probs)}
            confidence = max(probs.values())
        except:
            probs = {r: 0.0 for r in range(self.n_regimes)}
            probs[regime] = 1.0
            confidence = 0.5
        
        return regime, probs, confidence
    
    def _rule_based_regime(self,
                          current_return: float,
                          current_volatility: float) -> int:
        """Rule-based regime detection fallback"""
        if current_volatility > 0.03:
            if current_return < -0.02:
                return RegimeState.CRISIS.value
            elif current_return > 0.02:
                return RegimeState.RECOVERY.value
            else:
                return RegimeState.BEAR_HIGH_VOL.value
        elif current_volatility < 0.01:
            if current_return > 0:
                return RegimeState.BULL_LOW_VOL.value
            else:
                return RegimeState.NORMAL.value
        else:
            return RegimeState.NORMAL.value
    
    def _hmm_to_signal(self, hmm_regime: int) -> float:
        """Convert HMM regime to directional signal"""
        regime_signals = {
            RegimeState.BULL_LOW_VOL.value: 0.8,
            RegimeState.NORMAL.value: 0.2,
            RegimeState.BEAR_HIGH_VOL.value: -0.4,
            RegimeState.CRISIS.value: -0.8,
            RegimeState.RECOVERY.value: 0.5
        }
        return regime_signals.get(hmm_regime, 0.0)
    
    def _score_to_regime(self,
                        combined_score: float,
                        volatility: float) -> RegimeState:
        """Convert combined score to regime state"""
        if volatility > 0.03:
            if combined_score < -0.3:
                return RegimeState.CRISIS
            elif combined_score > 0.3:
                return RegimeState.RECOVERY
            else:
                return RegimeState.BEAR_HIGH_VOL
        elif volatility < 0.012:
            if combined_score > 0.2:
                return RegimeState.BULL_LOW_VOL
            else:
                return RegimeState.NORMAL
        else:
            if combined_score > 0.4:
                return RegimeState.BULL_LOW_VOL
            elif combined_score < -0.4:
                return RegimeState.BEAR_HIGH_VOL
            else:
                return RegimeState.NORMAL
    
    def _calculate_regime_probabilities(self,
                                       hmm_probs: Dict,
                                       futures_signal: float,
                                       flow_signal: float,
                                       sentiment: float) -> Dict[RegimeState, float]:
        """Calculate probability distribution over regimes"""
        probs = {}
        
        for regime in RegimeState:
            base_prob = hmm_probs.get(regime.value, 0.0)
            
            # Adjust based on other signals
            adjustment = 0.0
            
            if regime in [RegimeState.BULL_LOW_VOL, RegimeState.RECOVERY]:
                adjustment = 0.1 * (futures_signal + flow_signal + sentiment)
            elif regime in [RegimeState.BEAR_HIGH_VOL, RegimeState.CRISIS]:
                adjustment = -0.1 * (futures_signal + flow_signal + sentiment)
            
            probs[regime] = max(0, min(1, base_prob + adjustment))
        
        # Normalize
        total = sum(probs.values())
        if total > 0:
            probs = {k: v/total for k, v in probs.items()}
        
        return probs
    
    def _calculate_risk_level(self,
                             regime: RegimeState,
                             volatility: float,
                             combined_score: float) -> float:
        """Calculate current risk level (0-1)"""
        base_risk = {
            RegimeState.BULL_LOW_VOL: 0.2,
            RegimeState.NORMAL: 0.4,
            RegimeState.BEAR_HIGH_VOL: 0.7,
            RegimeState.CRISIS: 0.9,
            RegimeState.RECOVERY: 0.5
        }
        
        risk = base_risk.get(regime, 0.5)
        
        # Adjust for volatility
        vol_adjustment = (volatility - 0.02) * 5  # +/- based on vol
        risk += vol_adjustment
        
        # Adjust for combined score (negative score = higher risk)
        score_adjustment = -combined_score * 0.2
        risk += score_adjustment
        
        return np.clip(risk, 0, 1)
    
    def _calculate_recommended_exposure(self,
                                        regime: RegimeState,
                                        risk_level: float,
                                        combined_score: float) -> float:
        """Calculate recommended market exposure (0-1)"""
        # Base exposure by regime
        base_exposure = {
            RegimeState.BULL_LOW_VOL: 1.0,
            RegimeState.NORMAL: 0.85,
            RegimeState.BEAR_HIGH_VOL: 0.60,
            RegimeState.CRISIS: 0.40,
            RegimeState.RECOVERY: 0.80
        }
        
        exposure = base_exposure.get(regime, 0.7)
        
        # Reduce exposure as risk increases
        risk_adjustment = (0.5 - risk_level) * 0.3
        exposure += risk_adjustment
        
        # Adjust based on combined signal
        signal_adjustment = combined_score * 0.15
        exposure += signal_adjustment
        
        return np.clip(exposure, 0.2, 1.0)  # Min 20%, max 100%


if __name__ == "__main__":
    # Test enhanced regime detector
    print("Enhanced Regime Detection Test")
    print("=" * 50)
    
    detector = EnhancedRegimeDetector(n_regimes=5)
    
    # Simulate data
    np.random.seed(42)
    
    # Test various market conditions
    test_cases = [
        {"return": 0.02, "vol": 0.01, "desc": "Bull low vol"},
        {"return": -0.03, "vol": 0.04, "desc": "Crisis"},
        {"return": 0.05, "vol": 0.035, "desc": "Recovery"},
        {"return": 0.001, "vol": 0.015, "desc": "Normal"},
        {"return": -0.01, "vol": 0.025, "desc": "Bear high vol"}
    ]
    
    for tc in test_cases:
        assessment = detector.detect_regime(
            current_return=tc['return'],
            current_volatility=tc['vol'],
            sentiment_score=0.0
        )
        
        print(f"\n{tc['desc']}:")
        print(f"  Regime: {assessment.primary_regime.name}")
        print(f"  Combined Score: {assessment.combined_score:.3f}")
        print(f"  Risk Level: {assessment.risk_level:.2f}")
        print(f"  Recommended Exposure: {assessment.recommended_exposure:.1%}")
