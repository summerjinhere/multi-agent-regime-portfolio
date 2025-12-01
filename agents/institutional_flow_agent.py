"""
Institutional Flow Tracking and Piggyback Trading Module
=========================================================

Features:
1. Detect institutional accumulation/distribution patterns
2. Piggyback trading with 10% profit target
3. Short position tracking
4. Volume-based institutional activity detection
5. Smart money flow analysis

Author: Anonymous
Date: November 2025
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FlowType(Enum):
    """Type of institutional flow"""
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    NEUTRAL = "neutral"
    SHORT_COVERING = "short_covering"
    SHORT_SELLING = "short_selling"


class PositionType(Enum):
    """Position direction"""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class InstitutionalSignal:
    """Container for institutional flow signal"""
    ticker: str
    timestamp: datetime
    flow_type: FlowType
    strength: float  # 0 to 1
    volume_ratio: float  # Relative to average
    price_impact: float  # Price change during flow
    institutional_ratio: float  # Estimated institutional volume %
    confidence: float  # Signal confidence
    details: Dict = field(default_factory=dict)


@dataclass
class PiggybackPosition:
    """Tracks a piggyback position"""
    ticker: str
    position_type: PositionType
    entry_price: float
    entry_time: datetime
    entry_signal: InstitutionalSignal
    quantity: int
    target_price: float  # 10% profit target
    stop_loss_price: float  # Stop loss level
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    status: str = "open"  # open, closed_profit, closed_loss, closed_signal


class VolumeAnalyzer:
    """
    Analyzes volume patterns to detect institutional activity.
    Uses multiple indicators:
    - Volume spike detection
    - On-Balance Volume (OBV)
    - Volume-Weighted Average Price (VWAP) analysis
    - Accumulation/Distribution Line
    """
    
    def __init__(self, lookback_days: int = 20):
        self.lookback_days = lookback_days
    
    def calculate_volume_metrics(self, 
                                 prices: pd.DataFrame,
                                 volumes: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Calculate comprehensive volume metrics.
        
        Parameters:
        -----------
        prices : pd.DataFrame
            OHLC prices with columns [Open, High, Low, Close]
        volumes : pd.DataFrame
            Volume data with same index as prices
        
        Returns:
        --------
        Dict with volume metrics DataFrames
        """
        metrics = {}
        
        # 1. Volume Moving Average and Ratio
        vol_ma = volumes.rolling(window=self.lookback_days).mean()
        vol_ratio = volumes / vol_ma
        metrics['volume_ratio'] = vol_ratio
        
        # 2. On-Balance Volume (OBV)
        if isinstance(prices, pd.DataFrame) and 'Close' in prices.columns:
            close = prices['Close']
        else:
            close = prices
        
        price_change = close.diff()
        obv = pd.Series(0.0, index=close.index)
        
        for i in range(1, len(close)):
            if price_change.iloc[i] > 0:
                obv.iloc[i] = obv.iloc[i-1] + volumes.iloc[i]
            elif price_change.iloc[i] < 0:
                obv.iloc[i] = obv.iloc[i-1] - volumes.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        metrics['obv'] = obv
        metrics['obv_ma'] = obv.rolling(window=self.lookback_days).mean()
        
        # 3. Accumulation/Distribution Line
        if isinstance(prices, pd.DataFrame) and all(c in prices.columns for c in ['High', 'Low', 'Close']):
            high = prices['High']
            low = prices['Low']
            close = prices['Close']
            
            # Money Flow Multiplier
            mfm = ((close - low) - (high - close)) / (high - low + 1e-10)
            
            # Money Flow Volume
            mfv = mfm * volumes
            
            # A/D Line
            ad_line = mfv.cumsum()
            metrics['ad_line'] = ad_line
            metrics['mfv'] = mfv
        
        # 4. Volume Price Trend
        vpt = (price_change / close.shift(1) * volumes).cumsum()
        metrics['vpt'] = vpt
        
        # 5. Relative Volume
        rel_vol = volumes / volumes.rolling(window=self.lookback_days).std()
        metrics['relative_volume'] = rel_vol
        
        return metrics
    
    def detect_volume_anomalies(self,
                                volumes: pd.Series,
                                threshold: float = 2.0) -> pd.Series:
        """
        Detect abnormal volume spikes that may indicate institutional activity.
        
        Returns boolean series where True = anomaly detected.
        """
        vol_mean = volumes.rolling(window=self.lookback_days).mean()
        vol_std = volumes.rolling(window=self.lookback_days).std()
        
        z_score = (volumes - vol_mean) / (vol_std + 1e-10)
        
        return z_score > threshold


class InstitutionalFlowDetector:
    """
    Detects institutional accumulation and distribution patterns.
    Uses volume analysis, price action, and flow indicators.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.volume_analyzer = VolumeAnalyzer(
            lookback_days=self.config.get('lookback_days', 20)
        )
        self.accumulation_threshold = self.config.get('accumulation_threshold', 0.15)
        self.distribution_threshold = self.config.get('distribution_threshold', -0.15)
        self.signal_history: Dict[str, List[InstitutionalSignal]] = {}
    
    def analyze_flow(self,
                    ticker: str,
                    prices: pd.DataFrame,
                    volumes: pd.Series) -> InstitutionalSignal:
        """
        Analyze institutional flow for a single ticker.
        
        Parameters:
        -----------
        ticker : str
            Stock ticker
        prices : pd.DataFrame
            OHLC data
        volumes : pd.Series
            Volume data
        
        Returns:
        --------
        InstitutionalSignal
        """
        if len(prices) < 20:
            return self._create_neutral_signal(ticker)
        
        # Calculate volume metrics
        metrics = self.volume_analyzer.calculate_volume_metrics(prices, volumes)
        
        # Get latest values
        current_vol_ratio = metrics['volume_ratio'].iloc[-1]
        obv_trend = self._calculate_trend(metrics['obv'], window=5)
        
        # A/D line trend if available
        ad_trend = 0.0
        if 'ad_line' in metrics:
            ad_trend = self._calculate_trend(metrics['ad_line'], window=5)
        
        # Price change
        if isinstance(prices, pd.DataFrame) and 'Close' in prices.columns:
            close = prices['Close']
        else:
            close = prices
        
        price_change = (close.iloc[-1] / close.iloc[-5] - 1) if len(close) >= 5 else 0
        
        # Detect flow type
        flow_type, strength = self._classify_flow(
            vol_ratio=current_vol_ratio,
            obv_trend=obv_trend,
            ad_trend=ad_trend,
            price_change=price_change
        )
        
        # Estimate institutional ratio (simplified)
        # In production, use actual institutional vs retail volume data
        inst_ratio = self._estimate_institutional_ratio(
            vol_ratio=current_vol_ratio,
            price_impact=abs(price_change)
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            vol_ratio=current_vol_ratio,
            strength=strength,
            agreement=self._check_indicator_agreement(obv_trend, ad_trend, price_change)
        )
        
        signal = InstitutionalSignal(
            ticker=ticker,
            timestamp=datetime.now(),
            flow_type=flow_type,
            strength=strength,
            volume_ratio=current_vol_ratio,
            price_impact=price_change,
            institutional_ratio=inst_ratio,
            confidence=confidence,
            details={
                'obv_trend': obv_trend,
                'ad_trend': ad_trend,
                'price_change': price_change
            }
        )
        
        # Store in history
        if ticker not in self.signal_history:
            self.signal_history[ticker] = []
        self.signal_history[ticker].append(signal)
        
        return signal
    
    def _classify_flow(self,
                      vol_ratio: float,
                      obv_trend: float,
                      ad_trend: float,
                      price_change: float) -> Tuple[FlowType, float]:
        """Classify the type and strength of institutional flow"""
        
        # Combine signals
        combined_signal = (
            0.3 * np.sign(vol_ratio - 1) * min(abs(vol_ratio - 1), 1) +
            0.3 * np.tanh(obv_trend * 10) +
            0.2 * np.tanh(ad_trend * 10) +
            0.2 * np.tanh(price_change * 20)
        )
        
        strength = abs(combined_signal)
        
        # Volume must be elevated for institutional activity
        if vol_ratio < 1.2:
            return FlowType.NEUTRAL, 0.0
        
        # High volume + price up + positive OBV/AD = Accumulation
        if combined_signal > self.accumulation_threshold:
            # Check for short covering (price up rapidly after prior decline)
            if price_change > 0.03 and obv_trend > 0:
                return FlowType.SHORT_COVERING, strength
            return FlowType.ACCUMULATION, strength
        
        # High volume + price down + negative OBV/AD = Distribution
        elif combined_signal < self.distribution_threshold:
            # Check for short selling (price down with increasing volume)
            if price_change < -0.02 and vol_ratio > 1.5:
                return FlowType.SHORT_SELLING, strength
            return FlowType.DISTRIBUTION, strength
        
        return FlowType.NEUTRAL, 0.0
    
    def _calculate_trend(self, series: pd.Series, window: int = 5) -> float:
        """Calculate normalized trend of a series"""
        if len(series) < window:
            return 0.0
        
        recent = series.iloc[-window:]
        if recent.std() == 0:
            return 0.0
        
        # Linear regression slope, normalized
        x = np.arange(window)
        slope = np.polyfit(x, recent.values, 1)[0]
        normalized_slope = slope / (recent.std() + 1e-10)
        
        return normalized_slope
    
    def _estimate_institutional_ratio(self,
                                     vol_ratio: float,
                                     price_impact: float) -> float:
        """
        Estimate the proportion of institutional volume.
        Simplified heuristic - in production use actual data.
        """
        # Higher volume with less price impact = more institutional
        if vol_ratio > 2.0 and price_impact < 0.02:
            return 0.7  # High institutional activity
        elif vol_ratio > 1.5:
            return 0.5
        elif vol_ratio > 1.2:
            return 0.35
        else:
            return 0.25
    
    def _check_indicator_agreement(self,
                                   obv_trend: float,
                                   ad_trend: float,
                                   price_change: float) -> float:
        """Check if indicators agree on direction"""
        signs = [np.sign(obv_trend), np.sign(ad_trend), np.sign(price_change)]
        signs = [s for s in signs if s != 0]
        
        if not signs:
            return 0.0
        
        # Proportion of agreeing signs
        positive = sum(1 for s in signs if s > 0)
        negative = sum(1 for s in signs if s < 0)
        
        return max(positive, negative) / len(signs)
    
    def _calculate_confidence(self,
                             vol_ratio: float,
                             strength: float,
                             agreement: float) -> float:
        """Calculate overall signal confidence"""
        # Base confidence from volume
        vol_conf = min(1.0, (vol_ratio - 1) / 2) if vol_ratio > 1 else 0.0
        
        # Combine factors
        confidence = (
            0.4 * vol_conf +
            0.3 * strength +
            0.3 * agreement
        )
        
        return min(1.0, confidence)
    
    def _create_neutral_signal(self, ticker: str) -> InstitutionalSignal:
        """Create a neutral signal when data is insufficient"""
        return InstitutionalSignal(
            ticker=ticker,
            timestamp=datetime.now(),
            flow_type=FlowType.NEUTRAL,
            strength=0.0,
            volume_ratio=1.0,
            price_impact=0.0,
            institutional_ratio=0.0,
            confidence=0.0
        )


class PiggybackTrader:
    """
    Implements piggyback trading strategy:
    - Enter when institutional accumulation detected
    - Target 10% profit
    - Track both long and short positions
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.take_profit_pct = self.config.get('take_profit_pct', 0.10)  # 10%
        self.stop_loss_pct = self.config.get('stop_loss_pct', 0.05)  # 5%
        self.entry_threshold = self.config.get('entry_threshold', 0.10)
        self.positions: Dict[str, PiggybackPosition] = {}
        self.closed_positions: List[PiggybackPosition] = []
        self.flow_detector = InstitutionalFlowDetector(config)
    
    def evaluate_entry(self,
                      ticker: str,
                      signal: InstitutionalSignal,
                      current_price: float,
                      available_capital: float) -> Optional[PiggybackPosition]:
        """
        Evaluate whether to enter a piggyback position.
        
        Returns:
        --------
        PiggybackPosition if entry is warranted, None otherwise
        """
        # Skip if already have position
        if ticker in self.positions:
            return None
        
        # Check signal strength and confidence
        if signal.strength < self.entry_threshold:
            return None
        
        if signal.confidence < 0.5:
            return None
        
        # Determine position type based on flow
        if signal.flow_type in [FlowType.ACCUMULATION, FlowType.SHORT_COVERING]:
            position_type = PositionType.LONG
            target_price = current_price * (1 + self.take_profit_pct)
            stop_loss_price = current_price * (1 - self.stop_loss_pct)
        
        elif signal.flow_type in [FlowType.DISTRIBUTION, FlowType.SHORT_SELLING]:
            if not self.config.get('short_tracking_enabled', True):
                return None
            position_type = PositionType.SHORT
            target_price = current_price * (1 - self.take_profit_pct)
            stop_loss_price = current_price * (1 + self.stop_loss_pct)
        
        else:
            return None
        
        # Calculate position size (simplified - use proper sizing in production)
        position_value = available_capital * 0.05  # 5% of capital per position
        quantity = int(position_value / current_price)
        
        if quantity < 1:
            return None
        
        position = PiggybackPosition(
            ticker=ticker,
            position_type=position_type,
            entry_price=current_price,
            entry_time=datetime.now(),
            entry_signal=signal,
            quantity=quantity,
            target_price=target_price,
            stop_loss_price=stop_loss_price,
            current_price=current_price
        )
        
        self.positions[ticker] = position
        
        logger.info(f"Entered {position_type.value} position in {ticker} "
                   f"at ${current_price:.2f}, target: ${target_price:.2f}")
        
        return position
    
    def update_positions(self, 
                        prices: Dict[str, float]) -> List[Tuple[str, str, float]]:
        """
        Update all positions with current prices.
        Check for exit conditions.
        
        Returns:
        --------
        List of (ticker, action, pnl) for closed positions
        """
        closed = []
        
        for ticker, position in list(self.positions.items()):
            if ticker not in prices:
                continue
            
            current_price = prices[ticker]
            position.current_price = current_price
            
            # Calculate unrealized P&L
            if position.position_type == PositionType.LONG:
                position.unrealized_pnl = (
                    (current_price - position.entry_price) / position.entry_price
                )
                
                # Check exit conditions
                if current_price >= position.target_price:
                    pnl = self._close_position(ticker, "target_reached")
                    closed.append((ticker, "profit_target", pnl))
                
                elif current_price <= position.stop_loss_price:
                    pnl = self._close_position(ticker, "stop_loss")
                    closed.append((ticker, "stop_loss", pnl))
            
            elif position.position_type == PositionType.SHORT:
                position.unrealized_pnl = (
                    (position.entry_price - current_price) / position.entry_price
                )
                
                # Check exit conditions (reversed for short)
                if current_price <= position.target_price:
                    pnl = self._close_position(ticker, "target_reached")
                    closed.append((ticker, "profit_target", pnl))
                
                elif current_price >= position.stop_loss_price:
                    pnl = self._close_position(ticker, "stop_loss")
                    closed.append((ticker, "stop_loss", pnl))
        
        return closed
    
    def check_signal_reversal(self,
                             ticker: str,
                             new_signal: InstitutionalSignal) -> bool:
        """
        Check if institutional flow has reversed, warranting early exit.
        """
        if ticker not in self.positions:
            return False
        
        position = self.positions[ticker]
        
        # Long position + distribution signal = exit
        if position.position_type == PositionType.LONG:
            if new_signal.flow_type in [FlowType.DISTRIBUTION, FlowType.SHORT_SELLING]:
                if new_signal.strength > 0.15 and new_signal.confidence > 0.5:
                    pnl = self._close_position(ticker, "signal_reversal")
                    logger.info(f"Closed {ticker} on signal reversal, PnL: {pnl:.2%}")
                    return True
        
        # Short position + accumulation signal = exit
        elif position.position_type == PositionType.SHORT:
            if new_signal.flow_type in [FlowType.ACCUMULATION, FlowType.SHORT_COVERING]:
                if new_signal.strength > 0.15 and new_signal.confidence > 0.5:
                    pnl = self._close_position(ticker, "signal_reversal")
                    logger.info(f"Closed {ticker} on signal reversal, PnL: {pnl:.2%}")
                    return True
        
        return False
    
    def _close_position(self, ticker: str, reason: str) -> float:
        """Close a position and return the P&L"""
        if ticker not in self.positions:
            return 0.0
        
        position = self.positions.pop(ticker)
        position.status = f"closed_{reason}"
        
        pnl = position.unrealized_pnl
        self.closed_positions.append(position)
        
        return pnl
    
    def get_position_summary(self) -> Dict:
        """Get summary of all positions"""
        open_positions = len(self.positions)
        closed_positions = len(self.closed_positions)
        
        if self.closed_positions:
            closed_pnl = [p.unrealized_pnl for p in self.closed_positions]
            win_rate = sum(1 for p in closed_pnl if p > 0) / len(closed_pnl)
            avg_pnl = np.mean(closed_pnl)
            total_pnl = sum(closed_pnl)
        else:
            win_rate = 0.0
            avg_pnl = 0.0
            total_pnl = 0.0
        
        return {
            'open_positions': open_positions,
            'closed_positions': closed_positions,
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'total_pnl': total_pnl,
            'current_positions': list(self.positions.keys())
        }


class ShortInterestTracker:
    """
    Tracks short interest and short selling activity.
    Detects potential short squeezes.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.short_interest_data: Dict[str, pd.Series] = {}
    
    def update_short_interest(self,
                             ticker: str,
                             short_interest: float,
                             shares_outstanding: float,
                             avg_daily_volume: float) -> Dict:
        """
        Update short interest data and calculate metrics.
        
        Parameters:
        -----------
        short_interest : float
            Number of shares sold short
        shares_outstanding : float
            Total shares outstanding
        avg_daily_volume : float
            Average daily trading volume
        
        Returns:
        --------
        Dict with short interest metrics
        """
        # Short interest ratio (% of float)
        short_ratio = short_interest / shares_outstanding if shares_outstanding > 0 else 0
        
        # Days to cover
        days_to_cover = short_interest / avg_daily_volume if avg_daily_volume > 0 else 0
        
        # Short squeeze potential
        squeeze_score = self._calculate_squeeze_score(short_ratio, days_to_cover)
        
        return {
            'ticker': ticker,
            'short_interest': short_interest,
            'short_ratio': short_ratio,
            'days_to_cover': days_to_cover,
            'squeeze_score': squeeze_score,
            'squeeze_alert': squeeze_score > 0.7
        }
    
    def _calculate_squeeze_score(self,
                                short_ratio: float,
                                days_to_cover: float) -> float:
        """
        Calculate short squeeze potential score (0-1).
        Higher score = higher squeeze potential.
        """
        # High short ratio increases squeeze potential
        ratio_score = min(1.0, short_ratio / 0.30)  # 30% short interest = max score
        
        # High days to cover increases squeeze potential
        dtc_score = min(1.0, days_to_cover / 10)  # 10 days = max score
        
        # Combine
        squeeze_score = 0.5 * ratio_score + 0.5 * dtc_score
        
        return squeeze_score


if __name__ == "__main__":
    # Test the institutional flow detector
    np.random.seed(42)
    
    # Generate sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    
    # Simulate accumulation pattern
    prices = pd.DataFrame({
        'Open': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'High': 100 + np.cumsum(np.random.randn(100) * 0.5) + 1,
        'Low': 100 + np.cumsum(np.random.randn(100) * 0.5) - 1,
        'Close': 100 + np.cumsum(np.random.randn(100) * 0.5 + 0.1)  # Slight uptrend
    }, index=dates)
    
    # Volume with spike at end (simulating accumulation)
    volumes = pd.Series(
        1000000 + np.random.randn(100) * 100000,
        index=dates
    )
    volumes.iloc[-10:] *= 2  # Volume spike
    
    # Detect flow
    detector = InstitutionalFlowDetector()
    signal = detector.analyze_flow('AAPL', prices, volumes)
    
    print("Institutional Flow Detection Test")
    print("=" * 50)
    print(f"Ticker: {signal.ticker}")
    print(f"Flow Type: {signal.flow_type.value}")
    print(f"Strength: {signal.strength:.3f}")
    print(f"Volume Ratio: {signal.volume_ratio:.2f}")
    print(f"Institutional Ratio: {signal.institutional_ratio:.2%}")
    print(f"Confidence: {signal.confidence:.3f}")
    print(f"Details: {signal.details}")
