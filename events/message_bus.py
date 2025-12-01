"""
Event System for Multi-Agent Communication
===========================================
Base events and message bus for inter-agent communication.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Type, Callable
from datetime import datetime
from abc import ABC
import logging
from queue import PriorityQueue
import threading

logger = logging.getLogger(__name__)


# =============================================================================
# Base Event
# =============================================================================

@dataclass
class Event(ABC):
    """Base class for all events"""
    source_agent: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 0
    
    def __lt__(self, other):
        return self.priority < other.priority


# =============================================================================
# Specific Events
# =============================================================================

@dataclass
class MarketDataEvent(Event):
    """Event containing market data"""
    prices: Optional[pd.DataFrame] = None
    volumes: Optional[pd.DataFrame] = None
    returns: Optional[pd.DataFrame] = None
    sector_mapping: Dict[str, List[str]] = field(default_factory=dict)
    data_quality: Dict[str, float] = field(default_factory=dict)
    as_of_date: Optional[datetime] = None
    
    def __post_init__(self):
        self.priority = 10  # High priority


@dataclass
class RegimeEvent(Event):
    """Event containing regime detection results"""
    current_regime: int = 0
    regime_name: str = "Unknown"
    regime_probabilities: Dict[int, float] = field(default_factory=dict)
    transition_matrix: Optional[np.ndarray] = None
    regime_duration: int = 0
    regime_history: Optional[pd.Series] = None
    model_confidence: float = 0.0
    features_used: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self.priority = 8


@dataclass
class QuantSignalEvent(Event):
    """Event containing quantitative signals"""
    expected_returns: Optional[pd.Series] = None
    covariance_matrix: Optional[pd.DataFrame] = None
    cointegration_rank: int = 0
    is_cointegrated: bool = False
    forecast_horizon: int = 5
    forecast_confidence: float = 0.5
    asset_classes: List[str] = field(default_factory=list)
    methodology: str = "Mean-Variance"
    
    def __post_init__(self):
        self.priority = 7


@dataclass
class RiskAssessmentEvent(Event):
    """Event containing risk assessment"""
    portfolio_var: float = 0.0
    portfolio_cvar: float = 0.0
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    risk_budget: Dict[str, float] = field(default_factory=dict)
    position_limits: Dict[str, float] = field(default_factory=dict)
    risk_alerts: List[str] = field(default_factory=list)
    regime_risk_params: Optional[Dict] = None
    
    def __post_init__(self):
        self.priority = 6


@dataclass
class PortfolioSignalEvent(Event):
    """Event containing portfolio allocation signals"""
    target_weights: Optional[pd.Series] = None
    current_weights: Optional[pd.Series] = None
    rebalance_required: bool = False
    turnover: float = 0.0
    expected_return: float = 0.0
    expected_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    optimization_method: str = "Mean-Variance"
    
    def __post_init__(self):
        self.priority = 5


@dataclass
class TradeEvent(Event):
    """Event for trade execution"""
    ticker: str = ""
    side: str = "BUY"
    quantity: int = 0
    price: float = 0.0
    execution_algo: str = "TWAP"
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    slippage_bps: float = 0.0
    status: str = "PENDING"
    
    def __post_init__(self):
        self.priority = 3


@dataclass
class InstitutionalFlowEvent(Event):
    """Event for institutional flow signals"""
    ticker: str = ""
    flow_type: str = "NEUTRAL"  # ACCUMULATION, DISTRIBUTION, NEUTRAL
    flow_strength: float = 0.0
    volume_ratio: float = 1.0
    institutional_ratio: float = 0.0
    confidence: float = 0.0
    piggyback_signal: bool = False
    
    def __post_init__(self):
        self.priority = 7


@dataclass
class SentimentEvent(Event):
    """Event for sentiment analysis"""
    ticker: str = ""
    sentiment_score: float = 0.0  # -1 to +1
    confidence: float = 0.0
    source: str = ""
    mention_count: int = 0
    keywords: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self.priority = 6


# =============================================================================
# Message Bus
# =============================================================================

class MessageBus:
    """
    Central message bus for event distribution.
    Implements pub/sub pattern with priority queue.
    """
    
    def __init__(self):
        self._subscribers: Dict[Type[Event], List[Callable]] = {}
        self._event_queue: PriorityQueue = PriorityQueue()
        self._lock = threading.Lock()
        self._running = False
        self._event_history: List[Event] = []
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def subscribe(self, event_type: Type[Event], callback: Callable) -> None:
        """Subscribe to an event type"""
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(callback)
            self.logger.debug(f"Subscribed {callback.__qualname__} to {event_type.__name__}")
    
    def unsubscribe(self, event_type: Type[Event], callback: Callable) -> None:
        """Unsubscribe from an event type"""
        with self._lock:
            if event_type in self._subscribers:
                self._subscribers[event_type].remove(callback)
    
    def publish(self, event: Event) -> None:
        """Publish an event to all subscribers"""
        event_type = type(event)
        
        with self._lock:
            self._event_history.append(event)
            
            # Get all subscribers for this event type and parent types
            subscribers = []
            for evt_type, callbacks in self._subscribers.items():
                if isinstance(event, evt_type):
                    subscribers.extend(callbacks)
        
        # Notify subscribers
        for callback in subscribers:
            try:
                callback(event)
            except Exception as e:
                self.logger.error(f"Error in subscriber {callback.__qualname__}: {e}")
        
        self.logger.debug(f"Published {event_type.__name__} from {event.source_agent} "
                         f"to {len(subscribers)} subscribers")
    
    def publish_async(self, event: Event) -> None:
        """Add event to queue for async processing"""
        self._event_queue.put((event.priority, datetime.now(), event))
    
    def process_queue(self) -> None:
        """Process all events in queue"""
        while not self._event_queue.empty():
            _, _, event = self._event_queue.get()
            self.publish(event)
    
    def get_event_history(self, event_type: Optional[Type[Event]] = None) -> List[Event]:
        """Get event history, optionally filtered by type"""
        if event_type is None:
            return self._event_history.copy()
        return [e for e in self._event_history if isinstance(e, event_type)]
    
    def clear_history(self) -> None:
        """Clear event history"""
        with self._lock:
            self._event_history.clear()


# =============================================================================
# Global Message Bus Instance
# =============================================================================

_message_bus: Optional[MessageBus] = None


def get_message_bus() -> MessageBus:
    """Get or create global message bus instance"""
    global _message_bus
    if _message_bus is None:
        _message_bus = MessageBus()
    return _message_bus


def reset_message_bus() -> None:
    """Reset global message bus"""
    global _message_bus
    _message_bus = None
