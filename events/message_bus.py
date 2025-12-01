"""
Message Bus Implementation
==========================
Publish-Subscribe pattern for agent communication.
"""

from typing import Dict, List, Callable, Type, Any, Optional
from collections import defaultdict
from datetime import datetime
import threading
import logging

from events.base_event import Event

logger = logging.getLogger(__name__)


class MessageBus:
    """
    Synchronous message bus for agent communication.
    """
    
    def __init__(self, max_history: int = 1000):
        self._subscribers: Dict[Type[Event], List[Callable]] = defaultdict(list)
        self._event_history: List[Event] = []
        self._max_history = max_history
        self._lock = threading.Lock()
        self._error_handlers: List[Callable] = []
        
    def subscribe(self, event_type: Type[Event], handler: Callable[[Event], None]) -> None:
        """Subscribe a handler to an event type."""
        with self._lock:
            if handler not in self._subscribers[event_type]:
                self._subscribers[event_type].append(handler)
                logger.debug(f"Subscribed {handler.__name__} to {event_type.__name__}")
    
    def unsubscribe(self, event_type: Type[Event], handler: Callable) -> None:
        """Remove a handler from an event type."""
        with self._lock:
            if handler in self._subscribers[event_type]:
                self._subscribers[event_type].remove(handler)
    
    def publish(self, event: Event) -> None:
        """Publish an event to all subscribers."""
        with self._lock:
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history = self._event_history[-self._max_history:]
            
            handlers = list(self._subscribers[type(event)])
        
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in handler {handler.__name__}: {e}")
    
    def get_history(self, event_type: Optional[Type[Event]] = None, 
                    limit: int = 100) -> List[Event]:
        """Get event history, optionally filtered by type."""
        with self._lock:
            if event_type is None:
                return list(self._event_history[-limit:])
            return [e for e in self._event_history if isinstance(e, event_type)][-limit:]
    
    def get_last_event(self, event_type: Type[Event]) -> Optional[Event]:
        """Get the most recent event of a given type."""
        history = self.get_history(event_type, limit=1)
        return history[0] if history else None
    
    def clear_history(self) -> None:
        """Clear event history."""
        with self._lock:
            self._event_history.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get message bus statistics."""
        with self._lock:
            event_counts = defaultdict(int)
            for event in self._event_history:
                event_counts[type(event).__name__] += 1
            
            return {
                'total_events': len(self._event_history),
                'event_counts': dict(event_counts),
                'subscriber_counts': {
                    k.__name__: len(v) 
                    for k, v in self._subscribers.items()
                }
            }
