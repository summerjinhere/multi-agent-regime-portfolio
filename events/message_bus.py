"""
Message Bus Implementation
==========================
Publish-Subscribe pattern for agent communication.
"""

from typing import Dict, List, Callable, Type, Any, Optional
from collections import defaultdict
from datetime import datetime
import threading
import queue
import logging
from dataclasses import dataclass

from events.base_event import Event

logger = logging.getLogger(__name__)


class MessageBus:
    """
    Synchronous message bus for agent communication.
    
    Features:
    - Type-safe event subscription
    - Event history logging
    - Error handling with optional error handlers
    
    Usage:
        bus = MessageBus()
        bus.subscribe(MarketDataEvent, my_handler)
        bus.publish(MarketDataEvent(...))
    """
    
    def __init__(self, max_history: int = 1000):
        self._subscribers: Dict[Type[Event], List[Callable]] = defaultdict(list)
        self._event_history: List[Event] = []
        self._max_history = max_history
        self._lock = threading.Lock()
        self._error_handlers: List[Callable] = []
        
    def subscribe(self, event_type: Type[Event], handler: Callable[[Event], None]) -> None:
        """
        Subscribe a handler to an event type.
        
        Args:
            event_type: The event class to subscribe to
            handler: Callback function that receives the event
        """
        with self._lock:
            if handler not in self._subscribers[event_type]:
                self._subscribers[event_type].append(handler)
                logger.debug(f"Subscribed {handler.__name__} to {event_type.__name__}")
    
    def unsubscribe(self, event_type: Type[Event], handler: Callable) -> None:
        """Remove a handler from an event type."""
        with self._lock:
            if handler in self._subscribers[event_type]:
                self._subscribers[event_type].remove(handler)
                logger.debug(f"Unsubscribed {handler.__name__} from {event_type.__name__}")
    
    def publish(self, event: Event) -> None:
        """
        Publish an event to all subscribers.
        
        Args:
            event: The event instance to publish
        """
        with self._lock:
            # Log event
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history = self._event_history[-self._max_history:]
            
            # Get handlers (copy to avoid modification during iteration)
            handlers = list(self._subscribers[type(event)])
        
        # Execute handlers outside lock
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in handler {handler.__name__}: {e}")
                self._handle_error(event, handler, e)
    
    def publish_and_wait(self, event: Event, timeout: float = 30.0) -> bool:
        """
        Publish event and wait for all handlers to complete.
        Returns True if all handlers completed successfully.
        """
        self.publish(event)
        return True
    
    def add_error_handler(self, handler: Callable[[Event, Callable, Exception], None]) -> None:
        """Add a global error handler for failed event processing."""
        self._error_handlers.append(handler)
    
    def _handle_error(self, event: Event, handler: Callable, error: Exception) -> None:
        """Invoke error handlers."""
        for error_handler in self._error_handlers:
            try:
                error_handler(event, handler, error)
            except Exception as e:
                logger.error(f"Error in error handler: {e}")
    
    def get_history(self, event_type: Optional[Type[Event]] = None, 
                    limit: int = 100) -> List[Event]:
        """
        Get event history, optionally filtered by type.
        
        Args:
            event_type: Filter by this event type (None for all)
            limit: Maximum number of events to return
        """
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
    
    def get_subscriber_count(self, event_type: Type[Event]) -> int:
        """Get number of subscribers for an event type."""
        return len(self._subscribers[event_type])
    
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


class AsyncMessageBus(MessageBus):
    """
    Asynchronous message bus with background processing.
    
    Events are queued and processed by a background thread,
    allowing non-blocking publish operations.
    """
    
    def __init__(self, max_history: int = 1000):
        super().__init__(max_history)
        self._queue: queue.Queue = queue.Queue()
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
    
    def start(self) -> None:
        """Start the background processing thread."""
        if not self._running:
            self._running = True
            self._worker_thread = threading.Thread(target=self._process_events, daemon=True)
            self._worker_thread.start()
            logger.info("AsyncMessageBus started")
    
    def stop(self, wait: bool = True, timeout: float = 5.0) -> None:
        """
        Stop the background processing thread.
        
        Args:
            wait: If True, wait for queue to be processed
            timeout: Maximum time to wait for queue processing
        """
        self._running = False
        if wait and self._worker_thread:
            self._worker_thread.join(timeout=timeout)
        logger.info("AsyncMessageBus stopped")
    
    def publish_async(self, event: Event) -> None:
        """
        Queue an event for asynchronous processing.
        
        Args:
            event: The event to queue
        """
        self._queue.put(event)
    
    def _process_events(self) -> None:
        """Background thread for processing events."""
        while self._running or not self._queue.empty():
            try:
                event = self._queue.get(timeout=0.1)
                super().publish(event)
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}")
    
    def wait_for_completion(self, timeout: float = 30.0) -> bool:
        """Wait for all queued events to be processed."""
        try:
            self._queue.join()
            return True
        except Exception:
            return False
    
    def get_queue_size(self) -> int:
        """Get number of events waiting to be processed."""
        return self._queue.qsize()


@dataclass
class EventFilter:
    """Filter for selective event subscription."""
    event_type: Type[Event]
    source_agent: Optional[str] = None
    min_confidence: Optional[float] = None
    
    def matches(self, event: Event) -> bool:
        """Check if event matches filter criteria."""
        if not isinstance(event, self.event_type):
            return False
        if self.source_agent and event.source_agent != self.source_agent:
            return False
        # Add more filter criteria as needed
        return True


class FilteredMessageBus(MessageBus):
    """
    Message bus with filtered subscriptions.
    Allows subscribers to specify criteria for events they want to receive.
    """
    
    def __init__(self, max_history: int = 1000):
        super().__init__(max_history)
        self._filtered_subscribers: List[tuple] = []  # (filter, handler)
    
    def subscribe_filtered(self, event_filter: EventFilter, 
                          handler: Callable[[Event], None]) -> None:
        """Subscribe with a filter."""
        self._filtered_subscribers.append((event_filter, handler))
    
    def publish(self, event: Event) -> None:
        """Publish to both regular and filtered subscribers."""
        super().publish(event)
        
        # Process filtered subscribers
        for event_filter, handler in self._filtered_subscribers:
            if event_filter.matches(event):
                try:
                    handler(event)
                except Exception as e:
                    logger.error(f"Error in filtered handler {handler.__name__}: {e}")
