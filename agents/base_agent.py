"""
Base Agent Class
================
Abstract base class for all agents in the multi-agent portfolio system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Type
from datetime import datetime
from enum import Enum
import logging
import threading
import time

from events.message_bus import MessageBus
from events.base_event import Event, AgentStatusEvent


class AgentState(Enum):
    """Agent lifecycle states."""
    CREATED = "created"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class BaseAgent(ABC):
    """
    Abstract base class for all portfolio agents.
    
    Provides:
    - Lifecycle management (start, stop, pause)
    - Event subscription and publication
    - Logging and error handling
    - State persistence
    - Health monitoring
    
    Subclasses must implement:
    - _initialize(): Setup logic
    - _process(): Main processing logic
    - _cleanup(): Cleanup on stop
    """
    
    def __init__(self, name: str, message_bus: MessageBus, config: Optional[Dict] = None):
        """
        Initialize base agent.
        
        Args:
            name: Unique agent identifier
            message_bus: Shared message bus for communication
            config: Optional configuration dictionary
        """
        self.name = name
        self.bus = message_bus
        self.config = config or {}
        
        self._state = AgentState.CREATED
        self._error: Optional[Exception] = None
        self._last_run: Optional[datetime] = None
        self._run_count = 0
        
        # Logging
        self.logger = logging.getLogger(f"agent.{name}")
        
        # Thread safety
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        
        # Subscribed event types
        self._subscriptions: List[Type[Event]] = []
        
        # Cache for received events
        self._event_cache: Dict[Type[Event], Event] = {}
    
    # =========================================================================
    # Lifecycle Methods
    # =========================================================================
    
    def start(self) -> bool:
        """
        Start the agent.
        
        Returns:
            True if started successfully
        """
        try:
            self._state = AgentState.INITIALIZING
            self._publish_status("Initializing...")
            
            self._initialize()
            self._setup_subscriptions()
            
            self._state = AgentState.READY
            self._publish_status("Ready")
            self.logger.info(f"{self.name} started successfully")
            return True
            
        except Exception as e:
            self._state = AgentState.ERROR
            self._error = e
            self.logger.error(f"Failed to start {self.name}: {e}")
            self._publish_status(f"Error: {e}", error=True)
            return False
    
    def stop(self) -> None:
        """Stop the agent and cleanup resources."""
        self._stop_event.set()
        try:
            self._cleanup()
            self._state = AgentState.STOPPED
            self._publish_status("Stopped")
            self.logger.info(f"{self.name} stopped")
        except Exception as e:
            self.logger.error(f"Error during {self.name} cleanup: {e}")
    
    def pause(self) -> None:
        """Pause agent processing."""
        self._state = AgentState.PAUSED
        self._publish_status("Paused")
        self.logger.info(f"{self.name} paused")
    
    def resume(self) -> None:
        """Resume agent processing."""
        if self._state == AgentState.PAUSED:
            self._state = AgentState.READY
            self._publish_status("Resumed")
            self.logger.info(f"{self.name} resumed")
    
    def run(self) -> None:
        """
        Execute the agent's main processing logic.
        Called by orchestrator or in response to events.
        """
        if self._state not in [AgentState.READY, AgentState.RUNNING]:
            self.logger.warning(f"Cannot run {self.name}: state is {self._state}")
            return
        
        try:
            self._state = AgentState.RUNNING
            self._publish_status("Running")
            
            self._process()
            
            self._last_run = datetime.now()
            self._run_count += 1
            self._state = AgentState.READY
            self._publish_status("Completed")
            
        except Exception as e:
            self._state = AgentState.ERROR
            self._error = e
            self.logger.error(f"Error in {self.name}: {e}", exc_info=True)
            self._publish_status(f"Error: {e}", error=True)
            raise
    
    # =========================================================================
    # Abstract Methods (Must be implemented by subclasses)
    # =========================================================================
    
    @abstractmethod
    def _initialize(self) -> None:
        """
        Initialize agent-specific resources.
        Called once during start().
        """
        pass
    
    @abstractmethod
    def _process(self) -> None:
        """
        Main processing logic.
        Called each time run() is invoked.
        """
        pass
    
    def _cleanup(self) -> None:
        """
        Cleanup agent resources.
        Called during stop(). Override if needed.
        """
        pass
    
    def get_subscribed_events(self) -> List[Type[Event]]:
        """
        Return list of event types this agent subscribes to.
        Override to enable event-driven processing.
        """
        return []
    
    # =========================================================================
    # Event Handling
    # =========================================================================
    
    def _setup_subscriptions(self) -> None:
        """Setup event subscriptions based on get_subscribed_events()."""
        for event_type in self.get_subscribed_events():
            self.bus.subscribe(event_type, self._handle_event)
            self._subscriptions.append(event_type)
            self.logger.debug(f"Subscribed to {event_type.__name__}")
    
    def _handle_event(self, event: Event) -> None:
        """
        Handle incoming events.
        Caches the event and optionally triggers processing.
        """
        with self._lock:
            self._event_cache[type(event)] = event
        
        self.logger.debug(f"Received {type(event).__name__} from {event.source_agent}")
        self.on_event(event)
    
    def on_event(self, event: Event) -> None:
        """
        Event callback. Override for reactive processing.
        
        Args:
            event: The received event
        """
        pass
    
    def publish(self, event: Event) -> None:
        """
        Publish an event to the message bus.
        
        Args:
            event: Event to publish
        """
        event.source_agent = self.name
        event.timestamp = datetime.now()
        self.bus.publish(event)
        self.logger.debug(f"Published {type(event).__name__}")
    
    def get_cached_event(self, event_type: Type[Event]) -> Optional[Event]:
        """Get the most recent cached event of a given type."""
        return self._event_cache.get(event_type)
    
    def wait_for_event(self, event_type: Type[Event], 
                       timeout: float = 30.0) -> Optional[Event]:
        """
        Wait for an event of a specific type.
        
        Args:
            event_type: The event type to wait for
            timeout: Maximum time to wait in seconds
            
        Returns:
            The event if received, None if timeout
        """
        start = time.time()
        while time.time() - start < timeout:
            event = self.get_cached_event(event_type)
            if event and event.timestamp > datetime.fromtimestamp(start):
                return event
            time.sleep(0.1)
        return None
    
    # =========================================================================
    # Status & Monitoring
    # =========================================================================
    
    def _publish_status(self, message: str, error: bool = False) -> None:
        """Publish agent status update."""
        status = "ERROR" if error else self._state.value.upper()
        event = AgentStatusEvent(
            source_agent=self.name,
            agent_name=self.name,
            status=status,
            message=message,
            error_details=str(self._error) if error else None
        )
        self.bus.publish(event)
    
    def get_state(self) -> AgentState:
        """Get current agent state."""
        return self._state
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            'name': self.name,
            'state': self._state.value,
            'last_run': self._last_run.isoformat() if self._last_run else None,
            'run_count': self._run_count,
            'subscriptions': [e.__name__ for e in self._subscriptions],
            'cached_events': list(self._event_cache.keys()),
            'error': str(self._error) if self._error else None
        }
    
    def is_healthy(self) -> bool:
        """Check if agent is in a healthy state."""
        return self._state in [AgentState.READY, AgentState.RUNNING]
    
    # =========================================================================
    # Configuration
    # =========================================================================
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration."""
        self.config.update(updates)
        self.logger.info(f"Config updated: {updates}")
    
    # =========================================================================
    # String Representation
    # =========================================================================
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, state={self._state.value})"
