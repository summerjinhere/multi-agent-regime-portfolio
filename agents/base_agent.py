"""
Base Agent Class
=================
Abstract base class for all agents in the system.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type, Any
from datetime import datetime

from events.message_bus import MessageBus, Event


class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    
    Provides:
    - Event subscription and publishing
    - Configuration management
    - Logging
    - State management
    """
    
    def __init__(self, 
                 name: str, 
                 message_bus: MessageBus, 
                 config: Optional[Dict] = None):
        self.name = name
        self.message_bus = message_bus
        self.config = config or {}
        self.logger = logging.getLogger(f"Agent.{name}")
        
        # State
        self.is_initialized = False
        self.last_run_time: Optional[datetime] = None
        self.run_count = 0
        self.errors: List[str] = []
        
        # Subscribe to events
        for event_type in self.get_subscribed_events():
            self.message_bus.subscribe(event_type, self.on_event)
    
    # =========================================================================
    # Abstract Methods (must be implemented by subclasses)
    # =========================================================================
    
    @abstractmethod
    def get_subscribed_events(self) -> List[Type[Event]]:
        """Return list of event types this agent subscribes to"""
        pass
    
    @abstractmethod
    def on_event(self, event: Event) -> None:
        """Handle incoming events"""
        pass
    
    @abstractmethod
    def _process(self) -> None:
        """Main processing logic"""
        pass
    
    @abstractmethod
    def _initialize(self) -> None:
        """Initialize agent resources"""
        pass
    
    # =========================================================================
    # Public Methods
    # =========================================================================
    
    def initialize(self) -> None:
        """Initialize the agent"""
        if self.is_initialized:
            return
        
        try:
            self._initialize()
            self.is_initialized = True
            self.logger.info(f"{self.name} initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.name}: {e}")
            self.errors.append(str(e))
            raise
    
    def run(self) -> None:
        """Execute the agent's main processing"""
        if not self.is_initialized:
            self.initialize()
        
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Running {self.name}...")
            self._process()
            self.run_count += 1
            self.last_run_time = datetime.now()
            
            elapsed = (self.last_run_time - start_time).total_seconds()
            self.logger.info(f"{self.name} completed in {elapsed:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Error in {self.name}: {e}")
            self.errors.append(str(e))
            raise
    
    def publish(self, event: Event) -> None:
        """Publish an event to the message bus"""
        event.source_agent = self.name
        event.timestamp = datetime.now()
        self.message_bus.publish(event)
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value with optional default"""
        return self.config.get(key, default)
    
    def get_status(self) -> Dict:
        """Get agent status"""
        return {
            'name': self.name,
            'initialized': self.is_initialized,
            'run_count': self.run_count,
            'last_run': self.last_run_time.isoformat() if self.last_run_time else None,
            'errors': self.errors[-5:],  # Last 5 errors
            'error_count': len(self.errors)
        }
    
    def reset(self) -> None:
        """Reset agent state"""
        self.is_initialized = False
        self.run_count = 0
        self.last_run_time = None
        self.errors.clear()
        self.logger.info(f"{self.name} reset")
