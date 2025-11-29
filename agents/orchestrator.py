"""
Orchestrator Agent
==================
Coordinates the multi-agent pipeline execution.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import time
import logging
from enum import Enum

from agents.base_agent import BaseAgent, AgentState
from events.message_bus import MessageBus
from events.base_event import (
    Event, AgentStatusEvent, MarketDataEvent, RegimeEvent,
    QuantSignalEvent, RiskAdjustedSignalEvent, TradeSignalEvent
)


class PipelineStage(Enum):
    """Pipeline execution stages."""
    IDLE = "idle"
    DATA_ACQUISITION = "data_acquisition"
    REGIME_DETECTION = "regime_detection"
    QUANT_ANALYSIS = "quant_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    PORTFOLIO_OPTIMIZATION = "portfolio_optimization"
    COMPLETE = "complete"
    ERROR = "error"


class Orchestrator(BaseAgent):
    """
    Orchestrator Agent that coordinates the multi-agent pipeline.
    
    Pipeline Flow:
    1. MarketDataAgent → MarketDataEvent
    2. MLAgent → RegimeEvent
    3. QuantAgent → QuantSignalEvent  
    4. RiskManagerAgent → RiskAdjustedSignalEvent
    5. PortfolioAgent → TradeSignalEvent
    
    Responsibilities:
    - Agent lifecycle management
    - Pipeline execution control
    - Error handling and recovery
    - Progress monitoring
    - Result aggregation
    
    Config Options:
        timeout_seconds: Max time per stage
        retry_attempts: Number of retries on failure
        parallel_execution: Run independent stages in parallel
    """
    
    def __init__(self, message_bus: MessageBus, config: Optional[Dict] = None):
        super().__init__("Orchestrator", message_bus, config)
        
        # Configuration
        self.timeout_seconds = self.get_config('timeout_seconds', 60)
        self.retry_attempts = self.get_config('retry_attempts', 3)
        
        # Agent registry
        self.agents: Dict[str, BaseAgent] = {}
        
        # Pipeline state
        self.current_stage = PipelineStage.IDLE
        self.pipeline_results: Dict[str, Any] = {}
        self.execution_log: List[Dict] = []
        
        # Callbacks
        self.on_complete_callbacks: List[Callable] = []
        self.on_error_callbacks: List[Callable] = []
    
    def _initialize(self) -> None:
        """Initialize orchestrator."""
        self.logger.info("Orchestrator initialized")
    
    def register_agent(self, agent: BaseAgent) -> None:
        """
        Register an agent with the orchestrator.
        
        Args:
            agent: Agent instance to register
        """
        self.agents[agent.name] = agent
        self.logger.info(f"Registered agent: {agent.name}")
    
    def get_subscribed_events(self) -> List[type]:
        """Subscribe to status events from all agents."""
        return [AgentStatusEvent, TradeSignalEvent]
    
    def on_event(self, event: Event) -> None:
        """Handle status updates and final signals."""
        if isinstance(event, AgentStatusEvent):
            self._handle_status_update(event)
        elif isinstance(event, TradeSignalEvent):
            self._handle_trade_signal(event)
    
    def _handle_status_update(self, event: AgentStatusEvent) -> None:
        """Process agent status updates."""
        self.execution_log.append({
            'timestamp': event.timestamp,
            'agent': event.agent_name,
            'status': event.status,
            'message': event.message
        })
        
        if event.status == "ERROR":
            self.logger.error(f"Agent {event.agent_name} error: {event.message}")
    
    def _handle_trade_signal(self, event: TradeSignalEvent) -> None:
        """Process final trade signals."""
        self.pipeline_results['trade_signal'] = event
        self.current_stage = PipelineStage.COMPLETE
        self._invoke_callbacks(self.on_complete_callbacks, event)
    
    def _process(self) -> None:
        """Execute the full pipeline."""
        self.run_pipeline()
    
    def run_pipeline(self, start_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute the complete agent pipeline.
        
        Args:
            start_date: Optional override for data start date
            
        Returns:
            Dictionary containing pipeline results
        """
        self.pipeline_results = {}
        self.execution_log = []
        start_time = datetime.now()
        
        self.logger.info("="*50)
        self.logger.info("STARTING PIPELINE EXECUTION")
        self.logger.info("="*50)
        
        try:
            # Stage 1: Data Acquisition
            self._execute_stage(
                PipelineStage.DATA_ACQUISITION,
                "MarketDataAgent",
                MarketDataEvent
            )
            
            # Stage 2: Regime Detection
            self._execute_stage(
                PipelineStage.REGIME_DETECTION,
                "MLAgent",
                RegimeEvent
            )
            
            # Stage 3: Quantitative Analysis
            self._execute_stage(
                PipelineStage.QUANT_ANALYSIS,
                "QuantAgent",
                QuantSignalEvent
            )
            
            # Stage 4: Risk Assessment
            self._execute_stage(
                PipelineStage.RISK_ASSESSMENT,
                "RiskManagerAgent",
                RiskAdjustedSignalEvent
            )
            
            # Stage 5: Portfolio Optimization
            self._execute_stage(
                PipelineStage.PORTFOLIO_OPTIMIZATION,
                "PortfolioAgent",
                TradeSignalEvent
            )
            
            self.current_stage = PipelineStage.COMPLETE
            
        except Exception as e:
            self.current_stage = PipelineStage.ERROR
            self.pipeline_results['error'] = str(e)
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            self._invoke_callbacks(self.on_error_callbacks, e)
            raise
        
        finally:
            elapsed = (datetime.now() - start_time).total_seconds()
            self.pipeline_results['execution_time'] = elapsed
            self.logger.info(f"Pipeline completed in {elapsed:.2f} seconds")
        
        return self.pipeline_results
    
    def _execute_stage(self, stage: PipelineStage, agent_name: str, 
                       expected_event: type) -> Event:
        """
        Execute a single pipeline stage.
        
        Args:
            stage: Pipeline stage enum
            agent_name: Name of agent to execute
            expected_event: Expected output event type
            
        Returns:
            The output event from the agent
        """
        self.current_stage = stage
        self.logger.info(f"\n[{stage.value.upper()}] Running {agent_name}...")
        
        if agent_name not in self.agents:
            raise ValueError(f"Agent not registered: {agent_name}")
        
        agent = self.agents[agent_name]
        
        # Retry logic
        last_error = None
        for attempt in range(self.retry_attempts):
            try:
                # Run agent
                agent.run()
                
                # Get output event
                event = self.bus.get_last_event(expected_event)
                
                if event is None:
                    raise ValueError(f"No {expected_event.__name__} produced")
                
                # Store result
                self.pipeline_results[stage.value] = event
                self.logger.info(f"[{stage.value.upper()}] Completed successfully")
                
                return event
                
            except Exception as e:
                last_error = e
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(1)  # Brief pause before retry
        
        raise RuntimeError(f"Stage {stage.value} failed after {self.retry_attempts} attempts: {last_error}")
    
    def _invoke_callbacks(self, callbacks: List[Callable], arg: Any) -> None:
        """Invoke registered callbacks."""
        for callback in callbacks:
            try:
                callback(arg)
            except Exception as e:
                self.logger.error(f"Callback error: {e}")
    
    # =========================================================================
    # Agent Management
    # =========================================================================
    
    def start_all_agents(self) -> Dict[str, bool]:
        """
        Start all registered agents.
        
        Returns:
            Dict mapping agent name -> success status
        """
        results = {}
        for name, agent in self.agents.items():
            try:
                success = agent.start()
                results[name] = success
                if success:
                    self.logger.info(f"Started {name}")
                else:
                    self.logger.error(f"Failed to start {name}")
            except Exception as e:
                results[name] = False
                self.logger.error(f"Error starting {name}: {e}")
        
        return results
    
    def stop_all_agents(self) -> None:
        """Stop all registered agents."""
        for name, agent in self.agents.items():
            try:
                agent.stop()
                self.logger.info(f"Stopped {name}")
            except Exception as e:
                self.logger.error(f"Error stopping {name}: {e}")
    
    def get_agent_status(self) -> Dict[str, str]:
        """Get status of all agents."""
        return {name: agent.get_state().value for name, agent in self.agents.items()}
    
    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get agent by name."""
        return self.agents.get(name)
    
    # =========================================================================
    # Pipeline Control
    # =========================================================================
    
    def run_backtest(self, start_date: str, end_date: Optional[str] = None,
                     rebalance_freq: str = 'M') -> pd.DataFrame:
        """
        Run a backtest using the agent pipeline.
        
        Args:
            start_date: Backtest start date
            end_date: Backtest end date (None for today)
            rebalance_freq: Rebalancing frequency ('D', 'W', 'M')
            
        Returns:
            DataFrame with backtest results
        """
        from backtesting.backtest_engine import AgentBacktester
        
        backtester = AgentBacktester(
            orchestrator=self,
            message_bus=self.bus
        )
        
        return backtester.run(
            start_date=start_date,
            end_date=end_date,
            rebalance_freq=rebalance_freq
        )
    
    def on_complete(self, callback: Callable) -> None:
        """Register callback for pipeline completion."""
        self.on_complete_callbacks.append(callback)
    
    def on_error(self, callback: Callable) -> None:
        """Register callback for pipeline errors."""
        self.on_error_callbacks.append(callback)
    
    # =========================================================================
    # Monitoring
    # =========================================================================
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get summary of last pipeline execution."""
        return {
            'stage': self.current_stage.value,
            'results': {k: type(v).__name__ for k, v in self.pipeline_results.items()},
            'execution_time': self.pipeline_results.get('execution_time'),
            'agents': self.get_agent_status(),
            'log_entries': len(self.execution_log)
        }
    
    def get_execution_log(self) -> List[Dict]:
        """Get detailed execution log."""
        return self.execution_log
    
    def get_final_trades(self) -> Optional[List]:
        """Get final trade orders from last run."""
        trade_signal = self.pipeline_results.get('portfolio_optimization')
        if trade_signal and isinstance(trade_signal, TradeSignalEvent):
            return trade_signal.trades
        return None
    
    def print_trade_summary(self) -> None:
        """Print human-readable trade summary."""
        trades = self.get_final_trades()
        if not trades:
            print("No trades available")
            return
        
        print("\n" + "="*60)
        print("TRADE SIGNALS")
        print("="*60)
        
        for action_type in ["BUY", "SELL", "HOLD"]:
            action_trades = [t for t in trades if t.action == action_type]
            if action_trades:
                print(f"\n{action_type}:")
                for t in action_trades:
                    print(f"  {t.ticker}: {t.current_weight:.1%} -> {t.target_weight:.1%} "
                          f"(Δ{t.delta_weight:+.1%}) [{t.urgency}]")
        
        print("\n" + "="*60)


def create_pipeline(config: Optional[Dict] = None) -> Orchestrator:
    """
    Factory function to create a complete agent pipeline.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured Orchestrator with all agents registered
    """
    from agents.market_data_agent import MarketDataAgent
    from agents.ml_agent import MLAgent
    from agents.quant_agent import QuantAgent
    from agents.risk_manager_agent import RiskManagerAgent
    from agents.portfolio_agent import PortfolioAgent
    
    # Create message bus
    bus = MessageBus()
    
    # Create orchestrator
    orchestrator = Orchestrator(bus, config)
    
    # Create and register agents
    market_data_agent = MarketDataAgent(bus, config.get('market_data', {}) if config else {})
    ml_agent = MLAgent(bus, config.get('ml', {}) if config else {})
    quant_agent = QuantAgent(bus, config.get('quant', {}) if config else {})
    risk_manager = RiskManagerAgent(bus, config.get('risk', {}) if config else {})
    portfolio_agent = PortfolioAgent(bus, config.get('portfolio', {}) if config else {})
    
    orchestrator.register_agent(market_data_agent)
    orchestrator.register_agent(ml_agent)
    orchestrator.register_agent(quant_agent)
    orchestrator.register_agent(risk_manager)
    orchestrator.register_agent(portfolio_agent)
    
    return orchestrator
