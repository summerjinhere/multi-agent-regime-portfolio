"""
Multi-Agent Portfolio Optimization System
==========================================

Main entry point for running the agent-based portfolio optimization.

Usage:
    # Run live prediction
    python main.py
    
    # Run backtest
    python main.py --backtest
    
    # Run with custom config
    python main.py --config config/custom.yaml
"""

import argparse
import logging
import sys
import os
import yaml
from datetime import datetime
from typing import Dict, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from events.message_bus import MessageBus
from agents.orchestrator import create_pipeline, Orchestrator
from agents.market_data_agent import MarketDataAgent
from agents.ml_agent import MLAgent
from agents.quant_agent import QuantAgent
from agents.risk_manager_agent import RiskManagerAgent
from agents.portfolio_agent import PortfolioAgent


def setup_logging(level: str = "INFO") -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_agents(bus: MessageBus, config: Dict) -> Orchestrator:
    """
    Create and wire up all agents.
    
    Args:
        bus: Shared message bus
        config: Configuration dictionary
        
    Returns:
        Configured orchestrator
    """
    # Create orchestrator
    orchestrator = Orchestrator(bus, config.get('orchestrator', {}))
    
    # Create agents with their configs
    market_data_agent = MarketDataAgent(bus, config.get('market_data', {}))
    ml_agent = MLAgent(bus, config.get('ml', {}))
    quant_agent = QuantAgent(bus, config.get('quant', {}))
    risk_manager = RiskManagerAgent(bus, config.get('risk', {}))
    portfolio_agent = PortfolioAgent(bus, config.get('portfolio', {}))
    
    # Register with orchestrator
    orchestrator.register_agent(market_data_agent)
    orchestrator.register_agent(ml_agent)
    orchestrator.register_agent(quant_agent)
    orchestrator.register_agent(risk_manager)
    orchestrator.register_agent(portfolio_agent)
    
    return orchestrator


def run_live(orchestrator: Orchestrator) -> None:
    """
    Run live portfolio optimization.
    
    Args:
        orchestrator: Configured orchestrator
    """
    print("\n" + "="*60)
    print("MULTI-AGENT PORTFOLIO OPTIMIZATION SYSTEM")
    print("Mode: LIVE PREDICTION")
    print("="*60 + "\n")
    
    # Start all agents
    print("Starting agents...")
    start_results = orchestrator.start_all_agents()
    
    for agent, success in start_results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {agent}")
    
    # Run pipeline
    print("\nRunning pipeline...")
    results = orchestrator.run_pipeline()
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    orchestrator.print_trade_summary()
    
    # Get summary
    summary = orchestrator.get_pipeline_summary()
    print(f"\nExecution time: {summary['execution_time']:.2f} seconds")
    
    # Save results
    trade_signal = results.get('portfolio_optimization')
    if trade_signal:
        import pandas as pd
        weights = pd.Series(trade_signal.target_weights)
        weights.to_csv('final_portfolio.csv')
        print("\nPortfolio saved to final_portfolio.csv")
        
        if hasattr(trade_signal, 'detailed_weights') and trade_signal.detailed_weights:
            detailed = pd.Series(trade_signal.detailed_weights)
            detailed.to_csv('final_portfolio_detailed.csv')
            print("Detailed allocation saved to final_portfolio_detailed.csv")
        else:
            print("Detailed weights missing or empty. Using target weights as fallback.")
            detailed = pd.Series(trade_signal.target_weights)
            detailed.to_csv('final_portfolio_detailed.csv')
            print("Fallback allocation saved to final_portfolio_detailed.csv")
            
    else:
        print("No trade signal generated from backtest.")
        # Fail-safe: Generate equal weight portfolio from config
        print("Generating fail-safe portfolio for dashboard...")
        try:
            market_agent = orchestrator.get_agent("MarketDataAgent")
            if market_agent and market_agent.prices is not None:
                assets = market_agent.prices.columns.tolist()
                n = len(assets)
                if n > 0:
                    weights = {asset: 1.0/n for asset in assets}
                    detailed = pd.Series(weights)
                    detailed.to_csv('final_portfolio_detailed.csv')
                    # Also save summary portfolio
                    pd.Series(weights).to_csv('final_portfolio.csv')
                    print(f"Fail-safe portfolio saved with {n} assets.")
        except Exception as e:
            print(f"Could not generate fail-safe portfolio: {e}")
    
    # Cleanup
    orchestrator.stop_all_agents()


def run_backtest(orchestrator: Orchestrator, config: Dict) -> None:
    """
    Run historical backtest.
    
    Args:
        orchestrator: Configured orchestrator
        config: Backtest configuration
    """
    print("\n" + "="*60)
    print("MULTI-AGENT PORTFOLIO OPTIMIZATION SYSTEM")
    print("Mode: BACKTEST")
    print("="*60 + "\n")
    
    from backtesting.backtest_engine import AgentBacktester
    
    backtest_config = config.get('backtest', {})
    
    # Start agents
    print("Starting agents...")
    orchestrator.start_all_agents()
    
    # Create backtester
    backtester = AgentBacktester(
        orchestrator=orchestrator,
        message_bus=orchestrator.bus,
        initial_capital=backtest_config.get('initial_capital', 100000),
        transaction_cost_bps=config.get('portfolio', {}).get('transaction_cost_bps', 10)
    )
    
    # Run backtest
    print(f"\nRunning backtest from {backtest_config.get('start_date', '2020-01-01')}...")
    
    results = backtester.run(
        start_date=backtest_config.get('start_date', '2020-01-01'),
        rebalance_freq=backtest_config.get('rebalance_freq', 'M')
    )
    
    # Calculate metrics
    metrics = backtester.calculate_metrics()
    
    # Print results
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    
    for metric, value in metrics.items():
        if isinstance(value, float):
            if 'Return' in metric or 'Drawdown' in metric or 'Rate' in metric:
                print(f"  {metric}: {value:.2%}")
            else:
                print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    # Save results
    results.to_csv('backtest_results.csv')
    print("\nResults saved to backtest_results.csv")
    
    # Save final portfolio allocation
    from events.base_event import TradeSignalEvent
    final_trade_signal = orchestrator.bus.get_last_event(TradeSignalEvent)
    
    if final_trade_signal:
        import pandas as pd
        weights = pd.Series(final_trade_signal.target_weights)
        weights.to_csv('final_portfolio.csv')
        print("Final portfolio saved to final_portfolio.csv")
        
        # Save detailed weights if available
        if hasattr(final_trade_signal, 'detailed_weights') and final_trade_signal.detailed_weights:
            detailed = pd.Series(final_trade_signal.detailed_weights)
            detailed.to_csv('final_portfolio_detailed.csv')
            print("Detailed allocation saved to final_portfolio_detailed.csv")
        else:
            # Use target weights as fallback
            detailed = pd.Series(final_trade_signal.target_weights)
            detailed.to_csv('final_portfolio_detailed.csv')
            print("Target weights saved as detailed allocation")
    else:
        print("Warning: No final trade signal found in backtest")
    
    # Cleanup
    orchestrator.stop_all_agents()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Multi-Agent Portfolio Optimization')
    parser.add_argument('--backtest', action='store_true', help='Run backtest mode')
    parser.add_argument('--config', type=str, default='config/agent_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Load config
    config_path = args.config
    if os.path.exists(config_path):
        config = load_config(config_path)
        print(f"Loaded config from {config_path}")
    else:
        config = {}
        print("Using default configuration")
    
    # Create message bus and agents
    bus = MessageBus()
    orchestrator = create_agents(bus, config)
    
    # Run appropriate mode
    if args.backtest:
        run_backtest(orchestrator, config)
    else:
        run_live(orchestrator)


if __name__ == "__main__":
    main()
