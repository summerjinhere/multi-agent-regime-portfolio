"""
Multi-Agent Portfolio Optimization System (FIXED)
==================================================

Main entry point for running the agent-based portfolio optimization.

Key Fixes:
1. Proper detailed_weights saving for dashboard
2. Better error handling
3. Fallback portfolio generation
"""

import argparse
import logging
import sys
import os
from datetime import datetime
from typing import Dict, Optional

# Optional YAML support
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print("Warning: PyYAML not installed. Using default configuration.")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from events.message_bus import MessageBus
from agents.orchestrator import Orchestrator
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
    if not YAML_AVAILABLE:
        return get_default_config()
    
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return get_default_config()


def get_default_config() -> Dict:
    """Return default configuration."""
    return {
        'market_data': {
            'start_date': '2015-01-01',
            'end_date': '2025-11-29',
            'cache_ttl_hours': 24,
            'min_data_quality': 0.5,
        },
        'ml': {
            'n_components': 3,
            'covariance_type': 'full',
            'n_iter': 1000,
            'feature_window': 10,
        },
        'quant': {
            'lookback_window': 252,
            'forecast_horizon': 5,
            'use_shrinkage': True,
            'regime_adjustment': True,
        },
        'risk': {
            'max_drawdown_limit': 0.20,
            'var_confidence': 0.95,
            'max_position': 0.30,
            'max_leverage': 1.0,
        },
        'portfolio': {
            'use_black_litterman': True,
            'rebalance_threshold': 0.03,
            'max_turnover': 0.40,
            'transaction_cost_bps': 10,
            'risk_free_rate': 0.02,
        },
        'orchestrator': {
            'timeout_seconds': 60,
            'retry_attempts': 3,
        },
        'backtest': {
            'initial_capital': 1000000,
            'start_date': '2020-01-01',
            'rebalance_freq': 'M',
        }
    }


def create_agents(bus: MessageBus, config: Dict) -> Orchestrator:
    """Create and wire up all agents."""
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


def save_portfolio_files(trade_signal, orchestrator) -> None:
    """
    Save portfolio files for dashboard connectivity.
    
    Creates:
    - final_portfolio.csv (asset class level)
    - final_portfolio_detailed.csv (stock level)
    """
    import pandas as pd
    
    if trade_signal is None:
        print("Warning: No trade signal to save")
        return
    
    # Save asset-class level weights
    if trade_signal.target_weights:
        weights = pd.Series(trade_signal.target_weights)
        weights.to_csv('final_portfolio.csv')
        print(f"Saved final_portfolio.csv with {len(weights)} asset classes")
    
    # Save detailed stock-level weights (CRITICAL for dashboard)
    if hasattr(trade_signal, 'detailed_weights') and trade_signal.detailed_weights:
        detailed = pd.Series(trade_signal.detailed_weights)
        detailed.to_csv('final_portfolio_detailed.csv')
        print(f"Saved final_portfolio_detailed.csv with {len(detailed)} positions")
        
        # Show top positions
        print("\nTop 10 Positions:")
        for ticker, weight in detailed.nlargest(10).items():
            print(f"  {ticker}: {weight:.2%}")
    else:
        # Fallback: use target_weights
        print("Warning: No detailed_weights, using target_weights as fallback")
        weights = pd.Series(trade_signal.target_weights)
        weights.to_csv('final_portfolio_detailed.csv')


def generate_fallback_portfolio(orchestrator) -> None:
    """Generate fallback portfolio if pipeline fails."""
    import pandas as pd
    
    print("\nGenerating fallback portfolio...")
    
    try:
        market_agent = orchestrator.get_agent("MarketDataAgent")
        if market_agent and market_agent.prices is not None:
            assets = market_agent.prices.columns.tolist()
            n = len(assets)
            if n > 0:
                weights = {asset: 1.0/n for asset in assets}
                detailed = pd.Series(weights)
                detailed.to_csv('final_portfolio_detailed.csv')
                pd.Series(weights).to_csv('final_portfolio.csv')
                print(f"Fallback portfolio saved with {n} equal-weighted assets")
                return
    except Exception as e:
        print(f"Error generating fallback: {e}")
    
    # Ultimate fallback: simple portfolio
    fallback = {
        'AAPL': 0.1, 'MSFT': 0.1, 'AMZN': 0.1, 'GOOGL': 0.1, 'JPM': 0.1,
        'TLT': 0.2, 'GLD': 0.1, 'Cash': 0.2
    }
    pd.Series(fallback).to_csv('final_portfolio_detailed.csv')
    pd.Series(fallback).to_csv('final_portfolio.csv')
    print("Saved default fallback portfolio")


def run_live(orchestrator: Orchestrator) -> None:
    """Run live portfolio optimization."""
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
    
    try:
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
        save_portfolio_files(trade_signal, orchestrator)
        
    except Exception as e:
        print(f"\nPipeline error: {e}")
        generate_fallback_portfolio(orchestrator)
    
    finally:
        # Cleanup
        orchestrator.stop_all_agents()


def run_backtest(orchestrator: Orchestrator, config: Dict) -> None:
    """Run historical backtest."""
    print("\n" + "="*60)
    print("MULTI-AGENT PORTFOLIO OPTIMIZATION SYSTEM")
    print("Mode: BACKTEST")
    print("="*60 + "\n")
    
    from backtesting.backtest_engine import AgentBacktester
    import pandas as pd
    
    backtest_config = config.get('backtest', {})
    
    # Start agents
    print("Starting agents...")
    orchestrator.start_all_agents()
    
    try:
        # Create backtester
        backtester = AgentBacktester(
            orchestrator=orchestrator,
            message_bus=orchestrator.bus,
            initial_capital=backtest_config.get('initial_capital', 1000000),
            transaction_cost_bps=config.get('portfolio', {}).get('transaction_cost_bps', 10)
        )
        
        # Run backtest
        start_date = backtest_config.get('start_date', '2020-01-01')
        rebalance_freq = backtest_config.get('rebalance_freq', 'M')
        
        print(f"\nRunning backtest from {start_date}...")
        print(f"Rebalance frequency: {rebalance_freq}")
        
        results = backtester.run(
            start_date=start_date,
            rebalance_freq=rebalance_freq
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
                elif 'Capital' in metric:
                    print(f"  {metric}: ${value:,.0f}")
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
        save_portfolio_files(final_trade_signal, orchestrator)
        
        # Save additional data
        weights_history = backtester.get_weights_history()
        weights_history.to_csv('weights_history.csv')
        print("Weights history saved to weights_history.csv")
        
        regime_history = backtester.get_regime_history()
        if len(regime_history) > 0:
            regime_history.to_csv('regime_history.csv')
            print("Regime history saved to regime_history.csv")
        
    except Exception as e:
        print(f"\nBacktest error: {e}")
        import traceback
        traceback.print_exc()
        generate_fallback_portfolio(orchestrator)
    
    finally:
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
    if YAML_AVAILABLE and os.path.exists(config_path):
        config = load_config(config_path)
        print(f"Loaded config from {config_path}")
    else:
        config = get_default_config()
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
