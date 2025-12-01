#!/usr/bin/env python
"""
Multi-Agent Portfolio System v3.0
==================================

Main entry point for the system.

Usage:
------
# Run backtest
python main.py --backtest

# Run dashboard
python main.py --dashboard

# Run both
python main.py --backtest --dashboard
"""

import argparse
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_backtest():
    """Run the backtest"""
    from orchestrator import main as orchestrator_main
    return orchestrator_main()


def run_dashboard():
    """Run Streamlit dashboard"""
    import subprocess
    dashboard_path = os.path.join(os.path.dirname(__file__), 'dashboard', 'app.py')
    subprocess.run([sys.executable, '-m', 'streamlit', 'run', dashboard_path])


def main():
    parser = argparse.ArgumentParser(
        description='Multi-Agent Portfolio System v3.0'
    )
    parser.add_argument(
        '--backtest',
        action='store_true',
        help='Run backtest'
    )
    parser.add_argument(
        '--dashboard',
        action='store_true',
        help='Launch Streamlit dashboard'
    )
    parser.add_argument(
        '--live',
        action='store_true',
        help='Run in live mode'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/agent_config.yaml',
        help='Path to config file'
    )
    
    args = parser.parse_args()
    
    # Default: run backtest if no arguments
    if not args.backtest and not args.dashboard and not args.live:
        args.backtest = True
    
    if args.backtest:
        print("Running backtest...")
        orchestrator, result = run_backtest()
        
        if args.dashboard:
            print("\nLaunching dashboard...")
            run_dashboard()
            
    elif args.live:
        print("Starting Live Mode...")
        if args.dashboard:
            print("Launching dashboard in background...")
            import threading
            dashboard_thread = threading.Thread(target=run_dashboard)
            dashboard_thread.daemon = True
            dashboard_thread.start()
            
        run_live_mode()
    
    elif args.dashboard:
        run_dashboard()


def run_live_mode():
    """Run the system in live mode (simulated loop)"""
    from orchestrator import Orchestrator
    from config.settings import SystemConfig
    import time
    import pandas as pd
    from datetime import datetime
    
    config = SystemConfig()
    orchestrator = Orchestrator(config)
    orchestrator.initialize()
    
    print("Live mode initialized. Press Ctrl+C to stop.")
    print("Updating every 60 seconds...")
    
    # Create results directory
    os.makedirs('./results', exist_ok=True)
    live_results_file = './results/live_performance.csv'
    
    # Initialize or load existing live results
    if os.path.exists(live_results_file):
        live_history = pd.read_csv(live_results_file, index_col=0, parse_dates=True)
    else:
        live_history = pd.DataFrame(columns=['Portfolio Value', 'Return', 'Regime'])
    
    initial_value = 100000.0  # Starting capital for live simulation
    current_value = initial_value
    
    if not live_history.empty:
        current_value = live_history.iloc[-1]['Portfolio Value']
    
    try:
        while True:
            current_time = datetime.now()
            print(f"\n[{current_time}] Running update...")
            
            # Run agent pipeline
            state = orchestrator.run_once()
            
            # Simulate portfolio value update (using random walk for demo if no real broker)
            # In a real system, this would fetch account balance from broker
            # Here we use the expected return from the model to simulate change
            exp_ret = state.get('expected_return', 0)
            vol = state.get('volatility', 0.15)
            
            # Simulate daily return (scaled to 1 minute for demo)
            dt = 1/252/390 * 60  # 1 minute fraction of trading year
            sim_ret = np.random.normal(exp_ret * dt, vol * np.sqrt(dt))
            
            current_value *= (1 + sim_ret)
            
            # Record result
            new_row = pd.DataFrame({
                'Portfolio Value': [current_value],
                'Return': [sim_ret],
                'Regime': [state.get('regime', 0)]
            }, index=[current_time])
            
            live_history = pd.concat([live_history, new_row])
            live_history.to_csv(live_results_file)
            
            print(f"  Regime: {state.get('regime_name')}")
            print(f"  Portfolio Value: ${current_value:,.2f}")
            print(f"  Saved to {live_results_file}")
            
            time.sleep(60)
            
    except KeyboardInterrupt:
        print("\nLive mode stopped.")

if __name__ == "__main__":
    import numpy as np # Needed for live simulation
    main()
