#!/usr/bin/env python
"""
Run Backtest Script
===================

Easy-to-use script to run the multi-agent portfolio backtest.
Can use synthetic data or load from Yahoo Finance.

Usage:
------
python run_backtest.py                    # Use synthetic data
python run_backtest.py --yahoo            # Use Yahoo Finance data
python run_backtest.py --start 2020-01-01 --end 2025-11-30
python run_backtest.py --export-csv       # Export results to CSV
python run_backtest.py --export-excel     # Export results to Excel
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import MultiAgentOrchestrator
from config.settings import SystemConfig
from data.data_loader import DataLoader, BacktestExporter, PerformanceAnalyzer
import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run Multi-Agent Portfolio Backtest'
    )
    parser.add_argument(
        '--yahoo', 
        action='store_true',
        help='Use Yahoo Finance data instead of synthetic'
    )
    parser.add_argument(
        '--start',
        type=str,
        default='2018-01-01',
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end',
        type=str,
        default='2025-11-30',
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--capital',
        type=float,
        default=1_000_000,
        help='Initial capital'
    )
    parser.add_argument(
        '--rebalance',
        type=str,
        choices=['daily', 'weekly', 'monthly'],
        default='weekly',
        help='Rebalancing frequency'
    )
    parser.add_argument(
        '--export-csv',
        action='store_true',
        help='Export results to CSV'
    )
    parser.add_argument(
        '--export-excel',
        action='store_true',
        help='Export results to Excel'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results',
        help='Output directory for exports'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for synthetic data'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("ENHANCED MULTI-AGENT PORTFOLIO SYSTEM")
    print("=" * 70)
    
    # Configuration
    config = SystemConfig(
        initial_capital=args.capital,
        rebalance_frequency=args.rebalance
    )
    
    print(f"\n[Configuration]")
    print(f"  Universe:          {len(config.universe)} assets")
    print(f"  Initial Capital:   ${config.initial_capital:,.0f}")
    print(f"  Rebalance:         {config.rebalance_frequency}")
    print(f"  Period:            {args.start} to {args.end}")
    
    # Load data
    loader = DataLoader()
    
    if args.yahoo:
        print(f"\n[Loading Data from Yahoo Finance...]")
        try:
            prices = loader.load_yahoo(
                tickers=config.universe,
                start_date=args.start,
                end_date=args.end
            )
            # Generate volume data (Yahoo provides this too)
            import yfinance as yf
            data = yf.download(config.universe, start=args.start, end=args.end, progress=False)
            volumes = data['Volume'] if 'Volume' in data.columns else data['Volume']
        except Exception as e:
            print(f"  Warning: Could not load from Yahoo ({e})")
            print(f"  Falling back to synthetic data...")
            prices, volumes = loader.generate_synthetic(
                tickers=config.universe,
                start_date=args.start,
                end_date=args.end,
                seed=args.seed
            )
    else:
        print(f"\n[Generating Synthetic Data...]")
        prices, volumes = loader.generate_synthetic(
            tickers=config.universe,
            start_date=args.start,
            end_date=args.end,
            seed=args.seed
        )
    
    print(f"  Data loaded: {len(prices)} days, {len(prices.columns)} assets")
    
    # Initialize orchestrator
    print(f"\n[Initializing Multi-Agent System...]")
    orchestrator = MultiAgentOrchestrator(config)
    
    # Run backtest
    print(f"\n[Running Backtest...]")
    result = orchestrator.run_backtest(prices, volumes)
    
    # Print results
    orchestrator.print_results(result)
    
    # Additional analysis
    if args.verbose:
        print("\n" + "=" * 70)
        print("ADDITIONAL ANALYSIS")
        print("=" * 70)
        
        print("\n[Monthly Returns]")
        monthly = PerformanceAnalyzer.calculate_monthly_returns(result.returns)
        print((monthly * 100).round(2).to_string())
        
        print("\n[Top 5 Drawdowns]")
        dd = PerformanceAnalyzer.drawdown_analysis(result.returns)
        if len(dd) > 0:
            print(dd.head().to_string())
        
        print("\n[Rolling Metrics (252-day)]")
        rolling = PerformanceAnalyzer.calculate_rolling_metrics(result.returns)
        print(f"  Latest Rolling Sharpe: {rolling['rolling_sharpe'].iloc[-1]:.2f}")
        print(f"  Latest Rolling Vol:    {rolling['rolling_volatility'].iloc[-1]:.1%}")
    
    # Export results
    if args.export_csv:
        print(f"\n[Exporting to CSV...]")
        BacktestExporter.to_csv(result, output_dir=args.output_dir)
        print(f"  Exported to {args.output_dir}/")
    
    if args.export_excel:
        print(f"\n[Exporting to Excel...]")
        excel_path = f"{args.output_dir}/backtest_results.xlsx"
        BacktestExporter.to_excel(result, filepath=excel_path)
        print(f"  Exported to {excel_path}")
    
    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)
    
    return result


if __name__ == "__main__":
    result = main()
