import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from backtester import Backtester
from data_loader import DataFetcher
import os

def run_sensitivity_analysis():
    print("Starting Sensitivity Analysis (Robustness Check)...")
    
    # 1. Load Data
    fetcher = DataFetcher()
    if os.path.exists("data/market_data.csv"):
        prices = pd.read_csv("data/market_data.csv", index_col=0, parse_dates=True)
    else:
        prices = fetcher.fetch_data()
        
    # 2. Define Grid
    # Testing HMM Components (States) vs Lookback Window
    n_components_grid = [2, 3, 4]
    lookback_grid = [126, 252, 504] # 6m, 1y, 2y
    
    results_grid = np.zeros((len(n_components_grid), len(lookback_grid)))
    
    print(f"Grid Search: {len(n_components_grid)} x {len(lookback_grid)} combinations.")
    
    for i, n_comp in enumerate(n_components_grid):
        for j, lookback in enumerate(lookback_grid):
            print(f"\nTesting: HMM States={n_comp}, Lookback={lookback}")
            
            # Initialize Backtester with specific params
            # We use a slightly higher transaction cost here to be conservative
            bt = Backtester(prices, transaction_cost=0.001, n_components=n_comp)
            
            # Run Backtest (Shortened period for speed if needed, but full is better for thesis)
            # Using 2021-Present to save time, or 2018 if possible.
            # Let's use 2020-01-01 as before.
            try:
                res = bt.run_backtest(start_date="2020-01-01", rebalance_freq='M', lookback_window=lookback)
                metrics = bt.calculate_metrics()
                sharpe = metrics['Sharpe Ratio']
            except Exception as e:
                print(f"Run failed: {e}")
                sharpe = 0
                
            results_grid[i, j] = sharpe
            print(f"Result: Sharpe = {sharpe:.4f}")
            
    # 3. Plot Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(results_grid, annot=True, fmt=".2f", cmap="RdYlGn", 
                xticklabels=lookback_grid, yticklabels=n_components_grid)
    
    plt.title('Sensitivity Analysis: Sharpe Ratio Stability', fontsize=16)
    plt.xlabel('Lookback Window (Days)')
    plt.ylabel('HMM States (Regimes)')
    
    plt.tight_layout()
    plt.savefig('sensitivity_heatmap.png')
    print("Sensitivity Analysis Complete. Heatmap saved to sensitivity_heatmap.png")

if __name__ == "__main__":
    run_sensitivity_analysis()
