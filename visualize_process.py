import pandas as pd
import numpy as np
from utils.visualizer import Visualizer
from backtester import Backtester
from data_loader import DataFetcher
import os

def run_process_viz():
    print("Running Process Visualization (Deep Dive)...")
    
    # 1. Load Data
    fetcher = DataFetcher()
    if os.path.exists("data/market_data.csv"):
        prices = pd.read_csv("data/market_data.csv", index_col=0, parse_dates=True)
    else:
        prices = fetcher.fetch_data()
        
    # 2. Run Backtest (Need to re-run to capture granular regime probs)
    print("Re-running Backtest to capture granular diagnostics...")
    backtester = Backtester(prices)
    results = backtester.run_backtest(start_date="2020-01-01", rebalance_freq='M')
    metrics = backtester.calculate_metrics()
    
    # 3. Generate Diagnostics Plots
    viz = Visualizer()
    
    # Regime Stability (Stacked Area)
    print("Plotting Regime Stability...")
    viz.plot_regime_stability(backtester.regime_history)
    
    # Rolling Metrics
    print("Plotting Rolling Metrics...")
    viz.plot_rolling_metrics(results['Value'])
    
    # --- New: Correlation Stability ---
    print("Plotting Correlation Stability...")
    # Calculate average correlation of the asset universe over time
    # Use a rolling window (e.g., 6 months)
    returns = prices.pct_change().dropna()
    rolling_corr = returns.rolling(window=126).corr()
    
    # Extract average off-diagonal correlation at each time step
    avg_corrs = []
    dates = returns.index[126:]
    for date in dates:
        # Get correlation matrix for this date
        # rolling_corr is a MultiIndex (Date, Asset) -> Asset
        try:
            matrix = rolling_corr.loc[date]
            # Mean of off-diagonal elements
            # matrix.values[np.triu_indices_from(matrix.values, k=1)] gets upper triangle
            avg_corr = matrix.values[np.triu_indices_from(matrix.values, k=1)].mean()
            avg_corrs.append(avg_corr)
        except:
            avg_corrs.append(np.nan)
            
    corr_series = pd.Series(avg_corrs, index=dates)
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(corr_series, color='teal', label='Avg Asset Correlation (6m Rolling)')
    plt.axhline(corr_series.mean(), color='black', linestyle='--', label='Long-Term Mean')
    plt.title('Correlation Instability: Why Static Models Fail', fontsize=14)
    plt.ylabel('Average Correlation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('correlation_stability.png')
    plt.close()
    
    # Standard Plots
    import yfinance as yf
    spy = yf.download("SPY", start="2020-01-01", end=None, auto_adjust=True)
    if isinstance(spy.columns, pd.MultiIndex): spy = spy['Close'] if 'Close' in spy.columns.levels[0] else spy.iloc[:, 0]
    elif 'Close' in spy.columns: spy = spy['Close']
    
    viz.plot_backtest(results['Value'], spy)
    
    # Regime Scatter
    regime_df = pd.DataFrame(backtester.regime_history).set_index('Date')
    viz.plot_regimes(prices, regime_df)
    
    # 4. Generate Enhanced Report
    viz.generate_html_report(metrics)
    print("Process Visualization Complete. Check reports/report.html")

if __name__ == "__main__":
    run_process_viz()
