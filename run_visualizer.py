import pandas as pd
import numpy as np
import yfinance as yf
from utils.visualizer import Visualizer
from models.hmm_regime import RegimeDetector
import os

def run_viz():
    print("Starting Standalone Visualizer...")
    viz = Visualizer()
    
    # 1. Load Data
    print("Loading Data...")
    if os.path.exists("data/market_data.csv"):
        prices = pd.read_csv("data/market_data.csv", index_col=0, parse_dates=True)
    else:
        print("Error: market_data.csv not found.")
        return

    # 2. Load Backtest Results
    print("Loading Backtest Results...")
    if os.path.exists("backtest_results.csv"):
        results = pd.read_csv("backtest_results.csv", index_col=0, parse_dates=True)
        
        # Calculate Metrics for Report
        total_ret = (results['Value'].iloc[-1] / results['Value'].iloc[0]) - 1
        days = (results.index[-1] - results.index[0]).days
        ann_ret = (1 + total_ret) ** (365/days) - 1
        ann_vol = results['Value'].pct_change().std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol != 0 else 0
        cum_ret = (1 + results['Value'].pct_change()).cumprod()
        drawdown = (cum_ret - cum_ret.cummax()) / cum_ret.cummax()
        max_dd = drawdown.min()
        
        metrics = {
            "Total Return": total_ret,
            "Annualized Return": ann_ret,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": max_dd
        }
        
        # Fetch Benchmark (SPY)
        print("Fetching Benchmark (SPY)...")
        spy = yf.download("SPY", start=results.index[0], end=results.index[-1] + pd.Timedelta(days=5), auto_adjust=True)
        if isinstance(spy.columns, pd.MultiIndex): 
            spy = spy['Close'] if 'Close' in spy.columns.levels[0] else spy.iloc[:, 0]
        elif 'Close' in spy.columns: 
            spy = spy['Close']
            
        viz.plot_backtest(results['Value'], spy)
    else:
        print("Warning: backtest_results.csv not found. Skipping backtest plot.")
        metrics = {}

    # 3. Re-run Regime Detection for Plotting
    print("Detecting Regimes...")
    regime_detector = RegimeDetector(n_components=3)
    market_proxy = prices.mean(axis=1).to_frame(name='Market')
    regime_detector.prepare_features(market_proxy, asset_col='Market')
    regime_detector.fit()
    
    # Create regime history format expected by visualizer
    # Visualizer expects a DataFrame or list of dicts with 'Date' and 'Regime'
    # The fitted model has .hidden_states which matches the data index
    regime_df = pd.DataFrame({
        'Date': regime_detector.data.index,
        'Regime': regime_detector.hidden_states
    }).set_index('Date')
    
    viz.plot_regimes(prices, regime_df)

    # 4. Load Final Portfolio
    print("Loading Final Portfolio...")
    if os.path.exists("final_portfolio.csv"):
        # Load series
        portfolio = pd.read_csv("final_portfolio.csv", index_col=0, header=None).squeeze("columns")
        # If it read as DataFrame with 1 col, squeeze converts to Series
        # Check if it has a name 0 or 1, usually header=None makes 0: index, 1: value
        if isinstance(portfolio, pd.DataFrame):
            portfolio = pd.read_csv("final_portfolio.csv", index_col=0, header=None)
            portfolio = portfolio.iloc[:, 0]
            
        viz.plot_allocation(portfolio.to_dict())
    else:
        print("Warning: final_portfolio.csv not found. Skipping allocation plot.")

    # 5. Generate Report
    viz.generate_html_report(metrics)
    print("Visualization Complete.")

if __name__ == "__main__":
    run_viz()
