import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_loader import DataFetcher
from models.hmm_regime import RegimeDetector
from models.ms_vecm import MSVECMAllocator
from models.portfolio_optimizer import PortfolioOptimizer
import os

def main():
    print("Starting Magnum Opus Portfolio Optimization Model...")
    
    # 1. Data Acquisition
    print("\n[1] Fetching Data...")
    fetcher = DataFetcher(start_date="2015-01-01")
    # Check if data exists to avoid re-downloading
    data_path = os.path.join(fetcher.data_dir, "market_data.csv")
    if os.path.exists(data_path):
        print(f"Loading data from {data_path}")
        prices = pd.read_csv(data_path, index_col=0, parse_dates=True)
        print(f"Loaded Prices Head:\n{prices.head()}")
        print(f"Loaded Prices Columns:\n{prices.columns}")
    else:
        prices = fetcher.fetch_data()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_loader import DataFetcher
from models.hmm_regime import RegimeDetector
from models.ms_vecm import MSVECMAllocator
from models.portfolio_optimizer import PortfolioOptimizer
import os

def main():
    print("Starting Magnum Opus Portfolio Optimization Model...")
    
    # 1. Data Acquisition
    print("\n[1] Fetching Data...")
    fetcher = DataFetcher(start_date="2015-01-01")
    # Check if data exists to avoid re-downloading
    data_path = os.path.join(fetcher.data_dir, "market_data.csv")
    if os.path.exists(data_path):
        print(f"Loading data from {data_path}")
        prices = pd.read_csv(data_path, index_col=0, parse_dates=True)
        print(f"Loaded Prices Head:\n{prices.head()}")
        print(f"Loaded Prices Columns:\n{prices.columns}")
    else:
        prices = fetcher.fetch_data()

    # ... (Previous code) ...
    
    # Check for Backtest Mode (Argument or Flag)
    # For now, we will run Backtest AND Current Prediction to show both
    
    print("\n" + "="*50)
    print("RUNNING STATISTICAL VERIFICATION (BACKTEST)")
    print("="*50)
    
    from backtester import Backtester
    from utils.visualizer import Visualizer
    
    viz = Visualizer()
    backtester = Backtester(prices)
    
    # Run backtest
    results = backtester.run_backtest(start_date="2020-01-01", rebalance_freq='M')
    metrics = backtester.calculate_metrics()
    
    # Save Results
    results.to_csv("backtest_results.csv")
    
    # Visualize Backtest
    # Fetch SPY for benchmark comparison
    import yfinance as yf
    spy = yf.download("SPY", start="2020-01-01", end=None, auto_adjust=True)
    if isinstance(spy.columns, pd.MultiIndex): spy = spy['Close'] if 'Close' in spy.columns.levels[0] else spy.iloc[:, 0]
    elif 'Close' in spy.columns: spy = spy['Close']
    
    viz.plot_backtest(results['Value'], spy)
    viz.plot_regimes(prices, backtester.regime_history)
    
    print("\n" + "="*50)
    print("GENERATING CURRENT PORTFOLIO (LIVE)")
    print("="*50)

    # ... (Regime Detection and Optimization Logic) ...
    # We need to capture the final weights to plot allocation
    
    # ... (After optimization loop) ...
    # Let's assume we have 'final_portfolio' dict from the original code
    # We need to ensure we reach that part.
    
    # RE-INSERTING THE ORIGINAL LOGIC FLOW HERE BUT CONNECTED
    
    # 2. Regime Detection (HMM) - Live
    print("\n[2] Detecting Market Regimes (Live)...")
    regime_detector = RegimeDetector(n_components=3)
    market_proxy = prices.mean(axis=1).to_frame(name='Market')
    regime_detector.prepare_features(market_proxy, asset_col='Market')
    regime_detector.fit()
    current_regime = regime_detector.get_current_regime()
    print(f"Current Regime: {current_regime}")

    # 3. Asset Allocation
    # ... (Same as before) ...
    # Aggregating Sectors
    sector_prices = pd.DataFrame()
    for sector, tickers in fetcher.sectors.items():
        valid_tickers = [t for t in tickers if t in prices.columns]
        if valid_tickers:
            sector_prices[sector] = prices[valid_tickers].mean(axis=1)
            
    asset_class_prices = sector_prices.copy()
    if 'TLT' in prices.columns: asset_class_prices['Bonds'] = prices['TLT']
    if 'GLD' in prices.columns: asset_class_prices['Commodities'] = prices['GLD']
    if '^VIX' in prices.columns: asset_class_prices['Volatility'] = prices['^VIX']
    asset_class_prices.dropna(inplace=True)
    
    vecm_allocator = MSVECMAllocator()
    exp_returns, cov_matrix = vecm_allocator.fit_predict(asset_class_prices, current_regime)
    
    optimizer = PortfolioOptimizer()
    asset_weights = optimizer.optimize_asset_classes(exp_returns, cov_matrix, use_black_litterman=True)
    
    # 4. Stock Selection
    final_portfolio = {}
    for sector, weight in asset_weights.items():
        if sector in fetcher.sectors:
            tickers = fetcher.sectors[sector]
            valid_tickers = [t for t in tickers if t in prices.columns]
            sector_data = prices[valid_tickers]
            top_stocks = optimizer.select_top_stocks(sector_data, current_regime, n_top=5)
            stock_weight = weight / len(top_stocks)
            for stock in top_stocks:
                final_portfolio[stock] = stock_weight
        else:
            if sector == 'Bonds' and 'TLT' in prices.columns: final_portfolio['TLT'] = weight
            elif sector == 'Commodities' and 'GLD' in prices.columns: final_portfolio['GLD'] = weight
            elif sector == 'Volatility' and '^VIX' in prices.columns: final_portfolio['^VIX'] = weight

    # Visualize Allocation
    viz.plot_allocation(final_portfolio)
    
    # Generate Report
    viz.generate_html_report(metrics)
    
    print("\n[5] Final Portfolio Allocation:")
    sorted_portfolio = sorted(final_portfolio.items(), key=lambda x: x[1], reverse=True)
    for asset, w in sorted_portfolio:
        print(f"{asset}: {w:.2%}")
        
    pd.Series(final_portfolio).to_csv("final_portfolio.csv")
    print("Portfolio saved to final_portfolio.csv")

if __name__ == "__main__":
    main()
