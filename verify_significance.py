import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from scipy import stats

def calculate_p_value():
    print("Loading Backtest Results...")
    try:
        strategy = pd.read_csv("backtest_results.csv", index_col='Date', parse_dates=True)
    except FileNotFoundError:
        print("Error: backtest_results.csv not found. Please run the model first.")
        return

    # Calculate Strategy Returns
    strategy_ret = strategy['Value'].pct_change().dropna()
    
    print("Fetching Benchmark Data (SPY)...")
    start_date = strategy_ret.index[0]
    end_date = strategy_ret.index[-1]
    
    # Fetch SPY
    spy = yf.download("SPY", start=start_date, end=end_date + pd.Timedelta(days=5), auto_adjust=True)
    
    # Handle MultiIndex if present (yfinance update)
    if isinstance(spy.columns, pd.MultiIndex):
        if 'Close' in spy.columns.levels[0]:
            spy = spy['Close']
        elif 'Adj Close' in spy.columns.levels[0]:
            spy = spy['Adj Close']
        else:
            spy = spy.iloc[:, 0] # Fallback
    elif 'Close' in spy.columns:
        spy = spy['Close']
        
    # Resample to match strategy frequency (Monthly)
    # Strategy is monthly based on backtest_results.csv (dates like 2020-01-31, 2020-02-29)
    spy_monthly = spy.resample('M').last()
    
    # Align dates
    # We need to make sure indices match. 
    # Strategy index is month-end. SPY resampled should be month-end.
    
    # Calculate Benchmark Returns
    benchmark_ret = spy_monthly.pct_change().dropna()
    
    # Align dataframes
    common_index = strategy_ret.index.intersection(benchmark_ret.index)
    
    if len(common_index) < 10:
        print("Not enough overlapping data points for regression.")
        return

    y = strategy_ret.loc[common_index]
    x = benchmark_ret.loc[common_index]
    
    # Risk Free Rate (approx 0.02/12 monthly)
    rf = 0.02 / 12
    
    # Excess Returns
    y_excess = y - rf
    x_excess = x - rf
    
    # OLS Regression
    X = sm.add_constant(x_excess) # Adds Alpha
    model = sm.OLS(y_excess, X).fit()
    
    print("\n" + "="*40)
    print("STATISTICAL SIGNIFICANCE REPORT")
    print("="*40)
    print(model.summary())
    
    alpha = model.params['const']
    p_value = model.pvalues['const']
    beta = model.params.iloc[1]
    
    print("\n" + "-"*40)
    print(f"Alpha (Monthly): {alpha:.5f}")
    print(f"Alpha (Annualized): {alpha*12:.5f}")
    print(f"Beta: {beta:.4f}")
    print(f"P-Value (Alpha): {p_value:.6f}")
    print("-"*40)
    
    if p_value < 0.05:
        print("RESULT: Statistically Significant Outperformance (p < 0.05)")
    else:
        print("RESULT: Not Statistically Significant (p >= 0.05)")

if __name__ == "__main__":
    calculate_p_value()
