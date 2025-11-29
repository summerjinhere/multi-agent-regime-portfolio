import pandas as pd
import numpy as np
import statsmodels.api as sm
import yfinance as yf
import matplotlib.pyplot as plt

def run_ols_test():
    print("Running OLS Regression (Alpha/Beta Test)...")
    
    # 1. Load Backtest Results
    try:
        results = pd.read_csv("backtest_results.csv", index_col=0, parse_dates=True)
        portfolio_returns = results['Value'].pct_change().dropna()
    except FileNotFoundError:
        print("Error: backtest_results.csv not found. Please run the backtest first.")
        return

    # 2. Load Benchmark (SPY)
    print("Fetching Benchmark Data (SPY)...")
    spy = yf.download("SPY", start=portfolio_returns.index[0], end=portfolio_returns.index[-1] + pd.Timedelta(days=1), auto_adjust=True)
    
    if isinstance(spy.columns, pd.MultiIndex):
        spy_close = spy.xs('Close', level=0, axis=1) if 'Close' in spy.columns.levels[0] else spy.iloc[:, 0]
    else:
        spy_close = spy['Close']
        
    spy_returns = spy_close.pct_change().dropna()
    
    # Align Data
    aligned_data = pd.concat([portfolio_returns, spy_returns], axis=1).dropna()
    aligned_data.columns = ['Portfolio', 'Market']
    
    # 3. Run OLS
    # Y = Portfolio Returns
    # X = Market Returns + Constant (Alpha)
    Y = aligned_data['Portfolio']
    X = sm.add_constant(aligned_data['Market'])
    
    model = sm.OLS(Y, X)
    results = model.fit()
    
    print("\n" + "="*50)
    print("OLS REGRESSION RESULTS")
    print("="*50)
    print(results.summary())
    
    # Extract Key Metrics
    alpha = results.params['const']
    beta = results.params['Market']
    p_value_alpha = results.pvalues['const']
    
    print("\nKey Findings:")
    print(f"Alpha (Daily): {alpha:.6f} (Annualized: {alpha*252:.4f})")
    print(f"Beta: {beta:.4f}")
    print(f"Alpha p-value: {p_value_alpha:.6f}")
    
    if p_value_alpha < 0.05:
        print(">> RESULT: Alpha is STATISTICALLY SIGNIFICANT (p < 0.05)")
    else:
        print(">> RESULT: Alpha is NOT statistically significant.")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(aligned_data['Market'], aligned_data['Portfolio'], alpha=0.5, label='Daily Returns')
    
    # Regression Line
    x_range = np.linspace(aligned_data['Market'].min(), aligned_data['Market'].max(), 100)
    y_pred = alpha + beta * x_range
    plt.plot(x_range, y_pred, color='red', linewidth=2, label=f'OLS Line (Beta={beta:.2f})')
    
    plt.title('Portfolio vs Market Regression (CAPM)', fontsize=14)
    plt.xlabel('Market Returns (SPY)')
    plt.ylabel('Portfolio Returns')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('ols_regression.png')
    print("Regression plot saved to ols_regression.png")

if __name__ == "__main__":
    run_ols_test()
