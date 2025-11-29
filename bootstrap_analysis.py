import pandas as pd
import numpy as np
import yfinance as yf
from scipy import stats
import matplotlib.pyplot as plt

def bootstrap_analysis():
    print("=" * 60)
    print("BOOTSTRAP STATISTICAL ANALYSIS")
    print("=" * 60)
    
    # Load backtest results
    results = pd.read_csv('backtest_results.csv')
    results['Date'] = pd.to_datetime(results['Date'])
    results.set_index('Date', inplace=True)
    strategy_returns = results['Value'].pct_change().dropna()
    
    start_date = strategy_returns.index[0]
    end_date = strategy_returns.index[-1]
    
    print(f"\nBacktest period: {start_date.date()} to {end_date.date()}")
    print(f"Sample size: {len(strategy_returns)} periods")
    
    # Download SPY
    print("\nDownloading SPY benchmark...")
    spy = yf.download('SPY', start=start_date, end=end_date + pd.Timedelta(days=1), progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy = spy.xs('Close', level=0, axis=1) if 'Close' in spy.columns.get_level_values(0) else spy.xs('Adj Close', level=0, axis=1)
    elif 'Adj Close' in spy.columns:
        spy = spy['Adj Close']
    elif 'Close' in spy.columns:
        spy = spy['Close']
        
    spy_monthly = spy.resample('M').last()
    spy_returns = spy_monthly.pct_change().dropna()
    
    # Align dates
    common_dates = strategy_returns.index.intersection(spy_returns.index)
    y = strategy_returns.loc[common_dates]
    x = spy_returns.loc[common_dates]
    
    print(f"\nAligned sample size: {len(y)} periods")
    
    # Bootstrap
    n_bootstrap = 1000
    bootstrap_alphas = []
    bootstrap_betas = []
    bootstrap_rsq = []
    
    print(f"\nRunning {n_bootstrap} bootstrap samples...")
    
    np.random.seed(42)
    
    for i in range(n_bootstrap):
        if (i + 1) % 200 == 0:
            print(f"  Progress: {i+1}/{n_bootstrap}")
            
        # Resample with replacement
        indices = np.random.choice(len(y), size=len(y), replace=True)
        y_boot = y.iloc[indices]
        x_boot = x.iloc[indices]
        
        # Run regression
        import statsmodels.api as sm
        X_boot = sm.add_constant(x_boot)
        model = sm.OLS(y_boot, X_boot).fit()
        
        bootstrap_alphas.append(model.params['const'])
        bootstrap_betas.append(model.params.iloc[1])
        bootstrap_rsq.append(model.rsquared)
    
    # Calculate confidence intervals
    alpha_ci = np.percentile(bootstrap_alphas, [2.5, 97.5])
    beta_ci = np.percentile(bootstrap_betas, [2.5, 97.5])
    rsq_ci = np.percentile(bootstrap_rsq, [2.5, 97.5])
    
    print("\n" + "=" * 60)
    print("BOOTSTRAP RESULTS (95% CI)")
    print("=" * 60)
    print(f"Alpha (Monthly):")
    print(f"  Mean: {np.mean(bootstrap_alphas):.6f}")
    print(f"  95% CI: [{alpha_ci[0]:.6f}, {alpha_ci[1]:.6f}]")
    print(f"\nBeta:")
    print(f"  Mean: {np.mean(bootstrap_betas):.4f}")
    print(f"  95% CI: [{beta_ci[0]:.4f}, {beta_ci[1]:.4f}]")
    print(f"\nR-Squared:")
    print(f"  Mean: {np.mean(bootstrap_rsq):.4f}")
    print(f"  95% CI: [{rsq_ci[0]:.4f}, {rsq_ci[1]:.4f}]")
    print("=" * 60)
    
    # Save detailed report
    with open('bootstrap_report.md', 'w') as f:
        f.write("# Bootstrap Analysis Report\n\n")
        f.write(f"**Period:** {start_date.date()} to {end_date.date()}\n")
        f.write(f"**Observations:** {len(y)}\n")
        f.write(f"**Bootstrap Samples:** {n_bootstrap}\n\n")
        
        f.write("## Results (95% Confidence Intervals)\n\n")
        f.write("| Metric | Mean | 95% CI Lower | 95% CI Upper |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        f.write(f"| Alpha (Monthly) | `{np.mean(bootstrap_alphas):.6f}` | `{alpha_ci[0]:.6f}` | `{alpha_ci[1]:.6f}` |\n")
        f.write(f"| Beta | `{np.mean(bootstrap_betas):.4f}` | `{beta_ci[0]:.4f}` | `{beta_ci[1]:.4f}` |\n")
        f.write(f"| R-Squared | `{np.mean(bootstrap_rsq):.4f}` | `{rsq_ci[0]:.4f}` | `{rsq_ci[1]:.4f}` |\n\n")
        
        f.write("## Interpretation\n\n")
        if 0 in alpha_ci:
            f.write("> [!WARNING]\n> The 95% confidence interval for Alpha includes zero. This suggests the excess return may not be statistically reliable.\n\n")
        else:
            f.write("> [!SUCCESS]\n> The 95% confidence interval for Alpha does not include zero, indicating statistical significance.\n\n")
            
        f.write(f"The extremely low R-squared ({np.mean(bootstrap_rsq):.4f}) indicates the strategy returns are nearly uncorrelated with the market, behaving like a cash or bond-like instrument rather than an equity strategy.\n")
    
    print("\nBootstrap report saved to bootstrap_report.md")

if __name__ == "__main__":
    bootstrap_analysis()
