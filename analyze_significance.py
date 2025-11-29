import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
import os

def analyze_significance():
    print("Starting statistical analysis...")
    
    # Load backtest results
    try:
        results = pd.read_csv('backtest_results.csv')
        results['Date'] = pd.to_datetime(results['Date'])
        results.set_index('Date', inplace=True)
        strategy_returns = results['Value'].pct_change().dropna()
        
        start_date = strategy_returns.index[0]
        end_date = strategy_returns.index[-1]
        
        print(f"Backtest period: {start_date.date()} to {end_date.date()}")
        
    except Exception as e:
        print(f"Error loading backtest results: {e}")
        return

    # Download Benchmark (SPY)
    print("Downloading SPY benchmark data...")
    try:
        spy = yf.download('SPY', start=start_date, end=end_date + pd.Timedelta(days=1), progress=False)
        if isinstance(spy.columns, pd.MultiIndex):
            spy = spy.xs('Close', level=0, axis=1) if 'Close' in spy.columns.get_level_values(0) else spy.xs('Adj Close', level=0, axis=1)
        elif 'Adj Close' in spy.columns:
            spy = spy['Adj Close']
        elif 'Close' in spy.columns:
            spy = spy['Close']
            
        # Resample to match strategy frequency (Monthly)
        # Assuming backtest is monthly based on previous context
        spy_monthly = spy.resample('M').last()
        spy_returns = spy_monthly.pct_change().dropna()
        
        # Align dates
        common_dates = strategy_returns.index.intersection(spy_returns.index)
        y = strategy_returns.loc[common_dates]
        x = spy_returns.loc[common_dates]
        
        if len(y) < 12:
            print("Warning: Less than 12 data points for regression. Results may be unreliable.")
            
    except Exception as e:
        print(f"Error downloading benchmark: {e}")
        return

    # OLS Regression
    print("Running OLS regression...")
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    
    alpha = model.params['const']
    beta = model.params.iloc[1]
    t_stat_alpha = model.tvalues['const']
    p_value_alpha = model.pvalues['const']
    r_squared = model.rsquared
    
    # Annualize Alpha (assuming monthly data)
    annual_alpha = (1 + alpha) ** 12 - 1
    
    print("-" * 40)
    print("STATISTICAL SIGNIFICANCE REPORT")
    print("-" * 40)
    print(f"Observations: {len(y)}")
    print(f"Alpha (Monthly): {alpha:.4f}")
    print(f"Alpha (Annualized): {annual_alpha:.2%}")
    print(f"Beta: {beta:.4f}")
    print(f"R-Squared: {r_squared:.4f}")
    print(f"Alpha t-stat: {t_stat_alpha:.4f}")
    print(f"Alpha p-value: {p_value_alpha:.4f}")
    print("-" * 40)
    
    significance_msg = "SIGNIFICANT" if p_value_alpha < 0.05 else "NOT SIGNIFICANT"
    print(f"Result is statistically {significance_msg} at 5% level.")

    # Generate Report
    with open('statistical_report.md', 'w') as f:
        f.write("# Statistical Significance Report\n\n")
        f.write(f"**Period:** {start_date.date()} to {end_date.date()}\n")
        f.write(f"**Benchmark:** SPY (S&P 500 ETF)\n\n")
        
        f.write("## Regression Results (vs SPY)\n")
        f.write("| Metric | Value | Description |\n")
        f.write("| :--- | :--- | :--- |\n")
        f.write(f"| **Alpha (Annualized)** | `{annual_alpha:.2%}` | Excess return over benchmark |\n")
        f.write(f"| **Beta** | `{beta:.4f}` | Market sensitivity |\n")
        f.write(f"| **R-Squared** | `{r_squared:.4f}` | Correlation with market |\n")
        f.write(f"| **Alpha t-stat** | `{t_stat_alpha:.4f}` | Significance score |\n")
        f.write(f"| **Alpha p-value** | `{p_value_alpha:.4f}` | Probability of random chance |\n\n")
        
        f.write("## Interpretation\n")
        if p_value_alpha < 0.05:
            f.write(f"> [!SUCCESS]\n> The strategy generated statistically significant alpha (p < 0.05). The t-stat of {t_stat_alpha:.2f} indicates strong evidence against the null hypothesis of zero excess returns.\n")
        elif p_value_alpha < 0.10:
             f.write(f"> [!WARNING]\n> The strategy generated marginally significant alpha (p < 0.10). There is some evidence of skill, but it is not conclusive at the standard 5% level.\n")
        else:
             f.write(f"> [!NOTE]\n> The alpha is not statistically significant (p > 0.10). The excess returns could be due to random chance or the sample size is too small.\n")
             
        f.write("\n## Model Summary\n")
        f.write("```\n")
        f.write(str(model.summary()))
        f.write("\n```\n")

    print("Report saved to statistical_report.md")

if __name__ == "__main__":
    analyze_significance()
