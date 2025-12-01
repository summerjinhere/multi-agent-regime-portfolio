"""
Statistical Analysis Module
===========================
Provides functions for statistical validation of portfolio performance.
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

def run_ols(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> dict:
    """
    Run OLS regression of Strategy ~ Benchmark.
    Returns Alpha, Beta, and their statistics.
    """
    # Align data
    common_index = strategy_returns.index.intersection(benchmark_returns.index)
    y = strategy_returns.loc[common_index]
    x = benchmark_returns.loc[common_index]
    
    if len(y) < 30:
        return {"error": "Insufficient data for OLS (min 30 points)"}
    
    # Add constant for Alpha
    X = sm.add_constant(x)
    
    model = sm.OLS(y, X)
    # Use Newey-West HAC standard errors (maxlags=1 is a simple default, can be higher)
    results = model.fit(cov_type='HAC', cov_kwds={'maxlags': int(4 * (len(y)/100)**(2/9))}) 
    
    return {
        "alpha": results.params.iloc[0],
        "alpha_pvalue": results.pvalues.iloc[0],
        "alpha_tstat": results.tvalues.iloc[0],
        "beta": results.params.iloc[1],
        "beta_pvalue": results.pvalues.iloc[1],
        "beta_tstat": results.tvalues.iloc[1],
        "r_squared": results.rsquared,
        "adj_r_squared": results.rsquared_adj,
        "n_obs": int(results.nobs),
        "summary": results.summary().as_text()
    }

def run_rolling_ols(strategy_returns: pd.Series, benchmark_returns: pd.Series, window: int = 252) -> pd.DataFrame:
    """
    Run rolling OLS regression to track Alpha/Beta stability.
    """
    # Align data
    common_index = strategy_returns.index.intersection(benchmark_returns.index)
    y = strategy_returns.loc[common_index]
    x = benchmark_returns.loc[common_index]
    
    results = []
    
    from statsmodels.regression.rolling import RollingOLS
    
    X = sm.add_constant(x)
    model = RollingOLS(y, X, window=window)
    rolling_res = model.fit()
    
    params = rolling_res.params.copy()
    tvalues = rolling_res.tvalues.copy()
    
    # params columns: const (Alpha), benchmark (Beta)
    df = pd.DataFrame(index=common_index)
    df['Alpha'] = params['const'] * 252 # Annualized
    df['Alpha_t'] = tvalues['const']
    df['Beta'] = params.iloc[:, 1] # Assuming benchmark is the second column
    
    return df.dropna()

def run_bootstrap(returns: pd.Series, n_samples: int = 1000, risk_free_rate: float = 0.0) -> dict:
    """
    Perform Bootstrap analysis on Sharpe Ratio.
    """
    returns = returns.dropna()
    if len(returns) < 30:
        return {"error": "Insufficient data for Bootstrap"}
    
    observed_sharpe = (returns.mean() - risk_free_rate) / returns.std() * np.sqrt(252)
    
    bootstrap_sharpes = []
    n = len(returns)
    
    np.random.seed(42)
    for _ in range(n_samples):
        # Resample with replacement
        sample = np.random.choice(returns, size=n, replace=True)
        sample_mean = np.mean(sample)
        sample_std = np.std(sample, ddof=1)
        
        if sample_std > 0:
            sharpe = (sample_mean - risk_free_rate) / sample_std * np.sqrt(252)
            bootstrap_sharpes.append(sharpe)
    
    bootstrap_sharpes = np.array(bootstrap_sharpes)
    
    # Calculate confidence intervals
    ci_lower = np.percentile(bootstrap_sharpes, 2.5)
    ci_upper = np.percentile(bootstrap_sharpes, 97.5)
    p_value = (bootstrap_sharpes <= 0).mean() # Prob that Sharpe <= 0
    
    return {
        "observed_sharpe": observed_sharpe,
        "bootstrap_mean": np.mean(bootstrap_sharpes),
        "bootstrap_std": np.std(bootstrap_sharpes),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value_zero": p_value, # Probability that true Sharpe is <= 0
        "distribution": bootstrap_sharpes
    }
