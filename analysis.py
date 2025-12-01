import pandas as pd
import numpy as np
import yfinance as yf
from scipy import stats
import statsmodels.api as sm
from typing import Dict, Tuple, Optional

def load_backtest_results(file_path: str = 'backtest_results.csv') -> pd.DataFrame:
    """Load backtest results from CSV."""
    try:
        df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
        return df
    except Exception as e:
        print(f"Error loading backtest results: {e}")
        return pd.DataFrame()

def fetch_benchmark_data(start_date: str, end_date: str, ticker: str = 'SPY') -> pd.Series:
    """Fetch benchmark data from Yahoo Finance."""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data = data['Close'] # Handle multi-index if present
        
        # If it's still a DataFrame (e.g. single level col but with ticker name), try to get the series
        if isinstance(data, pd.DataFrame):
            if ticker in data.columns:
                return data[ticker]
            elif 'Close' in data.columns:
                return data['Close']
            else:
                return data.iloc[:, 0] # Fallback
        return data
    except Exception as e:
        print(f"Error fetching benchmark data: {e}")
        return pd.Series()

def _get_frequency(returns: pd.Series) -> int:
    """Infer frequency from returns."""
    if len(returns) < 2:
        return 252
    
    days_diff = (returns.index[-1] - returns.index[0]).days
    if days_diff > 0:
        freq = len(returns) / (days_diff / 365.25)
    else:
        freq = 252
        
    # Snap to standard frequencies
    if 10 <= freq <= 14:
        return 12
    elif 200 <= freq <= 300:
        return 252
    elif 48 <= freq <= 56:
        return 52
    else:
        return int(freq)

def calculate_metrics(returns: pd.Series, risk_free_rate: float = 0.02) -> Dict[str, float]:
    """Calculate performance metrics."""
    if len(returns) < 2:
        return {}
    
    freq = _get_frequency(returns)
        
    excess_returns = returns - (risk_free_rate / freq)
    
    total_return = (1 + returns).prod() - 1
    cagr = (1 + total_return) ** (freq / len(returns)) - 1
    volatility = returns.std() * np.sqrt(freq)
    sharpe = excess_returns.mean() / returns.std() * np.sqrt(freq) if returns.std() > 0 else 0
    
    downside_returns = returns[returns < 0]
    sortino = excess_returns.mean() / downside_returns.std() * np.sqrt(freq) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
    
    # Max Drawdown
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    return {
        'Total Return': total_return,
        'CAGR': cagr,
        'Volatility': volatility,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Max Drawdown': max_drawdown
    }

def calculate_significance(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> Dict[str, float]:
    """
    Calculate statistical significance (Alpha, Beta) using OLS with HAC standard errors.
    """
    # Align data
    common_index = strategy_returns.index.intersection(benchmark_returns.index)
    strat = strategy_returns.loc[common_index]
    bench = benchmark_returns.loc[common_index]
    
    if len(strat) < 10:
        return {}
    
    # Prepare regression
    X = sm.add_constant(bench.values)
    y = strat.values
    
    # OLS with Newey-West (HAC) standard errors
    # maxlags=1 is a conservative default for monthly data
    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 1})
    
    alpha = model.params[0]
    beta = model.params[1]
    r_squared = model.rsquared
    alpha_p_value = model.pvalues[0]
    
    # T-test for difference in means (Welch's t-test)
    t_stat, t_p_value = stats.ttest_ind(strat.values, bench.values, equal_var=False)
    
    return {
        'Alpha': alpha,
        'Beta': beta,
        'R-Squared': r_squared,
        'Alpha P-Value': alpha_p_value,
        'T-Test P-Value': t_p_value,
        'Correlation': np.corrcoef(strat, bench)[0, 1]
    }

def calculate_probabilistic_sharpe_ratio(returns: pd.Series, benchmark_sharpe: float = 0.0) -> float:
    """
    Calculate Probabilistic Sharpe Ratio (PSR).
    
    Args:
        returns: Strategy returns series
        benchmark_sharpe: Annualized benchmark Sharpe Ratio
        
    Returns:
        PSR probability (0.0 to 1.0)
    """
    if len(returns) < 2:
        return 0.0
        
    freq = _get_frequency(returns)
    
    # Convert annualized benchmark Sharpe to period Sharpe
    # SR_annual = SR_period * sqrt(freq) -> SR_period = SR_annual / sqrt(freq)
    benchmark_sharpe_period = benchmark_sharpe / np.sqrt(freq)
    
    # Calculate strategy period moments
    sr_period = returns.mean() / returns.std()
    skew = returns.skew()
    kurt = returns.kurtosis()
    n = len(returns)
    
    # Standard deviation of the estimated Sharpe Ratio
    # Reference: Bailey, D. H., & Lopez de Prado, M. (2012). The Sharpe Ratio Efficient Frontier.
    sr_std = np.sqrt((1 - skew * sr_period + (kurt - 1) / 4 * sr_period**2) / (n - 1))
    
    # Z-statistic
    if sr_std == 0:
        return 0.0
        
    z = (sr_period - benchmark_sharpe_period) / sr_std
    
    # PSR
    psr = stats.norm.cdf(z)
    return psr

def calculate_information_ratio(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Calculate Information Ratio.
    Active Return / Tracking Error.
    """
    common_index = returns.index.intersection(benchmark_returns.index)
    strat = returns.loc[common_index]
    bench = benchmark_returns.loc[common_index]
    
    if len(strat) < 2:
        return 0.0
    
    freq = _get_frequency(strat)
        
    active_return = strat - bench
    tracking_error = active_return.std()
    
    if tracking_error == 0:
        return 0.0
        
    # Annualize
    return (active_return.mean() / tracking_error) * np.sqrt(freq)

def bootstrap_analysis(returns: pd.Series, n_samples: int = 1000) -> Dict[str, Tuple[float, float]]:
    """
    Bootstrap analysis for Sharpe Ratio and CAGR.
    Returns 95% confidence intervals.
    """
    if len(returns) < 10:
        return {}
        
    sharpes = []
    cagrs = []
    
    freq = _get_frequency(returns)
    
    values = returns.values
    n = len(values)
    
    for _ in range(n_samples):
        # Simple bootstrap (resample with replacement)
        # For time series, block bootstrap is often preferred, but simple is a good baseline
        sample = np.random.choice(values, size=n, replace=True)
        
        # Calculate metrics for sample
        sample_mean = np.mean(sample)
        sample_std = np.std(sample)
        
        sharpe = sample_mean / sample_std * np.sqrt(freq) if sample_std > 0 else 0
        sharpes.append(sharpe)
        
        total_ret = (1 + sample).prod() - 1
        cagr = (1 + total_ret) ** (freq / n) - 1
        cagrs.append(cagr)
        
    return {
        'Sharpe Ratio CI': (np.percentile(sharpes, 2.5), np.percentile(sharpes, 97.5)),
        'CAGR CI': (np.percentile(cagrs, 2.5), np.percentile(cagrs, 97.5))
    }
