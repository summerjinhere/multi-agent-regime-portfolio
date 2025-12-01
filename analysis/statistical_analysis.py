"""
Statistical Analysis Module
============================

Features:
1. OLS Regression with Newey-West standard errors
2. Block Bootstrap for Sharpe Ratio
3. Deflated Sharpe Ratio (Bailey & López de Prado)
4. Factor model regression (Fama-French)
5. Statistical significance testing

Author: Anonymous
Date: November 2025
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


@dataclass
class RegressionResult:
    """Container for regression results"""
    coefficients: np.ndarray
    std_errors: np.ndarray
    t_stats: np.ndarray
    p_values: np.ndarray
    r_squared: float
    adj_r_squared: float
    f_stat: float
    f_pvalue: float
    residuals: np.ndarray
    n_obs: int
    dof: int
    variable_names: List[str] = field(default_factory=list)


@dataclass
class BootstrapResult:
    """Container for bootstrap results"""
    original_stat: float
    bootstrap_mean: float
    bootstrap_std: float
    bias: float
    ci_lower: float
    ci_upper: float
    ci_level: float
    n_bootstrap: int
    bootstrap_distribution: np.ndarray = None


@dataclass
class DSRResult:
    """Container for Deflated Sharpe Ratio results"""
    sharpe_ratio: float
    sharpe_std_error: float
    t_statistic: float
    p_value: float
    psr: float  # Probabilistic Sharpe Ratio
    dsr: float  # Deflated Sharpe Ratio
    expected_max_sr: float
    skewness: float
    kurtosis: float
    n_obs: int
    n_trials: int


class OLSRegression:
    """
    OLS Regression with robust standard errors.
    Supports Newey-West HAC standard errors for time series.
    """
    
    def __init__(self, add_constant: bool = True):
        self.add_constant = add_constant
        self.coefficients = None
        self.residuals = None
        self.fitted_values = None
    
    def fit(self,
           X: Union[np.ndarray, pd.DataFrame],
           y: Union[np.ndarray, pd.Series],
           newey_west_lags: int = 0,
           variable_names: Optional[List[str]] = None) -> RegressionResult:
        """
        Fit OLS regression.
        
        Parameters:
        -----------
        X : array-like
            Independent variables (n x k)
        y : array-like
            Dependent variable (n x 1)
        newey_west_lags : int
            Number of lags for Newey-West HAC standard errors
        variable_names : list
            Names for variables (for reporting)
        
        Returns:
        --------
        RegressionResult
        """
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            if variable_names is None:
                variable_names = X.columns.tolist()
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        X = np.atleast_2d(X)
        if X.shape[0] == 1:
            X = X.T
        
        y = y.flatten()
        n = len(y)
        
        # Add constant if requested
        if self.add_constant:
            X = np.column_stack([np.ones(n), X])
            if variable_names:
                variable_names = ['const'] + variable_names
        
        k = X.shape[1]
        
        # OLS estimation: β = (X'X)^(-1)X'y
        XtX = np.dot(X.T, X)
        XtX_inv = np.linalg.inv(XtX)
        Xty = np.dot(X.T, y)
        
        self.coefficients = np.dot(XtX_inv, Xty)
        
        # Fitted values and residuals
        self.fitted_values = np.dot(X, self.coefficients)
        self.residuals = y - self.fitted_values
        
        # Calculate R-squared
        ss_res = np.sum(self.residuals ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k)
        
        # Standard errors
        if newey_west_lags > 0:
            # Newey-West HAC standard errors
            std_errors = self._newey_west_se(X, self.residuals, newey_west_lags)
        else:
            # Classical OLS standard errors
            mse = ss_res / (n - k)
            var_coef = mse * XtX_inv
            std_errors = np.sqrt(np.diag(var_coef))
        
        # T-statistics and p-values
        t_stats = self.coefficients / (std_errors + 1e-10)
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k))
        
        # F-statistic (excluding constant)
        if k > 1:
            r_matrix = np.eye(k)[1:, :] if self.add_constant else np.eye(k)
            restricted_ss = np.sum((y - y.mean()) ** 2)
            f_stat = ((restricted_ss - ss_res) / (k - 1)) / (ss_res / (n - k))
            f_pvalue = 1 - stats.f.cdf(f_stat, k - 1, n - k)
        else:
            f_stat = 0.0
            f_pvalue = 1.0
        
        return RegressionResult(
            coefficients=self.coefficients,
            std_errors=std_errors,
            t_stats=t_stats,
            p_values=p_values,
            r_squared=r_squared,
            adj_r_squared=adj_r_squared,
            f_stat=f_stat,
            f_pvalue=f_pvalue,
            residuals=self.residuals,
            n_obs=n,
            dof=n - k,
            variable_names=variable_names or [f'x{i}' for i in range(k)]
        )
    
    def _newey_west_se(self,
                      X: np.ndarray,
                      residuals: np.ndarray,
                      max_lag: int) -> np.ndarray:
        """
        Calculate Newey-West HAC standard errors.
        
        Accounts for heteroskedasticity and autocorrelation.
        """
        n, k = X.shape
        
        # Calculate meat of sandwich estimator
        meat = np.zeros((k, k))
        
        # Lag 0 (heteroskedasticity)
        for t in range(n):
            xt = X[t, :].reshape(-1, 1)
            meat += (residuals[t] ** 2) * np.dot(xt, xt.T)
        
        # Lags 1 to max_lag (autocorrelation)
        for lag in range(1, max_lag + 1):
            weight = 1 - lag / (max_lag + 1)  # Bartlett kernel
            
            for t in range(lag, n):
                xt = X[t, :].reshape(-1, 1)
                xt_lag = X[t - lag, :].reshape(-1, 1)
                
                gamma = residuals[t] * residuals[t - lag] * (
                    np.dot(xt, xt_lag.T) + np.dot(xt_lag, xt.T)
                )
                meat += weight * gamma
        
        # Sandwich: (X'X)^(-1) * meat * (X'X)^(-1)
        XtX_inv = np.linalg.inv(np.dot(X.T, X))
        robust_var = np.dot(XtX_inv, np.dot(meat, XtX_inv))
        
        return np.sqrt(np.diag(robust_var))
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using fitted model"""
        if self.coefficients is None:
            raise ValueError("Model not fitted")
        
        if self.add_constant:
            X = np.column_stack([np.ones(len(X)), X])
        
        return np.dot(X, self.coefficients)


class BlockBootstrap:
    """
    Block Bootstrap for time series data.
    Preserves autocorrelation structure.
    """
    
    def __init__(self,
                 block_size: int = 21,
                 n_bootstrap: int = 10000,
                 confidence_level: float = 0.95,
                 random_state: int = 42):
        self.block_size = block_size
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.rng = np.random.RandomState(random_state)
    
    def bootstrap_statistic(self,
                           data: np.ndarray,
                           statistic_func: callable,
                           **kwargs) -> BootstrapResult:
        """
        Bootstrap any statistic with block resampling.
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data
        statistic_func : callable
            Function that computes the statistic from data
        **kwargs : dict
            Additional arguments for statistic_func
        
        Returns:
        --------
        BootstrapResult
        """
        n = len(data)
        n_blocks = int(np.ceil(n / self.block_size))
        
        # Original statistic
        original_stat = statistic_func(data, **kwargs)
        
        # Bootstrap
        bootstrap_stats = np.zeros(self.n_bootstrap)
        
        for b in range(self.n_bootstrap):
            # Sample block starting indices
            block_starts = self.rng.randint(0, max(1, n - self.block_size + 1), n_blocks)
            
            # Construct bootstrap sample
            boot_sample = []
            for start in block_starts:
                end = min(start + self.block_size, n)
                boot_sample.extend(data[start:end])
            
            boot_sample = np.array(boot_sample[:n])
            
            # Compute statistic
            try:
                bootstrap_stats[b] = statistic_func(boot_sample, **kwargs)
            except:
                bootstrap_stats[b] = np.nan
        
        # Remove NaN values
        bootstrap_stats = bootstrap_stats[~np.isnan(bootstrap_stats)]
        
        # Compute statistics
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
        
        return BootstrapResult(
            original_stat=original_stat,
            bootstrap_mean=np.mean(bootstrap_stats),
            bootstrap_std=np.std(bootstrap_stats),
            bias=np.mean(bootstrap_stats) - original_stat,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ci_level=self.confidence_level,
            n_bootstrap=len(bootstrap_stats),
            bootstrap_distribution=bootstrap_stats
        )
    
    def bootstrap_sharpe(self,
                        returns: np.ndarray,
                        annualization_factor: float = np.sqrt(252)
                        ) -> BootstrapResult:
        """
        Bootstrap Sharpe ratio specifically.
        """
        def sharpe_func(r, ann=annualization_factor):
            return r.mean() / (r.std() + 1e-10) * ann
        
        return self.bootstrap_statistic(returns, sharpe_func)
    
    def bootstrap_difference(self,
                            data1: np.ndarray,
                            data2: np.ndarray,
                            statistic_func: callable) -> Dict:
        """
        Bootstrap test for difference between two samples.
        
        H0: stat(data1) = stat(data2)
        """
        n = min(len(data1), len(data2))
        d1 = data1[:n]
        d2 = data2[:n]
        
        n_blocks = int(np.ceil(n / self.block_size))
        
        diff_stats = []
        
        for _ in range(self.n_bootstrap):
            # Sample with same blocks for both series
            block_starts = self.rng.randint(0, max(1, n - self.block_size + 1), n_blocks)
            
            boot1, boot2 = [], []
            for start in block_starts:
                end = min(start + self.block_size, n)
                boot1.extend(d1[start:end])
                boot2.extend(d2[start:end])
            
            boot1 = np.array(boot1[:n])
            boot2 = np.array(boot2[:n])
            
            try:
                stat1 = statistic_func(boot1)
                stat2 = statistic_func(boot2)
                diff_stats.append(stat1 - stat2)
            except:
                pass
        
        diff_stats = np.array(diff_stats)
        
        # Original difference
        orig_diff = statistic_func(d1) - statistic_func(d2)
        
        # Two-sided p-value
        p_value = 2 * min(
            np.mean(diff_stats <= 0),
            np.mean(diff_stats >= 0)
        )
        
        return {
            'original_diff': orig_diff,
            'bootstrap_mean': np.mean(diff_stats),
            'bootstrap_std': np.std(diff_stats),
            'p_value': p_value,
            'significant': p_value < (1 - self.confidence_level)
        }


class DeflatedSharpeRatio:
    """
    Deflated Sharpe Ratio implementation.
    Bailey & López de Prado (2014)
    
    Adjusts for:
    - Multiple testing / strategy trials
    - Non-normality of returns
    - Sample size
    """
    
    @staticmethod
    def compute(returns: np.ndarray,
               n_trials: int = 1,
               benchmark_sr: float = 0.0,
               annualization_factor: float = np.sqrt(252)) -> DSRResult:
        """
        Compute Deflated Sharpe Ratio.
        
        Parameters:
        -----------
        returns : np.ndarray
            Return series
        n_trials : int
            Number of independent strategy trials
        benchmark_sr : float
            Benchmark Sharpe ratio (usually 0)
        annualization_factor : float
            Factor to annualize Sharpe ratio
        
        Returns:
        --------
        DSRResult
        """
        returns = np.array(returns).flatten()
        n = len(returns)
        
        # Calculate Sharpe ratio
        mu = returns.mean() * annualization_factor ** 2 / np.sqrt(252)  # Annualized return
        sigma = returns.std() * annualization_factor
        sharpe = mu / (sigma + 1e-10)
        
        # Higher moments
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns) + 3  # Raw kurtosis
        
        # Standard error of Sharpe ratio (Lo, 2002, with non-normality adjustment)
        sr_var = (
            1 + 0.5 * sharpe**2 - skew * sharpe + (kurt - 3) / 4 * sharpe**2
        ) / (n - 1)
        sr_std = np.sqrt(max(0, sr_var))
        
        # T-statistic
        t_stat = sharpe / (sr_std + 1e-10)
        
        # P-value (two-sided)
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 1))
        
        # Probabilistic Sharpe Ratio
        z_psr = (sharpe - benchmark_sr) / (sr_std + 1e-10)
        psr = stats.norm.cdf(z_psr)
        
        # Expected maximum Sharpe ratio under null (for DSR)
        if n_trials > 1:
            # Euler-Mascheroni approximation
            euler_gamma = 0.5772156649
            
            # Expected max of n_trials standard normals
            try:
                z_max = (1 - euler_gamma) * stats.norm.ppf(1 - 1/n_trials) + \
                        euler_gamma * stats.norm.ppf(1 - 1/(n_trials * np.e))
            except:
                z_max = np.sqrt(2 * np.log(n_trials))
            
            expected_max_sr = z_max * sr_std
        else:
            expected_max_sr = 0.0
        
        # Deflated Sharpe Ratio
        z_dsr = (sharpe - expected_max_sr) / (sr_std + 1e-10)
        dsr = stats.norm.cdf(z_dsr)
        
        return DSRResult(
            sharpe_ratio=sharpe,
            sharpe_std_error=sr_std,
            t_statistic=t_stat,
            p_value=p_value,
            psr=psr,
            dsr=dsr,
            expected_max_sr=expected_max_sr,
            skewness=skew,
            kurtosis=kurt,
            n_obs=n,
            n_trials=n_trials
        )
    
    @staticmethod
    def minimum_track_record(target_sr: float,
                            observed_sr: float,
                            sr_std: float,
                            confidence: float = 0.95) -> float:
        """
        Calculate minimum track record length (MinTRL).
        
        How long must the track record be for the Sharpe ratio
        to be statistically significant?
        """
        z = stats.norm.ppf(confidence)
        
        if observed_sr <= target_sr:
            return float('inf')
        
        min_trl = (z * sr_std / (observed_sr - target_sr)) ** 2
        
        return min_trl


class FactorAnalysis:
    """
    Factor model analysis for attribution.
    Supports Fama-French factors and custom factors.
    """
    
    STANDARD_FACTORS = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']
    
    def __init__(self, add_alpha: bool = True):
        self.add_alpha = add_alpha
        self.regression = OLSRegression(add_constant=add_alpha)
    
    def factor_regression(self,
                         returns: np.ndarray,
                         factor_returns: pd.DataFrame,
                         newey_west_lags: int = 5) -> RegressionResult:
        """
        Regress returns on factor returns.
        
        Parameters:
        -----------
        returns : np.ndarray
            Strategy excess returns
        factor_returns : pd.DataFrame
            Factor returns (aligned with strategy returns)
        newey_west_lags : int
            Lags for Newey-West standard errors
        
        Returns:
        --------
        RegressionResult with factor loadings
        """
        # Align data
        if isinstance(returns, pd.Series):
            returns = returns.values
        
        X = factor_returns.values
        variable_names = factor_returns.columns.tolist()
        
        return self.regression.fit(
            X, returns,
            newey_west_lags=newey_west_lags,
            variable_names=variable_names
        )
    
    def attribution(self,
                   returns: np.ndarray,
                   factor_returns: pd.DataFrame) -> Dict:
        """
        Decompose returns into factor contributions and alpha.
        """
        result = self.factor_regression(returns, factor_returns)
        
        # Factor contributions
        contributions = {}
        total_explained = 0
        
        start_idx = 1 if self.add_alpha else 0
        factor_names = result.variable_names[start_idx:]
        factor_betas = result.coefficients[start_idx:]
        
        for i, (name, beta) in enumerate(zip(factor_names, factor_betas)):
            contrib = beta * factor_returns.iloc[:, i].mean() * 252  # Annualized
            contributions[name] = {
                'beta': beta,
                'contribution': contrib,
                't_stat': result.t_stats[start_idx + i],
                'p_value': result.p_values[start_idx + i]
            }
            total_explained += contrib
        
        # Alpha (if included)
        if self.add_alpha:
            alpha_annual = result.coefficients[0] * 252
            contributions['alpha'] = {
                'value': alpha_annual,
                't_stat': result.t_stats[0],
                'p_value': result.p_values[0],
                'significant': result.p_values[0] < 0.05
            }
        
        return {
            'contributions': contributions,
            'r_squared': result.r_squared,
            'total_explained': total_explained,
            'residual_vol': np.std(result.residuals) * np.sqrt(252)
        }


class StatisticalValidator:
    """
    Comprehensive statistical validation for strategy performance.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.bootstrap = BlockBootstrap(confidence_level=confidence_level)
    
    def full_validation(self,
                       returns: np.ndarray,
                       benchmark_returns: Optional[np.ndarray] = None,
                       n_trials: int = 1,
                       factor_returns: Optional[pd.DataFrame] = None
                       ) -> Dict:
        """
        Run comprehensive statistical validation.
        
        Parameters:
        -----------
        returns : np.ndarray
            Strategy returns
        benchmark_returns : np.ndarray, optional
            Benchmark returns for comparison
        n_trials : int
            Number of strategy trials (for DSR)
        factor_returns : pd.DataFrame, optional
            Factor returns for attribution
        
        Returns:
        --------
        Dict with all validation results
        """
        results = {}
        
        # 1. Deflated Sharpe Ratio
        dsr = DeflatedSharpeRatio.compute(returns, n_trials=n_trials)
        results['dsr'] = {
            'sharpe_ratio': dsr.sharpe_ratio,
            'std_error': dsr.sharpe_std_error,
            't_statistic': dsr.t_statistic,
            'p_value': dsr.p_value,
            'psr': dsr.psr,
            'dsr': dsr.dsr,
            'skewness': dsr.skewness,
            'kurtosis': dsr.kurtosis
        }
        
        # 2. Bootstrap Sharpe ratio
        boot = self.bootstrap.bootstrap_sharpe(returns)
        results['bootstrap'] = {
            'original_sharpe': boot.original_stat,
            'bootstrap_mean': boot.bootstrap_mean,
            'bootstrap_std': boot.bootstrap_std,
            'ci_lower': boot.ci_lower,
            'ci_upper': boot.ci_upper,
            'significant': boot.ci_lower > 0
        }
        
        # 3. CAPM regression (if benchmark provided)
        if benchmark_returns is not None:
            ols = OLSRegression(add_constant=True)
            capm = ols.fit(
                benchmark_returns.reshape(-1, 1),
                returns,
                newey_west_lags=5,
                variable_names=['market']
            )
            
            results['capm'] = {
                'alpha': capm.coefficients[0] * 252,  # Annualized
                'alpha_t': capm.t_stats[0],
                'alpha_p': capm.p_values[0],
                'beta': capm.coefficients[1],
                'beta_t': capm.t_stats[1],
                'r_squared': capm.r_squared
            }
        
        # 4. Factor attribution (if factors provided)
        if factor_returns is not None:
            factor_analysis = FactorAnalysis()
            attribution = factor_analysis.attribution(returns, factor_returns)
            results['factor_attribution'] = attribution
        
        # 5. Summary statistics
        results['summary'] = {
            'n_obs': len(returns),
            'mean_return': np.mean(returns) * 252,
            'volatility': np.std(returns) * np.sqrt(252),
            'min_return': np.min(returns),
            'max_return': np.max(returns),
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns),
            'jarque_bera': stats.jarque_bera(returns)
        }
        
        return results
    
    def print_validation_report(self, results: Dict) -> None:
        """Print formatted validation report"""
        
        print("\n" + "=" * 60)
        print("STATISTICAL VALIDATION REPORT")
        print("=" * 60)
        
        # DSR
        print("\n1. DEFLATED SHARPE RATIO ANALYSIS")
        print("-" * 40)
        dsr = results['dsr']
        print(f"   Sharpe Ratio:     {dsr['sharpe_ratio']:.3f}")
        print(f"   Standard Error:   {dsr['std_error']:.3f}")
        print(f"   t-statistic:      {dsr['t_statistic']:.2f}")
        sig = "***" if dsr['p_value'] < 0.01 else "**" if dsr['p_value'] < 0.05 else "*" if dsr['p_value'] < 0.10 else ""
        print(f"   p-value:          {dsr['p_value']:.4f} {sig}")
        print(f"   PSR (vs 0):       {dsr['psr']*100:.1f}%")
        print(f"   DSR:              {dsr['dsr']*100:.1f}%")
        
        # Bootstrap
        print("\n2. BOOTSTRAP CONFIDENCE INTERVAL")
        print("-" * 40)
        boot = results['bootstrap']
        print(f"   Original Sharpe:  {boot['original_sharpe']:.3f}")
        print(f"   Bootstrap Mean:   {boot['bootstrap_mean']:.3f}")
        print(f"   Bootstrap SE:     {boot['bootstrap_std']:.3f}")
        print(f"   95% CI:           [{boot['ci_lower']:.3f}, {boot['ci_upper']:.3f}]")
        print(f"   Significant:      {'Yes ✓' if boot['significant'] else 'No'}")
        
        # CAPM
        if 'capm' in results:
            print("\n3. CAPM REGRESSION")
            print("-" * 40)
            capm = results['capm']
            print(f"   Alpha (annual):   {capm['alpha']*100:.2f}%")
            print(f"   Alpha t-stat:     {capm['alpha_t']:.2f}")
            print(f"   Alpha p-value:    {capm['alpha_p']:.4f}")
            print(f"   Beta:             {capm['beta']:.2f}")
            print(f"   R-squared:        {capm['r_squared']:.4f}")


if __name__ == "__main__":
    # Test statistical analysis
    np.random.seed(42)
    
    # Generate sample returns
    n = 252 * 5  # 5 years of daily data
    returns = np.random.randn(n) * 0.015 + 0.0003  # Slight positive drift
    
    # Add some structure
    for i in range(n):
        if 200 <= i < 250:  # Simulated crisis
            returns[i] *= 2.5
            returns[i] -= 0.01
    
    benchmark = np.random.randn(n) * 0.012 + 0.0002
    
    print("Statistical Analysis Module Test")
    print("=" * 60)
    
    # Full validation
    validator = StatisticalValidator()
    results = validator.full_validation(
        returns,
        benchmark_returns=benchmark,
        n_trials=10
    )
    
    validator.print_validation_report(results)
