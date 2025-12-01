"""
Data Loading and Backtesting Utilities
=======================================

Features:
1. Data loading from various sources
2. Data validation and cleaning
3. Synthetic data generation for testing
4. Backtest result export

Author: Anonymous
Date: November 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import os
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Loads and preprocesses market data.
    Supports CSV, Yahoo Finance, and synthetic data.
    """
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        
    def load_csv(self, 
                filepath: str,
                date_column: str = 'Date',
                parse_dates: bool = True) -> pd.DataFrame:
        """Load data from CSV file"""
        df = pd.read_csv(filepath, parse_dates=[date_column] if parse_dates else False)
        if parse_dates:
            df = df.set_index(date_column)
        return df
    
    def load_yahoo(self,
                  tickers: List[str],
                  start_date: str,
                  end_date: str,
                  data_type: str = 'Adj Close') -> pd.DataFrame:
        """
        Load data from Yahoo Finance.
        Requires yfinance package.
        """
        try:
            import yfinance as yf
            
            data = yf.download(
                tickers,
                start=start_date,
                end=end_date,
                progress=False
            )
            
            if data_type in data.columns.get_level_values(0):
                return data[data_type]
            return data
            
        except ImportError:
            logger.warning("yfinance not installed. Use: pip install yfinance")
            return self.generate_synthetic(tickers, start_date, end_date)
    
    def generate_synthetic(self,
                          tickers: List[str],
                          start_date: str,
                          end_date: str,
                          seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate synthetic price and volume data for testing.
        Includes realistic market dynamics:
        - Volatility clustering
        - Regime changes
        - Correlations
        - Fat tails
        """
        np.random.seed(seed)
        
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        n_days = len(dates)
        n_assets = len(tickers)
        
        # Generate correlated returns with regime switching
        prices_data = {}
        volumes_data = {}
        
        # Base correlation structure
        base_corr = self._generate_correlation_matrix(n_assets)
        chol = np.linalg.cholesky(base_corr)
        
        # Parameters by asset (simulate different sectors)
        mus = np.random.uniform(0.0002, 0.0006, n_assets)  # Daily drift
        sigmas = np.random.uniform(0.015, 0.035, n_assets)  # Daily vol
        
        # Generate regime sequence
        regimes = self._generate_regime_sequence(n_days)
        
        for i, ticker in enumerate(tickers):
            # Base price
            price = 100 * (1 + np.random.uniform(-0.3, 0.3))
            prices = [price]
            volumes = []
            
            for t in range(1, n_days):
                regime = regimes[t]
                
                # Adjust volatility for regime
                if regime == 'crisis':
                    vol_mult = 2.5
                    drift_mult = -2.0
                elif regime == 'bear':
                    vol_mult = 1.5
                    drift_mult = -0.5
                elif regime == 'recovery':
                    vol_mult = 1.3
                    drift_mult = 1.5
                else:  # bull/normal
                    vol_mult = 1.0
                    drift_mult = 1.0
                
                # Generate return
                z = np.random.randn()
                vol = sigmas[i] * vol_mult
                drift = mus[i] * drift_mult
                
                ret = drift + vol * z
                
                # Add fat tails (occasional jumps)
                if np.random.random() < 0.02:
                    ret += np.random.choice([-1, 1]) * np.random.exponential(0.03)
                
                # Update price
                price = price * (1 + ret)
                prices.append(price)
                
                # Generate volume (higher in volatile periods)
                base_volume = 1_000_000 * (1 + np.random.uniform(-0.2, 0.2))
                volume = base_volume * vol_mult * (1 + abs(ret) * 10)
                volumes.append(int(volume))
            
            volumes.insert(0, int(1_000_000))  # First day volume
            
            prices_data[ticker] = prices
            volumes_data[ticker] = volumes
        
        prices_df = pd.DataFrame(prices_data, index=dates)
        volumes_df = pd.DataFrame(volumes_data, index=dates)
        
        return prices_df, volumes_df
    
    def _generate_correlation_matrix(self, n: int) -> np.ndarray:
        """Generate realistic correlation matrix with sector structure"""
        # Base correlation
        base_corr = np.eye(n)
        
        # Add sector correlations (5 sectors, 5 assets each assumed)
        sector_size = n // 5
        for s in range(5):
            start = s * sector_size
            end = min(start + sector_size, n)
            for i in range(start, end):
                for j in range(start, end):
                    if i != j:
                        base_corr[i, j] = 0.6 + np.random.uniform(-0.1, 0.1)
        
        # Add market correlation
        for i in range(n):
            for j in range(n):
                if i != j and base_corr[i, j] == 0:
                    base_corr[i, j] = 0.3 + np.random.uniform(-0.1, 0.1)
        
        # Ensure positive definiteness
        eigenvalues = np.linalg.eigvalsh(base_corr)
        if eigenvalues.min() < 0:
            base_corr += (-eigenvalues.min() + 0.01) * np.eye(n)
            # Normalize to correlation matrix
            d = np.sqrt(np.diag(base_corr))
            base_corr = base_corr / np.outer(d, d)
        
        return base_corr
    
    def _generate_regime_sequence(self, n_days: int) -> List[str]:
        """Generate regime sequence with realistic transitions"""
        regimes = []
        current_regime = 'normal'
        
        # Transition probabilities
        transitions = {
            'bull': {'bull': 0.95, 'normal': 0.04, 'bear': 0.01},
            'normal': {'bull': 0.05, 'normal': 0.90, 'bear': 0.04, 'crisis': 0.01},
            'bear': {'normal': 0.05, 'bear': 0.90, 'crisis': 0.05},
            'crisis': {'crisis': 0.70, 'recovery': 0.30},
            'recovery': {'recovery': 0.80, 'bull': 0.15, 'normal': 0.05}
        }
        
        for _ in range(n_days):
            regimes.append(current_regime)
            
            # Transition
            probs = transitions.get(current_regime, transitions['normal'])
            rand = np.random.random()
            cumsum = 0
            for regime, prob in probs.items():
                cumsum += prob
                if rand < cumsum:
                    current_regime = regime
                    break
        
        return regimes


class BacktestExporter:
    """
    Export backtest results to various formats.
    """
    
    @staticmethod
    def to_csv(result, output_dir: str = "./results"):
        """Export results to CSV files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Returns
        result.returns.to_csv(f"{output_dir}/returns.csv")
        
        # Cumulative returns
        result.cumulative_returns.to_csv(f"{output_dir}/cumulative_returns.csv")
        
        # Portfolio values
        result.portfolio_values.to_csv(f"{output_dir}/portfolio_values.csv")
        
        # Weights history
        result.weights_history.to_csv(f"{output_dir}/weights_history.csv")
        
        # Regime history
        result.regime_history.to_csv(f"{output_dir}/regime_history.csv")
        
        # Metrics
        metrics_df = pd.DataFrame([result.metrics])
        metrics_df.to_csv(f"{output_dir}/metrics.csv", index=False)
        
        # Statistical validation
        if result.statistical_validation:
            val_data = []
            for category, values in result.statistical_validation.items():
                if isinstance(values, dict):
                    for key, val in values.items():
                        val_data.append({
                            'category': category,
                            'metric': key,
                            'value': val
                        })
            val_df = pd.DataFrame(val_data)
            val_df.to_csv(f"{output_dir}/statistical_validation.csv", index=False)
        
        logger.info(f"Results exported to {output_dir}/")
    
    @staticmethod
    def to_excel(result, filepath: str = "./results/backtest_results.xlsx"):
        """Export results to Excel file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            result.returns.to_frame('returns').to_excel(writer, sheet_name='Returns')
            result.cumulative_returns.to_frame('cumulative').to_excel(writer, sheet_name='Cumulative')
            result.portfolio_values.to_frame('value').to_excel(writer, sheet_name='Portfolio Value')
            result.weights_history.to_excel(writer, sheet_name='Weights')
            result.regime_history.to_frame('regime').to_excel(writer, sheet_name='Regimes')
            
            pd.DataFrame([result.metrics]).T.to_excel(
                writer, sheet_name='Metrics', header=['Value']
            )
        
        logger.info(f"Results exported to {filepath}")


class PerformanceAnalyzer:
    """
    Analyze backtest performance with detailed metrics.
    """
    
    @staticmethod
    def calculate_rolling_metrics(returns: pd.Series, 
                                 window: int = 252) -> pd.DataFrame:
        """Calculate rolling performance metrics"""
        
        rolling_return = returns.rolling(window).mean() * 252
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        rolling_sharpe = rolling_return / rolling_vol
        
        # Rolling max drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.rolling(window).max()
        rolling_dd = (cumulative / rolling_max - 1).rolling(window).min()
        
        return pd.DataFrame({
            'rolling_return': rolling_return,
            'rolling_volatility': rolling_vol,
            'rolling_sharpe': rolling_sharpe,
            'rolling_max_dd': rolling_dd
        })
    
    @staticmethod
    def calculate_monthly_returns(returns: pd.Series) -> pd.DataFrame:
        """Calculate monthly return matrix"""
        monthly = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Reshape to year x month matrix
        monthly_df = monthly.to_frame('return')
        monthly_df['year'] = monthly_df.index.year
        monthly_df['month'] = monthly_df.index.month
        
        pivot = monthly_df.pivot(index='year', columns='month', values='return')
        pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Add annual return
        annual = returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
        pivot['Year'] = annual.values[:len(pivot)]
        
        return pivot
    
    @staticmethod
    def drawdown_analysis(returns: pd.Series) -> pd.DataFrame:
        """Detailed drawdown analysis"""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = cumulative / rolling_max - 1
        
        # Find drawdown periods
        in_drawdown = drawdown < 0
        drawdown_start = in_drawdown & ~in_drawdown.shift(1).fillna(False)
        drawdown_end = ~in_drawdown & in_drawdown.shift(1).fillna(False)
        
        starts = drawdown.index[drawdown_start]
        ends = drawdown.index[drawdown_end]
        
        drawdowns = []
        for i, start in enumerate(starts):
            # Find corresponding end
            future_ends = ends[ends > start]
            if len(future_ends) > 0:
                end = future_ends[0]
            else:
                end = drawdown.index[-1]
            
            period_dd = drawdown[start:end]
            trough_date = period_dd.idxmin()
            max_dd = period_dd.min()
            
            drawdowns.append({
                'start': start,
                'trough': trough_date,
                'end': end,
                'depth': max_dd,
                'length_days': (end - start).days,
                'recovery_days': (end - trough_date).days
            })
        
        df = pd.DataFrame(drawdowns)
        if len(df) > 0:
            df = df.sort_values('depth').head(10)  # Top 10 worst
        
        return df


def run_example():
    """Run example with synthetic data"""
    
    from main import MultiAgentOrchestrator
    from config.settings import SystemConfig
    
    print("=" * 60)
    print("MULTI-AGENT PORTFOLIO SYSTEM - EXAMPLE RUN")
    print("=" * 60)
    
    # Configuration
    config = SystemConfig()
    
    # Generate synthetic data
    loader = DataLoader()
    prices, volumes = loader.generate_synthetic(
        tickers=config.universe,
        start_date='2018-01-01',
        end_date='2025-11-30',
        seed=42
    )
    
    print(f"\nData generated:")
    print(f"  Period: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"  Assets: {len(config.universe)}")
    print(f"  Days: {len(prices)}")
    
    # Initialize orchestrator
    orchestrator = MultiAgentOrchestrator(config)
    
    # Run backtest
    print("\nRunning backtest...")
    result = orchestrator.run_backtest(prices, volumes)
    
    # Print results
    orchestrator.print_results(result)
    
    # Export results
    BacktestExporter.to_csv(result, output_dir="./results")
    
    # Additional analysis
    print("\n[MONTHLY RETURNS]")
    monthly = PerformanceAnalyzer.calculate_monthly_returns(result.returns)
    print(monthly.round(3).to_string())
    
    print("\n[TOP 5 DRAWDOWNS]")
    dd_analysis = PerformanceAnalyzer.drawdown_analysis(result.returns)
    if len(dd_analysis) > 0:
        print(dd_analysis.head().to_string())
    
    return result


if __name__ == "__main__":
    result = run_example()
