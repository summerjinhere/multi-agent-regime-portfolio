"""
Orchestrator and Backtest Engine
=================================
Coordinates all agents and runs backtests.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
import os

from config.settings import SystemConfig, load_config
from events.message_bus import MessageBus, get_message_bus, reset_message_bus
from agents.market_data_agent import MarketDataAgent
from agents.ml_agent import MLAgent
from agents.quant_agent import QuantAgent
from agents.risk_manager_agent import RiskManagerAgent
from agents.portfolio_agent import PortfolioAgent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container for backtest results"""
    returns: pd.Series = None
    cumulative_returns: pd.Series = None
    portfolio_values: pd.Series = None
    weights_history: pd.DataFrame = None
    regime_history: pd.Series = None
    trades: List[Dict] = field(default_factory=list)
    metrics: Dict = field(default_factory=dict)
    benchmark_returns: pd.Series = None
    benchmark_cumulative: pd.Series = None


class Orchestrator:
    """
    Main orchestrator that coordinates all agents.
    """
    
    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()
        
        # Reset and get message bus
        reset_message_bus()
        self.message_bus = get_message_bus()
        
        # Initialize agents
        self._init_agents()
        
        # State
        self.is_initialized = False
        self.last_run_time: Optional[datetime] = None
        
    def _init_agents(self) -> None:
        """Initialize all agents."""
        # Market Data Agent
        self.market_data_agent = MarketDataAgent(
            self.message_bus,
            {
                'start_date': self.config.market_data.start_date,
                'end_date': self.config.market_data.end_date,
                'sectors': self.config.market_data.sectors,
                'bonds': self.config.market_data.bonds,
                'commodities': self.config.market_data.commodities,
                'cache_dir': self.config.market_data.cache_dir
            }
        )
        
        # ML Agent (HMM)
        self.ml_agent = MLAgent(
            self.message_bus,
            {
                'n_components': self.config.ml.n_components,
                'covariance_type': self.config.ml.covariance_type,
                'n_iter': self.config.ml.n_iter,
                'feature_window': self.config.ml.feature_window
            }
        )
        
        # Quant Agent (MS-VECM)
        self.quant_agent = QuantAgent(
            self.message_bus,
            {
                'lookback_window': self.config.quant.lookback_window,
                'forecast_horizon': self.config.quant.forecast_horizon,
                'use_shrinkage': self.config.quant.use_shrinkage,
                'regime_adjustment': self.config.quant.regime_adjustment
            }
        )
        
        # Risk Manager Agent
        self.risk_agent = RiskManagerAgent(
            self.message_bus,
            {
                'max_drawdown_limit': self.config.risk.max_drawdown_limit,
                'var_confidence': self.config.risk.var_confidence,
                'var_limit': self.config.risk.var_limit,
                'max_position': self.config.risk.max_position,
                'regime_params': {
                    k: v.__dict__ if hasattr(v, '__dict__') else v
                    for k, v in self.config.risk.regime_params.items()
                }
            }
        )
        
        # Portfolio Agent
        self.portfolio_agent = PortfolioAgent(
            self.message_bus,
            {
                'use_black_litterman': self.config.portfolio.use_black_litterman,
                'rebalance_threshold': self.config.portfolio.rebalance_threshold,
                'max_turnover': self.config.portfolio.max_turnover,
                'transaction_cost_bps': self.config.portfolio.transaction_cost_bps,
                'risk_free_rate': self.config.portfolio.risk_free_rate,
                'max_weight': self.config.portfolio.max_weight
            }
        )
        
        logger.info("All agents initialized")
    
    def initialize(self) -> None:
        """Initialize all agents."""
        if self.is_initialized:
            return
        
        self.market_data_agent.initialize()
        self.ml_agent.initialize()
        self.quant_agent.initialize()
        self.risk_agent.initialize()
        self.portfolio_agent.initialize()
        
        self.is_initialized = True
        logger.info("Orchestrator initialized")
    
    def run_once(self) -> Dict:
        """Run all agents once in sequence."""
        if not self.is_initialized:
            self.initialize()
        
        logger.info("Running agent pipeline...")
        
        # 1. Market Data
        self.market_data_agent.run()
        
        # 2. ML Agent (Regime Detection)
        self.ml_agent.run()
        
        # 3. Quant Agent (Expected Returns)
        self.quant_agent.run()
        
        # 4. Risk Manager
        self.risk_agent.run()
        
        # 5. Portfolio Agent
        self.portfolio_agent.run()
        
        self.last_run_time = datetime.now()
        
        # Return current state
        return self.get_current_state()
    
    def get_current_state(self) -> Dict:
        """Get current state of all agents."""
        return {
            'regime': self.ml_agent.hidden_states[-1] if self.ml_agent.hidden_states is not None else None,
            'regime_name': MLAgent.REGIME_NAMES.get(
                self.ml_agent.hidden_states[-1] if self.ml_agent.hidden_states is not None else 0
            ),
            'target_weights': self.portfolio_agent.target_weights,
            'expected_return': self.portfolio_agent.get_portfolio_metrics().get('expected_return'),
            'volatility': self.portfolio_agent.get_portfolio_metrics().get('volatility'),
            'last_update': self.last_run_time
        }
    
    def run_backtest(self,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None,
                    rebalance_freq: str = 'M') -> BacktestResult:
        """
        Run historical backtest.
        """
        logger.info("Starting backtest...")
        
        # Initialize
        if not self.is_initialized:
            self.initialize()
        
        # Run market data agent to get all data
        self.market_data_agent.run()
        
        # Get prices
        prices = self.market_data_agent.prices
        returns = self.market_data_agent.returns
        
        if prices is None or returns is None:
            raise ValueError("No price data available")
        
        # Date filtering
        start_date = start_date or self.config.backtest.start_date
        end_date = end_date or self.config.backtest.end_date
        
        prices = prices[prices.index >= start_date]
        prices = prices[prices.index <= end_date]
        returns = returns.loc[prices.index[1]:]
        
        # Get benchmark
        benchmark = self.config.backtest.benchmark
        if benchmark in returns.columns:
            benchmark_returns = returns[benchmark]
        else:
            benchmark_returns = returns.mean(axis=1)
            
        # Calculate asset class returns for portfolio simulation
        asset_prices = self.quant_agent.aggregate_assets(prices, self.config.market_data.sectors)
        asset_returns = asset_prices.pct_change().fillna(0)
        
        # Initialize tracking
        initial_capital = self.config.backtest.initial_capital
        portfolio_value = initial_capital
        cash = initial_capital
        
        portfolio_values = []
        weights_history = []
        regime_history = []
        all_returns = []
        trades = []
        
        # Get rebalancing dates
        rebalance_dates = self._get_rebalance_dates(returns.index, rebalance_freq)
        
        # Initial weights (equal weighted)
        n_stocks = len([t for sector in self.config.market_data.sectors.values() for t in sector])
        current_weights = None
        
        logger.info(f"Backtest period: {returns.index[0]} to {returns.index[-1]}")
        logger.info(f"Rebalancing on {len(rebalance_dates)} dates")
        
        # Run backtest
        for i, date in enumerate(returns.index):
            
            # Calculate daily return
            if current_weights is not None:
                # Portfolio return
                day_return = 0
                for asset, weight in current_weights.items():
                    if asset in asset_returns.columns:
                        day_return += weight * asset_returns.loc[date, asset]
                
                all_returns.append(day_return)
                portfolio_value *= (1 + day_return)
            else:
                all_returns.append(0)
            
            portfolio_values.append(portfolio_value)
            
            # Store current weights
            if current_weights is not None:
                weights_history.append(current_weights.copy())
            else:
                weights_history.append({})
            
            # Detect regime (Quantile-based N=5) with Persistence
            lookback = min(60, i+1)
            raw_regime = 2 # Default Neutral
            
            if lookback > 20:
                # Calculate rolling volatility
                recent_vol = returns.iloc[i-lookback:i].std().mean() * np.sqrt(252)
                
                if recent_vol < 0.10:
                    raw_regime = 0
                elif recent_vol < 0.15:
                    raw_regime = 1
                elif recent_vol < 0.20:
                    raw_regime = 2
                elif recent_vol < 0.30:
                    raw_regime = 3
                else:
                    raw_regime = 4
            
            # Apply Persistence Filter (Min 10 days)
            if not regime_history:
                regime = raw_regime
                regime_duration = 1
            else:
                last_regime = regime_history[-1]
                if raw_regime != last_regime:
                    if regime_duration >= 10:
                        regime = raw_regime
                        regime_duration = 1
                    else:
                        regime = last_regime
                        regime_duration += 1
                else:
                    regime = last_regime
                    regime_duration += 1
            
            regime_history.append(regime)
            
            # Rebalancing
            if date in rebalance_dates and i > 60:
                # Get historical data up to now
                history = prices.iloc[:i]
                
                # Aggregate to asset classes
                asset_history = self.quant_agent.aggregate_assets(history, self.config.market_data.sectors)
                
                # 1. Get Quant Signal (VECM)
                exp_ret, cov, rank, is_coint = self.quant_agent.calculate_signal(asset_history)
                
                if exp_ret is not None and cov is not None:
                    # 2. Get Portfolio Weights (Black-Litterman + MVO)
                    # Use regime to adjust risk aversion
                    risk_aversion = 2.0 + regime * 1.0
                        
                    # Pass current_weights for turnover control (if available)
                    # Note: PortfolioAgent needs to be updated to accept current_weights
                    new_weights = self.portfolio_agent.calculate_weights(
                        exp_ret, cov, risk_aversion, current_weights=current_weights
                    )
                    
                    # Convert to dict
                    new_weights = new_weights.to_dict()
                    
                    # 3. Apply Equity Floors (Regime-based)
                    # Bull/Strong Bull (0,1): 90%
                    # Neutral (2): 65%
                    # Bear/Strong Bear (3,4): 45%
                    
                    min_equity = 0.45
                    if regime <= 1:
                        min_equity = 0.90
                    elif regime == 2:
                        min_equity = 0.65
                        
                    # Calculate current equity allocation (sum of all weights)
                    # Assuming all assets are "equity" or risk assets for now. 
                    # If we had explicit Cash asset, we would exclude it.
                    total_equity = sum(new_weights.values())
                    
                    if total_equity < min_equity:
                        # Scale up to meet floor
                        scale_factor = min_equity / total_equity
                        for asset in new_weights:
                            new_weights[asset] *= scale_factor
                    
                    # 4. Dynamic Beta Hedging (Reduced impact due to floors)
                    # Only hedge if volatility is EXTREME, to avoid fighting the floor
                    if 'SPY' in returns.columns:
                        mkt_vol = returns['SPY'].iloc[i-21:i].std() * np.sqrt(252)
                    else:
                        mkt_vol = returns.mean(axis=1).iloc[i-21:i].std() * np.sqrt(252)
                    
                    if mkt_vol > 0.35 or regime == 4: # Only in extreme stress
                        hedge_ratio = 0.3
                        net_exposure = 1.0 - hedge_ratio
                        for asset in new_weights:
                            new_weights[asset] *= net_exposure

                else:
                    new_weights = None
                    print(f"DEBUG: {date} - No signal. Assets: {len(asset_history) if 'asset_history' in locals() else 0}")
                
                if new_weights is not None:
                    # Record trade
                    if current_weights is not None:
                        turnover = sum(abs(new_weights.get(a, 0) - current_weights.get(a, 0)) 
                                      for a in set(new_weights.keys()) | set(current_weights.keys()))
                    else:
                        turnover = 1.0
                    
                    trades.append({
                        'date': date,
                        'turnover': turnover,
                        'regime': regime,
                        'n_positions': len(new_weights)
                    })
                    
                    current_weights = new_weights
                else:
                    # Keep previous weights if signal failed, or stay empty
                    pass
        
        # Create result
        result = BacktestResult()
        result.returns = pd.Series(all_returns, index=returns.index)
        result.cumulative_returns = (1 + result.returns).cumprod()
        result.portfolio_values = pd.Series(portfolio_values, index=returns.index)
        result.weights_history = pd.DataFrame(weights_history, index=returns.index)
        result.regime_history = pd.Series(regime_history, index=returns.index)
        result.trades = trades
        result.benchmark_returns = benchmark_returns
        result.benchmark_cumulative = (1 + benchmark_returns).cumprod()
        
        # Calculate metrics
        result.metrics = self._calculate_metrics(result)
        
        logger.info("Backtest complete!")
        
        return result
    
    def _get_rebalance_dates(self, dates: pd.DatetimeIndex, freq: str) -> set:
        """Get rebalancing dates."""
        if freq == 'D':
            return set(dates)
        elif freq == 'W':
            return set(dates[dates.dayofweek == 4])  # Fridays
        elif freq == 'M':
            # Last trading day of each month
            monthly = dates.to_series().groupby(dates.to_period('M')).last()
            return set(monthly.values)
        else:
            return set(dates[dates.dayofweek == 4])
    
    def _calculate_weights(self, 
                          returns: pd.DataFrame,
                          prices: pd.DataFrame,
                          regime: int) -> Dict[str, float]:
        """Calculate portfolio weights."""
        # Get stock columns only
        stock_cols = []
        for sector_tickers in self.config.market_data.sectors.values():
            for t in sector_tickers:
                if t in returns.columns:
                    stock_cols.append(t)
        
        if len(stock_cols) < 5:
            return None
        
        stock_returns = returns[stock_cols].dropna(axis=1)
        
        # Simple mean-variance with regime adjustment
        lookback = min(252, len(stock_returns))
        recent_returns = stock_returns.tail(lookback)
        
        # Expected returns (regime adjusted)
        exp_ret = recent_returns.mean() * 252
        
        regime_mult = {0: 1.0, 1: 0.8, 2: 0.5}
        exp_ret = exp_ret * regime_mult.get(regime, 0.8)
        
        # Covariance
        cov = recent_returns.cov() * 252
        
        # Simple inverse volatility weighting
        vols = np.sqrt(np.diag(cov.values))
        inv_vol = 1.0 / (vols + 0.01)
        weights = inv_vol / inv_vol.sum()
        
        # Apply max position constraint
        max_weight = self.config.portfolio.max_weight
        weights = np.minimum(weights, max_weight)
        weights = weights / weights.sum()
        
        return dict(zip(stock_returns.columns, weights))
    
    def _calculate_metrics(self, result: BacktestResult) -> Dict:
        """Calculate performance metrics."""
        returns = result.returns.dropna()
        
        if len(returns) < 20:
            return {}
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        n_years = len(returns) / 252
        cagr = (1 + total_return) ** (1/n_years) - 1 if n_years > 0 else 0
        volatility = returns.std() * np.sqrt(252)
        sharpe = (cagr - self.config.portfolio.risk_free_rate) / volatility if volatility > 0 else 0
        
        # Downside metrics
        neg_returns = returns[returns < 0]
        downside_dev = neg_returns.std() * np.sqrt(252) if len(neg_returns) > 0 else volatility
        sortino = (cagr - self.config.portfolio.risk_free_rate) / downside_dev if downside_dev > 0 else 0
        
        # Drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = cumulative / rolling_max - 1
        max_dd = drawdown.min()
        
        # Calmar
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        
        # VaR and CVaR
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
        
        # Win rate
        win_rate = (returns > 0).mean()
        
        # Benchmark comparison
        if result.benchmark_returns is not None:
            bench = result.benchmark_returns.loc[returns.index].dropna()
            if len(bench) > 20:
                bench_total = (1 + bench).prod() - 1
                bench_cagr = (1 + bench_total) ** (1/n_years) - 1 if n_years > 0 else 0
                bench_vol = bench.std() * np.sqrt(252)
                bench_sharpe = (bench_cagr - self.config.portfolio.risk_free_rate) / bench_vol if bench_vol > 0 else 0
                
                # Alpha/Beta
                if len(bench) == len(returns):
                    cov = np.cov(returns, bench)[0, 1]
                    var_bench = bench.var()
                    beta = cov / var_bench if var_bench > 0 else 1
                    alpha = cagr - (self.config.portfolio.risk_free_rate + beta * (bench_cagr - self.config.portfolio.risk_free_rate))
                else:
                    beta = 1.0
                    alpha = cagr - bench_cagr
            else:
                bench_cagr = 0
                bench_sharpe = 0
                beta = 1.0
                alpha = 0
        else:
            bench_cagr = 0
            bench_sharpe = 0
            beta = 1.0
            alpha = 0
        
        return {
            'Total Return': total_return,
            'CAGR': cagr,
            'Volatility': volatility,
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'Max Drawdown': max_dd,
            'Calmar Ratio': calmar,
            'VaR (95%)': var_95,
            'CVaR (95%)': cvar_95,
            'Win Rate': win_rate,
            'Beta': beta,
            'Alpha': alpha,
            'Benchmark Return': bench_cagr,
            'Benchmark Sharpe': bench_sharpe,
            'N Trades': len(result.trades)
        }
    
    def print_results(self, result: BacktestResult) -> None:
        """Print formatted results."""
        print("\n" + "=" * 70)
        print("MULTI-AGENT PORTFOLIO SYSTEM - BACKTEST RESULTS")
        print("=" * 70)
        
        m = result.metrics
        
        print("\n[PERFORMANCE SUMMARY]")
        print("-" * 40)
        print(f"  Total Return:      {m.get('Total Return', 0)*100:>10.2f}%")
        print(f"  CAGR:              {m.get('CAGR', 0)*100:>10.2f}%")
        print(f"  Volatility:        {m.get('Volatility', 0)*100:>10.2f}%")
        print(f"  Sharpe Ratio:      {m.get('Sharpe Ratio', 0):>10.2f}")
        print(f"  Sortino Ratio:     {m.get('Sortino Ratio', 0):>10.2f}")
        print(f"  Max Drawdown:      {m.get('Max Drawdown', 0)*100:>10.2f}%")
        print(f"  Calmar Ratio:      {m.get('Calmar Ratio', 0):>10.2f}")
        
        print("\n[RISK METRICS]")
        print("-" * 40)
        print(f"  VaR (95%):         {m.get('VaR (95%)', 0)*100:>10.2f}%")
        print(f"  CVaR (95%):        {m.get('CVaR (95%)', 0)*100:>10.2f}%")
        print(f"  Win Rate:          {m.get('Win Rate', 0)*100:>10.1f}%")
        print(f"  Beta:              {m.get('Beta', 0):>10.2f}")
        print(f"  Alpha (annual):    {m.get('Alpha', 0)*100:>10.2f}%")
        
        print("\n[BENCHMARK COMPARISON]")
        print("-" * 40)
        print(f"  Benchmark Return:  {m.get('Benchmark Return', 0)*100:>10.2f}%")
        print(f"  Benchmark Sharpe:  {m.get('Benchmark Sharpe', 0):>10.2f}")
        print(f"  Number of Trades:  {m.get('N Trades', 0):>10d}")


def main():
    """Main entry point."""
    # Load config
    config = SystemConfig()
    
    print("=" * 60)
    print("MULTI-AGENT PORTFOLIO SYSTEM")
    print("=" * 60)
    print(f"\nUniverse: {config.n_assets} assets")
    print(f"Sectors: {list(config.market_data.sectors.keys())}")
    for sector, tickers in config.market_data.sectors.items():
        print(f"  {sector}: {len(tickers)} stocks")
    print(f"Bonds: {len(config.market_data.bonds)}")
    print(f"Commodities: {len(config.market_data.commodities)}")
    
    # Create orchestrator
    orchestrator = Orchestrator(config)
    
    # Run backtest
    print("\nRunning backtest...")
    result = orchestrator.run_backtest(
        start_date=config.backtest.start_date,
        end_date=config.backtest.end_date,
        rebalance_freq=config.backtest.rebalance_freq
    )
    
    # Print results
    orchestrator.print_results(result)
    
    # Save results
    os.makedirs('./results', exist_ok=True)
    result.returns.to_csv('./results/returns.csv')
    result.portfolio_values.to_csv('./results/portfolio_values.csv')
    
    if not result.weights_history.empty:
        result.weights_history.to_csv('./results/weights_history.csv')
    
    pd.DataFrame([result.metrics]).T.to_csv('./results/metrics.csv')
    
    print("\nResults saved to ./results/")
    
    return orchestrator, result


if __name__ == "__main__":
    orchestrator, result = main()
