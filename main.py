"""
Multi-Agent Portfolio System - Main Orchestrator
=================================================

Coordinates all agents and modules:
1. Market Data Agent (with NLP sentiment)
2. ML Agent (Enhanced regime detection)
3. Quant Agent (Factor analysis)
4. Risk Agent (Regime-adaptive risk management)
5. Portfolio Agent (Optimization with TWAP/VWAP execution)
6. Institutional Flow Agent (Piggyback trading)

Author: Jaeha
Date: November 2025
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
import os

# Import our modules
from config.settings import (
    SystemConfig, RegimeConfig, RegimeType, 
    ExecutionConfig, InstitutionalFlowConfig, SentimentConfig,
    FuturesConfig, StatisticalConfig, load_config
)
from models.enhanced_regime_detector import (
    EnhancedRegimeDetector, RegimeState, RegimeAssessment
)
from models.regime_optimizer import (
    RegimeAdaptiveOptimizer, OptimizationObjective, OptimizationResult
)
from execution.execution_algos import (
    ExecutionEngine, ExecutionAlgo, OrderSide, ParentOrder
)
from agents.institutional_flow_agent import (
    InstitutionalFlowDetector, PiggybackTrader, FlowType, InstitutionalSignal
)
from nlp.sentiment_analyzer import (
    MarketSentimentAgent, SentimentScore, NuanceSentimentAnalyzer
)
from analysis.statistical_analysis import (
    StatisticalValidator, DeflatedSharpeRatio, BlockBootstrap, OLSRegression
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EventType(Enum):
    """Event types for inter-agent communication"""
    MARKET_DATA_UPDATE = "market_data_update"
    REGIME_CHANGE = "regime_change"
    SENTIMENT_UPDATE = "sentiment_update"
    FLOW_SIGNAL = "flow_signal"
    RISK_ALERT = "risk_alert"
    REBALANCE_TRIGGER = "rebalance_trigger"
    ORDER_EXECUTION = "order_execution"
    PIGGYBACK_SIGNAL = "piggyback_signal"


@dataclass
class Event:
    """Event for message passing between agents"""
    event_type: EventType
    timestamp: datetime
    source: str
    data: Dict
    priority: int = 0


@dataclass
class PortfolioState:
    """Current portfolio state"""
    timestamp: datetime
    weights: Dict[str, float]
    positions: Dict[str, int]
    cash: float
    total_value: float
    regime: RegimeState
    risk_level: float
    exposure: float


@dataclass
class BacktestResult:
    """Backtest results container"""
    returns: pd.Series
    cumulative_returns: pd.Series
    portfolio_values: pd.Series
    weights_history: pd.DataFrame
    regime_history: pd.Series
    trades: List[Dict]
    metrics: Dict
    statistical_validation: Dict


class MarketDataAgent:
    """
    Agent responsible for market data collection and preprocessing.
    Includes NLP sentiment analysis.
    """
    
    def __init__(self, tickers: List[str], config: SentimentConfig):
        self.tickers = tickers
        self.config = config
        self.sentiment_agent = MarketSentimentAgent(tickers, config.__dict__)
        self.price_data: Dict[str, pd.DataFrame] = {}
        self.volume_data: Dict[str, pd.Series] = {}
        
    def update(self, prices: pd.DataFrame, volumes: pd.DataFrame) -> Event:
        """Update market data and compute features"""
        
        # Store data
        for ticker in self.tickers:
            if ticker in prices.columns:
                self.price_data[ticker] = prices[ticker]
            if ticker in volumes.columns:
                self.volume_data[ticker] = volumes[ticker]
        
        # Compute returns and volatility
        returns = prices.pct_change().dropna()
        volatility = returns.rolling(21).std() * np.sqrt(252)
        
        # Get sentiment (if enabled)
        sentiment_scores = {}
        if self.config.enabled:
            for ticker in self.tickers:
                try:
                    score = self.sentiment_agent.get_sentiment_signal(ticker)
                    sentiment_scores[ticker] = score
                except:
                    sentiment_scores[ticker] = 0.0
        
        return Event(
            event_type=EventType.MARKET_DATA_UPDATE,
            timestamp=datetime.now(),
            source="MarketDataAgent",
            data={
                'prices': prices,
                'returns': returns,
                'volatility': volatility,
                'volumes': volumes,
                'sentiment': sentiment_scores
            }
        )


class MLAgent:
    """
    Machine Learning Agent for regime detection.
    Integrates HMM, futures positioning, and investor flows.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.regime_detector = EnhancedRegimeDetector(
            n_regimes=config.get('n_regimes', 5),
            config=config
        )
        self.current_regime: Optional[RegimeAssessment] = None
        self.regime_history: List[RegimeAssessment] = []
        
    def fit(self, returns: pd.Series, volatility: pd.Series):
        """Fit the regime detection model"""
        self.regime_detector.fit_hmm(returns, volatility)
        
    def update(self, 
               current_return: float,
               current_volatility: float,
               futures_data: Optional[Dict] = None,
               flow_data: Optional[Dict] = None,
               sentiment_score: float = 0.0) -> Event:
        """Update regime assessment"""
        
        assessment = self.regime_detector.detect_regime(
            current_return=current_return,
            current_volatility=current_volatility,
            futures_data=futures_data,
            flow_data=flow_data,
            sentiment_score=sentiment_score
        )
        
        # Check for regime change
        regime_changed = (
            self.current_regime is None or 
            self.current_regime.primary_regime != assessment.primary_regime
        )
        
        self.current_regime = assessment
        self.regime_history.append(assessment)
        
        event_type = EventType.REGIME_CHANGE if regime_changed else EventType.MARKET_DATA_UPDATE
        
        return Event(
            event_type=event_type,
            timestamp=datetime.now(),
            source="MLAgent",
            data={
                'regime': assessment.primary_regime,
                'probabilities': assessment.regime_probabilities,
                'risk_level': assessment.risk_level,
                'recommended_exposure': assessment.recommended_exposure,
                'combined_score': assessment.combined_score,
                'regime_changed': regime_changed
            },
            priority=1 if regime_changed else 0
        )


class QuantAgent:
    """
    Quantitative Analysis Agent.
    Computes expected returns, covariance, and factor exposures.
    """
    
    def __init__(self, tickers: List[str], lookback_days: int = 252):
        self.tickers = tickers
        self.lookback_days = lookback_days
        self.expected_returns: Optional[np.ndarray] = None
        self.covariance_matrix: Optional[np.ndarray] = None
        
    def update(self, 
               returns: pd.DataFrame,
               regime: RegimeState) -> Dict:
        """Update return and risk estimates"""
        
        # Use regime-specific lookback
        if regime in [RegimeState.CRISIS, RegimeState.BEAR_HIGH_VOL]:
            lookback = min(63, len(returns))  # Shorter lookback in volatile periods
        else:
            lookback = min(self.lookback_days, len(returns))
        
        recent_returns = returns.iloc[-lookback:]
        
        # Expected returns (shrinkage toward mean)
        raw_mu = recent_returns.mean().values * 252
        grand_mean = raw_mu.mean()
        shrinkage = 0.5  # Shrink 50% toward grand mean
        self.expected_returns = shrinkage * grand_mean + (1 - shrinkage) * raw_mu
        
        # Covariance (Ledoit-Wolf shrinkage)
        self.covariance_matrix = self._shrunk_covariance(recent_returns) * 252
        
        return {
            'expected_returns': self.expected_returns,
            'covariance_matrix': self.covariance_matrix,
            'lookback_used': lookback
        }
    
    def _shrunk_covariance(self, returns: pd.DataFrame) -> np.ndarray:
        """Ledoit-Wolf shrinkage estimator"""
        n, p = returns.shape
        
        # Sample covariance
        sample_cov = returns.cov().values
        
        # Shrinkage target (diagonal)
        mu = np.trace(sample_cov) / p
        target = mu * np.eye(p)
        
        # Optimal shrinkage intensity (simplified)
        delta = sample_cov - target
        shrinkage_intensity = min(1, max(0, 
            (np.sum(delta ** 2) / n) / (np.sum((sample_cov - target) ** 2) + 1e-10)
        ))
        
        # Shrunk covariance
        shrunk_cov = shrinkage_intensity * target + (1 - shrinkage_intensity) * sample_cov
        
        return shrunk_cov


class RiskAgent:
    """
    Risk Management Agent.
    Monitors portfolio risk and enforces constraints.
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.current_drawdown: float = 0.0
        self.peak_value: float = 0.0
        self.risk_alerts: List[Dict] = []
        
    def update(self, 
               portfolio_value: float,
               weights: np.ndarray,
               covariance: np.ndarray,
               regime: RegimeState) -> Event:
        """Update risk monitoring"""
        
        # Update drawdown
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
        self.current_drawdown = (portfolio_value / self.peak_value) - 1
        
        # Portfolio volatility
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(covariance, weights)))
        
        # VaR estimate (parametric)
        var_95 = -1.645 * portfolio_vol / np.sqrt(252)  # Daily VaR
        
        # Risk alerts
        alerts = []
        
        if self.current_drawdown < -self.config.max_drawdown_limit:
            alerts.append({
                'type': 'MAX_DRAWDOWN_BREACH',
                'severity': 'HIGH',
                'message': f'Drawdown {self.current_drawdown:.1%} exceeds limit'
            })
        
        if var_95 < -self.config.var_limit_95:
            alerts.append({
                'type': 'VAR_LIMIT_BREACH',
                'severity': 'MEDIUM',
                'message': f'VaR {var_95:.2%} exceeds limit'
            })
        
        # Get regime-specific risk budget
        regime_config = self.config.regime_configs.get(
            RegimeType(regime.value), 
            self.config.regime_configs[RegimeType.NORMAL]
        )
        
        self.risk_alerts.extend(alerts)
        
        return Event(
            event_type=EventType.RISK_ALERT if alerts else EventType.MARKET_DATA_UPDATE,
            timestamp=datetime.now(),
            source="RiskAgent",
            data={
                'drawdown': self.current_drawdown,
                'portfolio_vol': portfolio_vol,
                'var_95': var_95,
                'alerts': alerts,
                'risk_aversion': regime_config.risk_aversion,
                'risk_multiplier': regime_config.risk_multiplier,
                'max_position': regime_config.max_position
            },
            priority=2 if alerts else 0
        )


class PortfolioAgent:
    """
    Portfolio Construction Agent.
    Optimizes and executes portfolio allocations.
    """
    
    def __init__(self, 
                 tickers: List[str],
                 sector_mapping: Dict[str, str],
                 config: SystemConfig):
        self.tickers = tickers
        self.config = config
        self.optimizer = RegimeAdaptiveOptimizer(
            tickers=tickers,
            sector_mapping=sector_mapping,
            risk_free_rate=config.risk_free_rate
        )
        self.execution_engine = ExecutionEngine(config.execution.__dict__)
        self.current_weights = np.ones(len(tickers)) / len(tickers)
        self.target_weights: Optional[np.ndarray] = None
        
    def optimize(self,
                regime: int,
                expected_returns: np.ndarray,
                covariance: np.ndarray,
                risk_constraints: Dict) -> OptimizationResult:
        """Run portfolio optimization"""
        
        from models.regime_optimizer import PortfolioConstraints
        
        constraints = PortfolioConstraints(
            min_weight=0.0,
            max_weight=risk_constraints.get('max_position', 0.30),
            max_turnover=self.config.max_turnover,
            max_sector_weight={s: self.config.max_sector_exposure 
                             for s in set(self.optimizer.sector_mapping.values())}
        )
        
        result = self.optimizer.optimize(
            regime=regime,
            expected_returns=expected_returns,
            covariance_matrix=covariance,
            constraints=constraints,
            objective=OptimizationObjective.MEAN_VARIANCE,
            custom_risk_aversion=risk_constraints.get('risk_aversion')
        )
        
        if result.success:
            self.target_weights = result.weights
        
        return result
    
    def execute_rebalance(self,
                         target_weights: np.ndarray,
                         current_prices: Dict[str, float],
                         portfolio_value: float,
                         algo: ExecutionAlgo = ExecutionAlgo.TWAP
                         ) -> List[ParentOrder]:
        """Execute rebalancing trades using TWAP/VWAP"""
        
        orders = []
        
        for i, ticker in enumerate(self.tickers):
            current_weight = self.current_weights[i]
            target_weight = target_weights[i]
            weight_diff = target_weight - current_weight
            
            if abs(weight_diff) < 0.01:  # Skip small changes
                continue
            
            price = current_prices.get(ticker, 100)
            dollar_amount = abs(weight_diff) * portfolio_value
            quantity = int(dollar_amount / price)
            
            if quantity < 1:
                continue
            
            side = OrderSide.BUY if weight_diff > 0 else OrderSide.SELL
            
            order = self.execution_engine.submit_order(
                ticker=ticker,
                side=side,
                quantity=quantity,
                algo=algo,
                duration_minutes=30
            )
            orders.append(order)
        
        return orders


class InstitutionalFlowAgentWrapper:
    """
    Wrapper for institutional flow detection and piggyback trading.
    """
    
    def __init__(self, config: InstitutionalFlowConfig):
        self.config = config
        self.flow_detector = InstitutionalFlowDetector(config.__dict__)
        self.piggyback_trader = PiggybackTrader(config.__dict__)
        
    def analyze(self,
               ticker: str,
               prices: pd.DataFrame,
               volumes: pd.Series) -> Tuple[InstitutionalSignal, Optional[Event]]:
        """Analyze institutional flow and generate signals"""
        
        signal = self.flow_detector.analyze_flow(ticker, prices, volumes)
        
        event = None
        if signal.strength > self.config.piggyback_entry_threshold:
            event = Event(
                event_type=EventType.PIGGYBACK_SIGNAL,
                timestamp=datetime.now(),
                source="InstitutionalFlowAgent",
                data={
                    'ticker': ticker,
                    'flow_type': signal.flow_type.value,
                    'strength': signal.strength,
                    'confidence': signal.confidence
                },
                priority=1
            )
        
        return signal, event


class MultiAgentOrchestrator:
    """
    Main orchestrator that coordinates all agents.
    """
    
    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()
        
        # Initialize agents
        self.market_agent = MarketDataAgent(
            self.config.universe, 
            self.config.sentiment
        )
        self.ml_agent = MLAgent({
            'n_regimes': self.config.n_regimes,
            'futures_weight': self.config.futures.positioning_weight,
            'flow_weight': 0.25
        })
        self.quant_agent = QuantAgent(
            self.config.universe,
            lookback_days=252
        )
        self.risk_agent = RiskAgent(self.config)
        self.portfolio_agent = PortfolioAgent(
            tickers=self.config.universe,
            sector_mapping=self._create_sector_mapping(),
            config=self.config
        )
        self.flow_agent = InstitutionalFlowAgentWrapper(self.config.institutional)
        
        # Statistical validator
        self.validator = StatisticalValidator(
            confidence_level=self.config.statistical.confidence_level
        )
        
        # State
        self.portfolio_value = self.config.initial_capital
        self.cash = self.config.initial_capital
        self.positions: Dict[str, int] = {}
        self.event_queue: List[Event] = []
        self.trade_history: List[Dict] = []
        
    def _create_sector_mapping(self) -> Dict[str, str]:
        """Create ticker to sector mapping"""
        sectors = {
            'Technology': ['AAPL', 'MSFT', 'NVDA', 'AMD', 'AVGO'],
            'Energy': ['XOM', 'CVX', 'SLB', 'MPC', 'PSX'],
            'Financials': ['JPM', 'BAC', 'GS', 'MS', 'WFC'],
            'Consumer': ['AMZN', 'TSLA', 'MCD', 'NKE', 'SBUX'],
            'Healthcare': ['JNJ', 'LLY', 'MRK', 'AMGN', 'TMO']
        }
        
        mapping = {}
        for sector, tickers in sectors.items():
            for ticker in tickers:
                mapping[ticker] = sector
        
        return mapping
    
    def run_backtest(self,
                    prices: pd.DataFrame,
                    volumes: pd.DataFrame,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> BacktestResult:
        """
        Run full backtest with all agents.
        """
        logger.info("Starting backtest...")
        
        # Prepare data
        if start_date:
            prices = prices[prices.index >= start_date]
            volumes = volumes[volumes.index >= start_date]
        if end_date:
            prices = prices[prices.index <= end_date]
            volumes = volumes[volumes.index <= end_date]
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        # Fit ML agent on initial data
        market_returns = returns.mean(axis=1)
        market_vol = market_returns.rolling(21).std()
        self.ml_agent.fit(market_returns.iloc[:252], market_vol.iloc[:252])
        
        # Initialize tracking
        portfolio_values = []
        weights_history = []
        regime_history = []
        all_returns = []
        
        # Initial weights
        n_assets = len(self.config.universe)
        current_weights = np.ones(n_assets) / n_assets
        self.portfolio_agent.current_weights = current_weights
        
        # Run backtest
        rebalance_dates = self._get_rebalance_dates(returns.index)
        
        for i, date in enumerate(returns.index[252:], start=252):
            
            # Get current data
            current_returns = returns.iloc[i]
            current_prices = prices.iloc[i]
            current_volumes = volumes.iloc[i] if i < len(volumes) else volumes.iloc[-1]
            
            # Market data update
            market_return = current_returns.mean()
            market_vol_current = market_returns.iloc[i-21:i].std() if i >= 21 else 0.02
            
            # ML Agent: Regime detection
            regime_event = self.ml_agent.update(
                current_return=market_return,
                current_volatility=market_vol_current,
                sentiment_score=0.0  # Would be from sentiment agent
            )
            
            regime = regime_event.data['regime']
            regime_history.append(regime.value)
            
            # Calculate portfolio return
            portfolio_return = np.dot(current_weights, current_returns.values)
            all_returns.append(portfolio_return)
            
            # Update portfolio value
            self.portfolio_value *= (1 + portfolio_return)
            portfolio_values.append(self.portfolio_value)
            weights_history.append(current_weights.copy())
            
            # Check for rebalancing
            if date in rebalance_dates:
                # Quant Agent: Update estimates
                quant_data = self.quant_agent.update(
                    returns.iloc[i-252:i],
                    regime
                )
                
                # Risk Agent: Get constraints
                risk_event = self.risk_agent.update(
                    portfolio_value=self.portfolio_value,
                    weights=current_weights,
                    covariance=quant_data['covariance_matrix'],
                    regime=regime
                )
                
                # Portfolio Agent: Optimize
                opt_result = self.portfolio_agent.optimize(
                    regime=regime.value,
                    expected_returns=quant_data['expected_returns'],
                    covariance=quant_data['covariance_matrix'],
                    risk_constraints=risk_event.data
                )
                
                if opt_result.success:
                    # Record trade
                    turnover = np.sum(np.abs(opt_result.weights - current_weights))
                    if turnover > 0.02:  # Only trade if meaningful
                        self.trade_history.append({
                            'date': date,
                            'old_weights': current_weights.copy(),
                            'new_weights': opt_result.weights.copy(),
                            'turnover': turnover,
                            'regime': regime.name
                        })
                        current_weights = opt_result.weights
                        self.portfolio_agent.current_weights = current_weights
        
        # Create results
        returns_series = pd.Series(all_returns, index=returns.index[252:])
        cumulative = (1 + returns_series).cumprod()
        
        # Statistical validation
        validation = self.validator.full_validation(
            returns_series.values,
            benchmark_returns=market_returns.iloc[252:].values,
            n_trials=10
        )
        
        # Calculate metrics
        metrics = self._calculate_metrics(returns_series, market_returns.iloc[252:])
        
        logger.info("Backtest complete!")
        
        return BacktestResult(
            returns=returns_series,
            cumulative_returns=cumulative,
            portfolio_values=pd.Series(portfolio_values, index=returns.index[252:]),
            weights_history=pd.DataFrame(weights_history, 
                                        index=returns.index[252:],
                                        columns=self.config.universe),
            regime_history=pd.Series(regime_history, index=returns.index[252:]),
            trades=self.trade_history,
            metrics=metrics,
            statistical_validation=validation
        )
    
    def _get_rebalance_dates(self, dates: pd.DatetimeIndex) -> set:
        """Get rebalancing dates based on frequency"""
        if self.config.rebalance_frequency == 'daily':
            return set(dates)
        elif self.config.rebalance_frequency == 'weekly':
            return set(dates[dates.dayofweek == 4])  # Fridays
        elif self.config.rebalance_frequency == 'monthly':
            return set(dates.to_series().groupby(dates.to_period('M')).last())
        else:
            return set(dates[dates.dayofweek == 4])
    
    def _calculate_metrics(self, 
                          returns: pd.Series, 
                          benchmark: pd.Series) -> Dict:
        """Calculate performance metrics"""
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        n_years = len(returns) / 252
        cagr = (1 + total_return) ** (1/n_years) - 1 if n_years > 0 else 0
        volatility = returns.std() * np.sqrt(252)
        sharpe = (cagr - self.config.risk_free_rate) / volatility if volatility > 0 else 0
        
        # Downside metrics
        neg_returns = returns[returns < 0]
        downside_dev = neg_returns.std() * np.sqrt(252) if len(neg_returns) > 0 else volatility
        sortino = (cagr - self.config.risk_free_rate) / downside_dev if downside_dev > 0 else 0
        
        # Drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = cumulative / rolling_max - 1
        max_dd = drawdown.min()
        
        # Calmar
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        
        # VaR and CVaR
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        
        # Win rate
        win_rate = (returns > 0).mean()
        
        # Benchmark comparison
        bench_total = (1 + benchmark).prod() - 1
        bench_cagr = (1 + bench_total) ** (1/n_years) - 1 if n_years > 0 else 0
        bench_vol = benchmark.std() * np.sqrt(252)
        bench_sharpe = (bench_cagr - self.config.risk_free_rate) / bench_vol if bench_vol > 0 else 0
        
        # Alpha/Beta (CAPM)
        cov = np.cov(returns, benchmark)[0, 1]
        var_bench = benchmark.var()
        beta = cov / var_bench if var_bench > 0 else 1
        alpha = cagr - (self.config.risk_free_rate + beta * (bench_cagr - self.config.risk_free_rate))
        
        return {
            'total_return': total_return,
            'cagr': cagr,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd,
            'calmar_ratio': calmar,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'win_rate': win_rate,
            'beta': beta,
            'alpha': alpha,
            'benchmark_return': bench_cagr,
            'benchmark_sharpe': bench_sharpe,
            'information_ratio': (cagr - bench_cagr) / (returns - benchmark).std() / np.sqrt(252) if (returns - benchmark).std() > 0 else 0,
            'n_trades': len(self.trade_history)
        }
    
    def print_results(self, result: BacktestResult):
        """Print formatted backtest results"""
        
        print("\n" + "=" * 70)
        print("MULTI-AGENT PORTFOLIO SYSTEM - BACKTEST RESULTS")
        print("=" * 70)
        
        m = result.metrics
        
        print("\n[PERFORMANCE SUMMARY]")
        print("-" * 40)
        print(f"  Total Return:      {m['total_return']*100:>10.2f}%")
        print(f"  CAGR:              {m['cagr']*100:>10.2f}%")
        print(f"  Volatility:        {m['volatility']*100:>10.2f}%")
        print(f"  Sharpe Ratio:      {m['sharpe_ratio']:>10.2f}")
        print(f"  Sortino Ratio:     {m['sortino_ratio']:>10.2f}")
        print(f"  Max Drawdown:      {m['max_drawdown']*100:>10.2f}%")
        print(f"  Calmar Ratio:      {m['calmar_ratio']:>10.2f}")
        
        print("\n[RISK METRICS]")
        print("-" * 40)
        print(f"  VaR (95%):         {m['var_95']*100:>10.2f}%")
        print(f"  CVaR (95%):        {m['cvar_95']*100:>10.2f}%")
        print(f"  Win Rate:          {m['win_rate']*100:>10.1f}%")
        print(f"  Beta:              {m['beta']:>10.2f}")
        print(f"  Alpha (annual):    {m['alpha']*100:>10.2f}%")
        
        print("\n[BENCHMARK COMPARISON]")
        print("-" * 40)
        print(f"  Benchmark Return:  {m['benchmark_return']*100:>10.2f}%")
        print(f"  Benchmark Sharpe:  {m['benchmark_sharpe']:>10.2f}")
        print(f"  Information Ratio: {m['information_ratio']:>10.2f}")
        
        print("\n[TRADING ACTIVITY]")
        print("-" * 40)
        print(f"  Number of Trades:  {m['n_trades']:>10d}")
        
        # Statistical validation
        if result.statistical_validation:
            v = result.statistical_validation
            print("\n[STATISTICAL VALIDATION]")
            print("-" * 40)
            
            if 'dsr' in v:
                dsr = v['dsr']
                sig = "***" if dsr['p_value'] < 0.01 else "**" if dsr['p_value'] < 0.05 else "*" if dsr['p_value'] < 0.10 else ""
                print(f"  t-statistic:       {dsr['t_statistic']:>10.2f}")
                print(f"  p-value:           {dsr['p_value']:>10.4f} {sig}")
                print(f"  PSR:               {dsr['psr']*100:>10.1f}%")
                print(f"  DSR:               {dsr['dsr']*100:>10.1f}%")
            
            if 'bootstrap' in v:
                boot = v['bootstrap']
                print(f"  Bootstrap 95% CI:  [{boot['ci_lower']:.2f}, {boot['ci_upper']:.2f}]")
                print(f"  Significant:       {'Yes ✓' if boot['significant'] else 'No'}")


    def run_live_dashboard(self, refresh_rate: int = 5):
        """
        Run live dashboard and trading loop.
        """
        import time
        import random
        
        print("\n" + "=" * 70)
        print("MULTI-AGENT PORTFOLIO SYSTEM - LIVE DASHBOARD")
        print("=" * 70)
        print("Press Ctrl+C to stop")
        
        # Initial setup
        self.portfolio_agent.current_weights = np.ones(len(self.config.universe)) / len(self.config.universe)
        
        try:
            while True:
                # Clear screen (simulated with newlines for now to be safe across terminals)
                # os.system('cls' if os.name == 'nt' else 'clear') 
                print("\n" + "-" * 70)
                print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("-" * 70)
                
                # Simulate market data update (replace with real feed in production)
                # Here we just generate a random return for demonstration
                market_return = np.random.normal(0.0005, 0.01)
                market_vol = 0.15 + np.random.normal(0, 0.02)
                
                # Update ML Agent
                regime_event = self.ml_agent.update(
                    current_return=market_return,
                    current_volatility=market_vol
                )
                
                regime = regime_event.data['regime']
                probs = regime_event.data['probabilities']
                
                # Update Portfolio Value (simulated)
                self.portfolio_value *= (1 + market_return)
                
                # Display Status
                print(f"Portfolio Value: ${self.portfolio_value:,.2f}")
                print(f"Current Regime:  {regime.name}")
                print(f"Regime Probs:    Bull={probs.get('bull', 0):.2f}, Bear={probs.get('bear', 0):.2f}, Crisis={probs.get('crisis', 0):.2f}")
                print(f"Risk Level:      {regime_event.data['risk_level']:.2f}")
                
                # Risk Alerts
                risk_event = self.risk_agent.update(
                    portfolio_value=self.portfolio_value,
                    weights=self.portfolio_agent.current_weights,
                    covariance=np.eye(len(self.config.universe)), # Dummy cov
                    regime=regime
                )
                
                if risk_event.event_type == EventType.RISK_ALERT:
                    print("\n[RISK ALERTS]")
                    for alert in risk_event.data['alerts']:
                        print(f"  ! {alert['severity']}: {alert['message']}")
                
                # Top Positions
                print("\n[TOP POSITIONS]")
                weights = self.portfolio_agent.current_weights
                sorted_idx = np.argsort(weights)[::-1][:5]
                for idx in sorted_idx:
                    ticker = self.config.universe[idx]
                    weight = weights[idx]
                    print(f"  {ticker:<6} {weight:>6.1%}")
                
                time.sleep(refresh_rate)
                
        except KeyboardInterrupt:
            print("\nDashboard stopped by user.")


def main():
    """Main function to run the system"""
    
    print("Initializing Multi-Agent Portfolio System...")
    
    # Create orchestrator
    config = SystemConfig()
    orchestrator = MultiAgentOrchestrator(config)
    
    print(f"Universe: {len(config.universe)} assets")
    print(f"Initial Capital: ${config.initial_capital:,.0f}")
    print(f"Rebalance Frequency: {config.rebalance_frequency}")
    
    # Run live dashboard
    orchestrator.run_live_dashboard()
    
    return orchestrator


if __name__ == "__main__":
    orchestrator = main()
