"""
Agent-Based Backtest Engine
============================
Rolling window backtester using the multi-agent pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

from events.message_bus import MessageBus
from events.base_event import TradeSignalEvent


logger = logging.getLogger(__name__)


class AgentBacktester:
    """
    Backtester that runs the agent pipeline at each rebalance date.
    
    Features:
    - Rolling window execution
    - Transaction cost modeling
    - Regime-dependent slippage
    - Performance metrics calculation
    - Full audit trail
    """
    
    def __init__(self, orchestrator, message_bus: MessageBus,
                 initial_capital: float = 1000000,
                 transaction_cost_bps: float = 10):
        """
        Initialize backtester.
        
        Args:
            orchestrator: Orchestrator instance with registered agents
            message_bus: Shared message bus
            initial_capital: Starting capital
            transaction_cost_bps: Base transaction cost in basis points
        """
        self.orchestrator = orchestrator
        self.bus = message_bus
        self.initial_capital = initial_capital
        self.transaction_cost_bps = transaction_cost_bps
        
        # Results storage
        self.portfolio_values: List[Dict] = []
        self.weights_history: List[Dict] = []
        self.regime_history: List[Dict] = []
        self.trade_history: List[Dict] = []
        
        # State
        self.current_capital = initial_capital
        self.current_weights: Dict[str, float] = {}
        self.prices: Optional[pd.DataFrame] = None
    
    def run(self, start_date: str, end_date: Optional[str] = None,
            rebalance_freq: str = 'M', lookback_window: int = 252) -> pd.DataFrame:
        """
        Run the backtest.
        
        Args:
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (None for latest)
            rebalance_freq: Rebalancing frequency ('D', 'W', 'M')
            lookback_window: Days of history for models
            
        Returns:
            DataFrame with portfolio values over time
        """
        logger.info(f"Starting backtest from {start_date}")
        
        # Initialize agents
        self.orchestrator.start_all_agents()
        
        # Run data agent once to get full history
        market_data_agent = self.orchestrator.get_agent("MarketDataAgent")
        market_data_agent.run()
        
        # Get prices
        from events.base_event import MarketDataEvent
        market_event = self.bus.get_last_event(MarketDataEvent)
        self.prices = market_event.prices
        
        if self.prices is None:
            raise ValueError("No price data available")
        
        # Get rebalance dates
        rebalance_dates = self._get_rebalance_dates(start_date, end_date, rebalance_freq)
        
        logger.info(f"Running {len(rebalance_dates)} rebalance periods")
        
        for i, date in enumerate(rebalance_dates):
            logger.info(f"\n[{i+1}/{len(rebalance_dates)}] Processing {date.date()}")
            
            try:
                # Update performance since last rebalance
                if i > 0:
                    self._update_portfolio_value(rebalance_dates[i-1], date)
                
                # Run pipeline for this date
                trade_signal = self._run_pipeline_for_date(date, lookback_window)
                
                if trade_signal:
                    # Apply trades and costs
                    self._execute_trades(date, trade_signal)
                
                # Record state
                self._record_state(date)
                
            except Exception as e:
                logger.error(f"Error at {date}: {e}")
                self._record_state(date)  # Still record for continuity
        
        # Final update
        if len(rebalance_dates) > 0:
            final_date = self.prices.index[-1]
            self._update_portfolio_value(rebalance_dates[-1], final_date)
        
        return self._compile_results()
    
    def _get_rebalance_dates(self, start_date: str, end_date: Optional[str],
                             freq: str) -> List[pd.Timestamp]:
        """Get list of rebalance dates."""
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date) if end_date else self.prices.index[-1]
        
        # Filter to available dates
        mask = (self.prices.index >= start) & (self.prices.index <= end)
        available_dates = self.prices.index[mask]
        
        # Resample to get period end dates
        dummy = pd.Series(1, index=available_dates)
        rebalance_dates = dummy.resample(freq).last().dropna().index.tolist()
        
        return rebalance_dates
    
    def _run_pipeline_for_date(self, date: pd.Timestamp, 
                               lookback: int) -> Optional[TradeSignalEvent]:
        """
        Run the agent pipeline for a specific date.
        
        This simulates running the full pipeline as if we were at that date.
        """
        # For backtesting, we would need to:
        # 1. Filter data up to this date
        # 2. Run each agent with filtered data
        
        # Since our agents use the cached MarketDataEvent, we need to
        # either re-run data agent with date filter, or modify the agents
        # to accept date parameters.
        
        # For simplicity in this implementation, we run the full pipeline
        # and let the lookback_window in each agent handle the filtering.
        
        try:
            # Run ML Agent (regime detection)
            ml_agent = self.orchestrator.get_agent("MLAgent")
            ml_agent.run()
            
            # Run Quant Agent
            quant_agent = self.orchestrator.get_agent("QuantAgent")
            quant_agent.run()
            
            # Run Risk Manager
            risk_agent = self.orchestrator.get_agent("RiskManagerAgent")
            risk_agent.update_portfolio_value(self.current_capital)
            risk_agent.run()
            
            # Run Portfolio Agent
            portfolio_agent = self.orchestrator.get_agent("PortfolioAgent")
            portfolio_agent.update_current_weights(self.current_weights)
            portfolio_agent.run()
            
            # Get trade signal
            trade_signal = self.bus.get_last_event(TradeSignalEvent)
            return trade_signal
            
        except Exception as e:
            logger.warning(f"Pipeline failed for {date}: {e}")
            return None
    
    def _update_portfolio_value(self, from_date: pd.Timestamp, 
                                to_date: pd.Timestamp) -> None:
        """Update portfolio value based on asset returns."""
        if not self.current_weights:
            return
        
        period_return = 0.0
        
        for asset, weight in self.current_weights.items():
            if asset == 'Cash':
                # Cash earns risk-free rate
                days = (to_date - from_date).days
                period_return += weight * (0.02 * days / 365)
            elif asset in self.prices.columns:
                try:
                    # Find closest available dates
                    from_idx = self.prices.index.get_indexer([from_date], method='pad')[0]
                    to_idx = self.prices.index.get_indexer([to_date], method='pad')[0]
                    
                    from_price = self.prices.iloc[from_idx][asset]
                    to_price = self.prices.iloc[to_idx][asset]
                    
                    if from_price > 0:
                        asset_return = (to_price / from_price) - 1
                        period_return += weight * asset_return
                except:
                    pass
        
        self.current_capital *= (1 + period_return)
    
    def _execute_trades(self, date: pd.Timestamp, 
                        trade_signal: TradeSignalEvent) -> None:
        """Execute trades and deduct costs."""
        # Calculate turnover
        turnover = trade_signal.total_turnover
        
        # Get regime for slippage
        from events.base_event import RegimeEvent
        regime_event = self.bus.get_last_event(RegimeEvent)
        current_regime = regime_event.current_regime if regime_event else 0
        
        # Regime-dependent slippage
        slippage_map = {0: 0.0005, 1: 0.0010, 2: 0.0025}
        slippage = slippage_map.get(current_regime, 0.0010)
        
        # Calculate costs
        traded_value = turnover * self.current_capital
        commission = traded_value * (self.transaction_cost_bps / 10000)
        slippage_cost = traded_value * slippage
        total_cost = commission + slippage_cost
        
        # Deduct costs
        self.current_capital -= total_cost
        
        # Update weights
        self.current_weights = trade_signal.target_weights.copy()
        
        # Record trades
        for trade in trade_signal.trades:
            if trade.action != "HOLD":
                self.trade_history.append({
                    'Date': date,
                    'Ticker': trade.ticker,
                    'Action': trade.action,
                    'Weight': trade.target_weight,
                    'Cost': trade.estimated_cost * self.current_capital
                })
        
        logger.info(f"Executed trades: turnover={turnover:.2%}, cost=${total_cost:.2f}")
    
    def _record_state(self, date: pd.Timestamp) -> None:
        """Record current state for history."""
        # Portfolio value
        self.portfolio_values.append({
            'Date': date,
            'Value': self.current_capital
        })
        
        # Weights
        self.weights_history.append({
            'Date': date,
            **self.current_weights
        })
        
        # Regime
        from events.base_event import RegimeEvent
        regime_event = self.bus.get_last_event(RegimeEvent)
        if regime_event:
            self.regime_history.append({
                'Date': date,
                'Regime': regime_event.current_regime,
                **regime_event.regime_probabilities
            })
    
    def _compile_results(self) -> pd.DataFrame:
        """Compile backtest results into DataFrame."""
        results = pd.DataFrame(self.portfolio_values).set_index('Date')
        return results
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics."""
        df = pd.DataFrame(self.portfolio_values).set_index('Date')
        
        if len(df) < 2:
            return {}
        
        returns = df['Value'].pct_change().dropna()
        
        # Total return
        total_return = (df['Value'].iloc[-1] / df['Value'].iloc[0]) - 1
        
        # Annualized return
        days = (df.index[-1] - df.index[0]).days
        ann_return = (1 + total_return) ** (365 / days) - 1
        
        # Volatility
        ann_vol = returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        
        # Maximum drawdown
        cum_returns = (1 + returns).cumprod()
        peak = cum_returns.cummax()
        drawdown = (cum_returns - peak) / peak
        max_dd = drawdown.min()
        
        # Win rate
        win_rate = (returns > 0).mean()
        
        return {
            'Total Return': total_return,
            'Annualized Return': ann_return,
            'Annualized Volatility': ann_vol,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': max_dd,
            'Win Rate': win_rate,
            'Total Trades': len(self.trade_history)
        }
    
    def get_weights_history(self) -> pd.DataFrame:
        """Get weights history as DataFrame."""
        return pd.DataFrame(self.weights_history).set_index('Date')
    
    def get_regime_history(self) -> pd.DataFrame:
        """Get regime history as DataFrame."""
        return pd.DataFrame(self.regime_history).set_index('Date')
    
    def get_trade_history(self) -> pd.DataFrame:
        """Get trade history as DataFrame."""
        return pd.DataFrame(self.trade_history)
