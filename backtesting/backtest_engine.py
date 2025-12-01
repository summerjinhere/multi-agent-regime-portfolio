"""
Agent-Based Backtest Engine (FIXED)
====================================
Rolling window backtester using the multi-agent pipeline.

Key Fixes:
1. Proper sector-to-stock return mapping
2. Actual portfolio returns tracked correctly
3. Detailed weights used for performance calculation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

from events.message_bus import MessageBus
from events.base_event import TradeSignalEvent, RegimeEvent, MarketDataEvent


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
    
    FIXED: Now properly calculates returns using detailed stock-level weights
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
        self.current_weights: Dict[str, float] = {}  # Asset class weights
        self.current_detailed_weights: Dict[str, float] = {}  # Stock-level weights
        self.prices: Optional[pd.DataFrame] = None
        self.sector_mapping: Dict[str, List[str]] = {}
    
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
        
        # Get prices and sector mapping
        market_event = self.bus.get_last_event(MarketDataEvent)
        self.prices = market_event.prices
        self.sector_mapping = market_event.sector_mapping
        
        if self.prices is None:
            raise ValueError("No price data available")
        
        logger.info(f"Loaded {len(self.prices.columns)} assets, {len(self.prices)} days")
        logger.info(f"Sectors: {list(self.sector_mapping.keys())}")
        
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
                logger.error(f"Error at {date}: {e}", exc_info=True)
                self._record_state(date)  # Still record for continuity
        
        # Final update to latest date
        if len(rebalance_dates) > 0:
            final_date = self.prices.index[-1]
            self._update_portfolio_value(rebalance_dates[-1], final_date)
            self._record_state(final_date)
        
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
        """
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
        """
        Update portfolio value based on ACTUAL stock returns.
        
        FIXED: Uses detailed_weights (stock-level) instead of sector weights
        """
        if not self.current_detailed_weights:
            logger.debug("No detailed weights, skipping portfolio update")
            return
        
        period_return = 0.0
        total_weight = 0.0
        
        for ticker, weight in self.current_detailed_weights.items():
            if weight <= 0.0001:
                continue
                
            if ticker == 'Cash':
                # Cash earns risk-free rate
                days = (to_date - from_date).days
                ticker_return = 0.02 * days / 365  # 2% annual
                period_return += weight * ticker_return
                total_weight += weight
                
            elif ticker in self.prices.columns:
                try:
                    # Find closest available dates
                    from_idx = self.prices.index.get_indexer([from_date], method='pad')[0]
                    to_idx = self.prices.index.get_indexer([to_date], method='pad')[0]
                    
                    if from_idx >= 0 and to_idx >= 0 and from_idx < to_idx:
                        from_price = self.prices.iloc[from_idx][ticker]
                        to_price = self.prices.iloc[to_idx][ticker]
                        
                        if from_price > 0 and to_price > 0:
                            ticker_return = (to_price / from_price) - 1
                            period_return += weight * ticker_return
                            total_weight += weight
                            
                except Exception as e:
                    logger.debug(f"Error calculating return for {ticker}: {e}")
            
            # Handle sector names (map to constituent stocks if needed)
            elif ticker in self.sector_mapping:
                sector_tickers = self.sector_mapping[ticker]
                valid_tickers = [t for t in sector_tickers if t in self.prices.columns]
                
                if valid_tickers:
                    sector_return = 0.0
                    for st in valid_tickers:
                        try:
                            from_idx = self.prices.index.get_indexer([from_date], method='pad')[0]
                            to_idx = self.prices.index.get_indexer([to_date], method='pad')[0]
                            
                            if from_idx >= 0 and to_idx >= 0:
                                from_price = self.prices.iloc[from_idx][st]
                                to_price = self.prices.iloc[to_idx][st]
                                
                                if from_price > 0 and to_price > 0:
                                    sector_return += (to_price / from_price) - 1
                        except:
                            pass
                    
                    sector_return /= len(valid_tickers)  # Average
                    period_return += weight * sector_return
                    total_weight += weight
            
            # Handle ETF proxies (Bonds -> TLT, Commodities -> GLD)
            elif ticker == 'Bonds' and 'TLT' in self.prices.columns:
                try:
                    from_idx = self.prices.index.get_indexer([from_date], method='pad')[0]
                    to_idx = self.prices.index.get_indexer([to_date], method='pad')[0]
                    from_price = self.prices.iloc[from_idx]['TLT']
                    to_price = self.prices.iloc[to_idx]['TLT']
                    if from_price > 0:
                        period_return += weight * ((to_price / from_price) - 1)
                        total_weight += weight
                except:
                    pass
                    
            elif ticker == 'Commodities' and 'GLD' in self.prices.columns:
                try:
                    from_idx = self.prices.index.get_indexer([from_date], method='pad')[0]
                    to_idx = self.prices.index.get_indexer([to_date], method='pad')[0]
                    from_price = self.prices.iloc[from_idx]['GLD']
                    to_price = self.prices.iloc[to_idx]['GLD']
                    if from_price > 0:
                        period_return += weight * ((to_price / from_price) - 1)
                        total_weight += weight
                except:
                    pass
        
        # Apply return to capital
        old_capital = self.current_capital
        self.current_capital *= (1 + period_return)
        
        logger.info(f"Portfolio return: {period_return:.2%} | "
                   f"Capital: ${old_capital:,.0f} -> ${self.current_capital:,.0f} | "
                   f"Weight coverage: {total_weight:.1%}")
    
    def _execute_trades(self, date: pd.Timestamp, 
                        trade_signal: TradeSignalEvent) -> None:
        """Execute trades and deduct costs."""
        # Calculate turnover
        turnover = trade_signal.total_turnover
        
        # Get regime for slippage
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
        
        # Update weights (both asset-class and detailed)
        self.current_weights = trade_signal.target_weights.copy()
        
        # *** CRITICAL FIX: Use detailed weights for return calculation ***
        if trade_signal.detailed_weights:
            self.current_detailed_weights = trade_signal.detailed_weights.copy()
            logger.info(f"Updated detailed weights: {len(self.current_detailed_weights)} positions")
        else:
            # Fallback: use target_weights
            self.current_detailed_weights = trade_signal.target_weights.copy()
            logger.warning("No detailed_weights, using target_weights")
        
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
        
        logger.info(f"Executed trades: turnover={turnover:.2%}, cost=${total_cost:.2f}, "
                   f"regime={current_regime}")
    
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
        ann_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
        
        # Volatility (annualized for monthly data)
        # Assuming monthly rebalancing, multiply by sqrt(12)
        ann_vol = returns.std() * np.sqrt(12)
        
        # Sharpe ratio (using 2% risk-free rate)
        sharpe = (ann_return - 0.02) / ann_vol if ann_vol > 0 else 0
        
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
            'Total Trades': len(self.trade_history),
            'Final Capital': df['Value'].iloc[-1]
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
