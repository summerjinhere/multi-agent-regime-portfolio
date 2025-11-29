import pandas as pd
import numpy as np
from models.hmm_regime import RegimeDetector
from models.ms_vecm import MSVECMAllocator
from models.portfolio_optimizer import PortfolioOptimizer
from datetime import timedelta

class Backtester:
    def __init__(self, prices, initial_capital=100000, transaction_cost=0.001, n_components=3):
        self.prices = prices
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.portfolio_value = []
        self.weights_history = []
        self.regime_history = []
        
        # Initialize models
        self.regime_detector = RegimeDetector(n_components=n_components)
        self.vecm_allocator = MSVECMAllocator()
        self.optimizer = PortfolioOptimizer()

    def run_backtest(self, start_date, rebalance_freq='M', lookback_window=252):
        """
        Run rolling window backtest.
        :param start_date: Date to start the backtest.
        :param rebalance_freq: 'M' for Monthly, 'W' for Weekly.
        """
        # Resample dates for rebalancing
        rebalance_dates = self.prices.loc[start_date:].resample(rebalance_freq).last().index
        
        current_capital = self.initial_capital
        current_weights = {} # Key: Ticker, Value: Weight
        
        print(f"Starting Backtest from {start_date} with {len(rebalance_dates)} periods. Cost: {self.transaction_cost*10000}bps")
        
        for i, date in enumerate(rebalance_dates):
            if date not in self.prices.index:
                # Find closest previous date
                date = self.prices.index[self.prices.index.get_indexer([date], method='pad')[0]]
            
            # 1. Get Historical Data up to t
            history = self.prices.loc[:date]
            if len(history) < lookback_window:
                continue
                
            # 2. Detect Regime
            # Use a proxy (e.g., mean of all assets)
            market_proxy = history.mean(axis=1).to_frame(name='Market')
            self.regime_detector.prepare_features(market_proxy, asset_col='Market')
            try:
                self.regime_detector.fit()
                current_regime = self.regime_detector.get_current_regime()
                # Capture probabilities of the last state
                probs = self.regime_detector.model.predict_proba(self.regime_detector.data.values)[-1]
            except:
                current_regime = 0 # Default to Bull if fail
                probs = [1, 0, 0] # Default to 100% Bull confidence
            
            self.regime_history.append({
                'Date': date, 
                'Regime': current_regime,
                'Prob_Bull': probs[0] if len(probs) > 0 else 0,
                'Prob_Trans': probs[1] if len(probs) > 1 else 0,
                'Prob_Bear': probs[2] if len(probs) > 2 else 0
            })
            
            # 3. Asset Allocation (MS-VECM + Black-Litterman)
            # Aggregate to Asset Classes
            # (Simplified aggregation logic similar to main.py)
            asset_class_prices = pd.DataFrame()
            if 'TLT' in history.columns: asset_class_prices['Bonds'] = history['TLT']
            if 'GLD' in history.columns: asset_class_prices['Commodities'] = history['GLD']
            if '^VIX' in history.columns: asset_class_prices['Volatility'] = history['^VIX']
            
            # Sector proxies (using first valid ticker as proxy for now for speed)
            # In production, use proper sector indices
            sectors = ["Technology", "Financials", "Healthcare", "Consumer Discretionary", "Energy"]
            sector_map = {
                "Technology": "AAPL", "Financials": "JPM", "Healthcare": "JNJ", 
                "Consumer Discretionary": "AMZN", "Energy": "XOM"
            }
            for s, t in sector_map.items():
                if t in history.columns:
                    asset_class_prices[s] = history[t]
            
            asset_class_prices.dropna(inplace=True)
            
            # Forecast
            exp_returns, cov_matrix = self.vecm_allocator.fit_predict(asset_class_prices, current_regime)
            
            # Optimize (Black-Litterman is inside optimize_asset_classes now)
            # We pass market caps or equilibrium assumptions if needed, 
            # but for now we'll use the enhanced optimizer which handles BL internally or via args.
            weights = self.optimizer.optimize_asset_classes(exp_returns, cov_matrix, use_black_litterman=True)
            
            # 4. Record Performance & Deduct Costs
            # Calculate return since last rebalance
            if i > 0:
                prev_date = rebalance_dates[i-1]
                if prev_date in self.prices.index:
                    period_return = 0
                    for asset, w in current_weights.items():
                        # Map asset class weight to proxy ticker
                        ticker = asset # In this simplified loop, asset is the column name in asset_class_prices
                        # We need to map back to actual tickers in 'prices'
                        # For simplicity in this backtest, we assume we hold the proxy tickers directly
                        
                        # Find the ticker for this asset class
                        real_ticker = None
                        if asset in sector_map: real_ticker = sector_map[asset]
                        elif asset == 'Bonds': real_ticker = 'TLT'
                        elif asset == 'Commodities': real_ticker = 'GLD'
                        elif asset == 'Volatility': real_ticker = '^VIX'
                        
                        if real_ticker and real_ticker in self.prices.columns:
                            r = self.prices.loc[date, real_ticker] / self.prices.loc[prev_date, real_ticker] - 1
                            period_return += w * r
                    
                    current_capital *= (1 + period_return)
            
            # Calculate Turnover and Transaction Cost
            # Turnover = sum(|w_new - w_old|)
            # We need to align keys
            all_assets = set(weights.keys()).union(set(current_weights.keys()))
            turnover = 0
            for asset in all_assets:
                w_new = weights.get(asset, 0)
                w_old = current_weights.get(asset, 0)
                turnover += abs(w_new - w_old)
            
            # --- Advanced Realism: Regime-Dependent Slippage ---
            # Logic: Liquidity dries up in Bear/Crisis markets, increasing slippage.
            # Regime 0 (Bull): High Liquidity -> Low Slippage (e.g., 5bps)
            # Regime 1 (Transition/Bear): Medium Liquidity -> Medium Slippage (e.g., 10bps)
            # Regime 2 (Crisis): Low Liquidity -> High Slippage (e.g., 25bps)
            
            slippage_map = {0: 0.0005, 1: 0.0010, 2: 0.0025}
            current_slippage = slippage_map.get(current_regime, 0.0010)
            
            # Total Cost = Commission (Transaction Cost) + Slippage
            # Cost is applied on the Traded Volume (Turnover * Capital)
            traded_volume = turnover * current_capital
            
            commission_cost = traded_volume * self.transaction_cost
            slippage_cost = traded_volume * current_slippage
            
            total_cost = commission_cost + slippage_cost
            current_capital -= total_cost
            
            self.portfolio_value.append({'Date': date, 'Value': current_capital})
            current_weights = weights
            self.weights_history.append({**{'Date': date}, **weights})
            
            print(f"Date: {date.date()} | Regime: {current_regime} | Value: ${current_capital:,.2f} | Cost: ${total_cost:.2f} (Slip: {current_slippage*10000:.0f}bps)")

        return pd.DataFrame(self.portfolio_value).set_index('Date')

    def calculate_metrics(self, benchmark_col='SPY'):
        """
        Calculate Sharpe, Drawdown, etc.
        """
        df = pd.DataFrame(self.portfolio_value).set_index('Date')
        df['Returns'] = df['Value'].pct_change()
        
        # Annualized Return
        total_ret = (df['Value'].iloc[-1] / df['Value'].iloc[0]) - 1
        days = (df.index[-1] - df.index[0]).days
        ann_ret = (1 + total_ret) ** (365/days) - 1
        
        # Volatility
        ann_vol = df['Returns'].std() * np.sqrt(252)
        
        # Sharpe
        sharpe = ann_ret / ann_vol if ann_vol != 0 else 0
        
        # Max Drawdown
        cum_ret = (1 + df['Returns']).cumprod()
        peak = cum_ret.cummax()
        drawdown = (cum_ret - peak) / peak
        max_dd = drawdown.min()
        
        return {
            "Total Return": total_ret,
            "Annualized Return": ann_ret,
            "Annualized Volatility": ann_vol,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": max_dd
        }
