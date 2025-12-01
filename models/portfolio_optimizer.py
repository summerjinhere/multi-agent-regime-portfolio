import pandas as pd
import numpy as np
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import objective_functions

class PortfolioOptimizer:
    def __init__(self):
        pass

    def optimize_asset_classes(self, expected_returns, cov_matrix, risk_aversion=1.0, use_black_litterman=False):
        """
        Optimize allocation across asset classes.
        If use_black_litterman is True, we use the VECM forecasts as 'Views' 
        and blend them with market equilibrium.
        """
        mu = expected_returns
        S = cov_matrix
        
        if use_black_litterman:
            from pypfopt import black_litterman, BlackLittermanModel
            
            # 1. Market Equilibrium (Prior)
            # We assume equal weights as the "Market" for simplicity, or use heuristic
            # In a real scenario, we'd fetch market caps.
            # Here we derive implied returns from an Equal-Weighted portfolio as the "Market Benchmark"
            market_weights = {k: 1.0/len(mu) for k in mu.index}
            
            # Delta (Risk Aversion)
            delta = black_litterman.market_implied_risk_aversion(pd.Series(market_weights), risk_free_rate=0.025)
            
            # Prior returns (Pi)
            prior_returns = black_litterman.market_implied_prior_returns(market_caps=pd.Series(market_weights), risk_aversion=delta, cov_matrix=S, risk_free_rate=0.02)
            
            # 2. Views (VECM Forecasts)
            # We treat our VECM expected returns as absolute views
            views = mu.to_dict()
            
            # 3. Confidence (Omega)
            # We can use Idzorek's method or simple diagonal
            # Here we assume we are 50% confident in our views vs market
            # Or we can set omega based on variance
            
            try:
                bl = BlackLittermanModel(S, pi=prior_returns, absolute_views=views)
                ret_bl = bl.bl_returns()
                cov_bl = bl.bl_cov()
                
                # Use Posterior for Optimization
                mu = ret_bl
                S = cov_bl
            except Exception as e:
                print(f"Black-Litterman failed: {e}. Falling back to Mean-Variance.")

        # Optimize for Maximal Sharpe Ratio or specific risk aversion
        try:
            ef = EfficientFrontier(mu, S)
            # ef.add_objective(objective_functions.L2_reg, gamma=0.1) # Regularization
            
            # Max Sharpe
            weights = ef.max_sharpe()
            cleaned_weights = ef.clean_weights()
            return cleaned_weights
        except Exception as e:
            print(f"Optimization failed: {e}")
            # Equal weight fallback
            n = len(expected_returns)
            return {k: 1.0/n for k in expected_returns.index}

    def select_top_stocks(self, sector_data, regime, n_top=20):
        """
        Select top stocks within a sector based on regime.
        Regime 0 (Low Vol/Bull): Momentum / High Beta
        Regime 1 (High Vol/Bear): Low Volatility / Quality / Low Beta
        """
        # Calculate metrics
        returns = sector_data.pct_change()
        mean_ret = returns.mean()
        vol = returns.std()
        
        scores = pd.Series(index=sector_data.columns, dtype=float)
        
        if regime == 0: # Bull
            # Score = Return / Vol (Sharpe-like) or just Momentum
            scores = mean_ret
        elif regime == 1: # Bear/High Vol
            # Score = -Volatility (Low Vol preference)
            scores = -vol
        else: # Stagnation or other
            # Balanced
            scores = mean_ret / vol
            
        top_stocks = scores.nlargest(n_top).index.tolist()
        return top_stocks

    def hierarchical_allocation(self, asset_class_weights, sector_allocations, stock_selections):
        """
        Combine asset class weights with stock selections to get final portfolio weights.
        """
        final_weights = {}
        
        # Example: asset_class_weights = {'Equity': 0.6, 'Bonds': 0.3, ...}
        # We need to know which asset class each ticker belongs to.
        # For this implementation, we'll assume a simplified mapping or passed in structure.
        
        return final_weights
