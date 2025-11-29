from models.portfolio_optimizer import PortfolioOptimizer
import pandas as pd

class PortfolioAgent:
    def __init__(self, sector_map):
        self.optimizer = PortfolioOptimizer()
        self.sector_map = sector_map
        self.current_weights = {}

    def optimize_allocation(self, exp_returns, cov_matrix):
        """
        Optimizes asset class weights using Black-Litterman.
        """
        print("[PortfolioAgent] Optimizing Asset Allocation...")
        weights = self.optimizer.optimize_asset_classes(exp_returns, cov_matrix, use_black_litterman=True)
        return weights

    def select_stocks(self, asset_weights, prices, regime):
        """
        Selects individual stocks based on regime and allocates capital.
        """
        print("[PortfolioAgent] Selecting Stocks...")
        final_portfolio = {}
        
        for sector, weight in asset_weights.items():
            if sector in self.sector_map:
                tickers = self.sector_map[sector]
                valid_tickers = [t for t in tickers if t in prices.columns]
                if not valid_tickers: continue
                
                sector_data = prices[valid_tickers]
                top_stocks = self.optimizer.select_top_stocks(sector_data, regime, n_top=5)
                
                stock_weight = weight / len(top_stocks)
                for stock in top_stocks:
                    final_portfolio[stock] = stock_weight
            else:
                # Direct mapping for Bonds, Commodities, Volatility
                # We need to know the ticker for the asset class
                # Simplified mapping for now
                ticker = None
                if sector == 'Bonds': ticker = 'TLT'
                elif sector == 'Commodities': ticker = 'GLD'
                elif sector == 'Volatility': ticker = '^VIX'
                
                if ticker and ticker in prices.columns:
                    final_portfolio[ticker] = weight
                    
        self.current_weights = final_portfolio
        return final_portfolio
