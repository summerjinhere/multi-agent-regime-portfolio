from agents.market_agent import MarketDataAgent
from agents.quant_agent import QuantAgent
from agents.risk_agent import RiskManagerAgent
from agents.portfolio_agent import PortfolioAgent
import pandas as pd

def run_orchestrator():
    print("="*50)
    print("INITIALIZING MULTI-AGENT SYSTEM")
    print("="*50)

    # 1. Initialize Agents
    market_agent = MarketDataAgent()
    quant_agent = QuantAgent()
    risk_agent = RiskManagerAgent()
    
    # We need the sector map for PortfolioAgent
    # Accessing it from MarketAgent's fetcher
    portfolio_agent = PortfolioAgent(market_agent.fetcher.sectors)

    # 2. Market Agent: Get Data
    prices = market_agent.get_data()
    
    # 3. Quant Agent: Analyze & Forecast
    regime = quant_agent.analyze_market(prices)
    
    # Prepare Asset Class Data for VECM
    # (Simplified aggregation logic reused)
    sector_prices = pd.DataFrame()
    for sector, tickers in market_agent.fetcher.sectors.items():
        valid_tickers = [t for t in tickers if t in prices.columns]
        if valid_tickers:
            sector_prices[sector] = prices[valid_tickers].mean(axis=1)
            
    asset_class_prices = sector_prices.copy()
    if 'TLT' in prices.columns: asset_class_prices['Bonds'] = prices['TLT']
    if 'GLD' in prices.columns: asset_class_prices['Commodities'] = prices['GLD']
    if '^VIX' in prices.columns: asset_class_prices['Volatility'] = prices['^VIX']
    asset_class_prices.dropna(inplace=True)
    
    exp_returns, cov_matrix = quant_agent.forecast_returns(asset_class_prices)
    
    # 4. Portfolio Agent: Optimize
    asset_weights = portfolio_agent.optimize_allocation(exp_returns, cov_matrix)
    
    # 5. Risk Agent: Check Limits (Pass-through for now)
    safe_weights = risk_agent.check_limits(asset_weights)
    
    # 6. Portfolio Agent: Select Stocks
    final_portfolio = portfolio_agent.select_stocks(safe_weights, prices, regime)
    
    # 7. Execution (Simulation)
    print("\n" + "="*50)
    print("FINAL AGENT DECISION")
    print("="*50)
    sorted_portfolio = sorted(final_portfolio.items(), key=lambda x: x[1], reverse=True)
    for asset, w in sorted_portfolio:
        print(f"{asset}: {w:.2%}")
        
    # Calculate Slippage for this theoretical trade
    risk_agent.calculate_cost(turnover=2.0, capital=100000, regime=regime)

    # 8. Benchmark Comparison (Historical Context)
    print("\n" + "="*50)
    print("AGENT VS BENCHMARK (PERFORMANCE REVIEW)")
    print("="*50)
    
    try:
        # Load Agent Performance
        agent_perf = pd.read_csv("backtest_results.csv", index_col=0, parse_dates=True)
        agent_total_return = (agent_perf['Value'].iloc[-1] / agent_perf['Value'].iloc[0]) - 1
        
        # Load Benchmark (SPY)
        import yfinance as yf
        start_date = agent_perf.index[0]
        end_date = agent_perf.index[-1]
        spy = yf.download("SPY", start=start_date, end=end_date + pd.Timedelta(days=1), auto_adjust=True, progress=False)
        
        if isinstance(spy.columns, pd.MultiIndex):
            spy_close = spy.xs('Close', level=0, axis=1) if 'Close' in spy.columns.levels[0] else spy.iloc[:, 0]
        else:
            spy_close = spy['Close']
            
        # Ensure scalar float
        agent_total_return = float((agent_perf['Value'].iloc[-1] / agent_perf['Value'].iloc[0]) - 1)
        spy_total_return = float((spy_close.iloc[-1] / spy_close.iloc[0]) - 1)
        
        # Calculate CAGR
        days = (end_date - start_date).days
        agent_cagr = (1 + agent_total_return) ** (365 / days) - 1
        spy_cagr = (1 + spy_total_return) ** (365 / days) - 1
        
        print(f"{'Metric':<20} | {'Agent':<15} | {'SPY (Benchmark)':<15}")
        print("-" * 56)
        print(f"{'Total Return':<20} | {agent_total_return:>14.2%} | {spy_total_return:>14.2%}")
        print(f"{'CAGR':<20} | {agent_cagr:>14.2%} | {spy_cagr:>14.2%}")
        
        if agent_total_return > spy_total_return:
            print(f"\n>> SUCCESS: Agent outperformed Benchmark by {agent_total_return - spy_total_return:.2%}")
        else:
            print(f"\n>> NOTICE: Agent underperformed Benchmark.")
            
    except Exception as e:
        print(f"Could not run comparison: {e}")

if __name__ == "__main__":
    run_orchestrator()
