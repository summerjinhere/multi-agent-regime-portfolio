import yaml
import pandas as pd
import os

def generate_fallback():
    print("Generating fallback portfolio...")
    
    # Load config
    with open('config/vis_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    tickers = []
    # Add sectors
    for sector, sector_tickers in config['market_data']['sectors'].items():
        tickers.extend(sector_tickers)
    
    # Add other assets
    if 'bonds' in config['market_data']:
        tickers.extend(config['market_data']['bonds'])
    if 'commodities' in config['market_data']:
        tickers.extend(config['market_data']['commodities'])
    if 'others' in config['market_data']:
        tickers.extend(config['market_data']['others'])
        
    # Remove duplicates
    tickers = list(set(tickers))
    
    print(f"Found {len(tickers)} tickers in config.")
    
    if not tickers:
        print("No tickers found!")
        return
        
    # Equal weight
    weight = 1.0 / len(tickers)
    weights = {ticker: weight for ticker in tickers}
    
    # Save detailed
    detailed = pd.Series(weights)
    detailed.to_csv('final_portfolio_detailed.csv')
    print("Saved final_portfolio_detailed.csv")
    
    # Save summary (aggregated by sector for backward compatibility if needed)
    # But dashboard uses detailed if available.
    detailed.to_csv('final_portfolio.csv') 
    print("Saved final_portfolio.csv")

if __name__ == "__main__":
    generate_fallback()
