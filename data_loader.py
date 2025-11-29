import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

class DataFetcher:
    def __init__(self, start_date="2010-01-01", end_date=None, data_dir="data"):
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime("%Y-%m-%d")
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        # Define Asset Universe
        self.sectors = {
            "Technology": ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "ADBE", "CRM", "AMD", "QCOM", "TXN", "INTC", "IBM", "AMAT", "NOW", "MU", "LRCX", "ADI", "PANW", "SNPS", "KLAC"],
            "Healthcare": ["LLY", "UNH", "JNJ", "MRK", "ABBV", "TMO", "AMGN", "PFE", "ISRG", "DHR", "BMY", "GILD", "VRTX", "REGN", "SYK", "EVP", "HUM", "BSX", "ZTS", "MCK"],
            "Financials": ["JPM", "V", "MA", "BAC", "WFC", "MS", "GS", "AXP", "BLK", "C", "SPGI", "PGR", "CB", "MMC", "USB", "ICE", "COF", "AIG", "TRV", "BK"],
            "Consumer Discretionary": ["AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "BKNG", "TJX", "TGT", "MAR", "LVS", "HLT", "CMG", "YUM", "ORLY", "ROST", "RCL", "EXPE", "ABNB"],
            "Energy": ["XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "OXY", "HES", "WMB", "KMI", "HAL", "BKR", "OKE", "DVN", "FANG", "TRGP", "CTRA", "MRO"]
        }
        
        self.bonds = ["TLT", "IEF", "SHY", "LQD", "HYG"] # 20y, 7-10y, 1-3y, Corp Inv Grade, High Yield
        self.commodities = ["GLD", "USO", "SLV", "DBA", "UNG"] # Gold, Oil, Silver, Agriculture, Nat Gas
        self.others = ["^VIX", "VNQ", "BTC-USD", "ETH-USD", "EURUSD=X"] # Volatility, REIT, Crypto, Forex

    def get_all_tickers(self):
        all_tickers = []
        for sector_tickers in self.sectors.values():
            all_tickers.extend(sector_tickers)
        all_tickers.extend(self.bonds)
        all_tickers.extend(self.commodities)
        all_tickers.extend(self.others)
        return list(set(all_tickers)) # Remove duplicates if any

    def fetch_data(self):
        tickers = self.get_all_tickers()
        print(f"Fetching data for {len(tickers)} assets...")
        
        # Download data
        # auto_adjust=True returns 'Close' which is adjusted.
        print("Downloading data...")
        data = yf.download(tickers, start=self.start_date, end=self.end_date, auto_adjust=True)
        
        print(f"Data Downloaded Shape: {data.shape}")
        print(f"Data Columns Levels: {data.columns.nlevels}")
        if data.columns.nlevels > 0:
            print(f"Level 0: {data.columns.get_level_values(0)[:5]}")
        
        close_prices = pd.DataFrame()

        if isinstance(data.columns, pd.MultiIndex):
            # Check if (Price, Ticker) or (Ticker, Price)
            # We look for 'Close' or 'Adj Close' in levels
            
            level0 = data.columns.get_level_values(0)
            level1 = data.columns.get_level_values(1)
            
            if 'Close' in level0 or 'Adj Close' in level0:
                # Structure: (Price, Ticker)
                target = 'Close' if 'Close' in level0 else 'Adj Close'
                close_prices = data.xs(target, level=0, axis=1)
            elif 'Close' in level1 or 'Adj Close' in level1:
                # Structure: (Ticker, Price)
                target = 'Close' if 'Close' in level1 else 'Adj Close'
                close_prices = data.xs(target, level=1, axis=1)
            else:
                print("Could not identify Price level in MultiIndex.")
                # Fallback: iterate and check
                pass
        else:
            # Flat DataFrame
            if 'Close' in data.columns:
                close_prices = data[['Close']]
                if len(tickers) == 1:
                    close_prices.columns = tickers
            elif 'Adj Close' in data.columns:
                close_prices = data[['Adj Close']]
                if len(tickers) == 1:
                    close_prices.columns = tickers
            else:
                # Maybe it is already just prices?
                close_prices = data

        # Drop columns with all NaNs
        close_prices.dropna(axis=1, how='all', inplace=True)

        # Forward fill to handle weekends/holidays if we have mixed assets (Crypto vs Stocks)
        close_prices.fillna(method='ffill', inplace=True)
        close_prices.fillna(method='bfill', inplace=True)

        # Drop columns with too many NaNs (e.g., recently listed stocks)
        # Relax threshold to 50% to be safe with mixed calendars
        close_prices.dropna(axis=1, thresh=int(len(close_prices)*0.5), inplace=True)

        output_path = os.path.join(self.data_dir, "market_data.csv")
        close_prices.to_csv(output_path)
        print(f"Data saved to {output_path}")
        return close_prices

if __name__ == "__main__":
    fetcher = DataFetcher()
    df = fetcher.fetch_data()
    print(df.head())
