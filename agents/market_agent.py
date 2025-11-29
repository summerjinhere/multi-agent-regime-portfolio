from data_loader import DataFetcher
import pandas as pd
import os

class MarketDataAgent:
    def __init__(self, start_date="2015-01-01"):
        self.fetcher = DataFetcher(start_date=start_date)
        self.data = None

    def get_data(self):
        """
        Fetches data using DataFetcher or loads from CSV if available.
        """
        print("[MarketDataAgent] Scanning market data...")
        data_path = os.path.join(self.fetcher.data_dir, "market_data.csv")
        
        if os.path.exists(data_path):
            print(f"[MarketDataAgent] Loading cached data from {data_path}")
            self.data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        else:
            print("[MarketDataAgent] Downloading fresh data...")
            self.data = self.fetcher.fetch_data()
            
        print(f"[MarketDataAgent] Data ready. Shape: {self.data.shape}")
        return self.data

    def get_latest_prices(self):
        if self.data is not None:
            return self.data.iloc[-1]
        return None
