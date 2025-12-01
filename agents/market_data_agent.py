"""
Market Data Agent
==================
Responsible for fetching and preprocessing market data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Type
from datetime import datetime, timedelta
import os
import pickle
import logging

from agents.base_agent import BaseAgent
from events.message_bus import MessageBus, Event, MarketDataEvent

logger = logging.getLogger(__name__)


class MarketDataAgent(BaseAgent):
    """
    Market Data Agent for fetching and preprocessing data.
    
    Responsibilities:
    - Fetch price data from various sources
    - Handle data caching
    - Calculate returns and basic statistics
    - Quality checks and data cleaning
    
    Config Options:
        start_date: Data start date
        end_date: Data end date
        cache_dir: Directory for data cache
        cache_ttl_hours: Cache time-to-live in hours
        sectors: Dict of sector -> tickers
        bonds: List of bond ETF tickers
        commodities: List of commodity tickers
    """
    
    def __init__(self, message_bus: MessageBus, config: Optional[Dict] = None):
        super().__init__("MarketDataAgent", message_bus, config)
        
        # Configuration
        self.start_date = self.get_config('start_date', '2005-01-01')
        self.end_date = self.get_config('end_date', '2025-11-30')
        self.cache_dir = self.get_config('cache_dir', './data/cache')
        self.cache_ttl_hours = self.get_config('cache_ttl_hours', 24)
        self.min_data_quality = self.get_config('min_data_quality', 0.5)
        
        # Asset universe
        self.sectors = self.get_config('sectors', {
            'Technology': ['AAPL', 'MSFT', 'NVDA', 'AVGO', 'ORCL', 'ADBE', 'CRM', 'AMD'],
            'Healthcare': ['LLY', 'UNH', 'JNJ', 'MRK', 'ABBV', 'TMO', 'AMGN', 'PFE'],
            'Financials': ['JPM', 'V', 'MA', 'BAC', 'WFC', 'MS', 'GS', 'AXP'],
            'Consumer': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'LOW', 'SBUX', 'TGT'],
            'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'PSX', 'OXY']
        })
        self.bonds = self.get_config('bonds', ['TLT', 'IEF', 'SHY', 'LQD', 'HYG'])
        self.commodities = self.get_config('commodities', ['GLD', 'USO', 'SLV', 'DBC'])
        self.others = self.get_config('others', ['VNQ', 'SPY'])
        
        # State
        self.prices: Optional[pd.DataFrame] = None
        self.volumes: Optional[pd.DataFrame] = None
        self.returns: Optional[pd.DataFrame] = None
        self.data_quality: Dict[str, float] = {}
        
    def _initialize(self) -> None:
        """Initialize data agent"""
        os.makedirs(self.cache_dir, exist_ok=True)
        self.logger.info(f"Initialized with {len(self.get_all_tickers())} tickers")
    
    def get_subscribed_events(self) -> List[Type[Event]]:
        """No subscriptions - this is a data source agent"""
        return []
    
    def on_event(self, event: Event) -> None:
        """No event handling needed"""
        pass
    
    def get_all_tickers(self) -> List[str]:
        """Get complete list of tickers"""
        tickers = []
        for sector_tickers in self.sectors.values():
            tickers.extend(sector_tickers)
        tickers.extend(self.bonds)
        tickers.extend(self.commodities)
        tickers.extend(self.others)
        return tickers
    
    def _process(self) -> None:
        """
        Main processing:
        1. Check cache
        2. Fetch data if needed
        3. Clean and validate
        4. Calculate returns
        5. Publish MarketDataEvent
        """
        all_tickers = self.get_all_tickers()
        
        # Try to load from cache
        cached_data = self._load_cache()
        
        if cached_data is not None:
            self.prices, self.volumes = cached_data
            self.logger.info("Loaded data from cache")
        else:
            # Fetch fresh data
            self.prices, self.volumes = self._fetch_data(all_tickers)
            
            # Save to cache
            self._save_cache(self.prices, self.volumes)
        
        # Clean data
        self.prices = self._clean_data(self.prices)
        
        # Calculate returns
        self.returns = self.prices.pct_change().dropna()
        
        # Data quality check
        self.data_quality = self._check_data_quality()
        
        # Create sector mapping
        sector_mapping = {}
        for sector, tickers in self.sectors.items():
            valid_tickers = [t for t in tickers if t in self.prices.columns]
            sector_mapping[sector] = valid_tickers
        
        # Publish event
        event = MarketDataEvent(
            source_agent=self.name,
            prices=self.prices,
            volumes=self.volumes,
            returns=self.returns,
            sector_mapping=sector_mapping,
            data_quality=self.data_quality,
            as_of_date=datetime.now()
        )
        
        self.publish(event)
        self.logger.info(f"Published MarketDataEvent with {len(self.prices.columns)} tickers, "
                        f"{len(self.prices)} observations")
    
    def _fetch_data(self, tickers: List[str]) -> tuple:
        """Fetch data from Yahoo Finance or generate synthetic"""
        try:
            import yfinance as yf
            
            self.logger.info(f"Fetching data for {len(tickers)} tickers from Yahoo Finance...")
            
            data = yf.download(
                tickers,
                start=self.start_date,
                end=self.end_date,
                progress=True,
                group_by='column'
            )
            
            if 'Adj Close' in data.columns.get_level_values(0):
                prices = data['Adj Close']
            elif 'Close' in data.columns.get_level_values(0):
                prices = data['Close']
            else:
                prices = data
            
            if 'Volume' in data.columns.get_level_values(0):
                volumes = data['Volume']
            else:
                volumes = pd.DataFrame()
            
            self.logger.info(f"Fetched {len(prices)} observations for {len(prices.columns)} tickers")
            
            return prices, volumes
            
        except Exception as e:
            self.logger.warning(f"Failed to fetch from Yahoo Finance: {e}")
            self.logger.info("Generating synthetic data...")
            return self._generate_synthetic_data(tickers)
    
    def _generate_synthetic_data(self, tickers: List[str]) -> tuple:
        """Generate synthetic price data for testing"""
        np.random.seed(42)
        
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='B')
        n_days = len(dates)
        n_assets = len(tickers)
        
        # Generate correlated returns
        base_mu = np.random.uniform(0.0002, 0.0006, n_assets)
        base_sigma = np.random.uniform(0.015, 0.035, n_assets)
        
        # Create correlation matrix with sector structure
        corr = np.eye(n_assets) * 0.3 + 0.7 * np.eye(n_assets)
        
        # Generate returns
        returns = np.random.multivariate_normal(
            base_mu, 
            np.diag(base_sigma) @ corr @ np.diag(base_sigma),
            n_days
        )
        
        # Convert to prices
        prices = 100 * np.cumprod(1 + returns, axis=0)
        
        prices_df = pd.DataFrame(prices, index=dates, columns=tickers)
        volumes_df = pd.DataFrame(
            np.random.randint(100000, 10000000, (n_days, n_assets)),
            index=dates,
            columns=tickers
        )
        
        return prices_df, volumes_df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess data"""
        # Forward fill missing values (max 5 days)
        df = df.ffill(limit=5)
        
        # Drop columns with too many missing values
        threshold = len(df) * self.min_data_quality
        df = df.dropna(axis=1, thresh=int(threshold))
        
        # Drop remaining rows with NaN
        df = df.dropna()
        
        # Remove any columns with zero variance
        df = df.loc[:, df.std() > 0]
        
        return df
    
    def _check_data_quality(self) -> Dict[str, float]:
        """Check data quality for each ticker"""
        quality = {}
        
        for col in self.prices.columns:
            # Completeness
            non_null = self.prices[col].notna().sum() / len(self.prices)
            
            # Variance check
            has_variance = self.prices[col].std() > 0
            
            # Combined quality score
            quality[col] = non_null if has_variance else 0.0
        
        return quality
    
    def _load_cache(self) -> Optional[tuple]:
        """Load data from cache if valid"""
        cache_file = os.path.join(self.cache_dir, 'market_data.pkl')
        
        if not os.path.exists(cache_file):
            return None
        
        # Check cache age
        cache_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        age_hours = (datetime.now() - cache_time).total_seconds() / 3600
        
        if age_hours > self.cache_ttl_hours:
            self.logger.info(f"Cache expired ({age_hours:.1f} hours old)")
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            return data['prices'], data['volumes']
        except Exception as e:
            self.logger.warning(f"Failed to load cache: {e}")
            return None
    
    def _save_cache(self, prices: pd.DataFrame, volumes: pd.DataFrame) -> None:
        """Save data to cache"""
        cache_file = os.path.join(self.cache_dir, 'market_data.pkl')
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({'prices': prices, 'volumes': volumes}, f)
            self.logger.info("Saved data to cache")
        except Exception as e:
            self.logger.warning(f"Failed to save cache: {e}")
    
    # =========================================================================
    # Public API
    # =========================================================================
    
    def get_sector_returns(self) -> pd.DataFrame:
        """Get sector-level returns (equal-weighted)"""
        if self.returns is None:
            return pd.DataFrame()
        
        sector_returns = pd.DataFrame()
        
        for sector, tickers in self.sectors.items():
            valid_tickers = [t for t in tickers if t in self.returns.columns]
            if valid_tickers:
                sector_returns[sector] = self.returns[valid_tickers].mean(axis=1)
        
        return sector_returns
    
    def get_market_return(self) -> pd.Series:
        """Get market return (equal-weighted all stocks)"""
        if self.returns is None:
            return pd.Series()
        
        return self.returns.mean(axis=1)
