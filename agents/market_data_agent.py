"""
Market Data Agent
=================
Responsible for data acquisition, validation, and distribution.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import os
import logging

from agents.base_agent import BaseAgent
from events.message_bus import MessageBus
from events.base_event import MarketDataEvent


class MarketDataAgent(BaseAgent):
    """
    Agent responsible for fetching and preprocessing market data.
    
    Responsibilities:
    - Fetch price data from Yahoo Finance
    - Validate data quality
    - Handle missing data and outliers
    - Publish cleaned data to other agents
    
    Config Options:
        start_date: Data start date (YYYY-MM-DD)
        end_date: Data end date (None for today)
        cache_dir: Directory for cached data
        cache_ttl_hours: Cache time-to-live
        min_data_quality: Minimum acceptable data quality score (0-1)
    """
    
    # Default asset universe
    DEFAULT_SECTORS = {
        "Technology": ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "ADBE", "CRM", "AMD"],
        "Healthcare": ["LLY", "UNH", "JNJ", "MRK", "ABBV", "TMO", "AMGN", "PFE"],
        "Financials": ["JPM", "V", "MA", "BAC", "WFC", "MS", "GS", "AXP"],
        "Consumer Discretionary": ["AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX"],
        "Energy": ["XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX"]
    }
    
    DEFAULT_BONDS = ["TLT", "IEF", "SHY", "LQD", "HYG"]
    DEFAULT_COMMODITIES = ["GLD", "USO", "SLV"]
    DEFAULT_OTHERS = ["^VIX", "VNQ"]
    
    def __init__(self, message_bus: MessageBus, config: Optional[Dict] = None):
        super().__init__("MarketDataAgent", message_bus, config)
        
        # Data storage
        self.prices: Optional[pd.DataFrame] = None
        self.returns: Optional[pd.DataFrame] = None
        self.sector_mapping: Dict[str, List[str]] = {}
        
        # Configuration
        self.start_date = self.get_config('start_date', '2015-01-01')
        self.end_date = self.get_config('end_date', '2025-11-29')
        self.cache_dir = self.get_config('cache_dir', 'data')
        self.cache_ttl_hours = self.get_config('cache_ttl_hours', 24)
        self.min_data_quality = self.get_config('min_data_quality', 0.5)
        
        # Custom universe (optional)
        self.custom_sectors = self.get_config('sectors', self.DEFAULT_SECTORS)
        self.custom_bonds = self.get_config('bonds', self.DEFAULT_BONDS)
        self.custom_commodities = self.get_config('commodities', self.DEFAULT_COMMODITIES)
        self.custom_others = self.get_config('others', self.DEFAULT_OTHERS)
    
    def _initialize(self) -> None:
        """Initialize data directories and validate config."""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            self.logger.info(f"Created cache directory: {self.cache_dir}")
        
        # Build sector mapping
        self.sector_mapping = self.custom_sectors.copy()
        self.logger.info(f"Initialized with {len(self._get_all_tickers())} tickers")
    
    def _process(self) -> None:
        """
        Main processing: fetch data and publish MarketDataEvent.
        """
        # Try to load from cache first
        cache_path = os.path.join(self.cache_dir, 'market_data.csv')
        cache_valid = self._is_cache_valid(cache_path)
        
        if cache_valid:
            self.logger.info("Loading data from cache...")
            self.prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        else:
            self.logger.info("Fetching fresh data from Yahoo Finance...")
            self.prices = self._fetch_data()
            self._save_cache(cache_path)
        
        # Calculate returns
        self.returns = self.prices.pct_change().dropna()
        
        # Validate data quality
        quality_score = self._validate_data_quality()
        missing_tickers = self._get_missing_tickers()
        
        if quality_score < self.min_data_quality:
            self.logger.warning(f"Data quality ({quality_score:.2f}) below threshold ({self.min_data_quality})")
        
        # Publish event
        event = MarketDataEvent(
            source_agent=self.name,
            prices=self.prices,
            returns=self.returns,
            asset_universe=list(self.prices.columns),
            sector_mapping=self.sector_mapping,
            data_quality_score=quality_score,
            missing_tickers=missing_tickers,
            date_range=(
                str(self.prices.index[0].date()),
                str(self.prices.index[-1].date())
            )
        )
        
        self.publish(event)
        self.logger.info(f"Published MarketDataEvent: {len(self.prices.columns)} assets, "
                        f"quality={quality_score:.2f}")
    
    def _get_all_tickers(self) -> List[str]:
        """Get complete list of tickers."""
        tickers = []
        for sector_tickers in self.custom_sectors.values():
            tickers.extend(sector_tickers)
        tickers.extend(self.custom_bonds)
        tickers.extend(self.custom_commodities)
        tickers.extend(self.custom_others)
        return list(set(tickers))
    
    def _fetch_data(self) -> pd.DataFrame:
        """
        Fetch data from Yahoo Finance.
        
        Returns:
            DataFrame of adjusted close prices
        """
        tickers = self._get_all_tickers()
        end_date = self.end_date or datetime.now().strftime("%Y-%m-%d")
        
        self.logger.info(f"Downloading {len(tickers)} tickers from {self.start_date} to {end_date}")
        
        try:
            data = yf.download(
                tickers, 
                start=self.start_date, 
                end=end_date, 
                auto_adjust=True,
                progress=False
            )
        except Exception as e:
            self.logger.error(f"Download failed: {e}")
            raise
        
        # Extract close prices from MultiIndex
        close_prices = self._extract_close_prices(data)
        
        # Clean data
        close_prices = self._clean_data(close_prices)
        
        return close_prices
    
    def _extract_close_prices(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract close prices from yfinance data structure."""
        if isinstance(data.columns, pd.MultiIndex):
            level0 = data.columns.get_level_values(0)
            if 'Close' in level0:
                return data.xs('Close', level=0, axis=1)
            elif 'Adj Close' in level0:
                return data.xs('Adj Close', level=0, axis=1)
            else:
                # Try level 1
                level1 = data.columns.get_level_values(1)
                if 'Close' in level1:
                    return data.xs('Close', level=1, axis=1)
        
        # Fallback for single ticker or flat structure
        if 'Close' in data.columns:
            return data[['Close']]
        
        return data
    
    def _clean_data(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess price data.
        
        Steps:
        1. Remove all-NaN columns
        2. Forward/backward fill missing values
        3. Remove tickers with too many missing values
        4. Handle outliers
        """
        # Remove all-NaN columns
        prices = prices.dropna(axis=1, how='all')
        
        # Forward fill then backward fill
        prices = prices.fillna(method='ffill').fillna(method='bfill')
        
        # Remove columns with >50% missing (before fill)
        # Already filled, so check for zeros or suspicious patterns
        
        # Remove columns with constant values (no trading)
        non_constant = prices.std() > 0
        prices = prices.loc[:, non_constant]
        
        self.logger.info(f"Cleaned data: {prices.shape[1]} tickers, {len(prices)} days")
        return prices
    
    def _validate_data_quality(self) -> float:
        """
        Calculate data quality score (0-1).
        
        Factors:
        - Coverage: % of requested tickers obtained
        - Completeness: % of non-missing values
        - Recency: How recent is the latest data
        """
        if self.prices is None or len(self.prices) == 0:
            return 0.0
        
        requested_tickers = self._get_all_tickers()
        obtained_tickers = list(self.prices.columns)
        
        # Coverage score
        coverage = len(set(obtained_tickers) & set(requested_tickers)) / len(requested_tickers)
        
        # Completeness score (after cleaning, should be ~1.0)
        completeness = 1.0 - (self.prices.isna().sum().sum() / self.prices.size)
        
        # Recency score
        latest_date = self.prices.index[-1]
        days_old = (datetime.now() - pd.Timestamp(latest_date).to_pydatetime()).days
        recency = max(0, 1 - (days_old / 7))  # Penalize if >7 days old
        
        # Weighted average
        quality = 0.4 * coverage + 0.4 * completeness + 0.2 * recency
        
        return round(quality, 4)
    
    def _get_missing_tickers(self) -> List[str]:
        """Get list of tickers that failed to download."""
        if self.prices is None:
            return self._get_all_tickers()
        
        requested = set(self._get_all_tickers())
        obtained = set(self.prices.columns)
        return list(requested - obtained)
    
    def _is_cache_valid(self, cache_path: str) -> bool:
        """Check if cached data is still valid."""
        if not os.path.exists(cache_path):
            return False
        
        # Check file age
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        age_hours = (datetime.now() - file_time).total_seconds() / 3600
        
        if age_hours > self.cache_ttl_hours:
            self.logger.info(f"Cache expired ({age_hours:.1f} hours old)")
            return False
        
        return True
    
    def _save_cache(self, cache_path: str) -> None:
        """Save data to cache."""
        if self.prices is not None:
            self.prices.to_csv(cache_path)
            self.logger.info(f"Data cached to {cache_path}")
    
    # =========================================================================
    # Public API
    # =========================================================================
    
    def get_sector_prices(self, sector: str) -> Optional[pd.DataFrame]:
        """Get prices for a specific sector."""
        if self.prices is None or sector not in self.sector_mapping:
            return None
        
        tickers = [t for t in self.sector_mapping[sector] if t in self.prices.columns]
        return self.prices[tickers] if tickers else None
    
    def get_asset_class_prices(self) -> pd.DataFrame:
        """
        Get aggregated asset class prices.
        Used by Quant Agent for cointegration analysis.
        """
        if self.prices is None:
            return pd.DataFrame()
        
        asset_classes = pd.DataFrame()
        
        # Sector averages
        for sector, tickers in self.sector_mapping.items():
            valid_tickers = [t for t in tickers if t in self.prices.columns]
            if valid_tickers:
                asset_classes[sector] = self.prices[valid_tickers].mean(axis=1)
        
        # Add bonds, commodities
        if 'TLT' in self.prices.columns:
            asset_classes['Bonds'] = self.prices['TLT']
        if 'GLD' in self.prices.columns:
            asset_classes['Commodities'] = self.prices['GLD']
        if '^VIX' in self.prices.columns:
            asset_classes['Volatility'] = self.prices['^VIX']
        
        return asset_classes.dropna()
    
    def refresh_data(self) -> None:
        """Force refresh data from source."""
        cache_path = os.path.join(self.cache_dir, 'market_data.csv')
        if os.path.exists(cache_path):
            os.remove(cache_path)
        self._process()
