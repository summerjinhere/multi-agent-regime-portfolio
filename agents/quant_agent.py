from models.hmm_regime import RegimeDetector
from models.ms_vecm import MSVECMAllocator
import pandas as pd
import numpy as np

class QuantAgent:
    def __init__(self):
        self.regime_detector = RegimeDetector(n_components=3)
        self.vecm_allocator = MSVECMAllocator()
        self.current_regime = None
        self.regime_probs = None

    def analyze_market(self, prices):
        """
        Analyzes market data to determine regime and forecast returns.
        """
        print("[QuantAgent] Analyzing market structure...")
        
        # 1. Detect Regime
        market_proxy = prices.mean(axis=1).to_frame(name='Market')
        self.regime_detector.prepare_features(market_proxy, asset_col='Market')
        
        try:
            self.regime_detector.fit()
            self.current_regime = self.regime_detector.get_current_regime()
            self.regime_probs = self.regime_detector.model.predict_proba(self.regime_detector.data.values)[-1]
            print(f"[QuantAgent] Market Regime Detected: {self.current_regime} (Probs: {self.regime_probs})")
        except Exception as e:
            print(f"[QuantAgent] HMM Failed: {e}. Defaulting to Bull.")
            self.current_regime = 0
            self.regime_probs = [1, 0, 0]

        return self.current_regime

    def forecast_returns(self, asset_class_prices):
        """
        Forecasts expected returns and covariance using MS-VECM.
        """
        print(f"[QuantAgent] Forecasting returns for {len(asset_class_prices.columns)} assets...")
        exp_returns, cov_matrix = self.vecm_allocator.fit_predict(asset_class_prices, self.current_regime)
        return exp_returns, cov_matrix
