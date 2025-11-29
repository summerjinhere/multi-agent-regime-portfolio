import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
import warnings

class MSVECMAllocator:
    def __init__(self, lookback_window=252):
        self.lookback_window = lookback_window
        self.models = {} # Store models for each regime if needed, or just fit on demand

    def check_cointegration(self, data, det_order=-1, k_ar_diff=1):
        """
        Perform Johansen cointegration test.
        :param data: DataFrame of asset prices (Asset Classes)
        :return: Boolean (True if cointegrated), Rank (r)
        """
        # Johansen test requires strictly positive data (usually logs are used)
        # and no NaNs.
        df_log = np.log(data).dropna()
        
        # det_order: -1 (no deterministic part), 0 (const), 1 (const+trend)
        try:
            jores = coint_johansen(df_log, det_order, k_ar_diff)
            # Trace statistic > Critical Value (95%)
            # r=0 (no cointegration) hypothesis rejected?
            # jores.lr1 is trace statistic, jores.cvt is critical values
            
            # Check if r=0 is rejected (meaning r>=1)
            # 0-th element corresponds to r=0
            is_cointegrated = jores.lr1[0] > jores.cvt[0, 1] # 1 is 95% confidence
            
            # Determine rank r
            r = 0
            for i in range(len(jores.lr1)):
                if jores.lr1[i] > jores.cvt[i, 1]:
                    r += 1
                else:
                    break
            
            return is_cointegrated, r
        except Exception as e:
            print(f"Cointegration test failed: {e}")
            return False, 0

    def fit_predict(self, data, regime, forecast_horizon=5):
        """
        Fit VECM on data corresponding to the current regime (or recent history weighted by regime).
        In a pure MS-VECM, we would filter data by regime.
        However, VECM needs contiguous data for lags. 
        
        Strategy:
        1. Use a rolling window.
        2. If we want "Regime-Dependent", we can:
           a) Filter historical periods that match the current regime to estimate *parameters* (A, B matrices),
              then apply these parameters to the *current* last observations to forecast.
           b) Or simply fit VECM on the recent window, assuming the current regime dominates the recent dynamics.
           
        Approach (a) is closer to true MS-VECM.
        """
        # For simplicity and robustness in this implementation, we will use the recent window 
        # but we could adjust the lookback or weights based on regime stability.
        # A true switching VECM is complex to implement from scratch.
        # We will stick to: Fit VECM on the recent lookback_window.
        
        # Log prices
        df_log = np.log(data.tail(self.lookback_window)).dropna()
        
        is_coint, rank = self.check_cointegration(df_log)
        
        if not is_coint or rank == 0:
            # Fallback to VAR or simple Mean-Variance if not cointegrated
            # print("Not cointegrated, falling back to simple returns mean.")
            returns = data.pct_change().mean() * 252
            cov = data.pct_change().cov() * 252
            return returns, cov
        
        # Fit VECM
        # k_ar_diff = 1 (1 lag in differences = 2 lags in levels)
        try:
            model = VECM(df_log, k_ar_diff=1, coint_rank=rank, deterministic="ci")
            vecm_res = model.fit()
            
            # Forecast
            # Forecast is in Log Levels
            forecast_log = vecm_res.predict(steps=forecast_horizon)
            
            # Convert back to expected returns
            # Expected Price at T+h
            last_price = data.iloc[-1]
            pred_price = np.exp(forecast_log[-1])
            
            # Annualized Expected Return
            expected_ret = (pred_price / last_price) ** (252/forecast_horizon) - 1
            
            # Covariance of residuals (Innovation covariance)
            # This is the covariance of the shocks
            cov_matrix = vecm_res.sigma_u * 252 # Annualize? Sigma_u is per step.
            # VECM covariance is complex, but sigma_u is a good proxy for risk.
            
            # Convert covariance to simple returns space roughly
            # Cov(r) approx Cov(diff log p)
            cov_matrix = pd.DataFrame(cov_matrix, index=data.columns, columns=data.columns) * 252
            
            return expected_ret, cov_matrix
            
        except Exception as e:
            print(f"VECM fitting failed: {e}")
            # Fallback
            returns = data.pct_change().mean() * 252
            cov = data.pct_change().cov() * 252
            return returns, cov
