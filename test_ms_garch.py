"""
Integration script: Replace simple HMM with MS-GARCH in MLAgent
"""

import sys
sys.path.append('../')

from models.ms_garch import MarkovSwitchingGARCH
import pandas as pd

# This script shows how to integrate MS-GARCH
# It should be imported and used in MLAgent.fit()

def test_ms_garch():
    # Generate sample returns
    returns = pd.DataFrame({
        'AAPL': pd.Series([0.01, -0.02, 0.015, -0.01, 0.02] * 20),
        'MSFT': pd.Series([0.012, -0.018, 0.014, -0.012, 0.019] * 20)
    })
    
    # Fit model
    model = MarkovSwitchingGARCH(n_regimes=3)
    model.fit(returns)
    
    # Get results
    print("Regime Features:")
    print(model.get_regime_features())
    
    print(f"\nCurrent Regime: {model.current_regime}")
    print(f"Transition Matrix:\n{model.transition_matrix}")
    
    # Predict volatility
    vols = model.predict_volatility()
    print(f"\nPredicted Volatilities: {vols}")

if __name__ == "__main__":
    test_ms_garch()
