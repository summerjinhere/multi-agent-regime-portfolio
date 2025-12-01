import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
import seaborn as sns

class RegimeDetector:
    def __init__(self, n_components=3, covariance_type="full", n_iter=10000):
        """
        Initialize HMM Regime Detector.
        :param n_components: Number of regimes (e.g., 3: Bull, Bear, Stagnation)
        :param covariance_type: Covariance type for HMM ('full', 'diag', 'spherical', 'tied')
        """
        self.n_components = n_components
        self.model = GaussianHMM(n_components=n_components, covariance_type=covariance_type, n_iter=n_iter, random_state=42)
        self.data = None
        self.hidden_states = None

    def prepare_features(self, prices_df, asset_col='^GSPC'):
        """
        Prepare features for HMM. Typically returns and volatility.
        If asset_col is not in df, use the first column or mean of all.
        """
        if asset_col in prices_df.columns:
            series = prices_df[asset_col]
        else:
            # Create a synthetic index if specific ticker not found (e.g. equal weight)
            series = prices_df.mean(axis=1)
        
        returns = series.pct_change().dropna()
        volatility = returns.rolling(window=10).std().dropna()
        
        # Align data
        common_index = returns.index.intersection(volatility.index)
        self.data = pd.DataFrame({
            'Returns': returns.loc[common_index],
            'Volatility': volatility.loc[common_index]
        })
        
        # Standardize features? HMM works better with standardized data sometimes, 
        # but raw returns/vol are interpretable. Let's keep raw for now or standardize internally if needed.
        return self.data

    def fit(self):
        """
        Fit the HMM model.
        """
        if self.data is None:
            raise ValueError("Data not prepared. Call prepare_features() first.")
        
        X = self.data.values
        self.model.fit(X)
        self.hidden_states = self.model.predict(X)
        
        # Order regimes by volatility (0: Low Vol, ..., N: High Vol) to make them interpretable
        # Calculate volatility for each state
        state_vars = []
        for i in range(self.n_components):
            # Assuming 2nd feature is volatility
            # Or just calculate std of returns in that state
            mask = (self.hidden_states == i)
            state_vars.append(self.data['Returns'][mask].std())
        
        # Sort states by volatility
        sorted_indices = np.argsort(state_vars)
        mapping = {old: new for new, old in enumerate(sorted_indices)}
        
        # Remap hidden states
        self.hidden_states = np.array([mapping[s] for s in self.hidden_states])
        
        # Reorder model parameters (means, covars) - This is tricky with hmmlearn, 
        # easier to just remap the output states.
        
        return self.hidden_states

    def predict(self, new_data):
        """
        Predict regime for new data.
        """
        return self.model.predict(new_data)

    def plot_regimes(self, prices_df, asset_col='^GSPC'):
        """
        Plot asset prices colored by regime.
        """
        if self.hidden_states is None:
            print("Model not fitted.")
            return

        if asset_col in prices_df.columns:
            price_series = prices_df[asset_col]
        else:
            price_series = prices_df.mean(axis=1)
            
        # Align dates
        aligned_prices = price_series.loc[self.data.index]
        
        plt.figure(figsize=(15, 8))
        
        # Create a color map
        colors = ['green', 'yellow', 'red'] # Assuming 3 states: Low, Med, High Vol
        if self.n_components > 3:
            colors = sns.color_palette("hsv", self.n_components)
            
        for i in range(self.n_components):
            mask = (self.hidden_states == i)
            plt.plot(aligned_prices.index[mask], aligned_prices[mask], '.', label=f'Regime {i}', color=colors[i] if i < 3 else colors[i], markersize=2)
            
        plt.title('Market Regimes identified by HMM')
        plt.legend()
        plt.show()

    def get_current_regime(self):
        return self.hidden_states[-1] if self.hidden_states is not None else None
