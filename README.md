# HMM + MS-VECM Portfolio Optimization Model

This project implements a "Magnum Opus" portfolio optimization model combining **Hidden Markov Models (HMM)** for market regime detection and **Markov Switching Vector Error Correction Models (MS-VECM)** for dynamic asset allocation.

## Project Structure

- `data_loader.py`: Fetches historical data for 100+ assets (Stocks, Bonds, Commodities, Crypto) using `yfinance`.
- `models/hmm_regime.py`: Implements `RegimeDetector` using Gaussian HMM to identify Bull, Bear, and Crisis regimes.
- `models/ms_vecm.py`: Implements `MSVECMAllocator` to perform cointegration tests and estimate expected returns/covariance based on the current regime.
- `models/portfolio_optimizer.py`: Implements Mean-Variance Optimization and hierarchical portfolio construction.
- `main.py`: The main orchestration script.

## How to Run

1.  **Install Python**: Ensure Python 3.8+ is installed and added to your PATH.
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Model**:
    ```bash
    python main.py
    ```
    Or simply double-click `run.bat`.

## Methodology

1.  **Regime Detection**: The model uses HMM on the aggregate market data to classify the current state into one of 3 regimes (e.g., Low Volatility, High Volatility).
2.  **Asset Class Allocation**: It uses VECM (Vector Error Correction Model) to capture long-term equilibrium relationships between asset classes (Equities, Bonds, Commodities). The parameters are estimated based on the current regime's history.
3.  **Stock Selection**: Within each sector, the model selects the top 5 stocks based on regime-specific factors (e.g., Momentum in Bull markets, Low Volatility in Bear markets).
4.  **Optimization**: The final portfolio weights are calculated using Mean-Variance Optimization on the forecasted returns and covariance matrix.

## Dependencies
- yfinance
- hmmlearn
- statsmodels
- PyPortfolioOpt
- pandas, numpy, matplotlib
