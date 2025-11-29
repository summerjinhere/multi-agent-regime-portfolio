import pandas as pd
import numpy as np

def calculate_metrics():
    try:
        df = pd.read_csv('backtest_results.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # Calculate returns
        df['Return'] = df['Value'].pct_change().fillna(0)
        
        # Total Return
        start_val = df['Value'].iloc[0]
        end_val = df['Value'].iloc[-1]
        total_return = (end_val - start_val) / start_val
        
        # CAGR (Annualized Return)
        days = (df.index[-1] - df.index[0]).days
        cagr = (end_val / start_val) ** (365.25 / days) - 1
        
        # Volatility (Annualized for Monthly Data)
        volatility = df['Return'].std() * np.sqrt(12)
        
        # Sharpe Ratio (Risk Free Rate = 2%)
        rf = 0.02
        sharpe = (cagr - rf) / volatility if volatility != 0 else 0
        
        # Max Drawdown
        rolling_max = df['Value'].cummax()
        drawdown = (df['Value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        print("-" * 30)
        print("BACKTEST PERFORMANCE SUMMARY")
        print("-" * 30)
        print(f"Period: {df.index[0].date()} to {df.index[-1].date()}")
        print(f"Total Return: {total_return:.2%}")
        print(f"CAGR: {cagr:.2%}")
        print(f"Volatility: {volatility:.2%}")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2%}")
        print("-" * 30)
        
    except Exception as e:
        print(f"Error calculating metrics: {e}")

if __name__ == "__main__":
    calculate_metrics()
