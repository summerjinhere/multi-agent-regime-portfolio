import pandas as pd
import numpy as np
import yaml
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def cross_validation_analysis():
    print("=" * 60)
    print("TIME-SERIES CROSS-VALIDATION ANALYSIS")
    print("=" * 60)
    
    # Load backtest results
    results = pd.read_csv('backtest_results.csv')
    results['Date'] = pd.to_datetime(results['Date'])
    results.set_index('Date', inplace=True)
    
    print(f"\nTotal sample: {len(results)} months")
    print(f"Period: {results.index[0].date()} to {results.index[-1].date()}")
    
    # Time-series CV with expanding window
    # Start with minimum 12 months training, test on next 3 months
    min_train = 12
    test_window = 3
    
    cv_results = []
    
    for i in range(min_train, len(results) - test_window, test_window):
        train_data = results.iloc[:i]
        test_data = results.iloc[i:i+test_window]
        
        # Calculate training metrics
        train_returns = train_data['Value'].pct_change().dropna()
        train_sharpe = (train_returns.mean() / train_returns.std()) * np.sqrt(12) if train_returns.std() > 0 else 0
        train_vol = train_returns.std() * np.sqrt(12)
        
        # Calculate test metrics
        test_returns = test_data['Value'].pct_change().dropna()
        test_sharpe = (test_returns.mean() / test_returns.std()) * np.sqrt(12) if test_returns.std() > 0 and len(test_returns) > 0 else 0
        test_vol = test_returns.std() * np.sqrt(12) if len(test_returns) > 0 else 0
        
        cv_results.append({
            'fold': len(cv_results) + 1,
            'train_end': train_data.index[-1],
            'test_start': test_data.index[0],
            'test_end': test_data.index[-1],
            'train_sharpe': train_sharpe,
            'test_sharpe': test_sharpe,
            'train_vol': train_vol,
            'test_vol': test_vol
        })
    
    cv_df = pd.DataFrame(cv_results)
    
    print(f"\nNumber of CV folds: {len(cv_df)}")
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 60)
    print(f"\nAverage Training Sharpe: {cv_df['train_sharpe'].mean():.4f}")
    print(f"Average Test Sharpe: {cv_df['test_sharpe'].mean():.4f}")
    print(f"Sharpe Degradation: {(cv_df['train_sharpe'].mean() - cv_df['test_sharpe'].mean()):.4f}")
    print(f"\nAverage Training Volatility: {cv_df['train_vol'].mean():.2%}")
    print(f"Average Test Volatility: {cv_df['test_vol'].mean():.2%}")
    print("=" * 60)
    
    # Save detailed report
    with open('cv_report.md', 'w') as f:
        f.write("# Cross-Validation Report\n\n")
        f.write(f"**Method:** Expanding Window Time-Series CV\n")
        f.write(f"**Min Training Size:** {min_train} months\n")
        f.write(f"**Test Window:** {test_window} months\n")
        f.write(f"**Number of Folds:** {len(cv_df)}\n\n")
        
        f.write("## Summary Statistics\n\n")
        f.write("| Metric | Training | Testing | Degradation |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        f.write(f"| Sharpe Ratio | `{cv_df['train_sharpe'].mean():.4f}` | `{cv_df['test_sharpe'].mean():.4f}` | `{(cv_df['train_sharpe'].mean() - cv_df['test_sharpe'].mean()):.4f}` |\n")
        f.write(f"| Volatility | `{cv_df['train_vol'].mean():.2%}` | `{cv_df['test_vol'].mean():.2%}` | `{(cv_df['train_vol'].mean() - cv_df['test_vol'].mean()):.2%}` |\n\n")
        
        f.write("## Fold Details\n\n")
        f.write("| Fold | Test Period | Train Sharpe | Test Sharpe | Train Vol | Test Vol |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        
        for _, row in cv_df.iterrows():
            f.write(f"| {row['fold']} | {row['test_start'].strftime('%Y-%m')} to {row['test_end'].strftime('%Y-%m')} | ")
            f.write(f"`{row['train_sharpe']:.2f}` | `{row['test_sharpe']:.2f}` | ")
            f.write(f"`{row['train_vol']:.1%}` | `{row['test_vol']:.1%}` |\n")
        
        f.write("\n## Interpretation\n\n")
        if abs(cv_df['test_sharpe'].mean()) < 0.1:
            f.write("> [!WARNING]\n> Out-of-sample Sharpe ratio is near zero, indicating poor predictive performance.\n\n")
        else:
            f.write("> [!NOTE]\n> The model shows some predictive ability in out-of-sample testing.\n\n")
    
    print("\nCV report saved to cv_report.md")
    
    # Save CV results to CSV
    cv_df.to_csv('cv_results.csv', index=False)
    print("Detailed results saved to cv_results.csv")

if __name__ == "__main__":
    cross_validation_analysis()
