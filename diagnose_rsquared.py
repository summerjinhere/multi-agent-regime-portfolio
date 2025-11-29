import pandas as pd
import numpy as np

def diagnose():
    print("=" * 50)
    print("DIAGNOSING LOW R-SQUARED")
    print("=" * 50)
    
    # Load portfolio
    portfolio = pd.read_csv('final_portfolio.csv', header=None, names=['Asset', 'Weight'])
    portfolio = portfolio[portfolio['Weight'] > 0.001]
    
    print("\nCurrent Portfolio Allocation:")
    print("-" * 50)
    for _, row in portfolio.iterrows():
        print(f"{row['Asset']:30s} {row['Weight']*100:6.2f}%")
    
    # Analyze
    cash_pct = portfolio[portfolio['Asset'] == 'Cash']['Weight'].sum() if 'Cash' in portfolio['Asset'].values else 0
    total_equity = portfolio[portfolio['Asset'].isin(['Technology', 'Healthcare', 'Financials', 'Consumer_Discretionary', 'Energy'])]['Weight'].sum()
    total_bonds = portfolio[portfolio['Asset'] == 'Bonds']['Weight'].sum() if 'Bonds' in portfolio['Asset'].values else 0
    total_commodities = portfolio[portfolio['Asset'] == 'Commodities']['Weight'].sum() if 'Commodities' in portfolio['Asset'].values else 0
    total_volatility = portfolio[portfolio['Asset'] == 'Volatility']['Weight'].sum() if 'Volatility' in portfolio['Asset'].values else 0
    
    print("\n" + "=" * 50)
    print("ASSET CLASS BREAKDOWN:")
    print("=" * 50)
    print(f"Cash:        {cash_pct*100:6.2f}%")
    print(f"Equities:    {total_equity*100:6.2f}%")
    print(f"Bonds:       {total_bonds*100:6.2f}%")
    print(f"Commodities: {total_commodities*100:6.2f}%")
    print(f"Volatility:  {total_volatility*100:6.2f}%")
    
    print("\n" + "=" * 50)
    print("DIAGNOSIS:")
    print("=" * 50)
    
    if cash_pct > 0.15:
        print(f"⚠️  HIGH CASH ALLOCATION ({cash_pct*100:.1f}%)")
        print("   -> Reduces market correlation (low R-squared)")
        print("   -> Strategy behaves like risk-free asset")
        
    if total_equity < 0.5:
        print(f"⚠️  LOW EQUITY EXPOSURE ({total_equity*100:.1f}%)")
        print("   -> Insufficient market participation")
        print("   -> R-squared will be low vs SPY")
        
    print("\n" + "=" * 50)
    print("RECOMMENDATIONS:")
    print("=" * 50)
    print("1. Reduce cash allocation to <10%")
    print("2. Increase equity exposure to >60%")
    print("3. Adjust risk_budget parameter in RiskManagerAgent")
    print("4. Review circuit breaker activation in backtest")

if __name__ == "__main__":
    diagnose()
