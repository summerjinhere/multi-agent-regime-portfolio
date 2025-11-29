import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class Visualizer:
    def __init__(self, output_dir="reports"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Set style
        sns.set_theme(style="darkgrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.family'] = 'sans-serif'

    def plot_backtest(self, strategy_series, benchmark_series=None):
        """
        Plot Cumulative Returns and Drawdown.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Cumulative Returns
        ax1.plot(strategy_series.index, strategy_series, label='Strategy', color='#00ff41', linewidth=2) # Matrix Green
        if benchmark_series is not None:
            # Normalize benchmark to strategy start
            benchmark_series = benchmark_series / benchmark_series.iloc[0] * strategy_series.iloc[0]
            ax1.plot(benchmark_series.index, benchmark_series, label='Benchmark (SPY)', color='gray', alpha=0.6)
            
        ax1.set_title('Cumulative Performance (Log Scale)', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.set_yscale('log')
        ax1.legend()
        
        
        # Drawdown
        drawdown = strategy_series / strategy_series.cummax() - 1
        ax2.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        ax2.plot(drawdown.index, drawdown, color='red', linewidth=1)
        ax2.set_title('Drawdown', fontsize=14)
        ax2.set_ylabel('% Drawdown')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'backtest_performance.png'))
        plt.close()

    def plot_regimes(self, prices, regimes, date_col='Date'):
        """
        Plot Asset Price colored by Regime.
        """
        # Align data
        # regimes is a list of dicts or a dataframe with Date and Regime
        if isinstance(regimes, list):
            regime_df = pd.DataFrame(regimes).set_index('Date')
        else:
            regime_df = regimes
            
        # Use a market proxy (e.g. mean of prices)
        market = prices.mean(axis=1)
        
        # Align
        common_idx = market.index.intersection(regime_df.index)
        market = market.loc[common_idx]
        regime_vals = regime_df.loc[common_idx, 'Regime']
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Create colored segments
        # 0: Bull (Green), 1: Transition (Yellow), 2: Bear (Red)
        colors = ['#00ff41', '#ffd700', '#ff0000'] 
        labels = ['Bull/Low Vol', 'Transition', 'Bear/High Vol']
        
        for i in range(3):
            mask = (regime_vals == i)
            if mask.any():
                ax.scatter(market.index[mask], market[mask], c=colors[i], label=labels[i], s=10, alpha=0.6)
        
        ax.set_title('Market Regimes Identification', fontsize=16, fontweight='bold')
        ax.set_ylabel('Market Index Level')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'regime_analysis.png'))
        plt.close()

    def plot_allocation(self, weights):
        """
        Plot Current Portfolio Allocation as a Donut Chart.
        """
        # Filter zero weights
        weights = {k: v for k, v in weights.items() if v > 0.01}
        
        labels = list(weights.keys())
        sizes = list(weights.values())
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Colors
        colors = sns.color_palette("husl", len(labels))
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, pctdistance=0.85, textprops={'fontsize': 12})
        
        # Draw circle
        centre_circle = plt.Circle((0,0),0.70,fc='white')
        fig.gca().add_artist(centre_circle)
        
        ax.set_title('Optimal Portfolio Allocation', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'portfolio_allocation.png'))
        plt.close()

    def plot_regime_stability(self, regime_history):
        """
        Plot Stacked Area Chart of Regime Probabilities over time.
        """
        df = pd.DataFrame(regime_history).set_index('Date')
        if 'Prob_Bull' not in df.columns:
            return
            
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.stackplot(df.index, 
                     df['Prob_Bull'], df['Prob_Trans'], df['Prob_Bear'],
                     labels=['Bull (Low Vol)', 'Transition', 'Bear (High Vol)'],
                     colors=['#00ff41', '#ffd700', '#ff0000'], alpha=0.7)
        
        ax.set_title('Regime Probability Evolution (Model Confidence)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Probability')
        ax.legend(loc='upper left')
        ax.set_xlim(df.index[0], df.index[-1])
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'regime_stability.png'))
        plt.close()

    def plot_rolling_metrics(self, strategy_series, window=6):
        """
        Plot Rolling Sharpe and Volatility.
        """
        returns = strategy_series.pct_change().dropna()
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(12) # Annualized
        rolling_mean = returns.rolling(window=window).mean() * 12
        rolling_sharpe = rolling_mean / rolling_vol
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        ax1.plot(rolling_sharpe.index, rolling_sharpe, color='purple', label=f'{window}-Month Rolling Sharpe')
        ax1.axhline(0, color='black', linestyle='--', linewidth=0.8)
        ax1.set_title('Rolling Sharpe Ratio (Stability Test)', fontsize=14)
        ax1.legend()
        
        ax2.plot(rolling_vol.index, rolling_vol, color='orange', label=f'{window}-Month Rolling Volatility')
        ax2.set_title('Rolling Volatility (Risk Regime)', fontsize=14)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'rolling_metrics.png'))
        plt.close()

    def generate_html_report(self, metrics):
        """
        Generate a simple HTML report embedding the images.
        """
        html = f"""
        <html>
        <head>
            <title>Magnum Opus Portfolio Report</title>
            <style>
                body {{ font-family: 'Segoe UI', sans-serif; background-color: #f4f4f9; color: #333; margin: 0; padding: 20px; }}
                .container {{ max_width: 1200px; margin: 0 auto; background: white; padding: 40px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); border-radius: 10px; }}
                h1 {{ color: #2c3e50; text-align: center; border-bottom: 2px solid #eee; padding-bottom: 20px; }}
                h2 {{ color: #34495e; margin-top: 40px; }}
                .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 40px; }}
                .metric-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; border-left: 5px solid #3498db; }}
                .metric-val {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
                .metric-label {{ font-size: 14px; color: #7f8c8d; }}
                img {{ max_width: 100%; height: auto; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .footer {{ text-align: center; margin-top: 50px; color: #95a5a6; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Magnum Opus Portfolio Optimization Model</h1>
                
                <h2>Performance Metrics</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-val">{metrics.get('Total Return', 0):.2%}</div>
                        <div class="metric-label">Total Return</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-val">{metrics.get('Annualized Return', 0):.2%}</div>
                        <div class="metric-label">CAGR</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-val">{metrics.get('Sharpe Ratio', 0):.2f}</div>
                        <div class="metric-label">Sharpe Ratio</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-val">{metrics.get('Max Drawdown', 0):.2%}</div>
                        <div class="metric-label">Max Drawdown</div>
                    </div>
                </div>

                <h2>1. Backtest Performance</h2>
                <p>Cumulative returns and drawdown analysis (2020 - Present).</p>
                <img src="backtest_performance.png" alt="Backtest Performance">

                <h2>2. Model Diagnostics (Process Visualization)</h2>
                <p><strong>Regime Stability:</strong> Evolution of HMM state probabilities over time. Shows how the model "changes its mind" about the market.</p>
                <img src="regime_stability.png" alt="Regime Stability">
                
                <p><strong>Rolling Metrics:</strong> 6-Month Rolling Sharpe and Volatility. Demonstrates consistency of performance.</p>
                <img src="rolling_metrics.png" alt="Rolling Metrics">

                <h2>3. Market Regimes Overlay</h2>
                <p>HMM-detected market regimes overlaid on the market index.</p>
                <img src="regime_analysis.png" alt="Regime Analysis">

                <h2>4. Current Allocation</h2>
                <p>Optimal asset allocation for the current market regime.</p>
                <img src="portfolio_allocation.png" alt="Portfolio Allocation">
                
                <div class="footer">
                    Generated by Antigravity AI | HMM + MS-VECM + Black-Litterman
                </div>
            </div>
        </body>
        </html>
        """
        with open(os.path.join(self.output_dir, 'report.html'), 'w') as f:
            f.write(html)
        print(f"Report generated at {os.path.join(self.output_dir, 'report.html')}")
