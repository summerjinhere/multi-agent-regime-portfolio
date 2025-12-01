"""
Streamlit Dashboard
====================
Interactive dashboard for portfolio monitoring.

Features:
- Backtest Results View
- Real-time Monitoring View
- Regime Visualization
- Performance Analytics

Usage:
    streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import SystemConfig
from orchestrator import Orchestrator, BacktestResult

# Page config
st.set_page_config(
    page_title="Multi-Agent Portfolio Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 5px;
    }
    .positive { color: #00c853; }
    .negative { color: #ff1744; }
    .stMetric > div { background-color: #f8f9fa; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Helper Functions
# =============================================================================

@st.cache_resource
def get_orchestrator():
    """Get cached orchestrator instance"""
    config = SystemConfig()
    return Orchestrator(config)


@st.cache_data(ttl=3600)
def run_backtest(start_date: str, end_date: str, rebalance_freq: str):
    """Run backtest with caching"""
    orchestrator = get_orchestrator()
    return orchestrator.run_backtest(start_date, end_date, rebalance_freq)


def format_pct(value, decimals=2):
    """Format as percentage"""
    return f"{value*100:.{decimals}f}%"


def format_number(value, decimals=2):
    """Format number"""
    return f"{value:,.{decimals}f}"


# =============================================================================
# Sidebar
# =============================================================================

st.sidebar.title("⚙️ Configuration")

# View selection
view_mode = st.sidebar.radio(
    "Select View",
    ["📈 Backtest Results", "🔴 Real-time Monitor", "📊 Analytics", "🔬 Statistical Validation"]
)

st.sidebar.markdown("---")

# Backtest settings
st.sidebar.subheader("Backtest Settings")

start_date = st.sidebar.date_input(
    "Start Date",
    value=datetime(2005, 1, 1)
).strftime('%Y-%m-%d')

end_date = st.sidebar.date_input(
    "End Date",
    value=datetime(2025, 11, 30)
).strftime('%Y-%m-%d')

rebalance_freq = st.sidebar.selectbox(
    "Rebalance Frequency",
    ["M", "W", "D"],
    format_func=lambda x: {"M": "Monthly", "W": "Weekly", "D": "Daily"}[x]
)

benchmark = st.sidebar.text_input("Benchmark", "SPY")

st.sidebar.markdown("---")

# Run backtest button
if st.sidebar.button("🚀 Run Backtest", type="primary"):
    with st.spinner("Running backtest..."):
        st.session_state['backtest_result'] = run_backtest(start_date, end_date, rebalance_freq)
    st.success("Backtest completed!")


# =============================================================================
# Main Content
# =============================================================================

st.title("🤖 Multi-Agent Portfolio Dashboard")

# Get or run backtest
if 'backtest_result' not in st.session_state:
    with st.spinner("Running initial backtest..."):
        st.session_state['backtest_result'] = run_backtest(start_date, end_date, rebalance_freq)

result = st.session_state['backtest_result']


# =============================================================================
# View: Backtest Results
# =============================================================================

if view_mode == "📈 Backtest Results":
    
    st.header("Backtest Performance Summary")
    
    # Metrics row
    if result.metrics:
        m = result.metrics
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Total Return",
                format_pct(m.get('Total Return', 0)),
                delta=format_pct(m.get('Total Return', 0) - m.get('Benchmark Return', 0))
            )
        
        with col2:
            st.metric(
                "CAGR",
                format_pct(m.get('CAGR', 0)),
                delta=format_pct(m.get('CAGR', 0) - m.get('Benchmark Return', 0))
            )
        
        with col3:
            st.metric(
                "Sharpe Ratio",
                format_number(m.get('Sharpe Ratio', 0)),
                delta=format_number(m.get('Sharpe Ratio', 0) - m.get('Benchmark Sharpe', 0))
            )
        
        with col4:
            st.metric(
                "Max Drawdown",
                format_pct(m.get('Max Drawdown', 0)),
                delta=None
            )
        
        with col5:
            st.metric(
                "Volatility",
                format_pct(m.get('Volatility', 0)),
                delta=None
            )
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Cumulative Returns")
        
        fig = go.Figure()
        
        # Strategy
        if result.cumulative_returns is not None:
            fig.add_trace(go.Scatter(
                x=result.cumulative_returns.index,
                y=result.cumulative_returns.values,
                name='Strategy',
                line=dict(color='blue', width=2)
            ))
        
        # Benchmark
        if result.benchmark_cumulative is not None:
            fig.add_trace(go.Scatter(
                x=result.benchmark_cumulative.index,
                y=result.benchmark_cumulative.values,
                name='Benchmark (SPY)',
                line=dict(color='gray', width=2, dash='dash')
            ))
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Growth of $1",
            hovermode='x unified',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Drawdown")
        
        if result.returns is not None:
            cum_ret = (1 + result.returns).cumprod()
            running_max = cum_ret.cummax()
            drawdown = (cum_ret / running_max - 1) * 100
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                fill='tozeroy',
                fillcolor='rgba(255, 0, 0, 0.3)',
                line=dict(color='red', width=1),
                name='Drawdown'
            ))
            
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Regime analysis
    st.subheader("Regime Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if result.regime_history is not None:
            # Regime distribution
            regime_counts = result.regime_history.value_counts()
            regime_names = {
                0: 'Strong Bull', 
                1: 'Bull', 
                2: 'Neutral', 
                3: 'Bear', 
                4: 'Strong Bear'
            }
            
            fig = px.pie(
                values=regime_counts.values,
                names=[regime_names.get(i, f'Regime {i}') for i in regime_counts.index],
                title="Regime Distribution",
                color_discrete_sequence=['darkgreen', 'green', 'yellow', 'orange', 'red']
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if result.regime_history is not None and result.returns is not None:
            # Returns by regime
            regime_returns = pd.DataFrame({
                'Return': result.returns,
                'Regime': result.regime_history
            }).dropna()
            
            regime_stats = regime_returns.groupby('Regime')['Return'].agg(['mean', 'std', 'count'])
            regime_stats['mean'] = regime_stats['mean'] * 252  # Annualize
            regime_stats['std'] = regime_stats['std'] * np.sqrt(252)
            regime_stats.index = [regime_names.get(i, f'Regime {i}') for i in regime_stats.index]
            
            fig = go.Figure(data=[
                go.Bar(
                    name='Ann. Return',
                    x=regime_stats.index,
                    y=regime_stats['mean'] * 100,
                    marker_color=['darkgreen', 'green', 'yellow', 'orange', 'red'][:len(regime_stats)]
                )
            ])
            
            fig.update_layout(
                title="Annualized Returns by Regime",
                yaxis_title="Return (%)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Asset Allocation
    st.subheader("Asset Allocation")
    
    if result.weights_history is not None and not result.weights_history.empty:
        # Stacked area chart of weights
        weights_df = result.weights_history
        
        # Filter out assets that never had significant weight
        max_weights = weights_df.max()
        significant_assets = max_weights[max_weights > 0.01].index
        weights_df = weights_df[significant_assets]
        
        fig = px.area(
            weights_df,
            title="Portfolio Weights Over Time",
            labels={"value": "Weight", "index": "Date", "variable": "Asset"}
        )
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
        
        # Current Allocation Pie Chart
        if len(weights_df) > 0:
            current_weights = weights_df.iloc[-1]
            current_weights = current_weights[current_weights > 0.001]  # Filter small dust
            
            fig_pie = px.pie(
                values=current_weights.values,
                names=current_weights.index,
                title=f"Current Allocation ({weights_df.index[-1].date()})"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
    # Detailed metrics table
    st.subheader("Detailed Metrics")
    
    if result.metrics:
        metrics_df = pd.DataFrame([result.metrics]).T
        metrics_df.columns = ['Value']
        
        # Format values
        for idx in metrics_df.index:
            val = metrics_df.loc[idx, 'Value']
            if 'Return' in idx or 'Drawdown' in idx or 'VaR' in idx or 'CVaR' in idx or 'Rate' in idx or 'Alpha' in idx:
                metrics_df.loc[idx, 'Value'] = format_pct(val)
            elif 'Ratio' in idx or 'Beta' in idx:
                metrics_df.loc[idx, 'Value'] = format_number(val)
            elif 'N Trades' in idx:
                metrics_df.loc[idx, 'Value'] = int(val)
        
        st.dataframe(metrics_df, use_container_width=True)


# =============================================================================
# View: Real-time Monitor
# =============================================================================

elif view_mode == "🔴 Real-time Monitor":
    
    st.header("Real-time Portfolio Monitor")
    
    # Status indicator
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 20px; background-color: #d4edda; border-radius: 10px;'>
            <h2>🟢 System Online</h2>
            <p>Last Update: {}</p>
        </div>
        """.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Current regime
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Current Regime")
        
        if result.regime_history is not None and len(result.regime_history) > 0:
            current_regime = result.regime_history.iloc[-1]
            regime_names = {
                0: ('Strong Bull 🟢🟢', 'darkgreen'), 
                1: ('Bull 🟢', 'green'), 
                2: ('Neutral 🟡', 'orange'), 
                3: ('Bear 🔴', 'orangered'), 
                4: ('Strong Bear 🔴🔴', 'red')
            }
            name, color = regime_names.get(current_regime, ('Unknown', 'gray'))
            
            st.markdown(f"""
            <div style='text-align: center; padding: 30px; background-color: {color}; color: white; border-radius: 10px; font-size: 24px;'>
                {name}
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("Portfolio Value")
        
        if result.portfolio_values is not None and len(result.portfolio_values) > 0:
            current_value = result.portfolio_values.iloc[-1]
            prev_value = result.portfolio_values.iloc[-2] if len(result.portfolio_values) > 1 else current_value
            change = (current_value / prev_value - 1) * 100
            
            st.metric(
                "Current Value",
                f"${current_value:,.0f}",
                delta=f"{change:.2f}%"
            )
    
    with col3:
        st.subheader("Risk Level")
        
        if result.metrics:
            var = abs(result.metrics.get('VaR (95%)', 0.02))
            risk_level = "Low 🟢" if var < 0.02 else "Medium 🟡" if var < 0.04 else "High 🔴"
            
            st.metric(
                "Daily VaR (95%)",
                format_pct(var),
                delta=risk_level
            )
    
    st.markdown("---")
    
    # Live chart (simulated)
    st.subheader("Live Price Chart")
    
    if result.portfolio_values is not None:
        # Show last 30 days
        recent = result.portfolio_values.tail(30)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recent.index,
            y=recent.values,
            mode='lines+markers',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ))
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Auto-refresh option
    if st.checkbox("Auto-refresh (every 60 seconds)"):
        st.markdown("⏱️ Auto-refresh enabled")
        import time
        time.sleep(60)
        st.rerun()

    # Regime History Chart
    st.subheader("Regime History")
    if result.regime_history is not None and not result.regime_history.empty:
        # Create a step chart for regime
        regime_df = pd.DataFrame({'Regime': result.regime_history})
        regime_df['Regime Label'] = regime_df['Regime'].map({
            0: 'Strong Bull', 
            1: 'Bull', 
            2: 'Neutral', 
            3: 'Bear', 
            4: 'Strong Bear'
        })
        
        fig_regime = px.line(
            regime_df, 
            y='Regime', 
            title="Market Regime Over Time",
            labels={'index': 'Date', 'Regime': 'Regime (0=S.Bull, 4=S.Bear)'}
        )
        fig_regime.update_traces(line_shape='hv') # Step chart
        fig_regime.update_yaxes(
            ticktext=['S.Bull', 'Bull', 'Neutral', 'Bear', 'S.Bear'], 
            tickvals=[0, 1, 2, 3, 4]
        )
        st.plotly_chart(fig_regime, use_container_width=True)

    # =========================================================================
    # Live vs Backtest Comparison
    # =========================================================================
    st.markdown("---")
    st.header("Live vs Backtest Comparison")
    
    live_file = './results/live_performance.csv'
    if os.path.exists(live_file):
        try:
            live_df = pd.read_csv(live_file, index_col=0, parse_dates=True)
            
            if not live_df.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Live Performance")
                    
                    # Calculate live metrics
                    live_ret = (live_df['Portfolio Value'].iloc[-1] / live_df['Portfolio Value'].iloc[0] - 1)
                    live_duration = (live_df.index[-1] - live_df.index[0]).total_seconds() / 3600 / 24 # Days
                    
                    st.metric("Live Return", format_pct(live_ret))
                    st.metric("Duration (Days)", f"{live_duration:.4f}")
                    
                    # Plot Live Value
                    fig_live = px.line(live_df, y='Portfolio Value', title="Live Portfolio Value")
                    st.plotly_chart(fig_live, use_container_width=True)
                
                with col2:
                    st.subheader("Comparison")
                    
                    if result.metrics:
                        backtest_ret = result.metrics.get('Total Return', 0)
                        backtest_cagr = result.metrics.get('CAGR', 0)
                        
                        comp_df = pd.DataFrame({
                            'Metric': ['Total Return', 'CAGR (Ann.)'],
                            'Backtest': [format_pct(backtest_ret), format_pct(backtest_cagr)],
                            'Live (Proj.)': [format_pct(live_ret), format_pct(live_ret * 365 / live_duration if live_duration > 0 else 0)]
                        })
                        st.table(comp_df)
                    else:
                        st.info("Run backtest to see comparison.")
            else:
                st.warning("Live data file is empty.")
        except Exception as e:
            st.error(f"Error loading live data: {e}")
    else:
        st.info("No live data found. Run with --live to generate.")


# =============================================================================
# View: Analytics
# =============================================================================

# =============================================================================
# View: Analytics
# =============================================================================

elif view_mode == "📊 Analytics":
    
    st.header("Performance Analytics")
    
    # Monthly returns heatmap
    st.subheader("Monthly Returns Heatmap")
    
    if result.returns is not None:
        monthly = result.returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Create monthly matrix
        monthly_df = pd.DataFrame({
            'Return': monthly,
            'Year': monthly.index.year,
            'Month': monthly.index.month
        })
        
        pivot = monthly_df.pivot(index='Year', columns='Month', values='Return')
        pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        fig = px.imshow(
            pivot * 100,
            labels=dict(x="Month", y="Year", color="Return (%)"),
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0,
            text_auto='.1f'
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Rolling metrics
    st.subheader("Rolling Metrics (252-day)")
    
    if result.returns is not None:
        rolling_return = result.returns.rolling(252).mean() * 252
        rolling_vol = result.returns.rolling(252).std() * np.sqrt(252)
        rolling_sharpe = rolling_return / rolling_vol
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Rolling Return', 'Rolling Volatility', 'Rolling Sharpe'),
            shared_xaxes=True
        )
        
        fig.add_trace(
            go.Scatter(x=rolling_return.index, y=rolling_return.values * 100, name='Return'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=rolling_vol.index, y=rolling_vol.values * 100, name='Volatility'),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values, name='Sharpe'),
            row=3, col=1
        )
        
        # Add horizontal line at Sharpe = 1
        fig.add_hline(y=1, line_dash="dash", line_color="gray", row=3, col=1)
        
        fig.update_layout(height=800, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Return distribution
    st.subheader("Return Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if result.returns is not None:
            fig = px.histogram(
                result.returns * 100,
                nbins=50,
                title="Daily Return Distribution"
            )
            fig.update_layout(xaxis_title="Return (%)", yaxis_title="Frequency")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if result.returns is not None:
            # QQ Plot
            from scipy import stats
            
            returns_sorted = np.sort(result.returns.dropna())
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(returns_sorted)))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=theoretical_quantiles,
                y=returns_sorted * 100,
                mode='markers',
                name='Returns'
            ))
            
            # Add 45-degree line
            min_val = min(theoretical_quantiles.min(), (returns_sorted * 100).min())
            max_val = max(theoretical_quantiles.max(), (returns_sorted * 100).max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Normal'
            ))
            
            fig.update_layout(
                title="Q-Q Plot",
                xaxis_title="Theoretical Quantiles",
                yaxis_title="Sample Quantiles (%)"
            )
            st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# View: Statistical Validation
# =============================================================================

elif view_mode == "🔬 Statistical Validation":
    
    st.header("Statistical Validation")
    
    from analysis.statistics import run_ols, run_bootstrap
    
    if result.returns is not None and result.benchmark_returns is not None:
        
        # 1. OLS Regression
        st.subheader("1. OLS Regression (Alpha/Beta)")
        st.markdown("Regression of Strategy Returns vs Benchmark Returns using **Newey-West HAC Robust Standard Errors**.")
        
        ols_res = run_ols(result.returns, result.benchmark_returns)
        
        if "error" not in ols_res:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Alpha (Ann.)", format_pct(ols_res['alpha'] * 252), 
                         help=f"t-stat: {ols_res['alpha_tstat']:.2f}, p-value: {ols_res['alpha_pvalue']:.4f}")
            with col2:
                st.metric("Beta", f"{ols_res['beta']:.2f}",
                         help=f"t-stat: {ols_res['beta_tstat']:.2f}, p-value: {ols_res['beta_pvalue']:.4f}")
            with col3:
                st.metric("R-Squared", f"{ols_res['r_squared']:.4f}")
            with col4:
                is_sig = ols_res['alpha_pvalue'] < 0.05
                st.metric("Alpha Significant?", "✅ Yes" if is_sig else "❌ No", 
                         help="At 95% confidence level")
            
            with st.expander("View Full OLS Summary"):
                st.text(ols_res['summary'])
        else:
            st.error(ols_res['error'])
            
        st.markdown("---")
        
        # 2. Rolling Alpha Analysis
        st.subheader("2. Rolling Alpha (252-day)")
        st.markdown("Time-varying Alpha generation. Green zones indicate statistically significant positive alpha (t > 1.96).")
        
        from analysis.statistics import run_rolling_ols
        rolling_res = run_rolling_ols(result.returns, result.benchmark_returns)
        
        if not rolling_res.empty:
            # Create figure with secondary y-axis
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Alpha Line
            fig.add_trace(
                go.Scatter(x=rolling_res.index, y=rolling_res['Alpha'], name="Rolling Alpha (Ann.)", line=dict(color='blue')),
                secondary_y=False
            )
            
            # Beta Line
            fig.add_trace(
                go.Scatter(x=rolling_res.index, y=rolling_res['Beta'], name="Rolling Beta", line=dict(color='gray', dash='dot')),
                secondary_y=True
            )
            
            # Highlight significant periods
            # We can't easily do background color in plotly for time series without shapes, 
            # so let's add markers or a separate trace for significant alpha
            sig_alpha = rolling_res[rolling_res['Alpha_t'] > 1.96]['Alpha']
            if not sig_alpha.empty:
                fig.add_trace(
                    go.Scatter(x=sig_alpha.index, y=sig_alpha, mode='markers', name="Significant Alpha", marker=dict(color='green', size=4)),
                    secondary_y=False
                )
            
            fig.update_layout(title="Rolling Alpha & Beta")
            fig.update_yaxes(title_text="Alpha", secondary_y=False)
            fig.update_yaxes(title_text="Beta", secondary_y=True)
            
            st.plotly_chart(fig, use_container_width=True)
            
        st.markdown("---")
        
        # 2. Bootstrap Analysis
        st.subheader("2. Bootstrap Analysis (Sharpe Ratio)")
        st.markdown("Resampling returns 1000 times to estimate the distribution of the Sharpe Ratio.")
        
        if st.button("Run Bootstrap Analysis"):
            with st.spinner("Running bootstrap simulation..."):
                boot_res = run_bootstrap(result.returns, n_samples=1000)
                
                if "error" not in boot_res:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Observed Sharpe", f"{boot_res['observed_sharpe']:.2f}")
                    with col2:
                        st.metric("Bootstrap Mean", f"{boot_res['bootstrap_mean']:.2f}")
                    with col3:
                        st.metric("Prob(Sharpe <= 0)", format_pct(boot_res['p_value_zero']),
                                 help="Probability that the strategy has no skill (Sharpe <= 0)")
                    
                    # Histogram
                    fig = px.histogram(
                        boot_res['distribution'], 
                        nbins=50, 
                        title="Bootstrap Distribution of Sharpe Ratio",
                        labels={'value': 'Sharpe Ratio'}
                    )
                    
                    # Add observed line
                    fig.add_vline(x=boot_res['observed_sharpe'], line_dash="solid", line_color="red", annotation_text="Observed")
                    
                    # Add CI lines
                    fig.add_vline(x=boot_res['ci_lower'], line_dash="dash", line_color="green", annotation_text="2.5%")
                    fig.add_vline(x=boot_res['ci_upper'], line_dash="dash", line_color="green", annotation_text="97.5%")
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.info(f"95% Confidence Interval: [{boot_res['ci_lower']:.2f}, {boot_res['ci_upper']:.2f}]")
                else:
                    st.error(boot_res['error'])
    else:
        st.warning("Need both Strategy and Benchmark returns for statistical analysis.")


# =============================================================================
# Footer
# =============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    Multi-Agent Portfolio System v3.0 | Built with Streamlit
</div>
""", unsafe_allow_html=True)
