import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
import seaborn as sns

from analysis import load_backtest_results, fetch_benchmark_data, calculate_metrics, calculate_significance

st.set_page_config(page_title="Portfolio Dashboard", layout="wide")

st.title("Jae ha K Multi-Agent Portfolio Dashboard")

# Sidebar
st.sidebar.header("Configuration")
benchmark_ticker = st.sidebar.text_input("Benchmark Ticker", "SPY")
risk_free_rate = st.sidebar.number_input("Risk-Free Rate", 0.02)

# Load Data
@st.cache_data
def get_data():
    results = load_backtest_results()
    return results

results_df = get_data()

if results_df.empty:
    st.error("No backtest results found. Please run 'main.py --backtest' first.")
    st.stop()

# Calculate Returns
results_df['Returns'] = results_df['Value'].pct_change()
start_date = results_df.index[0]
end_date = results_df.index[-1]

# Fetch Benchmark
with st.spinner("Fetching benchmark data..."):
    benchmark_data = fetch_benchmark_data(start_date, end_date, benchmark_ticker)
    if not benchmark_data.empty:
        benchmark_returns = benchmark_data.pct_change()
    else:
        benchmark_returns = pd.Series(dtype=float)

# --- Main Dashboard ---

# 1. Performance Summary
st.header("1. Performance Summary")

col1, col2, col3, col4 = st.columns(4)

metrics = calculate_metrics(results_df['Returns'].dropna(), risk_free_rate)
bench_metrics = calculate_metrics(benchmark_returns.dropna(), risk_free_rate) if not benchmark_returns.empty else {}

with col1:
    st.metric("Total Return", f"{metrics.get('Total Return', 0):.2%}", 
              delta=f"{metrics.get('Total Return', 0) - bench_metrics.get('Total Return', 0):.2%}" if bench_metrics else None)
with col2:
    st.metric("Sharpe Ratio", f"{metrics.get('Sharpe Ratio', 0):.2f}",
              delta=f"{metrics.get('Sharpe Ratio', 0) - bench_metrics.get('Sharpe Ratio', 0):.2f}" if bench_metrics else None)
with col3:
    st.metric("Max Drawdown", f"{metrics.get('Max Drawdown', 0):.2%}")
with col4:
    st.metric("Volatility", f"{metrics.get('Volatility', 0):.2%}")

# Statistical Significance
if not benchmark_returns.empty:
    stats_sig = calculate_significance(results_df['Returns'].dropna(), benchmark_returns.dropna())
    
    st.subheader("Statistical Significance (vs Benchmark)")
    s_col1, s_col2, s_col3, s_col4 = st.columns(4)
    s_col1.metric("Alpha", f"{stats_sig.get('Alpha', 0):.4f}")
    s_col2.metric("Beta", f"{stats_sig.get('Beta', 0):.2f}")
    s_col3.metric("R-Squared", f"{stats_sig.get('R-Squared', 0):.2f}")
    
    p_val = stats_sig.get('T-Test P-Value', 1.0)
    s_col4.metric("P-Value (T-Test)", f"{p_val:.4f}", 
                  delta="Significant" if p_val < 0.05 else "Not Significant", delta_color="inverse")

# Thesis Validation Metrics
st.header("Thesis Validation Metrics")
from analysis import calculate_probabilistic_sharpe_ratio, calculate_information_ratio, bootstrap_analysis

if not benchmark_returns.empty:
    # Calculate Metrics
    psr = calculate_probabilistic_sharpe_ratio(results_df['Returns'].dropna(), bench_metrics.get('Sharpe Ratio', 0))
    ir = calculate_information_ratio(results_df['Returns'].dropna(), benchmark_returns.dropna())
    bootstrap_res = bootstrap_analysis(results_df['Returns'].dropna())
    
    t_col1, t_col2 = st.columns(2)
    
    with t_col1:
        st.metric("Probabilistic Sharpe Ratio (PSR)", f"{psr:.2%}", 
                 help="Probability that the strategy's Sharpe Ratio is truly greater than the benchmark's.")
        if psr > 0.95:
            st.success("PSR > 95%: Strategy is statistically superior to benchmark with high confidence.")
        else:
            st.warning("PSR < 95%: Statistical superiority is not conclusive.")
            
    with t_col2:
        st.metric("Information Ratio", f"{ir:.2f}",
                 help="Active Return per unit of Tracking Error.")
                 
    st.subheader("Bootstrap Confidence Intervals (95%)")
    if bootstrap_res:
        b_col1, b_col2 = st.columns(2)
        b_col1.info(f"Sharpe Ratio CI: [{bootstrap_res['Sharpe Ratio CI'][0]:.2f}, {bootstrap_res['Sharpe Ratio CI'][1]:.2f}]")
        b_col2.info(f"CAGR CI: [{bootstrap_res['CAGR CI'][0]:.2%}, {bootstrap_res['CAGR CI'][1]:.2%}]")

# Report Generation
if st.button("Generate Thesis Report"):
    with st.spinner("Generating Report..."):
        import subprocess
        try:
            subprocess.run(["python", "generate_report.py"], check=True)
            st.success("Report generated successfully! Check 'thesis_report.md' and 'thesis_report.pdf' (if available).")
            with open("thesis_report.md", "r") as f:
                st.download_button("Download Report (Markdown)", f, file_name="thesis_report.md")
        except Exception as e:
            st.error(f"Failed to generate report: {e}")

# 2. Charts
st.header("2. Performance Charts")

tab1, tab2 = st.tabs(["Cumulative Returns", "Drawdowns"])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=results_df.index, y=results_df['Value'] / results_df['Value'].iloc[0], mode='lines', name='Strategy'))
    
    if not benchmark_data.empty:
        # Rebase benchmark
        bench_aligned = benchmark_data.loc[results_df.index[0]:results_df.index[-1]]
        if not bench_aligned.empty:
            bench_norm = bench_aligned / bench_aligned.iloc[0]
            fig.add_trace(go.Scatter(x=bench_aligned.index, y=bench_norm, mode='lines', name=benchmark_ticker, line=dict(dash='dash')))
            
    fig.update_layout(title="Cumulative Returns (Normalized)", xaxis_title="Date", yaxis_title="Growth of $1")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Calculate drawdown series
    cum_ret = (1 + results_df['Returns']).cumprod()
    run_max = cum_ret.cummax()
    dd = (cum_ret - run_max) / run_max
    
    fig_dd = px.area(x=dd.index, y=dd, title="Drawdown")
    st.plotly_chart(fig_dd, use_container_width=True)

# 3. HMM Visualization
st.header("3. HMM Model Process Visualization")

st.markdown("""
This section visualizes the Hidden Markov Model (HMM) training and regime detection process.
We will retrain the model on the benchmark data to demonstrate the process.
""")

n_components = st.slider("Number of Regimes", 2, 5, 3)
train_model = st.button("Train HMM Model")

if train_model:
    if benchmark_data.empty:
        st.error("No data available for HMM training.")
    else:
        with st.spinner("Training HMM..."):
            # Prepare data
            data = benchmark_data.to_frame(name='Close')
            data['Returns'] = data['Close'].pct_change()
            data['Volatility'] = data['Returns'].rolling(window=10).std()
            data = data.dropna()
            
            X = data[['Returns', 'Volatility']].values
            
            # Fit Model
            model = GaussianHMM(n_components=n_components, covariance_type="full", n_iter=1000, random_state=42)
            model.fit(X)
            
            hidden_states = model.predict(X)
            
            # Reorder states by volatility
            state_vars = []
            for i in range(n_components):
                mask = (hidden_states == i)
                if mask.any():
                    state_vars.append((i, data['Returns'][mask].std()))
                else:
                    state_vars.append((i, 0))
            
            sorted_states = sorted(state_vars, key=lambda x: x[1])
            mapping = {old: new for new, (old, _) in enumerate(sorted_states)}
            hidden_states = np.array([mapping[s] for s in hidden_states])
            
            data['Regime'] = hidden_states
            
            st.success("Model Trained Successfully!")
            
            # Visualizations
            st.subheader("Regime Identification")
            
            # Scatter plot of regimes
            fig_scatter = px.scatter(data, x='Returns', y='Volatility', color=data['Regime'].astype(str),
                                   title="Regime Clusters (Returns vs Volatility)",
                                   color_discrete_sequence=px.colors.qualitative.G10)
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Price colored by regime
            st.subheader("Market Regimes Over Time")
            
            # Create segments for coloring line chart
            # Plotly doesn't support multi-color lines easily, so we use scatter
            fig_regime = px.scatter(data, x=data.index, y='Close', color=data['Regime'].astype(str),
                                  title=f"{benchmark_ticker} Price with Regimes",
                                  color_discrete_sequence=['green', 'orange', 'red', 'purple', 'blue'])
            st.plotly_chart(fig_regime, use_container_width=True)
            
            # Transition Matrix
            st.subheader("Transition Matrix")
            trans_mat = model.transmat_
            # Reorder matrix
            n = n_components
            reordered_trans = np.zeros((n, n))
            # Need inverse mapping for matrix reordering
            inv_mapping = {v: k for k, v in mapping.items()}
            
            for i in range(n):
                for j in range(n):
                    old_i, old_j = inv_mapping[i], inv_mapping[j]
                    reordered_trans[i, j] = trans_mat[old_i, old_j]
            
            fig_heatmap = px.imshow(reordered_trans, 
                                  labels=dict(x="To Regime", y="From Regime", color="Probability"),
                                  x=[f"Regime {i}" for i in range(n)],
                                  y=[f"Regime {i}" for i in range(n)],
                                  text_auto='.2f',
                                  title="Regime Transition Probabilities")
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Component Distributions
            st.subheader("Regime Statistics")
            stats_df = data.groupby('Regime')[['Returns', 'Volatility']].agg(['mean', 'std', 'count'])
            st.dataframe(stats_df)

# 4. Detailed Data
st.header("4. Detailed Data")
with st.expander("View Backtest Data"):
    st.dataframe(results_df)

with st.expander("View Portfolio Weights"):
    try:
        weights = pd.read_csv('final_portfolio_detailed.csv', index_col=0)
        st.dataframe(weights)
        
        fig_pie = px.pie(values=weights.iloc[:, 0], names=weights.index, title="Final Portfolio Allocation")
        st.plotly_chart(fig_pie)
    except:
        st.warning("No detailed weights found.")
