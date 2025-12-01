import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Import our system
from main import MultiAgentOrchestrator, EventType
from config.settings import SystemConfig

# Page config
st.set_page_config(
    page_title="Multi-Agent Portfolio System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .regime-bull { color: #00cc96; font-weight: bold; }
    .regime-bear { color: #ef553b; font-weight: bold; }
    .regime-crisis { color: #ab63fa; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def get_orchestrator():
    """Initialize the orchestrator once"""
    config = SystemConfig()
    return MultiAgentOrchestrator(config)

def main():
    st.title("🤖 Multi-Agent Portfolio System")
    
    # Initialize system
    orchestrator = get_orchestrator()
    
    # Sidebar controls
    st.sidebar.header("Control Panel")
    is_running = st.sidebar.toggle("Run Live Simulation", value=False)
    refresh_rate = st.sidebar.slider("Refresh Rate (s)", 1, 10, 2)
    
    # Initialize session state for history if not exists
    if 'history' not in st.session_state:
        st.session_state.history = {
            'timestamps': [],
            'values': [],
            'regimes': []
        }
        # Initialize portfolio value if not set
        if not hasattr(orchestrator, 'portfolio_value_history'):
             orchestrator.portfolio_value = orchestrator.config.initial_capital
    
    # Main layout
    col1, col2, col3, col4 = st.columns(4)
    
    # Placeholders for metrics
    with col1:
        metric_value = st.empty()
    with col2:
        metric_regime = st.empty()
    with col3:
        metric_risk = st.empty()
    with col4:
        metric_cash = st.empty()
        
    # Charts layout
    tab1, tab2, tab3 = st.tabs(["Portfolio Performance", "Asset Allocation", "Risk & Alerts"])
    
    with tab1:
        chart_performance = st.empty()
    
    with tab2:
        col_alloc_1, col_alloc_2 = st.columns(2)
        with col_alloc_1:
            chart_allocation = st.empty()
        with col_alloc_2:
            table_positions = st.empty()
            
    with tab3:
        container_alerts = st.container()
    
    # Simulation Loop
    if is_running:
        # Simulate market data
        market_return = np.random.normal(0.0005, 0.01)
        market_vol = 0.15 + np.random.normal(0, 0.02)
        
        # Update Agents
        regime_event = orchestrator.ml_agent.update(
            current_return=market_return,
            current_volatility=market_vol
        )
        
        regime = regime_event.data['regime']
        probs = regime_event.data['probabilities']
        
        # Update Portfolio Value
        orchestrator.portfolio_value *= (1 + market_return)
        
        # Update Risk
        risk_event = orchestrator.risk_agent.update(
            portfolio_value=orchestrator.portfolio_value,
            weights=orchestrator.portfolio_agent.current_weights,
            covariance=np.eye(len(orchestrator.config.universe)), # Dummy cov for viz
            regime=regime
        )
        
        # Update History
        st.session_state.history['timestamps'].append(datetime.now())
        st.session_state.history['values'].append(orchestrator.portfolio_value)
        st.session_state.history['regimes'].append(regime.name)
        
        # Keep history manageable
        if len(st.session_state.history['timestamps']) > 100:
            st.session_state.history['timestamps'].pop(0)
            st.session_state.history['values'].pop(0)
            st.session_state.history['regimes'].pop(0)
            
        # --- RENDER UPDATES ---
        
        # 1. Metrics
        metric_value.metric("Portfolio Value", f"${orchestrator.portfolio_value:,.2f}", 
                           f"{market_return:.2%}")
        
        regime_color = "normal"
        if "BULL" in regime.name: regime_color = "regime-bull"
        elif "BEAR" in regime.name: regime_color = "regime-bear"
        elif "CRISIS" in regime.name: regime_color = "regime-crisis"
        
        metric_regime.markdown(f"**Regime**<br><span class='{regime_color}' style='font-size: 24px'>{regime.name}</span>", unsafe_allow_html=True)
        
        metric_risk.metric("Risk Level", f"{regime_event.data['risk_level']:.2f}")
        
        # 2. Performance Chart
        df_hist = pd.DataFrame({
            'Time': st.session_state.history['timestamps'],
            'Value': st.session_state.history['values'],
            'Regime': st.session_state.history['regimes']
        })
        
        fig_perf = px.line(df_hist, x='Time', y='Value', title='Portfolio Value Over Time')
        fig_perf.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
        chart_performance.plotly_chart(fig_perf, use_container_width=True)
        
        # 3. Allocation
        weights = orchestrator.portfolio_agent.current_weights
        assets = orchestrator.config.universe
        
        # Filter small positions for pie chart
        df_alloc = pd.DataFrame({'Asset': assets, 'Weight': weights})
        df_alloc = df_alloc[df_alloc['Weight'] > 0.01]
        
        fig_pie = px.pie(df_alloc, values='Weight', names='Asset', title='Current Allocation')
        fig_pie.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
        chart_allocation.plotly_chart(fig_pie, use_container_width=True)
        
        # 4. Positions Table
        df_pos = pd.DataFrame({
            'Asset': assets,
            'Weight': [f"{w:.1%}" for w in weights],
            'Value': [f"${w * orchestrator.portfolio_value:,.0f}" for w in weights]
        })
        table_positions.dataframe(df_pos, height=350, use_container_width=True)
        
        # 5. Alerts
        with container_alerts:
            st.subheader("System Alerts")
            if risk_event.event_type == EventType.RISK_ALERT:
                for alert in risk_event.data['alerts']:
                    if alert['severity'] == 'HIGH':
                        st.error(f"🚨 {alert['message']}")
                    else:
                        st.warning(f"⚠️ {alert['message']}")
            else:
                st.info("No active risk alerts. System nominal.")
                
            st.markdown("### Regime Probabilities")
            st.bar_chart(probs)

        time.sleep(refresh_rate)
        st.rerun()

    else:
        st.info("Click 'Run Live Simulation' in the sidebar to start the dashboard.")
        
        # Show static initial state
        metric_value.metric("Portfolio Value", f"${orchestrator.portfolio_value:,.2f}")
        metric_regime.metric("Regime", "WAITING")

if __name__ == "__main__":
    main()
