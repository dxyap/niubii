"""
Risk Management Page
====================
Portfolio risk monitoring and analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.risk import VaRCalculator, RiskLimits, RiskMonitor

st.set_page_config(page_title="Risk Management | Oil Trading", page_icon="🛡️", layout="wide")

# Initialize components
@st.cache_resource
def get_components():
    var_calc = VaRCalculator(confidence_level=0.95, holding_period=1)
    risk_limits = RiskLimits(config_path=str(project_root / "config" / "risk_limits.yaml"))
    risk_monitor = RiskMonitor()
    return var_calc, risk_limits, risk_monitor

var_calc, risk_limits, risk_monitor = get_components()

st.title("🛡️ Risk Management")
st.caption("Portfolio risk monitoring and stress testing")

# Mock positions for demonstration
mock_positions = {
    "CL1 Comdty": {"quantity": 45, "price": 73.45},
    "CL2 Comdty": {"quantity": 20, "price": 73.20},
    "CO1 Comdty": {"quantity": -15, "price": 77.80},
    "XB1 Comdty": {"quantity": 8, "price": 2.22},
    "HO1 Comdty": {"quantity": 5, "price": 2.52},
}

# Top metrics row
col1, col2, col3, col4, col5 = st.columns(5)

var_value = 245000
var_limit = 375000
var_util = var_value / var_limit * 100

with col1:
    st.metric(
        "Portfolio VaR (95%, 1-Day)",
        f"${var_value:,}",
        delta=f"{var_util:.0f}% of limit",
        delta_color="off"
    )

with col2:
    st.metric("Gross Exposure", "$12.5M", delta="62% of limit", delta_color="off")

with col3:
    st.metric("Net Exposure", "$8.2M", delta="Long", delta_color="off")

with col4:
    st.metric("Current Drawdown", "-1.8%", delta="36% of limit", delta_color="off")

with col5:
    st.metric("Active Alerts", "2", delta="1 Warning", delta_color="off")

st.divider()

# Main content
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Risk Overview",
    "📏 Position Limits",
    "🌪️ Stress Testing",
    "⚠️ Alerts"
])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("VaR Utilization")
        
        # VaR gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=var_util,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "VaR Utilization (%)", 'font': {'size': 16}},
            delta={'reference': 75, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "#00A3E0"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 75], 'color': 'rgba(0, 210, 106, 0.3)'},
                    {'range': [75, 90], 'color': 'rgba(255, 165, 0, 0.3)'},
                    {'range': [90, 100], 'color': 'rgba(255, 75, 75, 0.3)'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            template='plotly_dark',
            margin=dict(l=20, r=20, t=50, b=20),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # VaR breakdown
        st.subheader("Risk Contribution by Position")
        
        risk_data = pd.DataFrame({
            'Position': ['CLF5 (+45)', 'CLG5 (+20)', 'COH5 (-15)', 'XBF5 (+8)', 'HOF5 (+5)'],
            'Notional': [3305250, 1464000, -1167000, 745920, 529200],
            'VaR Contribution': [102500, 45200, 38500, 35800, 23000],
            'Weight': [42, 18, 16, 15, 9],
        })
        
        fig2 = px.pie(risk_data, values='VaR Contribution', names='Position',
                     title='VaR Contribution by Position',
                     color_discrete_sequence=px.colors.sequential.Blues_r)
        
        fig2.update_layout(
            template='plotly_dark',
            height=350,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        st.subheader("Risk Metrics")
        
        metrics = {
            "VaR (95%, 1-Day)": f"${var_value:,}",
            "VaR Limit": f"${var_limit:,}",
            "CVaR (Expected Shortfall)": "$312,000",
            "Daily Volatility": "1.8%",
            "Beta to Oil": "0.95",
            "Max Drawdown (30d)": "-3.2%",
        }
        
        for metric, value in metrics.items():
            st.text(f"{metric}: {value}")
        
        st.divider()
        
        st.subheader("Exposure Summary")
        
        exposure_metrics = {
            "Gross Exposure": "$12.5M",
            "Net Exposure": "$8.2M (Long)",
            "Crude Oil Exposure": "$10.1M",
            "Products Exposure": "$2.4M",
            "Long Exposure": "$10.3M",
            "Short Exposure": "$2.1M",
        }
        
        for metric, value in exposure_metrics.items():
            st.text(f"{metric}: {value}")
        
        st.divider()
        
        st.subheader("Correlation Alert")
        
        st.warning("""
        ⚠️ **High Correlation Warning**
        
        WTI and Brent positions have 0.95 correlation.
        Combined directional exposure: $10.2M (Net Long Oil)
        Effective diversification: LOW
        """)

with tab2:
    st.subheader("Position Limits Monitor")
    
    # Position limits table
    limits_data = pd.DataFrame({
        'Instrument': ['WTI (CL)', 'Brent (CO)', 'RBOB (XB)', 'Heating Oil (HO)', 'Spreads'],
        'Current': [65, 15, 8, 5, 10],
        'Limit': [100, 75, 50, 50, 50],
        'Utilization': [65, 20, 16, 10, 20],
        'Status': ['🟢 OK', '🟢 OK', '🟢 OK', '🟢 OK', '🟢 OK'],
    })
    
    # Add progress bars for visualization
    st.dataframe(
        limits_data,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Instrument': st.column_config.TextColumn('Instrument'),
            'Current': st.column_config.NumberColumn('Current Pos'),
            'Limit': st.column_config.NumberColumn('Max Limit'),
            'Utilization': st.column_config.ProgressColumn('Utilization %', min_value=0, max_value=100, format='%d%%'),
            'Status': st.column_config.TextColumn('Status'),
        }
    )
    
    st.divider()
    
    st.subheader("Exposure Limits")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Gross Exposure**")
        gross_util = 62.5
        st.progress(gross_util / 100, text=f"${12.5}M / $20M ({gross_util:.0f}%)")
        
        st.markdown("**Net Exposure**")
        net_util = 54.7
        st.progress(net_util / 100, text=f"${8.2}M / $15M ({net_util:.0f}%)")
    
    with col2:
        st.markdown("**Concentration Limits**")
        
        conc_data = {
            'WTI Concentration': (42, 40),
            'Crude Oil Group': (52, 60),
            'Single Strategy': (35, 50),
        }
        
        for name, (current, limit) in conc_data.items():
            color = "normal" if current <= limit else "inverse"
            status = "⚠️" if current > limit else "✓"
            st.metric(
                name, 
                f"{current}%", 
                delta=f"Limit: {limit}%",
                delta_color="off"
            )

with tab3:
    st.subheader("Stress Test Scenarios")
    
    # Define scenarios
    scenarios = {
        "Oil +10% Shock": {"factors": {"crude_oil": 0.10, "products": 0.10}},
        "Oil -10% Shock": {"factors": {"crude_oil": -0.10, "products": -0.10}},
        "Oil -20% Crash": {"factors": {"crude_oil": -0.20, "products": -0.18}},
        "2020 COVID Replay": {"factors": {"crude_oil": -0.65, "products": -0.50}},
        "WTI-Brent +$5": {"factors": {"wti_brent_spread": 5.0}},
    }
    
    # Run stress tests
    stress_results = var_calc.run_stress_test(mock_positions, scenarios)
    
    # Display results
    for _, row in stress_results.iterrows():
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f"**{row['scenario']}**")
        with col2:
            pnl = row['pnl']
            color = "#00D26A" if pnl > 0 else "#FF4B4B"
            st.markdown(f"<span style='color: {color}; font-size: 18px;'>${pnl:,.0f}</span>", unsafe_allow_html=True)
        with col3:
            if pnl > 0:
                st.success("Favorable")
            elif abs(row['pnl_pct']) < 5:
                st.info("Within Limits")
            elif abs(row['pnl_pct']) < 10:
                st.warning("Caution")
            else:
                st.error("Exceeds Limit")
    
    st.divider()
    
    # Visualization
    st.subheader("Stress Test Impact Visualization")
    
    fig = go.Figure()
    
    colors = ['#00D26A' if x > 0 else '#FF4B4B' for x in stress_results['pnl']]
    
    fig.add_trace(go.Bar(
        x=stress_results['scenario'],
        y=stress_results['pnl'],
        marker_color=colors,
        text=[f"${x:,.0f}" for x in stress_results['pnl']],
        textposition='outside',
    ))
    
    fig.add_hline(y=0, line_dash='solid', line_color='white')
    fig.add_hline(y=-var_limit, line_dash='dash', line_color='red', 
                 annotation_text='VaR Limit')
    
    fig.update_layout(
        template='plotly_dark',
        height=400,
        yaxis_title='P&L Impact ($)',
        margin=dict(l=0, r=0, t=10, b=0),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Custom scenario
    st.subheader("Custom Scenario Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        oil_shock = st.slider("Oil Price Shock (%)", -50, 50, 0, 5)
    with col2:
        spread_shock = st.slider("WTI-Brent Spread ($)", -10, 10, 0, 1)
    with col3:
        if st.button("Run Custom Scenario", use_container_width=True):
            custom_pnl = sum(
                pos["quantity"] * pos["price"] * 1000 * (oil_shock / 100)
                for pos in mock_positions.values()
            )
            
            st.metric("Custom Scenario P&L", f"${custom_pnl:,.0f}")

with tab4:
    st.subheader("Risk Alerts")
    
    # Active alerts
    alerts = [
        {
            "severity": "WARNING",
            "type": "Concentration",
            "message": "WTI concentration at 42% (limit: 40%)",
            "time": "14:32:15",
        },
        {
            "severity": "INFO",
            "type": "Correlation",
            "message": "High correlation between WTI and Brent positions",
            "time": "09:15:00",
        },
    ]
    
    for alert in alerts:
        if alert["severity"] == "CRITICAL":
            st.error(f"🚨 **{alert['type']}** - {alert['message']} ({alert['time']})")
        elif alert["severity"] == "WARNING":
            st.warning(f"⚠️ **{alert['type']}** - {alert['message']} ({alert['time']})")
        else:
            st.info(f"ℹ️ **{alert['type']}** - {alert['message']} ({alert['time']})")
    
    st.divider()
    
    st.subheader("Alert Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Alert Thresholds**")
        
        var_warning = st.slider("VaR Warning Level (%)", 50, 95, 75, 5)
        var_critical = st.slider("VaR Critical Level (%)", 80, 100, 90, 5)
        
    with col2:
        st.markdown("**Notification Settings**")
        
        st.checkbox("Email Alerts", value=True)
        st.checkbox("Dashboard Notifications", value=True)
        st.checkbox("Sound Alerts for Critical", value=False)
    
    st.divider()
    
    st.subheader("Alert History")
    
    history = pd.DataFrame({
        'Time': ['Today 14:32', 'Today 09:15', 'Yesterday 16:45', 'Yesterday 11:30'],
        'Type': ['Concentration', 'Correlation', 'VaR Breach', 'Position Limit'],
        'Severity': ['⚠️ Warning', 'ℹ️ Info', '🚨 Critical', '⚠️ Warning'],
        'Message': [
            'WTI concentration at 42%',
            'High correlation detected',
            'VaR exceeded 90% of limit',
            'Brent position at 80% of limit'
        ],
        'Status': ['Active', 'Acknowledged', 'Resolved', 'Resolved'],
    })
    
    st.dataframe(history, use_container_width=True, hide_index=True)
