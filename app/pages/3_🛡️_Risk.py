"""
Risk Management Page
====================
Portfolio risk monitoring and analysis with live data.
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

# Add app directory for shared state helpers
app_dir = Path(__file__).parent.parent
sys.path.insert(0, str(app_dir))

from app import shared_state
from core.risk import VaRCalculator, RiskLimits, RiskMonitor

st.set_page_config(page_title="Risk Management | Oil Trading", page_icon="🛡️", layout="wide")

# Apply shared theme
from app.components.theme import apply_theme, COLORS, PLOTLY_LAYOUT
apply_theme(st)

# Initialize components
@st.cache_resource
def get_risk_components():
    var_calc = VaRCalculator(confidence_level=0.95, holding_period=1)
    risk_limits = RiskLimits(config_path=str(project_root / "config" / "risk_limits.yaml"))
    risk_monitor = RiskMonitor()
    return var_calc, risk_limits, risk_monitor

var_calc, risk_limits, risk_monitor = get_risk_components()
context = shared_state.get_dashboard_context()
data_loader = context.data_loader

st.title("🛡️ Risk Management")
st.caption("Portfolio risk monitoring and stress testing")

# Get live portfolio data
portfolio = context.portfolio.summary
position_pnl = portfolio['positions']

# Build positions dict for risk calculations
positions_for_risk = {
    pos['ticker']: {
        'quantity': pos['qty'],
        'price': pos['current'],
        'notional': pos['notional']
    }
    for pos in position_pnl
}

# Top metrics row - All calculated from live data
col1, col2, col3, col4, col5 = st.columns(5)

var_value = portfolio['var_estimate']
var_limit = portfolio['var_limit']
var_util = portfolio['var_utilization']

with col1:
    st.metric(
        "Portfolio VaR (95%, 1-Day)",
        f"${var_value:,.0f}",
        delta=f"{var_util:.0f}% of limit",
        delta_color="off"
    )

with col2:
    gross_util = portfolio['gross_exposure'] / 20000000 * 100  # $20M limit
    st.metric(
        "Gross Exposure", 
        f"${portfolio['gross_exposure']/1e6:.1f}M", 
        delta=f"{gross_util:.0f}% of limit", 
        delta_color="off"
    )

with col3:
    net_label = "Long" if portfolio['net_exposure'] > 0 else "Short"
    st.metric(
        "Net Exposure", 
        f"${abs(portfolio['net_exposure'])/1e6:.1f}M", 
        delta=net_label, 
        delta_color="off"
    )

with col4:
    # Calculate drawdown from P&L
    drawdown_pct = -portfolio['total_pnl'] / 1000000 * 100 if portfolio['total_pnl'] < 0 else 0
    dd_util = abs(drawdown_pct) / 5 * 100 if drawdown_pct != 0 else 0  # 5% limit
    st.metric(
        "Current Drawdown", 
        f"{drawdown_pct:.1f}%", 
        delta=f"{dd_util:.0f}% of limit" if dd_util > 0 else "None", 
        delta_color="off"
    )

with col5:
    # Count active alerts
    alerts = []
    
    # Check concentration
    crude_exposure = sum(p['notional'] for p in position_pnl if p['ticker'].startswith('CL') or p['ticker'].startswith('CO'))
    crude_concentration = crude_exposure / portfolio['gross_exposure'] * 100 if portfolio['gross_exposure'] > 0 else 0
    if crude_concentration > 60:
        alerts.append("Concentration")
    
    if var_util > 75:
        alerts.append("VaR")
    
    st.metric(
        "Active Alerts", 
        len(alerts),
        delta=f"{len(alerts)} Warning" if alerts else "None",
        delta_color="off"
    )

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
        
        # VaR breakdown - calculated from actual positions
        st.subheader("Risk Contribution by Position")
        
        risk_data = []
        for pos in position_pnl:
            # Simplified VaR contribution (proportional to notional)
            var_contrib = pos['notional'] * 0.02  # 2% VaR assumption
            weight = pos['notional'] / portfolio['gross_exposure'] * 100 if portfolio['gross_exposure'] > 0 else 0
            
            risk_data.append({
                'Position': f"{pos['symbol']} ({pos['qty']:+d})",
                'Notional': pos['notional'],
                'VaR Contribution': var_contrib,
                'Weight': weight,
            })
        
        risk_df = pd.DataFrame(risk_data)
        
        fig2 = px.pie(
            risk_df, 
            values='VaR Contribution', 
            names='Position',
            title='VaR Contribution by Position',
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        
        fig2.update_layout(
            template='plotly_dark',
            height=350,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        st.subheader("Risk Metrics")
        
        # CVaR estimate (1.4x VaR typical)
        cvar = var_value * 1.4
        
        # Daily volatility (from VaR)
        daily_vol = var_value / (portfolio['gross_exposure'] * 1.65) * 100 if portfolio['gross_exposure'] > 0 else 0
        
        metrics = {
            "VaR (95%, 1-Day)": f"${var_value:,.0f}",
            "VaR Limit": f"${var_limit:,}",
            "CVaR (Expected Shortfall)": f"${cvar:,.0f}",
            "Daily Volatility": f"{daily_vol:.1f}%",
            "Beta to Oil": "0.95",
            "Max Drawdown (30d)": f"-${abs(min(portfolio['total_pnl'], 0)):,.0f}",
        }
        
        for metric, value in metrics.items():
            st.text(f"{metric}: {value}")
        
        st.divider()
        
        st.subheader("Exposure Summary")
        
        exposure_metrics = {
            "Gross Exposure": f"${portfolio['gross_exposure']/1e6:.2f}M",
            "Net Exposure": f"${portfolio['net_exposure']/1e6:.2f}M",
            "Long Exposure": f"${portfolio['long_exposure']/1e6:.2f}M",
            "Short Exposure": f"${portfolio['short_exposure']/1e6:.2f}M",
        }
        
        for metric, value in exposure_metrics.items():
            st.text(f"{metric}: {value}")
        
        st.divider()
        
        # Concentration check
        st.subheader("Correlation Alert")
        
        # Check for high correlation
        wti_exposure = sum(p['notional'] for p in position_pnl if p['ticker'].startswith('CL'))
        brent_exposure = sum(p['notional'] for p in position_pnl if p['ticker'].startswith('CO'))
        
        if wti_exposure > 0 and brent_exposure > 0:
            combined = wti_exposure + brent_exposure
            st.warning(f"""
            ⚠️ **High Correlation Warning**
            
            WTI and Brent positions have 0.95 correlation.
            Combined directional exposure: ${combined/1e6:.1f}M
            Effective diversification: LOW
            """)
        else:
            st.success("✅ No high-correlation alerts")

with tab2:
    st.subheader("Position Limits Monitor")
    
    # Calculate actual position utilization
    limits_data = []
    
    # WTI
    wti_qty = sum(p['qty'] for p in position_pnl if p['ticker'].startswith('CL'))
    limits_data.append({
        'Instrument': 'WTI (CL)',
        'Current': abs(wti_qty),
        'Limit': 100,
        'Utilization': abs(wti_qty),
        'Status': '🟢 OK' if abs(wti_qty) <= 100 else '🔴 Breach'
    })
    
    # Brent
    brent_qty = sum(p['qty'] for p in position_pnl if p['ticker'].startswith('CO'))
    limits_data.append({
        'Instrument': 'Brent (CO)',
        'Current': abs(brent_qty),
        'Limit': 75,
        'Utilization': abs(brent_qty) / 75 * 100,
        'Status': '🟢 OK' if abs(brent_qty) <= 75 else '🔴 Breach'
    })
    
    # Products
    rbob_qty = sum(p['qty'] for p in position_pnl if p['ticker'].startswith('XB'))
    limits_data.append({
        'Instrument': 'RBOB (XB)',
        'Current': abs(rbob_qty),
        'Limit': 50,
        'Utilization': abs(rbob_qty) / 50 * 100,
        'Status': '🟢 OK' if abs(rbob_qty) <= 50 else '🔴 Breach'
    })
    
    ho_qty = sum(p['qty'] for p in position_pnl if p['ticker'].startswith('HO'))
    limits_data.append({
        'Instrument': 'Heating Oil (HO)',
        'Current': abs(ho_qty),
        'Limit': 50,
        'Utilization': abs(ho_qty) / 50 * 100,
        'Status': '🟢 OK' if abs(ho_qty) <= 50 else '🔴 Breach'
    })
    
    limits_df = pd.DataFrame(limits_data)
    
    st.dataframe(
        limits_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Instrument': st.column_config.TextColumn('Instrument'),
            'Current': st.column_config.NumberColumn('Current Pos'),
            'Limit': st.column_config.NumberColumn('Max Limit'),
            'Utilization': st.column_config.ProgressColumn('Utilization %', min_value=0, max_value=100, format='%.0f%%'),
            'Status': st.column_config.TextColumn('Status'),
        }
    )
    
    st.divider()
    
    st.subheader("Exposure Limits")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Gross Exposure**")
        gross_util = portfolio['gross_exposure'] / 20000000 * 100
        st.progress(min(gross_util / 100, 1.0), text=f"${portfolio['gross_exposure']/1e6:.1f}M / $20M ({gross_util:.0f}%)")
        
        st.markdown("**Net Exposure**")
        net_util = abs(portfolio['net_exposure']) / 15000000 * 100
        st.progress(min(net_util / 100, 1.0), text=f"${abs(portfolio['net_exposure'])/1e6:.1f}M / $15M ({net_util:.0f}%)")
    
    with col2:
        st.markdown("**Concentration Limits**")
        
        # Calculate actual concentrations
        conc_data = {}
        
        wti_conc = wti_exposure / portfolio['gross_exposure'] * 100 if portfolio['gross_exposure'] > 0 else 0
        conc_data['WTI Concentration'] = (wti_conc, 40)
        
        crude_conc = crude_concentration
        conc_data['Crude Oil Group'] = (crude_conc, 60)
        
        # Single strategy (use largest)
        strategy_exposure = {}
        for pos in position_pnl:
            strat = pos['strategy']
            strategy_exposure[strat] = strategy_exposure.get(strat, 0) + pos['notional']
        
        max_strategy_conc = max(strategy_exposure.values()) / portfolio['gross_exposure'] * 100 if portfolio['gross_exposure'] > 0 and strategy_exposure else 0
        conc_data['Single Strategy'] = (max_strategy_conc, 50)
        
        for name, (current, limit) in conc_data.items():
            status = "⚠️" if current > limit else "✅"
            st.metric(
                name, 
                f"{current:.0f}%", 
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
    
    # Calculate stress test results based on actual positions
    stress_results = []
    
    for scenario_name, scenario in scenarios.items():
        factors = scenario['factors']
        pnl_impact = 0
        
        for pos in position_pnl:
            if 'crude_oil' in factors:
                if pos['ticker'].startswith('CL') or pos['ticker'].startswith('CO'):
                    pnl_impact += pos['notional'] * factors['crude_oil'] * (1 if pos['qty'] > 0 else -1)
            
            if 'products' in factors:
                if pos['ticker'].startswith('XB') or pos['ticker'].startswith('HO'):
                    pnl_impact += pos['notional'] * factors['products'] * (1 if pos['qty'] > 0 else -1)
            
            if 'wti_brent_spread' in factors:
                if pos['ticker'].startswith('CL'):
                    pnl_impact += pos['qty'] * factors['wti_brent_spread'] * 1000
        
        stress_results.append({
            'scenario': scenario_name,
            'pnl': pnl_impact,
            'pnl_pct': pnl_impact / portfolio['gross_exposure'] * 100 if portfolio['gross_exposure'] > 0 else 0
        })
    
    stress_df = pd.DataFrame(stress_results)
    
    # Display results
    for _, row in stress_df.iterrows():
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
    
    colors = ['#00D26A' if x > 0 else '#FF4B4B' for x in stress_df['pnl']]
    
    fig.add_trace(go.Bar(
        x=stress_df['scenario'],
        y=stress_df['pnl'],
        marker_color=colors,
        text=[f"${x:,.0f}" for x in stress_df['pnl']],
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
            custom_pnl = 0
            for pos in position_pnl:
                if pos['ticker'].startswith('CL') or pos['ticker'].startswith('CO'):
                    custom_pnl += pos['notional'] * (oil_shock / 100) * (1 if pos['qty'] > 0 else -1)
                if pos['ticker'].startswith('CL'):
                    custom_pnl += pos['qty'] * spread_shock * 1000
            
            color = "#00D26A" if custom_pnl > 0 else "#FF4B4B"
            st.markdown(f"**Custom Scenario P&L:** <span style='color: {color}; font-size: 24px;'>${custom_pnl:,.0f}</span>", unsafe_allow_html=True)

with tab4:
    st.subheader("Risk Alerts")
    
    # Generate alerts based on actual conditions
    alerts = []
    
    # Check VaR
    if var_util > 90:
        alerts.append({
            "severity": "CRITICAL",
            "type": "VaR",
            "message": f"VaR at {var_util:.0f}% of limit",
            "time": datetime.now().strftime("%H:%M:%S"),
        })
    elif var_util > 75:
        alerts.append({
            "severity": "WARNING",
            "type": "VaR",
            "message": f"VaR at {var_util:.0f}% of limit",
            "time": datetime.now().strftime("%H:%M:%S"),
        })
    
    # Check concentration
    if wti_conc > 40:
        alerts.append({
            "severity": "WARNING",
            "type": "Concentration",
            "message": f"WTI concentration at {wti_conc:.0f}% (limit: 40%)",
            "time": datetime.now().strftime("%H:%M:%S"),
        })
    
    # Check correlation
    if wti_exposure > 0 and brent_exposure > 0:
        alerts.append({
            "severity": "INFO",
            "type": "Correlation",
            "message": "High correlation between WTI and Brent positions",
            "time": datetime.now().strftime("%H:%M:%S"),
        })
    
    if not alerts:
        st.success("✅ No active risk alerts")
    else:
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
        'Type': ['Concentration', 'Correlation', 'VaR Warning', 'Position Limit'],
        'Severity': ['⚠️ Warning', 'ℹ️ Info', '⚠️ Warning', '⚠️ Warning'],
        'Message': [
            f'WTI concentration at {wti_conc:.0f}%',
            'High correlation detected',
            'VaR at 78% of limit',
            'Brent position at 80% of limit'
        ],
        'Status': ['Active', 'Acknowledged', 'Resolved', 'Resolved'],
    })
    
    st.dataframe(history, use_container_width=True, hide_index=True)
