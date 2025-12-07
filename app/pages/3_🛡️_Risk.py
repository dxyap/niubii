"""
Risk Management Page
====================
Portfolio risk monitoring and analysis with live data.
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from app.components.charts import BASE_LAYOUT, CHART_COLORS, create_gauge_chart
from app.page_utils import get_chart_config, init_page
from core.risk import RiskLimits, RiskMonitor, VaRCalculator

# Initialize page
ctx = init_page(
    title="🛡️ Risk Management",
    page_title="Risk Management | Oil Trading",
    icon="🛡️",
)

st.caption("Portfolio risk monitoring and stress testing")

# Initialize risk components (cached)
project_root = Path(__file__).parent.parent.parent

@st.cache_resource
def get_risk_components():
    var_calc = VaRCalculator(confidence_level=0.95, holding_period=1)
    risk_limits = RiskLimits(config_path=str(project_root / "config" / "risk_limits.yaml"))
    risk_monitor = RiskMonitor()
    return var_calc, risk_limits, risk_monitor

var_calc, risk_limits, risk_monitor = get_risk_components()

# Get live portfolio data
portfolio = ctx.portfolio.summary
position_pnl = portfolio['positions']

# Top metrics row
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
    gross_util = portfolio['gross_exposure'] / 20000000 * 100
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
    drawdown_pct = -portfolio['total_pnl'] / 1000000 * 100 if portfolio['total_pnl'] < 0 else 0
    dd_util = abs(drawdown_pct) / 5 * 100 if drawdown_pct != 0 else 0
    st.metric(
        "Current Drawdown",
        f"{drawdown_pct:.1f}%",
        delta=f"{dd_util:.0f}% of limit" if dd_util > 0 else "None",
        delta_color="off"
    )

with col5:
    alerts = []
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

# Main content tabs
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

        fig = create_gauge_chart(
            value=var_util,
            title="VaR Utilization",
            min_val=0,
            max_val=100,
            warning_threshold=75,
            critical_threshold=90,
            height=300,
        )

        st.plotly_chart(fig, width='stretch', config=get_chart_config())

        st.subheader("Risk Contribution by Position")

        risk_data = []
        for pos in position_pnl:
            var_contrib = pos['notional'] * 0.02
            weight = pos['notional'] / portfolio['gross_exposure'] * 100 if portfolio['gross_exposure'] > 0 else 0

            risk_data.append({
                'Position': f"{pos['symbol']} ({pos['qty']:+d})",
                'Notional': pos['notional'],
                'VaR Contribution': var_contrib,
                'Weight': weight,
            })

        risk_df = pd.DataFrame(risk_data)

        if not risk_df.empty:
            fig2 = px.pie(
                risk_df,
                values='VaR Contribution',
                names='Position',
                title='VaR Contribution by Position',
                color_discrete_sequence=[CHART_COLORS['primary'], CHART_COLORS['ma_fast'], CHART_COLORS['secondary'], CHART_COLORS['profit']],
            )

            fig2.update_layout(
                **BASE_LAYOUT,
                height=350,
                showlegend=True,
                legend={
                    "orientation": 'v',
                    "yanchor": 'middle',
                    "y": 0.5,
                    "xanchor": 'left',
                    "x": 1.02,
                    "font": {"size": 11, "color": CHART_COLORS['text_secondary']},
                },
            )

            fig2.update_traces(
                textposition='inside',
                textfont={"size": 12, "color": 'white'},
                hovertemplate='%{label}<br>$%{value:,.0f}<extra></extra>',
            )

            st.plotly_chart(fig2, width='stretch', config=get_chart_config())

    with col2:
        st.subheader("Risk Metrics")

        cvar = var_value * 1.4
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

with tab2:
    st.subheader("Position Limits Monitor")

    limits_data = []

    wti_qty = sum(p['qty'] for p in position_pnl if p['ticker'].startswith('CL'))
    limits_data.append({
        'Instrument': 'WTI (CL)',
        'Current': abs(wti_qty),
        'Limit': 100,
        'Utilization': abs(wti_qty),
        'Status': '🟢 OK' if abs(wti_qty) <= 100 else '🔴 Breach'
    })

    brent_qty = sum(p['qty'] for p in position_pnl if p['ticker'].startswith('CO'))
    limits_data.append({
        'Instrument': 'Brent (CO)',
        'Current': abs(brent_qty),
        'Limit': 75,
        'Utilization': abs(brent_qty) / 75 * 100,
        'Status': '🟢 OK' if abs(brent_qty) <= 75 else '🔴 Breach'
    })

    limits_df = pd.DataFrame(limits_data)

    st.dataframe(
        limits_df,
        width='stretch',
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

with tab3:
    st.subheader("Stress Test Scenarios")

    scenarios = {
        "Oil +10% Shock": {"factors": {"crude_oil": 0.10}},
        "Oil -10% Shock": {"factors": {"crude_oil": -0.10}},
        "Oil -20% Crash": {"factors": {"crude_oil": -0.20}},
        "2020 COVID Replay": {"factors": {"crude_oil": -0.65}},
    }

    stress_results = []

    for scenario_name, scenario in scenarios.items():
        factors = scenario['factors']
        pnl_impact = 0

        for pos in position_pnl:
            if 'crude_oil' in factors:
                if pos['ticker'].startswith('CL') or pos['ticker'].startswith('CO'):
                    pnl_impact += pos['notional'] * factors['crude_oil'] * (1 if pos['qty'] > 0 else -1)

        stress_results.append({
            'scenario': scenario_name,
            'pnl': pnl_impact,
            'pnl_pct': pnl_impact / portfolio['gross_exposure'] * 100 if portfolio['gross_exposure'] > 0 else 0
        })

    stress_df = pd.DataFrame(stress_results)

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
            else:
                st.warning("Caution")

with tab4:
    st.subheader("Risk Alerts")

    wti_exposure = sum(p['notional'] for p in position_pnl if p['ticker'].startswith('CL'))
    wti_conc = wti_exposure / portfolio['gross_exposure'] * 100 if portfolio['gross_exposure'] > 0 else 0

    alert_list = []

    if var_util > 90:
        alert_list.append({"severity": "CRITICAL", "type": "VaR", "message": f"VaR at {var_util:.0f}% of limit"})
    elif var_util > 75:
        alert_list.append({"severity": "WARNING", "type": "VaR", "message": f"VaR at {var_util:.0f}% of limit"})

    if wti_conc > 40:
        alert_list.append({"severity": "WARNING", "type": "Concentration", "message": f"WTI concentration at {wti_conc:.0f}% (limit: 40%)"})

    if not alert_list:
        st.success("✅ No active risk alerts")
    else:
        for alert in alert_list:
            if alert["severity"] == "CRITICAL":
                st.error(f"🚨 **{alert['type']}** - {alert['message']}")
            elif alert["severity"] == "WARNING":
                st.warning(f"⚠️ **{alert['type']}** - {alert['message']}")
            else:
                st.info(f"ℹ️ **{alert['type']}** - {alert['message']}")
