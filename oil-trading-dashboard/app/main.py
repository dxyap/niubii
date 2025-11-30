"""
Oil Trading Dashboard - Main Application
========================================
Quantitative Oil Trading Dashboard built with Streamlit.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Oil Trading Dashboard",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Quantitative Oil Trading Dashboard v1.0"
    }
)

# Custom CSS for trading terminal aesthetic
st.markdown("""
<style>
    /* Dark theme with trading terminal aesthetic */
    .stApp {
        background-color: #0E1117;
    }
    
    /* Custom metrics styling */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: bold;
        font-family: 'JetBrains Mono', 'SF Mono', monospace;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 14px;
        font-family: 'JetBrains Mono', 'SF Mono', monospace;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Inter', -apple-system, sans-serif;
        font-weight: 600;
    }
    
    /* Card-like containers */
    .metric-card {
        background-color: #1E2127;
        border-radius: 8px;
        padding: 20px;
        border: 1px solid #2D3139;
    }
    
    /* Profit/Loss colors */
    .profit {
        color: #00D26A !important;
    }
    
    .loss {
        color: #FF4B4B !important;
    }
    
    /* Data tables */
    .dataframe {
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0E1117;
        border-right: 1px solid #2D3139;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #00A3E0;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        background-color: #0088C2;
    }
    
    /* Input fields */
    .stTextInput > div > div > input {
        background-color: #1E2127;
        border: 1px solid #2D3139;
        border-radius: 6px;
        color: #FAFAFA;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #1E2127;
        border-radius: 6px 6px 0 0;
        border: 1px solid #2D3139;
        border-bottom: none;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #00A3E0;
        border-color: #00A3E0;
    }
    
    /* Status indicators */
    .status-ok {
        color: #00D26A;
    }
    
    .status-warning {
        color: #FFA500;
    }
    
    .status-critical {
        color: #FF4B4B;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loader' not in st.session_state:
    from core.data import DataLoader
    st.session_state.data_loader = DataLoader(
        config_dir=str(project_root / "config"),
        data_dir=str(project_root / "data"),
        use_mock=True
    )

if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()


def main():
    """Main application entry point."""
    
    # Sidebar navigation
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/oil-barrel.png", width=60)
        st.title("Oil Trading")
        st.caption("Quantitative Dashboard")
        
        st.divider()
        
        # Data status
        data_loader = st.session_state.data_loader
        
        # Get market data
        try:
            prices = data_loader.get_oil_prices()
            data_status = "🟢 Live"
        except Exception:
            prices = {"WTI": 72.50, "Brent": 77.20}
            data_status = "🟡 Mock"
        
        st.caption(f"Data Status: {data_status}")
        st.caption(f"Last Update: {st.session_state.last_refresh.strftime('%H:%M:%S')}")
        
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.session_state.last_refresh = datetime.now()
            data_loader.cache.clear()
            st.rerun()
        
        st.divider()
        
        # Quick prices
        st.subheader("Quick View")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("WTI", f"${prices.get('WTI', 0):.2f}")
        with col2:
            st.metric("Brent", f"${prices.get('Brent', 0):.2f}")
        
        # WTI-Brent spread
        spread = prices.get('WTI', 0) - prices.get('Brent', 0)
        st.metric("WTI-Brent", f"${spread:.2f}")
        
        st.divider()
        
        # Navigation links
        st.page_link("main.py", label="📊 Overview", icon="🏠")
        st.page_link("pages/1_📈_Market_Insights.py", label="📈 Market Insights")
        st.page_link("pages/2_📡_Signals.py", label="📡 Signals")
        st.page_link("pages/3_🛡️_Risk.py", label="🛡️ Risk Management")
        st.page_link("pages/4_💼_Trade_Entry.py", label="💼 Trade Entry")
        st.page_link("pages/5_📋_Blotter.py", label="📋 Trade Blotter")
        st.page_link("pages/6_📊_Analytics.py", label="📊 Analytics")
    
    # Main content - Overview Dashboard
    st.title("🛢️ Oil Trading Dashboard")
    st.caption("Real-time quantitative analysis for oil markets")
    
    # Top row - Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        wti_price = prices.get('WTI', 72.50)
        wti_change = np.random.uniform(-0.5, 0.5)
        st.metric(
            "WTI Crude",
            f"${wti_price:.2f}",
            f"{wti_change:+.2f} ({wti_change/wti_price*100:+.2f}%)"
        )
    
    with col2:
        brent_price = prices.get('Brent', 77.20)
        brent_change = np.random.uniform(-0.5, 0.5)
        st.metric(
            "Brent Crude",
            f"${brent_price:.2f}",
            f"{brent_change:+.2f} ({brent_change/brent_price*100:+.2f}%)"
        )
    
    with col3:
        spread = wti_price - brent_price
        st.metric("WTI-Brent", f"${spread:.2f}", "-0.15")
    
    with col4:
        crack = np.random.uniform(25, 32)
        st.metric("3-2-1 Crack", f"${crack:.2f}", "+1.20")
    
    with col5:
        # Mock P&L
        daily_pnl = np.random.uniform(-50000, 150000)
        st.metric(
            "Day P&L",
            f"${daily_pnl:,.0f}",
            delta=f"{daily_pnl/1000000*100:+.2f}%",
            delta_color="normal"
        )
    
    st.divider()
    
    # Two column layout
    left_col, right_col = st.columns([2, 1])
    
    with left_col:
        # Price chart
        st.subheader("📈 WTI Crude Oil Price")
        
        # Generate mock price data
        dates = pd.date_range(end=datetime.now(), periods=90, freq='D')
        base_price = 72.50
        np.random.seed(42)
        returns = np.random.normal(0.0002, 0.015, 90)
        prices_series = base_price * np.exp(np.cumsum(returns))
        
        chart_data = pd.DataFrame({
            'Date': dates,
            'Price': prices_series
        }).set_index('Date')
        
        st.line_chart(chart_data, use_container_width=True, height=300)
        
        # Futures Curve
        st.subheader("📊 WTI Futures Curve")
        
        curve_data = data_loader.get_futures_curve("wti", 12)
        
        curve_chart = pd.DataFrame({
            'Month': curve_data['month'],
            'Price': curve_data['price']
        })
        
        st.bar_chart(curve_chart.set_index('Month'), use_container_width=True, height=250)
        
        # Curve metrics
        curve_cols = st.columns(4)
        with curve_cols[0]:
            m1_m2 = curve_data['price'].iloc[0] - curve_data['price'].iloc[1]
            st.metric("M1-M2 Spread", f"${m1_m2:.2f}")
        with curve_cols[1]:
            m1_m6 = curve_data['price'].iloc[0] - curve_data['price'].iloc[5]
            st.metric("M1-M6 Spread", f"${m1_m6:.2f}")
        with curve_cols[2]:
            m1_m12 = curve_data['price'].iloc[0] - curve_data['price'].iloc[11]
            st.metric("M1-M12 Spread", f"${m1_m12:.2f}")
        with curve_cols[3]:
            slope = (curve_data['price'].iloc[11] - curve_data['price'].iloc[0]) / 11
            structure = "Contango" if slope > 0.05 else "Backwardation" if slope < -0.05 else "Flat"
            st.metric("Structure", structure)
    
    with right_col:
        # Position Summary
        st.subheader("💼 Position Summary")
        
        # Mock positions
        positions_data = pd.DataFrame({
            'Instrument': ['CLF5', 'CLG5', 'COH5', 'XBF5'],
            'Qty': [45, 20, -15, 8],
            'Entry': [72.15, 72.50, 78.20, 2.15],
            'Current': [73.45, 73.20, 77.80, 2.22],
            'P&L': ['+$58,500', '+$14,000', '+$6,000', '+$2,352'],
        })
        
        st.dataframe(
            positions_data,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Instrument': st.column_config.TextColumn('Symbol'),
                'Qty': st.column_config.NumberColumn('Qty', format='%d'),
                'Entry': st.column_config.NumberColumn('Entry', format='$%.2f'),
                'Current': st.column_config.NumberColumn('Current', format='$%.2f'),
                'P&L': st.column_config.TextColumn('P&L'),
            }
        )
        
        # Total P&L
        total_pnl = 80852
        st.metric("Total Unrealized P&L", f"${total_pnl:,}")
        
        st.divider()
        
        # Risk Summary
        st.subheader("🛡️ Risk Summary")
        
        # VaR gauge
        var_util = 65
        st.progress(var_util / 100, text=f"VaR Utilization: {var_util}%")
        
        risk_metrics = {
            'VaR (95%, 1-day)': '$245,000',
            'VaR Limit': '$375,000',
            'Gross Exposure': '$12.5M',
            'Net Exposure': '$8.2M',
        }
        
        for metric, value in risk_metrics.items():
            st.text(f"{metric}: {value}")
        
        st.divider()
        
        # Active Alerts
        st.subheader("⚠️ Active Alerts")
        
        # Mock alerts
        alerts = [
            {"type": "WARNING", "msg": "WTI concentration at 42%"},
            {"type": "INFO", "msg": "Curve backwardation deepening"},
        ]
        
        for alert in alerts:
            if alert["type"] == "WARNING":
                st.warning(alert["msg"], icon="⚠️")
            elif alert["type"] == "CRITICAL":
                st.error(alert["msg"], icon="🚨")
            else:
                st.info(alert["msg"], icon="ℹ️")
    
    st.divider()
    
    # Bottom row - Recent activity
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Recent Trades")
        
        recent_trades = pd.DataFrame({
            'Time': ['14:32', '10:15', '09:45', '09:30'],
            'Symbol': ['CLF5', 'CLF5', 'COH5', 'XBF5'],
            'Side': ['BUY', 'SELL', 'SELL', 'BUY'],
            'Qty': [10, 5, 15, 8],
            'Price': [72.45, 73.20, 78.20, 2.15],
        })
        
        st.dataframe(recent_trades, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("📡 Active Signals")
        
        signals = pd.DataFrame({
            'Instrument': ['WTI (CL1)', 'WTI-Brent'],
            'Direction': ['🟢 LONG', '🔴 SHORT'],
            'Confidence': ['72%', '58%'],
            'Horizon': ['5-10 Days', '3-5 Days'],
        })
        
        st.dataframe(signals, use_container_width=True, hide_index=True)
    
    # Footer
    st.divider()
    st.caption(
        f"Oil Trading Dashboard v1.0 | "
        f"Data refreshed: {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')} | "
        f"⚠️ Signals are advisory only - human execution required"
    )


if __name__ == "__main__":
    main()
