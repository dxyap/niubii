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
import time

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
    
    /* Live indicator */
    .live-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        background-color: #00D26A;
        border-radius: 50%;
        margin-right: 6px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    /* Price change colors */
    .price-up {
        color: #00D26A !important;
    }
    
    .price-down {
        color: #FF4B4B !important;
    }
    
    /* Status indicators */
    .status-ok { color: #00D26A; }
    .status-warning { color: #FFA500; }
    .status-critical { color: #FF4B4B; }
    
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

if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True

if 'positions' not in st.session_state:
    # Initialize mock positions with entry prices
    st.session_state.positions = [
        {"symbol": "CLF5", "ticker": "CL1 Comdty", "qty": 45, "entry": 72.15, "strategy": "Momentum"},
        {"symbol": "CLG5", "ticker": "CL2 Comdty", "qty": 20, "entry": 72.50, "strategy": "Spread"},
        {"symbol": "COH5", "ticker": "CO1 Comdty", "qty": -15, "entry": 78.20, "strategy": "Arb"},
        {"symbol": "XBF5", "ticker": "XB1 Comdty", "qty": 8, "entry": 2.15, "strategy": "Crack"},
    ]


def calculate_position_pnl(positions: list, data_loader) -> pd.DataFrame:
    """Calculate P&L for all positions using current prices."""
    results = []
    
    for pos in positions:
        current_price = data_loader.get_price(pos["ticker"])
        entry_price = pos["entry"]
        qty = pos["qty"]
        
        # Contract multiplier (1000 for CL/CO, 42000 for XB/HO)
        if pos["ticker"].startswith("XB") or pos["ticker"].startswith("HO"):
            multiplier = 42000  # 42,000 gallons per contract
        else:
            multiplier = 1000  # 1,000 barrels per contract
        
        # Calculate P&L
        price_change = current_price - entry_price
        pnl = price_change * qty * multiplier
        pnl_pct = (price_change / entry_price * 100) if entry_price != 0 else 0
        
        results.append({
            "symbol": pos["symbol"],
            "qty": qty,
            "entry": entry_price,
            "current": round(current_price, 2),
            "pnl": round(pnl, 0),
            "pnl_pct": round(pnl_pct, 2),
            "strategy": pos["strategy"],
        })
    
    return pd.DataFrame(results)


def format_pnl(value: float) -> str:
    """Format P&L with color indicator."""
    if value >= 0:
        return f"+${value:,.0f}"
    return f"-${abs(value):,.0f}"


def main():
    """Main application entry point."""
    
    data_loader = st.session_state.data_loader
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/oil-barrel.png", width=60)
        st.title("Oil Trading")
        st.caption("Quantitative Dashboard")
        
        st.divider()
        
        # Data status with live indicator
        st.markdown(
            '<span class="live-indicator"></span> <span style="color: #00D26A;">Live Data</span>',
            unsafe_allow_html=True
        )
        st.caption(f"Last Update: {st.session_state.last_refresh.strftime('%H:%M:%S')}")
        
        # Auto-refresh toggle
        auto_refresh = st.toggle("Auto Refresh (5s)", value=st.session_state.auto_refresh)
        st.session_state.auto_refresh = auto_refresh
        
        if st.button("🔄 Refresh Now", use_container_width=True):
            st.session_state.last_refresh = datetime.now()
            st.rerun()
        
        st.divider()
        
        # Quick prices from actual data
        prices = data_loader.get_oil_prices()
        
        st.subheader("Quick View")
        col1, col2 = st.columns(2)
        with col1:
            wti = prices.get('WTI', {})
            delta_color = "normal" if wti.get('change', 0) >= 0 else "inverse"
            st.metric(
                "WTI", 
                f"${wti.get('current', 0):.2f}",
                f"{wti.get('change', 0):+.2f}",
                delta_color=delta_color
            )
        with col2:
            brent = prices.get('Brent', {})
            delta_color = "normal" if brent.get('change', 0) >= 0 else "inverse"
            st.metric(
                "Brent", 
                f"${brent.get('current', 0):.2f}",
                f"{brent.get('change', 0):+.2f}",
                delta_color=delta_color
            )
        
        # WTI-Brent spread
        spread_data = data_loader.get_wti_brent_spread()
        st.metric(
            "WTI-Brent", 
            f"${spread_data['spread']:.2f}",
            f"{spread_data['change']:+.2f}"
        )
        
        st.divider()
        
        # Navigation
        st.page_link("main.py", label="📊 Overview", icon="🏠")
        st.page_link("pages/1_📈_Market_Insights.py", label="📈 Market Insights")
        st.page_link("pages/2_📡_Signals.py", label="📡 Signals")
        st.page_link("pages/3_🛡️_Risk.py", label="🛡️ Risk Management")
        st.page_link("pages/4_💼_Trade_Entry.py", label="💼 Trade Entry")
        st.page_link("pages/5_📋_Blotter.py", label="📋 Trade Blotter")
        st.page_link("pages/6_📊_Analytics.py", label="📊 Analytics")
    
    # Main content
    st.title("🛢️ Oil Trading Dashboard")
    st.caption("Real-time quantitative analysis for oil markets")
    
    # Get all data
    prices = data_loader.get_oil_prices()
    crack_data = data_loader.get_crack_spread_321()
    
    # Calculate portfolio P&L
    positions_df = calculate_position_pnl(st.session_state.positions, data_loader)
    total_pnl = positions_df['pnl'].sum()
    
    # Top row - Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        wti = prices.get('WTI', {})
        change_pct = wti.get('change_pct', 0)
        st.metric(
            "WTI Crude",
            f"${wti.get('current', 0):.2f}",
            f"{wti.get('change', 0):+.2f} ({change_pct:+.2f}%)",
            delta_color="normal" if wti.get('change', 0) >= 0 else "inverse"
        )
    
    with col2:
        brent = prices.get('Brent', {})
        change_pct = brent.get('change_pct', 0)
        st.metric(
            "Brent Crude",
            f"${brent.get('current', 0):.2f}",
            f"{brent.get('change', 0):+.2f} ({change_pct:+.2f}%)",
            delta_color="normal" if brent.get('change', 0) >= 0 else "inverse"
        )
    
    with col3:
        spread_data = data_loader.get_wti_brent_spread()
        st.metric(
            "WTI-Brent", 
            f"${spread_data['spread']:.2f}", 
            f"{spread_data['change']:+.2f}"
        )
    
    with col4:
        st.metric(
            "3-2-1 Crack", 
            f"${crack_data['crack']:.2f}", 
            f"{crack_data['change']:+.2f}"
        )
    
    with col5:
        pnl_pct = (total_pnl / 1000000 * 100)  # Assuming $1M base
        st.metric(
            "Day P&L",
            f"${total_pnl:,.0f}",
            f"{pnl_pct:+.2f}%",
            delta_color="normal" if total_pnl >= 0 else "inverse"
        )
    
    st.divider()
    
    # Two column layout
    left_col, right_col = st.columns([2, 1])
    
    with left_col:
        # Price chart using historical data that connects to current price
        st.subheader("📈 WTI Crude Oil Price")
        
        # Get historical data (now connected to current price)
        hist_data = data_loader.get_historical(
            "CL1 Comdty",
            start_date=datetime.now() - timedelta(days=90),
            end_date=datetime.now()
        )
        
        if hist_data is not None and len(hist_data) > 0:
            chart_data = pd.DataFrame({
                'Price': hist_data['PX_LAST']
            })
            st.line_chart(chart_data, use_container_width=True, height=300)
            
            # Price stats
            stat_cols = st.columns(4)
            with stat_cols[0]:
                current = hist_data['PX_LAST'].iloc[-1]
                st.metric("Current", f"${current:.2f}")
            with stat_cols[1]:
                high = hist_data['PX_HIGH'].max()
                st.metric("90D High", f"${high:.2f}")
            with stat_cols[2]:
                low = hist_data['PX_LOW'].min()
                st.metric("90D Low", f"${low:.2f}")
            with stat_cols[3]:
                avg = hist_data['PX_LAST'].mean()
                st.metric("90D Avg", f"${avg:.2f}")
        
        # Futures Curve
        st.subheader("📊 WTI Futures Curve")
        
        curve_data = data_loader.get_futures_curve("wti", 12)
        
        curve_chart = pd.DataFrame({
            'Month': curve_data['month'],
            'Price': curve_data['price']
        })
        
        st.bar_chart(curve_chart.set_index('Month'), use_container_width=True, height=250)
        
        # Curve metrics - all calculated from actual data
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
        # Position Summary - calculated from actual prices
        st.subheader("💼 Position Summary")
        
        # Format for display
        display_df = positions_df.copy()
        display_df['P&L'] = display_df['pnl'].apply(format_pnl)
        display_df = display_df.rename(columns={
            'symbol': 'Symbol',
            'qty': 'Qty',
            'entry': 'Entry',
            'current': 'Current',
        })
        
        st.dataframe(
            display_df[['Symbol', 'Qty', 'Entry', 'Current', 'P&L']],
            use_container_width=True,
            hide_index=True,
            column_config={
                'Symbol': st.column_config.TextColumn('Symbol'),
                'Qty': st.column_config.NumberColumn('Qty', format='%d'),
                'Entry': st.column_config.NumberColumn('Entry', format='$%.2f'),
                'Current': st.column_config.NumberColumn('Current', format='$%.2f'),
                'P&L': st.column_config.TextColumn('P&L'),
            }
        )
        
        # Total P&L
        pnl_color = "#00D26A" if total_pnl >= 0 else "#FF4B4B"
        st.markdown(
            f"<h3 style='color: {pnl_color};'>Total P&L: {format_pnl(total_pnl)}</h3>",
            unsafe_allow_html=True
        )
        
        st.divider()
        
        # Risk Summary
        st.subheader("🛡️ Risk Summary")
        
        # Calculate actual VaR based on positions
        gross_exposure = sum(abs(p['qty']) * data_loader.get_price(p['ticker']) * (1000 if not p['ticker'].startswith('XB') else 42000) 
                            for p in st.session_state.positions)
        
        # Simplified VaR (2% of gross exposure for oil)
        var_estimate = gross_exposure * 0.02
        var_limit = 375000
        var_util = min(var_estimate / var_limit * 100, 100)
        
        st.progress(var_util / 100, text=f"VaR Utilization: {var_util:.0f}%")
        
        risk_metrics = {
            'VaR (95%, 1-day)': f'${var_estimate:,.0f}',
            'VaR Limit': f'${var_limit:,}',
            'Gross Exposure': f'${gross_exposure/1e6:.1f}M',
            'Net Position': f'{sum(p["qty"] for p in st.session_state.positions):,} contracts',
        }
        
        for metric, value in risk_metrics.items():
            st.text(f"{metric}: {value}")
        
        st.divider()
        
        # Active Alerts - based on actual conditions
        st.subheader("⚠️ Active Alerts")
        
        alerts = []
        
        # Check concentration
        wti_exposure = sum(
            abs(p['qty']) * data_loader.get_price(p['ticker']) * 1000 
            for p in st.session_state.positions 
            if p['ticker'].startswith('CL')
        )
        wti_concentration = wti_exposure / gross_exposure * 100 if gross_exposure > 0 else 0
        
        if wti_concentration > 40:
            alerts.append({"type": "WARNING", "msg": f"WTI concentration at {wti_concentration:.0f}%"})
        
        # Check curve structure
        if structure == "Backwardation":
            alerts.append({"type": "INFO", "msg": "Curve in backwardation"})
        elif structure == "Contango":
            alerts.append({"type": "INFO", "msg": "Curve in contango - roll costs apply"})
        
        # VaR alert
        if var_util > 75:
            alerts.append({"type": "WARNING", "msg": f"VaR at {var_util:.0f}% of limit"})
        
        if not alerts:
            st.success("No active alerts", icon="✅")
        else:
            for alert in alerts:
                if alert["type"] == "WARNING":
                    st.warning(alert["msg"], icon="⚠️")
                elif alert["type"] == "CRITICAL":
                    st.error(alert["msg"], icon="🚨")
                else:
                    st.info(alert["msg"], icon="ℹ️")
    
    st.divider()
    
    # Bottom row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Recent Trades")
        
        # Get current prices for realistic trade prices
        wti_price = data_loader.get_price("CL1 Comdty")
        brent_price = data_loader.get_price("CO1 Comdty")
        rbob_price = data_loader.get_price("XB1 Comdty")
        
        recent_trades = pd.DataFrame({
            'Time': ['14:32', '10:15', '09:45', '09:30'],
            'Symbol': ['CLF5', 'CLF5', 'COH5', 'XBF5'],
            'Side': ['BUY', 'SELL', 'SELL', 'BUY'],
            'Qty': [10, 5, 15, 8],
            'Price': [
                round(wti_price * 0.998, 2),
                round(wti_price * 1.002, 2),
                round(brent_price, 2),
                round(rbob_price, 2)
            ],
        })
        
        st.dataframe(recent_trades, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("📡 Active Signals")
        
        # Generate signals based on actual data
        wti_data = prices.get('WTI', {})
        wti_change = wti_data.get('change_pct', 0)
        
        signals = []
        
        # Momentum signal based on price change
        if wti_change > 0.3:
            signals.append({
                'Instrument': 'WTI (CL1)',
                'Direction': '🟢 LONG',
                'Confidence': f'{min(50 + wti_change * 10, 85):.0f}%',
                'Horizon': '5-10 Days',
            })
        elif wti_change < -0.3:
            signals.append({
                'Instrument': 'WTI (CL1)',
                'Direction': '🔴 SHORT',
                'Confidence': f'{min(50 + abs(wti_change) * 10, 85):.0f}%',
                'Horizon': '5-10 Days',
            })
        else:
            signals.append({
                'Instrument': 'WTI (CL1)',
                'Direction': '⚪ NEUTRAL',
                'Confidence': '45%',
                'Horizon': '-',
            })
        
        # Spread signal based on WTI-Brent
        spread = spread_data['spread']
        if spread < -5:
            signals.append({
                'Instrument': 'WTI-Brent',
                'Direction': '🟢 LONG (Buy WTI)',
                'Confidence': '62%',
                'Horizon': '3-5 Days',
            })
        elif spread > -3:
            signals.append({
                'Instrument': 'WTI-Brent',
                'Direction': '🔴 SHORT (Sell WTI)',
                'Confidence': '58%',
                'Horizon': '3-5 Days',
            })
        
        signals_df = pd.DataFrame(signals)
        st.dataframe(signals_df, use_container_width=True, hide_index=True)
    
    # Footer
    st.divider()
    st.caption(
        f"Oil Trading Dashboard v1.0 | "
        f"Data refreshed: {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')} | "
        f"⚠️ Signals are advisory only - human execution required"
    )
    
    # Auto-refresh
    if st.session_state.auto_refresh:
        time.sleep(5)
        st.session_state.last_refresh = datetime.now()
        st.rerun()


if __name__ == "__main__":
    main()
