"""
Trade Blotter Page
==================
Trade history and position monitor.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date, timedelta
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.trading import TradeBlotter, PositionManager, PnLCalculator
from core.data import DataLoader

st.set_page_config(page_title="Trade Blotter | Oil Trading", page_icon="📋", layout="wide")

# Initialize components
@st.cache_resource
def get_components():
    blotter = TradeBlotter(db_path=str(project_root / "data" / "trades" / "trades.db"))
    position_mgr = PositionManager(db_path=str(project_root / "data" / "trades" / "trades.db"))
    pnl_calc = PnLCalculator()
    data_loader = DataLoader(
        config_dir=str(project_root / "config"),
        data_dir=str(project_root / "data"),
        use_mock=True
    )
    return blotter, position_mgr, pnl_calc, data_loader

blotter, position_mgr, pnl_calc, data_loader = get_components()

st.title("📋 Trade Blotter & Position Monitor")
st.caption("Track trades, positions, and P&L")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Positions", "📋 Trade History", "📈 P&L Analysis"])

with tab1:
    # Position Monitor
    st.subheader("Open Positions")
    
    # Get current prices
    current_prices = {
        "CL1 Comdty": data_loader.get_price("CL1 Comdty"),
        "CL2 Comdty": data_loader.get_price("CL1 Comdty") - 0.25,
        "CO1 Comdty": data_loader.get_price("CO1 Comdty"),
        "XB1 Comdty": data_loader.get_price("XB1 Comdty"),
        "HO1 Comdty": data_loader.get_price("HO1 Comdty"),
    }
    
    # Mock positions for display
    positions_display = pd.DataFrame({
        'Symbol': ['CLF5', 'CLG5', 'COH5', 'XBF5', 'HOF5'],
        'Direction': ['🟢 Long', '🟢 Long', '🔴 Short', '🟢 Long', '🟢 Long'],
        'Qty': [45, 20, -15, 8, 5],
        'Avg Entry': [72.15, 72.50, 78.20, 2.15, 2.45],
        'Current': [73.45, 73.20, 77.80, 2.22, 2.52],
        'Unrealized P&L': ['+$58,500', '+$14,000', '+$6,000', '+$2,352', '+$1,470'],
        'P&L %': ['+1.80%', '+0.97%', '+0.51%', '+3.26%', '+2.86%'],
        'Weight': ['42%', '18%', '14%', '8%', '5%'],
    })
    
    st.dataframe(
        positions_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Symbol': st.column_config.TextColumn('Symbol'),
            'Direction': st.column_config.TextColumn('Direction'),
            'Qty': st.column_config.NumberColumn('Qty'),
            'Avg Entry': st.column_config.NumberColumn('Avg Entry', format='$%.2f'),
            'Current': st.column_config.NumberColumn('Current', format='$%.2f'),
            'Unrealized P&L': st.column_config.TextColumn('Unrealized P&L'),
            'P&L %': st.column_config.TextColumn('P&L %'),
            'Weight': st.column_config.TextColumn('Weight'),
        }
    )
    
    # P&L Summary
    st.divider()
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_unrealized = 82322
    total_realized = 39128
    total_commission = 485
    net_pnl = total_unrealized + total_realized - total_commission
    
    with col1:
        st.metric("Unrealized P&L", f"${total_unrealized:,}", delta="+2.1%")
    with col2:
        st.metric("Realized P&L", f"${total_realized:,}", delta="+0.8%")
    with col3:
        st.metric("Commissions", f"-${total_commission:,}")
    with col4:
        st.metric("Net P&L", f"${net_pnl:,}", delta="+2.5%")
    
    # Intraday P&L chart
    st.subheader("Intraday P&L")
    
    # Generate mock intraday P&L
    times = pd.date_range(start=datetime.now().replace(hour=9, minute=0), 
                         end=datetime.now(), freq='15min')
    
    pnl_values = np.cumsum(np.random.normal(5000, 10000, len(times)))
    pnl_values = pnl_values + abs(pnl_values.min()) + 20000  # Ensure mostly positive
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=times,
        y=pnl_values,
        fill='tozeroy',
        fillcolor='rgba(0, 210, 106, 0.3)',
        line=dict(color='#00D26A', width=2),
        name='P&L'
    ))
    
    fig.add_hline(y=0, line_dash='solid', line_color='white', line_width=1)
    
    fig.update_layout(
        template='plotly_dark',
        height=300,
        yaxis_title='P&L ($)',
        xaxis_title='Time',
        margin=dict(l=0, r=0, t=10, b=0),
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Trade History
    st.subheader("Trade History")
    
    # Filters
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
    
    with filter_col1:
        start_date = st.date_input("Start Date", value=datetime.now().date() - timedelta(days=30))
    with filter_col2:
        end_date = st.date_input("End Date", value=datetime.now().date())
    with filter_col3:
        strategy_filter = st.selectbox("Strategy", ["All", "Momentum", "Mean Reversion", "Term Structure", "Spread", "Signal-Based"])
    with filter_col4:
        instrument_filter = st.selectbox("Instrument", ["All", "WTI (CL)", "Brent (CO)", "RBOB (XB)", "Heating Oil (HO)"])
    
    # Get trades
    trades = blotter.get_trades(
        start_date=start_date,
        end_date=end_date,
        strategy=strategy_filter if strategy_filter != "All" else None,
        limit=500
    )
    
    if not trades.empty:
        # Display trades
        display_cols = ['trade_date', 'trade_time', 'instrument', 'side', 'quantity', 'price', 'commission', 'strategy', 'notes']
        available_cols = [c for c in display_cols if c in trades.columns]
        
        st.dataframe(
            trades[available_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                'trade_date': st.column_config.DateColumn('Date'),
                'trade_time': st.column_config.TextColumn('Time'),
                'instrument': st.column_config.TextColumn('Symbol'),
                'side': st.column_config.TextColumn('Side'),
                'quantity': st.column_config.NumberColumn('Qty'),
                'price': st.column_config.NumberColumn('Price', format='$%.2f'),
                'commission': st.column_config.NumberColumn('Comm', format='$%.2f'),
                'strategy': st.column_config.TextColumn('Strategy'),
                'notes': st.column_config.TextColumn('Notes'),
            }
        )
        
        # Trade statistics
        st.divider()
        st.subheader("Trade Statistics")
        
        stats = blotter.get_trade_statistics(start_date=start_date, end_date=end_date)
        
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        
        with stat_col1:
            st.metric("Total Trades", stats['total_trades'])
        with stat_col2:
            st.metric("Buy / Sell", f"{stats['buy_trades']} / {stats['sell_trades']}")
        with stat_col3:
            st.metric("Total Volume", f"{stats['total_volume']:,} contracts")
        with stat_col4:
            st.metric("Total Commissions", f"${stats['total_commission']:,.2f}")
        
        # Export buttons
        st.divider()
        
        export_col1, export_col2, export_col3 = st.columns([1, 1, 2])
        
        with export_col1:
            if st.button("📥 Export CSV", use_container_width=True):
                csv = trades.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv,
                    "trades.csv",
                    "text/csv",
                    use_container_width=True
                )
        
        with export_col2:
            if st.button("📥 Export JSON", use_container_width=True):
                json_str = trades.to_json(orient='records', date_format='iso')
                st.download_button(
                    "Download JSON",
                    json_str,
                    "trades.json",
                    "application/json",
                    use_container_width=True
                )
    else:
        st.info("No trades found for the selected period")

with tab3:
    # P&L Analysis
    st.subheader("P&L Analysis")
    
    # Monthly P&L summary
    monthly_pnl = pd.DataFrame({
        'Month': ['Aug 2024', 'Sep 2024', 'Oct 2024', 'Nov 2024'],
        'Trades': [42, 38, 45, 47],
        'Win Rate': ['58%', '55%', '62%', '60%'],
        'Gross P&L': ['+$145,200', '+$98,500', '+$178,300', '+$185,600'],
        'Commissions': ['-$1,050', '-$950', '-$1,125', '-$1,175'],
        'Net P&L': ['+$144,150', '+$97,550', '+$177,175', '+$184,425'],
    })
    
    st.dataframe(monthly_pnl, use_container_width=True, hide_index=True)
    
    # P&L by Strategy
    st.subheader("P&L by Strategy")
    
    strategy_pnl = pd.DataFrame({
        'Strategy': ['Momentum', 'Spread', 'Signal-Based', 'Mean Reversion', 'Discretionary'],
        'Trades': [22, 12, 8, 3, 2],
        'Win Rate': [64, 58, 63, 67, 50],
        'P&L': [92300, 45200, 38100, 8200, 1800],
    })
    
    # Bar chart
    fig = go.Figure()
    
    colors = ['#00D26A' if x > 0 else '#FF4B4B' for x in strategy_pnl['P&L']]
    
    fig.add_trace(go.Bar(
        x=strategy_pnl['Strategy'],
        y=strategy_pnl['P&L'],
        marker_color=colors,
        text=[f"${x:,.0f}" for x in strategy_pnl['P&L']],
        textposition='outside',
    ))
    
    fig.update_layout(
        template='plotly_dark',
        height=350,
        yaxis_title='P&L ($)',
        margin=dict(l=0, r=0, t=10, b=0),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance metrics
    st.subheader("Performance Metrics (MTD)")
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric("Sharpe Ratio", "1.82")
        st.metric("Win Rate", "60%")
    
    with metric_col2:
        st.metric("Sortino Ratio", "2.45")
        st.metric("Profit Factor", "2.1")
    
    with metric_col3:
        st.metric("Max Drawdown", "-$32,500")
        st.metric("Avg Trade", "+$3,928")
    
    with metric_col4:
        st.metric("Best Trade", "+$32,500")
        st.metric("Worst Trade", "-$12,800")
    
    # Cumulative P&L chart
    st.subheader("Cumulative P&L (30 Days)")
    
    days = pd.date_range(end=datetime.now(), periods=30, freq='D')
    daily_pnl = np.random.normal(5000, 8000, 30)
    cumulative = np.cumsum(daily_pnl)
    
    fig2 = go.Figure()
    
    fig2.add_trace(go.Scatter(
        x=days,
        y=cumulative,
        fill='tozeroy',
        fillcolor='rgba(0, 163, 224, 0.3)',
        line=dict(color='#00A3E0', width=2),
        name='Cumulative P&L'
    ))
    
    fig2.add_hline(y=0, line_dash='solid', line_color='white')
    
    fig2.update_layout(
        template='plotly_dark',
        height=350,
        yaxis_title='Cumulative P&L ($)',
        margin=dict(l=0, r=0, t=10, b=0),
    )
    
    st.plotly_chart(fig2, use_container_width=True)
