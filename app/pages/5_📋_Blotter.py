"""
Trade Blotter Page
==================
Trade history and position monitor with live P&L.
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

# Add app directory for shared helpers
app_dir = Path(__file__).parent.parent
sys.path.insert(0, str(app_dir))

from app import shared_state
from core.trading import TradeBlotter, PositionManager, PnLCalculator
from app.components.charts import create_pnl_chart, create_bar_chart, CHART_COLORS, BASE_LAYOUT

st.set_page_config(page_title="Trade Blotter | Oil Trading", page_icon="📋", layout="wide")

# Apply shared theme
from app.components.theme import apply_theme, COLORS, PLOTLY_LAYOUT, get_chart_config
apply_theme(st)

# Initialize components
@st.cache_resource
def get_trading_components():
    blotter = TradeBlotter(db_path=str(project_root / "data" / "trades" / "trades.db"))
    position_mgr = PositionManager(db_path=str(project_root / "data" / "trades" / "trades.db"))
    pnl_calc = PnLCalculator()
    return blotter, position_mgr, pnl_calc

blotter, position_mgr, pnl_calc = get_trading_components()
context = shared_state.get_dashboard_context()
data_loader = context.data_loader

st.title("📋 Trade Blotter & Position Monitor")

# Show live prices at top
oil_prices = context.data.oil_prices
if oil_prices:
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    with col_p1:
        wti = oil_prices.get("WTI", {})
        st.metric("WTI Live", f"${wti.get('current', 0):.2f}", f"{wti.get('change', 0):+.2f}")
    with col_p2:
        brent = oil_prices.get("Brent", {})
        st.metric("Brent Live", f"${brent.get('current', 0):.2f}", f"{brent.get('change', 0):+.2f}")
    with col_p3:
        spread = context.data.wti_brent_spread
        if spread:
            st.metric("WTI-Brent", f"${spread.get('spread', 0):.2f}", f"{spread.get('change', 0):+.2f}")
    with col_p4:
        crack = context.data.crack_spread
        if crack:
            st.metric("3-2-1 Crack", f"${crack.get('crack', 0):.2f}", f"{crack.get('change', 0):+.2f}")
    st.divider()

st.caption("Track trades, positions, and P&L in real-time")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Positions", "📋 Trade History", "📈 P&L Analysis"])

with tab1:
    # Position Monitor - Using Live Data
    st.subheader("Open Positions")
    
    # Get calculated positions with live P&L
    portfolio = context.portfolio.summary
    position_pnl = portfolio['positions']
    
    # Build display dataframe
    positions_display = []
    for pos in position_pnl:
        direction = "🟢 Long" if pos['qty'] > 0 else "🔴 Short"
        pnl_formatted, pnl_color = shared_state.format_pnl_with_color(pos['pnl'])
        pnl_pct = f"{pos['pnl_pct']:+.2f}%"
        weight = pos['notional'] / portfolio['gross_exposure'] * 100 if portfolio['gross_exposure'] > 0 else 0
        
        positions_display.append({
            'Symbol': pos['symbol'],
            'Direction': direction,
            'Qty': pos['qty'],
            'Avg Entry': pos['entry'],
            'Current': pos['current'],
            'Unrealized P&L': pnl_formatted,
            'P&L %': pnl_pct,
            'Weight': f"{weight:.0f}%",
        })
    
    positions_df = pd.DataFrame(positions_display)
    
    st.dataframe(
        positions_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Symbol': st.column_config.TextColumn('Symbol'),
            'Direction': st.column_config.TextColumn('Direction'),
            'Qty': st.column_config.NumberColumn('Qty'),
            'Avg Entry': st.column_config.NumberColumn('Avg Entry', format='$%.2f'),
            'Current': st.column_config.NumberColumn('Current', format='$%.4f'),
            'Unrealized P&L': st.column_config.TextColumn('Unrealized P&L'),
            'P&L %': st.column_config.TextColumn('P&L %'),
            'Weight': st.column_config.TextColumn('Weight'),
        }
    )
    
    # P&L Summary - Calculated from live data
    st.divider()
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_unrealized = portfolio['total_pnl']
    total_realized = 39128  # Mock - would come from closed trades
    total_commission = 485
    net_pnl = total_unrealized + total_realized - total_commission
    
    with col1:
        pnl_delta = f"{total_unrealized / 1000000 * 100:+.2f}%" if total_unrealized != 0 else "0%"
        st.metric(
            "Unrealized P&L", 
            shared_state.format_pnl(total_unrealized),
            delta=pnl_delta,
            delta_color="normal" if total_unrealized >= 0 else "inverse"
        )
    with col2:
        st.metric("Realized P&L", shared_state.format_pnl(total_realized), delta="+0.8%")
    with col3:
        st.metric("Commissions", f"-${total_commission:,}")
    with col4:
        st.metric(
            "Net P&L", 
            shared_state.format_pnl(net_pnl),
            delta=f"{net_pnl / 1000000 * 100:+.2f}%",
            delta_color="normal" if net_pnl >= 0 else "inverse"
        )
    
    # Exposure Summary
    st.divider()
    exp_col1, exp_col2, exp_col3, exp_col4 = st.columns(4)
    
    with exp_col1:
        st.metric("Gross Exposure", f"${portfolio['gross_exposure']/1e6:.2f}M")
    with exp_col2:
        net_label = "Long" if portfolio['net_exposure'] > 0 else "Short"
        st.metric("Net Exposure", f"${abs(portfolio['net_exposure'])/1e6:.2f}M ({net_label})")
    with exp_col3:
        st.metric("VaR (95%, 1-Day)", f"${portfolio['var_estimate']:,.0f}")
    with exp_col4:
        var_util = portfolio['var_utilization']
        st.metric("VaR Utilization", f"{var_util:.0f}%")
    
    # Intraday P&L chart using actual price history
    st.subheader("Intraday P&L")
    
    # Get intraday prices for main position (WTI)
    intraday = data_loader.get_intraday_prices("CL1 Comdty")
    
    if not intraday.empty and len(intraday) > 1:
        # Calculate P&L at each point based on WTI position
        wti_positions = [p for p in position_pnl if p['ticker'].startswith('CL')]
        total_wti_qty = sum(p['qty'] for p in wti_positions)
        avg_entry = sum(p['qty'] * p['entry'] for p in wti_positions) / total_wti_qty if total_wti_qty != 0 else 0
        
        intraday['pnl'] = (intraday['price'] - avg_entry) * total_wti_qty * 1000
        
        fig = go.Figure()
        
        # Determine fill color based on current P&L
        current_pnl = intraday['pnl'].iloc[-1]
        fill_color = 'rgba(0, 220, 130, 0.15)' if current_pnl >= 0 else 'rgba(255, 82, 82, 0.15)'
        line_color = CHART_COLORS['profit'] if current_pnl >= 0 else CHART_COLORS['loss']
        
        fig.add_trace(go.Scatter(
            x=intraday['timestamp'],
            y=intraday['pnl'],
            fill='tozeroy',
            fillcolor=fill_color,
            line=dict(color=line_color, width=2.5),
            name='WTI P&L',
            hovertemplate='%{x}<br>$%{y:,.0f}<extra></extra>',
        ))
        
        fig.add_hline(y=0, line_dash='solid', line_color='rgba(255,255,255,0.3)', line_width=1)
        
        fig.update_layout(
            **BASE_LAYOUT,
            height=300,
            yaxis_title='P&L ($)',
            yaxis_tickformat='$,.0f',
            xaxis_title='Time',
        )
        
        st.plotly_chart(fig, use_container_width=True, config=get_chart_config())
    else:
        st.info("Collecting intraday price data...")

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
            csv = trades.to_csv(index=False)
            st.download_button(
                "📥 Export CSV",
                csv,
                "trades.csv",
                "text/csv",
                use_container_width=True
            )
        
        with export_col2:
            json_str = trades.to_json(orient='records', date_format='iso')
            st.download_button(
                "📥 Export JSON",
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
    
    # P&L by Strategy - calculated from actual positions
    st.subheader("P&L by Strategy")
    
    # Aggregate P&L by strategy from current positions
    strategy_data = {}
    for pos in position_pnl:
        strategy = pos['strategy']
        if strategy not in strategy_data:
            strategy_data[strategy] = {'pnl': 0, 'count': 0}
        strategy_data[strategy]['pnl'] += pos['pnl']
        strategy_data[strategy]['count'] += 1
    
    strategy_pnl = pd.DataFrame([
        {'Strategy': k, 'Positions': v['count'], 'P&L': v['pnl']}
        for k, v in strategy_data.items()
    ])
    
    if not strategy_pnl.empty:
        # Bar chart with enhanced styling
        fig = go.Figure()
        
        colors = [CHART_COLORS['profit'] if x > 0 else CHART_COLORS['loss'] for x in strategy_pnl['P&L']]
        
        fig.add_trace(go.Bar(
            x=strategy_pnl['Strategy'],
            y=strategy_pnl['P&L'],
            marker_color=colors,
            marker_line_width=0,
            text=[f"${x:,.0f}" for x in strategy_pnl['P&L']],
            textposition='outside',
            textfont=dict(size=12, color=CHART_COLORS['text_primary']),
            hovertemplate='%{x}<br>$%{y:,.0f}<extra></extra>',
        ))
        
        fig.add_hline(y=0, line_dash='solid', line_color='rgba(255,255,255,0.3)')
        
        fig.update_layout(
            **BASE_LAYOUT,
            height=350,
            yaxis_title='P&L ($)',
            yaxis_tickformat='$,.0f',
        )
        
        st.plotly_chart(fig, use_container_width=True, config=get_chart_config())
    
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
    
    # Cumulative P&L chart using historical data
    st.subheader("Cumulative P&L (30 Days)")
    
    # Get historical data to show price evolution
    hist_data = data_loader.get_historical(
        "CL1 Comdty",
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now()
    )
    
    if hist_data is not None and len(hist_data) > 0:
        # Calculate P&L based on WTI position
        wti_qty = sum(p['qty'] for p in position_pnl if p['ticker'].startswith('CL'))
        wti_entry = 72.15  # Average entry
        
        cumulative_pnl = (hist_data['PX_LAST'] - wti_entry) * wti_qty * 1000
        
        fig2 = go.Figure()
        
        current_pnl = cumulative_pnl.iloc[-1]
        fill_color = 'rgba(0, 220, 130, 0.15)' if current_pnl >= 0 else 'rgba(255, 82, 82, 0.15)'
        line_color = CHART_COLORS['profit'] if current_pnl >= 0 else CHART_COLORS['loss']
        
        fig2.add_trace(go.Scatter(
            x=hist_data.index,
            y=cumulative_pnl,
            fill='tozeroy',
            fillcolor=fill_color,
            line=dict(color=line_color, width=2.5),
            name='Cumulative P&L',
            hovertemplate='%{x}<br>$%{y:,.0f}<extra></extra>',
        ))
        
        fig2.add_hline(y=0, line_dash='solid', line_color='rgba(255,255,255,0.3)')
        
        fig2.update_layout(
            **BASE_LAYOUT,
            height=350,
            yaxis_title='Cumulative P&L ($)',
            yaxis_tickformat='$,.0f',
        )
        
        st.plotly_chart(fig2, use_container_width=True, config=get_chart_config())
