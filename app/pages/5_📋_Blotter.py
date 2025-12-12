"""
Trade Blotter Page
==================
Trade history and position monitor with live P&L.
Enhanced with visual P&L displays and position breakdown.
"""

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import shared_state
from app.components.charts import BASE_LAYOUT, CHART_COLORS
from app.components.ui_components import (
    render_compact_stats,
    render_mini_pnl_card,
    render_pnl_display,
    render_position_heat_strip,
)
from app.page_utils import get_chart_config, init_page
from core.trading import PnLCalculator, PositionManager, TradeBlotter

# Initialize page
ctx = init_page(
    title="📋 Trade Blotter & Position Monitor",
    page_title="Trade Blotter | Oil Trading",
    icon="📋",
)

st.caption("Track trades, positions, and P&L in real-time")

# Initialize trading components (cached)
project_root = Path(__file__).parent.parent.parent

@st.cache_resource
def get_trading_components():
    blotter = TradeBlotter(db_path=str(project_root / "data" / "trades" / "trades.db"))
    position_mgr = PositionManager(db_path=str(project_root / "data" / "trades" / "trades.db"))
    pnl_calc = PnLCalculator()
    return blotter, position_mgr, pnl_calc

blotter, position_mgr, pnl_calc = get_trading_components()

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Positions", "📋 Trade History", "📈 P&L Analysis"])

with tab1:
    st.subheader("Open Positions")

    portfolio = ctx.portfolio.summary
    position_pnl = portfolio['positions']

    # Position Heat Strip - Quick visual overview
    if position_pnl:
        position_data = [
            {
                "symbol": p.get("symbol", p.get("ticker", "???")),
                "qty": p.get("qty", 0),
                "pnl": p.get("pnl", 0),
            }
            for p in position_pnl
        ]
        render_position_heat_strip(position_data)
        st.markdown("")  # Spacing

    positions_display = []
    for pos in position_pnl:
        direction = "🟢 Long" if pos['qty'] > 0 else "🔴 Short"
        pnl_formatted, _ = shared_state.format_pnl_with_color(pos['pnl'])
        weight = pos['notional'] / portfolio['gross_exposure'] * 100 if portfolio['gross_exposure'] > 0 else 0

        positions_display.append({
            'Symbol': pos['symbol'],
            'Direction': direction,
            'Qty': pos['qty'],
            'Avg Entry': pos['entry'],
            'Current': pos['current'],
            'Unrealized P&L': pnl_formatted,
            'P&L %': f"{pos['pnl_pct']:+.2f}%",
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

    st.divider()

    # P&L Summary with enhanced visuals
    total_unrealized = portfolio['total_pnl']
    total_realized = 39128
    total_commission = 485
    net_pnl = total_unrealized + total_realized - total_commission
    
    # Prominent Net P&L display
    col_main, col_details = st.columns([1, 2])
    
    with col_main:
        render_pnl_display(
            value=net_pnl,
            label="Net P&L (MTD)",
            show_percentage=False,
        )
    
    with col_details:
        pnl_col1, pnl_col2, pnl_col3 = st.columns(3)
        
        with pnl_col1:
            render_mini_pnl_card(
                label="Unrealized",
                value=total_unrealized,
                sub_text="Open positions",
            )
        with pnl_col2:
            render_mini_pnl_card(
                label="Realized",
                value=total_realized,
                sub_text="Closed trades",
            )
        with pnl_col3:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
                border-radius: 10px;
                padding: 14px;
                border: 1px solid rgba(51, 65, 85, 0.5);
                text-align: center;
            ">
                <div style="font-size: 10px; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">
                    Commissions
                </div>
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 18px; font-weight: 600; color: #f59e0b;">
                    -${total_commission:,}
                </div>
                <div style="font-size: 10px; color: #64748b; margin-top: 2px;">Transaction costs</div>
            </div>
            """, unsafe_allow_html=True)

    # Exposure Summary with compact stats
    st.divider()
    st.markdown("**Exposure & Risk Summary**")
    
    net_label = "Long" if portfolio['net_exposure'] > 0 else "Short"
    var_color = "#00DC82" if portfolio['var_utilization'] < 75 else "#f59e0b" if portfolio['var_utilization'] < 90 else "#ef4444"
    
    render_compact_stats([
        {"label": "Gross Exposure", "value": f"${portfolio['gross_exposure']/1e6:.2f}M"},
        {"label": "Net Exposure", "value": f"${abs(portfolio['net_exposure'])/1e6:.2f}M ({net_label})"},
        {"label": "VaR (95%)", "value": f"${portfolio['var_estimate']:,.0f}"},
        {"label": "VaR Util", "value": f"{portfolio['var_utilization']:.0f}%", "color": var_color},
    ])

    # Intraday P&L chart
    st.subheader("Intraday P&L")

    intraday = ctx.data_loader.get_intraday_prices("CL1 Comdty")

    if not intraday.empty and len(intraday) > 1:
        wti_positions = [p for p in position_pnl if p['ticker'].startswith('CL')]
        total_wti_qty = sum(p['qty'] for p in wti_positions)
        avg_entry = sum(p['qty'] * p['entry'] for p in wti_positions) / total_wti_qty if total_wti_qty != 0 else 0

        intraday['pnl'] = (intraday['price'] - avg_entry) * total_wti_qty * 1000

        fig = go.Figure()

        current_pnl = intraday['pnl'].iloc[-1]
        fill_color = 'rgba(0, 220, 130, 0.15)' if current_pnl >= 0 else 'rgba(255, 82, 82, 0.15)'
        line_color = CHART_COLORS['profit'] if current_pnl >= 0 else CHART_COLORS['loss']

        fig.add_trace(go.Scatter(
            x=intraday['timestamp'],
            y=intraday['pnl'],
            fill='tozeroy',
            fillcolor=fill_color,
            line={"color": line_color, "width": 2.5},
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
    st.subheader("Trade History")

    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
        start_date = st.date_input("Start Date", value=datetime.now().date() - timedelta(days=30))
    with filter_col2:
        end_date = st.date_input("End Date", value=datetime.now().date())
    with filter_col3:
        strategy_filter = st.selectbox("Strategy", ["All", "Momentum", "Mean Reversion", "Term Structure", "Spread"])

    trades = blotter.get_trades(
        start_date=start_date,
        end_date=end_date,
        strategy=strategy_filter if strategy_filter != "All" else None,
        limit=500
    )

    if not trades.empty:
        display_cols = ['trade_date', 'trade_time', 'instrument', 'side', 'quantity', 'price', 'commission', 'strategy']
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
            }
        )

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

        st.divider()

        export_col1, export_col2 = st.columns([1, 3])

        with export_col1:
            csv = trades.to_csv(index=False)
            st.download_button("📥 Export CSV", csv, "trades.csv", "text/csv", use_container_width=True)
    else:
        st.info("No trades found for the selected period")

with tab3:
    st.subheader("P&L Analysis")

    monthly_pnl = pd.DataFrame({
        'Month': ['Aug 2024', 'Sep 2024', 'Oct 2024', 'Nov 2024'],
        'Trades': [42, 38, 45, 47],
        'Win Rate': ['58%', '55%', '62%', '60%'],
        'Net P&L': ['+$144,150', '+$97,550', '+$177,175', '+$184,425'],
    })

    st.dataframe(monthly_pnl, use_container_width=True, hide_index=True)

    st.subheader("P&L by Strategy")

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
        fig = go.Figure()

        colors = [CHART_COLORS['profit'] if x > 0 else CHART_COLORS['loss'] for x in strategy_pnl['P&L']]

        fig.add_trace(go.Bar(
            x=strategy_pnl['Strategy'],
            y=strategy_pnl['P&L'],
            marker_color=colors,
            text=[f"${x:,.0f}" for x in strategy_pnl['P&L']],
            textposition='outside',
        ))

        fig.add_hline(y=0, line_dash='solid', line_color='rgba(255,255,255,0.3)')

        fig.update_layout(
            **BASE_LAYOUT,
            height=350,
            yaxis_title='P&L ($)',
            yaxis_tickformat='$,.0f',
        )

        st.plotly_chart(fig, use_container_width=True, config=get_chart_config())

    st.subheader("Performance Metrics (MTD)")

    # Performance metrics in a more visual format
    st.markdown("""
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;">
    """, unsafe_allow_html=True)
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.6); border-radius: 10px; padding: 16px; border: 1px solid rgba(51, 65, 85, 0.5);">
            <div style="font-size: 10px; color: #64748b; text-transform: uppercase; margin-bottom: 12px;">Risk-Adjusted</div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span style="color: #94a3b8; font-size: 12px;">Sharpe</span>
                <span style="color: #00DC82; font-weight: 600; font-family: 'IBM Plex Mono', monospace;">1.82</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #94a3b8; font-size: 12px;">Sortino</span>
                <span style="color: #00DC82; font-weight: 600; font-family: 'IBM Plex Mono', monospace;">2.45</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with metric_col2:
        st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.6); border-radius: 10px; padding: 16px; border: 1px solid rgba(51, 65, 85, 0.5);">
            <div style="font-size: 10px; color: #64748b; text-transform: uppercase; margin-bottom: 12px;">Win Metrics</div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span style="color: #94a3b8; font-size: 12px;">Win Rate</span>
                <span style="color: #f1f5f9; font-weight: 600; font-family: 'IBM Plex Mono', monospace;">60%</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #94a3b8; font-size: 12px;">Profit Factor</span>
                <span style="color: #00DC82; font-weight: 600; font-family: 'IBM Plex Mono', monospace;">2.1x</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with metric_col3:
        st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.6); border-radius: 10px; padding: 16px; border: 1px solid rgba(51, 65, 85, 0.5);">
            <div style="font-size: 10px; color: #64748b; text-transform: uppercase; margin-bottom: 12px;">Drawdown</div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span style="color: #94a3b8; font-size: 12px;">Max DD</span>
                <span style="color: #FF5252; font-weight: 600; font-family: 'IBM Plex Mono', monospace;">-$32,500</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #94a3b8; font-size: 12px;">Avg Trade</span>
                <span style="color: #00DC82; font-weight: 600; font-family: 'IBM Plex Mono', monospace;">+$3,928</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with metric_col4:
        st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.6); border-radius: 10px; padding: 16px; border: 1px solid rgba(51, 65, 85, 0.5);">
            <div style="font-size: 10px; color: #64748b; text-transform: uppercase; margin-bottom: 12px;">Best/Worst</div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span style="color: #94a3b8; font-size: 12px;">Best</span>
                <span style="color: #00DC82; font-weight: 600; font-family: 'IBM Plex Mono', monospace;">+$32,500</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #94a3b8; font-size: 12px;">Worst</span>
                <span style="color: #FF5252; font-weight: 600; font-family: 'IBM Plex Mono', monospace;">-$12,800</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
