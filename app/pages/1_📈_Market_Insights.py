"""
Market Insights Page
====================
Comprehensive market analysis and intelligence.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from app import shared_state
from core.analytics import CurveAnalyzer, SpreadAnalyzer, FundamentalAnalyzer
from core.data.bloomberg import DataUnavailableError
from app.components.charts import (
    create_candlestick_chart,
    create_futures_curve_chart,
    create_volume_chart,
    create_open_interest_chart,
    create_bar_chart,
    CHART_COLORS,
    BASE_LAYOUT,
)

st.set_page_config(page_title="Market Insights | Oil Trading", page_icon="📈", layout="wide")

# Apply shared theme
from app.components.theme import apply_theme, COLORS, PLOTLY_LAYOUT, get_chart_config
apply_theme(st)

# Auto-refresh configuration
REFRESH_INTERVAL_SECONDS = 15  # Refresh every 15 seconds

# Initialize last refresh time in session state
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True

# Force refresh on data context to get latest data
context = shared_state.get_dashboard_context(lookback_days=180, force_refresh=True)
data_loader = context.data_loader
price_cache = context.price_cache
curve_analyzer = CurveAnalyzer()
spread_analyzer = SpreadAnalyzer()
fundamental_analyzer = FundamentalAnalyzer()

# Check data mode
connection_status = data_loader.get_connection_status()
data_mode = connection_status.get("data_mode", "disconnected")

# Header with live status and controls
header_col1, header_col2, header_col3 = st.columns([3, 1, 1])

with header_col1:
    st.title("📈 Market Insights")

with header_col2:
    auto_refresh = st.toggle("Auto Refresh (15s)", value=st.session_state.auto_refresh, key="auto_refresh_toggle")
    st.session_state.auto_refresh = auto_refresh

with header_col3:
    if st.button("🔄 Refresh Now", use_container_width=True):
        st.session_state.last_refresh = datetime.now()
        shared_state.invalidate_context_cache()
        st.rerun()

# Live status bar
if data_mode == "live":
    time_since_update = (datetime.now() - st.session_state.last_refresh).seconds
    st.markdown(
        f"""<div style="display: flex; align-items: center; gap: 10px; padding: 8px 12px; 
        background: linear-gradient(90deg, rgba(0,210,130,0.15) 0%, rgba(0,210,130,0.05) 100%); 
        border-left: 3px solid #00D282; border-radius: 4px; margin-bottom: 1rem;">
        <span style="color: #00D282; font-weight: 600;">🟢 LIVE</span>
        <span style="color: #94A3B8;">Bloomberg Connected</span>
        <span style="color: #64748B; margin-left: auto;">Last update: {st.session_state.last_refresh.strftime('%H:%M:%S')} ({time_since_update}s ago)</span>
        </div>""",
        unsafe_allow_html=True
    )
elif data_mode == "mock":
    st.markdown(
        f"""<div style="display: flex; align-items: center; gap: 10px; padding: 8px 12px; 
        background: linear-gradient(90deg, rgba(245,158,11,0.15) 0%, rgba(245,158,11,0.05) 100%); 
        border-left: 3px solid #F59E0B; border-radius: 4px; margin-bottom: 1rem;">
        <span style="color: #F59E0B; font-weight: 600;">🟡 SIMULATED</span>
        <span style="color: #94A3B8;">Development Mode</span>
        <span style="color: #64748B; margin-left: auto;">Last update: {st.session_state.last_refresh.strftime('%H:%M:%S')}</span>
        </div>""",
        unsafe_allow_html=True
    )
elif data_mode == "disconnected":
    st.error("🔴 Bloomberg Terminal not connected. Live data required.")
    st.info(f"Connection error: {connection_status.get('connection_error', 'Unknown')}")
    st.stop()
else:
    st.warning(f"⚠️ Data mode: {data_mode}")

# Tabs for different analysis views
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Price Action",
    "📈 Term Structure",
    "🔥 Crack Spreads",
    "📦 Inventory",
    "🌍 OPEC Monitor"
])

with tab1:
    # Price Action Tab
    st.subheader("Price Action & Structure")
    
    # Instrument definitions
    # Note: Dubai uses 2nd month swap (DAT2) to avoid BALMO (Balance of Month)
    # Note: WTI uses ICE prices (ENA1 Comdty) not NYMEX (CL1)
    instruments = {
        "Brent": {"ticker": "CO1 Comdty", "name": "Brent Crude Oil (ICE)", "icon": "🇬🇧"},
        "WTI": {"ticker": "ENA1 Comdty", "name": "WTI Crude Oil (ICE)", "icon": "🇺🇸"},
        "Dubai": {"ticker": "DAT2 Comdty", "name": "Dubai Crude Swap (M2)", "icon": "🇦🇪"},
    }
    
    # Instrument tabs
    brent_tab, wti_tab, dubai_tab = st.tabs([
        f"{instruments['Brent']['icon']} Brent",
        f"{instruments['WTI']['icon']} WTI",
        f"{instruments['Dubai']['icon']} Dubai"
    ])
    
    def render_price_action(instrument_key: str):
        """Render price action charts for an instrument."""
        inst = instruments[instrument_key]
        ticker = inst["ticker"]
        name = inst["name"]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Price chart
            chart_title = f"**{name} - Daily Chart**"
            if data_mode == "mock":
                chart_title += " _(simulated)_"
            st.markdown(chart_title)
            
            # Get historical data
            hist_data = None
            data_warning_shown = False
            try:
                hist_data = data_loader.get_historical(
                    ticker,
                    start_date=datetime.now() - timedelta(days=180),
                    end_date=datetime.now()
                )
            except DataUnavailableError:
                st.warning(f"No historical data available for {name}.")
                data_warning_shown = True
            except Exception as exc:
                st.error(f"Failed to load history for {name}: {exc}")
                data_warning_shown = True
            
            if hist_data is not None and not hist_data.empty:
                # Candlestick chart
                fig = create_candlestick_chart(
                    data=hist_data,
                    title="",
                    height=450,
                    show_volume=False,
                    show_ma=True,
                    ma_periods=[20, 50],
                )
                
                st.plotly_chart(fig, use_container_width=True, config=get_chart_config())
                
                # Volume and Open Interest charts side by side
                vol_col, oi_col = st.columns(2)
                
                with vol_col:
                    st.markdown("**Volume**")
                    if 'PX_VOLUME' in hist_data.columns:
                        vol_fig = create_volume_chart(hist_data, height=120)
                        st.plotly_chart(vol_fig, use_container_width=True, config=get_chart_config())
                
                with oi_col:
                    st.markdown("**Open Interest**")
                    if 'OPEN_INT' in hist_data.columns and hist_data['OPEN_INT'].notna().any():
                        oi_fig = create_open_interest_chart(hist_data, height=120)
                        st.plotly_chart(oi_fig, use_container_width=True, config=get_chart_config())
                    else:
                        st.caption("Open interest data not available")
            else:
                if not data_warning_shown:
                    st.info(f"Historical data unavailable for {name}.")
                hist_data = None
        
        with col2:
            # Get live price first
            live_price = price_cache.get(ticker)
            
            # Display live price prominently at the top
            if live_price:
                # Calculate change from previous close
                prev_close = None
                daily_change = 0
                daily_change_pct = 0
                
                if hist_data is not None and not hist_data.empty and len(hist_data) >= 2:
                    prev_close = hist_data['PX_LAST'].iloc[-2]
                    daily_change = live_price - prev_close
                    daily_change_pct = (daily_change / prev_close * 100) if prev_close else 0
                
                # Live price with change
                change_color = "#00DC82" if daily_change >= 0 else "#FF5252"
                change_sign = "+" if daily_change >= 0 else ""
                
                st.markdown(
                    f"""<div style="background: linear-gradient(135deg, rgba(0,163,224,0.1) 0%, rgba(0,163,224,0.05) 100%); 
                    padding: 16px; border-radius: 8px; border-left: 4px solid #00A3E0; margin-bottom: 16px;">
                    <div style="color: #94A3B8; font-size: 12px; margin-bottom: 4px;">LIVE PRICE</div>
                    <div style="color: #E2E8F0; font-size: 28px; font-weight: 700; font-family: 'IBM Plex Mono', monospace;">${live_price:.2f}</div>
                    <div style="color: {change_color}; font-size: 14px; font-weight: 600;">{change_sign}{daily_change:.2f} ({change_sign}{daily_change_pct:.2f}%)</div>
                    </div>""",
                    unsafe_allow_html=True
                )
                
                # Yesterday's close
                if prev_close:
                    st.markdown(
                        f"""<div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.1);">
                        <span style="color: #94A3B8;">Yesterday Close</span>
                        <span style="color: #E2E8F0; font-family: 'IBM Plex Mono', monospace;">${prev_close:.2f}</span>
                        </div>""",
                        unsafe_allow_html=True
                    )
            
            st.markdown("**Key Levels**")
            
            if hist_data is not None and not hist_data.empty:
                current_price = live_price if live_price else hist_data['PX_LAST'].iloc[-1]
                high_range = hist_data['PX_HIGH'].max()
                low_range = hist_data['PX_LOW'].min()
                
                # Today's OHLC from last bar
                today_open = hist_data['PX_OPEN'].iloc[-1]
                today_high = hist_data['PX_HIGH'].iloc[-1]
                today_low = hist_data['PX_LOW'].iloc[-1]
                
                st.metric("Today Open", f"${today_open:.2f}")
                st.metric("Today High", f"${today_high:.2f}")
                st.metric("Today Low", f"${today_low:.2f}")
                
                st.divider()
                
                st.metric("180D High", f"${high_range:.2f}")
                st.metric("180D Low", f"${low_range:.2f}")
                
                # Price position
                if high_range != low_range:
                    position = (current_price - low_range) / (high_range - low_range) * 100
                    st.progress(int(min(max(position, 0), 100)) / 100, text=f"Range Position: {position:.0f}%")
            
            st.divider()
            
            st.markdown("**Technical Indicators**")
            
            # Calculate technical indicators
            if hist_data is not None and not hist_data.empty and len(hist_data) >= 14:
                closes = hist_data['PX_LAST']
                
                # RSI calculation
                delta = closes.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                current_rsi = rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50
                
                # Trend determination
                sma20 = closes.rolling(20).mean().iloc[-1] if len(closes) >= 20 else closes.iloc[-1]
                sma50 = closes.rolling(50).mean().iloc[-1] if len(closes) >= 50 else closes.iloc[-1]
                trend = "Bullish" if sma20 > sma50 else "Bearish" if sma20 < sma50 else "Neutral"
                
                st.text(f"RSI (14): {current_rsi:.1f}")
                st.text(f"Trend: {trend}")
                st.text(f"SMA20: ${sma20:.2f}")
                st.text(f"SMA50: ${sma50:.2f}")
            else:
                st.text("Insufficient data for indicators")
    
    with brent_tab:
        render_price_action("Brent")
    
    with wti_tab:
        render_price_action("WTI")
    
    with dubai_tab:
        render_price_action("Dubai")

with tab2:
    # Term Structure Tab
    st.subheader("Futures Curve Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # WTI vs Brent curves via shared context cache
        wti_curve = context.data.futures_curve
        brent_curve = context.data.brent_curve
        
        # Create enhanced futures curve chart with both WTI and Brent
        fig = create_futures_curve_chart(
            curve_data=brent_curve,
            secondary_curve=wti_curve,
            title="WTI vs Brent Futures Curve",
            height=400,
        )
        
        st.plotly_chart(fig, use_container_width=True, config=get_chart_config())
        
        # Calendar spreads
        st.markdown("**Calendar Spreads**")
        spreads = curve_analyzer.calculate_calendar_spreads(wti_curve)
        
        st.dataframe(
            spreads,
            use_container_width=True,
            hide_index=True,
            column_config={
                'spread_name': 'Spread',
                'spread_value': st.column_config.NumberColumn('Value', format='$%.2f'),
                'front_price': st.column_config.NumberColumn('Front', format='$%.2f'),
                'back_price': st.column_config.NumberColumn('Back', format='$%.2f'),
            }
        )
    
    with col2:
        st.markdown("**Curve Analysis**")
        
        curve_metrics = curve_analyzer.analyze_curve(wti_curve)
        
        st.metric("Structure", curve_metrics['structure'])
        st.metric("M1-M2 Spread", f"${curve_metrics['m1_m2_spread']:.2f}")
        st.metric("Roll Yield (Ann.)", f"{curve_metrics['roll_yield_annual_pct']:.1f}%")
        st.metric("Curve Slope", f"{curve_metrics['overall_slope']:.4f}")
        
        st.divider()
        
        # Roll yield
        roll_yield = curve_analyzer.calculate_roll_yield(wti_curve)
        
        st.markdown("**Roll Yield Analysis**")
        st.text(f"Roll Cost: ${roll_yield['roll_cost']:.2f}")
        st.text(f"Roll Yield: ${roll_yield['roll_yield']:.2f}")
        st.text(f"Annual Roll Yield: {roll_yield['roll_yield_annual_pct']:.1f}%")
        st.text(f"Curve Carry: {roll_yield['curve_carry']}")

with tab3:
    # Crack Spreads Tab
    st.subheader("Crack Spread Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**3-2-1 Crack Spread (USGC)**")
        
        # Get LIVE prices using price cache
        wti = price_cache.get("CL1 Comdty")
        rbob = price_cache.get("XB1 Comdty")
        ho = price_cache.get("HO1 Comdty")
        brent = price_cache.get("CO1 Comdty")
        
        if wti and rbob and ho:
            crack_321 = spread_analyzer.calculate_crack_spread(wti, rbob, ho, "3-2-1")
            
            # Get crack spread change from context
            crack_data = context.data.crack_spread
            crack_change = crack_data.get('change', 0) if crack_data else 0
            
            st.metric(
                "Current",
                f"${crack_321['crack_spread']:.2f}/bbl",
                delta=f"{crack_change:+.2f}"
            )
            
            # Component breakdown visualization
            st.markdown("**Spread Components**")
            rbob_bbl = rbob * 42
            ho_bbl = ho * 42
            
            component_fig = go.Figure()
            component_fig.add_trace(go.Bar(
                x=['RBOB×2', 'HO×1', 'WTI×3', 'Crack'],
                y=[rbob_bbl * 2, ho_bbl, -wti * 3, crack_321['crack_spread'] * 3],
                marker_color=[CHART_COLORS['profit'], CHART_COLORS['profit'], CHART_COLORS['loss'], CHART_COLORS['primary']],
                text=[f"${rbob_bbl*2:.2f}", f"${ho_bbl:.2f}", f"-${wti*3:.2f}", f"${crack_321['crack_spread']*3:.2f}"],
                textposition='outside',
                textfont=dict(size=12, color=CHART_COLORS['text_primary']),
                marker_line_width=0,
            ))
            
            component_fig.update_layout(
                **BASE_LAYOUT,
                height=250,
                yaxis_title='$/bbl equivalent',
            )
            
            st.plotly_chart(component_fig, use_container_width=True, config=get_chart_config())
            
            # Metrics
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric("Refining Margin", f"${crack_321['crack_spread']:.2f}/bbl")
            with m_col2:
                st.metric("Margin %", f"{crack_321['margin_pct']:.1f}%")
            with m_col3:
                st.metric("Product Value", f"${(rbob_bbl*2 + ho_bbl)/3:.2f}/bbl")
        else:
            st.warning("Price data unavailable for crack spread calculation")
    
    with col2:
        st.markdown("**Live Spreads**")
        
        # Calculate LIVE regional differentials
        if wti and brent:
            brent_wti_spread = brent - wti
            spread_data = context.data.wti_brent_spread
            spread_change = spread_data.get('change', 0) if spread_data else 0
            
            st.metric("Brent-WTI", f"${brent_wti_spread:.2f}", delta=f"{spread_change:+.2f}")
        
        st.divider()
        
        st.markdown("**Live Component Prices**")
        if wti:
            st.metric("WTI Crude", f"${wti:.2f}/bbl")
        if brent:
            st.metric("Brent Crude", f"${brent:.2f}/bbl")
        if rbob:
            st.metric("RBOB Gasoline", f"${rbob:.4f}/gal")
        if ho:
            st.metric("Heating Oil", f"${ho:.4f}/gal")

with tab4:
    # Inventory Tab
    st.subheader("Inventory Analytics")
    
    eia_data = data_loader.get_eia_inventory()
    
    if eia_data is None or (hasattr(eia_data, 'empty') and eia_data.empty):
        st.info("📊 EIA inventory data requires Bloomberg connection or external data feed.")
        st.markdown("""
        **Data Sources:**
        - EIA Weekly Petroleum Status Report
        - Bloomberg ECST <GO> function
        - API integration with EIA.gov
        """)
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**EIA Weekly Crude Inventory**")
            
            fig = go.Figure()
            
            # Inventory level with gradient fill
            fig.add_trace(go.Scatter(
                x=eia_data.index,
                y=eia_data['inventory_mmb'],
                name='Inventory',
                line=dict(color=CHART_COLORS['primary'], width=2.5),
                fill='tozeroy',
                fillcolor='rgba(0, 163, 224, 0.1)',
                hovertemplate='%{x}<br>%{y:.1f} MMbbl<extra></extra>',
            ))
            
            # 5-year range
            mean = eia_data['inventory_mmb'].mean()
            fig.add_hline(y=mean, line_dash='dash', line_color=CHART_COLORS['ma_fast'],
                         annotation_text='5-Year Avg')
            
            fig.update_layout(
                **BASE_LAYOUT,
                height=350,
                yaxis_title='Inventory (MMbbl)',
                yaxis_tickformat='.0f',
            )
            
            st.plotly_chart(fig, use_container_width=True, config=get_chart_config())
            
            # Weekly change
            st.markdown("**Weekly Change**")
            
            change_fig = go.Figure()
            change_fig.add_trace(go.Bar(
                x=eia_data.index,
                y=eia_data['change_mmb'],
                marker_color=[CHART_COLORS['profit'] if x < 0 else CHART_COLORS['loss'] for x in eia_data['change_mmb']],
                marker_line_width=0,
                name='Change',
                hovertemplate='%{x}<br>%{y:+.1f} MMbbl<extra></extra>',
            ))
            
            fig.add_hline(y=0, line_dash='solid', line_color='rgba(255,255,255,0.3)', line_width=1)
            
            change_fig.update_layout(
                **BASE_LAYOUT,
                height=200,
                yaxis_title='Change (MMbbl)',
            )
            
            st.plotly_chart(change_fig, use_container_width=True, config=get_chart_config())
        
        with col2:
            st.markdown("**Latest Report**")
            
            latest = eia_data.iloc[-1]
            
            inv_analysis = fundamental_analyzer.analyze_inventory(
                current_level=latest['inventory_mmb'],
                change=latest['change_mmb'],
                expectation=latest['expectation_mmb']
            )
            
            st.metric("Current Level", f"{inv_analysis['current_level']:.1f} MMbbl")
            st.metric("Change", f"{inv_analysis['change']:+.1f} MMbbl")
            st.metric("Surprise", f"{inv_analysis['surprise']:+.1f} MMbbl")
            
            # Signal
            if "Bullish" in inv_analysis['surprise_signal']:
                st.success(inv_analysis['surprise_signal'])
            elif "Bearish" in inv_analysis['surprise_signal']:
                st.error(inv_analysis['surprise_signal'])
            else:
                st.info(inv_analysis['surprise_signal'])
            
            st.divider()
            
            st.markdown("**Level Analysis**")
            st.text(f"Percentile: {inv_analysis['percentile']:.0f}th")
            st.text(f"vs 5-Year Avg: {inv_analysis['vs_5yr_avg']:+.1f} MMbbl")
            st.text(f"Assessment: {inv_analysis['level_signal']}")

with tab5:
    # OPEC Monitor Tab
    st.subheader("OPEC+ Production Monitor")
    
    opec_data = data_loader.get_opec_production()
    
    if opec_data is None or (hasattr(opec_data, 'empty') and opec_data.empty):
        st.info("📊 OPEC production data requires Bloomberg connection or external data feed.")
        st.markdown("""
        **Data Sources:**
        - OPEC Monthly Oil Market Report
        - IEA Oil Market Report
        - Bloomberg OPEC <GO> function
        """)
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**Production vs Quota by Country**")
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Quota',
                x=opec_data['country'],
                y=opec_data['quota_mbpd'],
                marker_color=CHART_COLORS['primary'],
                marker_line_width=0,
                hovertemplate='Quota: %{y:.2f} mb/d<extra></extra>',
            ))
            
            fig.add_trace(go.Bar(
                name='Actual',
                x=opec_data['country'],
                y=opec_data['actual_mbpd'],
                marker_color=CHART_COLORS['profit'],
                marker_line_width=0,
                hovertemplate='Actual: %{y:.2f} mb/d<extra></extra>',
            ))
            
            fig.update_layout(
                **BASE_LAYOUT,
                height=400,
                barmode='group',
                yaxis_title='Production (mb/d)',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            )
            
            st.plotly_chart(fig, use_container_width=True, config=get_chart_config())
            
            # Compliance table
            st.markdown("**Compliance by Country**")
            
            st.dataframe(
                opec_data,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'country': 'Country',
                    'quota_mbpd': st.column_config.NumberColumn('Quota (mb/d)', format='%.2f'),
                    'actual_mbpd': st.column_config.NumberColumn('Actual (mb/d)', format='%.2f'),
                    'compliance_pct': st.column_config.ProgressColumn('Compliance', min_value=0, max_value=110, format='%.0f%%'),
                }
            )
        
        with col2:
            st.markdown("**Overall Compliance**")
            
            opec_analysis = fundamental_analyzer.analyze_opec_compliance(opec_data)
            
            st.metric(
                "Overall Compliance",
                f"{opec_analysis['overall_compliance_pct']:.1f}%"
            )
            
            st.metric(
                "Total OPEC+ Production",
                f"{opec_analysis['total_actual_mbpd']:.2f} mb/d"
            )
            
            st.metric(
                "vs Quota",
                f"{opec_analysis['deviation_mbpd']:+.2f} mb/d"
            )
            
            # Market impact
            st.divider()
            st.markdown("**Market Impact Assessment**")
            
            if "Bullish" in opec_analysis['market_impact']:
                st.success(opec_analysis['market_impact'])
            elif "Bearish" in opec_analysis['market_impact']:
                st.error(opec_analysis['market_impact'])
            else:
                st.info(opec_analysis['market_impact'])
            
            if opec_analysis['over_producers']:
                st.warning(f"Over-producers: {', '.join(opec_analysis['over_producers'])}")

# =============================================================================
# AUTO-REFRESH MECHANISM
# =============================================================================

# Auto-refresh using JavaScript injection (works without additional packages)
if st.session_state.auto_refresh:
    # Calculate time until next refresh
    time_since_last = (datetime.now() - st.session_state.last_refresh).total_seconds()
    time_until_refresh = max(0, REFRESH_INTERVAL_SECONDS - time_since_last)
    
    if time_until_refresh <= 0:
        st.session_state.last_refresh = datetime.now()
        time.sleep(0.1)  # Small delay to prevent rapid refreshes
        st.rerun()
    else:
        # Inject JavaScript for countdown and auto-refresh
        st.markdown(
            f"""
            <script>
                // Auto-refresh countdown
                setTimeout(function() {{
                    window.parent.postMessage({{isStreamlitMessage: true, type: "streamlit:rerun"}}, "*");
                }}, {int(time_until_refresh * 1000)});
            </script>
            """,
            unsafe_allow_html=True
        )

# Footer with refresh info
st.markdown("---")
st.markdown(
    f"""<div style="text-align: center; color: #64748B; font-size: 12px;">
    Data refreshes every 15 seconds when auto-refresh is enabled | 
    Charts show up to 180 days of historical data
    </div>""",
    unsafe_allow_html=True
)
