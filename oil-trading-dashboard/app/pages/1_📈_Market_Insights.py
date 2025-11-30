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
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.data import DataLoader
from core.analytics import CurveAnalyzer, SpreadAnalyzer, FundamentalAnalyzer

st.set_page_config(page_title="Market Insights | Oil Trading", page_icon="📈", layout="wide")

# Initialize components
@st.cache_resource
def get_data_loader():
    return DataLoader(
        config_dir=str(project_root / "config"),
        data_dir=str(project_root / "data"),
        use_mock=True
    )

data_loader = get_data_loader()
curve_analyzer = CurveAnalyzer()
spread_analyzer = SpreadAnalyzer()
fundamental_analyzer = FundamentalAnalyzer()

st.title("📈 Market Insights")
st.caption("Real-time oil market analysis and intelligence")

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
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Price chart with Plotly
        st.markdown("**WTI Crude Oil - Daily Chart**")
        
        # Get historical data
        hist_data = data_loader.get_historical("CL1 Comdty", 
                                                start_date=datetime.now() - timedelta(days=180))
        
        if not hist_data.empty:
            fig = go.Figure()
            
            # Candlestick chart
            fig.add_trace(go.Candlestick(
                x=hist_data.index,
                open=hist_data['PX_OPEN'],
                high=hist_data['PX_HIGH'],
                low=hist_data['PX_LOW'],
                close=hist_data['PX_LAST'],
                name='WTI'
            ))
            
            # Add moving averages
            hist_data['SMA20'] = hist_data['PX_LAST'].rolling(20).mean()
            hist_data['SMA50'] = hist_data['PX_LAST'].rolling(50).mean()
            
            fig.add_trace(go.Scatter(
                x=hist_data.index, y=hist_data['SMA20'],
                name='SMA 20', line=dict(color='orange', width=1)
            ))
            
            fig.add_trace(go.Scatter(
                x=hist_data.index, y=hist_data['SMA50'],
                name='SMA 50', line=dict(color='purple', width=1)
            ))
            
            fig.update_layout(
                template='plotly_dark',
                height=500,
                xaxis_rangeslider_visible=False,
                margin=dict(l=0, r=0, t=0, b=0),
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Volume chart
        st.markdown("**Volume**")
        if not hist_data.empty and 'PX_VOLUME' in hist_data.columns:
            vol_fig = go.Figure()
            vol_fig.add_trace(go.Bar(
                x=hist_data.index,
                y=hist_data['PX_VOLUME'],
                marker_color='#00A3E0',
                name='Volume'
            ))
            vol_fig.update_layout(
                template='plotly_dark',
                height=150,
                margin=dict(l=0, r=0, t=0, b=0),
            )
            st.plotly_chart(vol_fig, use_container_width=True)
    
    with col2:
        st.markdown("**Key Levels**")
        
        if not hist_data.empty:
            current_price = hist_data['PX_LAST'].iloc[-1]
            high_52w = hist_data['PX_HIGH'].max()
            low_52w = hist_data['PX_LOW'].min()
            
            st.metric("Current Price", f"${current_price:.2f}")
            st.metric("52-Week High", f"${high_52w:.2f}")
            st.metric("52-Week Low", f"${low_52w:.2f}")
            
            # Price position
            position = (current_price - low_52w) / (high_52w - low_52w) * 100
            st.progress(int(position) / 100, text=f"Range Position: {position:.0f}%")
        
        st.divider()
        
        st.markdown("**Technical Indicators**")
        
        # Mock technical indicators
        indicators = {
            'RSI (14)': 58.5,
            'MACD': 'Bullish',
            'BB Position': '65%',
            'ADX': 28.5,
        }
        
        for ind, val in indicators.items():
            st.text(f"{ind}: {val}")

with tab2:
    # Term Structure Tab
    st.subheader("Futures Curve Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # WTI vs Brent curves
        wti_curve = data_loader.get_futures_curve("wti", 12)
        brent_curve = data_loader.get_futures_curve("brent", 12)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=wti_curve['month'], y=wti_curve['price'],
            name='WTI', mode='lines+markers',
            line=dict(color='#00D26A', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=brent_curve['month'], y=brent_curve['price'],
            name='Brent', mode='lines+markers',
            line=dict(color='#00A3E0', width=2)
        ))
        
        fig.update_layout(
            template='plotly_dark',
            height=400,
            xaxis_title='Month',
            yaxis_title='Price ($/bbl)',
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
            margin=dict(l=0, r=0, t=30, b=0),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
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
        
        # Get prices
        wti = data_loader.get_price("CL1 Comdty")
        rbob = data_loader.get_price("XB1 Comdty")
        ho = data_loader.get_price("HO1 Comdty")
        
        crack_321 = spread_analyzer.calculate_crack_spread(wti, rbob, ho, "3-2-1")
        
        st.metric(
            "Current",
            f"${crack_321['crack_spread']:.2f}/bbl",
            delta="+$1.20 (+4.4%)"
        )
        
        # Historical chart (mock)
        dates = pd.date_range(end=datetime.now(), periods=60, freq='D')
        crack_values = 28 + np.cumsum(np.random.normal(0, 0.5, 60))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=crack_values,
            fill='tozeroy',
            fillcolor='rgba(0, 163, 224, 0.3)',
            line=dict(color='#00A3E0', width=2),
            name='3-2-1 Crack'
        ))
        
        # Add mean line
        fig.add_hline(y=crack_values.mean(), line_dash='dash', 
                     line_color='orange', annotation_text='30-Day Avg')
        
        fig.update_layout(
            template='plotly_dark',
            height=300,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Metrics
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric("30-Day Avg", f"${crack_values.mean():.2f}")
        with m_col2:
            st.metric("Percentile", "72nd")
        with m_col3:
            st.metric("Margin %", f"{crack_321['margin_pct']:.1f}%")
    
    with col2:
        st.markdown("**Regional Differentials**")
        
        # Mock regional differentials
        differentials = pd.DataFrame({
            'Spread': ['Brent-WTI', 'WCS-WTI', 'Dubai-Brent', 'Mars-WTI', 'LLS-WTI'],
            'Value': [-4.35, -14.20, -1.80, 2.10, 1.50],
            'Change': ['▼ narrowing', '▲ widening', '─ stable', '▲ widening', '─ stable'],
        })
        
        for _, row in differentials.iterrows():
            col_a, col_b, col_c = st.columns([2, 1, 1])
            with col_a:
                st.text(row['Spread'])
            with col_b:
                st.text(f"${row['Value']:.2f}")
            with col_c:
                st.text(row['Change'])
        
        st.divider()
        
        st.markdown("**Component Prices**")
        st.text(f"WTI: ${wti:.2f}/bbl")
        st.text(f"RBOB: ${rbob:.4f}/gal (${rbob*42:.2f}/bbl)")
        st.text(f"Heating Oil: ${ho:.4f}/gal (${ho*42:.2f}/bbl)")

with tab4:
    # Inventory Tab
    st.subheader("Inventory Analytics")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("**EIA Weekly Crude Inventory**")
        
        eia_data = data_loader.get_eia_inventory()
        
        fig = go.Figure()
        
        # Inventory level
        fig.add_trace(go.Scatter(
            x=eia_data.index,
            y=eia_data['inventory_mmb'],
            name='Inventory',
            line=dict(color='#00A3E0', width=2)
        ))
        
        # 5-year range (mock)
        mean = eia_data['inventory_mmb'].mean()
        fig.add_hline(y=mean, line_dash='dash', line_color='orange',
                     annotation_text='5-Year Avg')
        
        fig.update_layout(
            template='plotly_dark',
            height=350,
            yaxis_title='Inventory (MMbbl)',
            margin=dict(l=0, r=0, t=10, b=0),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Weekly change
        st.markdown("**Weekly Change**")
        
        change_fig = go.Figure()
        change_fig.add_trace(go.Bar(
            x=eia_data.index,
            y=eia_data['change_mmb'],
            marker_color=['#00D26A' if x < 0 else '#FF4B4B' for x in eia_data['change_mmb']],
            name='Change'
        ))
        
        change_fig.update_layout(
            template='plotly_dark',
            height=200,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        
        st.plotly_chart(change_fig, use_container_width=True)
    
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
        
        st.divider()
        
        st.markdown("**Cushing Stocks**")
        cushing = fundamental_analyzer.analyze_cushing_stocks(23.1)
        st.metric("Level", f"{cushing['current_level']:.1f} MMbbl")
        st.progress(int(cushing['utilization_pct']) / 100, 
                   text=f"Utilization: {cushing['utilization_pct']:.0f}%")

with tab5:
    # OPEC Monitor Tab
    st.subheader("OPEC+ Production Monitor")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("**Production vs Quota by Country**")
        
        opec_data = data_loader.get_opec_production()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Quota',
            x=opec_data['country'],
            y=opec_data['quota_mbpd'],
            marker_color='#00A3E0',
        ))
        
        fig.add_trace(go.Bar(
            name='Actual',
            x=opec_data['country'],
            y=opec_data['actual_mbpd'],
            marker_color='#00D26A',
        ))
        
        fig.update_layout(
            template='plotly_dark',
            height=400,
            barmode='group',
            yaxis_title='Production (mb/d)',
            margin=dict(l=0, r=0, t=10, b=0),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
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
        
        st.divider()
        
        st.markdown("**Next Meeting**")
        st.text("Date: Dec 4, 2024")
        st.text("Expected: No change")
        
        if opec_analysis['over_producers']:
            st.warning(f"Over-producers: {', '.join(opec_analysis['over_producers'])}")
