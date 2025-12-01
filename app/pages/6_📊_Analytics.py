"""
Analytics Page
==============
Advanced analytics and research tools.
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

from dotenv import load_dotenv
load_dotenv()

from app import shared_state
from core.analytics import CurveAnalyzer, SpreadAnalyzer, FundamentalAnalyzer

st.set_page_config(page_title="Analytics | Oil Trading", page_icon="📊", layout="wide")

# Apply shared theme
from app.components.theme import apply_theme, COLORS, PLOTLY_LAYOUT
apply_theme(st)

# Initialize
context = shared_state.get_dashboard_context(lookback_days=365)
data_loader = context.data_loader
curve_analyzer = CurveAnalyzer()
spread_analyzer = SpreadAnalyzer()
fundamental_analyzer = FundamentalAnalyzer()

st.title("📊 Advanced Analytics")

# Show live prices at top
oil_prices = context.data.oil_prices
price_cache = context.price_cache

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

st.caption("Deep dive analysis and research tools")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Price Analysis",
    "🔄 Correlation",
    "📅 Seasonality",
    "🔬 Backtesting"
])

with tab1:
    st.subheader("Price Analysis Tools")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        # Settings
        st.markdown("**Settings**")
        
        instrument = st.selectbox(
            "Instrument",
            ["WTI Crude (CL)", "Brent Crude (CO)", "RBOB Gasoline (XB)", "Heating Oil (HO)"]
        )
        
        timeframe = st.selectbox(
            "Timeframe",
            ["Daily", "Weekly", "Monthly"]
        )
        
        lookback = st.slider("Lookback (days)", 30, 365, 180)
        
        show_ma = st.checkbox("Show Moving Averages", value=True)
        show_bb = st.checkbox("Show Bollinger Bands", value=False)
        show_volume = st.checkbox("Show Volume", value=True)
    
    with col1:
        # Price chart
        ticker = "CL1 Comdty" if "WTI" in instrument else "CO1 Comdty" if "Brent" in instrument else "XB1 Comdty" if "RBOB" in instrument else "HO1 Comdty"
        
        hist_data = None
        if ticker == "CL1 Comdty":
            cached = context.data.wti_history
            if cached is not None and not cached.empty:
                start_cutoff = datetime.now() - timedelta(days=lookback)
                hist_data = cached[cached.index >= start_cutoff]
        if hist_data is None or hist_data.empty:
            hist_data = data_loader.get_historical(
                ticker,
                start_date=datetime.now() - timedelta(days=lookback)
            )
        
        if not hist_data.empty:
            fig = go.Figure()
            
            # Candlestick
            fig.add_trace(go.Candlestick(
                x=hist_data.index,
                open=hist_data['PX_OPEN'],
                high=hist_data['PX_HIGH'],
                low=hist_data['PX_LOW'],
                close=hist_data['PX_LAST'],
                name='Price'
            ))
            
            if show_ma:
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
            
            if show_bb:
                sma = hist_data['PX_LAST'].rolling(20).mean()
                std = hist_data['PX_LAST'].rolling(20).std()
                
                fig.add_trace(go.Scatter(
                    x=hist_data.index, y=sma + 2*std,
                    name='BB Upper', line=dict(color='gray', width=1, dash='dash')
                ))
                fig.add_trace(go.Scatter(
                    x=hist_data.index, y=sma - 2*std,
                    name='BB Lower', line=dict(color='gray', width=1, dash='dash'),
                    fill='tonexty', fillcolor='rgba(128, 128, 128, 0.1)'
                ))
            
            fig.update_layout(
                template='plotly_dark',
                height=450,
                xaxis_rangeslider_visible=False,
                margin=dict(l=0, r=0, t=10, b=0),
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            if show_volume and 'PX_VOLUME' in hist_data.columns:
                vol_fig = go.Figure()
                vol_fig.add_trace(go.Bar(
                    x=hist_data.index,
                    y=hist_data['PX_VOLUME'],
                    marker_color='#00A3E0',
                    name='Volume'
                ))
                vol_fig.update_layout(
                    template='plotly_dark',
                    height=120,
                    margin=dict(l=0, r=0, t=0, b=0),
                )
                st.plotly_chart(vol_fig, use_container_width=True)
            
            # Statistics
            st.markdown("**Statistics**")
            
            # Use LIVE price from price cache
            live_price = price_cache.get(ticker)
            current_price = live_price if live_price else hist_data['PX_LAST'].iloc[-1]
            
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            
            with stat_col1:
                st.metric("Current (Live)", f"${current_price:.2f}")
                st.metric("Period High", f"${hist_data['PX_HIGH'].max():.2f}")
            
            with stat_col2:
                ret = (hist_data['PX_LAST'].iloc[-1] / hist_data['PX_LAST'].iloc[0] - 1) * 100
                st.metric("Period Return", f"{ret:+.2f}%")
                st.metric("Period Low", f"${hist_data['PX_LOW'].min():.2f}")
            
            with stat_col3:
                vol = hist_data['PX_LAST'].pct_change().std() * np.sqrt(252) * 100
                st.metric("Ann. Volatility", f"{vol:.1f}%")
                st.metric("Avg Volume", f"{hist_data['PX_VOLUME'].mean():,.0f}")
            
            with stat_col4:
                sharpe = (hist_data['PX_LAST'].pct_change().mean() / hist_data['PX_LAST'].pct_change().std()) * np.sqrt(252)
                st.metric("Sharpe Ratio", f"{sharpe:.2f}")

with tab2:
    st.subheader("Correlation Analysis")
    
    # Correlation matrix
    st.markdown("**Oil Market Correlations (60-day rolling)**")
    
    # Mock correlation matrix
    corr_data = pd.DataFrame({
        'WTI': [1.00, 0.95, 0.82, 0.78, -0.35, 0.42],
        'Brent': [0.95, 1.00, 0.80, 0.75, -0.32, 0.38],
        'RBOB': [0.82, 0.80, 1.00, 0.88, -0.28, 0.35],
        'Heating Oil': [0.78, 0.75, 0.88, 1.00, -0.25, 0.32],
        'USD Index': [-0.35, -0.32, -0.28, -0.25, 1.00, -0.55],
        'S&P 500': [0.42, 0.38, 0.35, 0.32, -0.55, 1.00],
    }, index=['WTI', 'Brent', 'RBOB', 'Heating Oil', 'USD Index', 'S&P 500'])
    
    fig = px.imshow(
        corr_data,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
    )
    
    fig.update_layout(
        template='plotly_dark',
        height=400,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Rolling correlation
    st.markdown("**Rolling WTI-Brent Correlation**")
    
    dates = pd.date_range(end=datetime.now(), periods=180, freq='D')
    corr_series = 0.95 + np.random.normal(0, 0.02, 180)
    corr_series = np.clip(corr_series, 0.85, 0.99)
    
    fig2 = go.Figure()
    
    fig2.add_trace(go.Scatter(
        x=dates, y=corr_series,
        fill='tozeroy',
        fillcolor='rgba(0, 163, 224, 0.3)',
        line=dict(color='#00A3E0', width=2),
        name='Correlation'
    ))
    
    fig2.add_hline(y=0.95, line_dash='dash', line_color='orange',
                  annotation_text='Long-term Avg')
    
    fig2.update_layout(
        template='plotly_dark',
        height=300,
        yaxis_title='Correlation',
        yaxis=dict(range=[0.8, 1.0]),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader("Seasonality Analysis")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        product = st.selectbox(
            "Select Product",
            ["WTI Crude", "Brent Crude", "RBOB Gasoline", "Heating Oil", "3-2-1 Crack"]
        )
        
        years_lookback = st.slider("Years of History", 3, 10, 5)
    
    with col1:
        st.markdown(f"**{product} Seasonal Pattern**")
        
        # Generate seasonal pattern
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Different patterns for different products
        if "Gasoline" in product or "RBOB" in product:
            seasonal = [0.5, 1.2, 2.5, 4.0, 3.5, 2.0, -0.5, -1.5, -2.0, -1.0, -0.5, 0.2]
        elif "Heating" in product:
            seasonal = [3.0, 2.5, 1.0, -1.5, -2.5, -3.0, -2.0, -1.0, 0.5, 1.5, 2.0, 2.8]
        else:
            seasonal = [1.0, 0.5, -0.5, 0.8, 1.5, 2.0, 1.0, -1.0, -1.5, 0.5, 1.2, 0.8]
        
        # Add noise for range
        seasonal_high = [s + 2.5 for s in seasonal]
        seasonal_low = [s - 2.5 for s in seasonal]
        
        fig = go.Figure()
        
        # Range fill
        fig.add_trace(go.Scatter(
            x=months + months[::-1],
            y=seasonal_high + seasonal_low[::-1],
            fill='toself',
            fillcolor='rgba(0, 163, 224, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Historical Range',
        ))
        
        # Average line
        fig.add_trace(go.Scatter(
            x=months, y=seasonal,
            mode='lines+markers',
            line=dict(color='#00A3E0', width=3),
            marker=dict(size=10),
            name='5-Year Average',
        ))
        
        # Current year (mock)
        current_month = datetime.now().month
        current_year = [s + np.random.uniform(-1, 1) for s in seasonal[:current_month]]
        
        fig.add_trace(go.Scatter(
            x=months[:current_month], y=current_year,
            mode='lines+markers',
            line=dict(color='#00D26A', width=3),
            marker=dict(size=10),
            name='Current Year',
        ))
        
        fig.add_hline(y=0, line_dash='solid', line_color='white', line_width=1)
        
        fig.update_layout(
            template='plotly_dark',
            height=400,
            yaxis_title='Return (%)',
            margin=dict(l=0, r=0, t=10, b=0),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Monthly stats table
        st.markdown("**Monthly Statistics**")
        
        stats_df = pd.DataFrame({
            'Month': months,
            'Avg Return': [f"{s:+.1f}%" for s in seasonal],
            'Win Rate': [f"{55 + np.random.randint(-10, 15)}%" for _ in months],
            'Best': [f"+{abs(s) + 3:.1f}%" for s in seasonal],
            'Worst': [f"-{abs(s) + 2:.1f}%" for s in seasonal],
        })
        
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Backtesting Tools")
    
    st.info("🔬 **Research Environment**: Full backtesting capabilities available in Jupyter Lab")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Quick Strategy Test**")
        
        # Strategy selection
        strategy = st.selectbox(
            "Strategy",
            [
                "Moving Average Crossover",
                "RSI Mean Reversion",
                "Term Structure Momentum",
                "Calendar Spread Mean Reversion",
                "Inventory Surprise Trading"
            ]
        )
        
        # Parameters
        st.markdown("**Parameters**")
        
        if "Moving Average" in strategy:
            fast_period = st.slider("Fast Period", 5, 50, 20)
            slow_period = st.slider("Slow Period", 20, 200, 50)
        elif "RSI" in strategy:
            rsi_period = st.slider("RSI Period", 7, 21, 14)
            oversold = st.slider("Oversold Level", 20, 40, 30)
            overbought = st.slider("Overbought Level", 60, 80, 70)
        
        # Backtest period
        start_date = st.date_input("Backtest Start", value=datetime.now().date() - timedelta(days=365))
        end_date = st.date_input("Backtest End", value=datetime.now().date())
        
        if st.button("Run Backtest", use_container_width=True):
            with st.spinner("Running backtest..."):
                # Simulate backtest (mock results)
                import time
                time.sleep(1)
                
                st.success("Backtest complete!")
    
    with col2:
        st.markdown("**Results Summary**")
        
        # Mock results
        results = {
            "Total Return": "+45.2%",
            "Annual Return": "+18.5%",
            "Sharpe Ratio": "1.42",
            "Sortino Ratio": "2.15",
            "Max Drawdown": "-12.5%",
            "Win Rate": "58%",
            "Profit Factor": "1.85",
            "Total Trades": "156",
            "Avg Trade": "+$2,450",
        }
        
        for metric, value in results.items():
            st.text(f"{metric}: {value}")
        
        st.divider()
        
        # Equity curve
        st.markdown("**Equity Curve**")
        
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        equity = 100000 * np.exp(np.cumsum(np.random.normal(0.0005, 0.01, len(dates))))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates, y=equity,
            fill='tozeroy',
            fillcolor='rgba(0, 210, 106, 0.3)',
            line=dict(color='#00D26A', width=2),
            name='Equity'
        ))
        
        fig.update_layout(
            template='plotly_dark',
            height=200,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Launch Jupyter
    st.markdown("**Full Research Environment**")
    
    st.code("jupyter lab --notebook-dir=research/notebooks", language="bash")
    
    st.markdown("""
    Available research notebooks:
    - `01_data_exploration.ipynb` - Data analysis and visualization
    - `02_backtest_momentum.ipynb` - Momentum strategy backtesting
    - `03_ml_experiments.ipynb` - Machine learning model development
    - `04_term_structure_strategies.ipynb` - Curve trading strategies
    """)
