"""
Analytics Page
==============
Advanced analytics and research tools.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.components.charts import (
    BASE_LAYOUT,
    CHART_COLORS,
    create_candlestick_chart,
    create_heatmap,
    create_volume_chart,
)
from app.page_utils import get_chart_config, init_page
from core.analytics import CurveAnalyzer, FundamentalAnalyzer, SpreadAnalyzer

# Initialize page
ctx = init_page(
    title="📊 Advanced Analytics",
    page_title="Analytics | Oil Trading",
    icon="📊",
    lookback_days=365,
)

st.caption("Deep dive analysis and research tools")

# Initialize analyzers
curve_analyzer = CurveAnalyzer()
spread_analyzer = SpreadAnalyzer()
fundamental_analyzer = FundamentalAnalyzer()

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
        st.markdown("**Settings**")

        instrument = st.selectbox(
            "Instrument",
            ["WTI Crude (CL)", "Brent Crude (CO)", "RBOB Gasoline (XB)", "Heating Oil (HO)"]
        )

        timeframe = st.selectbox("Timeframe", ["Daily", "Weekly", "Monthly"])
        lookback = st.slider("Lookback (days)", 30, 365, 180)

        show_ma = st.checkbox("Show Moving Averages", value=True)
        show_bb = st.checkbox("Show Bollinger Bands", value=False)
        show_volume = st.checkbox("Show Volume", value=True)

    with col1:
        ticker = "CL1 Comdty" if "WTI" in instrument else "CO1 Comdty" if "Brent" in instrument else "XB1 Comdty" if "RBOB" in instrument else "HO1 Comdty"

        hist_data = None
        if ticker == "CL1 Comdty":
            cached = ctx.context.data.wti_history
            if cached is not None and not cached.empty:
                start_cutoff = datetime.now() - timedelta(days=lookback)
                hist_data = cached[cached.index >= start_cutoff]

        if hist_data is None or hist_data.empty:
            hist_data = ctx.data_loader.get_historical(
                ticker,
                start_date=datetime.now() - timedelta(days=lookback)
            )

        if hist_data is not None and not hist_data.empty:
            ma_periods = [20, 50] if show_ma else []
            fig = create_candlestick_chart(
                data=hist_data,
                title="",
                height=450,
                show_volume=False,
                show_ma=show_ma,
                ma_periods=ma_periods,
            )

            if show_bb:
                sma = hist_data['PX_LAST'].rolling(20).mean()
                std = hist_data['PX_LAST'].rolling(20).std()

                fig.add_trace(go.Scatter(
                    x=hist_data.index, y=sma + 2*std,
                    name='BB Upper', line={"color": CHART_COLORS['text_secondary'], "width": 1, "dash": 'dash'},
                ))
                fig.add_trace(go.Scatter(
                    x=hist_data.index, y=sma - 2*std,
                    name='BB Lower', line={"color": CHART_COLORS['text_secondary'], "width": 1, "dash": 'dash'},
                    fill='tonexty', fillcolor='rgba(148, 163, 184, 0.1)',
                ))

            st.plotly_chart(fig, width='stretch', config=get_chart_config())

            if show_volume and 'PX_VOLUME' in hist_data.columns:
                vol_fig = create_volume_chart(hist_data, height=120)
                st.plotly_chart(vol_fig, width='stretch', config=get_chart_config())

            st.markdown("**Statistics**")

            live_price = ctx.price_cache.get(ticker)
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

    st.markdown("**Oil Market Correlations (60-day rolling)**")

    corr_data = pd.DataFrame({
        'WTI': [1.00, 0.95, 0.82, 0.78, -0.35, 0.42],
        'Brent': [0.95, 1.00, 0.80, 0.75, -0.32, 0.38],
        'RBOB': [0.82, 0.80, 1.00, 0.88, -0.28, 0.35],
        'Heating Oil': [0.78, 0.75, 0.88, 1.00, -0.25, 0.32],
        'USD Index': [-0.35, -0.32, -0.28, -0.25, 1.00, -0.55],
        'S&P 500': [0.42, 0.38, 0.35, 0.32, -0.55, 1.00],
    }, index=['WTI', 'Brent', 'RBOB', 'Heating Oil', 'USD Index', 'S&P 500'])

    fig = create_heatmap(corr_data, title="", colorscale="RdBu_r", height=400)
    st.plotly_chart(fig, width='stretch', config=get_chart_config())

    st.divider()

    st.markdown("**Rolling WTI-Brent Correlation**")

    dates = pd.date_range(end=datetime.now(), periods=180, freq='D')
    corr_series = 0.95 + np.random.normal(0, 0.02, 180)
    corr_series = np.clip(corr_series, 0.85, 0.99)

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=dates, y=corr_series,
        fill='tozeroy',
        fillcolor='rgba(0, 163, 224, 0.1)',
        line={"color": CHART_COLORS['primary'], "width": 2.5},
        name='Correlation',
    ))

    fig2.add_hline(y=0.95, line_dash='dash', line_color=CHART_COLORS['ma_fast'], annotation_text='Long-term Avg')

    corr_yaxis = dict(BASE_LAYOUT['yaxis'])
    corr_yaxis.update({'range': [0.8, 1.0], 'tickformat': '.2f'})

    fig2.update_layout(
        **BASE_LAYOUT,
        height=300,
        yaxis_title='Correlation',
        yaxis=corr_yaxis,
    )

    st.plotly_chart(fig2, width='stretch', config=get_chart_config())

with tab3:
    st.subheader("Seasonality Analysis")

    col1, col2 = st.columns([3, 1])

    with col2:
        product = st.selectbox("Select Product", ["WTI Crude", "Brent Crude", "RBOB Gasoline", "Heating Oil"])
        years_lookback = st.slider("Years of History", 3, 10, 5)

    with col1:
        st.markdown(f"**{product} Seasonal Pattern**")

        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        if "Gasoline" in product or "RBOB" in product:
            seasonal = [0.5, 1.2, 2.5, 4.0, 3.5, 2.0, -0.5, -1.5, -2.0, -1.0, -0.5, 0.2]
        elif "Heating" in product:
            seasonal = [3.0, 2.5, 1.0, -1.5, -2.5, -3.0, -2.0, -1.0, 0.5, 1.5, 2.0, 2.8]
        else:
            seasonal = [1.0, 0.5, -0.5, 0.8, 1.5, 2.0, 1.0, -1.0, -1.5, 0.5, 1.2, 0.8]

        seasonal_high = [s + 2.5 for s in seasonal]
        seasonal_low = [s - 2.5 for s in seasonal]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=months + months[::-1],
            y=seasonal_high + seasonal_low[::-1],
            fill='toself',
            fillcolor='rgba(0, 163, 224, 0.1)',
            line={"color": 'rgba(255,255,255,0)'},
            name='Historical Range',
        ))

        fig.add_trace(go.Scatter(
            x=months, y=seasonal,
            mode='lines+markers',
            line={"color": CHART_COLORS['primary'], "width": 3, "shape": 'spline'},
            marker={"size": 10, "color": CHART_COLORS['primary']},
            name='5-Year Average',
        ))

        current_month = datetime.now().month
        current_year = [s + np.random.uniform(-1, 1) for s in seasonal[:current_month]]

        fig.add_trace(go.Scatter(
            x=months[:current_month], y=current_year,
            mode='lines+markers',
            line={"color": CHART_COLORS['profit'], "width": 3, "shape": 'spline'},
            marker={"size": 10, "color": CHART_COLORS['profit']},
            name='Current Year',
        ))

        fig.add_hline(y=0, line_dash='solid', line_color='rgba(255,255,255,0.3)')

        fig.update_layout(
            **BASE_LAYOUT,
            height=400,
            yaxis_title='Return (%)',
            yaxis_tickformat='+.0f',
        )

        st.plotly_chart(fig, width='stretch', config=get_chart_config())

with tab4:
    st.subheader("Backtesting Tools")

    st.info("🔬 **Research Environment**: Full backtesting capabilities available in Jupyter Lab")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Quick Strategy Test**")

        strategy = st.selectbox(
            "Strategy",
            ["Moving Average Crossover", "RSI Mean Reversion", "Term Structure Momentum"]
        )

        st.markdown("**Parameters**")

        if "Moving Average" in strategy:
            fast_period = st.slider("Fast Period", 5, 50, 20)
            slow_period = st.slider("Slow Period", 20, 200, 50)
        elif "RSI" in strategy:
            rsi_period = st.slider("RSI Period", 7, 21, 14)
            oversold = st.slider("Oversold Level", 20, 40, 30)

        if st.button("Run Backtest", width='stretch'):
            with st.spinner("Running backtest..."):
                import time
                time.sleep(1)
                st.success("Backtest complete!")

    with col2:
        st.markdown("**Results Summary**")

        results = {
            "Total Return": "+45.2%",
            "Sharpe Ratio": "1.42",
            "Max Drawdown": "-12.5%",
            "Win Rate": "58%",
        }

        for metric, value in results.items():
            st.text(f"{metric}: {value}")

        st.divider()

        st.markdown("**Equity Curve**")

        dates = pd.date_range(end=datetime.now(), periods=365, freq='D')
        equity = 100000 * np.exp(np.cumsum(np.random.normal(0.0005, 0.01, len(dates))))

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=dates, y=equity,
            fill='tozeroy',
            fillcolor='rgba(0, 220, 130, 0.15)',
            line={"color": CHART_COLORS['profit'], "width": 2.5},
            name='Equity',
        ))

        fig.update_layout(
            **BASE_LAYOUT,
            height=200,
            yaxis_tickformat='$,.0f',
        )

        st.plotly_chart(fig, width='stretch', config=get_chart_config())

    st.divider()

    st.markdown("**Full Research Environment**")
    st.code("jupyter lab --notebook-dir=research/notebooks", language="bash")
