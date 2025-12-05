"""
Research Page
=============
Advanced analytics, AI-powered research, and alternative data analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

from app.page_utils import init_page, get_chart_config
from app.components.charts import CHART_COLORS, BASE_LAYOUT

# Initialize page
ctx = init_page(
    title="🔍 Research & Analytics",
    page_title="Research | Oil Trading",
    icon="🔍",
    lookback_days=365,
)

# Try to load research modules
try:
    from core.research import (
        NewsAnalyzer,
        SentimentAnalyzer,
        CorrelationAnalyzer,
        RegimeDetector,
        FactorModel,
        AlternativeDataProvider,
    )
    from core.research.llm import AnalysisConfig, SentimentConfig
    from core.research.correlations import CorrelationMethod
    from core.research.regimes import RegimeConfig
    from core.research.factors import FactorConfig
    RESEARCH_AVAILABLE = True
except ImportError as e:
    RESEARCH_AVAILABLE = False
    RESEARCH_ERROR = str(e)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📰 News & Sentiment",
    "📊 Correlations",
    "🔄 Regimes",
    "📈 Factor Models",
    "🛰️ Alternative Data"
])

# ============================================================================
# TAB 1: News & Sentiment
# ============================================================================
with tab1:
    st.subheader("LLM-Powered News Analysis")
    
    if not RESEARCH_AVAILABLE:
        st.warning(f"Research modules not fully loaded: {RESEARCH_ERROR}")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**Analyze News Article**")
            
            sample_articles = [
                "OPEC+ agrees to extend production cuts through Q2 2024, citing weak global demand outlook. Saudi Arabia to maintain voluntary 1 million bpd cut.",
                "US crude inventories rise by 5.2 million barrels, exceeding expectations of 2.1 million build. Gasoline stocks also increased.",
                "Tensions in the Middle East escalate as conflict spreads. Oil traders monitor shipping routes through Strait of Hormuz.",
                "China's crude imports hit record high as refineries boost processing ahead of holiday travel season.",
                "US shale producers report increasing well productivity, with Permian basin output reaching all-time highs.",
            ]
            
            use_sample = st.checkbox("Use sample article", value=True)
            
            if use_sample:
                article_text = st.selectbox(
                    "Select sample article",
                    sample_articles,
                    key="sample_article"
                )
            else:
                article_text = st.text_area(
                    "Enter article text",
                    height=150,
                    placeholder="Paste news article text here..."
                )
            
            if st.button("🔍 Analyze Article", type="primary") and article_text:
                with st.spinner("Analyzing..."):
                    try:
                        analyzer = NewsAnalyzer(AnalysisConfig(use_llm=False))
                        summary = analyzer.analyze_article(article_text)
                        
                        st.success("Analysis complete!")
                        
                        # Display results
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            impact_colors = {
                                "HIGH": "🔴",
                                "MEDIUM": "🟡", 
                                "LOW": "🟢",
                                "NONE": "⚪"
                            }
                            st.metric(
                                "Market Impact",
                                f"{impact_colors.get(summary.impact_level, '⚪')} {summary.impact_level}"
                            )
                        
                        with col_b:
                            direction_icons = {
                                "BULLISH": "📈",
                                "BEARISH": "📉",
                                "NEUTRAL": "➡️"
                            }
                            st.metric(
                                "Direction",
                                f"{direction_icons.get(summary.impact_direction, '➡️')} {summary.impact_direction}"
                            )
                        
                        with col_c:
                            st.metric("Confidence", f"{summary.confidence:.0%}")
                        
                        st.markdown("**Summary**")
                        st.write(summary.summary)
                        
                        if summary.key_points:
                            st.markdown("**Key Points**")
                            for point in summary.key_points:
                                st.markdown(f"• {point}")
                        
                        if summary.commodities:
                            st.markdown("**Commodities Mentioned**")
                            st.write(", ".join(summary.commodities))
                    
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")
        
        with col2:
            st.markdown("**Sentiment Analyzer**")
            
            try:
                sentiment_analyzer = SentimentAnalyzer()
                
                # Quick sentiment check
                quick_text = st.text_input(
                    "Quick sentiment check",
                    placeholder="Enter text..."
                )
                
                if quick_text:
                    result = sentiment_analyzer.analyze_text(quick_text)
                    
                    sentiment_colors = {
                        "VERY_BULLISH": "#00cc00",
                        "BULLISH": "#66ff66",
                        "NEUTRAL": "#999999",
                        "BEARISH": "#ff6666",
                        "VERY_BEARISH": "#cc0000"
                    }
                    
                    st.markdown(
                        f"<div style='padding:10px; background-color:{sentiment_colors.get(result.sentiment.name, '#999')};'>"
                        f"<strong>{result.sentiment.name}</strong> (Score: {result.score:.2f})</div>",
                        unsafe_allow_html=True
                    )
                
                st.divider()
                
                st.markdown("**Aggregate Sentiment**")
                
                # Generate mock aggregate sentiment
                texts = [
                    "Oil prices surge on supply concerns",
                    "OPEC to cut production",
                    "US inventories build weighs on prices",
                    "China demand outlook improves",
                    "Dollar strength pressures commodities"
                ]
                
                aggregate = sentiment_analyzer.get_aggregate_sentiment(texts)
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=aggregate["avg_score"],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [-1, 1]},
                        'bar': {'color': "#1f77b4"},
                        'steps': [
                            {'range': [-1, -0.3], 'color': "#ffcccc"},
                            {'range': [-0.3, 0.3], 'color': "#eeeeee"},
                            {'range': [0.3, 1], 'color': "#ccffcc"},
                        ],
                    },
                    title={'text': "Sentiment Score"}
                ))
                
                fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True, config=get_chart_config())
            
            except Exception as e:
                st.error(f"Sentiment analyzer error: {e}")

# ============================================================================
# TAB 2: Correlations
# ============================================================================
with tab2:
    st.subheader("Cross-Asset Correlation Analysis")
    
    if not RESEARCH_AVAILABLE:
        st.warning("Research modules not available")
    else:
        try:
            correlation_analyzer = CorrelationAnalyzer()
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("**Correlation Settings**")
                
                method = st.selectbox(
                    "Method",
                    ["Pearson", "Spearman", "Kendall"],
                    key="corr_method"
                )
                
                window = st.slider("Rolling Window (days)", 21, 252, 63)
                
                assets = st.multiselect(
                    "Assets",
                    ["Brent", "WTI", "Natural_Gas", "Dollar", "SP500", "Gold", "VIX"],
                    default=["Brent", "WTI", "Dollar", "SP500"]
                )
            
            with col2:
                st.markdown("**Current Correlation Matrix**")
                
                if len(assets) >= 2:
                    corr_matrix = correlation_analyzer.calculate_correlation_matrix(assets)
                    
                    if not corr_matrix.empty:
                        fig = px.imshow(
                            corr_matrix,
                            text_auto='.2f',
                            color_continuous_scale='RdBu_r',
                            zmin=-1, zmax=1,
                        )
                        
                        fig.update_layout(
                            height=350,
                            margin=dict(l=20, r=20, t=20, b=20),
                        )
                        
                        st.plotly_chart(fig, use_container_width=True, config=get_chart_config())
            
            st.divider()
            
            # Rolling correlation chart
            st.markdown("**Rolling Correlation Analysis**")
            
            col_a, col_b = st.columns([1, 4])
            
            with col_a:
                asset1 = st.selectbox("Asset 1", ["Brent", "WTI"], key="corr_asset1")
                asset2 = st.selectbox("Asset 2", ["Dollar", "SP500", "Gold", "VIX"], key="corr_asset2")
            
            with col_b:
                rolling_data = correlation_analyzer.calculate_rolling_correlation(
                    asset1, asset2,
                    window=window,
                    days=365
                )
                
                if rolling_data:
                    dates = [r.date for r in rolling_data]
                    correlations = [r.correlation for r in rolling_data]
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=dates,
                        y=correlations,
                        mode='lines',
                        name='Correlation',
                        line=dict(color=CHART_COLORS['primary'], width=2),
                        fill='tozeroy',
                        fillcolor='rgba(31, 119, 180, 0.2)'
                    ))
                    
                    fig.add_hline(y=0, line_dash="dash", line_color="gray")
                    
                    fig.update_layout(
                        **BASE_LAYOUT,
                        height=300,
                        yaxis=dict(title="Correlation", range=[-1, 1]),
                        xaxis_title="Date",
                    )
                    
                    st.plotly_chart(fig, use_container_width=True, config=get_chart_config())
            
            # Correlation regime detection
            st.markdown("**Correlation Regime Detection**")
            
            regime = correlation_analyzer.detect_regime(asset1, asset2)
            
            regime_cols = st.columns(4)
            
            with regime_cols[0]:
                st.metric("Current Regime", regime.get("regime", "Unknown"))
            
            with regime_cols[1]:
                st.metric("Correlation", f"{regime.get('current_correlation', 0):.2f}")
            
            with regime_cols[2]:
                st.metric("Regime Strength", f"{regime.get('regime_strength', 50):.0f}%")
            
            with regime_cols[3]:
                st.metric("Days in Regime", regime.get("days_in_regime", 0))
        
        except Exception as e:
            st.error(f"Correlation analysis error: {e}")

# ============================================================================
# TAB 3: Regimes
# ============================================================================
with tab3:
    st.subheader("Market Regime Detection")
    
    if not RESEARCH_AVAILABLE:
        st.warning("Research modules not available")
    else:
        try:
            regime_detector = RegimeDetector()
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("**Current Market Regime**")
                
                current = regime_detector.get_current_regime()
                
                regime_colors = {
                    "TRENDING_UP": "#00cc00",
                    "TRENDING_DOWN": "#cc0000",
                    "RANGING": "#0066cc",
                    "HIGH_VOLATILITY": "#ff9900",
                    "LOW_VOLATILITY": "#66ccff",
                    "CRISIS": "#990000"
                }
                
                regime_name = current.get("regime", "UNKNOWN")
                
                st.markdown(
                    f"<div style='padding:20px; background-color:{regime_colors.get(regime_name, '#999')};'>"
                    f"<h2 style='color:white; margin:0;'>{regime_name}</h2></div>",
                    unsafe_allow_html=True
                )
                
                st.markdown(f"**Confidence:** {current.get('confidence', 50):.0f}%")
                st.markdown(f"**Rationale:** {current.get('rationale', 'N/A')}")
                
                # Regime metrics
                st.divider()
                
                metrics = current.get("metrics", {})
                
                metric_cols = st.columns(4)
                
                with metric_cols[0]:
                    st.metric("Trend Strength", f"{metrics.get('trend_strength', 0):.1%}")
                
                with metric_cols[1]:
                    st.metric("Volatility", f"{metrics.get('volatility', 0)*100:.1f}%")
                
                with metric_cols[2]:
                    st.metric("Mean Reversion", f"{metrics.get('mean_reversion', 0):.2f}")
                
                with metric_cols[3]:
                    st.metric("Trend Direction", metrics.get("trend_direction", "N/A").title())
            
            with col2:
                st.markdown("**Volatility Regime**")
                
                vol_regime = regime_detector.get_volatility_regime()
                
                vol_regime_name = vol_regime.get("regime", "NORMAL")
                
                vol_colors = {
                    "EXTREMELY_LOW": "#00ffff",
                    "LOW": "#66ccff",
                    "NORMAL": "#999999",
                    "HIGH": "#ffcc00",
                    "EXTREMELY_HIGH": "#ff0000"
                }
                
                st.markdown(
                    f"<div style='padding:15px; background-color:{vol_colors.get(vol_regime_name, '#999')};'>"
                    f"<strong>{vol_regime_name}</strong></div>",
                    unsafe_allow_html=True
                )
                
                st.metric("Current Vol", f"{vol_regime.get('current_volatility', 0)*100:.1f}%")
                st.metric("Vol Percentile", f"{vol_regime.get('percentile', 50):.0f}%")
                st.metric("Trend", vol_regime.get("trend", "stable").title())
            
            st.divider()
            
            # Regime history
            st.markdown("**Regime History**")
            
            history = regime_detector.get_regime_history(days=180)
            
            if history:
                # Create regime timeline
                fig = go.Figure()
                
                dates = [h.get("date") for h in history]
                regimes = [h.get("regime") for h in history]
                
                # Assign numeric values to regimes for plotting
                regime_values = {
                    "TRENDING_UP": 2,
                    "TRENDING_DOWN": -2,
                    "RANGING": 0,
                    "HIGH_VOLATILITY": 1,
                    "LOW_VOLATILITY": -1,
                    "CRISIS": -3,
                }
                
                values = [regime_values.get(r, 0) for r in regimes]
                colors = [regime_colors.get(r, "#999") for r in regimes]
                
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=values,
                    mode='markers+lines',
                    marker=dict(
                        color=colors,
                        size=8,
                    ),
                    line=dict(color='gray', width=1),
                    text=regimes,
                    hovertemplate='%{text}<br>%{x}<extra></extra>'
                ))
                
                fig.update_layout(
                    **BASE_LAYOUT,
                    height=300,
                    yaxis=dict(
                        title="Regime",
                        ticktext=["Crisis", "Trend Down", "Low Vol", "Ranging", "High Vol", "Trend Up"],
                        tickvals=[-3, -2, -1, 0, 1, 2],
                    ),
                    xaxis_title="Date",
                )
                
                st.plotly_chart(fig, use_container_width=True, config=get_chart_config())
            
            # Regime transitions
            st.markdown("**Recent Transitions**")
            
            transitions = regime_detector.get_regime_transitions(limit=5)
            
            if transitions:
                trans_df = pd.DataFrame([
                    {
                        "Date": t.date.strftime("%Y-%m-%d %H:%M") if hasattr(t.date, 'strftime') else str(t.date),
                        "From": t.from_regime.name if hasattr(t.from_regime, 'name') else str(t.from_regime),
                        "To": t.to_regime.name if hasattr(t.to_regime, 'name') else str(t.to_regime),
                        "Confidence": f"{t.confidence:.0%}",
                    }
                    for t in transitions
                ])
                
                st.dataframe(trans_df, use_container_width=True, hide_index=True)
        
        except Exception as e:
            st.error(f"Regime detection error: {e}")

# ============================================================================
# TAB 4: Factor Models
# ============================================================================
with tab4:
    st.subheader("Factor Decomposition & Attribution")
    
    if not RESEARCH_AVAILABLE:
        st.warning("Research modules not available")
    else:
        try:
            factor_model = FactorModel()
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("**Factor Settings**")
                
                asset = st.selectbox(
                    "Asset to Analyze",
                    ["Brent", "WTI", "RBOB", "Heating_Oil"],
                    key="factor_asset"
                )
                
                window = st.slider(
                    "Analysis Window (days)",
                    30, 252, 63,
                    key="factor_window"
                )
                
                if st.button("📊 Run Factor Analysis", type="primary"):
                    with st.spinner("Running factor decomposition..."):
                        decomposition = factor_model.decompose_returns(asset, days=window)
                        st.session_state['factor_results'] = decomposition
            
            with col2:
                if 'factor_results' in st.session_state:
                    decomposition = st.session_state['factor_results']
                    
                    st.markdown("**Factor Exposures**")
                    
                    exposures = decomposition.factor_exposures
                    
                    fig = go.Figure(go.Bar(
                        x=list(exposures.values()),
                        y=list(exposures.keys()),
                        orientation='h',
                        marker_color=[
                            CHART_COLORS['up'] if v > 0 else CHART_COLORS['down']
                            for v in exposures.values()
                        ]
                    ))
                    
                    fig.update_layout(
                        **BASE_LAYOUT,
                        height=350,
                        xaxis_title="Exposure (Beta)",
                        margin=dict(l=120, r=20, t=20, b=40),
                    )
                    
                    st.plotly_chart(fig, use_container_width=True, config=get_chart_config())
                else:
                    st.info("Click 'Run Factor Analysis' to see results")
            
            st.divider()
            
            if 'factor_results' in st.session_state:
                decomposition = st.session_state['factor_results']
                
                # Return Attribution
                st.markdown("**Return Attribution**")
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown("**Contribution to Returns**")
                    
                    contributions = decomposition.factor_contributions
                    
                    if contributions:
                        fig = go.Figure(data=[go.Pie(
                            labels=list(contributions.keys()),
                            values=[abs(v) for v in contributions.values()],
                            hole=0.4,
                            marker_colors=px.colors.qualitative.Set3,
                        )])
                        
                        fig.update_layout(
                            height=300,
                            margin=dict(l=20, r=20, t=20, b=20),
                        )
                        
                        st.plotly_chart(fig, use_container_width=True, config=get_chart_config())
                
                with col_b:
                    st.markdown("**Model Statistics**")
                    
                    stats_cols = st.columns(2)
                    
                    with stats_cols[0]:
                        st.metric("R-Squared", f"{decomposition.r_squared:.1%}")
                        st.metric("Total Return", f"{decomposition.total_return:.2%}")
                    
                    with stats_cols[1]:
                        st.metric("Alpha", f"{decomposition.alpha:.2%}")
                        st.metric("Residual Vol", f"{decomposition.residual_volatility:.1%}")
                    
                    st.divider()
                    
                    st.markdown("**Top Factors**")
                    
                    sorted_factors = sorted(
                        contributions.items(),
                        key=lambda x: abs(x[1]),
                        reverse=True
                    )[:5]
                    
                    for factor, contrib in sorted_factors:
                        st.text(f"{factor}: {contrib:+.2%}")
        
        except Exception as e:
            st.error(f"Factor analysis error: {e}")

# ============================================================================
# TAB 5: Alternative Data
# ============================================================================
with tab5:
    st.subheader("Alternative Data Sources")
    
    if not RESEARCH_AVAILABLE:
        st.warning("Research modules not available")
    else:
        try:
            alt_data = AlternativeDataProvider()
            
            subtab1, subtab2, subtab3, subtab4 = st.tabs([
                "🛰️ Satellite",
                "🚢 Shipping",
                "📊 Positioning",
                "📡 Aggregate Signal"
            ])
            
            with subtab1:
                st.markdown("**Oil Storage Monitoring (Satellite Imagery)**")
                
                satellite = alt_data.satellite
                observations = satellite.get_latest_observations()
                
                locations = observations.get("locations", {})
                
                if locations:
                    # Storage overview
                    storage_data = []
                    
                    for loc, data in locations.items():
                        storage_data.append({
                            "Location": data.get("name", loc),
                            "Utilization": f"{data.get('utilization_pct', 0):.1f}%",
                            "Volume (MB)": data.get("estimated_volume_mb", 0),
                            "Capacity (MB)": data.get("capacity_mb", 0),
                            "Week Change": f"{data.get('change_week_pct', 0):+.1f}%",
                            "Confidence": f"{data.get('confidence', 0):.0%}",
                        })
                    
                    storage_df = pd.DataFrame(storage_data)
                    st.dataframe(storage_df, use_container_width=True, hide_index=True)
                    
                    # Storage utilization chart
                    fig = go.Figure()
                    
                    for loc, data in locations.items():
                        util = data.get("utilization_pct", 0)
                        fig.add_trace(go.Bar(
                            x=[data.get("name", loc)],
                            y=[util],
                            name=loc,
                            marker_color="#ff7f0e" if util > 70 else ("#1f77b4" if util < 50 else "#2ca02c")
                        ))
                    
                    fig.update_layout(
                        **BASE_LAYOUT,
                        height=300,
                        yaxis=dict(title="Utilization %", range=[0, 100]),
                        showlegend=False,
                    )
                    
                    st.plotly_chart(fig, use_container_width=True, config=get_chart_config())
                    
                    # Storage signal
                    signal = satellite.calculate_storage_signal()
                    
                    st.markdown("**Storage Signal**")
                    
                    signal_cols = st.columns(4)
                    
                    with signal_cols[0]:
                        signal_color = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}
                        st.metric("Signal", f"{signal_color.get(signal['signal'], '⚪')} {signal['signal'].upper()}")
                    
                    with signal_cols[1]:
                        st.metric("Confidence", f"{signal['confidence']}%")
                    
                    with signal_cols[2]:
                        st.metric("Global Utilization", f"{signal['global_utilization']:.1f}%")
                    
                    with signal_cols[3]:
                        st.info(signal['rationale'])
            
            with subtab2:
                st.markdown("**Tanker Tracking & Trade Flows**")
                
                shipping = alt_data.shipping
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Fleet Overview**")
                    
                    fleet = shipping.get_fleet_overview()
                    fleet_by_type = fleet.get("fleet_by_type", {})
                    
                    if fleet_by_type:
                        fleet_data = []
                        
                        for vessel_type, counts in fleet_by_type.items():
                            fleet_data.append({
                                "Type": vessel_type,
                                "At Sea": counts.get("at_sea", 0),
                                "Loading": counts.get("loading", 0),
                                "Discharging": counts.get("discharging", 0),
                                "Anchored": counts.get("anchored", 0),
                            })
                        
                        st.dataframe(pd.DataFrame(fleet_data), use_container_width=True, hide_index=True)
                
                with col2:
                    st.markdown("**Freight Rates**")
                    
                    freight = shipping.get_freight_rates()
                    spot_rates = freight.get("spot_rates", {})
                    
                    if spot_rates:
                        for vessel_type, rate_data in spot_rates.items():
                            change = rate_data.get("change_week_pct", 0)
                            delta_color = "normal" if change >= 0 else "inverse"
                            
                            st.metric(
                                f"{vessel_type} Rate",
                                f"${rate_data.get('rate_usd_day', 0):,.0f}/day",
                                f"{change:+.1f}%",
                                delta_color=delta_color
                            )
                
                st.divider()
                
                st.markdown("**Trade Flows**")
                
                flows = shipping.get_trade_flows()
                flow_data = flows.get("flows", {})
                
                if flow_data:
                    fig = go.Figure()
                    
                    for route_id, data in flow_data.items():
                        fig.add_trace(go.Bar(
                            x=[data.get("name", route_id)],
                            y=[data.get("current_mb_d", 0)],
                            name=route_id,
                            text=[f"{data.get('change_pct', 0):+.1f}%"],
                            textposition='outside',
                        ))
                    
                    fig.update_layout(
                        **BASE_LAYOUT,
                        height=300,
                        yaxis_title="Million barrels/day",
                        showlegend=False,
                    )
                    
                    st.plotly_chart(fig, use_container_width=True, config=get_chart_config())
                
                # Shipping signal
                signal = shipping.calculate_shipping_signal()
                
                st.markdown("**Shipping Signal**")
                
                signal_cols = st.columns(4)
                
                with signal_cols[0]:
                    signal_color = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}
                    st.metric("Signal", f"{signal_color.get(signal['signal'], '⚪')} {signal['signal'].upper()}")
                
                with signal_cols[1]:
                    st.metric("Confidence", f"{signal['confidence']}%")
                
                with signal_cols[2]:
                    st.metric("Flow Ratio", f"{signal['flow_ratio']:.2f}x")
                
                with signal_cols[3]:
                    st.info(signal['rationale'])
            
            with subtab3:
                st.markdown("**Commitment of Traders (COT) Data**")
                
                positioning = alt_data.positioning
                
                # Managed Money positions
                st.markdown("**Managed Money Positions**")
                
                managed = positioning.get_managed_money_positions()
                positions = managed.get("positions", {})
                
                if positions:
                    pos_data = []
                    
                    for commodity, data in positions.items():
                        pos_data.append({
                            "Commodity": commodity,
                            "Net Contracts": f"{data.get('net_contracts', 0):,}",
                            "Percentile": f"{data.get('percentile', 50):.0f}%",
                            "Stance": data.get("stance", "neutral").replace("_", " ").title(),
                            "Week Change": f"{data.get('week_change', 0):+,}",
                        })
                    
                    st.dataframe(pd.DataFrame(pos_data), use_container_width=True, hide_index=True)
                
                # Percentile visualization
                if positions:
                    fig = go.Figure()
                    
                    commodities = list(positions.keys())
                    percentiles = [positions[c].get("percentile", 50) for c in commodities]
                    
                    fig.add_trace(go.Bar(
                        x=commodities,
                        y=percentiles,
                        marker_color=[
                            "#cc0000" if p > 80 else ("#00cc00" if p < 20 else "#1f77b4")
                            for p in percentiles
                        ],
                        text=[f"{p:.0f}%" for p in percentiles],
                        textposition='outside',
                    ))
                    
                    fig.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Extreme Long")
                    fig.add_hline(y=20, line_dash="dash", line_color="green", annotation_text="Extreme Short")
                    
                    fig.update_layout(
                        **BASE_LAYOUT,
                        height=300,
                        yaxis=dict(title="Percentile", range=[0, 110]),
                    )
                    
                    st.plotly_chart(fig, use_container_width=True, config=get_chart_config())
                
                # Positioning signal
                signal = positioning.calculate_positioning_signal()
                
                st.markdown("**Positioning Signal (Contrarian)**")
                
                signal_cols = st.columns(4)
                
                with signal_cols[0]:
                    signal_color = {
                        "bullish": "🟢", "bearish": "🔴", "neutral": "🟡",
                        "cautious_bullish": "🟢", "cautious_bearish": "🔴"
                    }
                    st.metric("Signal", f"{signal_color.get(signal['signal'], '⚪')} {signal['signal'].upper()}")
                
                with signal_cols[1]:
                    st.metric("Confidence", f"{signal['confidence']}%")
                
                with signal_cols[2]:
                    st.metric("Avg Percentile", f"{signal['avg_percentile']:.0f}%")
                
                with signal_cols[3]:
                    st.info(signal['rationale'])
            
            with subtab4:
                st.markdown("**Aggregate Alternative Data Signal**")
                
                aggregate = alt_data.get_aggregate_signal()
                
                st.markdown("---")
                
                # Overall signal display
                signal_color = {"bullish": "#00cc00", "bearish": "#cc0000", "neutral": "#999999"}
                overall = aggregate.get("overall_signal", "neutral")
                
                st.markdown(
                    f"<div style='padding:30px; text-align:center; background-color:{signal_color.get(overall, '#999')};'>"
                    f"<h1 style='color:white; margin:0;'>{overall.upper()}</h1>"
                    f"<p style='color:white;'>Aggregate Confidence: {aggregate.get('overall_confidence', 50):.0f}%</p></div>",
                    unsafe_allow_html=True
                )
                
                st.markdown("---")
                
                # Component signals
                st.markdown("**Component Signals**")
                
                col1, col2, col3 = st.columns(3)
                
                signals = aggregate.get("signals", {})
                
                with col1:
                    sat_signal = signals.get("satellite", {})
                    st.markdown("**🛰️ Satellite (Storage)**")
                    st.metric("Signal", sat_signal.get("signal", "N/A").upper())
                    st.metric("Confidence", f"{sat_signal.get('confidence', 0)}%")
                
                with col2:
                    ship_signal = signals.get("shipping", {})
                    st.markdown("**🚢 Shipping**")
                    st.metric("Signal", ship_signal.get("signal", "N/A").upper())
                    st.metric("Confidence", f"{ship_signal.get('confidence', 0)}%")
                
                with col3:
                    pos_signal = signals.get("positioning", {})
                    st.markdown("**📊 Positioning**")
                    st.metric("Signal", pos_signal.get("signal", "N/A").upper())
                    st.metric("Confidence", f"{pos_signal.get('confidence', 0)}%")
                
                st.markdown("---")
                
                # Rationales
                st.markdown("**Signal Rationales**")
                
                for source, data in signals.items():
                    with st.expander(f"📝 {source.title()} Rationale"):
                        st.write(data.get("rationale", "No rationale available"))
        
        except Exception as e:
            st.error(f"Alternative data error: {e}")
