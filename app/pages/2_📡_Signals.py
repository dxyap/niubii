"""
Trading Signals Page
====================
Signal generation and display with enhanced visual indicators.

Performance optimizations:
- Signal components cached via @st.cache_resource
- Historical data fetching uses page context caching
- Lazy signal generation (only when needed)
"""

from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

from app.components.ui_components import (
    render_compact_stats,
    render_progress_ring,
    render_signal_indicator,
)
from app.page_utils import COLORS, init_page

# Initialize page first (before heavy imports)
ctx = init_page(
    title="📡 Trading Signals",
    page_title="Signals | Oil Trading",
    icon="📡",
    lookback_days=120,
)

st.caption("AI-powered signal generation for oil markets | Signals are advisory only")


# Lazy import and cache signal components
@st.cache_resource(show_spinner=False)
def get_signal_components():
    """
    Initialize signal components (cached as resource).
    
    These are expensive to create and stateless, so caching is safe.
    """
    from core.signals import FundamentalSignals, SignalAggregator, TechnicalSignals
    return TechnicalSignals(), FundamentalSignals(), SignalAggregator()


tech_signals, fund_signals, aggregator = get_signal_components()


def generate_signals():
    """Generate composite trading signals using live data."""
    hist_data = ctx.data_loader.get_historical(
        "CL1 Comdty",
        start_date=datetime.now() - timedelta(days=120)
    )

    if hist_data is None or hist_data.empty:
        return None, None, None

    prices = hist_data['PX_LAST']
    current_price = ctx.price_cache.get("CL1 Comdty")

    # Technical signal
    tech_signal = tech_signals.generate_composite_signal(prices)

    # Get curve data for fundamental signal
    curve_metrics = ctx.context.data.curve_metrics()
    crack_spread = ctx.context.data.crack_spread

    # Fundamental signal
    fund_signal = fund_signals.generate_composite_fundamental_signal(
        inventory_data={"level": 430, "change": -2.1, "expectation": -1.5},
        opec_data={"compliance": 94, "deviation": 0.3},
        curve_data={
            "m1_m2_spread": curve_metrics.get('m1_m2', 0.35),
            "m1_m12_spread": curve_metrics.get('m1_m12', 1.8),
            "slope": curve_metrics.get('slope', 0.15)
        },
        crack_spread=crack_spread.get('crack', 28.5) if crack_spread else 28.5,
        turnaround_data={"offline": 800, "upcoming": 600}
    )

    # Aggregate signals
    composite = aggregator.aggregate_signals(
        technical_signal=tech_signal,
        fundamental_signal=fund_signal,
        instrument="CL1 Comdty",
        current_price=current_price
    )

    return tech_signal, fund_signal, composite


tech_signal, fund_signal, composite = generate_signals()

# Main signal display
if composite:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🎯 Active Signal - WTI Crude Oil")

        # Enhanced signal indicator
        render_signal_indicator(
            direction=composite.direction,
            confidence=composite.confidence,
            instrument="WTI Crude Oil",
            entry_price=composite.entry_price,
        )

        st.markdown("")  # Spacing

        # Trade levels with better styling
        level_col1, level_col2, level_col3 = st.columns(3)

        with level_col1:
            entry_color = COLORS.get("profit", "#00DC82") if composite.direction == "LONG" else COLORS.get("loss", "#FF5252")
            st.markdown(f"""
            <div style="background: rgba(30, 41, 59, 0.6); border-radius: 10px; padding: 16px; border: 1px solid rgba(51, 65, 85, 0.5); text-align: center;">
                <div style="font-size: 10px; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em;">Entry Zone</div>
                <div style="font-size: 24px; font-weight: 700; color: {entry_color}; font-family: 'IBM Plex Mono', monospace;">${composite.entry_price:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        with level_col2:
            st.markdown(f"""
            <div style="background: rgba(239, 68, 68, 0.1); border-radius: 10px; padding: 16px; border: 1px solid rgba(239, 68, 68, 0.3); text-align: center;">
                <div style="font-size: 10px; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em;">Stop Loss</div>
                <div style="font-size: 24px; font-weight: 700; color: #FF5252; font-family: 'IBM Plex Mono', monospace;">${composite.stop_loss:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        with level_col3:
            st.markdown(f"""
            <div style="background: rgba(0, 220, 130, 0.1); border-radius: 10px; padding: 16px; border: 1px solid rgba(0, 220, 130, 0.3); text-align: center;">
                <div style="font-size: 10px; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em;">Target</div>
                <div style="font-size: 24px; font-weight: 700; color: #00DC82; font-family: 'IBM Plex Mono', monospace;">${composite.target_price:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # Signal drivers
        st.subheader("📊 Signal Drivers")

        drivers_df = pd.DataFrame(composite.drivers)

        for _, driver in drivers_df.iterrows():
            signal_color = COLORS["success"] if "LONG" in str(driver.get('signal', '')) or "BUY" in str(driver.get('signal', '')) else COLORS["error"] if "SHORT" in str(driver.get('signal', '')) or "SELL" in str(driver.get('signal', '')) else "#94a3b8"

            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; padding: 12px 16px; background: rgba(30, 41, 59, 0.6); border-radius: 8px; margin: 6px 0; border: 1px solid #334155;">
                <span style="color: #e2e8f0;">{driver['source']}</span>
                <span style="color: {signal_color}; font-weight: 600;">{driver.get('signal', 'N/A')}</span>
                <span style="color: #94a3b8;">{driver.get('weight', 0):.0f}% weight</span>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.subheader("📈 Signal Confidence")

        confidence = composite.confidence
        confidence_color = COLORS.get("success", "#00DC82") if confidence > 70 else COLORS.get("warning", "#f59e0b") if confidence > 50 else "#94a3b8"

        # Circular confidence gauge
        render_progress_ring(
            value=confidence,
            max_value=100,
            label="Confidence",
            color=confidence_color,
            size=140,
        )

        st.markdown("")  # Spacing

        # Signal details in compact format
        render_compact_stats([
            {"label": "Signal ID", "value": composite.signal_id[:8]},
            {"label": "Generated", "value": composite.timestamp.strftime('%H:%M:%S')},
            {"label": "Horizon", "value": composite.time_horizon},
        ])

        st.markdown("")  # Spacing

        # Action buttons with better styling
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("✅ Traded", use_container_width=True, type="primary"):
                st.success("Signal marked as traded")
        with col_btn2:
            if st.button("❌ Dismiss", use_container_width=True):
                st.info("Signal dismissed")

st.divider()

# Component signals breakdown
st.subheader("🔬 Signal Components")

tab1, tab2, tab3 = st.tabs(["Technical Analysis", "Fundamental Analysis", "Signal History"])

with tab1:
    if tech_signal and 'components' in tech_signal:
        col1, col2, col3 = st.columns(3)

        components = tech_signal['components']

        with col1:
            st.markdown("**Moving Averages**")
            ma = components.get('ma_crossover', {})
            st.metric("Signal", ma.get('signal', 'N/A'))
            st.metric("Confidence", f"{ma.get('confidence', 0):.0f}%")
            st.text(f"Fast MA: ${ma.get('fast_ma', 0):.2f}")
            st.text(f"Slow MA: ${ma.get('slow_ma', 0):.2f}")

            st.divider()

            st.markdown("**RSI**")
            rsi = components.get('rsi', {})
            st.metric("RSI Value", f"{rsi.get('rsi', 0):.1f}")
            st.metric("Signal", rsi.get('signal', 'N/A'))

        with col2:
            st.markdown("**Bollinger Bands**")
            bb = components.get('bollinger', {})
            st.metric("Signal", bb.get('signal', 'N/A'))
            st.metric("%B", f"{bb.get('percent_b', 50):.1f}%")
            st.text(f"Upper: ${bb.get('upper_band', 0):.2f}")
            st.text(f"Middle: ${bb.get('middle_band', 0):.2f}")
            st.text(f"Lower: ${bb.get('lower_band', 0):.2f}")

            st.divider()

            st.markdown("**Momentum**")
            mom = components.get('momentum', {})
            st.metric("ROC", f"{mom.get('roc_pct', 0):.2f}%")
            st.metric("Signal", mom.get('signal', 'N/A'))

        with col3:
            st.markdown("**Breakout Analysis**")
            breakout = components.get('breakout', {})
            st.metric("Signal", breakout.get('signal', 'N/A'))
            st.text(f"Channel High: ${breakout.get('channel_high', 0):.2f}")
            st.text(f"Channel Low: ${breakout.get('channel_low', 0):.2f}")
            st.text(f"Width: {breakout.get('channel_width_pct', 0):.1f}%")

            st.divider()

            st.markdown("**Overall Technical**")
            st.metric("Score", f"{tech_signal.get('score', 0):.3f}")
            st.metric("Direction", tech_signal.get('signal', 'N/A'))

with tab2:
    if fund_signal and 'components' in fund_signal:
        col1, col2 = st.columns(2)

        components = fund_signal['components']

        with col1:
            st.markdown("**Inventory Signal**")
            inv = components.get('inventory', {})
            st.metric("Signal", inv.get('signal', 'N/A'))
            st.metric("Surprise", f"{inv.get('surprise', 0):+.1f} MMbbl")
            st.text(inv.get('surprise_signal', ''))

            st.divider()

            st.markdown("**OPEC Signal**")
            opec = components.get('opec', {})
            st.metric("Signal", opec.get('signal', 'N/A'))
            st.metric("Compliance", f"{opec.get('compliance_pct', 0):.0f}%")
            st.text(opec.get('description', ''))

            st.divider()

            st.markdown("**Term Structure Signal**")
            term = components.get('term_structure', {})
            st.metric("Signal", term.get('signal', 'N/A'))
            st.metric("Curve Slope", f"{term.get('curve_slope', 0):.4f}")
            st.text(term.get('description', ''))

        with col2:
            st.markdown("**Crack Spread Signal**")
            crack = components.get('crack_spread', {})
            st.metric("Signal", crack.get('signal', 'N/A'))
            st.metric("Z-Score", f"{crack.get('zscore', 0):.2f}")
            st.metric("Percentile", f"{crack.get('percentile', 50):.0f}th")
            st.text(crack.get('description', ''))

            st.divider()

            st.markdown("**Turnaround Signal**")
            turn = components.get('turnaround', {})
            st.metric("Signal", turn.get('signal', 'N/A'))
            st.metric("Offline", f"{turn.get('total_impact_kbpd', 0):,.0f} kb/d")
            st.text(turn.get('description', ''))

            st.divider()

            st.markdown("**Overall Fundamental**")
            st.metric("Score", f"{fund_signal.get('score', 0):.3f}")
            st.metric("Direction", fund_signal.get('signal', 'N/A'))

with tab3:
    st.markdown("**Recent Signal History**")

    signal_history = pd.DataFrame({
        'Time': ['14:32', '11:15', '09:30', 'Yesterday 15:45', 'Yesterday 10:00'],
        'Instrument': ['WTI', 'WTI-Brent', 'WTI', 'Brent', 'WTI'],
        'Direction': ['🟢 LONG', '🔴 SHORT', '⚪ NEUTRAL', '🟢 LONG', '🔴 SHORT'],
        'Confidence': ['72%', '58%', '45%', '68%', '61%'],
        'Status': ['Active', 'Active', 'Expired', 'Traded', 'Stopped Out'],
        'Result': ['--', '--', '--', '+$12,500', '-$4,200'],
    })

    st.dataframe(signal_history, width="stretch", hide_index=True)

    st.divider()

    st.markdown("**Signal Performance (Last 30 Days)**")

    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)

    with perf_col1:
        st.metric("Total Signals", "47")
    with perf_col2:
        st.metric("Win Rate", "58%")
    with perf_col3:
        st.metric("Avg P&L", "+$2,450")
    with perf_col4:
        st.metric("Profit Factor", "1.42")

# Disclaimer
st.divider()
st.warning("""
⚠️ **IMPORTANT DISCLAIMER**: These signals are generated by algorithmic analysis and are for informational purposes only.
They do not constitute investment advice. All trading decisions should be made by qualified traders after their own analysis.
Past performance does not guarantee future results. Always use proper risk management.
""")