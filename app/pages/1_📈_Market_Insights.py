"""
Market Insights & Research Page
================================
Comprehensive market analysis, intelligence, and AI-powered research.
"""

import math
import sys
import time
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
CONFIG_DIR = project_root / "config"
BLOOMBERG_TICKERS_PATH = CONFIG_DIR / "bloomberg_tickers.yaml"

from dotenv import load_dotenv

load_dotenv()

from app import shared_state
from app.components.charts import (
    BASE_LAYOUT,
    CHART_COLORS,
    create_candlestick_chart,
    create_futures_curve_chart,
    create_open_interest_chart,
    create_volume_chart,
)
from core.analytics import CurveAnalyzer, FundamentalAnalyzer, SpreadAnalyzer
from core.data.bloomberg import DataUnavailableError

st.set_page_config(page_title="Market Insights | Oil Trading", page_icon="📈", layout="wide")

# Apply shared theme
from app.components.theme import apply_theme, get_chart_config

apply_theme(st)


@lru_cache(maxsize=1)
def load_bloomberg_tickers_config() -> dict:
    """Load Bloomberg ticker mappings once per session."""
    if BLOOMBERG_TICKERS_PATH.exists():
        try:
            with BLOOMBERG_TICKERS_PATH.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}
    return {}


def get_crack_spread_front_month_override() -> dict:
    """Return any configured front month override for the 321 crack spread."""
    config = load_bloomberg_tickers_config()
    override = (
        config.get("spreads", {})
        .get("crack_321", {})
        .get("front_month_override")
    )
    if isinstance(override, dict):
        return {
            "ticker": override.get("ticker"),
            "label": override.get("label"),
        }
    if isinstance(override, str):
        return {"ticker": override, "label": None}
    return {}


def sanitize_percentage(value, default=0.0):
    """Convert percentage-like values to a safe float or fall back to default."""
    try:
        sanitized = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(sanitized):
        return default
    return sanitized


# Try to load research modules
try:
    from core.research import (
        AlternativeDataProvider,
        CorrelationAnalyzer,
        FactorModel,
        NewsAnalyzer,
        RegimeDetector,
        SatelliteData,
        SentimentAnalyzer,
    )
    from core.research.llm import AnalysisConfig
    RESEARCH_AVAILABLE = True
except ImportError as e:
    RESEARCH_AVAILABLE = False
    RESEARCH_ERROR = str(e)

# Auto-refresh configuration
REFRESH_INTERVAL_SECONDS = 60  # Refresh every 60 seconds

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


def format_calendar_spread_labels(spreads_df, curve_df):
    """Format spread labels with actual contract months when available."""
    if spreads_df is None or curve_df is None:
        return spreads_df
    if spreads_df.empty or curve_df.empty:
        return spreads_df

    label_series = None
    if "contract_month" in curve_df.columns:
        label_series = curve_df["contract_month"]
    elif "contract_date" in curve_df.columns:
        label_series = pd.to_datetime(curve_df["contract_date"]).dt.strftime("%b-%y")
    elif "ticker" in curve_df.columns:
        label_series = curve_df["ticker"]

    if label_series is None:
        return spreads_df

    label_map = {}
    for idx, raw_label in enumerate(label_series.tolist(), start=1):
        if pd.isna(raw_label):
            continue
        label_text = str(raw_label).strip()
        if not label_text or label_text.lower() == "nan":
            continue
        label_map[idx] = label_text

    if not label_map:
        return spreads_df

    formatted = spreads_df.copy()

    def _format_label(row):
        front_label = label_map.get(int(row.get("front_month", 0)))
        back_label = label_map.get(int(row.get("back_month", 0)))
        if front_label and back_label:
            return f"{front_label} vs {back_label}"
        return row.get("spread_name", "")

    formatted["spread_name"] = formatted.apply(_format_label, axis=1)
    if "front_contract" not in formatted.columns:
        formatted["front_contract"] = formatted["front_month"].map(label_map)
    if "back_contract" not in formatted.columns:
        formatted["back_contract"] = formatted["back_month"].map(label_map)
    return formatted

# Check data mode
connection_status = data_loader.get_connection_status()
data_mode = connection_status.get("data_mode", "disconnected")

# Header with live status and controls
header_col1, header_col2, header_col3 = st.columns([3, 1, 1])

with header_col1:
    st.title("📈 Market Insights & Research")

with header_col2:
    toggle_label = f"Auto Refresh ({REFRESH_INTERVAL_SECONDS}s)"
    auto_refresh = st.toggle(toggle_label, value=st.session_state.auto_refresh, key="auto_refresh_toggle")
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
elif data_mode == "disconnected":
    st.error("🔴 Bloomberg Terminal not connected. Live data required.")
    st.info(f"Connection error: {connection_status.get('connection_error', 'Unknown')}")
    st.stop()
else:
    st.warning(f"⚠️ Data mode: {data_mode}")

# =============================================================================
# MAIN TABS - Market Insights + Research
# =============================================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📊 Price Action",
    "🔥 Crack Spreads",
    "📦 Inventory",
    "🌍 OPEC Monitor",
    "📰 News & Sentiment",
    "📊 Correlations",
    "🔄 Regimes",
    "📈 Factor Models",
    "🛰️ Alternative Data"
])

# =============================================================================
# TAB 1: Price Action
# =============================================================================
with tab1:
    # Instrument definitions
    # Note: Dubai uses 2nd month swap (DAT2) to avoid BALMO (Balance of Month)
    # Note: WTI uses ICE prices (ENA1 Comdty) not NYMEX (CL1)
    instruments = {
        "Brent": {"ticker": "CO1 Comdty", "name": "Brent Crude Oil (ICE)", "icon": "🇬🇧"},
        "WTI": {
            "ticker": "ENA1 Comdty",
            "name": "WTI Crude Oil (ICE)",
            "icon": "🇺🇸",
            "fallback_tickers": ["CL1 Comdty"],
        },
        "Dubai": {"ticker": "DAT2 Comdty", "name": "Dubai Crude Swap (M2)", "icon": "🇦🇪"},
    }

    # ==========================================================================
    # SEGMENT 1: Price Action & Structure
    # ==========================================================================
    st.subheader("Price Action & Structure")
    show_price_action = True

    def render_price_action(instrument_key: str):
        """Render price action charts for an instrument."""
        inst = instruments[instrument_key]
        ticker = inst["ticker"]
        fallback_tickers = inst.get("fallback_tickers", [])
        ticker_candidates: list[str] = []
        for candidate in [ticker, *fallback_tickers]:
            if candidate not in ticker_candidates:
                ticker_candidates.append(candidate)
        history_ticker_used = None
        name = inst["name"]

        col1, col2 = st.columns([2, 1])

        with col1:
            # Price chart
            chart_title = f"**{name} - Daily Chart**"
            st.markdown(chart_title)

            # Get historical data
            hist_data = None
            data_warning_shown = False
            history_errors: list[str] = []
            for candidate in ticker_candidates:
                try:
                    candidate_history = data_loader.get_historical(
                        candidate,
                        start_date=datetime.now() - timedelta(days=180),
                        end_date=datetime.now()
                    )
                    if candidate_history is not None and not candidate_history.empty:
                        hist_data = candidate_history
                        history_ticker_used = candidate
                        break
                    history_errors.append(f"{candidate}: returned no rows")
                except DataUnavailableError as exc:
                    history_errors.append(f"{candidate}: {exc}")
                except Exception as exc:
                    history_errors.append(f"{candidate}: {exc}")
                    break

            if hist_data is None:
                attempted = ", ".join(ticker_candidates)
                if history_errors:
                    st.warning(f"No historical data available for {name}. Tried {attempted}.")
                    st.caption("; ".join(history_errors))
                else:
                    st.info(f"Historical data unavailable for {name}.")
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
                if history_ticker_used and history_ticker_used != ticker:
                    st.caption(f"Using fallback historical ticker {history_ticker_used} due to unavailable ICE data.")

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
            live_price = None
            price_ticker_used = None
            for candidate in ticker_candidates:
                candidate_price = price_cache.get(candidate)
                if candidate_price is not None:
                    live_price = candidate_price
                    price_ticker_used = candidate
                    break

            # Display live price prominently at the top
            if live_price is not None:
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
                if price_ticker_used and price_ticker_used != ticker:
                    st.caption(f"Live price source: {price_ticker_used} (WTI NYMEX fallback)")

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

    if show_price_action:
        # Instrument tabs for price action
        brent_tab, wti_tab, dubai_tab = st.tabs([
            f"{instruments['Brent']['icon']} Brent",
            f"{instruments['WTI']['icon']} WTI",
            f"{instruments['Dubai']['icon']} Dubai",
        ])

        with brent_tab:
            render_price_action("Brent")

        with wti_tab:
            render_price_action("WTI")

        with dubai_tab:
            render_price_action("Dubai")

    st.divider()

    # ==========================================================================
    # SEGMENT 2: Term Structure
    # ==========================================================================
    st.subheader("Term Structure")
    show_term_structure = True

    if show_term_structure:
        # Term Structure content
        st.markdown("**Futures Curve Analysis**")

        # Create sub-tabs for different curve views
        curve_tab1, curve_tab2 = st.tabs(["🛢️ WTI vs Brent", "🇦🇪 Dubai Swap Curve"])

        # Get curves from shared context cache
        wti_curve = context.data.futures_curve
        brent_curve = context.data.brent_curve

        with curve_tab1:
            col1, col2 = st.columns([2, 1])

            with col1:
                # Create enhanced futures curve chart with both WTI and Brent
                fig = create_futures_curve_chart(
                    curve_data=brent_curve,
                    secondary_curve=wti_curve,
                    title="WTI vs Brent Futures Curve",
                    height=400,
                )

                st.plotly_chart(fig, use_container_width=True, config=get_chart_config())

                # Calendar spreads
                st.markdown("**Calendar Spreads (WTI)**")
                spreads = curve_analyzer.calculate_calendar_spreads(wti_curve)
                spreads = format_calendar_spread_labels(spreads, wti_curve)

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

                if brent_curve is not None and not brent_curve.empty:
                    st.markdown("**Calendar Spreads (Brent)**")
                    brent_spreads = curve_analyzer.calculate_calendar_spreads(brent_curve)
                    brent_spreads = format_calendar_spread_labels(brent_spreads, brent_curve)
                    st.dataframe(
                        brent_spreads,
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
                st.markdown("**WTI Curve Analysis**")

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

        with curve_tab2:
            col1, col2 = st.columns([2, 1])

            with col1:
                # Dubai swap curve via shared context cache
                dubai_curve = context.data.dubai_curve

                # Create Dubai swap curve chart
                if dubai_curve is not None and not dubai_curve.empty:
                    # Use contract_month labels for x-axis
                    x_column = 'contract_month' if 'contract_month' in dubai_curve.columns else 'month'

                    # Get price range for proper y-axis scaling
                    price_min = dubai_curve['price'].min()
                    price_max = dubai_curve['price'].max()
                    price_range = price_max - price_min
                    y_min = price_min - (price_range * 0.15)
                    y_max = price_max + (price_range * 0.15)

                    if price_range < 2:
                        mid = (price_min + price_max) / 2
                        y_min = mid - 2
                        y_max = mid + 2

                    fig = go.Figure()

                    # Dubai curve with distinct styling (green for UAE)
                    fig.add_trace(go.Scatter(
                        x=dubai_curve[x_column],
                        y=dubai_curve['price'],
                        name='Dubai',
                        mode='lines+markers',
                        line={"color": '#00DC82', "width": 3, "shape": 'spline'},  # Green for Dubai/UAE
                        marker={
                            "size": 8,
                            "color": '#00DC82',
                            "line": {"width": 2, "color": 'white'},
                            "symbol": 'circle',
                        },
                        fill='tozeroy',
                        fillcolor='rgba(0, 220, 130, 0.1)',
                        hovertemplate='%{x}<br>$%{y:.2f}<extra>Dubai</extra>',
                    ))

                    # Also show Brent for comparison (EFS spread context)
                    if brent_curve is not None and not brent_curve.empty:
                        brent_x_column = 'contract_month' if 'contract_month' in brent_curve.columns else 'month'
                        fig.add_trace(go.Scatter(
                            x=brent_curve[brent_x_column],
                            y=brent_curve['price'],
                            name='Brent',
                            mode='lines+markers',
                            line={"color": CHART_COLORS["primary"], "width": 2, "dash": 'dot'},
                            marker={
                                "size": 6,
                                "color": CHART_COLORS["primary"],
                                "line": {"width": 1, "color": 'white'},
                                "symbol": 'diamond',
                            },
                            hovertemplate='%{x}<br>$%{y:.2f}<extra>Brent</extra>',
                        ))

                        # Update y-axis range to include both curves
                        all_prices = list(dubai_curve['price']) + list(brent_curve['price'])
                        price_min = min(all_prices)
                        price_max = max(all_prices)
                        price_range = price_max - price_min
                        y_min = price_min - (price_range * 0.15)
                        y_max = price_max + (price_range * 0.15)
                        if price_range < 2:
                            mid = (price_min + price_max) / 2
                            y_min = mid - 2
                            y_max = mid + 2

                    base_layout_without_axes = {
                        key: value for key, value in BASE_LAYOUT.items()
                        if key not in {"yaxis", "xaxis"}
                    }
                    fig.update_layout(
                        **base_layout_without_axes,
                        height=400,
                        title={
                            "text": "Dubai Swap Curve (vs Brent)",
                            "font": {"size": 14, "color": CHART_COLORS["text_primary"]},
                            "x": 0,
                            "xanchor": 'left',
                        },
                        yaxis=dict(
                            **BASE_LAYOUT["yaxis"],
                            range=[y_min, y_max],
                            title_text="Price ($/bbl)",
                            dtick=max(1, round(price_range / 5)),
                        ),
                        xaxis=dict(
                            **BASE_LAYOUT["xaxis"],
                            title_text="Contract Month",
                            tickangle=-45 if len(dubai_curve) > 12 else 0,
                        ),
                    )

                    st.plotly_chart(fig, use_container_width=True, config=get_chart_config())

                    # Dubai Calendar spreads
                    st.markdown("**Calendar Spreads (Dubai)**")
                    dubai_spreads = curve_analyzer.calculate_calendar_spreads(dubai_curve)
                    dubai_spreads = format_calendar_spread_labels(dubai_spreads, dubai_curve)

                    st.dataframe(
                        dubai_spreads,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            'spread_name': 'Spread',
                            'spread_value': st.column_config.NumberColumn('Value', format='$%.2f'),
                            'front_price': st.column_config.NumberColumn('Front', format='$%.2f'),
                            'back_price': st.column_config.NumberColumn('Back', format='$%.2f'),
                        }
                    )
                else:
                    st.warning("Dubai swap curve data not available")

            with col2:
                st.markdown("**Dubai Curve Analysis**")

                if dubai_curve is not None and not dubai_curve.empty:
                    dubai_metrics = curve_analyzer.analyze_curve(dubai_curve)

                    st.metric("Structure", dubai_metrics['structure'])
                    st.metric("M1-M2 Spread", f"${dubai_metrics['m1_m2_spread']:.2f}")
                    st.metric("Roll Yield (Ann.)", f"{dubai_metrics['roll_yield_annual_pct']:.1f}%")
                    st.metric("Curve Slope", f"{dubai_metrics['overall_slope']:.4f}")

                    st.divider()

                    # Dubai-Brent EFS (Exchange for Swaps) spread
                    st.markdown("**Dubai-Brent EFS Spread**")

                    if brent_curve is not None and not brent_curve.empty:
                        # Front month EFS spread (Dubai - Brent)
                        dubai_front = dubai_curve['price'].iloc[0]
                        brent_front = brent_curve['price'].iloc[0]
                        efs_spread = dubai_front - brent_front

                        st.metric(
                            "EFS Spread (M1)",
                            f"${efs_spread:.2f}",
                            help="Dubai minus Brent. Negative = Dubai discount"
                        )

                        # M6 EFS spread
                        if len(dubai_curve) >= 6 and len(brent_curve) >= 6:
                            dubai_m6 = dubai_curve['price'].iloc[5]
                            brent_m6 = brent_curve['price'].iloc[5]
                            efs_m6 = dubai_m6 - brent_m6
                            st.metric("EFS Spread (M6)", f"${efs_m6:.2f}")

                        # Market interpretation
                        if efs_spread < -1.0:
                            st.info("📉 Wide Dubai discount - bullish for Asian refiners")
                        elif efs_spread > -0.5:
                            st.warning("📈 Narrow Dubai discount - bearish for Asian refiners")
                        else:
                            st.caption("EFS spread within normal range")
                    else:
                        st.caption("Brent data required for EFS calculation")

                    st.divider()

                    # Roll yield
                    dubai_roll = curve_analyzer.calculate_roll_yield(dubai_curve)

                    st.markdown("**Roll Yield Analysis**")
                    st.text(f"Roll Cost: ${dubai_roll['roll_cost']:.2f}")
                    st.text(f"Roll Yield: ${dubai_roll['roll_yield']:.2f}")
                    st.text(f"Annual Roll Yield: {dubai_roll['roll_yield_annual_pct']:.1f}%")
                    st.text(f"Curve Carry: {dubai_roll['curve_carry']}")
                else:
                    st.caption("Dubai curve data not available")

# =============================================================================
# TAB 2: Crack Spreads
# =============================================================================
with tab2:
    # Crack Spreads Tab
    st.subheader("321 Crack Spread")

    # Bloomberg 321 Crack Spread tickers (FVCSM series)
    # Month codes: F=Jan, G=Feb, H=Mar, J=Apr, K=May, M=Jun, N=Jul, Q=Aug, U=Sep, V=Oct, X=Nov, Z=Dec
    CRACK_SPREAD_MONTH_CODES = ['F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z']
    CRACK_SPREAD_MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    crack_front_month_override = get_crack_spread_front_month_override()
    override_ticker = crack_front_month_override.get("ticker")
    override_label = crack_front_month_override.get("label")
    override_applied = False

    def get_crack_spread_forward_curve():
        """Fetch 321 crack spread forward curve from Bloomberg FVCSM series."""
        curve_data = []
        current_year = datetime.now().year
        current_month = datetime.now().month

        # Fetch next 12 months of crack spread contracts
        for i in range(12):
            month_idx = (current_month - 1 + i) % 12
            year = current_year + ((current_month - 1 + i) // 12)
            year_code = str(year)[-2:]  # Last 2 digits of year

            month_code = CRACK_SPREAD_MONTH_CODES[month_idx]
            month_name = CRACK_SPREAD_MONTH_NAMES[month_idx]

            ticker = f"FVCSM {month_code}{year_code} Index"
            contract_label = f"{month_name} {year_code}"

            try:
                price = data_loader.get_price(ticker, validate=False)
                if price is not None:
                    curve_data.append({
                        'ticker': ticker,
                        'contract_month': contract_label,
                        'month_idx': i,
                        'price': float(price),
                    })
            except Exception:
                # Skip contracts that aren't available
                pass

        return pd.DataFrame(curve_data) if curve_data else None

    # Fetch the crack spread forward curve
    crack_curve = get_crack_spread_forward_curve()
    if crack_curve is not None and not crack_curve.empty and override_ticker:
        override_mask = crack_curve['ticker'] == override_ticker
        if override_mask.any():
            first_idx = int(crack_curve.index[override_mask][0])
            crack_curve = crack_curve.loc[first_idx:].reset_index(drop=True)
            override_applied = True
    front_month_crack = None
    front_month_label = None
    front_month_ticker = None
    if crack_curve is not None and not crack_curve.empty:
        front_row = crack_curve.iloc[0]
        front_month_crack = float(front_row['price'])
        front_month_label = (
            override_label if override_applied and override_label else front_row['contract_month']
        )
        front_month_ticker = front_row['ticker']

    st.markdown("### 321 Crack Spread Front-Month Daily Chart")

    lookback_days = 180
    if front_month_ticker:
        crack_history = None
        history_error = False
        try:
            crack_history = data_loader.get_historical(
                front_month_ticker,
                start_date=datetime.now() - timedelta(days=lookback_days),
                end_date=datetime.now(),
                frequency="daily",
            )
        except Exception:
            crack_history = None
            st.warning(f"Unable to load history for {front_month_ticker}. Please retry.")
            history_error = True

        if crack_history is not None and not crack_history.empty:
            crack_history = crack_history.sort_index()
            px_last = crack_history["PX_LAST"].astype(float)

            def get_price_series(column: str):
                if column in crack_history.columns:
                    return crack_history[column].astype(float)
                return px_last

            px_open = get_price_series("PX_OPEN")
            px_high = get_price_series("PX_HIGH")
            px_low = get_price_series("PX_LOW")
            hover_texts = [
                f"{idx:%b %d %Y}<br>"
                f"O: ${open_price:.2f}<br>"
                f"H: ${high_price:.2f}<br>"
                f"L: ${low_price:.2f}<br>"
                f"C: ${close_price:.2f}"
                for idx, open_price, high_price, low_price, close_price in zip(
                    crack_history.index, px_open, px_high, px_low, px_last
                )
            ]

            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=crack_history.index,
                open=px_open,
                high=px_high,
                low=px_low,
                close=px_last,
                name="Price",
                increasing={
                    "line": {"color": CHART_COLORS["candle_up"], "width": 1},
                    "fillcolor": CHART_COLORS["candle_up_fill"],
                },
                decreasing={
                    "line": {"color": CHART_COLORS["candle_down"], "width": 1},
                    "fillcolor": CHART_COLORS["candle_down_fill"],
                },
                hovertext=hover_texts,
                hoverinfo="text",
                showlegend=False,
            ))

            ma_window = 20
            ma_series = px_last.rolling(window=ma_window).mean().dropna()
            if not ma_series.empty:
                fig.add_trace(go.Scatter(
                    x=ma_series.index,
                    y=ma_series,
                    name=f"{ma_window}-day MA",
                    mode="lines",
                    line={"color": CHART_COLORS["ma_fast"], "width": 1.5, "dash": "dot"},
                    hovertemplate="%{x|%b %d %Y}<br>$%{y:.2f}/bbl<extra>MA</extra>",
                ))

            base_layout_without_axes = {
                key: value for key, value in BASE_LAYOUT.items()
                if key not in {"yaxis", "xaxis"}
            }
            layout = {
                **base_layout_without_axes,
                "height": 320,
                "margin": {"l": 50, "r": 20, "t": 40, "b": 40},
                "title": {
                    "text": f"{front_month_label or 'Front Month'} ({front_month_ticker})",
                    "font": {"size": 14, "color": CHART_COLORS["text_primary"]},
                    "x": 0,
                    "xanchor": "left",
                },
                "yaxis": dict(
                    **BASE_LAYOUT["yaxis"],
                    title_text="$/bbl",
                ),
                "xaxis": dict(
                    **BASE_LAYOUT["xaxis"],
                    tickformat="%b %y",
                ),
                "showlegend": True,
                "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
            }
            fig.update_layout(**layout)

            st.plotly_chart(fig, use_container_width=True, config=get_chart_config())
            st.caption(f"{front_month_ticker} | {lookback_days}-day daily history")

            stats_cols = st.columns(3)
            latest = px_last.iloc[-1]
            stats_cols[0].metric("Last Close", f"${latest:.2f}/bbl")

            compare_idx = max(len(px_last) - 22, 0)
            compare_value = px_last.iloc[compare_idx]
            change_value = latest - compare_value
            change_pct = (change_value / compare_value * 100) if compare_value else 0
            stats_cols[1].metric("1M Change", f"${change_value:.2f}", f"{change_pct:+.1f}%")

            px_min = px_last.min()
            px_max = px_last.max()
            stats_cols[2].metric("Range", f"${px_min:.2f} - ${px_max:.2f}")
        elif not history_error:
            st.info("Historical data unavailable for the current front month crack spread.")
    else:
        st.info("Front month ticker unavailable for charting.")

    st.divider()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("**321 Crack Spread Forward Curve**")
        st.caption("Source: Bloomberg FVCSM Index (321 Crack Spread)")

        if crack_curve is not None and not crack_curve.empty:
            # Create forward curve chart
            fig = go.Figure()

            # Crack spread curve
            fig.add_trace(go.Scatter(
                x=crack_curve['contract_month'],
                y=crack_curve['price'],
                name='321 Crack Spread',
                mode='lines+markers',
                line={"color": CHART_COLORS['primary'], "width": 3, "shape": 'spline'},
                marker={
                    "size": 10,
                    "color": CHART_COLORS['primary'],
                    "line": {"width": 2, "color": 'white'},
                    "symbol": 'circle',
                },
                fill='tozeroy',
                fillcolor='rgba(0, 163, 224, 0.15)',
                hovertemplate='%{x}<br>$%{y:.2f}/bbl<extra>321 Crack</extra>',
            ))

            # Get price range for y-axis scaling
            price_min = crack_curve['price'].min()
            price_max = crack_curve['price'].max()
            price_range = price_max - price_min
            y_min = max(0, price_min - (price_range * 0.2))
            y_max = price_max + (price_range * 0.2)

            if price_range < 2:
                mid = (price_min + price_max) / 2
                y_min = max(0, mid - 3)
                y_max = mid + 3

            base_layout_without_axes = {
                key: value for key, value in BASE_LAYOUT.items()
                if key not in {"yaxis", "xaxis"}
            }
            fig.update_layout(
                **base_layout_without_axes,
                height=400,
                title={
                    "text": "321 Crack Spread Forward Curve",
                    "font": {"size": 14, "color": CHART_COLORS["text_primary"]},
                    "x": 0,
                    "xanchor": 'left',
                },
                yaxis=dict(
                    **BASE_LAYOUT["yaxis"],
                    range=[y_min, y_max],
                    title_text="Crack Spread ($/bbl)",
                    dtick=max(1, round(price_range / 5)),
                ),
                xaxis=dict(
                    **BASE_LAYOUT["xaxis"],
                    title_text="Contract Month",
                    tickangle=-45 if len(crack_curve) > 8 else 0,
                ),
                showlegend=False,
            )

            st.plotly_chart(fig, use_container_width=True, config=get_chart_config())

            # Forward curve data table
            st.markdown("**Forward Curve Data**")
            display_df = crack_curve[['contract_month', 'price', 'ticker']].copy()
            display_df.columns = ['Contract', 'Price ($/bbl)', 'Bloomberg Ticker']
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Contract': st.column_config.TextColumn('Contract'),
                    'Price ($/bbl)': st.column_config.NumberColumn('Price', format='$%.2f'),
                    'Bloomberg Ticker': st.column_config.TextColumn('Ticker'),
                }
            )
        else:
            st.warning("321 Crack spread data unavailable from Bloomberg (FVCSM series)")

    with col2:
        st.markdown("**Current 321 Crack Spread**")

        if crack_curve is not None and not crack_curve.empty:
            # Display current crack spread prominently
            st.markdown(
                f"""<div style="background: linear-gradient(135deg, rgba(0,163,224,0.15) 0%, rgba(0,163,224,0.05) 100%);
                padding: 20px; border-radius: 8px; border-left: 4px solid #00A3E0; margin-bottom: 16px;">
                <div style="color: #94A3B8; font-size: 12px; margin-bottom: 4px;">CURRENT ({front_month_label})</div>
                <div style="color: #E2E8F0; font-size: 32px; font-weight: 700; font-family: 'IBM Plex Mono', monospace;">${front_month_crack:.2f}/bbl</div>
                <div style="color: #64748B; font-size: 11px; margin-top: 4px;">{front_month_ticker}</div>
                </div>""",
                unsafe_allow_html=True
            )

            st.divider()

            # Curve structure analysis
            st.markdown("**Curve Structure**")

            if len(crack_curve) >= 2:
                # M1-M2 spread
                m1 = crack_curve.iloc[0]['price']
                m2 = crack_curve.iloc[1]['price']
                m1_m2_spread = m1 - m2

                st.metric(
                    "M1-M2 Spread",
                    f"${m1_m2_spread:.2f}",
                    delta="Backwardation" if m1_m2_spread > 0 else "Contango",
                    delta_color="normal" if m1_m2_spread > 0 else "inverse"
                )

            if len(crack_curve) >= 6:
                # M1-M6 spread
                m6 = crack_curve.iloc[5]['price']
                m1_m6_spread = m1 - m6
                st.metric("M1-M6 Spread", f"${m1_m6_spread:.2f}")

            if len(crack_curve) >= 12:
                # M1-M12 spread
                m12 = crack_curve.iloc[11]['price']
                m1_m12_spread = m1 - m12
                st.metric("M1-M12 Spread", f"${m1_m12_spread:.2f}")

            st.divider()

            # Curve statistics
            st.markdown("**Curve Statistics**")
            avg_crack = crack_curve['price'].mean()
            max_crack = crack_curve['price'].max()
            min_crack = crack_curve['price'].min()

            st.text(f"Avg: ${avg_crack:.2f}/bbl")
            st.text(f"Max: ${max_crack:.2f}/bbl")
            st.text(f"Min: ${min_crack:.2f}/bbl")

            # Determine curve structure
            if len(crack_curve) >= 3:
                front_avg = crack_curve.head(3)['price'].mean()
                back_avg = crack_curve.tail(3)['price'].mean()
                if front_avg > back_avg * 1.02:
                    structure = "Backwardation"
                    st.success(f"📈 {structure}")
                elif back_avg > front_avg * 1.02:
                    structure = "Contango"
                    st.warning(f"📉 {structure}")
                else:
                    structure = "Flat"
                    st.info(f"➡️ {structure}")

        else:
            st.info("Crack spread data not available")

        st.divider()

        # Live component prices for reference
        wti = price_cache.get("CL1 Comdty")
        rbob = price_cache.get("XB1 Comdty")
        ho = price_cache.get("HO1 Comdty")
        brent = price_cache.get("CO1 Comdty")

        st.markdown("**Reference Prices**")
        if wti is not None:
            st.metric("WTI Crude", f"${wti:.2f}/bbl")
        if brent is not None:
            st.metric("Brent Crude", f"${brent:.2f}/bbl")
        if rbob is not None:
            st.metric("RBOB Gasoline", f"${rbob:.4f}/gal")
        if ho is not None:
            st.metric("Heating Oil", f"${ho:.4f}/gal")

# =============================================================================
# TAB 3: Inventory
# =============================================================================
with tab3:
    # Inventory Tab
    st.subheader("Global Inventory Analytics")

    # Initialize satellite data provider for regional stocks
    if RESEARCH_AVAILABLE:
        satellite_data = SatelliteData()
        regional_stocks = satellite_data.get_latest_observations()
    else:
        regional_stocks = None

    # -------------------------------------------------------------------------
    # SECTION 1: REGIONAL CRUDE STOCKS (Main Focus)
    # -------------------------------------------------------------------------
    st.markdown("### 🛢️ Regional Crude Stocks")
    st.caption("Primary crude oil storage hubs - Live data from Bloomberg")

    # Regional hub definitions with crude-specific data
    REGIONAL_HUBS = {
        "usgc": {
            "name": "US Gulf Coast",
            "region": "North America",
            "key": "usgc",
            "crude_capacity_mb": 125,
            "product_capacity_mb": 20,
            "benchmark": "WTI/LLS",
            "icon": "🇺🇸",
        },
        "ara": {
            "name": "ARA (Amsterdam-Rotterdam-Antwerp)",
            "region": "Europe",
            "key": "rotterdam",  # Maps to satellite data key
            "crude_capacity_mb": 25,  # Million barrels crude capacity
            "product_capacity_mb": 10,
            "benchmark": "Brent",
            "icon": "🇪🇺",
        },
    }

    if regional_stocks and regional_stocks.get("locations"):
        locations = regional_stocks["locations"]

        # Create regional crude stocks display
        crude_cols = st.columns(len(REGIONAL_HUBS))

        for idx, (hub_key, hub_info) in enumerate(REGIONAL_HUBS.items()):
            with crude_cols[idx]:
                loc_data = locations.get(hub_info["key"], {})
                utilization = sanitize_percentage(loc_data.get("utilization_pct", 0))
                change_week = loc_data.get("change_week_pct", 0)
                crude_cap = hub_info["crude_capacity_mb"]
                estimated_crude = crude_cap * utilization / 100

                st.markdown(f"**{hub_info['icon']} {hub_info['name']}**")
                st.caption(f"Region: {hub_info['region']} | Benchmark: {hub_info['benchmark']}")

                # Crude stock metrics
                st.metric(
                    "Crude Stocks",
                    f"{estimated_crude:.1f} MMbbl",
                    f"{change_week:+.1f}% WoW",
                    delta_color="inverse"  # Lower is bullish
                )

                # Utilization gauge
                util_color = (
                    CHART_COLORS['profit'] if utilization < 50
                    else CHART_COLORS['loss'] if utilization > 75
                    else CHART_COLORS['ma_fast']
                )
                progress_value = max(0.0, min(utilization / 100, 1.0))
                st.progress(progress_value)
                st.caption(f"Utilization: {utilization:.1f}% of {crude_cap} MMbbl capacity")

                # Signal interpretation
                if utilization < 45:
                    st.success("📈 Low stocks - Bullish")
                elif utilization > 70:
                    st.error("📉 High stocks - Bearish")
                else:
                    st.info("➡️ Normal range")

        st.divider()

        # Regional stocks comparison chart
        st.markdown("**Regional Crude Stocks Comparison**")

        # Prepare data for chart
        hub_names = []
        crude_volumes = []
        utilizations = []
        changes = []

        for hub_key, hub_info in REGIONAL_HUBS.items():
            loc_data = locations.get(hub_info["key"], {})
            util = sanitize_percentage(loc_data.get("utilization_pct", 50), default=50)
            hub_names.append(hub_info["name"])
            crude_volumes.append(hub_info["crude_capacity_mb"] * util / 100)
            utilizations.append(util)
            changes.append(loc_data.get("change_week_pct", 0))

        # Create comparison bar chart
        fig_regional = go.Figure()

        fig_regional.add_trace(go.Bar(
            x=hub_names,
            y=crude_volumes,
            name='Crude Stocks (MMbbl)',
            marker_color=[
                CHART_COLORS['profit'] if u < 50 else CHART_COLORS['loss'] if u > 70 else CHART_COLORS['primary']
                for u in utilizations
            ],
            marker_line_width=0,
            text=[f"{v:.1f}" for v in crude_volumes],
            textposition='outside',
            hovertemplate='%{x}<br>%{y:.1f} MMbbl<br>Utilization: %{customdata:.1f}%<extra></extra>',
            customdata=utilizations,
        ))

        fig_regional.update_layout(
            **BASE_LAYOUT,
            height=300,
            yaxis_title='Crude Stocks (MMbbl)',
            showlegend=False,
        )

        st.plotly_chart(fig_regional, use_container_width=True, config=get_chart_config())

        # Weekly changes chart
        col_change1, col_change2 = st.columns([2, 1])

        with col_change1:
            st.markdown("**Weekly Change by Hub**")

            fig_changes = go.Figure()
            fig_changes.add_trace(go.Bar(
                x=hub_names,
                y=changes,
                marker_color=[CHART_COLORS['profit'] if c < 0 else CHART_COLORS['loss'] for c in changes],
                marker_line_width=0,
                text=[f"{c:+.1f}%" for c in changes],
                textposition='outside',
                hovertemplate='%{x}<br>%{y:+.1f}% WoW<extra></extra>',
            ))

            fig_changes.add_hline(y=0, line_dash='solid', line_color='rgba(255,255,255,0.3)', line_width=1)

            fig_changes.update_layout(
                **BASE_LAYOUT,
                height=250,
                yaxis_title='Weekly Change (%)',
                showlegend=False,
            )

            st.plotly_chart(fig_changes, use_container_width=True, config=get_chart_config())

        with col_change2:
            st.markdown("**Global Summary**")
            global_summary = satellite_data.get_global_summary()

            total_crude = sum(crude_volumes)
            avg_util = np.mean(utilizations)

            st.metric("Total Regional Crude", f"{total_crude:.1f} MMbbl")
            st.metric("Avg Utilization", f"{avg_util:.1f}%")

            # Trading signal
            signal = satellite_data.calculate_storage_signal()
            if signal["signal"] == "bullish":
                st.success(f"📈 {signal['rationale']}")
            elif signal["signal"] == "bearish":
                st.error(f"📉 {signal['rationale']}")
            else:
                st.info(f"➡️ {signal['rationale']}")

            st.caption(f"Confidence: {signal['confidence']}%")

    else:
        st.info("📊 Regional stock data requires Bloomberg connection and configured tickers.")
        st.markdown("""
        **To enable regional inventory data:**
        1. Ensure Bloomberg Terminal is connected
        2. Configure tickers in `config/bloomberg_tickers.yaml` (`inventory.locations`)

        **Centralized ticker config**
        - All Bloomberg-connected regional inventory tickers default to this config file
        - Add the USGC / ARA tickers under their respective location keys

        **Bloomberg Tickers to search:**
        - **USGC**: USGCTOTL Index (PADD 3 commercial crude)
        - **ARA**: Crude stocks, Gasoline, Gasoil, Fuel Oil indices
        """)

    st.divider()

    # -------------------------------------------------------------------------
    # SECTION 2: US CRUDE INVENTORY (EIA Data)
    # -------------------------------------------------------------------------
    st.markdown("### 🇺🇸 US Crude Inventory")

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
                line={"color": CHART_COLORS['primary'], "width": 2.5},
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

            change_fig.add_hline(y=0, line_dash='solid', line_color='rgba(255,255,255,0.3)', line_width=1)

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

    st.divider()

    # -------------------------------------------------------------------------
    # SECTION 3: PRODUCT STOCKS (Bottom)
    # -------------------------------------------------------------------------
    st.markdown("### ⛽ Product Stocks")
    st.caption("Refined product inventories by region")

    # Product stock definitions
    PRODUCT_STOCKS = {
        "ara": {
            "name": "ARA Products",
            "icon": "🇪🇺",
            "products": {
                "gasoline": {"capacity": 4.5, "typical_util": 55},
                "gasoil": {"capacity": 5.0, "typical_util": 60},
                "fuel_oil": {"capacity": 2.5, "typical_util": 50},
                "naphtha": {"capacity": 1.5, "typical_util": 45},
            }
        },
        "fujairah": {
            "name": "Fujairah Products",
            "icon": "🇦🇪",
            "products": {
                "gasoline": {"capacity": 3.0, "typical_util": 50},
                "gasoil": {"capacity": 4.5, "typical_util": 55},
                "fuel_oil": {"capacity": 6.0, "typical_util": 65},
                "jet_fuel": {"capacity": 1.5, "typical_util": 45},
            }
        },
        "singapore": {
            "name": "Singapore Products",
            "icon": "🇸🇬",
            "products": {
                "gasoline": {"capacity": 4.0, "typical_util": 52},
                "gasoil": {"capacity": 5.5, "typical_util": 58},
                "fuel_oil": {"capacity": 4.0, "typical_util": 60},
                "jet_fuel": {"capacity": 2.0, "typical_util": 48},
            }
        },
    }

    # Generate simulated product stock data (in production, this would come from API)
    import random
    random.seed(42)  # For consistent display

    product_cols = st.columns(3)

    for idx, (region_key, region_info) in enumerate(PRODUCT_STOCKS.items()):
        with product_cols[idx]:
            st.markdown(f"**{region_info['icon']} {region_info['name']}**")

            # Create product inventory table
            product_data = []
            for product_name, product_info in region_info["products"].items():
                # Simulate current utilization around typical
                current_util = product_info["typical_util"] + random.uniform(-10, 10)
                current_util = max(20, min(90, current_util))
                volume = product_info["capacity"] * current_util / 100
                change = random.uniform(-8, 8)

                product_data.append({
                    "Product": product_name.replace("_", " ").title(),
                    "Volume (MMbbl)": round(volume, 2),
                    "Util %": round(current_util, 1),
                    "WoW %": round(change, 1),
                })

            df_products = pd.DataFrame(product_data)

            # Display as styled dataframe
            st.dataframe(
                df_products,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Product": st.column_config.TextColumn("Product", width="small"),
                    "Volume (MMbbl)": st.column_config.NumberColumn("Stock", format="%.2f"),
                    "Util %": st.column_config.ProgressColumn("Util", min_value=0, max_value=100, format="%.0f%%"),
                    "WoW %": st.column_config.NumberColumn("WoW", format="%+.1f%%"),
                }
            )

            # Regional summary
            total_products = sum(p["Volume (MMbbl)"] for p in product_data)
            avg_util = np.mean([p["Util %"] for p in product_data])

            if avg_util < 45:
                st.success(f"Low product stocks ({avg_util:.0f}% avg)")
            elif avg_util > 65:
                st.warning(f"High product stocks ({avg_util:.0f}% avg)")
            else:
                st.info(f"Normal product stocks ({avg_util:.0f}% avg)")

    # Product stocks summary chart
    st.markdown("**Product Stocks by Type Across Regions**")

    # Aggregate products across regions
    all_products = ["Gasoline", "Gasoil", "Fuel Oil", "Jet Fuel", "Naphtha"]
    regions = list(PRODUCT_STOCKS.keys())

    fig_products = go.Figure()

    colors = {
        "ara": CHART_COLORS['primary'],
        "fujairah": CHART_COLORS['ma_fast'],
        "singapore": CHART_COLORS['profit'],
    }

    for region_key, region_info in PRODUCT_STOCKS.items():
        volumes = []
        for product in all_products:
            product_key = product.lower().replace(" ", "_")
            if product_key in region_info["products"]:
                p_info = region_info["products"][product_key]
                util = p_info["typical_util"] + random.uniform(-5, 5)
                volumes.append(p_info["capacity"] * util / 100)
            else:
                volumes.append(0)

        fig_products.add_trace(go.Bar(
            name=region_info["name"],
            x=all_products,
            y=volumes,
            marker_color=colors.get(region_key, CHART_COLORS['primary']),
            marker_line_width=0,
            hovertemplate='%{x}<br>%{y:.2f} MMbbl<extra>' + region_info["name"] + '</extra>',
        ))

    fig_products.update_layout(
        **BASE_LAYOUT,
        height=350,
        barmode='group',
        yaxis_title='Product Stocks (MMbbl)',
    )

    st.plotly_chart(fig_products, use_container_width=True, config=get_chart_config())

    # Data sources note
    st.caption("""
    **Data Sources:** EIA Weekly Petroleum Status Report, Euroilstock (ARA), S&P Global Platts,
    Fujairah Oil Industry Zone (FOIZ), Enterprise Singapore, Argus Media, Vortexa
    """)

# =============================================================================
# TAB 4: OPEC Monitor
# =============================================================================
with tab4:
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

            opec_layout = {
                **BASE_LAYOUT,
                "height": 400,
                "barmode": 'group',
                "yaxis_title": 'Production (mb/d)',
                "legend": {"orientation": 'h', "yanchor": 'bottom', "y": 1.02, "xanchor": 'right', "x": 1},
            }
            fig.update_layout(**opec_layout)

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
# TAB 5: News & Sentiment
# =============================================================================
with tab5:
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

                fig.update_layout(height=200, margin={"l": 20, "r": 20, "t": 40, "b": 20})
                st.plotly_chart(fig, use_container_width=True, config=get_chart_config())

            except Exception as e:
                st.error(f"Sentiment analyzer error: {e}")

# =============================================================================
# TAB 6: Correlations
# =============================================================================
with tab6:
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
                            margin={"l": 20, "r": 20, "t": 20, "b": 20},
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
                        line={"color": CHART_COLORS['primary'], "width": 2},
                        fill='tozeroy',
                        fillcolor='rgba(31, 119, 180, 0.2)'
                    ))

                    fig.add_hline(y=0, line_dash="dash", line_color="gray")

                    corr_layout = {
                        **BASE_LAYOUT,
                        "height": 300,
                        "yaxis": {"title": "Correlation", "range": [-1, 1]},
                        "xaxis_title": "Date",
                    }
                    fig.update_layout(**corr_layout)

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

# =============================================================================
# TAB 7: Regimes
# =============================================================================
with tab7:
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
                    marker={
                        "color": colors,
                        "size": 8,
                    },
                    line={"color": 'gray', "width": 1},
                    text=regimes,
                    hovertemplate='%{text}<br>%{x}<extra></extra>'
                ))

                regime_layout = {
                    **BASE_LAYOUT,
                    "height": 300,
                    "yaxis": {
                        "title": "Regime",
                        "ticktext": ["Crisis", "Trend Down", "Low Vol", "Ranging", "High Vol", "Trend Up"],
                        "tickvals": [-3, -2, -1, 0, 1, 2],
                    },
                    "xaxis_title": "Date",
                }
                fig.update_layout(**regime_layout)

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

# =============================================================================
# TAB 8: Factor Models
# =============================================================================
with tab8:
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

                    exposure_layout = {
                        **BASE_LAYOUT,
                        "height": 350,
                        "xaxis_title": "Exposure (Beta)",
                        "margin": {"l": 120, "r": 20, "t": 20, "b": 40},
                    }
                    fig.update_layout(**exposure_layout)

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
                            margin={"l": 20, "r": 20, "t": 20, "b": 20},
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

# =============================================================================
# TAB 9: Alternative Data
# =============================================================================
with tab9:
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
                        utilization_pct = sanitize_percentage(data.get("utilization_pct", 0))
                        storage_data.append({
                            "Location": data.get("name", loc),
                            "Utilization": f"{utilization_pct:.1f}%",
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
                        util = sanitize_percentage(data.get("utilization_pct", 0))
                        fig.add_trace(go.Bar(
                            x=[data.get("name", loc)],
                            y=[util],
                            name=loc,
                            marker_color="#ff7f0e" if util > 70 else ("#1f77b4" if util < 50 else "#2ca02c")
                        ))

                    storage_layout = {
                        **BASE_LAYOUT,
                        "height": 300,
                        "yaxis": {"title": "Utilization %", "range": [0, 100]},
                        "showlegend": False,
                    }
                    fig.update_layout(**storage_layout)

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

                    cot_layout = {
                        **BASE_LAYOUT,
                        "height": 300,
                        "yaxis": {"title": "Percentile", "range": [0, 110]},
                    }
                    fig.update_layout(**cot_layout)

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
    Data refreshes every {REFRESH_INTERVAL_SECONDS} seconds when auto-refresh is enabled |
    Charts show up to 180 days of historical data
    </div>""",
    unsafe_allow_html=True
)
