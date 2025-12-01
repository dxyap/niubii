"""
Oil Trading Dashboard - Streamlit entry point.
Streamlit dashboard structured with reusable services for clarity and performance.

Performance Optimizations:
- Batch API calls for price fetching
- Streamlit caching for expensive operations
- Auto-refresh using st.fragment (non-blocking)
- Lazy loading of historical data
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app import shared_state
from app.dashboard_core import DashboardContext, PortfolioAnalytics


# =============================================================================
# STREAMLIT CACHING FOR EXPENSIVE OPERATIONS
# =============================================================================

@st.cache_data(ttl=300, show_spinner=False)
def get_historical_data_cached(ticker: str = "CO1 Comdty", lookback_days: int = 90):
    """
    Cache historical data with 5-minute TTL.
    Historical data doesn't change frequently, so caching is safe.
    """
    data_loader = shared_state.get_data_loader()
    end = datetime.now()
    start = end - timedelta(days=lookback_days)
    try:
        return data_loader.get_historical(ticker, start_date=start, end_date=end)
    except Exception:
        return None


@st.cache_data(ttl=60, show_spinner=False)
def get_futures_curve_cached(commodity: str = "brent", num_months: int = 12):
    """
    Cache futures curve with 1-minute TTL.
    Curve data changes throughout the day but not every second.
    """
    data_loader = shared_state.get_data_loader()
    try:
        return data_loader.get_futures_curve(commodity, num_months)
    except Exception:
        return None

# Page configuration must be the first Streamlit call
st.set_page_config(
    page_title="Oil Trading Dashboard",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Quantitative Oil Trading Dashboard"},
)

THEME_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Outfit:wght@400;500;600;700&display=swap');
    
    .stApp { 
        background: linear-gradient(135deg, #0a0f1a 0%, #111827 50%, #0f172a 100%);
    }
    
    [data-testid="stMetricValue"] {
        font-size: 26px;
        font-weight: 600;
        font-family: 'IBM Plex Mono', 'SF Mono', monospace;
        color: #e2e8f0 !important;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 13px;
        font-family: 'IBM Plex Mono', monospace;
    }
    
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-family: 'Outfit', sans-serif;
        font-weight: 500;
        text-transform: uppercase;
        font-size: 11px;
        letter-spacing: 0.05em;
    }
    
    h1 { 
        font-family: 'Outfit', sans-serif; 
        font-weight: 700; 
        color: #f1f5f9 !important;
        letter-spacing: -0.02em;
    }
    
    h2, h3 { 
        font-family: 'Outfit', sans-serif; 
        font-weight: 600; 
        color: #e2e8f0 !important;
    }
    
    p, span, div { color: #cbd5e1; }
    
    .stMarkdown { color: #cbd5e1; }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid #334155;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #f8fafc !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown { color: #94a3b8; }
    
    .stButton > button {
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-family: 'Outfit', sans-serif;
        padding: 0.5rem 1rem;
        transition: all 0.2s ease;
        box-shadow: 0 2px 8px rgba(14, 165, 233, 0.25);
    }
    
    .stButton > button:hover { 
        background: linear-gradient(135deg, #38bdf8 0%, #0ea5e9 100%);
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.4);
        transform: translateY(-1px);
    }
    
    .live-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        background-color: #22c55e;
        border-radius: 50%;
        margin-right: 8px;
        box-shadow: 0 0 8px rgba(34, 197, 94, 0.6);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; box-shadow: 0 0 8px rgba(34, 197, 94, 0.6); }
        50% { opacity: 0.7; box-shadow: 0 0 16px rgba(34, 197, 94, 0.4); }
        100% { opacity: 1; box-shadow: 0 0 8px rgba(34, 197, 94, 0.6); }
    }
    
    .profit { color: #22c55e !important; }
    .loss { color: #ef4444 !important; }
    
    .dataframe { 
        font-family: 'IBM Plex Mono', monospace; 
        font-size: 13px;
        color: #e2e8f0;
    }
    
    [data-testid="stDataFrame"] {
        background: rgba(30, 41, 59, 0.5);
        border-radius: 8px;
        border: 1px solid #334155;
    }
    
    .stProgress > div > div {
        background: linear-gradient(90deg, #0ea5e9 0%, #22c55e 100%);
    }
    
    .stDivider {
        border-color: #334155;
    }
    
    [data-testid="stExpander"] {
        background: rgba(30, 41, 59, 0.3);
        border: 1px solid #334155;
        border-radius: 8px;
    }
    
    .status-ok { color: #22c55e; }
    .status-warning { color: #f59e0b; }
    .status-critical { color: #ef4444; }
    
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    
    /* Chart container styling */
    [data-testid="stVegaLiteChart"] {
        background: rgba(15, 23, 42, 0.4);
        border-radius: 8px;
        padding: 8px;
        border: 1px solid #334155;
    }
</style>
"""


class DashboardApp:
    """
    Main application orchestrator.
    
    Performance optimizations:
    - Uses cached functions for expensive data fetches
    - Non-blocking auto-refresh with streamlit-autorefresh or manual rerun
    - Batch API calls for price data
    """

    NAV_LINKS = (
        ("main.py", "Overview"),
        ("pages/1_\U0001F4C8_Market_Insights.py", "Market Insights"),
        ("pages/2_\U0001F4E1_Signals.py", "Signals"),
        ("pages/3_\U0001F6E1\ufe0f_Risk.py", "Risk Management"),
        ("pages/4_\U0001F4BC_Trade_Entry.py", "Trade Entry"),
        ("pages/5_\U0001F4CB_Blotter.py", "Trade Blotter"),
        ("pages/6_\U0001F4CA_Analytics.py", "Analytics"),
    )

    def __init__(self):
        load_dotenv()
        self.refresh_interval = int(os.getenv("AUTO_REFRESH_INTERVAL", "10"))  # Increased default
        self._init_session_state()
        self.data_loader = shared_state.get_data_loader()
        self._ensure_core_subscriptions()
        
        # Use shared context for better caching
        self.context = shared_state.get_dashboard_context()
        self._curve_structure = "Unknown"

    def apply_theme(self) -> None:
        """Inject global CSS."""
        st.markdown(THEME_CSS, unsafe_allow_html=True)

    def run(self) -> None:
        """Render the dashboard."""
        self._render_sidebar()
        self._render_header()
        self._render_key_metrics()
        self._render_market_sections()
        self._render_bottom_sections()
        self._render_footer()
        self._setup_auto_refresh()

    # --------------------------------------------------------------------- #
    # SETUP
    # --------------------------------------------------------------------- #

    def _init_session_state(self) -> None:
        if "last_refresh" not in st.session_state:
            st.session_state.last_refresh = datetime.now()
        if "auto_refresh" not in st.session_state:
            st.session_state.auto_refresh = True
        if "refresh_count" not in st.session_state:
            st.session_state.refresh_count = 0

    def _ensure_core_subscriptions(self) -> None:
        if not st.session_state.get("core_ticker_subscription"):
            self.data_loader.subscribe_to_core_tickers()
            st.session_state.core_ticker_subscription = True

    # --------------------------------------------------------------------- #
    # SIDEBAR
    # --------------------------------------------------------------------- #

    def _render_sidebar(self) -> None:
        data_bundle = self.context.data
        connection_status = data_bundle.connection_status
        data_mode = connection_status.get("data_mode", "disconnected")
        connection_error = connection_status.get("connection_error")
        
        # Color based on connection status
        if data_mode == "live":
            indicator_color = "#00D26A"  # Green
            status_text = "Live Data"
        elif data_mode == "mock":
            indicator_color = "#FFA500"  # Orange
            status_text = "Mock Data (Dev)"
        else:
            indicator_color = "#FF4B4B"  # Red
            status_text = "Disconnected"

        with st.sidebar:
            st.image("https://img.icons8.com/fluency/96/000000/oil-barrel.png", width=60)
            st.title("Oil Trading")
            st.caption("Quantitative Dashboard")
            st.divider()

            st.markdown(
                f'<span class="live-indicator" style="background:{indicator_color};"></span>'
                f"<span style='color:{indicator_color};'>{status_text}</span>",
                unsafe_allow_html=True,
            )
            
            # Show connection error if disconnected
            if data_mode == "disconnected" and connection_error:
                st.error(f"⚠️ {connection_error[:100]}")
            
            st.caption(f"Last Update: {st.session_state.last_refresh.strftime('%H:%M:%S')}")

            toggle_label = f"Auto Refresh ({self.refresh_interval}s)"
            auto_refresh = st.toggle(toggle_label, value=st.session_state.auto_refresh)
            st.session_state.auto_refresh = auto_refresh

            if st.button("Refresh Now", use_container_width=True):
                st.session_state.last_refresh = datetime.now()
                st.rerun()

            st.divider()
            self._render_quick_prices()
            st.divider()

            for path, label in self.NAV_LINKS:
                st.page_link(path, label=label)

    def _render_quick_prices(self) -> None:
        prices = self.context.data.oil_prices
        st.subheader("Quick View")
        
        if prices is None:
            st.warning("Price data unavailable")
            return
        
        col1, col2 = st.columns(2)
        with col1:
            self._render_metric("WTI", prices.get("WTI", {}))
        with col2:
            self._render_metric("Brent", prices.get("Brent", {}))
        
        spread = self.context.data.wti_brent_spread
        if spread is not None:
            st.metric("WTI-Brent", f"${spread.get('spread', 0):.2f}", f"{spread.get('change', 0):+0.2f}")
        else:
            st.metric("WTI-Brent", "N/A", "Data unavailable")

    @staticmethod
    def _render_metric(label: str, values: dict) -> None:
        current = values.get("current", 0.0)
        change = values.get("change", 0.0)
        change_pct = values.get("change_pct")
        suffix = f" ({change_pct:+.2f}%)" if change_pct is not None else ""
        delta = f"{change:+.2f}{suffix}"
        st.metric(label, f"${current:.2f}", delta, delta_color="normal" if change >= 0 else "inverse")

    # --------------------------------------------------------------------- #
    # MAIN CONTENT
    # --------------------------------------------------------------------- #

    def _render_header(self) -> None:
        st.title("Oil Trading Dashboard")
        st.caption("Real-time quantitative analysis for oil markets")

    def _render_key_metrics(self) -> None:
        prices = self.context.data.oil_prices
        spread = self.context.data.wti_brent_spread
        crack = self.context.data.crack_spread
        summary = self.context.portfolio.summary

        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            if prices is not None:
                self._render_metric("WTI Crude", prices.get("WTI", {}))
            else:
                st.metric("WTI Crude", "N/A", "No data")
        with col2:
            if prices is not None:
                self._render_metric("Brent Crude", prices.get("Brent", {}))
            else:
                st.metric("Brent Crude", "N/A", "No data")
        with col3:
            if spread is not None:
                st.metric("WTI-Brent", f"${spread.get('spread', 0):.2f}", f"{spread.get('change', 0):+0.2f}")
            else:
                st.metric("WTI-Brent", "N/A", "No data")
        with col4:
            if crack is not None:
                st.metric("3-2-1 Crack", f"${crack.get('crack', 0):.2f}", f"{crack.get('change', 0):+0.2f}")
            else:
                st.metric("3-2-1 Crack", "N/A", "No data")
        with col5:
            st.metric(
                "Day P&L",
                f"{PortfolioAnalytics.format_pnl(summary['total_pnl'])}",
                f"{summary['pnl_return_pct']:+.2f}%",
                delta_color="normal" if summary["total_pnl"] >= 0 else "inverse",
            )
        st.divider()

    def _render_market_sections(self) -> None:
        left_col, right_col = st.columns([2, 1])
        with left_col:
            self._render_price_chart()
            self._render_curve_section()
        with right_col:
            self._render_positions_section()
            self._render_risk_section()

    def _render_price_chart(self) -> None:
        st.subheader("Brent Crude Oil Price")
        # Use cached historical data (5-min TTL) - much faster!
        hist_data = get_historical_data_cached("CO1 Comdty", 90)
        if hist_data is None or hist_data.empty:
            st.info("Historical data unavailable.")
            return

        chart_data = pd.DataFrame({"Price": hist_data["PX_LAST"]})
        st.line_chart(chart_data, height=300, use_container_width=True, color="#0ea5e9")

        # Use LIVE price from oil_prices for "Current", historical data for stats
        prices = self.context.data.oil_prices
        live_brent = prices.get("Brent", {}).get("current") if prices else None
        
        stat_cols = st.columns(4)
        with stat_cols[0]:
            current = live_brent if live_brent else hist_data["PX_LAST"].iloc[-1]
            st.metric("Current", f"${current:.2f}")
        with stat_cols[1]:
            high = hist_data["PX_HIGH"].max()
            st.metric("90D High", f"${high:.2f}")
        with stat_cols[2]:
            low = hist_data["PX_LOW"].min()
            st.metric("90D Low", f"${low:.2f}")
        with stat_cols[3]:
            avg = hist_data["PX_LAST"].mean()
            st.metric("90D Avg", f"${avg:.2f}")

    def _render_curve_section(self) -> None:
        st.subheader("Brent Futures Curve")
        # Use cached curve data (1-min TTL)
        curve = get_futures_curve_cached("brent", 12)
        if curve is None or curve.empty:
            self._curve_structure = "Unknown"
            st.info("Futures curve unavailable.")
            return
        curve_chart = pd.DataFrame({"Month": curve["month"], "Price": curve["price"]})
        st.bar_chart(curve_chart.set_index("Month"), height=250, use_container_width=True, color="#0ea5e9")

        # Calculate metrics from cached curve
        prices = curve["price"]
        slope = (prices.iloc[-1] - prices.iloc[0]) / max(len(prices) - 1, 1)
        
        if slope > 0.05:
            structure = "Contango"
        elif slope < -0.05:
            structure = "Backwardation"
        else:
            structure = "Flat"
        
        def spread(idx: int) -> float:
            return float(prices.iloc[0] - prices.iloc[idx]) if len(prices) > idx else 0.0
        
        cols = st.columns(4)
        cols[0].metric("M1-M2 Spread", f"${spread(1):.2f}")
        cols[1].metric("M1-M6 Spread", f"${spread(5):.2f}")
        cols[2].metric("M1-M12 Spread", f"${spread(11):.2f}")
        cols[3].metric("Structure", structure)
        self._curve_structure = structure

    def _render_positions_section(self) -> None:
        st.subheader("Position Summary")
        table = self.context.portfolio.formatted_table()
        if table.empty:
            st.info("No active positions.")
            return

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Symbol": st.column_config.TextColumn("Symbol"),
                "Qty": st.column_config.NumberColumn("Qty", format="%d"),
                "Entry": st.column_config.NumberColumn("Entry", format="$%.2f"),
                "Current": st.column_config.NumberColumn("Current", format="$%.2f"),
                "P&L": st.column_config.TextColumn("P&L"),
            },
        )

        total_pnl = self.context.portfolio.summary["total_pnl"]
        st.markdown(
            f"<h3 style='color:{'#22c55e' if total_pnl >= 0 else '#ef4444'}; font-family: IBM Plex Mono, monospace;'>"
            f"Total P&L: {PortfolioAnalytics.format_pnl(total_pnl)}</h3>",
            unsafe_allow_html=True,
        )

    def _render_risk_section(self) -> None:
        st.divider()
        st.subheader("Risk Summary")
        summary = self.context.portfolio.summary
        st.progress(summary["var_utilization"] / 100, text=f"VaR Utilization: {summary['var_utilization']:.0f}%")

        risk_metrics = {
            "VaR (95%, 1-day)": f"${summary['var_estimate']:,.0f}",
            "VaR Limit": f"${summary['var_limit']:,.0f}",
            "Gross Exposure": f"${summary['gross_exposure'] / 1e6:.1f}M",
            "Net Position": f"{summary['net_contracts']:.0f} contracts",
        }
        for label, value in risk_metrics.items():
            st.text(f"{label}: {value}")

        alerts = self.context.generate_alerts(self._curve_structure, summary["var_utilization"])
        st.divider()
        st.subheader("Active Alerts")
        if not alerts:
            st.success("No active alerts.")
        else:
            for alert in alerts:
                severity = alert["severity"]
                message = alert["message"]
                if severity == "warning":
                    st.warning(message)
                elif severity == "critical":
                    st.error(message)
                else:
                    st.info(message)

    # --------------------------------------------------------------------- #
    # LOWER GRID
    # --------------------------------------------------------------------- #

    def _render_bottom_sections(self) -> None:
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Recent Trades")
            trades = self.context.generate_recent_trades()
            st.dataframe(trades, use_container_width=True, hide_index=True)
        with col2:
            st.subheader("Active Signals")
            signals = self.context.generate_signals()
            if signals.empty:
                st.info("No active signals.")
            else:
                st.dataframe(signals, use_container_width=True, hide_index=True)

    # --------------------------------------------------------------------- #
    # FOOTER / REFRESH
    # --------------------------------------------------------------------- #

    def _render_footer(self) -> None:
        st.divider()
        st.caption(
            f"Oil Trading Dashboard | "
            f"Data refreshed at {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')} | "
            "Signals are informational only"
        )

    def _setup_auto_refresh(self) -> None:
        """
        Non-blocking auto-refresh using streamlit-autorefresh or JavaScript fallback.
        This replaces the old time.sleep() approach which blocked the entire UI.
        """
        if not st.session_state.auto_refresh:
            return
        
        # Try using streamlit-autorefresh (preferred)
        try:
            from streamlit_autorefresh import st_autorefresh
            count = st_autorefresh(
                interval=self.refresh_interval * 1000,  # milliseconds
                limit=None,  # No limit
                key="auto_refresh_counter"
            )
            if count > st.session_state.refresh_count:
                st.session_state.refresh_count = count
                st.session_state.last_refresh = datetime.now()
                # Clear caches to get fresh price data
                get_futures_curve_cached.clear()
        except ImportError:
            # Fallback: JavaScript-based auto-refresh (non-blocking)
            refresh_js = f"""
            <script>
                setTimeout(function() {{
                    window.parent.document.querySelector('button[kind="secondary"]')?.click() ||
                    window.parent.location.reload();
                }}, {self.refresh_interval * 1000});
            </script>
            """
            st.markdown(refresh_js, unsafe_allow_html=True)


def main() -> None:
    app = DashboardApp()
    app.apply_theme()
    app.run()


if __name__ == "__main__":
    main()
