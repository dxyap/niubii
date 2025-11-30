"""
Oil Trading Dashboard - Streamlit entry point.
Streamlit dashboard structured with reusable services for clarity and performance.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app import shared_state
from app.dashboard_core import DashboardContext, PortfolioAnalytics

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
    .stApp { background-color: #0E1117; }
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: bold;
        font-family: 'JetBrains Mono', 'SF Mono', monospace;
    }
    [data-testid="stMetricDelta"] {
        font-size: 14px;
        font-family: 'JetBrains Mono', 'SF Mono', monospace;
    }
    h1, h2, h3 { font-family: 'Inter', -apple-system, sans-serif; font-weight: 600; }
    .metric-card {
        background-color: #1E2127;
        border-radius: 8px;
        padding: 20px;
        border: 1px solid #2D3139;
    }
    .profit { color: #00D26A !important; }
    .loss { color: #FF4B4B !important; }
    .dataframe { font-family: 'JetBrains Mono', monospace; font-size: 13px; }
    [data-testid="stSidebar"] {
        background-color: #0E1117;
        border-right: 1px solid #2D3139;
    }
    .stButton > button {
        background-color: #00A3E0;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 500;
    }
    .stButton > button:hover { background-color: #0088C2; }
    .live-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        background-color: #00D26A;
        border-radius: 50%;
        margin-right: 6px;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    .status-ok { color: #00D26A; }
    .status-warning { color: #FFA500; }
    .status-critical { color: #FF4B4B; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
"""


class DashboardApp:
    """Main application orchestrator."""

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
        self.refresh_interval = int(os.getenv("AUTO_REFRESH_INTERVAL", "5"))
        self._init_session_state()
        self.data_loader = shared_state.get_data_loader()
        self._ensure_core_subscriptions()
        self.context = DashboardContext(self.data_loader, shared_state.get_positions())
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
        self._handle_auto_refresh()

    # --------------------------------------------------------------------- #
    # SETUP
    # --------------------------------------------------------------------- #

    def _init_session_state(self) -> None:
        if "last_refresh" not in st.session_state:
            st.session_state.last_refresh = datetime.now()
        if "auto_refresh" not in st.session_state:
            st.session_state.auto_refresh = True

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
        data_mode = connection_status.get("data_mode", "simulated").title()
        indicator_color = "#00D26A" if data_mode.lower() == "live" else "#00A3E0"

        with st.sidebar:
            st.image("https://img.icons8.com/fluency/96/000000/oil-barrel.png", width=60)
            st.title("Oil Trading")
            st.caption("Quantitative Dashboard")
            st.divider()

            st.markdown(
                f'<span class="live-indicator" style="background:{indicator_color};"></span>'
                f"<span style='color:{indicator_color};'>{data_mode} Data</span>",
                unsafe_allow_html=True,
            )
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
        col1, col2 = st.columns(2)
        with col1:
            self._render_metric("WTI", prices.get("WTI", {}))
        with col2:
            self._render_metric("Brent", prices.get("Brent", {}))
        spread = self.context.data.wti_brent_spread
        st.metric("WTI-Brent", f"${spread.get('spread', 0):.2f}", f"{spread.get('change', 0):+0.2f}")

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
            self._render_metric("WTI Crude", prices.get("WTI", {}))
        with col2:
            self._render_metric("Brent Crude", prices.get("Brent", {}))
        with col3:
            st.metric("WTI-Brent", f"${spread.get('spread', 0):.2f}", f"{spread.get('change', 0):+0.2f}")
        with col4:
            st.metric("3-2-1 Crack", f"${crack.get('crack', 0):.2f}", f"{crack.get('change', 0):+0.2f}")
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
        st.subheader("WTI Crude Oil Price")
        hist_data = self.context.data.wti_history
        if hist_data is None or hist_data.empty:
            st.info("Historical data unavailable.")
            return

        chart_data = pd.DataFrame({"Price": hist_data["PX_LAST"]})
        st.line_chart(chart_data, height=300, use_container_width=True)

        stat_cols = st.columns(4)
        with stat_cols[0]:
            current = hist_data["PX_LAST"].iloc[-1]
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
        st.subheader("WTI Futures Curve")
        curve = self.context.data.futures_curve
        if curve is None or curve.empty:
            self._curve_structure = "Unknown"
            st.info("Futures curve unavailable.")
            return
        curve_chart = pd.DataFrame({"Month": curve["month"], "Price": curve["price"]})
        st.bar_chart(curve_chart.set_index("Month"), height=250, use_container_width=True)

        metrics = self.context.data.curve_metrics()
        cols = st.columns(4)
        cols[0].metric("M1-M2 Spread", f"${metrics['m1_m2']:.2f}")
        cols[1].metric("M1-M6 Spread", f"${metrics['m1_m6']:.2f}")
        cols[2].metric("M1-M12 Spread", f"${metrics['m1_m12']:.2f}")
        cols[3].metric("Structure", metrics["structure"])
        self._curve_structure = metrics["structure"]

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
            f"<h3 style='color:{'#00D26A' if total_pnl >= 0 else '#FF4B4B'};'>"
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

    def _handle_auto_refresh(self) -> None:
        if st.session_state.auto_refresh:
            time.sleep(self.refresh_interval)
            st.session_state.last_refresh = datetime.now()
            st.rerun()


def main() -> None:
    app = DashboardApp()
    app.apply_theme()
    app.run()


if __name__ == "__main__":
    main()
