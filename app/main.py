"""
Oil Trading Dashboard - Streamlit entry point.
Streamlit dashboard structured with reusable services for clarity and performance.

Performance Optimizations:
- Batch API calls for price fetching
- Request deduplication for concurrent identical requests
- Streamlit caching for expensive operations
- Market-hours aware TTL for smart cache expiration
- Auto-refresh using st.fragment (non-blocking)
- Lazy loading of historical data
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Export cached functions for use by other pages
__all__ = [
    "get_historical_data_cached",
    "get_futures_curve_cached",
    "get_oil_prices_cached",
    "get_all_oil_prices_cached",
    "get_all_spreads_cached",
    "get_connection_status_cached",
    "get_term_structure_cached",
]

# Page configuration must be the first Streamlit call
st.set_page_config(
    page_title="Oil Trading Dashboard",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Quantitative Oil Trading Dashboard"},
)

from app import shared_state
from app.components.theme import apply_theme as apply_dashboard_theme
from app.page_utils import render_status_bar
from app.ui import (
    ActivityFeedView,
    KeyMetricsView,
    MarketOverviewView,
    PortfolioView,
    RefreshController,
    SidebarView,
)

# =============================================================================
# STREAMLIT CACHING FOR EXPENSIVE OPERATIONS
# =============================================================================
# These cached functions reduce Bloomberg API calls by caching results at the
# Streamlit layer. TTLs are tuned based on how frequently the data changes:
# - Real-time prices: 60 seconds (balance freshness vs. API calls)
# - Historical data: 300 seconds (rarely changes intraday)
# - Connection status: 30 seconds (doesn't change often)
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
def get_futures_curve_cached(commodity: str = "brent", num_months: int = 18):
    """
    Cache futures curve with 1-minute TTL.
    Curve data changes throughout the day but not every second.
    Default to 18 months ahead of front month.
    """
    data_loader = shared_state.get_data_loader()
    try:
        return data_loader.get_futures_curve(commodity, num_months)
    except Exception:
        return None


@st.cache_data(ttl=60, show_spinner=False)
def get_oil_prices_cached():
    """
    Cache oil prices with 1-minute TTL.
    Prices change throughout the day but 60s cache significantly
    reduces API calls during dashboard navigation.
    """
    data_loader = shared_state.get_data_loader()
    try:
        return data_loader.get_oil_prices()
    except Exception:
        return {}


@st.cache_data(ttl=60, show_spinner=False)
def get_all_oil_prices_cached():
    """
    Cache all oil prices (extended list) with 1-minute TTL.
    """
    data_loader = shared_state.get_data_loader()
    try:
        return data_loader.get_all_oil_prices()
    except Exception:
        return {}


@st.cache_data(ttl=60, show_spinner=False)
def get_all_spreads_cached():
    """
    Cache all spreads (WTI-Brent, 3-2-1 crack, 2-1-1 crack) with 1-minute TTL.
    Fetches all spreads in a single optimized batch call.
    """
    data_loader = shared_state.get_data_loader()
    try:
        return data_loader.get_all_spreads()
    except Exception:
        return {}


@st.cache_data(ttl=30, show_spinner=False)
def get_connection_status_cached():
    """
    Cache connection status with 30-second TTL.
    Status rarely changes, no need to check every render.
    """
    data_loader = shared_state.get_data_loader()
    try:
        return data_loader.get_connection_status()
    except Exception:
        return {"connected": False, "data_mode": "disconnected"}


@st.cache_data(ttl=60, show_spinner=False)
def get_term_structure_cached(commodity: str = "wti"):
    """
    Cache term structure analysis with 1-minute TTL.
    """
    data_loader = shared_state.get_data_loader()
    try:
        return data_loader.get_term_structure(commodity)
    except Exception:
        return {"structure": "Unknown", "slope": 0}


class DashboardApp:
    """Main dashboard orchestrator that wires views and shared state."""

    def __init__(self):
        load_dotenv()
        self.refresh_interval = int(os.getenv("AUTO_REFRESH_INTERVAL", "60"))
        self.refresh_controller = RefreshController(self.refresh_interval)
        self.data_loader = shared_state.get_data_loader()
        self._ensure_core_subscriptions()

        self.context = shared_state.get_dashboard_context()
        self.sidebar_view = SidebarView(self.context, self.refresh_controller)
        self.key_metrics_view = KeyMetricsView(self.context)
        self.market_view = MarketOverviewView(
            self.context,
            historical_loader=get_historical_data_cached,
            curve_loader=get_futures_curve_cached,
        )
        self.portfolio_view = PortfolioView(self.context)
        self.activity_view = ActivityFeedView(self.context)

    def apply_theme(self) -> None:
        """Inject global CSS via shared theme helper."""
        apply_dashboard_theme(st)

    def run(self) -> None:
        """Render the dashboard."""
        self.sidebar_view.render()
        self._render_header()
        self.key_metrics_view.render()
        self._render_market_sections()
        self.activity_view.render()
        self._render_footer()
        self.refresh_controller.schedule_auto_refresh(self._clear_caches)

    def _render_header(self) -> None:
        st.title("Oil Trading Dashboard")

        connection_status = self.data_loader.get_connection_status()
        data_mode = connection_status.get("data_mode", "disconnected")

        if data_mode == "disconnected":
            st.error("Bloomberg Terminal not connected. Live data required.")
            st.info(f"Connection error: {connection_status.get('connection_error', 'Unknown')}")
            if st.button("Retry Bloomberg Connection", key="retry_live_disconnected"):
                if self.data_loader.try_reconnect():
                    st.success("Bloomberg connection restored. Reloading data...")
                else:
                    st.warning("Bloomberg connection still unavailable.")
                self._trigger_rerun()
            st.stop()

        render_status_bar(
            data_mode=data_mode,
            last_refresh=self.refresh_controller.last_refresh,
            timezone=connection_status.get("timezone", "Asia/Singapore"),
            latency_ms=connection_status.get("latency_ms"),
        )

    def _render_market_sections(self) -> None:
        left_col, right_col = st.columns([2, 1])
        with left_col:
            self.market_view.render_price_chart()
            curve_structure = self.market_view.render_curve_section()
        with right_col:
            self.portfolio_view.render_positions()
            self.portfolio_view.render_risk(curve_structure)

    def _render_footer(self) -> None:
        st.divider()
        st.caption(
            "Oil Trading Dashboard | "
            f"Data refreshed at {self.refresh_controller.format_last_refresh('%Y-%m-%d %H:%M:%S')} | "
            "Signals are informational only"
        )

    def _ensure_core_subscriptions(self) -> None:
        if not st.session_state.get("core_ticker_subscription"):
            self.data_loader.subscribe_to_core_tickers()
            st.session_state.core_ticker_subscription = True

    @staticmethod
    def _clear_caches() -> None:
        """Clear short-lived caches when auto-refresh fires."""
        get_historical_data_cached.clear()
        get_futures_curve_cached.clear()

    @staticmethod
    def _trigger_rerun() -> None:
        """Trigger a Streamlit rerun (compatibility helper)."""
        if hasattr(st, "rerun"):
            st.rerun()
        else:
            st.experimental_rerun()


def main() -> None:
    app = DashboardApp()
    app.apply_theme()
    app.run()


if __name__ == "__main__":
    main()
