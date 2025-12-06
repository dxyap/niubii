"""
Page Utilities
==============
Shared utilities for all dashboard pages to eliminate code duplication.

Usage:
    from app.page_utils import init_page, PageContext

    # Initialize page with standard setup
    ctx = init_page(
        title="📈 Market Insights",
        page_title="Market Insights | Oil Trading",
        icon="📈"
    )

    # Access shared context
    data_loader = ctx.data_loader
    price_cache = ctx.price_cache
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st
from dotenv import load_dotenv

# Ensure project root is in path (do this once at module load)
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

load_dotenv()

from app import shared_state
from app.components.theme import COLORS, apply_theme, get_chart_config

if TYPE_CHECKING:
    from app.dashboard_core import DashboardContext, PriceCache
    from core.data import DataLoader


@dataclass
class PageContext:
    """Container for commonly needed page context objects."""
    context: DashboardContext
    data_loader: DataLoader
    price_cache: PriceCache
    data_mode: str
    connection_status: dict

    @property
    def oil_prices(self):
        """Get oil prices from data bundle."""
        return self.context.data.oil_prices

    @property
    def portfolio(self):
        """Get portfolio analytics."""
        return self.context.portfolio

    @property
    def is_live(self) -> bool:
        """Check if using live data."""
        return self.data_mode == "live"

    @property
    def is_mock(self) -> bool:
        """Check if using mock data."""
        return self.data_mode == "mock"


def init_page(
    title: str,
    page_title: str,
    icon: str,
    layout: str = "wide",
    lookback_days: int = 90,
    show_connection_status: bool = True,
    require_data: bool = True,
) -> PageContext:
    """
    Initialize a dashboard page with standard configuration.

    This function handles:
    - Page configuration
    - Theme application
    - Shared state initialization
    - Connection status display

    Args:
        title: Page header title (with emoji)
        page_title: Browser tab title
        icon: Page icon for tab
        layout: Page layout ("wide" or "centered")
        lookback_days: Days of historical data to load
        show_connection_status: Whether to show connection indicator
        require_data: Stop page if no data connection

    Returns:
        PageContext with common objects
    """
    # Page configuration (must be first Streamlit call)
    st.set_page_config(
        page_title=page_title,
        page_icon=icon,
        layout=layout,
    )

    # Apply shared theme
    apply_theme(st)

    # Get shared context
    context = shared_state.get_dashboard_context(lookback_days=lookback_days)
    data_loader = context.data_loader
    price_cache = context.price_cache

    # Get connection status
    connection_status = data_loader.get_connection_status()
    data_mode = connection_status.get("data_mode", "disconnected")

    # Render title
    st.title(title)

    # Show connection status if requested
    if show_connection_status:
        if data_mode == "live":
            st.caption("🟢 Live market data from Bloomberg")
        elif data_mode == "mock":
            st.caption("⚠️ Simulated data mode — Bloomberg not connected")
        elif data_mode == "disconnected" and require_data:
            st.error("🔴 Bloomberg Terminal not connected. Live data required.")
            st.info(f"Connection error: {connection_status.get('connection_error', 'Unknown')}")
            st.stop()

    return PageContext(
        context=context,
        data_loader=data_loader,
        price_cache=price_cache,
        data_mode=data_mode,
        connection_status=connection_status,
    )


def render_live_status_bar(ctx: PageContext) -> None:
    """
    Render a compact live status bar with last update time.
    Only use this on pages that need prominent live status (like Market Insights).
    Most pages should rely on sidebar status indicator.
    """
    last_refresh = st.session_state.get('last_refresh', datetime.now())
    time_since = (datetime.now() - last_refresh).seconds

    if ctx.is_live:
        st.markdown(
            f"""<div style="display: flex; align-items: center; gap: 10px; padding: 8px 12px;
            background: linear-gradient(90deg, rgba(0,210,130,0.15) 0%, rgba(0,210,130,0.05) 100%);
            border-left: 3px solid #00D282; border-radius: 4px; margin-bottom: 1rem;">
            <span style="color: #00D282; font-weight: 600;">🟢 LIVE</span>
            <span style="color: #94A3B8;">Bloomberg Connected</span>
            <span style="color: #64748B; margin-left: auto;">Last update: {last_refresh.strftime('%H:%M:%S')} ({time_since}s ago)</span>
            </div>""",
            unsafe_allow_html=True
        )
    elif ctx.is_mock:
        st.markdown(
            f"""<div style="display: flex; align-items: center; gap: 10px; padding: 8px 12px;
            background: linear-gradient(90deg, rgba(245,158,11,0.15) 0%, rgba(245,158,11,0.05) 100%);
            border-left: 3px solid #F59E0B; border-radius: 4px; margin-bottom: 1rem;">
            <span style="color: #F59E0B; font-weight: 600;">🟡 SIMULATED</span>
            <span style="color: #94A3B8;">Development Mode</span>
            <span style="color: #64748B; margin-left: auto;">Last update: {last_refresh.strftime('%H:%M:%S')}</span>
            </div>""",
            unsafe_allow_html=True
        )


def get_cached_component(component_class, *args, cache_key: str = None, **kwargs):
    """
    Get or create a cached component instance.
    Uses Streamlit's session state for caching.

    Args:
        component_class: Class to instantiate
        *args: Positional args for constructor
        cache_key: Optional cache key (defaults to class name)
        **kwargs: Keyword args for constructor

    Returns:
        Cached or new instance
    """
    key = cache_key or f"_cached_{component_class.__name__}"

    if key not in st.session_state:
        st.session_state[key] = component_class(*args, **kwargs)

    return st.session_state[key]


# Re-export commonly used items for convenience
__all__ = [
    'init_page',
    'PageContext',
    'render_live_status_bar',
    'get_cached_component',
    'COLORS',
    'get_chart_config',
]
