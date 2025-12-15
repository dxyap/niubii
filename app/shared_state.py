"""
Shared State Module
===================
Manages shared state across all dashboard pages.

This module provides:
- Centralized data loader access (cached with @st.cache_resource)
- Position management
- Portfolio P&L calculations
- Dashboard context caching

Performance Optimizations:
- DataLoader cached as resource (expensive to create)
- Context caching with time-based invalidation
- Lazy initialization patterns
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st
from dotenv import load_dotenv

# Add project root to path for imports (done once at module load)
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Load environment once at module import
load_dotenv()

from .dashboard_core import DashboardContext, PortfolioAnalytics, Position

if TYPE_CHECKING:
    from core.data import DataLoader


@st.cache_resource(show_spinner=False)
def _create_data_loader() -> "DataLoader":
    """
    Create DataLoader instance (cached as resource).
    
    This is expensive to create due to:
    - YAML config loading
    - Bloomberg connection setup
    - Cache initialization
    
    Using @st.cache_resource ensures it's created once per session.
    """
    from core.data import DataLoader
    return DataLoader(
        config_dir=str(project_root / "config"),
        data_dir=str(project_root / "data"),
    )


def get_data_loader() -> "DataLoader":
    """
    Get the shared data loader (cached resource).
    
    Uses @st.cache_resource internally for efficient caching.
    Falls back to session_state for backward compatibility.
    """
    # Use cached resource version (most efficient)
    return _create_data_loader()


def get_positions() -> list[dict]:
    """Get current positions from session state."""
    if 'positions' not in st.session_state:
        # Start with empty positions - no demo data
        st.session_state.positions = []
    return st.session_state.positions


def add_position(
    symbol: str,
    ticker: str,
    qty: int,
    entry: float,
    strategy: str | None = None
) -> None:
    """Add a new position."""
    positions = get_positions()
    positions.append({
        "symbol": symbol,
        "ticker": ticker,
        "qty": qty,
        "entry": entry,
        "strategy": strategy or "Manual",
    })
    st.session_state.positions = positions


def clear_positions() -> None:
    """Clear all positions."""
    st.session_state.positions = []


def calculate_position_pnl(data_loader=None):
    """Calculate P&L for all positions using shared portfolio analytics."""
    data_loader = data_loader or get_data_loader()
    positions = [Position(**pos) for pos in get_positions()]
    analytics = PortfolioAnalytics(positions, data_loader)

    df = analytics.positions_dataframe
    if df.empty:
        return []

    records = df.to_dict("records")
    for rec in records:
        try:
            rec["multiplier"] = data_loader.get_multiplier(rec["ticker"])
        except Exception:
            rec["multiplier"] = 1000
    return records


def get_portfolio_summary(data_loader=None):
    """Get portfolio-level summary metrics using shared analytics pipeline."""
    data_loader = data_loader or get_data_loader()
    positions = [Position(**pos) for pos in get_positions()]
    analytics = PortfolioAnalytics(positions, data_loader)
    summary = analytics.summary
    summary["positions"] = analytics.positions_dataframe.to_dict("records")
    return summary


def format_pnl(value: float) -> str:
    """Format P&L with sign."""
    if value >= 0:
        return f"+${value:,.0f}"
    return f"-${abs(value):,.0f}"


def format_pnl_with_color(value: float) -> tuple:
    """Return formatted P&L and color."""
    color = "#00D26A" if value >= 0 else "#FF4B4B"
    formatted = format_pnl(value)
    return formatted, color


def _positions_signature(positions):
    """Create a hashable signature for current positions."""
    sorted_positions = sorted(
        (
            pos.get("symbol"),
            pos.get("ticker"),
            float(pos.get("qty", 0)),
            float(pos.get("entry", 0)),
            pos.get("strategy", ""),
        )
        for pos in positions
    )
    return tuple(sorted_positions)


# Cache TTL constants (seconds)
_CONTEXT_CACHE_TTL = 30  # Reduced from 60 for fresher data
_CONTEXT_CACHE_TTL_OFF_HOURS = 300  # Longer cache during off-hours


def _get_context_cache_ttl() -> int:
    """Get appropriate cache TTL based on market hours."""
    try:
        from core.data.cache import is_market_hours
        return _CONTEXT_CACHE_TTL if is_market_hours() else _CONTEXT_CACHE_TTL_OFF_HOURS
    except ImportError:
        return _CONTEXT_CACHE_TTL


def get_dashboard_context(lookback_days: int = 90, force_refresh: bool = False) -> DashboardContext:
    """
    Return a cached DashboardContext shared across pages.

    The context is cached based on:
    - Lookback days
    - Position signature (changes when positions are modified)
    - Time-based refresh (market-hours aware TTL)

    Performance notes:
    - Avoid using force_refresh=True unless absolutely necessary
    - Cache TTL is shorter during market hours for fresher data
    - DataLoader is cached separately via @st.cache_resource
    """
    positions = get_positions()
    signature = _positions_signature(positions)
    current_time = datetime.now()

    store = st.session_state.setdefault("_dashboard_context_store", {})
    entry = store.get(lookback_days)

    if entry and not force_refresh:
        context, cached_time, cached_signature = entry
        time_diff = (current_time - cached_time).total_seconds()
        cache_ttl = _get_context_cache_ttl()

        # Cache is valid if:
        # 1. Position signature hasn't changed
        # 2. Cache is within TTL
        if cached_signature == signature and time_diff < cache_ttl:
            return context

    # Create new context with prefetched data
    # DataLoader is cached via @st.cache_resource, so this is efficient
    context = DashboardContext(
        get_data_loader(),
        positions,
        lookback_days=lookback_days,
    )
    store[lookback_days] = (context, current_time, signature)
    st.session_state["_dashboard_context_store"] = store
    return context


def invalidate_context_cache():
    """Force refresh of the dashboard context on next access."""
    if "_dashboard_context_store" in st.session_state:
        del st.session_state["_dashboard_context_store"]
