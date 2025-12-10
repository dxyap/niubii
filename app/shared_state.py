"""
Shared State Module
===================
Manages shared state across all dashboard pages.

This module provides:
- Centralized data loader access
- Position management
- Portfolio P&L calculations
- Dashboard context caching
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st
from dotenv import load_dotenv

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv()

from .dashboard_core import DashboardContext, PortfolioAnalytics, Position

if TYPE_CHECKING:
    from core.data import DataLoader


def get_data_loader() -> DataLoader:
    """Get or create the shared data loader."""
    if 'data_loader' not in st.session_state:
        from core.data import DataLoader
        st.session_state.data_loader = DataLoader(
            config_dir=str(project_root / "config"),
            data_dir=str(project_root / "data"),
        )
    return st.session_state.data_loader


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


def get_dashboard_context(lookback_days: int = 90, force_refresh: bool = False) -> DashboardContext:
    """
    Return a cached DashboardContext shared across pages.

    The context is cached based on:
    - Lookback days
    - Position signature (changes when positions are modified)
    - Time-based refresh (max 60 second cache for price data)

    This reduces redundant data fetching while ensuring data freshness.
    """
    positions = get_positions()
    signature = _positions_signature(positions)
    current_time = datetime.now()

    store = st.session_state.setdefault("_dashboard_context_store", {})
    entry = store.get(lookback_days)

    if entry and not force_refresh:
        context, cached_time, cached_signature = entry
        time_diff = (current_time - cached_time).total_seconds()

        # Cache is valid if:
        # 1. Position signature hasn't changed
        # 2. Cache is less than 60 seconds old (for live price data)
        if cached_signature == signature and time_diff < 60:
            return context

    # Create new context with prefetched data
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
