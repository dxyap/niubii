"""
Shared State Module
===================
Manages shared state across all dashboard pages.
"""

import streamlit as st
from datetime import datetime
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from .dashboard_core import DashboardContext


def get_data_loader():
    """Get or create the shared data loader."""
    if 'data_loader' not in st.session_state:
        from core.data import DataLoader
        # use_mock=None lets DataLoader read from BLOOMBERG_USE_MOCK env var
        st.session_state.data_loader = DataLoader(
            config_dir=str(project_root / "config"),
            data_dir=str(project_root / "data"),
            use_mock=None  # Auto-detect from environment (defaults to live data)
        )
    return st.session_state.data_loader


def get_positions():
    """Get current positions from session state."""
    if 'positions' not in st.session_state:
        # Start with empty positions - no demo data
        st.session_state.positions = []
    return st.session_state.positions


def add_position(symbol: str, ticker: str, qty: int, entry: float, strategy: str = None):
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


def clear_positions():
    """Clear all positions."""
    st.session_state.positions = []


def calculate_position_pnl(data_loader=None):
    """Calculate P&L for all positions using current prices."""
    if data_loader is None:
        data_loader = get_data_loader()
    
    positions = get_positions()
    results = []
    
    for pos in positions:
        current_price = data_loader.get_price(pos["ticker"])
        entry_price = pos["entry"]
        qty = pos["qty"]
        
        # Contract multiplier
        if pos["ticker"].startswith("XB") or pos["ticker"].startswith("HO"):
            multiplier = 42000  # 42,000 gallons per contract
        else:
            multiplier = 1000  # 1,000 barrels per contract
        
        # Calculate P&L
        price_change = current_price - entry_price
        pnl = price_change * qty * multiplier
        pnl_pct = (price_change / entry_price * 100) if entry_price != 0 else 0
        
        # Calculate notional
        notional = abs(qty) * current_price * multiplier
        
        results.append({
            "symbol": pos["symbol"],
            "ticker": pos["ticker"],
            "qty": qty,
            "entry": entry_price,
            "current": round(current_price, 4),
            "pnl": round(pnl, 0),
            "pnl_pct": round(pnl_pct, 2),
            "notional": round(notional, 0),
            "strategy": pos["strategy"],
            "multiplier": multiplier,
        })
    
    return results


def get_portfolio_summary(data_loader=None):
    """Get portfolio-level summary metrics."""
    if data_loader is None:
        data_loader = get_data_loader()
    
    position_pnl = calculate_position_pnl(data_loader)
    
    total_pnl = sum(p['pnl'] for p in position_pnl)
    gross_exposure = sum(p['notional'] for p in position_pnl)
    
    # Net exposure (long - short)
    long_exposure = sum(p['notional'] for p in position_pnl if p['qty'] > 0)
    short_exposure = sum(p['notional'] for p in position_pnl if p['qty'] < 0)
    net_exposure = long_exposure - short_exposure
    
    # VaR estimate (simplified: 2% of gross exposure)
    var_estimate = gross_exposure * 0.02
    var_limit = 375000
    
    return {
        "total_pnl": total_pnl,
        "gross_exposure": gross_exposure,
        "net_exposure": net_exposure,
        "long_exposure": long_exposure,
        "short_exposure": short_exposure,
        "var_estimate": var_estimate,
        "var_limit": var_limit,
        "var_utilization": (var_estimate / var_limit * 100) if var_limit > 0 else 0,
        "positions": position_pnl,
    }


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
    - Time-based refresh (max 30 second cache for price data)
    
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
        # 2. Cache is less than 30 seconds old (for live price data)
        if cached_signature == signature and time_diff < 30:
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
