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
        # Default demo positions
        st.session_state.positions = [
            {"symbol": "CLF5", "ticker": "CL1 Comdty", "qty": 45, "entry": 72.15, "strategy": "Momentum"},
            {"symbol": "CLG5", "ticker": "CL2 Comdty", "qty": 20, "entry": 72.50, "strategy": "Spread"},
            {"symbol": "COH5", "ticker": "CO1 Comdty", "qty": -15, "entry": 78.20, "strategy": "Arb"},
            {"symbol": "XBF5", "ticker": "XB1 Comdty", "qty": 8, "entry": 2.15, "strategy": "Crack"},
        ]
    return st.session_state.positions


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
