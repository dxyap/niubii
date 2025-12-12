"""
Core Utilities
==============
Centralized utility functions used across the trading dashboard.

This module provides a single source of truth for common operations like:
- Number formatting (P&L, currency, percentages)
- Path resolution
- Data conversion helpers
"""

from __future__ import annotations

from pathlib import Path


# =============================================================================
# PROJECT PATHS (Single source of truth)
# =============================================================================

# Project root is two levels up from this file (core/utils.py -> workspace)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"


def get_project_root() -> Path:
    """Get the project root directory."""
    return PROJECT_ROOT


def get_config_dir() -> Path:
    """Get the configuration directory."""
    return CONFIG_DIR


def get_data_dir() -> Path:
    """Get the data directory."""
    return DATA_DIR


# =============================================================================
# FORMATTING UTILITIES
# =============================================================================

def format_pnl(value: float) -> str:
    """
    Format P&L value with sign and currency symbol.
    
    Args:
        value: P&L value in dollars
        
    Returns:
        Formatted string like "+$1,234" or "-$567"
        
    Examples:
        >>> format_pnl(1234.56)
        '+$1,235'
        >>> format_pnl(-567.89)
        '-$568'
        >>> format_pnl(0)
        '+$0'
    """
    if value >= 0:
        return f"+${value:,.0f}"
    return f"-${abs(value):,.0f}"


def format_pnl_with_color(value: float) -> tuple[str, str]:
    """
    Format P&L and return with appropriate color.
    
    Args:
        value: P&L value in dollars
        
    Returns:
        Tuple of (formatted_string, hex_color)
        
    Examples:
        >>> format_pnl_with_color(100)
        ('+$100', '#00D26A')
        >>> format_pnl_with_color(-50)
        ('-$50', '#FF4B4B')
    """
    from .constants import COLORS  # Avoid circular import
    color = COLORS["profit"] if value >= 0 else COLORS["loss"]
    formatted = format_pnl(value)
    return formatted, color


def format_currency(value: float, precision: int = 0, abbreviate: bool = False) -> str:
    """
    Format a currency value.
    
    Args:
        value: Value in dollars
        precision: Decimal places
        abbreviate: If True, use K/M/B suffixes
        
    Returns:
        Formatted currency string
        
    Examples:
        >>> format_currency(1234567, abbreviate=True)
        '$1.2M'
        >>> format_currency(1234.56, precision=2)
        '$1,234.56'
    """
    if abbreviate:
        abs_value = abs(value)
        sign = "-" if value < 0 else ""
        if abs_value >= 1_000_000_000:
            return f"{sign}${abs_value / 1_000_000_000:.1f}B"
        elif abs_value >= 1_000_000:
            return f"{sign}${abs_value / 1_000_000:.1f}M"
        elif abs_value >= 1_000:
            return f"{sign}${abs_value / 1_000:.1f}K"
    
    if precision > 0:
        return f"${value:,.{precision}f}"
    return f"${value:,.0f}"


def format_percentage(value: float, precision: int = 2, with_sign: bool = False) -> str:
    """
    Format a percentage value.
    
    Args:
        value: Percentage value (e.g., 5.5 for 5.5%)
        precision: Decimal places
        with_sign: Include +/- sign
        
    Returns:
        Formatted percentage string
        
    Examples:
        >>> format_percentage(5.5)
        '5.50%'
        >>> format_percentage(5.5, with_sign=True)
        '+5.50%'
    """
    if with_sign:
        return f"{value:+.{precision}f}%"
    return f"{value:.{precision}f}%"


def format_quantity(value: int, with_sign: bool = True) -> str:
    """
    Format a quantity/lot value.
    
    Args:
        value: Quantity (positive or negative)
        with_sign: Include +/- sign
        
    Returns:
        Formatted quantity string
        
    Examples:
        >>> format_quantity(10)
        '+10'
        >>> format_quantity(-5)
        '-5'
    """
    if with_sign:
        return f"{value:+d}"
    return f"{value:d}"


# =============================================================================
# DATA CONVERSION HELPERS
# =============================================================================

def safe_float(value, default: float = 0.0) -> float:
    """
    Safely convert a value to float.
    
    Args:
        value: Value to convert
        default: Default if conversion fails
        
    Returns:
        Float value or default
    """
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value, default: int = 0) -> int:
    """
    Safely convert a value to int.
    
    Args:
        value: Value to convert
        default: Default if conversion fails
        
    Returns:
        Int value or default
    """
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp a value to a range.
    
    Args:
        value: Value to clamp
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        Clamped value
        
    Examples:
        >>> clamp(150, 0, 100)
        100
        >>> clamp(-10, 0, 100)
        0
    """
    return max(min_val, min(value, max_val))


# =============================================================================
# HASH/SIGNATURE UTILITIES
# =============================================================================

def create_positions_signature(positions: list[dict]) -> tuple:
    """
    Create a hashable signature for a list of positions.
    
    Used for cache invalidation when positions change.
    
    Args:
        positions: List of position dictionaries
        
    Returns:
        Hashable tuple representing the positions state
    """
    sorted_positions = sorted(
        (
            pos.get("symbol", ""),
            pos.get("ticker", ""),
            safe_float(pos.get("qty", 0)),
            safe_float(pos.get("entry", 0)),
            pos.get("strategy", ""),
        )
        for pos in positions
    )
    return tuple(sorted_positions)
