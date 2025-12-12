"""
Oil Trading Dashboard Core Module
=================================
Core business logic for the quantitative oil trading dashboard.

Submodules:
- data: Bloomberg integration, caching, and data loading
- analytics: Curve analysis, spreads, and fundamentals
- signals: Technical, fundamental, and ML signal generation
- risk: VaR, limits, and risk monitoring
- trading: Position management and P&L tracking
- ml: Machine learning models and feature engineering
- backtest: Strategy backtesting and optimization
- execution: Order management and execution algorithms
- alerts: Multi-channel alert system and notifications
- research: Advanced analytics, LLM, and alternative data
- infrastructure: Authentication, RBAC, audit logging, and monitoring
- indicators: Technical indicator calculations
- utils: Common utility functions (formatting, paths, etc.)
"""

from . import constants, indicators, utils
from .constants import COLORS
from .utils import (
    PROJECT_ROOT,
    CONFIG_DIR,
    DATA_DIR,
    format_pnl,
    format_pnl_with_color,
    format_currency,
    format_percentage,
    create_positions_signature,
)

__version__ = "3.0.0"  # Phase 9 complete - all phases implemented
__all__ = [
    "constants",
    "indicators",
    "utils",
    "COLORS",
    "PROJECT_ROOT",
    "CONFIG_DIR",
    "DATA_DIR",
    "format_pnl",
    "format_pnl_with_color",
    "format_currency",
    "format_percentage",
    "create_positions_signature",
]
