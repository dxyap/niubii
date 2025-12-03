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
- indicators: Technical indicator calculations
"""

from . import constants
from . import indicators

__version__ = "1.0.0"
__all__ = ["constants", "indicators"]
