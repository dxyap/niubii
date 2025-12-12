"""
Core Constants
==============
Centralized constants for the Oil Trading Dashboard.

This module provides a single source of truth for:
- Contract specifications
- Default configuration values
- Trading hours
- Risk parameters
"""

from typing import Final

# =============================================================================
# CONTRACT SPECIFICATIONS
# =============================================================================

# Contract multipliers (barrels or gallons per contract)
CONTRACT_MULTIPLIERS: Final[dict[str, int]] = {
    "CL": 1000,     # WTI Crude (NYMEX) - 1,000 barrels
    "ENA": 1000,    # WTI Crude (ICE) - 1,000 barrels
    "CO": 1000,     # Brent Crude (ICE) - 1,000 barrels
    "DAT": 1000,    # Dubai Crude Swap - 1,000 barrels
    "XB": 42000,    # RBOB Gasoline (NYMEX) - 42,000 gallons
    "HO": 42000,    # Heating Oil (NYMEX) - 42,000 gallons
    "QS": 100,      # Gasoil (ICE) - 100 metric tonnes
    "NG": 10000,    # Natural Gas (NYMEX) - 10,000 MMBtu
}

# Tick sizes (minimum price increment)
TICK_SIZES: Final[dict[str, float]] = {
    "CL": 0.01,
    "ENA": 0.01,
    "CO": 0.01,
    "DAT": 0.01,
    "XB": 0.0001,
    "HO": 0.0001,
    "QS": 0.25,
    "NG": 0.001,
}

# Tick values ($ value per tick)
TICK_VALUES: Final[dict[str, float]] = {
    "CL": 10.00,
    "ENA": 10.00,
    "CO": 10.00,
    "DAT": 10.00,
    "XB": 4.20,
    "HO": 4.20,
    "QS": 25.00,
    "NG": 10.00,
}

# Exchange mappings
EXCHANGES: Final[dict[str, str]] = {
    "CL": "NYMEX",
    "ENA": "ICE",
    "CO": "ICE",
    "DAT": "ICE",
    "XB": "NYMEX",
    "HO": "NYMEX",
    "QS": "ICE",
    "NG": "NYMEX",
}

# =============================================================================
# DEFAULT RISK PARAMETERS
# =============================================================================

DEFAULT_VAR_LIMIT: Final[float] = 375_000.0  # Maximum 1-day VaR (USD)
DEFAULT_GROSS_EXPOSURE_LIMIT: Final[float] = 20_000_000.0  # Maximum gross exposure
DEFAULT_NET_EXPOSURE_LIMIT: Final[float] = 15_000_000.0  # Maximum net exposure
DEFAULT_DAILY_DRAWDOWN_LIMIT: Final[float] = 0.05  # 5% daily drawdown

# VaR confidence levels
VAR_CONFIDENCE_95: Final[float] = 0.95
VAR_CONFIDENCE_99: Final[float] = 0.99

# Simplified VaR estimate factor (percentage of gross exposure)
SIMPLIFIED_VAR_FACTOR: Final[float] = 0.02  # 2%

# =============================================================================
# CACHE SETTINGS
# =============================================================================

CACHE_TTL_REALTIME: Final[int] = 15  # seconds
CACHE_TTL_INTRADAY: Final[int] = 60  # seconds
CACHE_TTL_HISTORICAL: Final[int] = 86400  # 24 hours
CACHE_TTL_REFERENCE: Final[int] = 604800  # 7 days

# =============================================================================
# BLOOMBERG FIELD MAPPINGS
# =============================================================================

BLOOMBERG_FIELDS: Final[dict[str, str]] = {
    "last": "PX_LAST",
    "bid": "PX_BID",
    "ask": "PX_ASK",
    "open": "PX_OPEN",
    "high": "PX_HIGH",
    "low": "PX_LOW",
    "close": "PX_LAST",
    "volume": "PX_VOLUME",
    "open_interest": "OPEN_INT",
    "vwap": "PX_VWAP",
    "settlement": "PX_SETTLE",
}

# =============================================================================
# MONTH CODES
# =============================================================================

MONTH_CODES: Final[dict[int, str]] = {
    1: 'F',   # January
    2: 'G',   # February
    3: 'H',   # March
    4: 'J',   # April
    5: 'K',   # May
    6: 'M',   # June
    7: 'N',   # July
    8: 'Q',   # August
    9: 'U',   # September
    10: 'V',  # October
    11: 'X',  # November
    12: 'Z',  # December
}

MONTH_CODES_REVERSE: Final[dict[str, int]] = {v: k for k, v in MONTH_CODES.items()}

# =============================================================================
# DASHBOARD SETTINGS
# =============================================================================

DEFAULT_REFRESH_INTERVAL: Final[int] = 10  # seconds
DEFAULT_LOOKBACK_DAYS: Final[int] = 90
MAX_POSITIONS_DISPLAY: Final[int] = 50

# =============================================================================
# ML MODEL SETTINGS
# =============================================================================

DEFAULT_PREDICTION_HORIZON: Final[int] = 5  # days
DEFAULT_TRAIN_TEST_SPLIT: Final[float] = 0.8
MIN_TRAINING_SAMPLES: Final[int] = 252  # ~1 year of daily data

# =============================================================================
# PRICE REFERENCE VALUES (for simulation)
# =============================================================================

REFERENCE_PRICES: Final[dict[str, float]] = {
    "CL": 72.50,    # WTI NYMEX
    "ENA": 72.55,   # WTI ICE
    "CO": 77.20,    # Brent
    "DAT": 76.80,   # Dubai
    "XB": 2.18,     # RBOB ($/gal)
    "HO": 2.52,     # Heating Oil ($/gal)
    "QS": 680.50,   # Gasoil ($/tonne)
    "NG": 3.25,     # Natural Gas
}

# Annualized volatility estimates (for VaR)
ANNUALIZED_VOLATILITY: Final[dict[str, float]] = {
    "CL": 0.25,     # 25% annual vol
    "ENA": 0.25,
    "CO": 0.25,
    "DAT": 0.25,
    "XB": 0.30,
    "HO": 0.30,
    "QS": 0.28,
    "NG": 0.40,
}

# =============================================================================
# UI COLORS (Single source of truth)
# =============================================================================

COLORS: Final[dict[str, str]] = {
    # Primary colors
    "primary": "#0ea5e9",
    "primary_light": "#38bdf8",
    "primary_dark": "#0284c7",

    # Secondary colors
    "secondary": "#8b5cf6",
    "secondary_light": "#a78bfa",

    # Semantic colors
    "success": "#00DC82",
    "warning": "#f59e0b",
    "error": "#FF5252",
    "info": "#00A3E0",

    # P&L specific colors
    "profit": "#00D26A",
    "loss": "#FF4B4B",
    "neutral": "#94a3b8",

    # Position colors
    "long": "#00DC82",
    "short": "#FF5252",
    "flat": "#64748b",

    # Candlestick colors (solid fills)
    "candle_up": "#00DC82",
    "candle_down": "#FF5252",

    # Text colors
    "text": "#e2e8f0",
    "text_muted": "#94a3b8",
    "text_bright": "#f8fafc",
    "text_dim": "#64748b",

    # Background colors
    "background": "#0f172a",
    "surface": "#1e293b",
    "surface_light": "#334155",
    "border": "#334155",
    "card": "rgba(30, 41, 59, 0.6)",

    # Chart colors
    "chart_bg": "rgba(15, 23, 42, 0.8)",
    "chart_grid": "rgba(51, 65, 85, 0.4)",

    # Moving average colors
    "ma_fast": "#FFB020",
    "ma_slow": "#A855F7",
    "ma_long": "#06B6D4",

    # Signal colors
    "signal_long": "#00DC82",
    "signal_short": "#FF5252",
    "signal_neutral": "#94a3b8",

    # Risk traffic light
    "risk_green": "#00DC82",
    "risk_yellow": "#f59e0b",
    "risk_red": "#ef4444",

    # Accent
    "gold": "#fbbf24",
    "cyan": "#06b6d4",
}
