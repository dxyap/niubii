"""
Data Infrastructure Module
==========================
Handles data loading, caching, and Bloomberg API integration.

Performance optimizations:
- Batch API calls for multiple tickers
- Thread-safe TTL caching for real-time data
- Request deduplication to prevent redundant API calls
- Market-hours aware TTL for smart cache expiration
- Streamlit caching integration for expensive operations
"""

from .bloomberg import (
    BloombergClient,
    BloombergConnectionError,
    BloombergSubscriptionService,
    DataUnavailableError,
    TickerMapper,
)
from .cache import (
    DataCache,
    ParquetStorage,
    RequestDeduplicator,
    TTLCache,
    get_smart_ttl,
    is_market_hours,
    set_weekend_closed_days,
)
from .loader import DataLoader

__all__ = [
    "DataLoader",
    "DataCache",
    "TTLCache",
    "ParquetStorage",
    "RequestDeduplicator",
    "get_smart_ttl",
    "is_market_hours",
    "set_weekend_closed_days",
    "BloombergClient",
    "TickerMapper",
    "BloombergSubscriptionService",
    "DataUnavailableError",
    "BloombergConnectionError",
]
