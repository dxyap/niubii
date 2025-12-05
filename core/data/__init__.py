"""
Data Infrastructure Module
==========================
Handles data loading, caching, and Bloomberg API integration.

Performance optimizations:
- Batch API calls for multiple tickers
- Thread-safe TTL caching for real-time data
- Streamlit caching integration for expensive operations
"""

from .bloomberg import (
    BloombergClient,
    BloombergConnectionError,
    BloombergSubscriptionService,
    DataUnavailableError,
    MockBloombergData,
    PriceSimulator,
    TickerMapper,
)
from .cache import DataCache, ParquetStorage, TTLCache
from .loader import DataLoader

__all__ = [
    "DataLoader",
    "DataCache",
    "TTLCache",
    "ParquetStorage",
    "BloombergClient",
    "TickerMapper",
    "MockBloombergData",
    "BloombergSubscriptionService",
    "PriceSimulator",
    "DataUnavailableError",
    "BloombergConnectionError",
]
