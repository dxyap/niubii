"""
Data Infrastructure Module
==========================
Handles data loading, caching, and Bloomberg API integration.

Performance optimizations:
- Batch API calls for multiple tickers
- Thread-safe TTL caching for real-time data
- Streamlit caching integration for expensive operations
"""

from .loader import DataLoader
from .cache import DataCache, TTLCache, ParquetStorage
from .bloomberg import (
    BloombergClient, 
    TickerMapper, 
    MockBloombergData,
    BloombergSubscriptionService,
    PriceSimulator,
    DataUnavailableError,
    BloombergConnectionError,
)

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
