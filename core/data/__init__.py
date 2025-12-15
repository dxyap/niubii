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

from __future__ import annotations

from importlib import import_module
from typing import Any

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

_LAZY_ATTR_MODULES = {
    "DataLoader": "core.data.loader",
    "DataCache": "core.data.cache",
    "TTLCache": "core.data.cache",
    "ParquetStorage": "core.data.cache",
    "RequestDeduplicator": "core.data.cache",
    "get_smart_ttl": "core.data.cache",
    "is_market_hours": "core.data.cache",
    "set_weekend_closed_days": "core.data.cache",
    "BloombergClient": "core.data.bloomberg",
    "TickerMapper": "core.data.bloomberg",
    "BloombergSubscriptionService": "core.data.bloomberg",
    "DataUnavailableError": "core.data.bloomberg",
    "BloombergConnectionError": "core.data.bloomberg",
}


def __getattr__(name: str) -> Any:
    """
    Lazily import data submodules on first attribute access.

    Prevents circular import deadlocks during concurrent Streamlit reruns
    by avoiding eager imports of heavy submodules at package import time.
    """
    module_path = _LAZY_ATTR_MODULES.get(name)
    if module_path is None:
        raise AttributeError(f"module 'core.data' has no attribute '{name}'")

    module = import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Provide better auto-complete support."""
    return sorted(set(globals()) | set(__all__))
