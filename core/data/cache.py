"""
Data Caching Layer
==================
Multi-layer caching strategy to minimize API calls and improve performance.

Performance optimizations:
- LRU cache for frequently accessed data
- TTL-based expiration
- Thread-safe operations
- Efficient memory management
"""

import contextlib
import hashlib
import logging
import threading
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import diskcache
    HAS_DISKCACHE = True
except ImportError:
    HAS_DISKCACHE = False

logger = logging.getLogger(__name__)


def is_market_hours() -> bool:
    """
    Check if oil markets are currently open.
    
    Oil futures trade nearly 24 hours on weekdays:
    - CME/NYMEX: Sunday 5pm - Friday 4pm CT (with daily break 4pm-5pm CT)
    - ICE: Sunday 7pm - Friday 5pm ET
    
    Returns True during weekday trading hours, False on weekends.
    """
    now = datetime.now()
    # Weekend: Saturday (5) or Sunday (6)
    return now.weekday() < 5


def get_smart_ttl(base_ttl: int, market_hours_multiplier: float = 1.0, off_hours_multiplier: float = 10.0) -> int:
    """
    Return TTL adjusted for market hours.
    
    During market hours, use base TTL (data changes frequently).
    During off-hours/weekends, use extended TTL (data is static).
    
    Args:
        base_ttl: Base TTL in seconds
        market_hours_multiplier: Multiplier during market hours (default 1.0)
        off_hours_multiplier: Multiplier during off-hours (default 10.0)
    
    Returns:
        Adjusted TTL in seconds
    """
    if is_market_hours():
        return int(base_ttl * market_hours_multiplier)
    return int(base_ttl * off_hours_multiplier)


class RequestDeduplicator:
    """
    Coalesces identical Bloomberg API requests within a time window.
    
    When multiple callers request the same data simultaneously, only one
    Bloomberg API call is made and the result is shared. This prevents
    redundant API calls during dashboard refresh cycles.
    
    Thread-safe implementation using locks and condition variables.
    """
    
    def __init__(self, window_seconds: float = 0.5):
        """
        Initialize request deduplicator.
        
        Args:
            window_seconds: Time window for coalescing identical requests
        """
        self._window = window_seconds
        self._pending: dict[str, dict] = {}  # key -> {result, error, complete, waiters}
        self._lock = threading.Lock()
        self._stats = {"hits": 0, "misses": 0, "errors": 0}
    
    def execute(self, key: str, fetch_fn: Callable[[], Any]) -> Any:
        """
        Execute fetch function with deduplication.
        
        If another caller is already fetching the same key, wait for their
        result instead of making a duplicate API call.
        
        Args:
            key: Unique identifier for this request (e.g., ticker + fields)
            fetch_fn: Function to call if no pending request exists
            
        Returns:
            Result from fetch_fn (shared if deduplicated)
            
        Raises:
            Exception: Re-raises any exception from fetch_fn
        """
        with self._lock:
            # Check if there's a pending request for this key
            if key in self._pending:
                pending = self._pending[key]
                if not pending["complete"]:
                    # Wait for the pending request to complete
                    self._stats["hits"] += 1
                    condition = pending["condition"]
                    # Release lock while waiting
                    condition.wait(timeout=30.0)  # 30s timeout
                    
                # Return the cached result or re-raise error
                if pending.get("error"):
                    raise pending["error"]
                return pending["result"]
            
            # No pending request - we'll be the one to fetch
            self._stats["misses"] += 1
            condition = threading.Condition(self._lock)
            self._pending[key] = {
                "result": None,
                "error": None,
                "complete": False,
                "condition": condition,
                "timestamp": datetime.now().timestamp(),
            }
        
        # Execute fetch outside the lock
        try:
            result = fetch_fn()
            
            with self._lock:
                self._pending[key]["result"] = result
                self._pending[key]["complete"] = True
                self._pending[key]["condition"].notify_all()
            
            # Schedule cleanup after window expires
            self._schedule_cleanup(key)
            return result
            
        except Exception as e:
            with self._lock:
                self._pending[key]["error"] = e
                self._pending[key]["complete"] = True
                self._pending[key]["condition"].notify_all()
                self._stats["errors"] += 1
            
            self._schedule_cleanup(key)
            raise
    
    def _schedule_cleanup(self, key: str) -> None:
        """Remove pending entry after window expires."""
        def cleanup():
            import time
            time.sleep(self._window)
            with self._lock:
                if key in self._pending:
                    del self._pending[key]
        
        # Run cleanup in background thread
        cleanup_thread = threading.Thread(target=cleanup, daemon=True)
        cleanup_thread.start()
    
    def get_stats(self) -> dict:
        """Get deduplication statistics."""
        with self._lock:
            total = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total if total > 0 else 0
            return {
                **self._stats,
                "total_requests": total,
                "hit_rate": round(hit_rate, 3),
                "pending_count": len(self._pending),
            }
    
    def clear(self) -> None:
        """Clear all pending requests."""
        with self._lock:
            self._pending.clear()


class TTLCache:
    """
    Thread-safe in-memory cache with TTL expiration.
    Optimized for high-frequency price data access.
    """

    def __init__(self, max_size: int = 1000, default_ttl: float = 5.0):
        """
        Initialize TTL cache.

        Args:
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds
        """
        self._cache: dict[str, tuple] = {}  # key -> (value, expiry_time)
        self._lock = threading.Lock()
        self._max_size = max_size
        self._default_ttl = default_ttl

    def get(self, key: str, default: Any = None) -> Any:
        """Get value if not expired."""
        with self._lock:
            if key in self._cache:
                value, expiry = self._cache[key]
                if datetime.now().timestamp() < expiry:
                    return value
                else:
                    # Expired - remove
                    del self._cache[key]
            return default

    def set(self, key: str, value: Any, ttl: float = None) -> None:
        """Set value with TTL."""
        if ttl is None:
            ttl = self._default_ttl

        with self._lock:
            # Evict oldest entries if at capacity
            if len(self._cache) >= self._max_size:
                self._evict_expired()
                if len(self._cache) >= self._max_size:
                    # Remove 10% of oldest entries
                    to_remove = list(self._cache.keys())[:self._max_size // 10]
                    for k in to_remove:
                        del self._cache[k]

            expiry = datetime.now().timestamp() + ttl
            self._cache[key] = (value, expiry)

    def _evict_expired(self) -> None:
        """Remove expired entries."""
        now = datetime.now().timestamp()
        expired = [k for k, (_, exp) in self._cache.items() if now >= exp]
        for k in expired:
            del self._cache[k]

    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)


class DataCache:
    """
    Multi-layer caching system for market data.

    Caching Strategy:
    - Real-time prices: 60 seconds (in-memory TTL cache)
    - Intraday: 60 seconds (in-memory TTL cache)
    - Historical OHLCV: 24 hours (disk)
    - Reference data: 7 days (disk)
    - Fundamental data: Until next release (disk)

    Performance features:
    - Thread-safe TTL cache for high-frequency access
    - Efficient memory management
    - Automatic cache eviction
    """

    # Cache durations in seconds
    DURATIONS = {
        "real_time": 60,
        "intraday": 60,
        "historical": 86400,  # 24 hours
        "reference": 604800,  # 7 days
        "fundamental": 604800,  # 7 days
    }

    def __init__(self, cache_dir: str = "data/cache"):
        """
        Initialize cache.

        Args:
            cache_dir: Directory for disk cache
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Efficient TTL cache for real-time data
        self._real_time_cache = TTLCache(max_size=1000, default_ttl=60.0)
        self._intraday_cache = TTLCache(max_size=500, default_ttl=60.0)

        # Legacy memory cache (for backward compatibility)
        self._memory_cache: dict = {}
        self._memory_timestamps: dict = {}

        # Disk cache for persistent data
        if HAS_DISKCACHE:
            try:
                self._disk_cache = diskcache.Cache(str(self.cache_dir / "diskcache"))
            except Exception as e:
                self._disk_cache = None
                logger.debug(f"Could not initialize diskcache: {e}, using file-based caching")
        else:
            self._disk_cache = None
            logger.debug("diskcache not available, using file-based caching")

    def _get_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate unique cache key."""
        key_data = f"{prefix}:{args}:{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def get(
        self,
        key: str,
        cache_type: str = "historical",
        default: Any = None
    ) -> Any | None:
        """
        Get value from cache.

        Args:
            key: Cache key
            cache_type: Type of cache (determines expiry)
            default: Default value if not found

        Returns:
            Cached value or default
        """
        # Use efficient TTL caches for high-frequency data
        if cache_type == "real_time":
            result = self._real_time_cache.get(key)
            if result is not None:
                return result
        elif cache_type == "intraday":
            result = self._intraday_cache.get(key)
            if result is not None:
                return result

        duration = self.DURATIONS.get(cache_type, self.DURATIONS["historical"])

        # Legacy memory cache fallback
        if cache_type in ("real_time", "intraday") and key in self._memory_cache:
            timestamp = self._memory_timestamps.get(key, 0)
            if datetime.now().timestamp() - timestamp < duration:
                return self._memory_cache[key]

        # Check disk cache
        if self._disk_cache is not None:
            try:
                value = self._disk_cache.get(key, default=None)
                if value is not None:
                    return value
            except Exception as e:
                logger.warning(f"Disk cache read error: {e}")

        # Fall back to file cache
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if datetime.now() - mtime < timedelta(seconds=duration):
                    return pd.read_pickle(cache_file)
            except Exception as e:
                logger.warning(f"File cache read error: {e}")

        return default

    def set(
        self,
        key: str,
        value: Any,
        cache_type: str = "historical"
    ) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            cache_type: Type of cache
        """
        duration = self.DURATIONS.get(cache_type, self.DURATIONS["historical"])

        # Use efficient TTL caches for high-frequency data
        if cache_type == "real_time":
            self._real_time_cache.set(key, value, ttl=duration)
            self._remember_for_fallback(key, value)
            return
        elif cache_type == "intraday":
            self._intraday_cache.set(key, value, ttl=duration)
            self._remember_for_fallback(key, value)
            return

        # Disk cache
        if self._disk_cache is not None:
            try:
                self._disk_cache.set(key, value, expire=duration)
                return
            except Exception as e:
                logger.warning(f"Disk cache write error: {e}")

        # Fall back to file cache
        try:
            cache_file = self.cache_dir / f"{key}.pkl"
            if isinstance(value, pd.DataFrame):
                value.to_pickle(cache_file)
            else:
                pd.to_pickle(value, cache_file)
        except Exception as e:
            logger.warning(f"File cache write error: {e}")

    def _remember_for_fallback(self, key: str, value: Any) -> None:
        """Store a copy in the legacy memory cache for TTL fallbacks."""
        self._memory_cache[key] = value
        self._memory_timestamps[key] = datetime.now().timestamp()

    def cached(
        self,
        cache_type: str = "historical",
        key_prefix: str = ""
    ) -> Callable:
        """
        Decorator for caching function results.

        Args:
            cache_type: Type of cache
            key_prefix: Prefix for cache key

        Returns:
            Decorated function
        """
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                key = self._get_cache_key(
                    key_prefix or func.__name__,
                    *args,
                    **kwargs
                )

                # Try cache first
                cached_value = self.get(key, cache_type)
                if cached_value is not None:
                    return cached_value

                # Call function and cache result
                result = func(*args, **kwargs)
                self.set(key, result, cache_type)
                return result

            return wrapper
        return decorator

    def clear(self, cache_type: str | None = None) -> None:
        """
        Clear cache.

        Args:
            cache_type: Type of cache to clear (None = all)
        """
        if cache_type == "real_time" or cache_type is None:
            self._real_time_cache.clear()

        if cache_type == "intraday" or cache_type is None:
            self._intraday_cache.clear()

        if cache_type in ("real_time", "intraday", None):
            self._memory_cache.clear()
            self._memory_timestamps.clear()

        if cache_type not in ("real_time", "intraday"):
            if self._disk_cache is not None:
                self._disk_cache.clear()

            # Clear file cache
            for f in self.cache_dir.glob("*.pkl"):
                with contextlib.suppress(Exception):
                    f.unlink()

    def get_stats(self) -> dict:
        """Get cache statistics."""
        stats = {
            "real_time_entries": len(self._real_time_cache),
            "intraday_entries": len(self._intraday_cache),
            "memory_entries": len(self._memory_cache),
            "disk_entries": 0,
            "file_entries": len(list(self.cache_dir.glob("*.pkl"))),
        }

        if self._disk_cache is not None:
            with contextlib.suppress(Exception):
                stats["disk_entries"] = len(self._disk_cache)

        return stats


class ParquetStorage:
    """
    Parquet-based storage for historical data.

    Optimized for:
    - Fast columnar queries
    - Efficient compression
    - Easy Snowflake migration
    """

    def __init__(self, base_dir: str = "data/historical"):
        """Initialize Parquet storage."""
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_ohlcv(
        self,
        ticker: str,
        data: pd.DataFrame,
        frequency: str = "daily"
    ) -> None:
        """
        Save OHLCV data to Parquet.

        Args:
            ticker: Instrument ticker
            data: OHLCV DataFrame
            frequency: Data frequency
        """
        ohlcv_dir = self.base_dir / "ohlcv"
        ohlcv_dir.mkdir(exist_ok=True)

        # Clean ticker for filename
        clean_ticker = ticker.replace(" ", "_").replace("/", "_")
        filepath = ohlcv_dir / f"{clean_ticker}_{frequency}.parquet"

        data.to_parquet(filepath, engine="pyarrow", compression="snappy")

    def load_ohlcv(
        self,
        ticker: str,
        frequency: str = "daily",
        start_date: datetime | None = None,
        end_date: datetime | None = None
    ) -> pd.DataFrame | None:
        """
        Load OHLCV data from Parquet.

        Args:
            ticker: Instrument ticker
            frequency: Data frequency
            start_date: Filter start date
            end_date: Filter end date

        Returns:
            OHLCV DataFrame or None
        """
        clean_ticker = ticker.replace(" ", "_").replace("/", "_")
        filepath = self.base_dir / "ohlcv" / f"{clean_ticker}_{frequency}.parquet"

        if not filepath.exists():
            return None

        df = pd.read_parquet(filepath)
        # Normalize index to pandas datetime for safe comparisons
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            df.reset_index(inplace=True)
            df.rename(columns={"index": "date"}, inplace=True)
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)

        # Apply date filters
        if start_date is not None:
            if not isinstance(start_date, pd.Timestamp):
                start_date = pd.Timestamp(start_date)
            df = df[df.index >= start_date]
        if end_date is not None:
            if not isinstance(end_date, pd.Timestamp):
                end_date = pd.Timestamp(end_date)
            df = df[df.index <= end_date]

        return df

    def save_curve(self, date: datetime, commodity: str, data: pd.DataFrame) -> None:
        """Save curve snapshot to Parquet."""
        curves_dir = self.base_dir / "curves"
        curves_dir.mkdir(exist_ok=True)

        date_str = date.strftime("%Y%m%d")
        filepath = curves_dir / f"{commodity}_curve_{date_str}.parquet"
        data.to_parquet(filepath, engine="pyarrow", compression="snappy")

    def load_curve(
        self,
        date: datetime,
        commodity: str
    ) -> pd.DataFrame | None:
        """Load curve snapshot from Parquet."""
        date_str = date.strftime("%Y%m%d")
        filepath = self.base_dir / "curves" / f"{commodity}_curve_{date_str}.parquet"

        if not filepath.exists():
            return None

        return pd.read_parquet(filepath)

    def save_fundamentals(self, data_type: str, data: pd.DataFrame) -> None:
        """Save fundamental data to Parquet."""
        fund_dir = self.base_dir / "fundamentals"
        fund_dir.mkdir(exist_ok=True)

        filepath = fund_dir / f"{data_type}.parquet"
        data.to_parquet(filepath, engine="pyarrow", compression="snappy")

    def load_fundamentals(self, data_type: str) -> pd.DataFrame | None:
        """Load fundamental data from Parquet."""
        filepath = self.base_dir / "fundamentals" / f"{data_type}.parquet"

        if not filepath.exists():
            return None

        return pd.read_parquet(filepath)

    def list_available_data(self) -> dict:
        """List all available data files."""
        available = {
            "ohlcv": [],
            "curves": [],
            "fundamentals": [],
        }

        ohlcv_dir = self.base_dir / "ohlcv"
        if ohlcv_dir.exists():
            available["ohlcv"] = [f.stem for f in ohlcv_dir.glob("*.parquet")]

        curves_dir = self.base_dir / "curves"
        if curves_dir.exists():
            available["curves"] = [f.stem for f in curves_dir.glob("*.parquet")]

        fund_dir = self.base_dir / "fundamentals"
        if fund_dir.exists():
            available["fundamentals"] = [f.stem for f in fund_dir.glob("*.parquet")]

        return available
