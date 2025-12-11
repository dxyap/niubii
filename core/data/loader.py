"""
Data Loading Utilities
======================
High-level data loading with caching and Bloomberg integration.
"""

import contextlib
import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .bloomberg import (
    BloombergClient,
    BloombergSubscriptionService,
    TickerMapper,
)
from .cache import DataCache, ParquetStorage, RequestDeduplicator, get_smart_ttl

logger = logging.getLogger(__name__)


def _require_mapping(value: Mapping | None, path: str) -> dict:
    """Ensure a mapping exists in configuration."""
    if isinstance(value, Mapping) and value:
        return dict(value)
    raise ValueError(f"Missing configuration for {path} in data_loader.yaml")


def _require_sequence(value: Sequence | None, path: str) -> list:
    """Ensure a sequence exists in configuration."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) > 0:
        return list(value)
    raise ValueError(f"Missing configuration for {path} in data_loader.yaml")


@dataclass(frozen=True)
class CrackSpreadConfig:
    """Configuration and calculator for crack spread ratios."""

    gasoline_ratio: float
    heating_oil_ratio: float
    crude_ratio: float
    divisor: float | None = None

    @classmethod
    def from_mapping(cls, data: Mapping | None, path: str) -> "CrackSpreadConfig":
        mapping = _require_mapping(data, path)
        divisor = mapping.get("divisor")
        divisor_value = float(divisor) if divisor not in (None, 0) else None
        return cls(
            gasoline_ratio=float(mapping.get("gasoline_ratio", 0)),
            heating_oil_ratio=float(mapping.get("heating_oil_ratio", 0)),
            crude_ratio=float(mapping.get("crude_ratio", 0)),
            divisor=divisor_value,
        )

    def calculate(self, gasoline_value: float, heating_oil_value: float, crude_value: float) -> float:
        """Calculate crack spread with configured ratios."""
        numerator = self.gasoline_ratio * gasoline_value + self.heating_oil_ratio * heating_oil_value
        numerator -= self.crude_ratio * crude_value
        if self.divisor:
            return numerator / self.divisor
        return numerator


@dataclass(frozen=True)
class LoaderSettings:
    """Immutable loader configuration with helper utilities."""

    gallons_per_barrel: int
    cache_base_ttl: int
    dedupe_window_seconds: float
    oil_price_tickers: dict[str, str]
    all_oil_price_tickers: dict[str, str]
    spread_batch_tickers: tuple[str, ...]
    core_subscription_tickers: tuple[str, ...]
    market_status_tickers: dict[str, str]
    crack_321: CrackSpreadConfig
    crack_211: CrackSpreadConfig
    curve_spread_offsets: dict[str, int]
    structure_threshold: float
    curve_months: int
    weekend_closed_days: frozenset[int]

    @classmethod
    def load(cls, config_dir: Path, gallons_default: int) -> "LoaderSettings":
        """Load settings from data_loader.yaml."""
        config_file = config_dir / "data_loader.yaml"
        if not config_file.exists():
            logger.warning("Data loader config not found at %s", config_file)
            raw: Mapping[str, Any] = {}
        else:
            with open(config_file) as f:
                raw = yaml.safe_load(f) or {}

        if not isinstance(raw, Mapping):
            raise ValueError("data_loader.yaml must define a mapping at the top level")

        return cls.from_mapping(raw, gallons_default)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], gallons_default: int) -> "LoaderSettings":
        """Build LoaderSettings from mapping data."""
        pricing_cfg = raw.get("pricing", {}) or {}
        caching_cfg = raw.get("caching", {}) or {}
        dedupe_cfg = raw.get("deduplicator", {}) or {}
        tickers_cfg = raw.get("tickers", {}) or {}
        spreads_cfg = raw.get("spreads", {}) or {}
        market_cfg = raw.get("market", {}) or {}
        market_hours_cfg = raw.get("market_hours", {}) or {}

        oil_price_tickers = _require_mapping(tickers_cfg.get("oil_prices"), "tickers.oil_prices")
        all_oil_price_tickers = _require_mapping(tickers_cfg.get("all_oil_prices"), "tickers.all_oil_prices")
        spread_batch_tickers = tuple(
            _require_sequence(tickers_cfg.get("spread_batch"), "tickers.spread_batch")
        )
        core_subscription_tickers = tuple(
            _require_sequence(tickers_cfg.get("core_subscriptions"), "tickers.core_subscriptions")
        )
        market_status_tickers = _require_mapping(tickers_cfg.get("market_status"), "tickers.market_status")

        crack_321 = CrackSpreadConfig.from_mapping(spreads_cfg.get("crack_321"), "spreads.crack_321")
        crack_211 = CrackSpreadConfig.from_mapping(spreads_cfg.get("crack_211"), "spreads.crack_211")

        curve_offsets_cfg = market_cfg.get("curve_spread_offsets") or {}
        curve_spread_offsets = {str(key): int(value) for key, value in curve_offsets_cfg.items()}
        if not curve_spread_offsets:
            curve_spread_offsets = {"m1_m2": 1, "m1_m6": 5, "m1_m12": 11}

        weekend_days = market_hours_cfg.get("weekend_closed_days", [5, 6])
        weekend_closed_days = frozenset(int(day) for day in weekend_days)

        return cls(
            gallons_per_barrel=int(pricing_cfg.get("gallons_per_barrel", gallons_default)),
            cache_base_ttl=int(caching_cfg.get("real_time_cache_base_ttl", 60)),
            dedupe_window_seconds=float(dedupe_cfg.get("window_seconds", 1.0)),
            oil_price_tickers=oil_price_tickers,
            all_oil_price_tickers=all_oil_price_tickers,
            spread_batch_tickers=spread_batch_tickers,
            core_subscription_tickers=core_subscription_tickers,
            market_status_tickers=market_status_tickers,
            crack_321=crack_321,
            crack_211=crack_211,
            curve_spread_offsets=curve_spread_offsets,
            structure_threshold=float(market_cfg.get("structure_threshold", 0.05)),
            curve_months=int(market_cfg.get("default_curve_months", 12)),
            weekend_closed_days=weekend_closed_days,
        )

    def is_market_open(self, when: datetime | None = None) -> bool:
        """Check market hours based on configured closed days."""
        reference = when or datetime.now()
        return reference.weekday() not in self.weekend_closed_days


class DataLoader:
    """
    Unified data loading interface.

    Handles:
    - Bloomberg API integration (live data)
    - Caching layer
    - Parquet storage for historical data
    - Ticker validation and mapping

    Raises DataUnavailableError when data cannot be retrieved.

    Unit Reference for Oil Products:
    - Crude oil (CL1, CO1): Quoted in $/barrel
    - RBOB Gasoline (XB1): Quoted in $/gallon (42 gallons per barrel)
    - Heating Oil (HO1): Quoted in $/gallon (42 gallons per barrel)
    - Gasoil (QS1): Quoted in $/metric tonne (~7.45 barrels per tonne)
    """

    # Standard conversion factor: gallons per barrel (overridden by config)
    GALLONS_PER_BARREL = 42

    def __init__(
        self,
        config_dir: str = "config",
        data_dir: str = "data",
    ):
        """
        Initialize data loader.

        Args:
            config_dir: Configuration directory
            data_dir: Data storage directory
        """
        self.config_dir = Path(config_dir)
        self.data_dir = Path(data_dir)

        # Load loader-specific configuration before initializing components
        self.settings = LoaderSettings.load(self.config_dir, self.GALLONS_PER_BARREL)
        # Backward compatibility: expose previous attribute name
        self.loader_settings = self.settings

        # Initialize components
        self.bloomberg = BloombergClient()
        self.cache = DataCache(cache_dir=str(self.data_dir / "cache"))
        self.storage = ParquetStorage(base_dir=str(self.data_dir / "historical"))

        # Request deduplicator to prevent redundant Bloomberg API calls
        # within a short time window (e.g., during dashboard refresh)
        self._deduplicator = RequestDeduplicator(window_seconds=self.settings.dedupe_window_seconds)

        # Initialize subscription service for real-time updates
        self.subscription_service = BloombergSubscriptionService(self.bloomberg)

        # Load configurations
        self._load_config()

        # Determine connection status
        if self.bloomberg.connected:
            self._data_mode = "live"
            self._connection_error = None
            logger.info("DataLoader initialized in LIVE mode (Bloomberg connected)")
        else:
            self._data_mode = "disconnected"
            self._connection_error = self.bloomberg.get_connection_error()
            logger.error(
                "DataLoader: Bloomberg not connected - %s",
                self._connection_error,
            )
            logger.error("Live Bloomberg connection required.")

    def _load_config(self):
        """Load configuration files."""
        self.instruments = {}
        self.tickers = {}

        instruments_file = self.config_dir / "instruments.yaml"
        if instruments_file.exists():
            with open(instruments_file) as f:
                self.instruments = yaml.safe_load(f)

        tickers_file = self.config_dir / "bloomberg_tickers.yaml"
        if tickers_file.exists():
            with open(tickers_file) as f:
                self.tickers = yaml.safe_load(f)

    # =========================================================================
    # TICKER UTILITIES
    # =========================================================================

    def validate_ticker(self, ticker: str) -> tuple:
        """Validate a Bloomberg ticker."""
        return TickerMapper.validate_ticker(ticker)

    def get_ticker(self, commodity: str, month: int = 1) -> str:
        """Get Bloomberg ticker for a commodity."""
        return TickerMapper.get_nth_month_ticker(commodity, month)

    def get_multiplier(self, ticker: str) -> int:
        """Get contract multiplier for a ticker."""
        return TickerMapper.get_multiplier(ticker)

    def parse_ticker(self, ticker: str) -> dict:
        """Parse ticker into components."""
        return TickerMapper.parse_ticker(ticker)

    # =========================================================================
    # REAL-TIME DATA METHODS
    # =========================================================================

    def get_price(self, ticker: str, validate: bool = True) -> float:
        """
        Get current price for ticker.
        
        Uses request deduplication to prevent redundant API calls within
        a short time window (useful during dashboard refresh cycles).
        """
        if validate:
            valid, msg = self.validate_ticker(ticker)
            if not valid:
                logger.warning(f"Invalid ticker {ticker}: {msg}")

        # Use deduplicator to coalesce identical requests
        dedupe_key = f"price:{ticker}"
        return self._deduplicator.execute(
            dedupe_key,
            lambda: self.bloomberg.get_price(ticker)
        )

    def get_price_with_change(self, ticker: str) -> dict[str, float]:
        """
        Get current price with change from open.
        
        Uses request deduplication to prevent redundant API calls.
        """
        dedupe_key = f"price_change:{ticker}"
        return self._deduplicator.execute(
            dedupe_key,
            lambda: self.bloomberg.get_price_with_change(ticker)
        )

    def get_prices(self, tickers: list[str]) -> pd.DataFrame:
        """
        Get current prices for multiple tickers.
        
        Uses request deduplication - the same batch request within the
        deduplication window will return the cached result.
        """
        # Create a stable key from sorted tickers
        dedupe_key = f"prices:{','.join(sorted(tickers))}"
        return self._deduplicator.execute(
            dedupe_key,
            lambda: self.bloomberg.get_prices(tickers)
        )

    def get_prices_batch(self, tickers: list[str]) -> dict[str, dict[str, float]]:
        """
        Get prices with changes for multiple tickers in a single batch call.
        Much more efficient than calling get_price_with_change() for each ticker.

        Uses request deduplication to prevent redundant API calls.

        Returns:
            Dict mapping ticker to price data dict
        """
        # Create a stable key from sorted tickers
        dedupe_key = f"prices_batch:{','.join(sorted(tickers))}"
        
        def fetch_batch():
            fields = ["PX_LAST", "PX_OPEN", "PX_HIGH", "PX_LOW"]
            df = self.bloomberg.get_prices(tickers, fields)

            result = {}
            for ticker in tickers:
                if ticker in df.index:
                    row = df.loc[ticker]
                    current = row.get("PX_LAST", 0)
                    open_price = row.get("PX_OPEN", current)
                    change = current - open_price if current and open_price else 0
                    change_pct = (change / open_price * 100) if open_price else 0

                    result[ticker] = {
                        "current": current,
                        "open": open_price,
                        "change": round(change, 4),
                        "change_pct": round(change_pct, 4),
                        "high": row.get("PX_HIGH", current),
                        "low": row.get("PX_LOW", current),
                    }

            return result
        
        return self._deduplicator.execute(dedupe_key, fetch_batch)

    def _get_price_batch_cached(self, cache_key: str, tickers: Sequence[str]) -> dict[str, dict[str, float]]:
        """
        Fetch a batch of prices with a short-lived real-time cache to avoid
        duplicate Bloomberg calls during a single refresh cycle.
        
        Uses market-hours aware TTL for smarter cache expiration.
        """
        cached = self.cache.get(cache_key, cache_type="real_time")
        if cached is not None:
            return cached

        batch = self.get_prices_batch(list(tickers))
        if batch:
            # Use smart TTL - longer during off-hours when prices don't change
            ttl = get_smart_ttl(self.settings.cache_base_ttl)  # Base TTL configurable
            self.cache.set(cache_key, batch, cache_type="real_time")
        return batch

    def _build_named_price_map(
        self,
        cache_key: str,
        ticker_map: Mapping[str, str],
    ) -> dict[str, dict[str, float]]:
        """Return price payload keyed by friendly names."""
        batch_prices = self._get_price_batch_cached(cache_key, list(ticker_map.values()))

        prices: dict[str, dict[str, float]] = {}
        for name, ticker in ticker_map.items():
            if ticker in batch_prices:
                prices[name] = batch_prices[ticker]
        return prices

    def get_oil_prices(self) -> dict[str, dict[str, float]]:
        """Get current oil prices for key benchmarks with changes (batch optimized)."""
        return self._build_named_price_map("oil_prices", self.settings.oil_price_tickers)

    def get_all_oil_prices(self) -> dict[str, dict[str, float]]:
        """Get prices for all tracked oil products (batch optimized)."""
        return self._build_named_price_map("all_oil_prices", self.settings.all_oil_price_tickers)

    def get_intraday_prices(self, ticker: str) -> pd.DataFrame:
        """Get intraday price history for charting."""
        return self.bloomberg.get_intraday_prices(ticker)

    # =========================================================================
    # HISTORICAL DATA METHODS
    # =========================================================================

    def get_historical(
        self,
        ticker: str,
        start_date: str | datetime = None,
        end_date: str | datetime = None,
        frequency: str = "daily"
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data.

        First checks local storage, then fetches from Bloomberg if needed.
        Ensures data includes recent dates (today/yesterday) when applicable.
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
        if end_date is None:
            end_date = datetime.now()

        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        now_ts = pd.Timestamp.now()
        today = now_ts.normalize()

        frequency_lower = frequency.lower()
        stored_df = None

        # Try loading from storage first (if parquet/pyarrow is available)
        try:
            if frequency_lower == "daily":
                stored_df = self.storage.load_ohlcv(ticker, frequency)
            else:
                stored_df = self.storage.load_ohlcv(ticker, frequency, start_date, end_date)
        except ImportError as e:
            # pyarrow not installed - skip storage, fetch directly from Bloomberg
            logger.debug(f"Parquet storage unavailable ({e}), fetching from Bloomberg")
        except Exception as e:
            logger.debug(f"Storage load failed for {ticker}: {e}")

        df = self._ensure_datetime_index(stored_df)

        # Incremental refresh for daily data to minimise Bloomberg calls
        if df is not None and len(df) > 0 and frequency_lower == "daily":
            df = df.sort_index()
            last_stored_date = pd.Timestamp(df.index[-1]).normalize()
            target_end = pd.Timestamp(end_date).normalize()
            needs_refresh = last_stored_date < target_end

            is_business_day = today.dayofweek < 5  # Mon=0, Fri=4
            if not needs_refresh and is_business_day and target_end >= today and last_stored_date < today:
                needs_refresh = True

            if needs_refresh:
                fetch_start = (last_stored_date - pd.Timedelta(days=3)).normalize()
                fetch_end = end_date if end_date >= now_ts else now_ts
                incremental_df = self.bloomberg.get_historical(
                    ticker,
                    fetch_start,
                    fetch_end,
                    frequency=frequency.upper()
                )
                incremental_df = self._ensure_datetime_index(incremental_df)

                if incremental_df is not None and len(incremental_df) > 0:
                    combined = pd.concat(
                        [df[df.index < fetch_start], incremental_df]
                    ).sort_index()
                    combined = combined[~combined.index.duplicated(keep="last")]

                    try:
                        self.storage.save_ohlcv(ticker, combined, frequency)
                    except ImportError:
                        pass  # pyarrow not installed - skip saving
                    except Exception as e:
                        logger.debug(f"Could not save to storage: {e}")

                    df = combined
                else:
                    logger.debug(f"No incremental data returned for {ticker} between {fetch_start} and {fetch_end}")

        if df is not None and len(df) > 0:
            filtered = df
            if start_date is not None:
                filtered = filtered[filtered.index >= start_date]
            if end_date is not None:
                filtered = filtered[filtered.index <= end_date]
            if len(filtered) > 0:
                return filtered

        # Fetch from Bloomberg when cache/storage missing or empty
        df = self.bloomberg.get_historical(
            ticker,
            start_date,
            end_date,
            frequency=frequency.upper()
        )
        df = self._ensure_datetime_index(df)

        # Save to storage for future use (if parquet/pyarrow is available)
        if df is not None and len(df) > 0:
            try:
                self.storage.save_ohlcv(ticker, df, frequency)
            except ImportError:
                pass  # pyarrow not installed - skip saving
            except Exception as e:
                logger.debug(f"Could not save to storage: {e}")

        return df

    def get_historical_multi(
        self,
        tickers: list[str],
        start_date: str | datetime = None,
        end_date: str | datetime = None,
        frequency: str = "daily"
    ) -> dict[str, pd.DataFrame]:
        """Get historical data for multiple tickers."""
        results = {}
        for ticker in tickers:
            try:
                results[ticker] = self.get_historical(ticker, start_date, end_date, frequency)
            except Exception as e:
                logger.warning(f"Could not get historical for {ticker}: {e}")
        return results

    # =========================================================================
    # CURVE DATA METHODS
    # =========================================================================

    def get_futures_curve(
        self,
        commodity: str = "wti",
        num_months: int | None = None
    ) -> pd.DataFrame:
        """
        Get futures curve data.
        
        Uses request deduplication to prevent redundant API calls.

        Args:
            commodity: 'wti', 'brent', etc.
            num_months: Number of months on curve

        Returns:
            DataFrame with curve data including changes
        """
        target_months = num_months if num_months is not None else self.settings.curve_months
        dedupe_key = f"curve:{commodity}:{target_months}"
        return self._deduplicator.execute(
            dedupe_key,
            lambda: self.bloomberg.get_curve(commodity, target_months)
        )

    def get_calendar_spreads(self, commodity: str = "wti") -> pd.DataFrame:
        """Calculate calendar spreads from curve."""
        curve = self.get_futures_curve(commodity, num_months=self.settings.curve_months)

        spreads = []
        for i in range(len(curve) - 1):
            spread = {
                "spread": f"M{i+1}-M{i+2}",
                "front_month": curve.iloc[i]["ticker"],
                "back_month": curve.iloc[i+1]["ticker"],
                "spread_value": curve.iloc[i]["price"] - curve.iloc[i+1]["price"],
                "front_price": curve.iloc[i]["price"],
                "back_price": curve.iloc[i+1]["price"],
            }
            spreads.append(spread)

        return pd.DataFrame(spreads)

    def get_term_structure(self, commodity: str = "wti") -> dict:
        """Get term structure analysis."""
        curve = self.get_futures_curve(commodity, num_months=self.settings.curve_months)
        curve_length = len(curve)

        if curve_length < 2:
            return {"structure": "Unknown", "slope": 0}

        # Calculate overall slope
        slope = (curve.iloc[-1]["price"] - curve.iloc[0]["price"]) / (curve_length - 1)

        # Determine structure
        if slope > self.settings.structure_threshold:
            structure = "Contango"
        elif slope < -self.settings.structure_threshold:
            structure = "Backwardation"
        else:
            structure = "Flat"

        spreads: dict[str, float] = {}
        for label, offset in self.settings.curve_spread_offsets.items():
            if curve_length > offset:
                spreads[f"{label}_spread"] = round(curve.iloc[0]["price"] - curve.iloc[offset]["price"], 2)
            else:
                spreads[f"{label}_spread"] = 0.0

        result = {
            "structure": structure,
            "slope": round(slope, 4),
            "curve_data": curve,
        }
        result.update(spreads)
        return result

    # =========================================================================
    # FUNDAMENTAL DATA METHODS
    # =========================================================================

    def get_eia_inventory(self) -> pd.DataFrame | None:
        """
        Get EIA crude oil inventory data.

        Returns:
            DataFrame with inventory data, or None if unavailable.

        Note: EIA data requires Bloomberg ECST <GO> or external data source.
        """
        cache_key = "eia_inventory"
        cached = self.cache.get(cache_key, cache_type="fundamental")

        if cached is not None:
            return cached

        # Try to load from storage
        stored = self.storage.load_fundamentals("eia_inventory")
        if stored is not None and not stored.empty:
            return stored

        # No data available - return None instead of mock data
        return None

    def get_opec_production(self) -> pd.DataFrame | None:
        """
        Get OPEC production and compliance data.

        Returns:
            DataFrame with OPEC data, or None if unavailable.

        Note: OPEC data requires Bloomberg or external data source.
        """
        # Try to load from storage
        stored = self.storage.load_fundamentals("opec_production")
        if stored is not None and not stored.empty:
            return stored

        # No data available
        return None

    def get_refinery_turnarounds(self) -> pd.DataFrame | None:
        """
        Get refinery turnaround schedule.

        Returns:
            DataFrame with turnaround data, or None if unavailable.

        Note: Turnaround data requires Bloomberg or external data source.
        """
        # Try to load from storage
        stored = self.storage.load_fundamentals("refinery_turnarounds")
        if stored is not None and not stored.empty:
            return stored

        # No data available
        return None

    # =========================================================================
    # SPREAD CALCULATIONS (Optimized - single batch fetch)
    # =========================================================================

    def _get_spread_price_batch(self) -> dict[str, dict[str, float]]:
        """
        Fetch the core spread tickers in one call with a short-lived cache to
        avoid redundant Bloomberg round trips within a refresh cycle.
        """
        return self._get_price_batch_cached("core_spread_prices", list(self.settings.spread_batch_tickers))

    def _build_spread_payloads(self, batch: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        """Compute the various spreads from a shared batch payload."""
        wti_data = batch.get("CL1 Comdty", {})
        brent_data = batch.get("CO1 Comdty", {})
        rbob_data = batch.get("XB1 Comdty", {})
        ho_data = batch.get("HO1 Comdty", {})

        # WTI/Brent values
        wti_current = wti_data.get("current", 0)
        brent_current = brent_data.get("current", 0)
        wti_open = wti_data.get("open", wti_current)
        brent_open = brent_data.get("open", brent_current)

        wti_brent_spread = wti_current - brent_current
        wti_brent_spread_open = wti_open - brent_open

        # Convert refined products to $/bbl
        gallon_factor = self.settings.gallons_per_barrel
        rbob_bbl = rbob_data.get("current", 0) * gallon_factor
        ho_bbl = ho_data.get("current", 0) * gallon_factor
        rbob_open_bbl = rbob_data.get("open", rbob_data.get("current", 0)) * gallon_factor
        ho_open_bbl = ho_data.get("open", ho_data.get("current", 0)) * gallon_factor

        crack_321 = self.settings.crack_321.calculate(rbob_bbl, ho_bbl, wti_current)
        crack_321_open = self.settings.crack_321.calculate(rbob_open_bbl, ho_open_bbl, wti_open)
        crack_211 = self.settings.crack_211.calculate(rbob_bbl, ho_bbl, wti_current)

        return {
            "wti_brent": {
                "spread": round(wti_brent_spread, 2),
                "change": round(wti_brent_spread - wti_brent_spread_open, 2),
                "wti": wti_current,
                "brent": brent_current,
            },
            "crack_321": {
                "crack": round(crack_321, 2),
                "change": round(crack_321 - crack_321_open, 2),
                "wti": wti_current,
                "rbob_bbl": round(rbob_bbl, 2),
                "ho_bbl": round(ho_bbl, 2),
            },
            "crack_211": {
                "crack": round(crack_211, 2),
                "wti": wti_current,
                "rbob_bbl": round(rbob_bbl, 2),
                "ho_bbl": round(ho_bbl, 2),
            },
        }

    def get_wti_brent_spread(self) -> dict[str, float]:
        """Calculate WTI-Brent spread with details (batch optimized)."""
        return self.get_all_spreads()["wti_brent"]

    def get_crack_spread_321(self) -> dict[str, float]:
        """
        Calculate 3-2-1 crack spread (batch optimized).

        The 3-2-1 crack spread represents the refining margin for converting
        3 barrels of crude oil into 2 barrels of gasoline and 1 barrel of heating oil.
        """
        return self.get_all_spreads()["crack_321"]

    def get_crack_spread_211(self) -> dict[str, float]:
        """
        Calculate 2-1-1 crack spread (batch optimized).

        The 2-1-1 crack spread represents the refining margin for converting
        2 barrels of crude oil into 1 barrel of gasoline and 1 barrel of heating oil.
        """
        return self.get_all_spreads()["crack_211"]

    def get_all_spreads(self) -> dict[str, dict]:
        """
        Get all spread data in a single optimized call.
        Fetches WTI-Brent spread, 3-2-1 crack, and 2-1-1 crack with one cached batch fetch.
        """
        return self._build_spread_payloads(self._get_spread_price_batch())

    # =========================================================================
    # MARKET SUMMARY
    # =========================================================================

    def get_market_summary(self) -> dict:
        """Get comprehensive market summary."""
        oil_prices = self.get_oil_prices()
        spread_payload = self.get_all_spreads()
        wti_structure = self.get_term_structure("wti")
        brent_structure = self.get_term_structure("brent")

        # Map configured curve spreads into friendly keys
        wti_spreads: dict[str, float] = {}
        for label in self.settings.curve_spread_offsets:
            spread_key = f"{label}_spread"
            value = wti_structure.get(spread_key)
            if value is not None:
                wti_spreads[f"wti_{label}"] = value

        return {
            "prices": oil_prices,
            "spreads": {
                "wti_brent": spread_payload["wti_brent"],
                "crack_321": spread_payload["crack_321"],
                "crack_211": spread_payload["crack_211"],
                **wti_spreads,
            },
            "curve": {
                "structure": wti_structure["structure"],
                "slope": wti_structure["slope"],
                "details": {
                    "wti": {
                        "structure": wti_structure["structure"],
                        "slope": wti_structure["slope"],
                    },
                    "brent": {
                        "structure": brent_structure["structure"],
                        "slope": brent_structure["slope"],
                    },
                },
            },
            "timestamp": datetime.now().isoformat(),
        }

    def get_market_status(self) -> dict:
        """Get quick market status overview."""
        wti_ticker = self.settings.market_status_tickers["wti"]
        brent_ticker = self.settings.market_status_tickers["brent"]

        wti = self.get_price_with_change(wti_ticker)
        brent = self.get_price_with_change(brent_ticker)

        return {
            "wti": wti,
            "brent": brent,
            "spread": round(wti["current"] - brent["current"], 2),
            "market_hours": self._is_market_hours(),
            "timestamp": datetime.now().isoformat(),
        }

    def _is_market_hours(self) -> bool:
        """Check if oil markets are open."""
        return self.settings.is_market_open()

    # =========================================================================
    # DATA REFRESH
    # =========================================================================

    def refresh_all(self) -> None:
        """Refresh all cached data."""
        logger.info("Refreshing all data...")

        # Clear cache
        self.cache.clear()

        # Refresh key data
        self.get_oil_prices()
        self.get_futures_curve("wti")
        self.get_futures_curve("brent")
        self.get_eia_inventory()

        logger.info("Data refresh complete")

    def get_connection_status(self) -> dict:
        """Get Bloomberg connection status."""
        return {
            "connected": self.bloomberg.connected,
            "host": os.environ.get("BLOOMBERG_HOST", "localhost"),
            "port": os.environ.get("BLOOMBERG_PORT", "8194"),
            "subscriptions_enabled": self.subscription_service.subscriptions_enabled,
            "subscribed_tickers": self.subscription_service.get_subscribed_tickers(),
            "data_mode": self._data_mode,
            "connection_error": self._connection_error,
            "data_available": self._data_mode == "live",
        }

    def get_api_efficiency_stats(self) -> dict:
        """
        Get API efficiency statistics including deduplication metrics.
        
        Useful for monitoring and debugging API call patterns.
        
        Returns:
            Dict with cache stats and deduplication metrics
        """
        return {
            "deduplicator": self._deduplicator.get_stats(),
            "cache": self.cache.get_stats(),
        }

    def subscribe_to_core_tickers(self) -> None:
        """Subscribe to core oil market tickers for real-time updates."""
        for ticker in self.settings.core_subscription_tickers:
            self.subscription_service.subscribe(ticker)

        logger.info(f"Subscribed to {len(self.settings.core_subscription_tickers)} core tickers")

    def get_live_prices(self) -> dict[str, dict[str, float]]:
        """Get live prices for all subscribed tickers."""
        return self.subscription_service.get_latest_prices()

    def is_live_data(self) -> bool:
        """Check if using live Bloomberg data."""
        return self._data_mode == "live"

    def is_data_available(self) -> bool:
        """Check if data source is available."""
        return self._data_mode == "live"

    def get_data_mode(self) -> str:
        """Get current data mode: 'live' or 'disconnected'."""
        return self._data_mode

    def get_connection_error_message(self) -> str | None:
        """Get connection error message if disconnected."""
        return self._connection_error

    # =========================================================================
    # CONNECTION MANAGEMENT
    # =========================================================================

    def try_reconnect(self) -> bool:
        """
        Attempt to reconnect to Bloomberg.

        Returns:
            True if reconnection succeeded, False otherwise.
        """
        candidate = BloombergClient()
        if candidate.connected:
            self.bloomberg = candidate
            self._data_mode = "live"
            self._connection_error = None
            self._reset_subscription_service()
            logger.info("Successfully reconnected to Bloomberg")
            return True

        self._connection_error = candidate.get_connection_error()
        logger.warning("Failed to reconnect to Bloomberg: %s", self._connection_error)
        return False

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _reset_subscription_service(self) -> None:
        """Rebuild subscription service for the current Bloomberg client."""
        if hasattr(self, "subscription_service") and self.subscription_service:
            with contextlib.suppress(Exception):
                self.subscription_service.stop()
        self.subscription_service = BloombergSubscriptionService(self.bloomberg)

    def _ensure_datetime_index(self, df: pd.DataFrame | None) -> pd.DataFrame | None:
        """Ensure dataframe index is a tz-naive DatetimeIndex for safe comparisons."""
        if df is None or len(df) == 0:
            return df

        index = pd.to_datetime(df.index)
        if getattr(index, "tz", None) is not None:
            index = index.tz_localize(None)

        if isinstance(df.index, pd.DatetimeIndex) and df.index.equals(index):
            return df

        df = df.copy()
        df.index = index
        return df
