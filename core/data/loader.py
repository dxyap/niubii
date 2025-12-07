"""
Data Loading Utilities
======================
High-level data loading with caching and Bloomberg integration.
"""

import contextlib
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml

from .bloomberg import (
    BloombergClient,
    BloombergSubscriptionService,
    TickerMapper,
)
from .cache import DataCache, ParquetStorage

logger = logging.getLogger(__name__)


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

    # Standard conversion factor: gallons per barrel
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

        # Initialize components
        self.bloomberg = BloombergClient()
        self.cache = DataCache(cache_dir=str(self.data_dir / "cache"))
        self.storage = ParquetStorage(base_dir=str(self.data_dir / "historical"))

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
        """Get current price for ticker."""
        if validate:
            valid, msg = self.validate_ticker(ticker)
            if not valid:
                logger.warning(f"Invalid ticker {ticker}: {msg}")

        return self.bloomberg.get_price(ticker)

    def get_price_with_change(self, ticker: str) -> dict[str, float]:
        """Get current price with change from open."""
        return self.bloomberg.get_price_with_change(ticker)

    def get_prices(self, tickers: list[str]) -> pd.DataFrame:
        """Get current prices for multiple tickers."""
        return self.bloomberg.get_prices(tickers)

    def get_prices_batch(self, tickers: list[str]) -> dict[str, dict[str, float]]:
        """
        Get prices with changes for multiple tickers in a single batch call.
        Much more efficient than calling get_price_with_change() for each ticker.

        Returns:
            Dict mapping ticker to price data dict
        """
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

    def get_oil_prices(self) -> dict[str, dict[str, float]]:
        """Get current oil prices for key benchmarks with changes (batch optimized)."""
        ticker_map = {
            "WTI": "CL1 Comdty",
            "Brent": "CO1 Comdty",
            "RBOB": "XB1 Comdty",
            "Heating Oil": "HO1 Comdty",
        }

        # Batch fetch all prices at once
        batch_prices = self.get_prices_batch(list(ticker_map.values()))

        # Map back to friendly names
        prices = {}
        for name, ticker in ticker_map.items():
            if ticker in batch_prices:
                prices[name] = batch_prices[ticker]

        return prices

    def get_all_oil_prices(self) -> dict[str, dict[str, float]]:
        """Get prices for all tracked oil products (batch optimized)."""
        ticker_map = {
            "WTI Front": "CL1 Comdty",
            "WTI 2nd": "CL2 Comdty",
            "Brent Front": "CO1 Comdty",
            "Brent 2nd": "CO2 Comdty",
            "RBOB": "XB1 Comdty",
            "Heating Oil": "HO1 Comdty",
            "Gasoil": "QS1 Comdty",
            "Natural Gas": "NG1 Comdty",
        }

        # Batch fetch all prices at once
        batch_prices = self.get_prices_batch(list(ticker_map.values()))

        # Map back to friendly names
        prices = {}
        for name, ticker in ticker_map.items():
            if ticker in batch_prices:
                prices[name] = batch_prices[ticker]

        return prices

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
        num_months: int = 12
    ) -> pd.DataFrame:
        """
        Get futures curve data.

        Args:
            commodity: 'wti', 'brent', etc.
            num_months: Number of months on curve

        Returns:
            DataFrame with curve data including changes
        """
        return self.bloomberg.get_curve(commodity, num_months)

    def get_calendar_spreads(self, commodity: str = "wti") -> pd.DataFrame:
        """Calculate calendar spreads from curve."""
        curve = self.get_futures_curve(commodity, num_months=12)

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
        curve = self.get_futures_curve(commodity, num_months=12)

        if len(curve) < 2:
            return {"structure": "Unknown", "slope": 0}

        # Calculate overall slope
        slope = (curve.iloc[-1]["price"] - curve.iloc[0]["price"]) / (len(curve) - 1)

        # Determine structure
        if slope > 0.05:
            structure = "Contango"
        elif slope < -0.05:
            structure = "Backwardation"
        else:
            structure = "Flat"

        # Key spreads
        m1_m2 = curve.iloc[0]["price"] - curve.iloc[1]["price"]
        m1_m6 = curve.iloc[0]["price"] - curve.iloc[5]["price"] if len(curve) > 5 else 0
        m1_m12 = curve.iloc[0]["price"] - curve.iloc[11]["price"] if len(curve) > 11 else 0

        return {
            "structure": structure,
            "slope": round(slope, 4),
            "m1_m2_spread": round(m1_m2, 2),
            "m1_m6_spread": round(m1_m6, 2),
            "m1_m12_spread": round(m1_m12, 2),
            "curve_data": curve,
        }

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

    def get_wti_brent_spread(self) -> dict[str, float]:
        """Calculate WTI-Brent spread with details (batch optimized)."""
        # Batch fetch both tickers in one call
        batch = self.get_prices_batch(["CL1 Comdty", "CO1 Comdty"])
        wti_data = batch.get("CL1 Comdty", {})
        brent_data = batch.get("CO1 Comdty", {})

        wti_current = wti_data.get("current", 0)
        brent_current = brent_data.get("current", 0)
        wti_open = wti_data.get("open", wti_current)
        brent_open = brent_data.get("open", brent_current)

        spread = wti_current - brent_current
        spread_open = wti_open - brent_open

        return {
            "spread": round(spread, 2),
            "change": round(spread - spread_open, 2),
            "wti": wti_current,
            "brent": brent_current,
        }

    def get_crack_spread_321(self) -> dict[str, float]:
        """
        Calculate 3-2-1 crack spread (batch optimized).

        The 3-2-1 crack spread represents the refining margin for converting
        3 barrels of crude oil into 2 barrels of gasoline and 1 barrel of heating oil.

        Formula: (2 × RBOB_$/bbl + 1 × HO_$/bbl - 3 × WTI_$/bbl) / 3

        Unit conversions applied:
        - WTI (CL1): Already in $/barrel, no conversion needed
        - RBOB (XB1): Quoted in $/gallon, multiply by 42 to get $/barrel
        - Heating Oil (HO1): Quoted in $/gallon, multiply by 42 to get $/barrel

        Returns:
            Dict with keys:
            - crack: The 3-2-1 crack spread in $/barrel
            - change: Change from session open in $/barrel
            - wti: WTI price in $/barrel
            - rbob_bbl: RBOB price converted to $/barrel
            - ho_bbl: Heating oil price converted to $/barrel
        """
        # Batch fetch all 3 tickers in one call (was 6 separate calls before!)
        batch = self.get_prices_batch(["CL1 Comdty", "XB1 Comdty", "HO1 Comdty"])

        wti_data = batch.get("CL1 Comdty", {})
        rbob_data = batch.get("XB1 Comdty", {})
        ho_data = batch.get("HO1 Comdty", {})

        # WTI is already in $/barrel
        wti = wti_data.get("current", 0)

        # Convert RBOB and HO from $/gallon to $/barrel (42 gallons per barrel)
        rbob = rbob_data.get("current", 0) * self.GALLONS_PER_BARREL
        ho = ho_data.get("current", 0) * self.GALLONS_PER_BARREL

        # 3-2-1 crack spread: margin per barrel of crude processed
        crack = (2 * rbob + ho - 3 * wti) / 3

        # Calculate change using data already fetched
        wti_open = wti_data.get("open", wti)
        rbob_open = rbob_data.get("open", rbob_data.get("current", 0)) * self.GALLONS_PER_BARREL
        ho_open = ho_data.get("open", ho_data.get("current", 0)) * self.GALLONS_PER_BARREL

        crack_open = (2 * rbob_open + ho_open - 3 * wti_open) / 3

        return {
            "crack": round(crack, 2),
            "change": round(crack - crack_open, 2),
            "wti": wti,
            "rbob_bbl": round(rbob, 2),
            "ho_bbl": round(ho, 2),
        }

    def get_crack_spread_211(self) -> dict[str, float]:
        """
        Calculate 2-1-1 crack spread (batch optimized).

        The 2-1-1 crack spread represents the refining margin for converting
        2 barrels of crude oil into 1 barrel of gasoline and 1 barrel of heating oil.

        Formula: (1 × RBOB_$/bbl + 1 × HO_$/bbl - 2 × WTI_$/bbl) / 2

        Unit conversions applied:
        - WTI (CL1): Already in $/barrel, no conversion needed
        - RBOB (XB1): Quoted in $/gallon, multiply by 42 to get $/barrel
        - Heating Oil (HO1): Quoted in $/gallon, multiply by 42 to get $/barrel

        Returns:
            Dict with keys:
            - crack: The 2-1-1 crack spread in $/barrel
            - wti: WTI price in $/barrel
            - rbob_bbl: RBOB price converted to $/barrel
            - ho_bbl: Heating oil price converted to $/barrel
        """
        # Batch fetch all 3 tickers in one call
        batch = self.get_prices_batch(["CL1 Comdty", "XB1 Comdty", "HO1 Comdty"])

        # WTI is already in $/barrel
        wti = batch.get("CL1 Comdty", {}).get("current", 0)

        # Convert RBOB and HO from $/gallon to $/barrel
        rbob = batch.get("XB1 Comdty", {}).get("current", 0) * self.GALLONS_PER_BARREL
        ho = batch.get("HO1 Comdty", {}).get("current", 0) * self.GALLONS_PER_BARREL

        # 2-1-1 crack spread: margin per barrel of crude processed
        crack = (rbob + ho - 2 * wti) / 2

        return {
            "crack": round(crack, 2),
            "wti": wti,
            "rbob_bbl": round(rbob, 2),
            "ho_bbl": round(ho, 2),
        }

    def get_all_spreads(self) -> dict[str, dict]:
        """
        Get all spread data in a single optimized call.
        Fetches WTI-Brent spread, 3-2-1 crack, and 2-1-1 crack with one batch fetch.
        """
        # Single batch fetch for all spread calculations
        batch = self.get_prices_batch([
            "CL1 Comdty", "CO1 Comdty", "XB1 Comdty", "HO1 Comdty"
        ])

        wti_data = batch.get("CL1 Comdty", {})
        brent_data = batch.get("CO1 Comdty", {})
        rbob_data = batch.get("XB1 Comdty", {})
        ho_data = batch.get("HO1 Comdty", {})

        # WTI-Brent spread
        wti_current = wti_data.get("current", 0)
        brent_current = brent_data.get("current", 0)
        wti_open = wti_data.get("open", wti_current)
        brent_open = brent_data.get("open", brent_current)

        wti_brent_spread = wti_current - brent_current
        wti_brent_spread_open = wti_open - brent_open

        # Crack spreads - convert product prices from $/gallon to $/barrel
        rbob_bbl = rbob_data.get("current", 0) * self.GALLONS_PER_BARREL
        ho_bbl = ho_data.get("current", 0) * self.GALLONS_PER_BARREL
        rbob_open_bbl = rbob_data.get("open", 0) * self.GALLONS_PER_BARREL
        ho_open_bbl = ho_data.get("open", 0) * self.GALLONS_PER_BARREL

        crack_321 = (2 * rbob_bbl + ho_bbl - 3 * wti_current) / 3
        crack_321_open = (2 * rbob_open_bbl + ho_open_bbl - 3 * wti_open) / 3

        crack_211 = (rbob_bbl + ho_bbl - 2 * wti_current) / 2

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

    # =========================================================================
    # MARKET SUMMARY
    # =========================================================================

    def get_market_summary(self) -> dict:
        """Get comprehensive market summary."""
        oil_prices = self.get_oil_prices()
        wti_curve = self.get_futures_curve("wti")
        self.get_futures_curve("brent")

        # Calculate spreads
        wti_brent = self.get_wti_brent_spread()
        crack_321 = self.get_crack_spread_321()

        # Calendar spreads
        wti_m1_m2 = wti_curve.iloc[0]["price"] - wti_curve.iloc[1]["price"]
        wti_m1_m12 = wti_curve.iloc[0]["price"] - wti_curve.iloc[11]["price"]

        # Curve shape
        curve_slope = (wti_curve.iloc[11]["price"] - wti_curve.iloc[0]["price"]) / 11
        structure = "Contango" if curve_slope > 0.05 else "Backwardation" if curve_slope < -0.05 else "Flat"

        return {
            "prices": oil_prices,
            "spreads": {
                "wti_brent": wti_brent,
                "crack_321": crack_321,
                "wti_m1_m2": round(wti_m1_m2, 2),
                "wti_m1_m12": round(wti_m1_m12, 2),
            },
            "curve": {
                "structure": structure,
                "slope": round(curve_slope, 3),
            },
            "timestamp": datetime.now().isoformat(),
        }

    def get_market_status(self) -> dict:
        """Get quick market status overview."""
        wti = self.get_price_with_change("CL1 Comdty")
        brent = self.get_price_with_change("CO1 Comdty")

        return {
            "wti": wti,
            "brent": brent,
            "spread": round(wti["current"] - brent["current"], 2),
            "market_hours": self._is_market_hours(),
            "timestamp": datetime.now().isoformat(),
        }

    def _is_market_hours(self) -> bool:
        """Check if oil markets are open."""
        now = datetime.now()
        # Simplified check - oil trades nearly 24 hours
        # Weekend closure: weekday >= 5 means Saturday or Sunday
        return now.weekday() < 5

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

    def subscribe_to_core_tickers(self) -> None:
        """Subscribe to core oil market tickers for real-time updates."""
        core_tickers = [
            "CL1 Comdty",  # WTI Front Month
            "CL2 Comdty",  # WTI 2nd Month
            "CO1 Comdty",  # Brent Front Month
            "CO2 Comdty",  # Brent 2nd Month
            "XB1 Comdty",  # RBOB Gasoline
            "HO1 Comdty",  # Heating Oil
        ]

        for ticker in core_tickers:
            self.subscription_service.subscribe(ticker)

        logger.info(f"Subscribed to {len(core_tickers)} core tickers")

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
