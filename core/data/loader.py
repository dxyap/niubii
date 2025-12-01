"""
Data Loading Utilities
======================
High-level data loading with caching and Bloomberg integration.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union
from pathlib import Path
import yaml
import logging
import os

from .bloomberg import (
    BloombergClient, 
    MockBloombergData, 
    TickerMapper, 
    BloombergSubscriptionService,
    DataUnavailableError,
    BloombergConnectionError,
)
from .cache import DataCache, ParquetStorage

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Unified data loading interface.
    
    Handles:
    - Bloomberg API integration (real data by default)
    - Caching layer
    - Parquet storage for historical data
    - Ticker validation and mapping
    
    Raises DataUnavailableError when data cannot be retrieved.
    """
    
    def __init__(
        self,
        config_dir: str = "config",
        data_dir: str = "data",
        use_mock: bool = None
    ):
        """
        Initialize data loader.
        
        Args:
            config_dir: Configuration directory
            data_dir: Data storage directory
            use_mock: Use mock data instead of Bloomberg.
                      If None, reads BLOOMBERG_USE_MOCK env var (default: false).
        """
        self.config_dir = Path(config_dir)
        self.data_dir = Path(data_dir)
        
        # Auto-detect mock mode from environment - DEFAULT IS FALSE (real data)
        if use_mock is None:
            use_mock = os.environ.get("BLOOMBERG_USE_MOCK", "false").lower() == "true"
        
        self._use_mock = use_mock
        
        # Initialize components
        self.bloomberg = BloombergClient(use_mock=use_mock)
        self.cache = DataCache(cache_dir=str(self.data_dir / "cache"))
        self.storage = ParquetStorage(base_dir=str(self.data_dir / "historical"))
        
        # Initialize subscription service for real-time updates
        self.subscription_service = BloombergSubscriptionService(self.bloomberg)
        
        # Load configurations
        self._load_config()
        
        # Determine actual mode and connection status
        if use_mock:
            self._data_mode = "mock"
            self._connection_error = None
            logger.info("DataLoader initialized in MOCK mode (development only)")
        elif self.bloomberg.connected:
            self._data_mode = "live"
            self._connection_error = None
            logger.info("DataLoader initialized in LIVE mode (Bloomberg connected)")
        else:
            # Fall back to mock mode when Bloomberg is not available
            # This ensures the dashboard always has data to display
            self._connection_error = self.bloomberg.get_connection_error()
            logger.warning(f"DataLoader: Bloomberg not connected - {self._connection_error}")
            logger.info("Falling back to MOCK mode for development")
            
            # Re-initialize Bloomberg client in mock mode
            self.bloomberg = BloombergClient(use_mock=True)
            self._use_mock = True
            self._data_mode = "mock"
    
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
    
    def parse_ticker(self, ticker: str) -> Dict:
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
    
    def get_price_with_change(self, ticker: str) -> Dict[str, float]:
        """Get current price with change from open."""
        return self.bloomberg.get_price_with_change(ticker)
    
    def get_prices(self, tickers: List[str]) -> pd.DataFrame:
        """Get current prices for multiple tickers."""
        return self.bloomberg.get_prices(tickers)
    
    def get_prices_batch(self, tickers: List[str]) -> Dict[str, Dict[str, float]]:
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
    
    def get_oil_prices(self) -> Dict[str, Dict[str, float]]:
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
    
    def get_all_oil_prices(self) -> Dict[str, Dict[str, float]]:
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
        start_date: Union[str, datetime] = None,
        end_date: Union[str, datetime] = None,
        frequency: str = "daily"
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data.
        
        First checks local storage, then fetches from Bloomberg if needed.
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
        if end_date is None:
            end_date = datetime.now()
        
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Try loading from storage first
        df = self.storage.load_ohlcv(ticker, frequency, start_date, end_date)
        
        if df is not None and len(df) > 0:
            return df
        
        # Fetch from Bloomberg
        df = self.bloomberg.get_historical(
            ticker,
            start_date,
            end_date,
            frequency=frequency.upper()
        )
        
        # Save to storage for future use
        if df is not None and len(df) > 0:
            self.storage.save_ohlcv(ticker, df, frequency)
        
        return df
    
    def get_historical_multi(
        self,
        tickers: List[str],
        start_date: Union[str, datetime] = None,
        end_date: Union[str, datetime] = None,
        frequency: str = "daily"
    ) -> Dict[str, pd.DataFrame]:
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
    
    def get_term_structure(self, commodity: str = "wti") -> Dict:
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
    
    def get_eia_inventory(self) -> Optional[pd.DataFrame]:
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
    
    def get_opec_production(self) -> Optional[pd.DataFrame]:
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
    
    def get_refinery_turnarounds(self) -> Optional[pd.DataFrame]:
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
    
    def get_wti_brent_spread(self) -> Dict[str, float]:
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
    
    def get_crack_spread_321(self) -> Dict[str, float]:
        """
        Calculate 3-2-1 crack spread (batch optimized).
        
        3-2-1 = (2 * RBOB + 1 * HO - 3 * WTI) / 3
        Converted to $/barrel
        """
        # Batch fetch all 3 tickers in one call (was 6 separate calls before!)
        batch = self.get_prices_batch(["CL1 Comdty", "XB1 Comdty", "HO1 Comdty"])
        
        wti_data = batch.get("CL1 Comdty", {})
        rbob_data = batch.get("XB1 Comdty", {})
        ho_data = batch.get("HO1 Comdty", {})
        
        wti = wti_data.get("current", 0)
        rbob = rbob_data.get("current", 0) * 42  # Convert $/gal to $/bbl
        ho = ho_data.get("current", 0) * 42
        
        crack = (2 * rbob + ho - 3 * wti) / 3
        
        # Calculate change using data already fetched
        wti_open = wti_data.get("open", wti)
        rbob_open = rbob_data.get("open", rbob_data.get("current", 0)) * 42
        ho_open = ho_data.get("open", ho_data.get("current", 0)) * 42
        
        crack_open = (2 * rbob_open + ho_open - 3 * wti_open) / 3
        
        return {
            "crack": round(crack, 2),
            "change": round(crack - crack_open, 2),
            "wti": wti,
            "rbob_bbl": round(rbob, 2),
            "ho_bbl": round(ho, 2),
        }
    
    def get_crack_spread_211(self) -> Dict[str, float]:
        """Calculate 2-1-1 crack spread (batch optimized)."""
        # Batch fetch all 3 tickers in one call
        batch = self.get_prices_batch(["CL1 Comdty", "XB1 Comdty", "HO1 Comdty"])
        
        wti = batch.get("CL1 Comdty", {}).get("current", 0)
        rbob = batch.get("XB1 Comdty", {}).get("current", 0) * 42
        ho = batch.get("HO1 Comdty", {}).get("current", 0) * 42
        
        crack = (rbob + ho - 2 * wti) / 2
        
        return {
            "crack": round(crack, 2),
            "wti": wti,
            "rbob_bbl": round(rbob, 2),
            "ho_bbl": round(ho, 2),
        }
    
    def get_all_spreads(self) -> Dict[str, Dict]:
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
        
        # Crack spreads
        rbob_bbl = rbob_data.get("current", 0) * 42
        ho_bbl = ho_data.get("current", 0) * 42
        rbob_open_bbl = rbob_data.get("open", 0) * 42
        ho_open_bbl = ho_data.get("open", 0) * 42
        
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
    
    def get_market_summary(self) -> Dict:
        """Get comprehensive market summary."""
        oil_prices = self.get_oil_prices()
        wti_curve = self.get_futures_curve("wti")
        brent_curve = self.get_futures_curve("brent")
        
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
    
    def get_market_status(self) -> Dict:
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
        # Weekend closure
        if now.weekday() >= 5:  # Saturday or Sunday
            return False
        return True
    
    # =========================================================================
    # DATA REFRESH
    # =========================================================================
    
    def refresh_all(self) -> None:
        """Refresh all cached data."""
        logger.info("Refreshing all data...")
        
        # Clear cache
        self.cache.clear()
        
        # Reset price simulator session
        if hasattr(self.bloomberg, 'simulator'):
            self.bloomberg.simulator.reset_session()
        
        # Refresh key data
        self.get_oil_prices()
        self.get_futures_curve("wti")
        self.get_futures_curve("brent")
        self.get_eia_inventory()
        
        logger.info("Data refresh complete")
    
    def get_connection_status(self) -> Dict:
        """Get Bloomberg connection status."""
        return {
            "mock_mode": self._use_mock,
            "connected": self.bloomberg.connected,
            "host": os.environ.get("BLOOMBERG_HOST", "localhost"),
            "port": os.environ.get("BLOOMBERG_PORT", "8194"),
            "subscriptions_enabled": self.subscription_service.subscriptions_enabled,
            "subscribed_tickers": self.subscription_service.get_subscribed_tickers(),
            "data_mode": self._data_mode,
            "connection_error": self._connection_error,
            "data_available": self._data_mode in ("live", "mock"),
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
    
    def get_live_prices(self) -> Dict[str, Dict[str, float]]:
        """Get live prices for all subscribed tickers."""
        return self.subscription_service.get_latest_prices()
    
    def is_live_data(self) -> bool:
        """Check if using live Bloomberg data."""
        return self._data_mode == "live"
    
    def is_data_available(self) -> bool:
        """Check if any data source is available (live or mock)."""
        return self._data_mode in ("live", "mock")
    
    def get_data_mode(self) -> str:
        """Get current data mode: 'live', 'mock', or 'disconnected'."""
        return self._data_mode
    
    def get_connection_error_message(self) -> Optional[str]:
        """Get connection error message if disconnected."""
        return self._connection_error
