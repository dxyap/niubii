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

from .bloomberg import BloombergClient, MockBloombergData, TickerMapper, BloombergSubscriptionService
from .cache import DataCache, ParquetStorage

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Unified data loading interface.
    
    Handles:
    - Bloomberg API integration (real or mock)
    - Caching layer
    - Parquet storage for historical data
    - Ticker validation and mapping
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
                      If None, auto-detects from environment.
        """
        self.config_dir = Path(config_dir)
        self.data_dir = Path(data_dir)
        
        # Auto-detect mock mode from environment
        if use_mock is None:
            use_mock = os.environ.get("BLOOMBERG_USE_MOCK", "true").lower() == "true"
        
        # Initialize components
        self.bloomberg = BloombergClient(use_mock=use_mock)
        self.cache = DataCache(cache_dir=str(self.data_dir / "cache"))
        self.storage = ParquetStorage(base_dir=str(self.data_dir / "historical"))
        
        # Initialize subscription service for real-time updates
        self.subscription_service = BloombergSubscriptionService(self.bloomberg)
        
        # Load configurations
        self._load_config()
        
        # Determine actual mode
        actual_mode = "live (Bloomberg)" if not self.bloomberg.use_mock and self.bloomberg.connected else "simulated"
        logger.info(f"DataLoader initialized (mode={actual_mode})")
    
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
    
    def get_oil_prices(self) -> Dict[str, Dict[str, float]]:
        """Get current oil prices for key benchmarks with changes."""
        tickers = {
            "WTI": "CL1 Comdty",
            "Brent": "CO1 Comdty",
            "RBOB": "XB1 Comdty",
            "Heating Oil": "HO1 Comdty",
        }
        
        prices = {}
        for name, ticker in tickers.items():
            prices[name] = self.get_price_with_change(ticker)
        
        return prices
    
    def get_all_oil_prices(self) -> Dict[str, Dict[str, float]]:
        """Get prices for all tracked oil products."""
        tickers = {
            "WTI Front": "CL1 Comdty",
            "WTI 2nd": "CL2 Comdty",
            "Brent Front": "CO1 Comdty",
            "Brent 2nd": "CO2 Comdty",
            "RBOB": "XB1 Comdty",
            "Heating Oil": "HO1 Comdty",
            "Gasoil": "QS1 Comdty",
            "Natural Gas": "NG1 Comdty",
        }
        
        prices = {}
        for name, ticker in tickers.items():
            try:
                prices[name] = self.get_price_with_change(ticker)
            except Exception as e:
                logger.warning(f"Could not get price for {ticker}: {e}")
        
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
    
    def get_eia_inventory(self) -> pd.DataFrame:
        """Get EIA crude oil inventory data."""
        cache_key = "eia_inventory"
        cached = self.cache.get(cache_key, cache_type="fundamental")
        
        if cached is not None:
            return cached
        
        # Use mock data for demonstration
        data = MockBloombergData.generate_eia_inventory_data()
        self.cache.set(cache_key, data, cache_type="fundamental")
        self.storage.save_fundamentals("eia_inventory", data)
        
        return data
    
    def get_opec_production(self) -> pd.DataFrame:
        """Get OPEC production and compliance data."""
        return MockBloombergData.generate_opec_production_data()
    
    def get_refinery_turnarounds(self) -> pd.DataFrame:
        """Get refinery turnaround schedule."""
        return MockBloombergData.generate_turnaround_data()
    
    # =========================================================================
    # SPREAD CALCULATIONS
    # =========================================================================
    
    def get_wti_brent_spread(self) -> Dict[str, float]:
        """Calculate WTI-Brent spread with details."""
        wti_data = self.get_price_with_change("CL1 Comdty")
        brent_data = self.get_price_with_change("CO1 Comdty")
        
        spread = wti_data["current"] - brent_data["current"]
        spread_open = wti_data["open"] - brent_data["open"]
        
        return {
            "spread": round(spread, 2),
            "change": round(spread - spread_open, 2),
            "wti": wti_data["current"],
            "brent": brent_data["current"],
        }
    
    def get_crack_spread_321(self) -> Dict[str, float]:
        """
        Calculate 3-2-1 crack spread.
        
        3-2-1 = (2 * RBOB + 1 * HO - 3 * WTI) / 3
        Converted to $/barrel
        """
        wti = self.get_price("CL1 Comdty")
        rbob = self.get_price("XB1 Comdty") * 42  # Convert $/gal to $/bbl
        ho = self.get_price("HO1 Comdty") * 42
        
        crack = (2 * rbob + ho - 3 * wti) / 3
        
        # Calculate change
        wti_data = self.get_price_with_change("CL1 Comdty")
        rbob_data = self.get_price_with_change("XB1 Comdty")
        ho_data = self.get_price_with_change("HO1 Comdty")
        
        crack_open = (2 * rbob_data["open"] * 42 + ho_data["open"] * 42 - 3 * wti_data["open"]) / 3
        
        return {
            "crack": round(crack, 2),
            "change": round(crack - crack_open, 2),
            "wti": wti,
            "rbob_bbl": round(rbob, 2),
            "ho_bbl": round(ho, 2),
        }
    
    def get_crack_spread_211(self) -> Dict[str, float]:
        """Calculate 2-1-1 crack spread."""
        wti = self.get_price("CL1 Comdty")
        rbob = self.get_price("XB1 Comdty") * 42
        ho = self.get_price("HO1 Comdty") * 42
        
        crack = (rbob + ho - 2 * wti) / 2
        
        return {
            "crack": round(crack, 2),
            "wti": wti,
            "rbob_bbl": round(rbob, 2),
            "ho_bbl": round(ho, 2),
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
            "mock_mode": self.bloomberg.use_mock,
            "connected": self.bloomberg.connected if not self.bloomberg.use_mock else True,
            "host": os.environ.get("BLOOMBERG_HOST", "localhost"),
            "port": os.environ.get("BLOOMBERG_PORT", "8194"),
            "subscriptions_enabled": self.subscription_service.subscriptions_enabled,
            "subscribed_tickers": self.subscription_service.get_subscribed_tickers(),
            "data_mode": "live" if (not self.bloomberg.use_mock and self.bloomberg.connected) else "simulated",
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
        return not self.bloomberg.use_mock and self.bloomberg.connected
