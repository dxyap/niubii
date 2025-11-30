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

from .bloomberg import BloombergClient, MockBloombergData
from .cache import DataCache, ParquetStorage

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Unified data loading interface.
    
    Handles:
    - Bloomberg API integration
    - Caching layer
    - Parquet storage for historical data
    - Mock data for development
    """
    
    def __init__(
        self,
        config_dir: str = "config",
        data_dir: str = "data",
        use_mock: bool = True
    ):
        """
        Initialize data loader.
        
        Args:
            config_dir: Configuration directory
            data_dir: Data storage directory
            use_mock: Use mock data instead of Bloomberg
        """
        self.config_dir = Path(config_dir)
        self.data_dir = Path(data_dir)
        
        # Initialize components
        self.bloomberg = BloombergClient(use_mock=use_mock)
        self.cache = DataCache(cache_dir=str(self.data_dir / "cache"))
        self.storage = ParquetStorage(base_dir=str(self.data_dir / "historical"))
        
        # Load configurations
        self._load_config()
    
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
    
    # Real-time Data Methods
    def get_price(self, ticker: str) -> float:
        """Get current price for ticker."""
        cache_key = f"price_{ticker}"
        cached = self.cache.get(cache_key, cache_type="real_time")
        
        if cached is not None:
            return cached
        
        price = self.bloomberg.get_price(ticker)
        self.cache.set(cache_key, price, cache_type="real_time")
        return price
    
    def get_prices(self, tickers: List[str]) -> pd.DataFrame:
        """Get current prices for multiple tickers."""
        return self.bloomberg.get_prices(tickers)
    
    def get_oil_prices(self) -> Dict[str, float]:
        """Get current oil prices for key benchmarks."""
        tickers = {
            "WTI": "CL1 Comdty",
            "Brent": "CO1 Comdty",
            "RBOB": "XB1 Comdty",
            "Heating Oil": "HO1 Comdty",
        }
        
        prices = {}
        for name, ticker in tickers.items():
            prices[name] = self.get_price(ticker)
        
        return prices
    
    # Historical Data Methods
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
    
    # Curve Data Methods
    def get_futures_curve(
        self,
        commodity: str = "wti",
        num_months: int = 12
    ) -> pd.DataFrame:
        """
        Get futures curve data.
        
        Args:
            commodity: 'wti' or 'brent'
            num_months: Number of months on curve
            
        Returns:
            DataFrame with curve data
        """
        cache_key = f"curve_{commodity}_{num_months}"
        cached = self.cache.get(cache_key, cache_type="intraday")
        
        if cached is not None:
            return cached
        
        curve = self.bloomberg.get_curve(commodity, num_months)
        self.cache.set(cache_key, curve, cache_type="intraday")
        
        return curve
    
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
    
    # Fundamental Data Methods
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
    
    # Spread Calculations
    def get_wti_brent_spread(self) -> float:
        """Calculate WTI-Brent spread."""
        wti = self.get_price("CL1 Comdty")
        brent = self.get_price("CO1 Comdty")
        return wti - brent
    
    def get_crack_spread_321(self) -> float:
        """
        Calculate 3-2-1 crack spread.
        
        3-2-1 = (2 * RBOB + 1 * HO - 3 * WTI) / 3
        Converted to $/barrel
        """
        wti = self.get_price("CL1 Comdty")
        rbob = self.get_price("XB1 Comdty") * 42  # Convert $/gal to $/bbl
        ho = self.get_price("HO1 Comdty") * 42
        
        crack = (2 * rbob + ho - 3 * wti) / 3
        return round(crack, 2)
    
    # Market Summary
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
                "wti_brent": round(wti_brent, 2),
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
    
    # Data Refresh
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
