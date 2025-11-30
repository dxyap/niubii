"""
Bloomberg API Client
====================
Wrapper for Bloomberg Desktop API (BLPAPI) with mock data fallback.

In production, this would connect to the actual Bloomberg Terminal.
For development/demo, it generates realistic mock data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)


class BloombergClient:
    """
    Bloomberg API Client with mock data fallback.
    
    When Bloomberg API is not available, generates realistic mock data
    for development and demonstration purposes.
    """
    
    def __init__(self, use_mock: bool = True):
        """
        Initialize Bloomberg client.
        
        Args:
            use_mock: If True, use mock data instead of real Bloomberg API
        """
        self.use_mock = use_mock
        self.connected = False
        self._cache = {}
        
        # Base prices for mock data generation
        self._base_prices = {
            "CL1 Comdty": 72.50,   # WTI
            "CL2 Comdty": 72.15,
            "CO1 Comdty": 77.20,   # Brent
            "CO2 Comdty": 76.85,
            "XB1 Comdty": 2.18,    # RBOB Gasoline ($/gal)
            "HO1 Comdty": 2.52,    # Heating Oil ($/gal)
            "QS1 Comdty": 680.50,  # Gasoil ($/tonne)
        }
        
        # Generate curve prices (contango structure)
        for i in range(3, 13):
            self._base_prices[f"CL{i} Comdty"] = 72.50 + (i - 1) * 0.15
            self._base_prices[f"CO{i} Comdty"] = 77.20 + (i - 1) * 0.12
        
        if not use_mock:
            self._connect()
    
    def _connect(self):
        """Attempt to connect to Bloomberg API."""
        try:
            # In production, this would use blpapi
            # import blpapi
            # self.session = blpapi.Session()
            # self.session.start()
            logger.info("Bloomberg connection not available, using mock data")
            self.use_mock = True
        except Exception as e:
            logger.warning(f"Could not connect to Bloomberg: {e}")
            self.use_mock = True
    
    def get_price(self, ticker: str, field: str = "PX_LAST") -> float:
        """
        Get current price for a ticker.
        
        Args:
            ticker: Bloomberg ticker
            field: Price field (PX_LAST, PX_BID, PX_ASK)
            
        Returns:
            Current price
        """
        if self.use_mock:
            return self._mock_price(ticker, field)
        
        # Real Bloomberg API call would go here
        raise NotImplementedError("Real Bloomberg API not implemented")
    
    def get_prices(self, tickers: List[str], fields: List[str] = None) -> pd.DataFrame:
        """
        Get current prices for multiple tickers.
        
        Args:
            tickers: List of Bloomberg tickers
            fields: List of price fields
            
        Returns:
            DataFrame with prices
        """
        if fields is None:
            fields = ["PX_LAST", "PX_BID", "PX_ASK", "PX_VOLUME"]
        
        if self.use_mock:
            return self._mock_prices(tickers, fields)
        
        raise NotImplementedError("Real Bloomberg API not implemented")
    
    def get_historical(
        self,
        ticker: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime] = None,
        fields: List[str] = None,
        frequency: str = "DAILY"
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data.
        
        Args:
            ticker: Bloomberg ticker
            start_date: Start date
            end_date: End date (defaults to today)
            fields: OHLCV fields
            frequency: Data frequency (DAILY, WEEKLY, MONTHLY)
            
        Returns:
            DataFrame with historical data
        """
        if fields is None:
            fields = ["PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST", "PX_VOLUME"]
        
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if end_date is None:
            end_date = datetime.now()
        elif isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        if self.use_mock:
            return self._mock_historical(ticker, start_date, end_date, fields, frequency)
        
        raise NotImplementedError("Real Bloomberg API not implemented")
    
    def get_curve(self, commodity: str = "wti", num_months: int = 12) -> pd.DataFrame:
        """
        Get futures curve data.
        
        Args:
            commodity: Commodity type (wti, brent)
            num_months: Number of months on curve
            
        Returns:
            DataFrame with curve data
        """
        ticker_prefix = "CL" if commodity.lower() == "wti" else "CO"
        
        data = []
        today = datetime.now()
        
        for i in range(1, num_months + 1):
            ticker = f"{ticker_prefix}{i} Comdty"
            price = self.get_price(ticker)
            expiry = today + timedelta(days=30 * i)
            
            data.append({
                "month": i,
                "ticker": ticker,
                "price": price,
                "expiry": expiry,
                "days_to_expiry": (expiry - today).days
            })
        
        return pd.DataFrame(data)
    
    def get_reference_data(self, ticker: str, fields: List[str]) -> Dict:
        """
        Get reference data for a ticker.
        
        Args:
            ticker: Bloomberg ticker
            fields: Reference data fields
            
        Returns:
            Dictionary of reference data
        """
        if self.use_mock:
            return self._mock_reference(ticker, fields)
        
        raise NotImplementedError("Real Bloomberg API not implemented")
    
    # Mock data generation methods
    def _mock_price(self, ticker: str, field: str) -> float:
        """Generate mock price with realistic noise."""
        base = self._base_prices.get(ticker, 70.0)
        noise = np.random.normal(0, base * 0.002)  # 0.2% noise
        
        if field == "PX_BID":
            return round(base + noise - 0.02, 2)
        elif field == "PX_ASK":
            return round(base + noise + 0.02, 2)
        return round(base + noise, 2)
    
    def _mock_prices(self, tickers: List[str], fields: List[str]) -> pd.DataFrame:
        """Generate mock prices for multiple tickers."""
        data = {}
        for ticker in tickers:
            row = {}
            for field in fields:
                if field == "PX_VOLUME":
                    row[field] = np.random.randint(10000, 100000)
                else:
                    row[field] = self._mock_price(ticker, field)
            data[ticker] = row
        
        return pd.DataFrame(data).T
    
    def _mock_historical(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        fields: List[str],
        frequency: str
    ) -> pd.DataFrame:
        """Generate mock historical data with realistic patterns."""
        base_price = self._base_prices.get(ticker, 70.0)
        
        # Generate date range
        if frequency == "DAILY":
            dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
        elif frequency == "WEEKLY":
            dates = pd.date_range(start=start_date, end=end_date, freq='W')
        else:
            dates = pd.date_range(start=start_date, end=end_date, freq='M')
        
        n = len(dates)
        
        # Generate price series with trend and mean reversion
        np.random.seed(42)  # For reproducibility
        returns = np.random.normal(0.0002, 0.015, n)  # Daily returns
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Add seasonality
        seasonal = 0.05 * np.sin(2 * np.pi * np.arange(n) / 252)
        prices = prices * (1 + seasonal)
        
        # Generate OHLC from close
        data = {
            "date": dates,
            "PX_LAST": prices,
            "PX_OPEN": prices * (1 + np.random.normal(0, 0.002, n)),
            "PX_HIGH": prices * (1 + np.abs(np.random.normal(0.003, 0.005, n))),
            "PX_LOW": prices * (1 - np.abs(np.random.normal(0.003, 0.005, n))),
            "PX_VOLUME": np.random.randint(50000, 200000, n),
        }
        
        df = pd.DataFrame(data)
        df.set_index("date", inplace=True)
        
        # Filter to requested fields
        available_fields = [f for f in fields if f in df.columns]
        return df[available_fields]
    
    def _mock_reference(self, ticker: str, fields: List[str]) -> Dict:
        """Generate mock reference data."""
        ref = {
            "NAME": f"Oil Futures {ticker[:2]}",
            "TICKER": ticker,
            "EXCH_CODE": "NYMEX" if ticker.startswith("CL") else "ICE",
            "CRNCY": "USD",
            "FUT_CONT_SIZE": 1000,
            "FUT_TICK_SIZE": 0.01,
        }
        return {f: ref.get(f, None) for f in fields}


class MockBloombergData:
    """
    Generate comprehensive mock market data for demonstration.
    """
    
    @staticmethod
    def generate_eia_inventory_data(periods: int = 52) -> pd.DataFrame:
        """Generate mock EIA inventory data."""
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='W-WED')
        
        # Base inventory level around 430 million barrels
        base = 430.0
        
        # Add seasonality (higher in spring, lower in fall)
        seasonal = 15 * np.sin(2 * np.pi * np.arange(periods) / 52 - np.pi/2)
        
        # Add trend and noise
        trend = np.linspace(-10, 5, periods)
        noise = np.random.normal(0, 3, periods)
        
        inventory = base + seasonal + trend + noise
        
        # Calculate week-over-week change
        change = np.diff(inventory, prepend=inventory[0])
        
        return pd.DataFrame({
            "date": dates,
            "inventory_mmb": inventory,
            "change_mmb": change,
            "expectation_mmb": change + np.random.normal(0, 1.5, periods),
        }).set_index("date")
    
    @staticmethod
    def generate_opec_production_data() -> pd.DataFrame:
        """Generate mock OPEC production data."""
        countries = {
            "Saudi Arabia": {"quota": 9.00, "compliance": 0.98},
            "Russia": {"quota": 9.50, "compliance": 0.95},
            "Iraq": {"quota": 4.00, "compliance": 0.85},
            "UAE": {"quota": 2.90, "compliance": 1.01},
            "Kuwait": {"quota": 2.40, "compliance": 0.99},
            "Nigeria": {"quota": 1.38, "compliance": 0.88},
            "Angola": {"quota": 1.28, "compliance": 0.92},
            "Algeria": {"quota": 0.96, "compliance": 0.97},
        }
        
        data = []
        for country, params in countries.items():
            actual = params["quota"] * params["compliance"]
            data.append({
                "country": country,
                "quota_mbpd": params["quota"],
                "actual_mbpd": round(actual, 2),
                "compliance_pct": round(params["compliance"] * 100, 1),
            })
        
        return pd.DataFrame(data)
    
    @staticmethod
    def generate_turnaround_data() -> pd.DataFrame:
        """Generate mock refinery turnaround data."""
        turnarounds = [
            {
                "region": "USGC",
                "refinery": "Motiva Port Arthur",
                "capacity_kbpd": 630,
                "start_date": datetime.now() + timedelta(days=5),
                "end_date": datetime.now() + timedelta(days=20),
                "type": "Planned",
            },
            {
                "region": "USGC",
                "refinery": "Marathon Galveston Bay",
                "capacity_kbpd": 585,
                "start_date": datetime.now() + timedelta(days=10),
                "end_date": datetime.now() + timedelta(days=35),
                "type": "Planned",
            },
            {
                "region": "NW Europe",
                "refinery": "Shell Pernis",
                "capacity_kbpd": 404,
                "start_date": datetime.now() + timedelta(days=1),
                "end_date": datetime.now() + timedelta(days=15),
                "type": "Planned",
            },
            {
                "region": "Asia",
                "refinery": "SK Ulsan",
                "capacity_kbpd": 840,
                "start_date": datetime.now() + timedelta(days=35),
                "end_date": datetime.now() + timedelta(days=55),
                "type": "Planned",
            },
            {
                "region": "USGC",
                "refinery": "ExxonMobil Beaumont",
                "capacity_kbpd": 369,
                "start_date": datetime.now() - timedelta(days=5),
                "end_date": datetime.now() + timedelta(days=10),
                "type": "Unplanned",
            },
        ]
        
        return pd.DataFrame(turnarounds)
