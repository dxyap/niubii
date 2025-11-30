"""
Bloomberg API Client
====================
Wrapper for Bloomberg Desktop API (BLPAPI) with mock data fallback.

In production, this would connect to the actual Bloomberg Terminal.
For development/demo, it generates realistic mock data with price persistence.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union
import logging
import hashlib

logger = logging.getLogger(__name__)


class PriceSimulator:
    """
    Realistic price simulator that maintains state across calls.
    Simulates market microstructure with drift, mean reversion, and volatility clustering.
    """
    
    def __init__(self):
        # Base reference prices (as of market close)
        self._reference_prices = {
            "CL1 Comdty": 72.50,   # WTI Front Month
            "CL2 Comdty": 72.65,   # WTI 2nd Month
            "CO1 Comdty": 77.20,   # Brent Front Month
            "CO2 Comdty": 77.35,   # Brent 2nd Month
            "XB1 Comdty": 2.18,    # RBOB Gasoline ($/gal)
            "HO1 Comdty": 2.52,    # Heating Oil ($/gal)
            "QS1 Comdty": 680.50,  # Gasoil ($/tonne)
        }
        
        # Generate curve prices with realistic term structure
        for i in range(3, 13):
            # WTI: slight contango
            self._reference_prices[f"CL{i} Comdty"] = 72.50 + (i - 1) * 0.12
            # Brent: slight contango
            self._reference_prices[f"CO{i} Comdty"] = 77.20 + (i - 1) * 0.10
        
        # Current simulated prices (will drift from reference)
        self._current_prices: Dict[str, float] = {}
        self._last_update: Dict[str, datetime] = {}
        self._price_history: Dict[str, List[tuple]] = {}  # [(timestamp, price), ...]
        
        # Volatility state for GARCH-like behavior
        self._volatility_state: Dict[str, float] = {}
        
        # Session start time for intraday simulation
        self._session_start = datetime.now()
        self._daily_open: Dict[str, float] = {}
        
        # Initialize prices
        self._initialize_session()
    
    def _initialize_session(self):
        """Initialize a new trading session with opening prices."""
        for ticker, ref_price in self._reference_prices.items():
            # Small gap from reference (overnight move)
            gap = np.random.normal(0, ref_price * 0.005)
            open_price = ref_price + gap
            
            self._daily_open[ticker] = open_price
            self._current_prices[ticker] = open_price
            self._last_update[ticker] = datetime.now()
            self._price_history[ticker] = [(datetime.now(), open_price)]
            self._volatility_state[ticker] = 0.0001  # Initial variance
    
    def get_price(self, ticker: str, field: str = "PX_LAST") -> float:
        """
        Get simulated price with realistic tick-by-tick movement.
        """
        if ticker not in self._current_prices:
            # Unknown ticker - use default
            base = 70.0
            self._current_prices[ticker] = base
            self._daily_open[ticker] = base
            self._last_update[ticker] = datetime.now()
            self._price_history[ticker] = [(datetime.now(), base)]
            self._volatility_state[ticker] = 0.0001
        
        # Time since last update
        now = datetime.now()
        last = self._last_update.get(ticker, now)
        elapsed = (now - last).total_seconds()
        
        # Update price if enough time has passed (simulate tick arrival)
        if elapsed > 0.5:  # Update every 500ms minimum
            self._update_price(ticker, elapsed)
        
        current = self._current_prices[ticker]
        
        # Apply bid/ask spread
        spread_pct = 0.0003  # 3 bps spread
        spread = current * spread_pct
        
        if field == "PX_BID":
            return round(current - spread / 2, 4)
        elif field == "PX_ASK":
            return round(current + spread / 2, 4)
        elif field == "PX_OPEN":
            return round(self._daily_open.get(ticker, current), 4)
        elif field == "PX_HIGH":
            history = self._price_history.get(ticker, [])
            if history:
                return round(max(p for _, p in history), 4)
            return round(current, 4)
        elif field == "PX_LOW":
            history = self._price_history.get(ticker, [])
            if history:
                return round(min(p for _, p in history), 4)
            return round(current, 4)
        
        return round(current, 4)
    
    def _update_price(self, ticker: str, elapsed_seconds: float):
        """
        Update price using a realistic market microstructure model.
        Combines: random walk + mean reversion + volatility clustering
        """
        current = self._current_prices[ticker]
        reference = self._reference_prices.get(ticker, current)
        
        # Scale volatility by time elapsed (but cap it)
        time_scale = min(elapsed_seconds / 60, 5)  # Cap at 5 minutes equivalent
        
        # GARCH-like volatility clustering
        vol_state = self._volatility_state.get(ticker, 0.0001)
        
        # Volatility parameters (annualized ~25% for oil)
        base_vol = 0.25 / np.sqrt(252 * 6.5 * 60)  # Per-minute vol
        
        # Update volatility state (GARCH(1,1) approximation)
        shock = np.random.standard_normal()
        vol_state = 0.9 * vol_state + 0.1 * (shock ** 2) * (base_vol ** 2)
        self._volatility_state[ticker] = vol_state
        
        current_vol = np.sqrt(vol_state) * np.sqrt(time_scale)
        
        # Mean reversion toward reference (weak)
        mean_reversion_speed = 0.01
        mean_reversion = mean_reversion_speed * (reference - current) * time_scale / 60
        
        # Random innovation
        innovation = current_vol * current * shock
        
        # Small drift (market microstructure)
        drift = 0
        
        # New price
        new_price = current + drift + mean_reversion + innovation
        
        # Ensure price stays positive
        new_price = max(new_price, current * 0.9)
        
        self._current_prices[ticker] = new_price
        self._last_update[ticker] = datetime.now()
        
        # Store in history (keep last 1000 points)
        history = self._price_history.get(ticker, [])
        history.append((datetime.now(), new_price))
        if len(history) > 1000:
            history = history[-1000:]
        self._price_history[ticker] = history
    
    def get_price_change(self, ticker: str) -> Dict[str, float]:
        """Get price change from session open."""
        current = self.get_price(ticker)
        open_price = self._daily_open.get(ticker, current)
        
        change = current - open_price
        change_pct = (change / open_price * 100) if open_price != 0 else 0
        
        return {
            "current": current,
            "open": open_price,
            "change": round(change, 4),
            "change_pct": round(change_pct, 4),
            "high": self.get_price(ticker, "PX_HIGH"),
            "low": self.get_price(ticker, "PX_LOW"),
        }
    
    def get_intraday_history(self, ticker: str) -> pd.DataFrame:
        """Get intraday price history."""
        history = self._price_history.get(ticker, [])
        if not history:
            return pd.DataFrame(columns=["timestamp", "price"])
        
        df = pd.DataFrame(history, columns=["timestamp", "price"])
        return df
    
    def reset_session(self):
        """Reset to simulate a new trading day."""
        self._session_start = datetime.now()
        self._initialize_session()


class BloombergClient:
    """
    Bloomberg API Client with mock data fallback.
    
    When Bloomberg API is not available, uses PriceSimulator for
    realistic, consistent mock data.
    """
    
    # Singleton price simulator for consistency across instances
    _simulator: PriceSimulator = None
    
    def __init__(self, use_mock: bool = True):
        """
        Initialize Bloomberg client.
        
        Args:
            use_mock: If True, use mock data instead of real Bloomberg API
        """
        self.use_mock = use_mock
        self.connected = False
        
        # Initialize shared simulator
        if BloombergClient._simulator is None:
            BloombergClient._simulator = PriceSimulator()
        
        self.simulator = BloombergClient._simulator
        
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
        """Get current price for a ticker."""
        if self.use_mock:
            return self.simulator.get_price(ticker, field)
        raise NotImplementedError("Real Bloomberg API not implemented")
    
    def get_price_with_change(self, ticker: str) -> Dict[str, float]:
        """Get current price with change from open."""
        if self.use_mock:
            return self.simulator.get_price_change(ticker)
        raise NotImplementedError("Real Bloomberg API not implemented")
    
    def get_prices(self, tickers: List[str], fields: List[str] = None) -> pd.DataFrame:
        """Get current prices for multiple tickers."""
        if fields is None:
            fields = ["PX_LAST", "PX_BID", "PX_ASK", "PX_VOLUME"]
        
        if self.use_mock:
            data = {}
            for ticker in tickers:
                row = {}
                for field in fields:
                    if field == "PX_VOLUME":
                        row[field] = np.random.randint(10000, 100000)
                    else:
                        row[field] = self.simulator.get_price(ticker, field)
                data[ticker] = row
            return pd.DataFrame(data).T
        
        raise NotImplementedError("Real Bloomberg API not implemented")
    
    def get_historical(
        self,
        ticker: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime] = None,
        fields: List[str] = None,
        frequency: str = "DAILY"
    ) -> pd.DataFrame:
        """Get historical OHLCV data."""
        if fields is None:
            fields = ["PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST", "PX_VOLUME"]
        
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if end_date is None:
            end_date = datetime.now()
        elif isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        if self.use_mock:
            return self._generate_historical(ticker, start_date, end_date, fields, frequency)
        
        raise NotImplementedError("Real Bloomberg API not implemented")
    
    def _generate_historical(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        fields: List[str],
        frequency: str
    ) -> pd.DataFrame:
        """Generate historical data that connects to current price."""
        # Get current price to anchor the series
        current_price = self.simulator.get_price(ticker)
        
        # Generate date range
        if frequency == "DAILY":
            dates = pd.date_range(start=start_date, end=end_date, freq='B')
        elif frequency == "WEEKLY":
            dates = pd.date_range(start=start_date, end=end_date, freq='W')
        else:
            dates = pd.date_range(start=start_date, end=end_date, freq='M')
        
        n = len(dates)
        if n == 0:
            return pd.DataFrame()
        
        # Use ticker hash for reproducible but unique series per ticker
        seed = int(hashlib.md5(ticker.encode()).hexdigest()[:8], 16) % (2**32)
        rng = np.random.RandomState(seed)
        
        # Generate returns working backwards from current price
        daily_vol = 0.02  # 2% daily volatility
        returns = rng.normal(0.0001, daily_vol, n)
        
        # Add some autocorrelation (momentum)
        for i in range(1, n):
            returns[i] += 0.1 * returns[i-1]
        
        # Build price series backwards from current price
        prices = np.zeros(n)
        prices[-1] = current_price
        for i in range(n - 2, -1, -1):
            prices[i] = prices[i + 1] / (1 + returns[i + 1])
        
        # Generate OHLC from close
        high_mult = 1 + np.abs(rng.normal(0.005, 0.003, n))
        low_mult = 1 - np.abs(rng.normal(0.005, 0.003, n))
        open_noise = rng.normal(0, 0.003, n)
        
        data = {
            "date": dates,
            "PX_LAST": prices,
            "PX_OPEN": prices * (1 + open_noise),
            "PX_HIGH": prices * high_mult,
            "PX_LOW": prices * low_mult,
            "PX_VOLUME": rng.randint(50000, 200000, n),
        }
        
        # Ensure OHLC consistency
        df = pd.DataFrame(data)
        df["PX_HIGH"] = df[["PX_OPEN", "PX_HIGH", "PX_LAST"]].max(axis=1)
        df["PX_LOW"] = df[["PX_OPEN", "PX_LOW", "PX_LAST"]].min(axis=1)
        
        df.set_index("date", inplace=True)
        
        available_fields = [f for f in fields if f in df.columns]
        return df[available_fields]
    
    def get_curve(self, commodity: str = "wti", num_months: int = 12) -> pd.DataFrame:
        """Get futures curve data."""
        ticker_prefix = "CL" if commodity.lower() == "wti" else "CO"
        
        data = []
        today = datetime.now()
        
        for i in range(1, num_months + 1):
            ticker = f"{ticker_prefix}{i} Comdty"
            price_data = self.simulator.get_price_change(ticker)
            expiry = today + timedelta(days=30 * i)
            
            data.append({
                "month": i,
                "ticker": ticker,
                "price": price_data["current"],
                "change": price_data["change"],
                "change_pct": price_data["change_pct"],
                "expiry": expiry,
                "days_to_expiry": (expiry - today).days
            })
        
        return pd.DataFrame(data)
    
    def get_reference_data(self, ticker: str, fields: List[str]) -> Dict:
        """Get reference data for a ticker."""
        ref = {
            "NAME": f"Oil Futures {ticker[:2]}",
            "TICKER": ticker,
            "EXCH_CODE": "NYMEX" if ticker.startswith("CL") else "ICE",
            "CRNCY": "USD",
            "FUT_CONT_SIZE": 1000,
            "FUT_TICK_SIZE": 0.01,
        }
        return {f: ref.get(f, None) for f in fields}
    
    def get_intraday_prices(self, ticker: str) -> pd.DataFrame:
        """Get intraday price history."""
        if self.use_mock:
            return self.simulator.get_intraday_history(ticker)
        raise NotImplementedError("Real Bloomberg API not implemented")


class MockBloombergData:
    """Generate comprehensive mock market data for demonstration."""
    
    @staticmethod
    def generate_eia_inventory_data(periods: int = 52) -> pd.DataFrame:
        """Generate mock EIA inventory data."""
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='W-WED')
        
        # Base inventory level around 430 million barrels
        base = 430.0
        
        # Add seasonality (higher in spring, lower in fall)
        seasonal = 15 * np.sin(2 * np.pi * np.arange(periods) / 52 - np.pi/2)
        
        # Add trend and noise
        rng = np.random.RandomState(42)
        trend = np.linspace(-10, 5, periods)
        noise = rng.normal(0, 3, periods)
        
        inventory = base + seasonal + trend + noise
        
        # Calculate week-over-week change
        change = np.diff(inventory, prepend=inventory[0])
        
        # Expectations (analyst estimates) with some error
        expectation = change + rng.normal(0, 1.5, periods)
        
        return pd.DataFrame({
            "date": dates,
            "inventory_mmb": inventory,
            "change_mmb": change,
            "expectation_mmb": expectation,
            "surprise_mmb": change - expectation,
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
