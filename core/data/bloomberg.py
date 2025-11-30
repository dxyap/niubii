"""
Bloomberg API Client
====================
Wrapper for Bloomberg Desktop API (BLPAPI) with mock data fallback.

Supports both real Bloomberg Terminal connection and realistic mock data
for development/demo environments.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union, Tuple
import logging
import hashlib
import os

logger = logging.getLogger(__name__)


# =============================================================================
# TICKER MAPPING AND VALIDATION
# =============================================================================

class TickerMapper:
    """
    Comprehensive ticker mapping and validation for Bloomberg tickers.
    Ensures consistent ticker usage across the application.
    """
    
    # Standard Bloomberg ticker formats
    TICKER_FORMATS = {
        # Crude Oil Futures
        "wti": "CL{n} Comdty",        # WTI Crude (NYMEX)
        "brent": "CO{n} Comdty",       # Brent Crude (ICE)
        
        # Refined Products
        "rbob": "XB{n} Comdty",        # RBOB Gasoline (NYMEX)
        "heating_oil": "HO{n} Comdty", # Heating Oil (NYMEX)
        "gasoil": "QS{n} Comdty",      # Gasoil (ICE)
        
        # Natural Gas
        "natgas": "NG{n} Comdty",      # Natural Gas (NYMEX)
    }
    
    # Generic month codes (for specific contract months)
    MONTH_CODES = {
        1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
        7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'
    }
    
    # Bloomberg field mappings
    FIELDS = {
        "last": "PX_LAST",
        "bid": "PX_BID",
        "ask": "PX_ASK",
        "open": "PX_OPEN",
        "high": "PX_HIGH",
        "low": "PX_LOW",
        "close": "PX_LAST",
        "volume": "PX_VOLUME",
        "open_interest": "OPEN_INT",
        "vwap": "PX_VWAP",
        "settlement": "PX_SETTLE",
    }
    
    # Contract multipliers (for position sizing)
    CONTRACT_MULTIPLIERS = {
        "CL": 1000,    # 1,000 barrels
        "CO": 1000,    # 1,000 barrels
        "XB": 42000,   # 42,000 gallons
        "HO": 42000,   # 42,000 gallons
        "QS": 100,     # 100 metric tonnes
        "NG": 10000,   # 10,000 MMBtu
    }
    
    # Exchange mappings
    EXCHANGES = {
        "CL": "NYMEX",
        "CO": "ICE",
        "XB": "NYMEX",
        "HO": "NYMEX",
        "QS": "ICE",
        "NG": "NYMEX",
    }
    
    @classmethod
    def get_front_month_ticker(cls, commodity: str) -> str:
        """Get front month ticker for a commodity."""
        fmt = cls.TICKER_FORMATS.get(commodity.lower())
        if fmt:
            return fmt.format(n=1)
        raise ValueError(f"Unknown commodity: {commodity}")
    
    @classmethod
    def get_nth_month_ticker(cls, commodity: str, n: int) -> str:
        """Get nth month ticker for a commodity (1-indexed)."""
        fmt = cls.TICKER_FORMATS.get(commodity.lower())
        if fmt:
            return fmt.format(n=n)
        raise ValueError(f"Unknown commodity: {commodity}")
    
    @classmethod
    def get_specific_month_ticker(cls, commodity: str, month: int, year: int) -> str:
        """Get ticker for specific contract month/year."""
        base = commodity.upper()[:2] if commodity.upper()[:2] in cls.CONTRACT_MULTIPLIERS else commodity[:2].upper()
        month_code = cls.MONTH_CODES.get(month)
        if month_code is None:
            raise ValueError(f"Invalid month: {month}")
        year_digit = year % 10  # Last digit of year
        return f"{base}{month_code}{year_digit} Comdty"
    
    @classmethod
    def parse_ticker(cls, ticker: str) -> Dict[str, str]:
        """Parse a Bloomberg ticker into its components."""
        if not ticker.endswith(" Comdty"):
            return {"ticker": ticker, "type": "unknown"}
        
        base = ticker.replace(" Comdty", "")
        
        # Try to determine if it's a generic ticker (e.g., CL1, CL12) or specific (e.g., CLF5)
        # Generic format: CL + number (1-12)
        # Specific format: CL + month_code + year_digit (e.g., CLF5, CLZ25)
        
        # First try to identify the commodity prefix (2 characters)
        if len(base) < 2:
            return {"ticker": ticker, "type": "unknown"}
        
        commodity = base[:2]
        remainder = base[2:]
        
        if not remainder:
            return {"ticker": ticker, "type": "unknown"}
        
        # Check if remainder is purely numeric (generic ticker)
        if remainder.isdigit():
            month_num = int(remainder)
            return {
                "ticker": ticker,
                "commodity": commodity,
                "type": "generic",
                "month_number": month_num,
                "exchange": cls.EXCHANGES.get(commodity, "Unknown"),
                "multiplier": cls.CONTRACT_MULTIPLIERS.get(commodity, 1000),
            }
        
        # Check if it's a specific contract (letter + digit(s))
        # Format: month_code (letter) + year (1 or 2 digits)
        if len(remainder) >= 2 and remainder[0].isalpha() and remainder[1:].isdigit():
            month_code = remainder[0]
            year_digit = remainder[1:]
            
            # Reverse lookup month code
            month = None
            for m, code in cls.MONTH_CODES.items():
                if code == month_code:
                    month = m
                    break
            
            if month is not None:
                return {
                    "ticker": ticker,
                    "commodity": commodity,
                    "type": "specific",
                    "month_code": month_code,
                    "month": month,
                    "year_digit": year_digit,
                    "exchange": cls.EXCHANGES.get(commodity, "Unknown"),
                    "multiplier": cls.CONTRACT_MULTIPLIERS.get(commodity, 1000),
                }
        
        return {"ticker": ticker, "type": "unknown"}
    
    @classmethod
    def validate_ticker(cls, ticker: str) -> Tuple[bool, str]:
        """Validate a Bloomberg ticker format."""
        if not ticker:
            return False, "Empty ticker"
        
        if not ticker.endswith(" Comdty"):
            return False, "Missing ' Comdty' suffix"
        
        parsed = cls.parse_ticker(ticker)
        
        if parsed["type"] == "unknown":
            return False, "Unknown ticker format"
        
        commodity = parsed.get("commodity", "")
        if commodity not in cls.CONTRACT_MULTIPLIERS:
            return False, f"Unknown commodity: {commodity}"
        
        return True, "Valid"
    
    @classmethod
    def get_field(cls, field_name: str) -> str:
        """Get Bloomberg field name from common name."""
        return cls.FIELDS.get(field_name.lower(), field_name)
    
    @classmethod
    def get_multiplier(cls, ticker: str) -> int:
        """Get contract multiplier for a ticker."""
        parsed = cls.parse_ticker(ticker)
        return parsed.get("multiplier", 1000)


# =============================================================================
# PRICE SIMULATOR
# =============================================================================

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
            "XB2 Comdty": 2.19,    # RBOB 2nd Month
            "HO1 Comdty": 2.52,    # Heating Oil ($/gal)
            "HO2 Comdty": 2.53,    # Heating Oil 2nd Month
            "QS1 Comdty": 680.50,  # Gasoil ($/tonne)
            "NG1 Comdty": 3.25,    # Natural Gas
        }
        
        # Generate curve prices with realistic term structure
        for i in range(3, 13):
            # WTI: slight contango
            self._reference_prices[f"CL{i} Comdty"] = 72.50 + (i - 1) * 0.12
            # Brent: slight contango
            self._reference_prices[f"CO{i} Comdty"] = 77.20 + (i - 1) * 0.10
            # Products
            self._reference_prices[f"XB{i} Comdty"] = 2.18 + (i - 1) * 0.005
            self._reference_prices[f"HO{i} Comdty"] = 2.52 + (i - 1) * 0.005
        
        # Current simulated prices (will drift from reference)
        self._current_prices: Dict[str, float] = {}
        self._last_update: Dict[str, datetime] = {}
        self._price_history: Dict[str, List[tuple]] = {}
        
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
        """Get simulated price with realistic tick-by-tick movement."""
        if ticker not in self._current_prices:
            # Try to infer base price from similar tickers
            base = self._infer_base_price(ticker)
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
        
        # Apply bid/ask spread based on product
        spread_pct = self._get_spread_pct(ticker)
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
        elif field == "PX_SETTLE":
            return round(current, 4)
        
        return round(current, 4)
    
    def _infer_base_price(self, ticker: str) -> float:
        """Infer base price for unknown ticker from similar instruments."""
        parsed = TickerMapper.parse_ticker(ticker)
        commodity = parsed.get("commodity", "CL")
        month_num = parsed.get("month_number", 1)
        
        # Base prices by commodity
        base_prices = {
            "CL": 72.50,
            "CO": 77.20,
            "XB": 2.18,
            "HO": 2.52,
            "QS": 680.50,
            "NG": 3.25,
        }
        
        base = base_prices.get(commodity, 70.0)
        
        # Apply term structure (contango)
        if month_num > 1:
            base += (month_num - 1) * 0.10
        
        return base
    
    def _get_spread_pct(self, ticker: str) -> float:
        """Get bid/ask spread percentage based on instrument."""
        parsed = TickerMapper.parse_ticker(ticker)
        commodity = parsed.get("commodity", "CL")
        
        # Spread varies by liquidity
        spreads = {
            "CL": 0.0002,  # 2 bps - very liquid
            "CO": 0.0003,  # 3 bps - liquid
            "XB": 0.0005,  # 5 bps - moderate
            "HO": 0.0005,  # 5 bps - moderate
            "QS": 0.001,   # 10 bps - less liquid
            "NG": 0.0004,  # 4 bps - liquid
        }
        
        return spreads.get(commodity, 0.0005)
    
    def _update_price(self, ticker: str, elapsed_seconds: float):
        """Update price using realistic market microstructure model."""
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
        
        # New price
        new_price = current + mean_reversion + innovation
        
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


# =============================================================================
# BLOOMBERG CLIENT
# =============================================================================

class BloombergClient:
    """
    Bloomberg API Client with mock data fallback.
    
    Attempts to connect to real Bloomberg Terminal via BLPAPI.
    Falls back to PriceSimulator for development/demo environments.
    """
    
    # Singleton price simulator for consistency across instances
    _simulator: PriceSimulator = None
    
    def __init__(self, use_mock: bool = None):
        """
        Initialize Bloomberg client.
        
        Args:
            use_mock: If True, use mock data. If None, auto-detect based on
                      environment and Bloomberg availability.
        """
        # Auto-detect mode from environment if not specified
        if use_mock is None:
            use_mock = os.environ.get("BLOOMBERG_USE_MOCK", "true").lower() == "true"
        
        self.use_mock = use_mock
        self.connected = False
        self._session = None
        self._ref_data_service = None
        
        # Initialize shared simulator
        if BloombergClient._simulator is None:
            BloombergClient._simulator = PriceSimulator()
        
        self.simulator = BloombergClient._simulator
        
        if not use_mock:
            self._connect()
    
    def _connect(self) -> bool:
        """Attempt to connect to Bloomberg API."""
        try:
            import blpapi
            
            # Session options from environment
            host = os.environ.get("BLOOMBERG_HOST", "localhost")
            port = int(os.environ.get("BLOOMBERG_PORT", "8194"))
            
            session_options = blpapi.SessionOptions()
            session_options.setServerHost(host)
            session_options.setServerPort(port)
            
            self._session = blpapi.Session(session_options)
            
            if not self._session.start():
                logger.warning("Failed to start Bloomberg session")
                self.use_mock = True
                return False
            
            if not self._session.openService("//blp/refdata"):
                logger.warning("Failed to open Bloomberg reference data service")
                self.use_mock = True
                return False
            
            self._ref_data_service = self._session.getService("//blp/refdata")
            self.connected = True
            logger.info("Successfully connected to Bloomberg API")
            return True
            
        except ImportError:
            logger.info("blpapi not installed, using mock data")
            self.use_mock = True
            return False
        except Exception as e:
            logger.warning(f"Could not connect to Bloomberg: {e}")
            self.use_mock = True
            return False
    
    def _ensure_connection(self) -> bool:
        """Ensure we have a valid connection."""
        if self.use_mock:
            return True
        
        if not self.connected:
            return self._connect()
        
        return True
    
    def get_price(self, ticker: str, field: str = "PX_LAST") -> float:
        """Get current price for a ticker."""
        # Validate ticker
        valid, msg = TickerMapper.validate_ticker(ticker)
        if not valid:
            logger.warning(f"Invalid ticker {ticker}: {msg}")
        
        if self.use_mock:
            return self.simulator.get_price(ticker, field)
        
        return self._get_bloomberg_price(ticker, field)
    
    def _get_bloomberg_price(self, ticker: str, field: str) -> float:
        """Get price from real Bloomberg API."""
        try:
            import blpapi
            
            request = self._ref_data_service.createRequest("ReferenceDataRequest")
            request.append("securities", ticker)
            request.append("fields", field)
            
            self._session.sendRequest(request)
            
            while True:
                event = self._session.nextEvent()
                for msg in event:
                    if msg.hasElement("securityData"):
                        security_data = msg.getElement("securityData")
                        if security_data.numValues() > 0:
                            sec = security_data.getValue(0)
                            field_data = sec.getElement("fieldData")
                            if field_data.hasElement(field):
                                return field_data.getElementAsFloat(field)
                
                if event.eventType() == blpapi.Event.RESPONSE:
                    break
            
            # Fallback to mock if no data
            logger.warning(f"No data from Bloomberg for {ticker}, using mock")
            return self.simulator.get_price(ticker, field)
            
        except Exception as e:
            logger.error(f"Bloomberg API error: {e}")
            return self.simulator.get_price(ticker, field)
    
    def get_price_with_change(self, ticker: str) -> Dict[str, float]:
        """Get current price with change from open."""
        if self.use_mock:
            return self.simulator.get_price_change(ticker)
        
        # Get multiple fields from Bloomberg
        fields = ["PX_LAST", "PX_OPEN", "PX_HIGH", "PX_LOW"]
        data = self.get_prices([ticker], fields)
        
        if data is not None and not data.empty:
            row = data.iloc[0]
            current = row.get("PX_LAST", 0)
            open_price = row.get("PX_OPEN", current)
            
            change = current - open_price
            change_pct = (change / open_price * 100) if open_price != 0 else 0
            
            return {
                "current": current,
                "open": open_price,
                "change": round(change, 4),
                "change_pct": round(change_pct, 4),
                "high": row.get("PX_HIGH", current),
                "low": row.get("PX_LOW", current),
            }
        
        return self.simulator.get_price_change(ticker)
    
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
        
        return self._get_bloomberg_prices(tickers, fields)
    
    def _get_bloomberg_prices(self, tickers: List[str], fields: List[str]) -> pd.DataFrame:
        """Get prices from real Bloomberg API."""
        try:
            import blpapi
            
            request = self._ref_data_service.createRequest("ReferenceDataRequest")
            
            for ticker in tickers:
                request.append("securities", ticker)
            for field in fields:
                request.append("fields", field)
            
            self._session.sendRequest(request)
            
            data = {}
            while True:
                event = self._session.nextEvent()
                for msg in event:
                    if msg.hasElement("securityData"):
                        security_data = msg.getElement("securityData")
                        for i in range(security_data.numValues()):
                            sec = security_data.getValue(i)
                            ticker = sec.getElementAsString("security")
                            field_data = sec.getElement("fieldData")
                            
                            row = {}
                            for field in fields:
                                if field_data.hasElement(field):
                                    try:
                                        row[field] = field_data.getElementAsFloat(field)
                                    except:
                                        row[field] = None
                            
                            data[ticker] = row
                
                if event.eventType() == blpapi.Event.RESPONSE:
                    break
            
            return pd.DataFrame(data).T
            
        except Exception as e:
            logger.error(f"Bloomberg API error: {e}")
            # Fallback to mock
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
        
        return self._get_bloomberg_historical(ticker, start_date, end_date, fields, frequency)
    
    def _get_bloomberg_historical(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        fields: List[str],
        frequency: str
    ) -> pd.DataFrame:
        """Get historical data from real Bloomberg API."""
        try:
            import blpapi
            
            request = self._ref_data_service.createRequest("HistoricalDataRequest")
            request.append("securities", ticker)
            
            for field in fields:
                request.append("fields", field)
            
            request.set("startDate", start_date.strftime("%Y%m%d"))
            request.set("endDate", end_date.strftime("%Y%m%d"))
            request.set("periodicitySelection", frequency)
            
            self._session.sendRequest(request)
            
            data = []
            while True:
                event = self._session.nextEvent()
                for msg in event:
                    if msg.hasElement("securityData"):
                        security_data = msg.getElement("securityData")
                        if security_data.hasElement("fieldData"):
                            field_data = security_data.getElement("fieldData")
                            
                            for i in range(field_data.numValues()):
                                point = field_data.getValue(i)
                                row = {"date": point.getElementAsDatetime("date")}
                                
                                for field in fields:
                                    if point.hasElement(field):
                                        try:
                                            row[field] = point.getElementAsFloat(field)
                                        except:
                                            row[field] = None
                                
                                data.append(row)
                
                if event.eventType() == blpapi.Event.RESPONSE:
                    break
            
            if data:
                df = pd.DataFrame(data)
                df.set_index("date", inplace=True)
                return df
            
            # Fallback to mock
            return self._generate_historical(ticker, start_date, end_date, fields, frequency)
            
        except Exception as e:
            logger.error(f"Bloomberg historical API error: {e}")
            return self._generate_historical(ticker, start_date, end_date, fields, frequency)
    
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
        ticker_prefix = "CL" if commodity.lower() == "wti" else "CO" if commodity.lower() == "brent" else commodity.upper()[:2]
        
        data = []
        today = datetime.now()
        
        for i in range(1, num_months + 1):
            ticker = f"{ticker_prefix}{i} Comdty"
            price_data = self.simulator.get_price_change(ticker) if self.use_mock else self.get_price_with_change(ticker)
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
        parsed = TickerMapper.parse_ticker(ticker)
        commodity = parsed.get("commodity", "CL")
        
        ref = {
            "NAME": f"Oil Futures {ticker[:2]}",
            "TICKER": ticker,
            "EXCH_CODE": parsed.get("exchange", "NYMEX"),
            "CRNCY": "USD",
            "FUT_CONT_SIZE": parsed.get("multiplier", 1000),
            "FUT_TICK_SIZE": 0.01,
        }
        return {f: ref.get(f, None) for f in fields}
    
    def get_intraday_prices(self, ticker: str) -> pd.DataFrame:
        """Get intraday price history."""
        if self.use_mock:
            return self.simulator.get_intraday_history(ticker)
        
        # For real Bloomberg, would use subscription or intraday bars
        # For now, fallback to simulator
        return self.simulator.get_intraday_history(ticker)
    
    def disconnect(self):
        """Disconnect from Bloomberg API."""
        if self._session:
            try:
                self._session.stop()
            except:
                pass
            self._session = None
            self.connected = False


# =============================================================================
# MOCK DATA GENERATORS
# =============================================================================

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
