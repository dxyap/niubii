"""
Bloomberg API Client
====================
Wrapper for Bloomberg Desktop API (BLPAPI).

Requires a Bloomberg Terminal connection for live data.
Raises DataUnavailableError when data cannot be retrieved.
"""

import builtins
import contextlib
import logging
import os
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)


class DataUnavailableError(Exception):
    """Raised when data cannot be retrieved from the data source."""
    pass


class BloombergConnectionError(Exception):
    """Raised when Bloomberg Terminal connection fails."""
    pass


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
        "wti": "CL{n} Comdty",         # WTI Crude (NYMEX)
        "wti_ice": "ENA{n} Comdty",    # WTI Crude (ICE)
        "brent": "CO{n} Comdty",       # Brent Crude (ICE)
        "dubai": "DAT{n} Comdty",      # Dubai Crude Swap (ICE)

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
        "CL": 1000,    # 1,000 barrels (WTI NYMEX)
        "ENA": 1000,   # 1,000 barrels (WTI ICE)
        "CO": 1000,    # 1,000 barrels (Brent ICE)
        "DAT": 1000,   # 1,000 barrels (Dubai Crude Swap)
        "XB": 42000,   # 42,000 gallons
        "HO": 42000,   # 42,000 gallons
        "QS": 100,     # 100 metric tonnes
        "NG": 10000,   # 10,000 MMBtu
    }

    # Exchange mappings
    EXCHANGES = {
        "CL": "NYMEX",
        "ENA": "ICE",    # WTI ICE
        "CO": "ICE",
        "DAT": "ICE",    # Dubai Crude Swap
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
    def parse_ticker(cls, ticker: str) -> dict[str, str]:
        """Parse a Bloomberg ticker into its components."""
        if not ticker.endswith(" Comdty"):
            return {"ticker": ticker, "type": "unknown"}

        base = ticker.replace(" Comdty", "")

        # Try to determine if it's a generic ticker (e.g., CL1, CL12) or specific (e.g., CLF5)
        # Generic format: CL + number (1-12)
        # Specific format: CL + month_code + year_digit (e.g., CLF5, CLZ25)

        if len(base) < 2:
            return {"ticker": ticker, "type": "unknown"}

        # Check for special prefixes first (DAT for Dubai, T for ICE WTI)
        commodity = None
        remainder = None

        # Try 3-char prefix first (e.g., DAT)
        if len(base) >= 4 and base[:3] in cls.SPECIAL_PREFIXES:
            commodity = base[:3]
            remainder = base[3:]
        # Then try 1-char prefix (e.g., T for ICE WTI)
        elif len(base) >= 2 and base[0] in cls.SPECIAL_PREFIXES:
            commodity = base[0]
            remainder = base[1:]
        # Default: 2-char prefix
        else:
            commodity = base[:2]
            remainder = base[2:]

        if not remainder:
            return {"ticker": ticker, "type": "unknown"}

        # Get exchange and multiplier from appropriate source
        if commodity in cls.SPECIAL_PREFIXES:
            exchange = cls.SPECIAL_PREFIXES[commodity]["exchange"]
            multiplier = cls.SPECIAL_PREFIXES[commodity]["multiplier"]
        else:
            exchange = cls.EXCHANGES.get(commodity, "Unknown")
            multiplier = cls.CONTRACT_MULTIPLIERS.get(commodity, 1000)

        # Check if remainder is purely numeric (generic ticker)
        if remainder.isdigit():
            month_num = int(remainder)
            return {
                "ticker": ticker,
                "commodity": commodity,
                "type": "generic",
                "month_number": month_num,
                "exchange": exchange,
                "multiplier": multiplier,
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
                    "exchange": exchange,
                    "multiplier": multiplier,
                }

        return {"ticker": ticker, "type": "unknown"}

    # Known index tickers (non-standard format)
    # Dubai uses M2 swap to avoid BALMO (Balance of Month) contract
    INDEX_TICKERS = {
        "PGCR2MOE Index": {"name": "Dubai Crude Swap M2 (Platts)", "multiplier": 1000},
        "PGCR3MOE Index": {"name": "Dubai Crude Swap M3 (Platts)", "multiplier": 1000},
    }

    # Special ticker patterns that don't follow standard 2-char prefix
    # DAT = Dubai Average Crude, ENA = ICE WTI
    SPECIAL_PREFIXES = {
        "DAT": {"name": "Dubai Crude Swap", "multiplier": 1000, "exchange": "ICE"},
        "ENA": {"name": "WTI Crude (ICE)", "multiplier": 1000, "exchange": "ICE"},
    }

    @classmethod
    def validate_ticker(cls, ticker: str) -> tuple[bool, str]:
        """Validate a Bloomberg ticker format."""
        if not ticker:
            return False, "Empty ticker"

        # Check for known index tickers
        if ticker in cls.INDEX_TICKERS:
            return True, "Valid (Index)"

        if not ticker.endswith(" Comdty"):
            return False, "Missing ' Comdty' suffix"

        parsed = cls.parse_ticker(ticker)

        if parsed["type"] == "unknown":
            return False, "Unknown ticker format"

        commodity = parsed.get("commodity", "")
        # Check both regular commodities and special prefixes
        if commodity not in cls.CONTRACT_MULTIPLIERS and commodity not in cls.SPECIAL_PREFIXES:
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
# BLOOMBERG CLIENT
# =============================================================================

class BloombergClient:
    """
    Bloomberg API Client.

    Connects to real Bloomberg Terminal via BLPAPI.
    Raises BloombergConnectionError if connection fails.
    """

    def __init__(self):
        """Initialize Bloomberg client and connect to Bloomberg Terminal."""
        self.connected = False
        self._session = None
        self._ref_data_service = None
        self._connection_error = None

        # Attempt to connect to Bloomberg
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
                self._connection_error = "Failed to start Bloomberg session. Is Bloomberg Terminal running?"
                logger.error(self._connection_error)
                return False

            if not self._session.openService("//blp/refdata"):
                self._connection_error = "Failed to open Bloomberg reference data service"
                logger.error(self._connection_error)
                return False

            self._ref_data_service = self._session.getService("//blp/refdata")
            self.connected = True
            self._connection_error = None
            logger.info("Successfully connected to Bloomberg API")
            return True

        except ImportError:
            self._connection_error = "Bloomberg API (blpapi) not installed. Install with: pip install blpapi"
            logger.error(self._connection_error)
            return False
        except Exception as e:
            self._connection_error = f"Could not connect to Bloomberg: {e}"
            logger.error(self._connection_error)
            return False

    def _ensure_connection(self) -> bool:
        """Ensure we have a valid connection. Raises if not connected."""
        if not self.connected:
            # Try to reconnect
            if not self._connect():
                raise BloombergConnectionError(
                    self._connection_error or "Bloomberg Terminal not connected"
                )

        return True

    def get_connection_error(self) -> str | None:
        """Get the connection error message if any."""
        return self._connection_error

    def get_price(self, ticker: str, field: str = "PX_LAST") -> float:
        """
        Get current price for a ticker.

        Raises:
            DataUnavailableError: If price data cannot be retrieved
            BloombergConnectionError: If not connected to Bloomberg
        """
        # Validate ticker
        valid, msg = TickerMapper.validate_ticker(ticker)
        if not valid:
            raise DataUnavailableError(f"Invalid ticker {ticker}: {msg}")

        self._ensure_connection()
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

            raise DataUnavailableError(f"No data available from Bloomberg for {ticker}")

        except DataUnavailableError:
            raise
        except Exception as e:
            raise DataUnavailableError(f"Bloomberg API error for {ticker}: {e}")

    def get_price_with_change(self, ticker: str) -> dict[str, float]:
        """
        Get current price with change from open.

        Raises:
            DataUnavailableError: If price data cannot be retrieved
        """
        self._ensure_connection()

        # Get multiple fields from Bloomberg
        fields = ["PX_LAST", "PX_OPEN", "PX_HIGH", "PX_LOW"]
        data = self.get_prices([ticker], fields)

        if data is not None and not data.empty:
            row = data.iloc[0]
            current = row.get("PX_LAST")

            if current is None or pd.isna(current):
                raise DataUnavailableError(f"No price data available for {ticker}")

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

        raise DataUnavailableError(f"No price data available for {ticker}")

    def get_prices(self, tickers: list[str], fields: list[str] = None) -> pd.DataFrame:
        """
        Get current prices for multiple tickers.

        Raises:
            DataUnavailableError: If price data cannot be retrieved
        """
        if fields is None:
            fields = ["PX_LAST", "PX_BID", "PX_ASK", "PX_VOLUME"]

        self._ensure_connection()
        return self._get_bloomberg_prices(tickers, fields)

    def _get_bloomberg_prices(self, tickers: list[str], fields: list[str]) -> pd.DataFrame:
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
                                    except Exception:
                                        row[field] = None

                            data[ticker] = row

                if event.eventType() == blpapi.Event.RESPONSE:
                    break

            if not data:
                raise DataUnavailableError(f"No data returned from Bloomberg for tickers: {tickers}")

            return pd.DataFrame(data).T

        except DataUnavailableError:
            raise
        except Exception as e:
            raise DataUnavailableError(f"Bloomberg API error: {e}")

    def get_historical(
        self,
        ticker: str,
        start_date: str | datetime,
        end_date: str | datetime = None,
        fields: list[str] = None,
        frequency: str = "DAILY"
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data.

        Raises:
            DataUnavailableError: If historical data cannot be retrieved
        """
        if fields is None:
            fields = ["PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST", "PX_VOLUME", "OPEN_INT"]

        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if end_date is None:
            end_date = datetime.now()
        elif isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        self._ensure_connection()
        return self._get_bloomberg_historical(ticker, start_date, end_date, fields, frequency)

    def _get_bloomberg_historical(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        fields: list[str],
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
                                        except Exception:
                                            row[field] = None

                                data.append(row)

                if event.eventType() == blpapi.Event.RESPONSE:
                    break

            if data:
                df = pd.DataFrame(data)
                df.set_index("date", inplace=True)
                return df

            raise DataUnavailableError(f"No historical data available for {ticker}")

        except DataUnavailableError:
            raise
        except Exception as e:
            raise DataUnavailableError(f"Bloomberg historical API error: {e}")

    def get_curve(self, commodity: str = "wti", num_months: int = 12) -> pd.DataFrame:
        """
        Get futures curve data (batch optimized).

        Fetches all curve points in a single batch API call for efficiency.
        Includes absolute contract month labels in MMM-YY format (e.g., "Jan-25").

        Contract Month Conventions:
        - WTI (CL/ENA): Front month is approximately 1 month ahead of current date
        - Brent (CO): Front month is approximately 2 months ahead (cash-settled contract)
        - Dubai (DAT): Uses M2 swap to avoid BALMO, so starts 2 months ahead

        Raises:
            DataUnavailableError: If curve data cannot be retrieved
        """
        commodity_lower = commodity.lower()

        # Determine ticker prefix and starting month index based on commodity
        if commodity_lower == "wti":
            ticker_prefix = "CL"
            start_month_index = 1
            # WTI front month offset: 1 month ahead (or 2 if past expiry)
            front_month_offset = 1
        elif commodity_lower == "wti_ice":
            ticker_prefix = "ENA"
            start_month_index = 1
            front_month_offset = 1
        elif commodity_lower == "brent":
            ticker_prefix = "CO"
            start_month_index = 1
            # Brent front month offset: 2 months ahead (cash-settled contract)
            front_month_offset = 2
        elif commodity_lower == "dubai":
            ticker_prefix = "DAT"
            # Dubai uses M2 swap to avoid BALMO, so start at index 2
            start_month_index = 2
            # Dubai front month offset: 2 months ahead (same as Brent)
            front_month_offset = 2
        else:
            ticker_prefix = commodity.upper()[:2]
            start_month_index = 1
            front_month_offset = 1

        # Build list of tickers for the curve
        tickers = [f"{ticker_prefix}{i} Comdty" for i in range(start_month_index, start_month_index + num_months)]

        # Batch fetch all curve prices in a single API call
        fields = ["PX_LAST", "PX_OPEN", "PX_HIGH", "PX_LOW", "OPEN_INT"]
        try:
            prices_df = self.get_prices(tickers, fields)
        except Exception as e:
            raise DataUnavailableError(f"Failed to fetch curve data for {commodity}: {e}")

        if prices_df is None or prices_df.empty:
            raise DataUnavailableError(f"No curve data available for {commodity}")

        # Build curve DataFrame from batch results
        data = []
        today = datetime.now()

        # Calculate the front month contract month based on commodity-specific conventions
        # Oil futures typically expire around the 20th of the month before delivery
        current_month = today.month
        current_year = today.year

        # Calculate front month based on:
        # 1. Commodity-specific offset (Brent/Dubai = 2 months, WTI = 1 month)
        # 2. Position within the month (if past ~20th, add 1 more month)
        if today.day <= 20:
            front_month = current_month + front_month_offset
            front_year = current_year
        else:
            front_month = current_month + front_month_offset + 1
            front_year = current_year

        # Adjust for year rollover
        while front_month > 12:
            front_month -= 12
            front_year += 1

        for i, ticker in enumerate(tickers):
            if ticker not in prices_df.index:
                continue

            row = prices_df.loc[ticker]
            current = row.get("PX_LAST", 0)
            open_price = row.get("PX_OPEN", current)

            if current is None or (hasattr(current, '__iter__') and not current):
                continue

            change = current - open_price if current and open_price else 0
            change_pct = (change / open_price * 100) if open_price else 0

            # Calculate contract month (i months ahead of front month)
            # For Dubai, i=0 corresponds to M2 which is the front month equivalent
            contract_month = front_month + i
            contract_year = front_year
            while contract_month > 12:
                contract_month -= 12
                contract_year += 1

            # Format as MMM-YY (e.g., "Jan-25")
            contract_date = datetime(contract_year, contract_month, 1)
            contract_label = contract_date.strftime("%b-%y")

            # Approximate expiry date (around 20th of prior month)
            expiry_month = contract_month - 1 if contract_month > 1 else 12
            expiry_year = contract_year if contract_month > 1 else contract_year - 1
            expiry = datetime(expiry_year, expiry_month, 20)

            open_interest = row.get("OPEN_INT")
            if pd.isna(open_interest):
                open_interest = None

            # Month number in the curve (1-indexed for display)
            month_number = i + 1

            data.append({
                "month": month_number,
                "contract_month": contract_label,
                "contract_date": contract_date,
                "ticker": ticker,
                "price": current,
                "change": round(change, 4),
                "change_pct": round(change_pct, 4),
                "expiry": expiry,
                "days_to_expiry": max(0, (expiry - today).days),
                "open_interest": open_interest
            })

        if not data:
            raise DataUnavailableError(f"No valid curve data available for {commodity}")

        curve_df = pd.DataFrame(data)
        if "contract_date" in curve_df.columns:
            curve_df = curve_df.sort_values("contract_date").reset_index(drop=True)

        return curve_df

    def get_reference_data(self, ticker: str, fields: list[str]) -> dict:
        """Get reference data for a ticker."""
        parsed = TickerMapper.parse_ticker(ticker)
        parsed.get("commodity", "CL")

        ref = {
            "NAME": f"Oil Futures {ticker[:2]}",
            "TICKER": ticker,
            "EXCH_CODE": parsed.get("exchange", "NYMEX"),
            "CRNCY": "USD",
            "FUT_CONT_SIZE": parsed.get("multiplier", 1000),
            "FUT_TICK_SIZE": 0.01,
        }
        return {f: ref.get(f) for f in fields}

    def get_intraday_prices(self, ticker: str) -> pd.DataFrame:
        """
        Get intraday price history.

        Note: Intraday data requires Bloomberg subscription service.
        Returns empty DataFrame if not available.
        """
        # Intraday bars require Bloomberg subscription service
        # Return empty DataFrame if not available
        return pd.DataFrame(columns=["timestamp", "price"])

    def disconnect(self):
        """Disconnect from Bloomberg API."""
        if self._session:
            with contextlib.suppress(builtins.BaseException):
                self._session.stop()
            self._session = None
            self.connected = False


# =============================================================================
# REAL-TIME SUBSCRIPTION SERVICE
# =============================================================================

class BloombergSubscriptionService:
    """
    Real-time subscription service for Bloomberg data.

    Provides streaming price updates when connected to a real Bloomberg Terminal.
    Falls back to polling-based updates when subscriptions are not available.
    """

    def __init__(self, bloomberg_client: 'BloombergClient'):
        """
        Initialize subscription service.

        Args:
            bloomberg_client: Bloomberg client instance for data access
        """
        self.client = bloomberg_client
        self._subscriptions: dict[str, dict] = {}
        self._callbacks: dict[str, list] = {}
        self._running = False
        self._subscription_session = None

        # Check if subscriptions are enabled
        self.subscriptions_enabled = os.environ.get("BLOOMBERG_ENABLE_SUBSCRIPTIONS", "false").lower() == "true"

    def subscribe(self, ticker: str, callback=None) -> bool:
        """
        Subscribe to real-time updates for a ticker.

        Args:
            ticker: Bloomberg ticker to subscribe to
            callback: Optional callback function for updates

        Returns:
            True if subscription successful
        """
        if ticker in self._subscriptions:
            if callback:
                self._callbacks.setdefault(ticker, []).append(callback)
            return True

        if not self.subscriptions_enabled:
            # Subscriptions not enabled - register for polling-based updates
            self._subscriptions[ticker] = {
                "mode": "polling",
                "last_update": datetime.now(),
            }
            if callback:
                self._callbacks.setdefault(ticker, []).append(callback)
            return True

        # Real Bloomberg subscription
        try:
            import blpapi

            if not self._subscription_session:
                self._start_subscription_session()

            subscription_list = blpapi.SubscriptionList()
            subscription_list.add(
                ticker,
                "LAST_PRICE,BID,ASK,VOLUME",
                "",
                blpapi.CorrelationId(ticker)
            )

            self._subscription_session.subscribe(subscription_list)

            self._subscriptions[ticker] = {
                "mode": "real",
                "last_update": datetime.now(),
            }

            if callback:
                self._callbacks.setdefault(ticker, []).append(callback)

            logger.info(f"Subscribed to {ticker}")
            return True

        except Exception as e:
            logger.warning(f"Could not subscribe to {ticker}: {e}")
            # Fall back to polling-based updates
            self._subscriptions[ticker] = {
                "mode": "polling",
                "last_update": datetime.now(),
            }
            if callback:
                self._callbacks.setdefault(ticker, []).append(callback)
            return True

    def unsubscribe(self, ticker: str) -> None:
        """Unsubscribe from a ticker."""
        if ticker in self._subscriptions:
            del self._subscriptions[ticker]
        if ticker in self._callbacks:
            del self._callbacks[ticker]

    def get_subscribed_tickers(self) -> list[str]:
        """Get list of subscribed tickers."""
        return list(self._subscriptions.keys())

    def get_latest_prices(self) -> dict[str, dict[str, float]]:
        """Get latest prices for all subscribed tickers."""
        prices = {}
        for ticker in self._subscriptions:
            prices[ticker] = self.client.get_price_with_change(ticker)
        return prices

    def _start_subscription_session(self) -> bool:
        """Start Bloomberg subscription session."""
        try:
            import blpapi

            host = os.environ.get("BLOOMBERG_HOST", "localhost")
            port = int(os.environ.get("BLOOMBERG_PORT", "8194"))

            session_options = blpapi.SessionOptions()
            session_options.setServerHost(host)
            session_options.setServerPort(port)

            self._subscription_session = blpapi.Session(session_options)

            if not self._subscription_session.start():
                logger.warning("Failed to start subscription session")
                return False

            if not self._subscription_session.openService("//blp/mktdata"):
                logger.warning("Failed to open market data service")
                return False

            logger.info("Subscription session started")
            return True

        except Exception as e:
            logger.warning(f"Could not start subscription session: {e}")
            return False

    def stop(self) -> None:
        """Stop all subscriptions and cleanup."""
        self._running = False
        self._subscriptions.clear()
        self._callbacks.clear()

        if self._subscription_session:
            with contextlib.suppress(builtins.BaseException):
                self._subscription_session.stop()
            self._subscription_session = None
