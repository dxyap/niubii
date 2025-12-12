"""
Dashboard domain utilities.
===========================
Provides object-oriented helpers that keep dashboard rendering code lean.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

# Import centralized formatting utilities
from core.utils import format_pnl


class DataNotAvailableError(Exception):
    """Raised when data is not available for display."""
    pass


class PriceCache:
    """
    Lightweight price cache that leverages DataLoader's built-in deduplication.
    
    This class provides a simple interface for getting prices while tracking
    errors. The actual caching and deduplication is handled by DataLoader's
    RequestDeduplicator, so this class only adds error tracking and a
    consistent interface.
    """

    def __init__(self, data_loader):
        self._data_loader = data_loader
        self._errors: dict[str, str] = {}
        self._session_prices: dict[str, float] = {}

    def get(self, ticker: str) -> float | None:
        """
        Return the latest price for ticker.
        
        Uses DataLoader's built-in request deduplication to prevent
        redundant Bloomberg API calls within the same refresh cycle.
        """
        if ticker in self._errors:
            return None
        
        # Check session cache first (for repeated calls in same render)
        if ticker in self._session_prices:
            return self._session_prices[ticker]
        
        try:
            # DataLoader.get_price() uses RequestDeduplicator internally
            price = float(self._data_loader.get_price(ticker))
            self._session_prices[ticker] = price
            return price
        except Exception as e:
            self._errors[ticker] = str(e)
            return None

    def get_batch(self, tickers: list[str]) -> dict[str, float]:
        """
        Get prices for multiple tickers in a single batch call.
        More efficient than calling get() for each ticker.
        """
        try:
            batch = self._data_loader.get_prices_batch(tickers)
            for ticker, data in batch.items():
                if "current" in data:
                    self._session_prices[ticker] = data["current"]
            return {t: d.get("current") for t, d in batch.items()}
        except Exception as e:
            for ticker in tickers:
                self._errors[ticker] = str(e)
            return {}

    def get_error(self, ticker: str) -> str | None:
        """Get error message for ticker if any."""
        return self._errors.get(ticker)

    def has_errors(self) -> bool:
        """Check if any errors occurred."""
        return len(self._errors) > 0

    def clear(self) -> None:
        """Clear session cache and errors for a fresh start."""
        self._session_prices.clear()
        self._errors.clear()


@dataclass(frozen=True)
class Position:
    """Immutable representation of a portfolio position."""

    symbol: str
    ticker: str
    qty: float
    entry: float
    strategy: str


class PortfolioAnalytics:
    """Encapsulates portfolio level calculations."""

    def __init__(self, positions: Sequence[Position], data_loader, price_cache: PriceCache | None = None):
        self.positions = list(positions)
        self.data_loader = data_loader
        self.price_cache = price_cache or PriceCache(data_loader)
        self._multiplier_cache: dict[str, int] = {}
        self._positions_df: pd.DataFrame | None = None
        self._summary: dict[str, float] | None = None
        self._var_limit = float(os.getenv("MAX_VAR_LIMIT", 375000))
        self._base_capital = float(os.getenv("PORTFOLIO_BASE_CAPITAL", 1_000_000))

    def _get_multiplier(self, ticker: str) -> int:
        if ticker not in self._multiplier_cache:
            try:
                self._multiplier_cache[ticker] = int(self.data_loader.get_multiplier(ticker))
            except Exception:
                # Default to crude contract size when mapping fails
                self._multiplier_cache[ticker] = 1000
        return self._multiplier_cache[ticker]

    def _build_dataframe(self) -> pd.DataFrame:
        rows = []
        for pos in self.positions:
            current_price = self.price_cache.get(pos.ticker)

            # Handle case when price data is unavailable
            if current_price is None:
                rows.append(
                    {
                        "symbol": pos.symbol,
                        "ticker": pos.ticker,
                        "qty": pos.qty,
                        "entry": round(pos.entry, 4),
                        "current": None,
                        "pnl": None,
                        "pnl_pct": None,
                        "notional": None,
                        "strategy": pos.strategy,
                        "error": self.price_cache.get_error(pos.ticker),
                    }
                )
                continue

            multiplier = self._get_multiplier(pos.ticker)
            notional = abs(pos.qty) * current_price * multiplier
            price_change = current_price - pos.entry
            pnl = price_change * pos.qty * multiplier
            pnl_pct = (price_change / pos.entry * 100) if pos.entry else 0.0

            rows.append(
                {
                    "symbol": pos.symbol,
                    "ticker": pos.ticker,
                    "qty": pos.qty,
                    "entry": round(pos.entry, 4),
                    "current": round(current_price, 4),
                    "pnl": round(pnl, 0),
                    "pnl_pct": round(pnl_pct, 2),
                    "notional": round(notional, 0),
                    "strategy": pos.strategy,
                    "error": None,
                }
            )

        if not rows:
            columns = ["symbol", "ticker", "qty", "entry", "current", "pnl", "pnl_pct", "notional", "strategy", "error"]
            return pd.DataFrame(columns=columns)

        return pd.DataFrame(rows)

    @property
    def positions_dataframe(self) -> pd.DataFrame:
        """Return a copy of the positions DataFrame."""
        if self._positions_df is None:
            self._positions_df = self._build_dataframe()
        return self._positions_df.copy()

    @staticmethod
    def format_pnl(value: float) -> str:
        """Format numeric P&L with sign. Delegates to centralized utility."""
        return format_pnl(value)

    def formatted_table(self) -> pd.DataFrame:
        """Return a presentation-ready table."""
        df = self.positions_dataframe
        if df.empty:
            return df

        formatted = df.copy()
        formatted["P&L"] = formatted["pnl"].apply(self.format_pnl)
        formatted = formatted.rename(
            columns={
                "symbol": "Symbol",
                "qty": "Qty",
                "entry": "Entry",
                "current": "Current",
            }
        )
        return formatted[["Symbol", "Qty", "Entry", "Current", "P&L"]]

    @property
    def summary(self) -> dict[str, float]:
        """Portfolio level summary metrics."""
        if self._summary is None:
            df = self.positions_dataframe

            # Filter out rows with unavailable data for calculations
            valid_df = df[df["pnl"].notna()] if not df.empty else df

            total_pnl = float(valid_df["pnl"].sum()) if not valid_df.empty else 0.0
            gross_exposure = float(valid_df["notional"].sum()) if not valid_df.empty else 0.0
            long_exposure = float(valid_df.loc[valid_df["qty"] > 0, "notional"].sum()) if not valid_df.empty else 0.0
            short_exposure = float(valid_df.loc[valid_df["qty"] < 0, "notional"].sum()) if not valid_df.empty else 0.0
            net_exposure_value = long_exposure - short_exposure
            net_contracts = float(df["qty"].sum()) if not df.empty else 0.0

            var_estimate = gross_exposure * 0.02
            var_utilization = (var_estimate / self._var_limit * 100) if self._var_limit else 0.0
            pnl_return_pct = (total_pnl / self._base_capital * 100) if self._base_capital else 0.0
            position_records = df.to_dict("records")

            # Count positions with data errors
            data_errors = len(df[df["error"].notna()]) if "error" in df.columns and not df.empty else 0

            self._summary = {
                "total_pnl": total_pnl,
                "pnl_return_pct": pnl_return_pct,
                "gross_exposure": gross_exposure,
                "long_exposure": long_exposure,
                "short_exposure": short_exposure,
                "net_exposure_value": net_exposure_value,
                "net_exposure": net_exposure_value,
                "net_contracts": net_contracts,
                "var_estimate": var_estimate,
                "var_limit": self._var_limit,
                "var_utilization": min(var_utilization, 100.0),
                "base_capital": self._base_capital,
                "positions": position_records,
                "data_errors": data_errors,
                "has_data_errors": data_errors > 0,
            }

        return self._summary.copy()

    def concentration(self, ticker_prefix: str) -> float:
        """Return concentration percentage for tickers with prefix."""
        df = self.positions_dataframe

        # Filter out rows with None notional values
        valid_df = df[df["notional"].notna()] if not df.empty else df

        total = float(valid_df["notional"].sum()) if not valid_df.empty else 0.0
        if total == 0:
            return 0.0

        mask = valid_df["ticker"].str.startswith(ticker_prefix)
        focused = float(valid_df.loc[mask, "notional"].sum())
        return focused / total * 100


class DashboardData:
    """
    Data bundle for the dashboard with optimized loading.

    Key optimizations:
    - Prefetch all price data in a single batch call via DataLoader
    - DataLoader's RequestDeduplicator prevents duplicate API calls
    - Lazy loading for expensive historical data
    - Caching of computed metrics
    
    Note: All data fetching goes through DataLoader which handles:
    - Request deduplication (prevents duplicate Bloomberg calls)
    - Multi-layer caching (TTL cache, disk cache)
    - Market-hours aware TTL (longer cache during off-hours)
    """

    # Sentinel to distinguish "not loaded" from "loaded but None/empty"
    _NOT_LOADED = object()

    def __init__(self, data_loader, lookback_days: int = 90, prefetch: bool = True):
        self.data_loader = data_loader
        self.lookback_days = lookback_days

        # Price data (loaded together in batch via DataLoader)
        self._oil_prices = self._NOT_LOADED
        self._all_spreads = self._NOT_LOADED

        # Curve data (lazy loaded)
        self._curves: dict[str, pd.DataFrame | object] = {
            "futures_curve": self._NOT_LOADED,  # WTI
            "brent_curve": self._NOT_LOADED,
            "dubai_curve": self._NOT_LOADED,
        }

        # Historical data (expensive - loaded lazily)
        self._wti_history = self._NOT_LOADED

        # Connection status (cached)
        self._connection_status: dict[str, str] | None = None

        # Store errors for display
        self._errors: dict[str, str] = {}

        # Prefetch price data if requested
        if prefetch:
            self._prefetch_price_data()
        self._crack_index = self._NOT_LOADED

    def _prefetch_price_data(self) -> None:
        """
        Prefetch all price-related data in optimized batch calls.
        
        DataLoader.get_oil_prices() and get_all_spreads() use:
        - _get_price_batch_cached() for batched Bloomberg calls
        - RequestDeduplicator to coalesce identical requests
        - Smart TTL based on market hours
        
        This significantly reduces API calls on page load.
        """
        try:
            # Fetch oil prices (batch call via DataLoader)
            self._oil_prices = self.data_loader.get_oil_prices()
        except Exception as e:
            self._errors["oil_prices"] = str(e)
            self._oil_prices = None

        try:
            # Fetch all spreads in a single batch
            # Note: This reuses prices from the same cache if called close together
            self._all_spreads = self.data_loader.get_all_spreads()
        except Exception as e:
            self._errors["spreads"] = str(e)
            self._all_spreads = None

    @property
    def oil_prices(self) -> dict[str, dict[str, float]] | None:
        if self._oil_prices is self._NOT_LOADED:
            try:
                self._oil_prices = self.data_loader.get_oil_prices()
            except Exception as e:
                self._errors["oil_prices"] = str(e)
                self._oil_prices = None
        return self._oil_prices

    @property
    def wti_brent_spread(self) -> dict[str, float] | None:
        # Use prefetched spread data if available
        if self._all_spreads is not self._NOT_LOADED and self._all_spreads is not None:
            return self._all_spreads.get("wti_brent")

        # Fallback to direct fetch
        try:
            return self.data_loader.get_wti_brent_spread()
        except Exception as e:
            self._errors["wti_brent_spread"] = str(e)
            return None

    @property
    def crack_spread(self) -> dict[str, float | str] | None:
        # Prefer Bloomberg FVCSM index (same source as Market Insights)
        if self._crack_index is self._NOT_LOADED:
            try:
                self._crack_index = self.data_loader.get_crack_spread_321_index()
            except Exception as e:
                self._errors["crack_spread_index"] = str(e)
                self._crack_index = None
        if self._crack_index is not None:
            return self._crack_index

        # Use prefetched spread data if available
        if self._all_spreads is not self._NOT_LOADED and self._all_spreads is not None:
            return self._all_spreads.get("crack_321")

        # Fallback to direct fetch
        try:
            return self.data_loader.get_crack_spread_321()
        except Exception as e:
            self._errors["crack_spread"] = str(e)
            return None

    def _get_curve(self, key: str, commodity: str, months: int) -> pd.DataFrame | None:
        """Shared curve loader with caching and error capture."""
        cached = self._curves.get(key, self._NOT_LOADED)
        if cached is self._NOT_LOADED:
            try:
                cached = self.data_loader.get_futures_curve(commodity, months)
            except Exception as e:
                self._errors[key] = str(e)
                cached = None
            self._curves[key] = cached
        return cached

    @property
    def futures_curve(self) -> pd.DataFrame | None:
        return self._get_curve("futures_curve", "wti", 18)

    @property
    def brent_curve(self) -> pd.DataFrame | None:
        return self._get_curve("brent_curve", "brent", 18)

    @property
    def dubai_curve(self) -> pd.DataFrame | None:
        return self._get_curve("dubai_curve", "dubai", 18)

    @property
    def wti_history(self) -> pd.DataFrame | None:
        """Historical data is loaded lazily as it's expensive."""
        if self._wti_history is self._NOT_LOADED:
            try:
                end = datetime.now()
                start = end - timedelta(days=self.lookback_days)
                self._wti_history = self.data_loader.get_historical("CL1 Comdty", start_date=start, end_date=end)
            except Exception as e:
                self._errors["wti_history"] = str(e)
                self._wti_history = None
        return self._wti_history

    @property
    def connection_status(self) -> dict[str, str]:
        if self._connection_status is None:
            self._connection_status = self.data_loader.get_connection_status()
        return self._connection_status

    def get_error(self, key: str) -> str | None:
        """Get error message for a data key if any."""
        return self._errors.get(key)

    def has_errors(self) -> bool:
        """Check if any errors occurred during data loading."""
        return len(self._errors) > 0

    def get_all_errors(self) -> dict[str, str]:
        """Get all error messages."""
        return self._errors.copy()

    def curve_metrics(self) -> dict[str, float | str]:
        curve = self.futures_curve
        if curve is None or (isinstance(curve, pd.DataFrame) and curve.empty):
            return {
                "structure": "Data Unavailable",
                "slope": 0.0,
                "m1_m2": 0.0,
                "m1_m6": 0.0,
                "m1_m12": 0.0,
                "error": self.get_error("futures_curve"),
            }

        prices = curve["price"]
        slope = (prices.iloc[-1] - prices.iloc[0]) / max(len(prices) - 1, 1)

        if slope > 0.05:
            structure = "Contango"
        elif slope < -0.05:
            structure = "Backwardation"
        else:
            structure = "Flat"

        def spread(idx: int) -> float:
            if len(prices) > idx:
                return float(prices.iloc[0] - prices.iloc[idx])
            return 0.0

        return {
            "structure": structure,
            "slope": float(round(slope, 4)),
            "m1_m2": round(spread(1), 2),
            "m1_m6": round(spread(5), 2),
            "m1_m12": round(spread(11), 2),
            "error": None,
        }


class DashboardContext:
    """Aggregates all data required to render the dashboard."""

    def __init__(self, data_loader, positions: Sequence[dict[str, float]], lookback_days: int = 90):
        self.data_loader = data_loader
        self.price_cache = PriceCache(data_loader)
        self.positions = [Position(**pos) for pos in positions]
        self.data = DashboardData(data_loader, lookback_days)
        self.portfolio = PortfolioAnalytics(self.positions, data_loader, self.price_cache)

    def is_data_available(self) -> bool:
        """Check if data is available for display."""
        return self.data_loader.is_data_available()

    def get_connection_error(self) -> str | None:
        """Get connection error message if any."""
        return self.data_loader.get_connection_error_message()

    def generate_recent_trades(self) -> pd.DataFrame:
        """
        Return recent trades from blotter.
        Returns empty DataFrame with message if no trades.
        """
        # Try to get real trades from the blotter
        try:
            from core.trading.blotter import TradeBlotter
            blotter = TradeBlotter()
            trades = blotter.get_todays_trades()
            if not trades.empty:
                return trades[["trade_time", "instrument", "side", "quantity", "price"]].rename(
                    columns={
                        "trade_time": "Time",
                        "instrument": "Symbol",
                        "side": "Side",
                        "quantity": "Qty",
                        "price": "Price",
                    }
                ).head(10)
        except Exception:
            pass

        # Return empty DataFrame if no trades
        return pd.DataFrame(columns=["Time", "Symbol", "Side", "Qty", "Price"])

    def generate_signals(self) -> pd.DataFrame:
        """Return a small table of rule-based signals."""
        oil_prices = self.data.oil_prices
        spread_data = self.data.wti_brent_spread

        # If data is not available, return empty signals
        if oil_prices is None or spread_data is None:
            return pd.DataFrame(columns=["Instrument", "Direction", "Confidence", "Horizon", "Status"])

        wti_data = oil_prices.get("WTI", {})
        wti_change = float(wti_data.get("change_pct", 0.0) or 0.0)
        spread = float(spread_data.get("spread", 0.0))

        signals: list[dict[str, str]] = []

        if wti_change > 0.3:
            confidence = min(50 + wti_change * 10, 85)
            signals.append(
                {"Instrument": "WTI (CL1)", "Direction": "LONG", "Confidence": f"{confidence:.0f}%", "Horizon": "5-10 Days"}
            )
        elif wti_change < -0.3:
            confidence = min(50 + abs(wti_change) * 10, 85)
            signals.append(
                {"Instrument": "WTI (CL1)", "Direction": "SHORT", "Confidence": f"{confidence:.0f}%", "Horizon": "5-10 Days"}
            )
        else:
            signals.append({"Instrument": "WTI (CL1)", "Direction": "NEUTRAL", "Confidence": "45%", "Horizon": "-"})

        if spread < -5:
            signals.append({"Instrument": "WTI-Brent", "Direction": "LONG WTI", "Confidence": "62%", "Horizon": "3-5 Days"})
        elif spread > -3:
            signals.append({"Instrument": "WTI-Brent", "Direction": "SHORT WTI", "Confidence": "58%", "Horizon": "3-5 Days"})

        return pd.DataFrame(signals)

    def generate_alerts(self, curve_structure: str, var_utilization: float) -> list[dict[str, str]]:
        """Assemble risk alerts from portfolio metrics."""
        alerts: list[dict[str, str]] = []

        # Check for data availability issues first
        if not self.is_data_available():
            error = self.get_connection_error()
            alerts.append({
                "severity": "critical",
                "message": f"Data unavailable: {error or 'Bloomberg not connected'}"
            })
            return alerts

        # Check for data loading errors
        if self.data.has_errors():
            for key, error in self.data.get_all_errors().items():
                alerts.append({
                    "severity": "warning",
                    "message": f"Data error ({key}): {error[:50]}..."
                })

        # Portfolio alerts only if we have positions
        if self.positions:
            wti_concentration = self.portfolio.concentration("CL")

            if wti_concentration > 40:
                alerts.append({"severity": "warning", "message": f"WTI concentration at {wti_concentration:.0f}%."})

            if var_utilization > 75:
                alerts.append({"severity": "warning", "message": f"VaR utilization at {var_utilization:.0f}% of limit."})

        # Curve structure alerts
        if curve_structure and curve_structure not in ("Unknown", "Data Unavailable"):
            if curve_structure == "Backwardation":
                alerts.append({"severity": "info", "message": "Curve is in backwardation."})
            elif curve_structure == "Contango":
                alerts.append({"severity": "info", "message": "Curve is in contango; expect roll costs."})

        return alerts
