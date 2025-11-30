"""
Dashboard domain utilities.
===========================
Provides object-oriented helpers that keep dashboard rendering code lean.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Sequence

import os
import pandas as pd


class PriceCache:
    """Cache ticker prices fetched from the data loader."""

    def __init__(self, data_loader):
        self._data_loader = data_loader
        self._prices: Dict[str, float] = {}

    def get(self, ticker: str) -> float:
        """Return the latest price for ticker, caching repeated requests."""
        if ticker not in self._prices:
            self._prices[ticker] = float(self._data_loader.get_price(ticker))
        return self._prices[ticker]


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
        self._multiplier_cache: Dict[str, int] = {}
        self._positions_df: pd.DataFrame | None = None
        self._summary: Dict[str, float] | None = None
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
                }
            )

        if not rows:
            columns = ["symbol", "ticker", "qty", "entry", "current", "pnl", "pnl_pct", "notional", "strategy"]
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
        """Format numeric P&L with sign."""
        if value >= 0:
            return f"+${value:,.0f}"
        return f"-${abs(value):,.0f}"

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
    def summary(self) -> Dict[str, float]:
        """Portfolio level summary metrics."""
        if self._summary is None:
            df = self.positions_dataframe
            total_pnl = float(df["pnl"].sum()) if not df.empty else 0.0
            gross_exposure = float(df["notional"].sum()) if not df.empty else 0.0
            long_exposure = float(df.loc[df["qty"] > 0, "notional"].sum()) if not df.empty else 0.0
            short_exposure = float(df.loc[df["qty"] < 0, "notional"].sum()) if not df.empty else 0.0
            net_exposure_value = long_exposure - short_exposure
            net_contracts = float(df["qty"].sum()) if not df.empty else 0.0

            var_estimate = gross_exposure * 0.02
            var_utilization = (var_estimate / self._var_limit * 100) if self._var_limit else 0.0
            pnl_return_pct = (total_pnl / self._base_capital * 100) if self._base_capital else 0.0
            position_records = df.to_dict("records")

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
            }

        return self._summary.copy()

    def concentration(self, ticker_prefix: str) -> float:
        """Return concentration percentage for tickers with prefix."""
        df = self.positions_dataframe
        total = float(df["notional"].sum()) if not df.empty else 0.0
        if total == 0:
            return 0.0

        mask = df["ticker"].str.startswith(ticker_prefix)
        focused = float(df.loc[mask, "notional"].sum())
        return focused / total * 100


class DashboardData:
    """Data bundle for the dashboard with caching."""

    def __init__(self, data_loader, lookback_days: int = 90):
        self.data_loader = data_loader
        self.lookback_days = lookback_days
        self._oil_prices: Dict[str, Dict[str, float]] | None = None
        self._wti_brent_spread: Dict[str, float] | None = None
        self._crack_321: Dict[str, float] | None = None
        self._futures_curve: pd.DataFrame | None = None
        self._brent_curve: pd.DataFrame | None = None
        self._wti_history: pd.DataFrame | None = None
        self._connection_status: Dict[str, str] | None = None

    @property
    def oil_prices(self) -> Dict[str, Dict[str, float]]:
        if self._oil_prices is None:
            self._oil_prices = self.data_loader.get_oil_prices()
        return self._oil_prices

    @property
    def wti_brent_spread(self) -> Dict[str, float]:
        if self._wti_brent_spread is None:
            self._wti_brent_spread = self.data_loader.get_wti_brent_spread()
        return self._wti_brent_spread

    @property
    def crack_spread(self) -> Dict[str, float]:
        if self._crack_321 is None:
            self._crack_321 = self.data_loader.get_crack_spread_321()
        return self._crack_321

    @property
    def futures_curve(self) -> pd.DataFrame:
        if self._futures_curve is None:
            self._futures_curve = self.data_loader.get_futures_curve("wti", 12)
        return self._futures_curve

    @property
    def brent_curve(self) -> pd.DataFrame:
        if self._brent_curve is None:
            self._brent_curve = self.data_loader.get_futures_curve("brent", 12)
        return self._brent_curve

    @property
    def wti_history(self) -> pd.DataFrame | None:
        if self._wti_history is None:
            end = datetime.now()
            start = end - timedelta(days=self.lookback_days)
            self._wti_history = self.data_loader.get_historical("CL1 Comdty", start_date=start, end_date=end)
        return self._wti_history

    @property
    def connection_status(self) -> Dict[str, str]:
        if self._connection_status is None:
            self._connection_status = self.data_loader.get_connection_status()
        return self._connection_status

    def curve_metrics(self) -> Dict[str, float | str]:
        curve = self.futures_curve
        if curve is None or curve.empty:
            return {"structure": "Unknown", "slope": 0.0, "m1_m2": 0.0, "m1_m6": 0.0, "m1_m12": 0.0}

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
        }


class DashboardContext:
    """Aggregates all data required to render the dashboard."""

    def __init__(self, data_loader, positions: Sequence[Dict[str, float]], lookback_days: int = 90):
        self.data_loader = data_loader
        self.price_cache = PriceCache(data_loader)
        self.positions = [Position(**pos) for pos in positions]
        self.data = DashboardData(data_loader, lookback_days)
        self.portfolio = PortfolioAnalytics(self.positions, data_loader, self.price_cache)

    def generate_recent_trades(self) -> pd.DataFrame:
        """Mock trade blotter with consistent pricing."""
        wti = self.price_cache.get("CL1 Comdty")
        brent = self.price_cache.get("CO1 Comdty")
        rbob = self.price_cache.get("XB1 Comdty")

        return pd.DataFrame(
            {
                "Time": ["14:32", "10:15", "09:45", "09:30"],
                "Symbol": ["CLF5", "CLF5", "COH5", "XBF5"],
                "Side": ["BUY", "SELL", "SELL", "BUY"],
                "Qty": [10, 5, 15, 8],
                "Price": [
                    round(wti * 0.998, 2),
                    round(wti * 1.002, 2),
                    round(brent, 2),
                    round(rbob, 2),
                ],
            }
        )

    def generate_signals(self) -> pd.DataFrame:
        """Return a small table of rule-based signals."""
        wti_data = self.data.oil_prices.get("WTI", {})
        wti_change = float(wti_data.get("change_pct", 0.0) or 0.0)
        spread = float(self.data.wti_brent_spread.get("spread", 0.0))

        signals: List[Dict[str, str]] = []

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

    def generate_alerts(self, curve_structure: str, var_utilization: float) -> List[Dict[str, str]]:
        """Assemble risk alerts from portfolio metrics."""
        alerts: List[Dict[str, str]] = []
        wti_concentration = self.portfolio.concentration("CL")

        if wti_concentration > 40:
            alerts.append({"severity": "warning", "message": f"WTI concentration at {wti_concentration:.0f}%."})

        if curve_structure == "Backwardation":
            alerts.append({"severity": "info", "message": "Curve is in backwardation."})
        elif curve_structure == "Contango":
            alerts.append({"severity": "info", "message": "Curve is in contango; expect roll costs."})

        if var_utilization > 75:
            alerts.append({"severity": "warning", "message": f"VaR utilization at {var_utilization:.0f}% of limit."})

        return alerts
