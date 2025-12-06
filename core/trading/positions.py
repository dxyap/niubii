"""
Position Management
===================
Track and manage open positions.
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


class PositionManager:
    """
    Position management and tracking.

    Features:
    - Position calculation from trades
    - Average entry price tracking
    - Position history
    - P&L by position
    """

    # Contract specifications
    CONTRACT_SPECS = {
        "CL": {"name": "WTI Crude", "multiplier": 1000, "tick": 0.01},
        "CO": {"name": "Brent Crude", "multiplier": 1000, "tick": 0.01},
        "XB": {"name": "RBOB Gasoline", "multiplier": 42000, "tick": 0.0001},
        "HO": {"name": "Heating Oil", "multiplier": 42000, "tick": 0.0001},
        "QS": {"name": "Gasoil", "multiplier": 100, "tick": 0.25},
    }

    def __init__(self, db_path: str = "data/trades/trades.db"):
        """
        Initialize position manager.

        Args:
            db_path: Path to trades database
        """
        self.db_path = Path(db_path)
        self._positions: dict[str, dict] = {}

    def calculate_positions(self) -> dict[str, dict]:
        """
        Calculate current positions from trade history.

        Returns:
            Dictionary of positions by instrument
        """
        if not self.db_path.exists():
            return {}

        conn = sqlite3.connect(self.db_path)

        # Get all trades
        query = """
            SELECT instrument, side, quantity, price, notional, trade_date
            FROM trades
            ORDER BY trade_date, trade_time
        """

        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            return {}

        positions = {}

        for instrument in df["instrument"].unique():
            inst_trades = df[df["instrument"] == instrument]

            # Calculate net position using FIFO
            total_qty = 0
            total_cost = 0
            realized_pnl = 0

            for _, trade in inst_trades.iterrows():
                qty = trade["quantity"] if trade["side"] == "BUY" else -trade["quantity"]
                price = trade["price"]

                if total_qty == 0:
                    # New position
                    total_qty = qty
                    total_cost = qty * price
                elif (total_qty > 0 and qty > 0) or (total_qty < 0 and qty < 0):
                    # Adding to position
                    total_qty += qty
                    total_cost += qty * price
                else:
                    # Reducing or closing position
                    avg_entry = total_cost / total_qty if total_qty != 0 else 0

                    if abs(qty) >= abs(total_qty):
                        # Full close or reversal
                        realized_pnl += total_qty * (price - avg_entry)
                        remaining = qty + total_qty
                        total_qty = remaining
                        total_cost = remaining * price if remaining != 0 else 0
                    else:
                        # Partial close
                        realized_pnl += (-qty) * (price - avg_entry)
                        total_qty += qty
                        total_cost = total_qty * avg_entry

            if total_qty != 0:
                avg_entry = total_cost / total_qty
                contract_type = instrument[:2]
                spec = self.CONTRACT_SPECS.get(contract_type, {"multiplier": 1000})

                positions[instrument] = {
                    "quantity": int(total_qty),
                    "avg_entry_price": round(avg_entry, 4),
                    "total_cost": round(total_cost * spec["multiplier"], 2),
                    "realized_pnl": round(realized_pnl * spec["multiplier"], 2),
                    "direction": "LONG" if total_qty > 0 else "SHORT",
                    "contract_type": contract_type,
                    "multiplier": spec["multiplier"],
                }

        self._positions = positions
        return positions

    def get_position(self, instrument: str) -> dict | None:
        """
        Get position for specific instrument.

        Args:
            instrument: Instrument ticker

        Returns:
            Position dictionary or None
        """
        positions = self.calculate_positions()
        return positions.get(instrument)

    def get_all_positions(self) -> dict[str, dict]:
        """Get all current positions."""
        return self.calculate_positions()

    def calculate_position_pnl(
        self,
        instrument: str,
        current_price: float
    ) -> dict:
        """
        Calculate P&L for a position.

        Args:
            instrument: Instrument ticker
            current_price: Current market price

        Returns:
            P&L dictionary
        """
        position = self.get_position(instrument)

        if not position:
            return {
                "unrealized_pnl": 0,
                "unrealized_pnl_pct": 0,
                "realized_pnl": 0,
                "total_pnl": 0,
            }

        qty = position["quantity"]
        avg_entry = position["avg_entry_price"]
        multiplier = position["multiplier"]
        realized = position["realized_pnl"]

        # Unrealized P&L
        unrealized = qty * (current_price - avg_entry) * multiplier
        unrealized_pct = (current_price / avg_entry - 1) * 100 * np.sign(qty)

        return {
            "instrument": instrument,
            "quantity": qty,
            "avg_entry_price": avg_entry,
            "current_price": current_price,
            "unrealized_pnl": round(unrealized, 2),
            "unrealized_pnl_pct": round(unrealized_pct, 2),
            "realized_pnl": realized,
            "total_pnl": round(unrealized + realized, 2),
        }

    def calculate_all_pnl(self, current_prices: dict[str, float]) -> pd.DataFrame:
        """
        Calculate P&L for all positions.

        Args:
            current_prices: Dict of current prices by instrument

        Returns:
            DataFrame of position P&L
        """
        positions = self.calculate_positions()

        if not positions:
            return pd.DataFrame()

        data = []
        for instrument, position in positions.items():
            current_price = current_prices.get(instrument, position["avg_entry_price"])
            pnl = self.calculate_position_pnl(instrument, current_price)

            data.append({
                "instrument": instrument,
                "direction": position["direction"],
                "quantity": position["quantity"],
                "avg_entry": position["avg_entry_price"],
                "current_price": current_price,
                "unrealized_pnl": pnl["unrealized_pnl"],
                "unrealized_pnl_pct": pnl["unrealized_pnl_pct"],
                "realized_pnl": pnl["realized_pnl"],
                "total_pnl": pnl["total_pnl"],
            })

        return pd.DataFrame(data)

    def get_exposure_summary(self, current_prices: dict[str, float]) -> dict:
        """
        Get portfolio exposure summary.

        Args:
            current_prices: Current prices by instrument

        Returns:
            Exposure summary
        """
        positions = self.calculate_positions()

        if not positions:
            return {
                "gross_exposure": 0,
                "net_exposure": 0,
                "long_exposure": 0,
                "short_exposure": 0,
                "num_positions": 0,
            }

        gross = 0
        net = 0
        long_exp = 0
        short_exp = 0

        for instrument, position in positions.items():
            price = current_prices.get(instrument, position["avg_entry_price"])
            notional = position["quantity"] * price * position["multiplier"]

            gross += abs(notional)
            net += notional

            if notional > 0:
                long_exp += notional
            else:
                short_exp += abs(notional)

        return {
            "gross_exposure": round(gross, 2),
            "net_exposure": round(net, 2),
            "long_exposure": round(long_exp, 2),
            "short_exposure": round(short_exp, 2),
            "num_positions": len(positions),
            "direction": "Net Long" if net > 0 else "Net Short" if net < 0 else "Flat",
        }

    def get_position_weights(self, current_prices: dict[str, float]) -> pd.DataFrame:
        """
        Calculate position weights in portfolio.

        Args:
            current_prices: Current prices

        Returns:
            DataFrame with position weights
        """
        positions = self.calculate_positions()

        if not positions:
            return pd.DataFrame()

        # Calculate notionals
        data = []
        total_notional = 0

        for instrument, position in positions.items():
            price = current_prices.get(instrument, position["avg_entry_price"])
            notional = abs(position["quantity"] * price * position["multiplier"])
            total_notional += notional

            data.append({
                "instrument": instrument,
                "direction": position["direction"],
                "notional": notional,
            })

        # Calculate weights
        df = pd.DataFrame(data)
        if total_notional > 0:
            df["weight_pct"] = (df["notional"] / total_notional * 100).round(1)
        else:
            df["weight_pct"] = 0

        return df.sort_values("weight_pct", ascending=False)
