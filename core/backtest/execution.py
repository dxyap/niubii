"""
Execution Simulation
====================
Order execution and fill simulation for backtesting.

Provides:
- Market order execution
- Limit order handling
- Stop order handling
- Partial fills
- Order book simulation
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd

from .costs import CostModel, OrderSide, SimpleCostModel, TransactionCosts
from .strategy import Order, Position

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


@dataclass
class Fill:
    """Represents an order fill."""
    order_id: str
    timestamp: datetime
    symbol: str
    side: str
    quantity: int
    price: float
    commission: float = 0.0
    slippage: float = 0.0
    costs: TransactionCosts | None = None

    @property
    def total_value(self) -> float:
        """Total value of fill including costs."""
        base_value = self.quantity * self.price
        cost = self.costs.total if self.costs else (self.commission + self.slippage)
        return base_value + cost if self.side == "BUY" else base_value - cost


@dataclass
class OrderRecord:
    """Complete order record with status and fills."""
    order: Order
    order_id: str
    status: OrderStatus = OrderStatus.PENDING
    fills: list[Fill] = field(default_factory=list)
    created_at: datetime = None
    updated_at: datetime = None
    rejection_reason: str | None = None

    @property
    def filled_quantity(self) -> int:
        """Total filled quantity."""
        return sum(f.quantity for f in self.fills)

    @property
    def remaining_quantity(self) -> int:
        """Remaining unfilled quantity."""
        return self.order.quantity - self.filled_quantity

    @property
    def avg_fill_price(self) -> float:
        """Average fill price."""
        if not self.fills:
            return 0.0
        total_value = sum(f.price * f.quantity for f in self.fills)
        return total_value / self.filled_quantity


class ExecutionSimulator:
    """
    Simulates order execution with realistic fills.

    Supports:
    - Market orders (immediate execution with slippage)
    - Limit orders (conditional execution)
    - Stop orders (trigger-based execution)
    - Transaction costs
    """

    def __init__(
        self,
        cost_model: CostModel | None = None,
        slippage_pct: float = 0.01,  # 1 basis point default
        fill_probability: float = 1.0,  # Probability of fill for limit orders
        use_high_low_for_fills: bool = True,  # Use H/L for limit order fills
    ):
        """
        Initialize execution simulator.

        Args:
            cost_model: Transaction cost model
            slippage_pct: Default slippage as percentage of price
            fill_probability: Probability of limit order fill
            use_high_low_for_fills: Check H/L for limit order execution
        """
        self.cost_model = cost_model or SimpleCostModel()
        self.slippage_pct = slippage_pct
        self.fill_probability = fill_probability
        self.use_high_low = use_high_low_for_fills

        self._pending_orders: dict[str, OrderRecord] = {}
        self._order_counter = 0
        self._all_orders: list[OrderRecord] = []
        self._all_fills: list[Fill] = []

    def submit_order(self, order: Order) -> OrderRecord:
        """
        Submit an order for execution.

        Args:
            order: Order to submit

        Returns:
            OrderRecord with order status
        """
        self._order_counter += 1
        order_id = f"ORD-{self._order_counter:06d}"

        record = OrderRecord(
            order=order,
            order_id=order_id,
            status=OrderStatus.PENDING,
            created_at=order.timestamp,
            updated_at=order.timestamp,
        )

        self._pending_orders[order_id] = record
        self._all_orders.append(record)

        return record

    def process_bar(
        self,
        timestamp: datetime,
        bar_data: dict[str, float]
    ) -> list[Fill]:
        """
        Process all pending orders against a bar of data.

        Args:
            timestamp: Bar timestamp
            bar_data: Dict with 'open', 'high', 'low', 'close' (or PX_ variants)

        Returns:
            List of fills generated
        """
        fills = []

        # Normalize column names
        open_price = bar_data.get("open") or bar_data.get("PX_OPEN")
        high_price = bar_data.get("high") or bar_data.get("PX_HIGH")
        low_price = bar_data.get("low") or bar_data.get("PX_LOW")
        bar_data.get("close") or bar_data.get("PX_LAST")

        orders_to_remove = []

        for order_id, record in self._pending_orders.items():
            order = record.order

            fill = None

            if order.order_type == "MARKET":
                fill = self._execute_market_order(
                    record, timestamp, open_price
                )
            elif order.order_type == "LIMIT":
                fill = self._execute_limit_order(
                    record, timestamp, open_price, high_price, low_price
                )
            elif order.order_type == "STOP":
                fill = self._execute_stop_order(
                    record, timestamp, open_price, high_price, low_price
                )

            if fill:
                fills.append(fill)
                record.fills.append(fill)
                record.updated_at = timestamp

                if record.remaining_quantity == 0:
                    record.status = OrderStatus.FILLED
                    orders_to_remove.append(order_id)
                else:
                    record.status = OrderStatus.PARTIALLY_FILLED

            # Check for expired orders
            if order.time_in_force == "DAY":
                # In a real system, would check if day has changed
                pass

        # Remove filled orders from pending
        for order_id in orders_to_remove:
            del self._pending_orders[order_id]

        self._all_fills.extend(fills)

        return fills

    def _execute_market_order(
        self,
        record: OrderRecord,
        timestamp: datetime,
        price: float
    ) -> Fill | None:
        """Execute a market order."""
        order = record.order

        # Apply slippage
        slippage = price * self.slippage_pct / 100
        if order.side == "BUY":
            fill_price = price + slippage
        else:
            fill_price = price - slippage

        # Calculate costs
        costs = self.cost_model.calculate_costs(
            price=fill_price,
            quantity=order.quantity,
            side=OrderSide.BUY if order.side == "BUY" else OrderSide.SELL
        )

        return Fill(
            order_id=record.order_id,
            timestamp=timestamp,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=round(fill_price, 4),
            commission=costs.commission,
            slippage=costs.slippage,
            costs=costs,
        )

    def _execute_limit_order(
        self,
        record: OrderRecord,
        timestamp: datetime,
        open_price: float,
        high_price: float,
        low_price: float
    ) -> Fill | None:
        """Execute a limit order if price conditions are met."""
        order = record.order
        limit_price = order.limit_price

        if limit_price is None:
            return None

        can_fill = False

        if order.side == "BUY":
            # Buy limit: fill if low <= limit price
            if self.use_high_low:
                can_fill = low_price <= limit_price
            else:
                can_fill = open_price <= limit_price
        else:
            # Sell limit: fill if high >= limit price
            if self.use_high_low:
                can_fill = high_price >= limit_price
            else:
                can_fill = open_price >= limit_price

        if not can_fill:
            return None

        # Probabilistic fill
        if np.random.random() > self.fill_probability:
            return None

        # Use limit price for fill (best case)
        fill_price = limit_price

        # Calculate costs (no slippage for limit orders)
        costs = self.cost_model.calculate_costs(
            price=fill_price,
            quantity=order.quantity,
            side=OrderSide.BUY if order.side == "BUY" else OrderSide.SELL
        )
        costs.slippage = 0  # No slippage for limit fills

        return Fill(
            order_id=record.order_id,
            timestamp=timestamp,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=round(fill_price, 4),
            commission=costs.commission,
            slippage=0.0,
            costs=costs,
        )

    def _execute_stop_order(
        self,
        record: OrderRecord,
        timestamp: datetime,
        open_price: float,
        high_price: float,
        low_price: float
    ) -> Fill | None:
        """Execute a stop order if trigger conditions are met."""
        order = record.order
        stop_price = order.stop_price

        if stop_price is None:
            return None

        triggered = False

        if order.side == "BUY":
            # Buy stop: trigger if high >= stop price
            if self.use_high_low:
                triggered = high_price >= stop_price
            else:
                triggered = open_price >= stop_price
        else:
            # Sell stop: trigger if low <= stop price
            if self.use_high_low:
                triggered = low_price <= stop_price
            else:
                triggered = open_price <= stop_price

        if not triggered:
            return None

        # Once triggered, treat as market order
        # Fill at stop price with slippage
        slippage = stop_price * self.slippage_pct / 100
        if order.side == "BUY":
            fill_price = stop_price + slippage
        else:
            fill_price = stop_price - slippage

        # Calculate costs
        costs = self.cost_model.calculate_costs(
            price=fill_price,
            quantity=order.quantity,
            side=OrderSide.BUY if order.side == "BUY" else OrderSide.SELL
        )

        return Fill(
            order_id=record.order_id,
            timestamp=timestamp,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=round(fill_price, 4),
            commission=costs.commission,
            slippage=costs.slippage,
            costs=costs,
        )

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        if order_id in self._pending_orders:
            record = self._pending_orders[order_id]
            record.status = OrderStatus.CANCELLED
            record.updated_at = datetime.now()
            del self._pending_orders[order_id]
            return True
        return False

    def cancel_all_orders(self, symbol: str | None = None):
        """Cancel all pending orders, optionally filtered by symbol."""
        to_cancel = []

        for order_id, record in self._pending_orders.items():
            if symbol is None or record.order.symbol == symbol:
                to_cancel.append(order_id)

        for order_id in to_cancel:
            self.cancel_order(order_id)

    def get_pending_orders(self, symbol: str | None = None) -> list[OrderRecord]:
        """Get all pending orders."""
        orders = list(self._pending_orders.values())
        if symbol:
            orders = [o for o in orders if o.order.symbol == symbol]
        return orders

    def get_all_fills(self) -> pd.DataFrame:
        """Get all fills as DataFrame."""
        if not self._all_fills:
            return pd.DataFrame()

        data = []
        for fill in self._all_fills:
            row = {
                "order_id": fill.order_id,
                "timestamp": fill.timestamp,
                "symbol": fill.symbol,
                "side": fill.side,
                "quantity": fill.quantity,
                "price": fill.price,
                "commission": fill.commission,
                "slippage": fill.slippage,
                "total_cost": fill.costs.total if fill.costs else 0,
            }
            data.append(row)

        return pd.DataFrame(data)

    def get_order_history(self) -> pd.DataFrame:
        """Get order history as DataFrame."""
        if not self._all_orders:
            return pd.DataFrame()

        data = []
        for record in self._all_orders:
            row = {
                "order_id": record.order_id,
                "timestamp": record.order.timestamp,
                "symbol": record.order.symbol,
                "side": record.order.side,
                "quantity": record.order.quantity,
                "order_type": record.order.order_type,
                "limit_price": record.order.limit_price,
                "stop_price": record.order.stop_price,
                "status": record.status.value,
                "filled_qty": record.filled_quantity,
                "avg_price": record.avg_fill_price,
            }
            data.append(row)

        return pd.DataFrame(data)

    def reset(self):
        """Reset simulator state."""
        self._pending_orders.clear()
        self._order_counter = 0
        self._all_orders.clear()
        self._all_fills.clear()


class PositionManager:
    """
    Manages positions based on fills.

    Tracks:
    - Current positions
    - Average entry prices
    - Realized and unrealized P&L
    """

    def __init__(self, contract_multiplier: float = 1000):
        """
        Initialize position manager.

        Args:
            contract_multiplier: Contract multiplier (barrels per contract)
        """
        self.multiplier = contract_multiplier
        self._positions: dict[str, Position] = {}
        self._trades: list[dict] = []

    def update_position(
        self,
        fill: Fill,
        current_price: float | None = None
    ) -> Position:
        """
        Update position based on a fill.

        Args:
            fill: Fill to process
            current_price: Current market price for MTM

        Returns:
            Updated position
        """
        symbol = fill.symbol

        if symbol not in self._positions:
            self._positions[symbol] = Position(symbol=symbol)

        pos = self._positions[symbol]
        old_qty = pos.quantity

        # Calculate P&L for closing trades
        trade_pnl = 0.0

        if fill.side == "BUY":
            new_qty = old_qty + fill.quantity

            if old_qty < 0:  # Covering short
                cover_qty = min(fill.quantity, abs(old_qty))
                trade_pnl = (pos.avg_price - fill.price) * cover_qty * self.multiplier
                trade_pnl -= fill.costs.total if fill.costs else 0
        else:
            new_qty = old_qty - fill.quantity

            if old_qty > 0:  # Closing long
                close_qty = min(fill.quantity, old_qty)
                trade_pnl = (fill.price - pos.avg_price) * close_qty * self.multiplier
                trade_pnl -= fill.costs.total if fill.costs else 0

        # Record trade
        if trade_pnl != 0:
            self._trades.append({
                "timestamp": fill.timestamp,
                "symbol": symbol,
                "side": fill.side,
                "quantity": fill.quantity,
                "price": fill.price,
                "pnl": trade_pnl,
            })

        # Update average price
        if new_qty == 0:
            new_avg = 0.0
        elif (old_qty >= 0 and fill.side == "BUY") or (old_qty <= 0 and fill.side == "SELL"):
            # Adding to position - calculate weighted average
            if old_qty == 0:
                new_avg = fill.price
            else:
                total_cost = abs(old_qty) * pos.avg_price + fill.quantity * fill.price
                new_avg = total_cost / abs(new_qty)
        else:
            # Reducing position - keep same average
            new_avg = pos.avg_price

        # Update position
        pos.quantity = new_qty
        pos.avg_price = new_avg
        pos.realized_pnl += trade_pnl

        # Update unrealized P&L if current price provided
        if current_price and new_qty != 0:
            pos.unrealized_pnl = (current_price - new_avg) * new_qty * self.multiplier
        else:
            pos.unrealized_pnl = 0.0

        return pos

    def get_position(self, symbol: str) -> Position:
        """Get current position for symbol."""
        if symbol not in self._positions:
            self._positions[symbol] = Position(symbol=symbol)
        return self._positions[symbol]

    def get_all_positions(self) -> dict[str, Position]:
        """Get all positions."""
        return self._positions.copy()

    def mark_to_market(self, prices: dict[str, float]):
        """Update unrealized P&L for all positions."""
        for symbol, pos in self._positions.items():
            if symbol in prices and pos.quantity != 0:
                price = prices[symbol]
                pos.unrealized_pnl = (price - pos.avg_price) * pos.quantity * self.multiplier

    def get_trades(self) -> pd.DataFrame:
        """Get all trades as DataFrame."""
        if not self._trades:
            return pd.DataFrame()
        return pd.DataFrame(self._trades)

    def get_total_pnl(self) -> dict[str, float]:
        """Get total P&L summary."""
        realized = sum(pos.realized_pnl for pos in self._positions.values())
        unrealized = sum(pos.unrealized_pnl for pos in self._positions.values())

        return {
            "realized": realized,
            "unrealized": unrealized,
            "total": realized + unrealized,
        }

    def reset(self):
        """Reset all positions."""
        self._positions.clear()
        self._trades.clear()
