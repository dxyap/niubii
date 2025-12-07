"""
Simulated Broker
================
Simulated broker for paper trading and testing.

Features:
- Realistic order execution simulation
- Configurable slippage and latency
- Position and P&L tracking
- Market simulation
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock

import numpy as np

from core.data.bloomberg import TickerMapper

from .base import Broker, BrokerConfig, BrokerStatus, ExecutionReport

logger = logging.getLogger(__name__)


@dataclass
class SimulatorConfig(BrokerConfig):
    """Configuration for simulated broker."""
    broker_id: str = "simulator"
    name: str = "Paper Trading Simulator"

    # Simulation parameters
    slippage_bps: float = 1.0          # Slippage in basis points
    fill_latency_ms: float = 50.0      # Fill latency in milliseconds
    partial_fill_prob: float = 0.0     # Probability of partial fill
    reject_prob: float = 0.0           # Probability of order rejection

    # Market simulation
    use_realistic_fills: bool = True
    volatility_impact: bool = True     # Higher vol = more slippage
    size_impact: bool = True           # Larger orders = more slippage

    # Initial capital
    initial_capital: float = 1_000_000

    # Contract specs
    contract_multiplier: float = 1000
    commission_per_contract: float = 2.50


@dataclass
class SimulatedPosition:
    """Simulated position tracking."""
    symbol: str
    quantity: int = 0
    avg_price: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    commission_paid: float = 0.0


@dataclass
class SimulatedOrder:
    """Internal order representation."""
    broker_order_id: str
    internal_order_id: str
    symbol: str
    side: str
    quantity: int
    order_type: str
    limit_price: float | None
    stop_price: float | None
    time_in_force: str
    status: str = "PENDING"
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    fills: list[ExecutionReport] = field(default_factory=list)


class SimulatedBroker(Broker):
    """
    Simulated broker for paper trading.

    Provides realistic order execution simulation with:
    - Configurable slippage and latency
    - Position and P&L tracking
    - Order book simulation
    - Commission calculation
    """

    def __init__(self, config: SimulatorConfig | None = None):
        config = config or SimulatorConfig()
        super().__init__(config)
        self.sim_config: SimulatorConfig = config

        # State
        self._orders: dict[str, SimulatedOrder] = {}
        self._positions: dict[str, SimulatedPosition] = {}
        self._account_value = config.initial_capital
        self._cash = config.initial_capital
        self._order_counter = 0
        self._execution_counter = 0

        # Market prices (would be updated by data feed in real implementation)
        self._prices: dict[str, float] = {}

        # Thread safety
        self._lock = Lock()

        # Auto-connect simulator
        self.connect()

    def connect(self) -> bool:
        """Connect to simulated broker (always succeeds)."""
        self.status = BrokerStatus.CONNECTED
        self._on_connect()
        return True

    def disconnect(self):
        """Disconnect from simulated broker."""
        self.status = BrokerStatus.DISCONNECTED
        self._on_disconnect()

    def set_price(self, symbol: str, price: float):
        """Set current price for a symbol."""
        with self._lock:
            self._prices[symbol] = price
            self._update_unrealized_pnl()

    def set_prices(self, prices: dict[str, float]):
        """Set prices for multiple symbols."""
        with self._lock:
            self._prices.update(prices)
            self._update_unrealized_pnl()

    def get_market_price(self, symbol: str) -> float | None:
        """Get current market price for symbol."""
        return self._prices.get(symbol)

    def submit_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str = "MARKET",
        limit_price: float | None = None,
        stop_price: float | None = None,
        time_in_force: str = "DAY",
        account: str | None = None,
        internal_order_id: str | None = None,
        **kwargs,
    ) -> str | None:
        """Submit order to simulated broker."""
        with self._lock:
            # Check for rejection
            if np.random.random() < self.sim_config.reject_prob:
                logger.warning(f"Order rejected (simulated): {symbol} {side} {quantity}")
                self._on_error(100, "Order rejected by risk check")
                return None

            # Generate broker order ID
            self._order_counter += 1
            broker_order_id = f"SIM-{datetime.now().strftime('%Y%m%d')}-{self._order_counter:06d}"

            # Create order
            order = SimulatedOrder(
                broker_order_id=broker_order_id,
                internal_order_id=internal_order_id or broker_order_id,
                symbol=symbol,
                side=side.upper(),
                quantity=quantity,
                order_type=order_type.upper(),
                limit_price=limit_price,
                stop_price=stop_price,
                time_in_force=time_in_force,
                status="WORKING",
            )

            self._orders[broker_order_id] = order

            # Notify order acknowledged
            self._on_order_status(broker_order_id, "ACKNOWLEDGED")

            # Process immediately for market orders
            if order_type.upper() == "MARKET":
                self._execute_market_order(order)
            else:
                # For limit/stop orders, check if they can be filled
                self._check_limit_order(order)

        return broker_order_id

    def _execute_market_order(self, order: SimulatedOrder):
        """Execute a market order."""
        price = self._prices.get(order.symbol)
        if price is None:
            price = 75.0  # Default price if not set
            logger.warning(f"No price for {order.symbol}, using default: {price}")

        # Calculate slippage
        slippage = self._calculate_slippage(order, price)

        if order.side == "BUY":
            fill_price = price + slippage
        else:
            fill_price = price - slippage

        # Execute fill
        self._process_fill(order, order.quantity, fill_price)

    def _check_limit_order(self, order: SimulatedOrder):
        """Check if limit order can be filled."""
        price = self._prices.get(order.symbol)
        if price is None:
            return

        can_fill = False

        if order.order_type == "LIMIT":
            if order.side == "BUY" and order.limit_price and price <= order.limit_price or order.side == "SELL" and order.limit_price and price >= order.limit_price:
                can_fill = True
        elif order.order_type == "STOP":
            if order.side == "BUY" and order.stop_price and price >= order.stop_price or order.side == "SELL" and order.stop_price and price <= order.stop_price:
                can_fill = True

        if can_fill:
            fill_price = order.limit_price or price
            slippage = self._calculate_slippage(order, fill_price)

            if order.side == "BUY":
                fill_price += slippage
            else:
                fill_price -= slippage

            self._process_fill(order, order.quantity, fill_price)

    def _calculate_slippage(self, order: SimulatedOrder, price: float) -> float:
        """Calculate slippage for an order."""
        base_slippage = price * self.sim_config.slippage_bps / 10000

        if self.sim_config.size_impact:
            # Larger orders get more slippage
            size_factor = 1 + (order.quantity / 50) * 0.5  # +50% slippage per 50 contracts
            base_slippage *= min(size_factor, 3.0)

        if self.sim_config.volatility_impact:
            # Add some randomness
            vol_factor = 1 + np.random.uniform(-0.3, 0.3)
            base_slippage *= vol_factor

        return base_slippage

    def _get_multiplier(self, symbol: str) -> int:
        """Resolve contract multiplier for a given symbol."""
        ticker = symbol if " " in symbol else f"{symbol} Comdty"
        try:
            return TickerMapper.get_multiplier(ticker)
        except Exception:
            return self.sim_config.contract_multiplier

    def _process_fill(self, order: SimulatedOrder, quantity: int, price: float):
        """Process a fill."""
        # Calculate commission
        commission = quantity * self.sim_config.commission_per_contract

        # Create execution report
        self._execution_counter += 1
        report = ExecutionReport(
            execution_id=f"EXEC-{self._execution_counter:08d}",
            order_id=order.internal_order_id,
            broker_order_id=order.broker_order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=quantity,
            price=round(price, 4),
            timestamp=datetime.now(),
            commission=commission,
            is_final=True,
            cumulative_quantity=order.filled_quantity + quantity,
            leaves_quantity=order.quantity - order.filled_quantity - quantity,
        )

        # Update order
        if order.filled_quantity == 0:
            order.avg_fill_price = price
        else:
            total_value = order.avg_fill_price * order.filled_quantity + price * quantity
            order.avg_fill_price = total_value / (order.filled_quantity + quantity)

        order.filled_quantity += quantity
        order.fills.append(report)

        if order.filled_quantity >= order.quantity:
            order.status = "FILLED"
        else:
            order.status = "PARTIALLY_FILLED"

        # Update position
        self._update_position(order.symbol, order.side, quantity, price, commission)

        # Update cash with trade value and commissions
        multiplier = self._get_multiplier(order.symbol)
        trade_value = quantity * price * multiplier
        if order.side == "BUY":
            self._cash -= trade_value
        else:
            self._cash += trade_value
        self._cash -= commission

        # Notify
        self._on_fill(report)
        self._on_order_status(order.broker_order_id, order.status)

    def _update_position(self, symbol: str, side: str, quantity: int, price: float, commission: float):
        """Update position after fill."""
        if symbol not in self._positions:
            self._positions[symbol] = SimulatedPosition(symbol=symbol)

        pos = self._positions[symbol]
        multiplier = self._get_multiplier(symbol)
        old_qty = pos.quantity

        if side == "BUY":
            new_qty = old_qty + quantity
        else:
            new_qty = old_qty - quantity

        # Calculate realized P&L for closing trades
        if (old_qty > 0 and side == "SELL") or (old_qty < 0 and side == "BUY"):
            close_qty = min(abs(old_qty), quantity)
            if side == "SELL":
                realized = (price - pos.avg_price) * close_qty * multiplier
            else:
                realized = (pos.avg_price - price) * close_qty * multiplier
            pos.realized_pnl += realized

        # Update average price
        if new_qty == 0:
            pos.avg_price = 0
        elif (old_qty >= 0 and side == "BUY") or (old_qty <= 0 and side == "SELL"):
            # Adding to position
            if old_qty == 0:
                pos.avg_price = price
            else:
                total_value = abs(old_qty) * pos.avg_price + quantity * price
                pos.avg_price = total_value / abs(new_qty)
        # If reducing, keep same average

        pos.quantity = new_qty
        pos.commission_paid += commission

    def _update_unrealized_pnl(self):
        """Update unrealized P&L for all positions."""
        for symbol, pos in self._positions.items():
            if pos.quantity != 0 and symbol in self._prices:
                price = self._prices[symbol]
                multiplier = self._get_multiplier(symbol)
                pos.unrealized_pnl = (price - pos.avg_price) * pos.quantity * multiplier

    def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel an order."""
        with self._lock:
            order = self._orders.get(broker_order_id)
            if not order:
                return False

            if order.status in ["FILLED", "CANCELLED", "REJECTED"]:
                return False

            order.status = "CANCELLED"
            self._on_order_status(broker_order_id, "CANCELLED")

            return True

    def modify_order(
        self,
        broker_order_id: str,
        quantity: int | None = None,
        limit_price: float | None = None,
        stop_price: float | None = None,
    ) -> bool:
        """Modify an order."""
        with self._lock:
            order = self._orders.get(broker_order_id)
            if not order:
                return False

            if order.status in ["FILLED", "CANCELLED", "REJECTED"]:
                return False

            if quantity is not None:
                order.quantity = quantity
            if limit_price is not None:
                order.limit_price = limit_price
            if stop_price is not None:
                order.stop_price = stop_price

            # Re-check if can fill
            if order.order_type != "MARKET":
                self._check_limit_order(order)

            return True

    def get_order_status(self, broker_order_id: str) -> dict | None:
        """Get order status."""
        order = self._orders.get(broker_order_id)
        if not order:
            return None

        return {
            "broker_order_id": order.broker_order_id,
            "symbol": order.symbol,
            "side": order.side,
            "quantity": order.quantity,
            "order_type": order.order_type,
            "status": order.status,
            "filled_quantity": order.filled_quantity,
            "avg_fill_price": order.avg_fill_price,
            "limit_price": order.limit_price,
            "stop_price": order.stop_price,
            "created_at": order.created_at.isoformat(),
        }

    def get_positions(self, account: str | None = None) -> dict[str, dict]:
        """Get all positions."""
        result = {}
        for symbol, pos in self._positions.items():
            if pos.quantity != 0:
                result[symbol] = {
                    "symbol": symbol,
                    "quantity": pos.quantity,
                    "avg_price": pos.avg_price,
                    "market_price": self._prices.get(symbol, pos.avg_price),
                    "unrealized_pnl": pos.unrealized_pnl,
                    "realized_pnl": pos.realized_pnl,
                    "commission_paid": pos.commission_paid,
                }
        return result

    def get_account_info(self, account: str | None = None) -> dict:
        """Get account information."""
        total_unrealized = sum(p.unrealized_pnl for p in self._positions.values())
        total_realized = sum(p.realized_pnl for p in self._positions.values())
        total_commission = sum(p.commission_paid for p in self._positions.values())
        positions_value = 0.0
        for symbol, pos in self._positions.items():
            price = self._prices.get(symbol, pos.avg_price)
            multiplier = self._get_multiplier(symbol)
            positions_value += pos.quantity * price * multiplier

        nav = self._cash + positions_value

        return {
            "account": account or self.sim_config.default_account,
            "initial_capital": self.sim_config.initial_capital,
            "cash": self._cash,
            "nav": nav,
            "unrealized_pnl": total_unrealized,
            "realized_pnl": total_realized,
            "total_commission": total_commission,
            "buying_power": self._cash * 10,  # 10x leverage assumption
            "margin_used": sum(
                abs(p.quantity) * self._prices.get(p.symbol, 0) * self._get_multiplier(p.symbol) * 0.1
                for p in self._positions.values()
            ),
        }

    def process_pending_orders(self):
        """Process all pending limit/stop orders against current prices."""
        with self._lock:
            for order in list(self._orders.values()):
                if order.status == "WORKING" and order.order_type != "MARKET":
                    self._check_limit_order(order)

    def reset(self):
        """Reset simulator to initial state."""
        with self._lock:
            self._orders.clear()
            self._positions.clear()
            self._cash = self.sim_config.initial_capital
            self._order_counter = 0
            self._execution_counter = 0

        logger.info("Simulator reset to initial state")
