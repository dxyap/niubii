"""
Broker Base Interface
=====================
Abstract base class for broker integrations.

Provides:
- Unified broker interface
- Common order handling
- Connection management
- Execution reporting
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class BrokerStatus(Enum):
    """Broker connection status."""
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    ERROR = "ERROR"
    RECONNECTING = "RECONNECTING"


@dataclass
class BrokerConfig:
    """Configuration for broker connection."""
    broker_id: str = "default"
    name: str = "Default Broker"

    # Connection
    host: str = "localhost"
    port: int = 7496
    client_id: int = 1

    # Timeouts
    connection_timeout: float = 30.0
    request_timeout: float = 10.0

    # Order handling
    default_account: str = "MAIN"
    route: str = "SMART"

    # Limits
    max_orders_per_second: int = 10
    max_pending_orders: int = 100


@dataclass
class ExecutionReport:
    """Report for an order execution."""
    execution_id: str
    order_id: str
    broker_order_id: str
    symbol: str
    side: str
    quantity: int
    price: float
    timestamp: datetime

    # Additional details
    commission: float = 0.0
    exchange: str | None = None
    liquidity_flag: str | None = None  # "ADD" or "REMOVE"

    # Status
    is_final: bool = True
    cumulative_quantity: int = 0
    leaves_quantity: int = 0

    def to_dict(self) -> dict:
        return {
            "execution_id": self.execution_id,
            "order_id": self.order_id,
            "broker_order_id": self.broker_order_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "price": self.price,
            "timestamp": self.timestamp.isoformat(),
            "commission": self.commission,
            "exchange": self.exchange,
            "is_final": self.is_final,
        }


class Broker(ABC):
    """
    Abstract base class for broker integrations.

    Provides a unified interface for order submission and management
    across different brokers.
    """

    def __init__(self, config: BrokerConfig):
        self.config = config
        self.status = BrokerStatus.DISCONNECTED
        self._callbacks: dict[str, list[Callable]] = {
            "on_connect": [],
            "on_disconnect": [],
            "on_order_status": [],
            "on_fill": [],
            "on_error": [],
        }

    @property
    def is_connected(self) -> bool:
        return self.status == BrokerStatus.CONNECTED

    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to broker.

        Returns:
            True if connection successful
        """
        pass

    @abstractmethod
    def disconnect(self):
        """Disconnect from broker."""
        pass

    @abstractmethod
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
        **kwargs,
    ) -> str | None:
        """
        Submit order to broker.

        Args:
            symbol: Trading symbol
            side: "BUY" or "SELL"
            quantity: Number of contracts
            order_type: Order type
            limit_price: Limit price
            stop_price: Stop price
            time_in_force: Order duration
            account: Trading account

        Returns:
            Broker order ID or None if failed
        """
        pass

    @abstractmethod
    def cancel_order(self, broker_order_id: str) -> bool:
        """
        Cancel an order.

        Args:
            broker_order_id: Broker's order ID

        Returns:
            True if cancel request submitted successfully
        """
        pass

    @abstractmethod
    def modify_order(
        self,
        broker_order_id: str,
        quantity: int | None = None,
        limit_price: float | None = None,
        stop_price: float | None = None,
    ) -> bool:
        """
        Modify an existing order.

        Args:
            broker_order_id: Order to modify
            quantity: New quantity
            limit_price: New limit price
            stop_price: New stop price

        Returns:
            True if modification submitted successfully
        """
        pass

    @abstractmethod
    def get_order_status(self, broker_order_id: str) -> dict | None:
        """
        Get order status from broker.

        Args:
            broker_order_id: Order ID

        Returns:
            Order status dictionary
        """
        pass

    @abstractmethod
    def get_positions(self, account: str | None = None) -> dict[str, dict]:
        """
        Get current positions.

        Args:
            account: Account to query

        Returns:
            Dictionary of positions by symbol
        """
        pass

    @abstractmethod
    def get_account_info(self, account: str | None = None) -> dict:
        """
        Get account information.

        Args:
            account: Account to query

        Returns:
            Account information dictionary
        """
        pass

    def get_market_price(self, symbol: str) -> float | None:
        """
        Get current market price.

        Args:
            symbol: Symbol to query

        Returns:
            Current price or None
        """
        # Default implementation - override for real brokers
        return None

    def register_callback(self, event: str, callback: Callable):
        """Register callback for broker events."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def _notify_callbacks(self, event: str, *args, **kwargs):
        """Notify all callbacks for an event."""
        for callback in self._callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Callback error for {event}: {e}")

    def _on_connect(self):
        """Handle connection established."""
        self.status = BrokerStatus.CONNECTED
        self._notify_callbacks("on_connect")
        logger.info(f"Connected to broker: {self.config.name}")

    def _on_disconnect(self):
        """Handle disconnection."""
        self.status = BrokerStatus.DISCONNECTED
        self._notify_callbacks("on_disconnect")
        logger.info(f"Disconnected from broker: {self.config.name}")

    def _on_order_status(self, broker_order_id: str, status: str, **kwargs):
        """Handle order status update."""
        self._notify_callbacks("on_order_status", broker_order_id, status, **kwargs)

    def _on_fill(self, report: ExecutionReport):
        """Handle fill notification."""
        self._notify_callbacks("on_fill", report)
        logger.info(f"Fill: {report.symbol} {report.side} {report.quantity} @ {report.price}")

    def _on_error(self, error_code: int, error_msg: str, **kwargs):
        """Handle error from broker."""
        self._notify_callbacks("on_error", error_code, error_msg, **kwargs)
        logger.error(f"Broker error [{error_code}]: {error_msg}")
