"""
Order Management System (OMS)
=============================
Complete order lifecycle management for trading operations.

Features:
- Order creation, validation, and submission
- Order status tracking and updates
- Order history and audit trail
- Risk checks integration
- Event-driven order updates
"""

import json
import logging
import sqlite3
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order status enumeration."""
    CREATED = "CREATED"           # Order created, not yet submitted
    PENDING = "PENDING"           # Submitted, awaiting acknowledgement
    ACKNOWLEDGED = "ACKNOWLEDGED" # Broker acknowledged receipt
    WORKING = "WORKING"           # Order is working in the market
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"             # Completely filled
    CANCELLED = "CANCELLED"       # User cancelled
    REJECTED = "REJECTED"         # Broker rejected
    EXPIRED = "EXPIRED"           # Order expired (e.g., day orders)
    PENDING_CANCEL = "PENDING_CANCEL"  # Cancel request pending
    PENDING_REPLACE = "PENDING_REPLACE"  # Replace request pending
    ERROR = "ERROR"               # System error


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    MOO = "MARKET_ON_OPEN"       # Market on open
    MOC = "MARKET_ON_CLOSE"     # Market on close
    LOO = "LIMIT_ON_OPEN"
    LOC = "LIMIT_ON_CLOSE"


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "BUY"
    SELL = "SELL"


class TimeInForce(Enum):
    """Time in force enumeration."""
    DAY = "DAY"                 # Day order
    GTC = "GTC"                 # Good till cancelled
    IOC = "IOC"                 # Immediate or cancel
    FOK = "FOK"                 # Fill or kill
    GTD = "GTD"                 # Good till date
    OPG = "OPG"                 # At the opening
    CLS = "CLS"                 # At the close


@dataclass
class Order:
    """
    Represents a trading order.

    Tracks all order attributes and state throughout its lifecycle.
    """
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    time_in_force: TimeInForce = TimeInForce.DAY

    # Order state
    status: OrderStatus = OrderStatus.CREATED
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    remaining_quantity: int = 0

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    submitted_at: datetime | None = None
    filled_at: datetime | None = None
    cancelled_at: datetime | None = None
    last_update: datetime = field(default_factory=datetime.now)

    # Metadata
    strategy: str | None = None
    signal_id: str | None = None
    account: str = "MAIN"
    notes: str | None = None
    parent_order_id: str | None = None  # For algo slices
    algo_id: str | None = None

    # Execution details
    broker_order_id: str | None = None
    commission: float = 0.0
    slippage: float = 0.0
    rejection_reason: str | None = None

    # Tags for filtering
    tags: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.remaining_quantity = self.quantity - self.filled_quantity

    @property
    def is_active(self) -> bool:
        """Check if order is still active."""
        return self.status in [
            OrderStatus.PENDING,
            OrderStatus.ACKNOWLEDGED,
            OrderStatus.WORKING,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.PENDING_CANCEL,
            OrderStatus.PENDING_REPLACE,
        ]

    @property
    def is_complete(self) -> bool:
        """Check if order is complete (filled, cancelled, or rejected)."""
        return self.status in [
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        ]

    @property
    def is_filled(self) -> bool:
        """Check if order is fully filled."""
        return self.status == OrderStatus.FILLED

    @property
    def fill_pct(self) -> float:
        """Get fill percentage."""
        return (self.filled_quantity / self.quantity * 100) if self.quantity > 0 else 0.0

    @property
    def notional_value(self) -> float:
        """Calculate notional value."""
        price = self.avg_fill_price if self.avg_fill_price > 0 else (self.limit_price or 0)
        return self.quantity * price * 1000  # Assuming 1000 barrel contracts

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "order_type": self.order_type.value,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "time_in_force": self.time_in_force.value,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "avg_fill_price": self.avg_fill_price,
            "remaining_quantity": self.remaining_quantity,
            "created_at": self.created_at.isoformat(),
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "strategy": self.strategy,
            "signal_id": self.signal_id,
            "account": self.account,
            "commission": self.commission,
            "slippage": self.slippage,
            "rejection_reason": self.rejection_reason,
            "broker_order_id": self.broker_order_id,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Order":
        """Create order from dictionary."""
        return cls(
            order_id=data["order_id"],
            symbol=data["symbol"],
            side=OrderSide(data["side"]),
            quantity=data["quantity"],
            order_type=OrderType(data.get("order_type", "MARKET")),
            limit_price=data.get("limit_price"),
            stop_price=data.get("stop_price"),
            time_in_force=TimeInForce(data.get("time_in_force", "DAY")),
            status=OrderStatus(data.get("status", "CREATED")),
            filled_quantity=data.get("filled_quantity", 0),
            avg_fill_price=data.get("avg_fill_price", 0.0),
            strategy=data.get("strategy"),
            signal_id=data.get("signal_id"),
            account=data.get("account", "MAIN"),
            commission=data.get("commission", 0.0),
            tags=data.get("tags", []),
        )


@dataclass
class OrderEvent:
    """Represents an order event/update."""
    event_id: str
    order_id: str
    event_type: str  # e.g., "SUBMITTED", "FILL", "CANCELLED", etc.
    timestamp: datetime
    details: dict = field(default_factory=dict)


@dataclass
class OrderUpdate:
    """Represents an update to apply to an order."""
    status: OrderStatus | None = None
    filled_quantity: int | None = None
    avg_fill_price: float | None = None
    rejection_reason: str | None = None
    broker_order_id: str | None = None
    commission: float | None = None


class OrderManager:
    """
    Order Management System (OMS).

    Manages the complete lifecycle of orders:
    - Order creation and validation
    - Order submission and tracking
    - Fill processing and position updates
    - Order history and audit trail
    """

    def __init__(
        self,
        db_path: str = "data/orders/orders.db",
        risk_checker: Callable | None = None,
    ):
        """
        Initialize Order Manager.

        Args:
            db_path: Path to orders database
            risk_checker: Optional function for pre-trade risk checks
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.risk_checker = risk_checker

        # In-memory order cache
        self._orders: dict[str, Order] = {}
        self._events: list[OrderEvent] = []
        self._order_counter = 0

        # Callbacks for order updates
        self._callbacks: list[Callable[[Order, OrderEvent], None]] = []

        # Initialize database
        self._init_database()
        self._load_active_orders()

    def _init_database(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Orders table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                order_type TEXT NOT NULL,
                limit_price REAL,
                stop_price REAL,
                time_in_force TEXT,
                status TEXT NOT NULL,
                filled_quantity INTEGER DEFAULT 0,
                avg_fill_price REAL DEFAULT 0,
                remaining_quantity INTEGER,
                created_at TIMESTAMP,
                submitted_at TIMESTAMP,
                filled_at TIMESTAMP,
                cancelled_at TIMESTAMP,
                last_update TIMESTAMP,
                strategy TEXT,
                signal_id TEXT,
                account TEXT DEFAULT 'MAIN',
                broker_order_id TEXT,
                commission REAL DEFAULT 0,
                slippage REAL DEFAULT 0,
                rejection_reason TEXT,
                parent_order_id TEXT,
                algo_id TEXT,
                notes TEXT,
                tags TEXT
            )
        """)

        # Order events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS order_events (
                event_id TEXT PRIMARY KEY,
                order_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                details TEXT,
                FOREIGN KEY (order_id) REFERENCES orders(order_id)
            )
        """)

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_strategy ON orders(strategy)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_order ON order_events(order_id)")

        conn.commit()
        conn.close()

    def _load_active_orders(self):
        """Load active orders from database into memory."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get active orders
        active_statuses = [s.value for s in [
            OrderStatus.PENDING,
            OrderStatus.ACKNOWLEDGED,
            OrderStatus.WORKING,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.PENDING_CANCEL,
        ]]

        placeholders = ",".join("?" * len(active_statuses))
        cursor.execute(f"""
            SELECT * FROM orders WHERE status IN ({placeholders})
        """, active_statuses)

        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]

        for row in rows:
            data = dict(zip(columns, row))
            order = Order(
                order_id=data["order_id"],
                symbol=data["symbol"],
                side=OrderSide(data["side"]),
                quantity=data["quantity"],
                order_type=OrderType(data["order_type"]),
                limit_price=data.get("limit_price"),
                stop_price=data.get("stop_price"),
                time_in_force=TimeInForce(data.get("time_in_force", "DAY")),
                status=OrderStatus(data["status"]),
                filled_quantity=data.get("filled_quantity", 0),
                avg_fill_price=data.get("avg_fill_price", 0.0),
                strategy=data.get("strategy"),
                signal_id=data.get("signal_id"),
                account=data.get("account", "MAIN"),
                broker_order_id=data.get("broker_order_id"),
                tags=json.loads(data.get("tags", "[]")),
            )
            self._orders[order.order_id] = order

        conn.close()
        logger.info(f"Loaded {len(self._orders)} active orders from database")

    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        stop_price: float | None = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
        strategy: str | None = None,
        signal_id: str | None = None,
        account: str = "MAIN",
        notes: str | None = None,
        tags: list[str] | None = None,
        validate: bool = True,
    ) -> Order:
        """
        Create a new order.

        Args:
            symbol: Trading symbol (e.g., "CL1")
            side: Buy or sell
            quantity: Number of contracts
            order_type: Order type (market, limit, etc.)
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            time_in_force: Order duration
            strategy: Strategy name/tag
            signal_id: Associated signal ID
            account: Trading account
            notes: Order notes
            tags: Tags for filtering
            validate: Whether to validate order

        Returns:
            Created order

        Raises:
            ValueError: If order validation fails
        """
        self._order_counter += 1
        order_id = f"ORD-{datetime.now().strftime('%Y%m%d')}-{self._order_counter:06d}"

        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            status=OrderStatus.CREATED,
            strategy=strategy,
            signal_id=signal_id,
            account=account,
            notes=notes,
            tags=tags or [],
        )

        if validate:
            self._validate_order(order)

        # Store in memory
        self._orders[order_id] = order

        # Record event
        self._record_event(order, "CREATED")

        # Persist to database
        self._save_order(order)

        logger.info(f"Created order: {order_id} - {side.value} {quantity} {symbol}")

        return order

    def _validate_order(self, order: Order):
        """Validate order parameters."""
        if order.quantity <= 0:
            raise ValueError("Order quantity must be positive")

        if order.order_type == OrderType.LIMIT and order.limit_price is None:
            raise ValueError("Limit price required for limit orders")

        if order.order_type == OrderType.STOP and order.stop_price is None:
            raise ValueError("Stop price required for stop orders")

        if order.order_type == OrderType.STOP_LIMIT and (order.limit_price is None or order.stop_price is None):
            raise ValueError("Both limit and stop prices required for stop-limit orders")

        # Run pre-trade risk checks if available
        if self.risk_checker:
            result = self.risk_checker(order)
            if not result.get("approved", True):
                raise ValueError(f"Risk check failed: {result.get('reason', 'Unknown')}")

    def submit_order(self, order_id: str) -> Order:
        """
        Submit an order for execution.

        Args:
            order_id: Order ID to submit

        Returns:
            Updated order
        """
        order = self.get_order(order_id)
        if not order:
            raise ValueError(f"Order not found: {order_id}")

        if order.status != OrderStatus.CREATED:
            raise ValueError(f"Cannot submit order in status: {order.status.value}")

        order.status = OrderStatus.PENDING
        order.submitted_at = datetime.now()
        order.last_update = datetime.now()

        self._record_event(order, "SUBMITTED")
        self._save_order(order)
        self._notify_callbacks(order)

        logger.info(f"Submitted order: {order_id}")

        return order

    def update_order(self, order_id: str, update: OrderUpdate) -> Order:
        """
        Update an order with new information.

        Args:
            order_id: Order ID to update
            update: Update to apply

        Returns:
            Updated order
        """
        order = self.get_order(order_id)
        if not order:
            raise ValueError(f"Order not found: {order_id}")

        old_status = order.status

        if update.status:
            order.status = update.status

        if update.filled_quantity is not None:
            order.filled_quantity = update.filled_quantity
            order.remaining_quantity = order.quantity - order.filled_quantity

        if update.avg_fill_price is not None:
            order.avg_fill_price = update.avg_fill_price

        if update.rejection_reason:
            order.rejection_reason = update.rejection_reason

        if update.broker_order_id:
            order.broker_order_id = update.broker_order_id

        if update.commission is not None:
            order.commission = update.commission

        order.last_update = datetime.now()

        # Set filled_at if newly filled
        if update.status == OrderStatus.FILLED and old_status != OrderStatus.FILLED:
            order.filled_at = datetime.now()

        # Record event
        event_type = "STATUS_CHANGE" if update.status else "UPDATE"
        if update.status == OrderStatus.FILLED:
            event_type = "FILLED"
        elif update.filled_quantity and update.filled_quantity > 0:
            event_type = "FILL"

        self._record_event(order, event_type, {
            "old_status": old_status.value,
            "new_status": order.status.value,
            "filled_qty": update.filled_quantity,
            "avg_price": update.avg_fill_price,
        })

        self._save_order(order)
        self._notify_callbacks(order)

        return order

    def cancel_order(self, order_id: str, reason: str = "User requested") -> Order:
        """
        Cancel an order.

        Args:
            order_id: Order ID to cancel
            reason: Cancellation reason

        Returns:
            Updated order
        """
        order = self.get_order(order_id)
        if not order:
            raise ValueError(f"Order not found: {order_id}")

        if not order.is_active:
            raise ValueError(f"Cannot cancel order in status: {order.status.value}")

        order.status = OrderStatus.PENDING_CANCEL
        order.last_update = datetime.now()

        self._record_event(order, "CANCEL_REQUESTED", {"reason": reason})
        self._save_order(order)

        logger.info(f"Cancel requested for order: {order_id}")

        return order

    def confirm_cancel(self, order_id: str) -> Order:
        """
        Confirm order cancellation.

        Args:
            order_id: Order ID to confirm cancel

        Returns:
            Updated order
        """
        order = self.get_order(order_id)
        if not order:
            raise ValueError(f"Order not found: {order_id}")

        order.status = OrderStatus.CANCELLED
        order.cancelled_at = datetime.now()
        order.last_update = datetime.now()

        self._record_event(order, "CANCELLED")
        self._save_order(order)
        self._notify_callbacks(order)

        logger.info(f"Order cancelled: {order_id}")

        return order

    def process_fill(
        self,
        order_id: str,
        fill_quantity: int,
        fill_price: float,
        commission: float = 0,
    ) -> Order:
        """
        Process a fill for an order.

        Args:
            order_id: Order ID that was filled
            fill_quantity: Quantity filled
            fill_price: Execution price
            commission: Commission for this fill

        Returns:
            Updated order
        """
        order = self.get_order(order_id)
        if not order:
            raise ValueError(f"Order not found: {order_id}")

        if not order.is_active:
            raise ValueError(f"Cannot fill order in status: {order.status.value}")

        # Calculate new average price
        old_filled = order.filled_quantity
        old_value = old_filled * order.avg_fill_price if old_filled > 0 else 0
        new_value = old_value + (fill_quantity * fill_price)
        new_filled = old_filled + fill_quantity

        order.filled_quantity = new_filled
        order.avg_fill_price = new_value / new_filled if new_filled > 0 else 0
        order.remaining_quantity = order.quantity - new_filled
        order.commission += commission
        order.last_update = datetime.now()

        # Update status
        if order.remaining_quantity == 0:
            order.status = OrderStatus.FILLED
            order.filled_at = datetime.now()
        else:
            order.status = OrderStatus.PARTIALLY_FILLED

        self._record_event(order, "FILL", {
            "fill_qty": fill_quantity,
            "fill_price": fill_price,
            "commission": commission,
            "total_filled": order.filled_quantity,
        })

        self._save_order(order)
        self._notify_callbacks(order)

        logger.info(f"Fill processed: {order_id} - {fill_quantity} @ {fill_price}")

        return order

    def get_order(self, order_id: str) -> Order | None:
        """Get order by ID."""
        return self._orders.get(order_id)

    def get_orders(
        self,
        status: OrderStatus | None = None,
        symbol: str | None = None,
        strategy: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        active_only: bool = False,
        limit: int = 100,
    ) -> list[Order]:
        """
        Get orders matching filters.

        Args:
            status: Filter by status
            symbol: Filter by symbol
            strategy: Filter by strategy
            start_date: Filter by start date
            end_date: Filter by end date
            active_only: Only active orders
            limit: Maximum results

        Returns:
            List of matching orders
        """
        orders = list(self._orders.values())

        # Apply filters
        if status:
            orders = [o for o in orders if o.status == status]

        if active_only:
            orders = [o for o in orders if o.is_active]

        if symbol:
            orders = [o for o in orders if o.symbol == symbol]

        if strategy:
            orders = [o for o in orders if o.strategy == strategy]

        if start_date:
            orders = [o for o in orders if o.created_at >= start_date]

        if end_date:
            orders = [o for o in orders if o.created_at <= end_date]

        # Sort by creation time
        orders.sort(key=lambda x: x.created_at, reverse=True)

        return orders[:limit]

    def get_active_orders(self, symbol: str | None = None) -> list[Order]:
        """Get all active orders."""
        return self.get_orders(symbol=symbol, active_only=True)

    def get_order_history(
        self,
        order_id: str,
        limit: int = 50,
    ) -> list[OrderEvent]:
        """Get order event history."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM order_events
            WHERE order_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (order_id, limit))

        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        conn.close()

        events = []
        for row in rows:
            data = dict(zip(columns, row))
            events.append(OrderEvent(
                event_id=data["event_id"],
                order_id=data["order_id"],
                event_type=data["event_type"],
                timestamp=datetime.fromisoformat(data["timestamp"]),
                details=json.loads(data.get("details", "{}")),
            ))

        return events

    def get_statistics(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> dict:
        """
        Get order statistics.

        Args:
            start_date: Start date filter
            end_date: End date filter

        Returns:
            Statistics dictionary
        """
        orders = self.get_orders(
            start_date=start_date,
            end_date=end_date,
            limit=10000,
        )

        if not orders:
            return {
                "total_orders": 0,
                "filled": 0,
                "cancelled": 0,
                "rejected": 0,
                "fill_rate": 0,
                "total_volume": 0,
                "total_commission": 0,
            }

        filled = len([o for o in orders if o.status == OrderStatus.FILLED])
        cancelled = len([o for o in orders if o.status == OrderStatus.CANCELLED])
        rejected = len([o for o in orders if o.status == OrderStatus.REJECTED])

        total = len(orders)
        fill_rate = filled / total * 100 if total > 0 else 0

        total_volume = sum(o.filled_quantity for o in orders)
        total_commission = sum(o.commission for o in orders)

        # By symbol
        by_symbol = {}
        for o in orders:
            if o.symbol not in by_symbol:
                by_symbol[o.symbol] = {"count": 0, "volume": 0}
            by_symbol[o.symbol]["count"] += 1
            by_symbol[o.symbol]["volume"] += o.filled_quantity

        # By strategy
        by_strategy = {}
        for o in orders:
            strategy = o.strategy or "Manual"
            if strategy not in by_strategy:
                by_strategy[strategy] = {"count": 0, "volume": 0}
            by_strategy[strategy]["count"] += 1
            by_strategy[strategy]["volume"] += o.filled_quantity

        return {
            "total_orders": total,
            "filled": filled,
            "cancelled": cancelled,
            "rejected": rejected,
            "active": len([o for o in orders if o.is_active]),
            "fill_rate": round(fill_rate, 1),
            "total_volume": total_volume,
            "total_commission": round(total_commission, 2),
            "by_symbol": by_symbol,
            "by_strategy": by_strategy,
        }

    def register_callback(self, callback: Callable[[Order, OrderEvent], None]):
        """Register callback for order updates."""
        self._callbacks.append(callback)

    def _notify_callbacks(self, order: Order):
        """Notify all callbacks of order update."""
        if self._events:
            event = self._events[-1]
            for callback in self._callbacks:
                try:
                    callback(order, event)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

    def _record_event(self, order: Order, event_type: str, details: dict | None = None):
        """Record an order event."""
        event = OrderEvent(
            event_id=f"EVT-{uuid.uuid4().hex[:12]}",
            order_id=order.order_id,
            event_type=event_type,
            timestamp=datetime.now(),
            details=details or {},
        )
        self._events.append(event)

        # Save to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO order_events (event_id, order_id, event_type, timestamp, details)
            VALUES (?, ?, ?, ?, ?)
        """, (
            event.event_id,
            event.order_id,
            event.event_type,
            event.timestamp.isoformat(),
            json.dumps(event.details),
        ))

        conn.commit()
        conn.close()

    def _save_order(self, order: Order):
        """Save order to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO orders (
                order_id, symbol, side, quantity, order_type, limit_price, stop_price,
                time_in_force, status, filled_quantity, avg_fill_price, remaining_quantity,
                created_at, submitted_at, filled_at, cancelled_at, last_update,
                strategy, signal_id, account, broker_order_id, commission, slippage,
                rejection_reason, parent_order_id, algo_id, notes, tags
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            order.order_id,
            order.symbol,
            order.side.value,
            order.quantity,
            order.order_type.value,
            order.limit_price,
            order.stop_price,
            order.time_in_force.value,
            order.status.value,
            order.filled_quantity,
            order.avg_fill_price,
            order.remaining_quantity,
            order.created_at.isoformat(),
            order.submitted_at.isoformat() if order.submitted_at else None,
            order.filled_at.isoformat() if order.filled_at else None,
            order.cancelled_at.isoformat() if order.cancelled_at else None,
            order.last_update.isoformat(),
            order.strategy,
            order.signal_id,
            order.account,
            order.broker_order_id,
            order.commission,
            order.slippage,
            order.rejection_reason,
            order.parent_order_id,
            order.algo_id,
            order.notes,
            json.dumps(order.tags),
        ))

        conn.commit()
        conn.close()

    def cancel_all_orders(self, symbol: str | None = None, strategy: str | None = None):
        """Cancel all active orders matching filters."""
        orders = self.get_active_orders(symbol=symbol)

        if strategy:
            orders = [o for o in orders if o.strategy == strategy]

        for order in orders:
            try:
                self.cancel_order(order.order_id, "Bulk cancel")
            except Exception as e:
                logger.error(f"Failed to cancel order {order.order_id}: {e}")

        logger.info(f"Cancelled {len(orders)} orders")
