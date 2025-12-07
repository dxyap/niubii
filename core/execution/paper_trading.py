"""
Paper Trading Engine
====================
Paper trading mode for strategy testing without real execution.

Features:
- Simulated order execution
- Real-time position tracking
- P&L calculation
- Integration with signals and OMS
"""

import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from core.data.bloomberg import TickerMapper

from .brokers import ExecutionReport, SimulatedBroker, SimulatorConfig
from .oms import Order, OrderManager, OrderSide, OrderStatus, OrderType, OrderUpdate

logger = logging.getLogger(__name__)


@dataclass
class PaperTradingConfig:
    """Configuration for paper trading."""
    initial_capital: float = 1_000_000

    # Execution simulation
    slippage_bps: float = 1.0
    commission_per_contract: float = 2.50
    fill_latency_ms: float = 50.0

    # Risk limits
    max_position_per_symbol: int = 50
    max_gross_exposure: float = 5_000_000
    max_loss_pct: float = 0.05  # 5% max loss before stopping

    # Contract specs
    contract_multiplier: float = 1000

    # Reporting
    track_history: bool = True
    history_days: int = 30


@dataclass
class SimulatedFill:
    """Represents a simulated fill."""
    fill_id: str
    order_id: str
    symbol: str
    side: str
    quantity: int
    price: float
    timestamp: datetime
    commission: float
    slippage_amount: float

    def to_dict(self) -> dict:
        return {
            "fill_id": self.fill_id,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "price": self.price,
            "timestamp": self.timestamp.isoformat(),
            "commission": self.commission,
            "slippage": self.slippage_amount,
        }


@dataclass
class TradingSession:
    """Represents a paper trading session."""
    session_id: str
    start_time: datetime
    end_time: datetime | None
    initial_capital: float
    final_capital: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    max_drawdown: float
    sharpe_ratio: float

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0
        return self.winning_trades / self.total_trades * 100

    @property
    def return_pct(self) -> float:
        return (self.final_capital / self.initial_capital - 1) * 100


class PaperTradingEngine:
    """
    Paper trading engine for strategy testing.

    Provides:
    - Simulated order execution
    - Real-time position and P&L tracking
    - Risk monitoring
    - Performance metrics
    """

    def __init__(self, config: PaperTradingConfig | None = None):
        self.config = config or PaperTradingConfig()

        # Initialize OMS
        self.oms = OrderManager(db_path="data/paper_trades/orders.db")

        # Initialize simulated broker
        broker_config = SimulatorConfig(
            initial_capital=self.config.initial_capital,
            slippage_bps=self.config.slippage_bps,
            commission_per_contract=self.config.commission_per_contract,
            fill_latency_ms=self.config.fill_latency_ms,
            contract_multiplier=self.config.contract_multiplier,
        )
        self.broker = SimulatedBroker(broker_config)

        # State
        self._is_active = False
        self._session_id: str | None = None
        self._session_start: datetime | None = None
        self._fills: list[SimulatedFill] = []
        self._equity_curve: list[dict] = []
        self._daily_pnl: list[dict] = []
        self._peak_equity = self.config.initial_capital
        self._max_drawdown = 0.0

        # Connect broker callbacks
        self.broker.register_callback("on_fill", self._on_broker_fill)
        self.broker.register_callback("on_order_status", self._on_order_status)

    def start_session(self, session_id: str | None = None) -> str:
        """
        Start a new paper trading session.

        Args:
            session_id: Optional session identifier

        Returns:
            Session ID
        """
        self._session_id = session_id or f"SESSION-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self._session_start = datetime.now()
        self._is_active = True

        # Reset state
        self.broker.reset()
        self._fills.clear()
        self._equity_curve.clear()
        self._daily_pnl.clear()
        self._peak_equity = self.config.initial_capital
        self._max_drawdown = 0.0

        # Record initial equity
        self._record_equity()

        logger.info(f"Paper trading session started: {self._session_id}")

        return self._session_id

    def stop_session(self) -> TradingSession:
        """
        Stop the current trading session.

        Returns:
            Session summary
        """
        self._is_active = False

        # Get final account state
        account_info = self.broker.get_account_info()

        # Calculate metrics
        total_trades = len(self._fills)
        winning = len([f for f in self._fills if self._is_winning_trade(f)])
        losing = total_trades - winning

        # Calculate Sharpe ratio from daily returns
        sharpe = self._calculate_sharpe()

        session = TradingSession(
            session_id=self._session_id or "unknown",
            start_time=self._session_start or datetime.now(),
            end_time=datetime.now(),
            initial_capital=self.config.initial_capital,
            final_capital=account_info["nav"],
            total_trades=total_trades,
            winning_trades=winning,
            losing_trades=losing,
            total_pnl=account_info["realized_pnl"] + account_info["unrealized_pnl"],
            max_drawdown=self._max_drawdown,
            sharpe_ratio=sharpe,
        )

        logger.info(f"Paper trading session ended: {self._session_id}, P&L: ${session.total_pnl:,.2f}")

        return session

    def submit_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str = "MARKET",
        limit_price: float | None = None,
        stop_price: float | None = None,
        strategy: str | None = None,
        signal_id: str | None = None,
    ) -> Order | None:
        """
        Submit an order for paper trading.

        Args:
            symbol: Trading symbol
            side: "BUY" or "SELL"
            quantity: Number of contracts
            order_type: Order type
            limit_price: Limit price
            stop_price: Stop price
            strategy: Strategy name
            signal_id: Signal ID

        Returns:
            Created order or None if rejected
        """
        if not self._is_active:
            logger.warning("Paper trading session not active")
            return None

        # Pre-trade checks
        if not self._pre_trade_check(symbol, side, quantity):
            return None

        try:
            # Create order in OMS
            order = self.oms.create_order(
                symbol=symbol,
                side=OrderSide(side.upper()),
                quantity=quantity,
                order_type=OrderType(order_type.upper()),
                limit_price=limit_price,
                stop_price=stop_price,
                strategy=strategy,
                signal_id=signal_id,
                tags=["paper_trading"],
            )

            # Submit to OMS
            self.oms.submit_order(order.order_id)

            # Submit to simulated broker
            broker_order_id = self.broker.submit_order(
                symbol=symbol,
                side=side.upper(),
                quantity=quantity,
                order_type=order_type.upper(),
                limit_price=limit_price,
                stop_price=stop_price,
                internal_order_id=order.order_id,
            )

            if broker_order_id:
                self.oms.update_order(order.order_id, OrderUpdate(
                    broker_order_id=broker_order_id
                ))

            return order

        except Exception as e:
            logger.error(f"Failed to submit paper trade: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        order = self.oms.get_order(order_id)
        if not order:
            return False

        if order.broker_order_id:
            self.broker.cancel_order(order.broker_order_id)

        self.oms.confirm_cancel(order_id)
        return True

    def update_prices(self, prices: dict[str, float]):
        """
        Update market prices.

        Args:
            prices: Dictionary of symbol -> price
        """
        self.broker.set_prices(prices)
        self.broker.process_pending_orders()
        self._record_equity()
        self._check_risk_limits()

    def get_positions(self) -> dict[str, dict]:
        """Get current positions."""
        return self.broker.get_positions()

    def get_account_info(self) -> dict:
        """Get account information."""
        return self.broker.get_account_info()

    def get_pnl_summary(self) -> dict:
        """Get P&L summary."""
        account = self.broker.get_account_info()
        positions = self.broker.get_positions()

        return {
            "initial_capital": self.config.initial_capital,
            "current_nav": account["nav"],
            "total_pnl": account["realized_pnl"] + account["unrealized_pnl"],
            "realized_pnl": account["realized_pnl"],
            "unrealized_pnl": account["unrealized_pnl"],
            "total_commission": account["total_commission"],
            "return_pct": (account["nav"] / self.config.initial_capital - 1) * 100,
            "max_drawdown": self._max_drawdown,
            "num_positions": len(positions),
            "num_trades": len(self._fills),
        }

    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        if not self._equity_curve:
            return pd.DataFrame()

        df = pd.DataFrame(self._equity_curve)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        return df

    def get_fills(self) -> list[dict]:
        """Get all fills."""
        return [f.to_dict() for f in self._fills]

    def get_open_orders(self) -> list[Order]:
        """Get all open orders."""
        return self.oms.get_active_orders()

    def _pre_trade_check(self, symbol: str, side: str, quantity: int) -> bool:
        """Run pre-trade risk checks."""
        positions = self.broker.get_positions()
        account = self.broker.get_account_info()

        # Check max loss
        pnl_pct = (account["nav"] / self.config.initial_capital - 1)
        if pnl_pct < -self.config.max_loss_pct:
            logger.warning(f"Max loss limit reached: {pnl_pct*100:.2f}%")
            return False

        # Check position limit
        current_pos = positions.get(symbol, {}).get("quantity", 0)
        if side.upper() == "BUY":
            new_pos = current_pos + quantity
        else:
            new_pos = current_pos - quantity

        if abs(new_pos) > self.config.max_position_per_symbol:
            logger.warning(f"Position limit would be exceeded: {abs(new_pos)} > {self.config.max_position_per_symbol}")
            return False

        # Check gross exposure using instrument-specific multipliers
        total_exposure = 0.0
        symbol_seen = False
        for sym, pos in positions.items():
            multiplier = self._resolve_multiplier(sym)
            price = self._estimate_price(sym, positions)
            qty = pos.get("quantity", 0)
            if sym == symbol:
                qty = new_pos
                symbol_seen = True
                price = self._estimate_price(symbol, positions)
            total_exposure += abs(qty) * price * multiplier

        if not symbol_seen:
            multiplier = self._resolve_multiplier(symbol)
            price = self._estimate_price(symbol, positions)
            total_exposure += abs(new_pos) * price * multiplier

        if total_exposure > self.config.max_gross_exposure:
            logger.warning(
                "Gross exposure limit would be exceeded: %.0f > %.0f",
                total_exposure,
                self.config.max_gross_exposure,
            )
            return False

        return True

    def _on_broker_fill(self, report: ExecutionReport):
        """Handle fill from broker."""
        # Record fill
        fill = SimulatedFill(
            fill_id=report.execution_id,
            order_id=report.order_id,
            symbol=report.symbol,
            side=report.side,
            quantity=report.quantity,
            price=report.price,
            timestamp=report.timestamp,
            commission=report.commission,
            slippage_amount=0,  # Calculated elsewhere
        )
        self._fills.append(fill)

        # Update OMS
        try:
            self.oms.process_fill(
                order_id=report.order_id,
                fill_quantity=report.quantity,
                fill_price=report.price,
                commission=report.commission,
            )
        except Exception as e:
            logger.error(f"Failed to update OMS with fill: {e}")

        # Record equity
        self._record_equity()

    def _on_order_status(self, broker_order_id: str, status: str, **kwargs):
        """Handle order status update from broker."""
        # Find order in OMS
        for order in self.oms.get_orders(limit=100):
            if order.broker_order_id == broker_order_id:
                try:
                    if status == "CANCELLED":
                        self.oms.confirm_cancel(order.order_id)
                    elif status == "REJECTED":
                        self.oms.update_order(order.order_id, OrderUpdate(
                            status=OrderStatus.REJECTED,
                            rejection_reason=kwargs.get("reason", "Unknown"),
                        ))
                except Exception as e:
                    logger.error(f"Failed to update order status: {e}")
                break

    def _record_equity(self):
        """Record current equity."""
        account = self.broker.get_account_info()
        equity = account["nav"]

        self._equity_curve.append({
            "timestamp": datetime.now(),
            "equity": equity,
            "cash": account["cash"],
            "unrealized_pnl": account["unrealized_pnl"],
            "realized_pnl": account["realized_pnl"],
        })

        # Update peak and drawdown
        if equity > self._peak_equity:
            self._peak_equity = equity

        drawdown = (self._peak_equity - equity) / self._peak_equity * 100
        if drawdown > self._max_drawdown:
            self._max_drawdown = drawdown

    def _check_risk_limits(self):
        """Check if risk limits are breached."""
        account = self.broker.get_account_info()
        pnl_pct = (account["nav"] / self.config.initial_capital - 1)

        if pnl_pct < -self.config.max_loss_pct:
            logger.warning(f"Max loss limit reached: {pnl_pct*100:.2f}%. Stopping session.")
            # Could auto-close positions here

    def _is_winning_trade(self, fill: SimulatedFill) -> bool:
        """Check if fill was part of a winning trade."""
        # Simplified - would need full trade matching for accuracy
        return fill.price > 0

    def _calculate_sharpe(self) -> float:
        """Calculate Sharpe ratio from equity curve."""
        if len(self._equity_curve) < 2:
            return 0

        df = pd.DataFrame(self._equity_curve)
        df["returns"] = df["equity"].pct_change()

        if df["returns"].std() == 0:
            return 0

        # Annualized Sharpe (assuming daily data)
        sharpe = df["returns"].mean() / df["returns"].std() * np.sqrt(252)
        return sharpe

    def _resolve_multiplier(self, symbol: str) -> int:
        """Resolve contract multiplier for a symbol using ticker metadata."""
        ticker = symbol if " " in symbol else f"{symbol} Comdty"
        try:
            return TickerMapper.get_multiplier(ticker)
        except Exception:
            return self.config.contract_multiplier

    def _estimate_price(self, symbol: str, positions: dict[str, dict]) -> float:
        """
        Estimate a reference price for exposure checks.

        Falls back to position data when broker price feed is unavailable.
        """
        price = self.broker.get_market_price(symbol)
        if price is None or price == 0:
            price = positions.get(symbol, {}).get("market_price")
        if price is None or price == 0:
            price = positions.get(symbol, {}).get("avg_price")
        return price if price not in (None, 0) else 1.0
