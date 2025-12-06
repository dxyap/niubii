"""
Automation Rules Engine
=======================
Automated trading based on signals and conditions.

⚠️ SIMULATION ONLY: All trading operations are paper trading simulations.
There is NO connection to real brokers or exchanges. No real trades are executed.
This module is for strategy testing and educational purposes only.

Features:
- Rule-based order generation (paper trading)
- Signal-to-order conversion (simulated)
- Multi-condition logic
- Scheduling and timing
- Risk integration
"""

import json
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from .oms import OrderManager, OrderSide, OrderType, TimeInForce
from .sizing import PositionSizer, SizingConfig, SizingMethod, get_position_sizer

logger = logging.getLogger(__name__)


class RuleStatus(Enum):
    """Automation rule status."""
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    DISABLED = "DISABLED"
    TRIGGERED = "TRIGGERED"
    ERROR = "ERROR"


class ConditionType(Enum):
    """Types of conditions for rules."""
    SIGNAL_DIRECTION = "SIGNAL_DIRECTION"      # Signal is LONG/SHORT
    SIGNAL_CONFIDENCE = "SIGNAL_CONFIDENCE"    # Confidence above threshold
    PRICE_ABOVE = "PRICE_ABOVE"                # Price above level
    PRICE_BELOW = "PRICE_BELOW"                # Price below level
    PRICE_CROSS_UP = "PRICE_CROSS_UP"          # Price crosses above level
    PRICE_CROSS_DOWN = "PRICE_CROSS_DOWN"      # Price crosses below level
    TIME_OF_DAY = "TIME_OF_DAY"                # Within time window
    DAY_OF_WEEK = "DAY_OF_WEEK"                # On specific days
    NO_POSITION = "NO_POSITION"                 # No existing position
    POSITION_FLAT = "POSITION_FLAT"            # Position is flat
    POSITION_LONG = "POSITION_LONG"            # Currently long
    POSITION_SHORT = "POSITION_SHORT"          # Currently short
    DRAWDOWN_BELOW = "DRAWDOWN_BELOW"          # Drawdown below limit
    VAR_BELOW = "VAR_BELOW"                    # VaR below limit
    CUSTOM = "CUSTOM"                           # Custom function


class ActionType(Enum):
    """Types of actions for rules."""
    ENTER_LONG = "ENTER_LONG"
    ENTER_SHORT = "ENTER_SHORT"
    EXIT_LONG = "EXIT_LONG"
    EXIT_SHORT = "EXIT_SHORT"
    CLOSE_POSITION = "CLOSE_POSITION"
    REDUCE_POSITION = "REDUCE_POSITION"
    REVERSE_POSITION = "REVERSE_POSITION"
    SCALE_IN = "SCALE_IN"
    SCALE_OUT = "SCALE_OUT"
    SEND_ALERT = "SEND_ALERT"


@dataclass
class RuleCondition:
    """A single condition for a rule."""
    condition_type: ConditionType
    parameters: dict[str, Any] = field(default_factory=dict)

    # For signal conditions
    min_confidence: float = 60.0
    required_direction: str | None = None  # "LONG", "SHORT"

    # For price conditions
    price_level: float | None = None

    # For time conditions
    start_time: time | None = None
    end_time: time | None = None
    days: list[int] = field(default_factory=list)  # 0=Monday, 6=Sunday

    # For risk conditions
    max_drawdown: float = 0.05
    max_var: float = 500000

    # For custom conditions
    custom_function: Callable | None = None

    def evaluate(self, context: dict[str, Any]) -> bool:
        """
        Evaluate the condition.

        Args:
            context: Context with current state (signal, price, position, etc.)

        Returns:
            True if condition is met
        """
        try:
            if self.condition_type == ConditionType.SIGNAL_DIRECTION:
                signal = context.get("signal", {})
                direction = signal.get("direction", "NEUTRAL")
                if self.required_direction:
                    return direction == self.required_direction
                return direction != "NEUTRAL"

            elif self.condition_type == ConditionType.SIGNAL_CONFIDENCE:
                signal = context.get("signal", {})
                confidence = signal.get("confidence", 0)
                return confidence >= self.min_confidence

            elif self.condition_type == ConditionType.PRICE_ABOVE:
                price = context.get("price", 0)
                return price > self.price_level if self.price_level else False

            elif self.condition_type == ConditionType.PRICE_BELOW:
                price = context.get("price", 0)
                return price < self.price_level if self.price_level else False

            elif self.condition_type == ConditionType.TIME_OF_DAY:
                now = datetime.now().time()
                if self.start_time and self.end_time:
                    return self.start_time <= now <= self.end_time
                return True

            elif self.condition_type == ConditionType.DAY_OF_WEEK:
                today = datetime.now().weekday()
                return today in self.days if self.days else True

            elif self.condition_type == ConditionType.NO_POSITION:
                position = context.get("position", {})
                quantity = position.get("quantity", 0)
                return quantity == 0

            elif self.condition_type == ConditionType.POSITION_LONG:
                position = context.get("position", {})
                quantity = position.get("quantity", 0)
                return quantity > 0

            elif self.condition_type == ConditionType.POSITION_SHORT:
                position = context.get("position", {})
                quantity = position.get("quantity", 0)
                return quantity < 0

            elif self.condition_type == ConditionType.DRAWDOWN_BELOW:
                drawdown = context.get("drawdown", 0)
                return drawdown < self.max_drawdown

            elif self.condition_type == ConditionType.VAR_BELOW:
                var = context.get("var", 0)
                return var < self.max_var

            elif self.condition_type == ConditionType.CUSTOM:
                if self.custom_function:
                    return self.custom_function(context)
                return False

            return False

        except Exception as e:
            logger.error(f"Condition evaluation error: {e}")
            return False


@dataclass
class RuleAction:
    """Action to take when rule triggers."""
    action_type: ActionType

    # Order parameters
    symbol: str = "CL1"
    order_type: OrderType = OrderType.MARKET
    time_in_force: TimeInForce = TimeInForce.DAY

    # Sizing
    sizing_method: SizingMethod = SizingMethod.VOLATILITY_TARGET
    fixed_quantity: int | None = None
    risk_pct: float = 0.02

    # Price parameters (for limit orders)
    limit_offset_pct: float = 0.0  # Offset from current price
    stop_offset_pct: float = 0.02  # Stop loss offset

    # Partial actions
    reduce_pct: float = 0.5  # For reduce/scale actions

    # Alert parameters
    alert_message: str | None = None

    def to_dict(self) -> dict:
        return {
            "action_type": self.action_type.value,
            "symbol": self.symbol,
            "order_type": self.order_type.value,
            "sizing_method": self.sizing_method.value,
            "fixed_quantity": self.fixed_quantity,
            "risk_pct": self.risk_pct,
        }


@dataclass
class RuleConfig:
    """Configuration for an automation rule."""
    rule_id: str
    name: str
    description: str = ""

    # Conditions (all must be met - AND logic)
    conditions: list[RuleCondition] = field(default_factory=list)

    # Action to take
    action: RuleAction | None = None

    # Rule settings
    status: RuleStatus = RuleStatus.ACTIVE
    priority: int = 0  # Higher = more priority
    cooldown_minutes: int = 5  # Minimum time between triggers
    max_triggers_per_day: int = 10

    # Validity
    valid_from: datetime | None = None
    valid_until: datetime | None = None

    # Metadata
    strategy: str | None = None
    tags: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_triggered: datetime | None = None
    trigger_count: int = 0


@dataclass
class AutomationRule:
    """
    An automation rule for signal-to-order conversion.

    Combines conditions with actions to automate trading.
    """
    config: RuleConfig

    @property
    def is_active(self) -> bool:
        return self.config.status == RuleStatus.ACTIVE

    def evaluate(self, context: dict[str, Any]) -> bool:
        """
        Evaluate all conditions.

        Args:
            context: Current market/signal context

        Returns:
            True if all conditions are met
        """
        if not self.is_active:
            return False

        # Check validity period
        now = datetime.now()
        if self.config.valid_from and now < self.config.valid_from:
            return False
        if self.config.valid_until and now > self.config.valid_until:
            return False

        # Check cooldown
        if self.config.last_triggered:
            cooldown = timedelta(minutes=self.config.cooldown_minutes)
            if now - self.config.last_triggered < cooldown:
                return False

        # Check daily limit
        if self._triggers_today() >= self.config.max_triggers_per_day:
            return False

        # Evaluate all conditions (AND logic)
        return all(condition.evaluate(context) for condition in self.config.conditions)

    def _triggers_today(self) -> int:
        """Count triggers today."""
        # Simplified - would need proper tracking
        if self.config.last_triggered and self.config.last_triggered.date() == datetime.now().date():
            return self.config.trigger_count
        return 0

    def to_dict(self) -> dict:
        return {
            "rule_id": self.config.rule_id,
            "name": self.config.name,
            "description": self.config.description,
            "status": self.config.status.value,
            "priority": self.config.priority,
            "conditions": len(self.config.conditions),
            "action": self.config.action.to_dict() if self.config.action else None,
            "strategy": self.config.strategy,
            "trigger_count": self.config.trigger_count,
            "last_triggered": self.config.last_triggered.isoformat() if self.config.last_triggered else None,
        }


class AutomationEngine:
    """
    Automation engine for executing trading rules.

    Manages rules, evaluates conditions, and generates orders.
    """

    def __init__(
        self,
        oms: OrderManager | None = None,
        config_path: str = "config/automation_rules.json",
    ):
        self.oms = oms or OrderManager()
        self.config_path = Path(config_path)

        # Rules storage
        self._rules: dict[str, AutomationRule] = {}
        self._execution_history: list[dict] = []

        # Position sizers cache
        self._sizers: dict[SizingMethod, PositionSizer] = {}

        # State
        self._is_running = False
        self._last_evaluation: datetime | None = None

        # Load saved rules
        self._load_rules()

    def add_rule(self, config: RuleConfig) -> AutomationRule:
        """
        Add a new automation rule.

        Args:
            config: Rule configuration

        Returns:
            Created rule
        """
        rule = AutomationRule(config=config)
        self._rules[config.rule_id] = rule
        self._save_rules()

        logger.info(f"Added automation rule: {config.name} ({config.rule_id})")

        return rule

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule."""
        if rule_id in self._rules:
            del self._rules[rule_id]
            self._save_rules()
            logger.info(f"Removed automation rule: {rule_id}")
            return True
        return False

    def update_rule_status(self, rule_id: str, status: RuleStatus):
        """Update rule status."""
        if rule_id in self._rules:
            self._rules[rule_id].config.status = status
            self._save_rules()

    def get_rule(self, rule_id: str) -> AutomationRule | None:
        """Get a rule by ID."""
        return self._rules.get(rule_id)

    def get_rules(
        self,
        status: RuleStatus | None = None,
        strategy: str | None = None,
    ) -> list[AutomationRule]:
        """Get rules matching filters."""
        rules = list(self._rules.values())

        if status:
            rules = [r for r in rules if r.config.status == status]

        if strategy:
            rules = [r for r in rules if r.config.strategy == strategy]

        return sorted(rules, key=lambda r: r.config.priority, reverse=True)

    def evaluate_rules(
        self,
        context: dict[str, Any],
        execute: bool = True,
    ) -> list[dict]:
        """
        Evaluate all active rules against current context.

        Args:
            context: Current market/signal context
            execute: Whether to execute triggered rules

        Returns:
            List of triggered rules and actions
        """
        triggered = []

        # Get active rules sorted by priority
        active_rules = self.get_rules(status=RuleStatus.ACTIVE)

        for rule in active_rules:
            if rule.evaluate(context):
                logger.info(f"Rule triggered: {rule.config.name}")

                result = {
                    "rule_id": rule.config.rule_id,
                    "rule_name": rule.config.name,
                    "timestamp": datetime.now().isoformat(),
                    "action": rule.config.action.to_dict() if rule.config.action else None,
                    "executed": False,
                    "order_id": None,
                    "error": None,
                }

                if execute and rule.config.action:
                    try:
                        order_id = self._execute_action(rule.config.action, context)
                        result["executed"] = True
                        result["order_id"] = order_id

                        # Update rule state
                        rule.config.last_triggered = datetime.now()
                        rule.config.trigger_count += 1

                    except Exception as e:
                        result["error"] = str(e)
                        logger.error(f"Failed to execute rule action: {e}")

                triggered.append(result)
                self._execution_history.append(result)

        self._last_evaluation = datetime.now()

        return triggered

    def _execute_action(self, action: RuleAction, context: dict[str, Any]) -> str | None:
        """Execute a rule action."""
        # Determine side and quantity
        side = None
        quantity = None

        current_position = context.get("position", {}).get("quantity", 0)
        price = context.get("price", 75.0)
        volatility = context.get("volatility", 0.25)

        if action.action_type == ActionType.ENTER_LONG:
            side = OrderSide.BUY
            quantity = self._calculate_quantity(action, price, volatility, context)

        elif action.action_type == ActionType.ENTER_SHORT:
            side = OrderSide.SELL
            quantity = self._calculate_quantity(action, price, volatility, context)

        elif action.action_type == ActionType.EXIT_LONG:
            if current_position > 0:
                side = OrderSide.SELL
                quantity = current_position

        elif action.action_type == ActionType.EXIT_SHORT:
            if current_position < 0:
                side = OrderSide.BUY
                quantity = abs(current_position)

        elif action.action_type == ActionType.CLOSE_POSITION:
            if current_position != 0:
                side = OrderSide.SELL if current_position > 0 else OrderSide.BUY
                quantity = abs(current_position)

        elif action.action_type == ActionType.REDUCE_POSITION:
            if current_position != 0:
                side = OrderSide.SELL if current_position > 0 else OrderSide.BUY
                quantity = int(abs(current_position) * action.reduce_pct)

        elif action.action_type == ActionType.REVERSE_POSITION:
            if current_position != 0:
                # Close current and enter opposite
                new_size = self._calculate_quantity(action, price, volatility, context)
                side = OrderSide.SELL if current_position > 0 else OrderSide.BUY
                quantity = abs(current_position) + new_size

        elif action.action_type == ActionType.SEND_ALERT:
            self._send_alert(action.alert_message or f"Rule triggered for {action.symbol}")
            return None

        # Create and submit order
        if side and quantity and quantity > 0:
            order = self.oms.create_order(
                symbol=action.symbol,
                side=side,
                quantity=quantity,
                order_type=action.order_type,
                time_in_force=action.time_in_force,
                strategy=context.get("strategy"),
                tags=["automation"],
            )

            self.oms.submit_order(order.order_id)

            return order.order_id

        return None

    def _calculate_quantity(
        self,
        action: RuleAction,
        price: float,
        volatility: float,
        context: dict[str, Any],
    ) -> int:
        """Calculate position size for action."""
        if action.fixed_quantity:
            return action.fixed_quantity

        # Use position sizer
        account_value = context.get("account_value", 1_000_000)

        sizer_config = SizingConfig(
            method=action.sizing_method,
            account_value=account_value,
            risk_per_trade_pct=action.risk_pct,
        )

        sizer = get_position_sizer(sizer_config)
        result = sizer.calculate_size(
            price=price,
            volatility=volatility,
            stop_loss_pct=action.stop_offset_pct,
        )

        return result.contracts

    def _send_alert(self, message: str):
        """Send an alert (placeholder for notification integration)."""
        logger.info(f"ALERT: {message}")
        # Would integrate with notification system

    def _load_rules(self):
        """Load rules from config file."""
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    data = json.load(f)

                for rule_data in data.get("rules", []):
                    # Reconstruct rule config
                    conditions = []
                    for cond_data in rule_data.get("conditions", []):
                        conditions.append(RuleCondition(
                            condition_type=ConditionType(cond_data["type"]),
                            parameters=cond_data.get("parameters", {}),
                            min_confidence=cond_data.get("min_confidence", 60),
                            required_direction=cond_data.get("required_direction"),
                            price_level=cond_data.get("price_level"),
                        ))

                    action_data = rule_data.get("action")
                    action = None
                    if action_data:
                        action = RuleAction(
                            action_type=ActionType(action_data["type"]),
                            symbol=action_data.get("symbol", "CL1"),
                            sizing_method=SizingMethod(action_data.get("sizing_method", "VOLATILITY_TARGET")),
                            fixed_quantity=action_data.get("fixed_quantity"),
                            risk_pct=action_data.get("risk_pct", 0.02),
                        )

                    config = RuleConfig(
                        rule_id=rule_data["rule_id"],
                        name=rule_data["name"],
                        description=rule_data.get("description", ""),
                        conditions=conditions,
                        action=action,
                        status=RuleStatus(rule_data.get("status", "ACTIVE")),
                        priority=rule_data.get("priority", 0),
                        strategy=rule_data.get("strategy"),
                    )

                    self._rules[config.rule_id] = AutomationRule(config=config)

                logger.info(f"Loaded {len(self._rules)} automation rules")

            except Exception as e:
                logger.error(f"Failed to load automation rules: {e}")

    def _save_rules(self):
        """Save rules to config file."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)

            rules_data = []
            for rule in self._rules.values():
                conditions = []
                for cond in rule.config.conditions:
                    conditions.append({
                        "type": cond.condition_type.value,
                        "parameters": cond.parameters,
                        "min_confidence": cond.min_confidence,
                        "required_direction": cond.required_direction,
                        "price_level": cond.price_level,
                    })

                action_data = None
                if rule.config.action:
                    action_data = {
                        "type": rule.config.action.action_type.value,
                        "symbol": rule.config.action.symbol,
                        "sizing_method": rule.config.action.sizing_method.value,
                        "fixed_quantity": rule.config.action.fixed_quantity,
                        "risk_pct": rule.config.action.risk_pct,
                    }

                rules_data.append({
                    "rule_id": rule.config.rule_id,
                    "name": rule.config.name,
                    "description": rule.config.description,
                    "conditions": conditions,
                    "action": action_data,
                    "status": rule.config.status.value,
                    "priority": rule.config.priority,
                    "strategy": rule.config.strategy,
                })

            with open(self.config_path, "w") as f:
                json.dump({"rules": rules_data}, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save automation rules: {e}")

    def get_execution_history(self, limit: int = 50) -> list[dict]:
        """Get recent execution history."""
        return self._execution_history[-limit:]

    def get_statistics(self) -> dict:
        """Get automation statistics."""
        total_rules = len(self._rules)
        active = len([r for r in self._rules.values() if r.config.status == RuleStatus.ACTIVE])

        total_triggers = sum(r.config.trigger_count for r in self._rules.values())

        return {
            "total_rules": total_rules,
            "active_rules": active,
            "paused_rules": total_rules - active,
            "total_triggers": total_triggers,
            "recent_executions": len(self._execution_history),
            "last_evaluation": self._last_evaluation.isoformat() if self._last_evaluation else None,
        }


# Factory function for creating common rules
def create_signal_rule(
    name: str,
    symbol: str,
    direction: str,
    min_confidence: float = 65,
    sizing_method: SizingMethod = SizingMethod.VOLATILITY_TARGET,
    risk_pct: float = 0.02,
) -> RuleConfig:
    """
    Create a rule that triggers on signal direction.

    Args:
        name: Rule name
        symbol: Trading symbol
        direction: "LONG" or "SHORT"
        min_confidence: Minimum confidence threshold
        sizing_method: Position sizing method
        risk_pct: Risk per trade

    Returns:
        Rule configuration
    """
    conditions = [
        RuleCondition(
            condition_type=ConditionType.SIGNAL_DIRECTION,
            required_direction=direction,
        ),
        RuleCondition(
            condition_type=ConditionType.SIGNAL_CONFIDENCE,
            min_confidence=min_confidence,
        ),
        RuleCondition(
            condition_type=ConditionType.NO_POSITION,
        ),
    ]

    action_type = ActionType.ENTER_LONG if direction == "LONG" else ActionType.ENTER_SHORT

    action = RuleAction(
        action_type=action_type,
        symbol=symbol,
        sizing_method=sizing_method,
        risk_pct=risk_pct,
    )

    return RuleConfig(
        rule_id=f"RULE-{uuid.uuid4().hex[:8]}",
        name=name,
        description=f"Enter {direction} on {symbol} when signal confidence > {min_confidence}%",
        conditions=conditions,
        action=action,
    )
