"""
Alert Rules
===========
Configurable alert conditions and triggers.

Features:
- Multiple condition types (price, signal, risk, time-based)
- Severity levels for prioritization
- Category classification
- Cooldown and rate limiting
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Any, Callable, Dict, List, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "INFO"           # Informational only
    LOW = "LOW"             # Low priority
    MEDIUM = "MEDIUM"       # Medium priority, action may be needed
    HIGH = "HIGH"           # High priority, action required
    CRITICAL = "CRITICAL"   # Critical, immediate action required


class AlertCategory(Enum):
    """Alert categories for classification."""
    SIGNAL = "SIGNAL"               # Trading signal alerts
    RISK = "RISK"                   # Risk breach alerts
    PRICE = "PRICE"                 # Price level alerts
    POSITION = "POSITION"           # Position-related alerts
    EXECUTION = "EXECUTION"         # Order execution alerts
    SYSTEM = "SYSTEM"               # System health alerts
    MARKET = "MARKET"               # Market event alerts
    PNL = "PNL"                     # P&L alerts
    COMPLIANCE = "COMPLIANCE"       # Compliance/limit alerts
    CUSTOM = "CUSTOM"               # Custom alerts


class ConditionType(Enum):
    """Types of conditions for alert rules."""
    # Price conditions
    PRICE_ABOVE = "PRICE_ABOVE"
    PRICE_BELOW = "PRICE_BELOW"
    PRICE_CHANGE_PCT = "PRICE_CHANGE_PCT"
    PRICE_CROSS_LEVEL = "PRICE_CROSS_LEVEL"
    
    # Signal conditions
    SIGNAL_GENERATED = "SIGNAL_GENERATED"
    SIGNAL_DIRECTION = "SIGNAL_DIRECTION"
    SIGNAL_CONFIDENCE_ABOVE = "SIGNAL_CONFIDENCE_ABOVE"
    SIGNAL_CHANGE = "SIGNAL_CHANGE"
    
    # Risk conditions
    VAR_ABOVE = "VAR_ABOVE"
    DRAWDOWN_ABOVE = "DRAWDOWN_ABOVE"
    EXPOSURE_ABOVE = "EXPOSURE_ABOVE"
    CONCENTRATION_ABOVE = "CONCENTRATION_ABOVE"
    LIMIT_BREACH = "LIMIT_BREACH"
    
    # Position conditions
    POSITION_SIZE_ABOVE = "POSITION_SIZE_ABOVE"
    POSITION_OPENED = "POSITION_OPENED"
    POSITION_CLOSED = "POSITION_CLOSED"
    POSITION_PNL_ABOVE = "POSITION_PNL_ABOVE"
    POSITION_PNL_BELOW = "POSITION_PNL_BELOW"
    
    # Execution conditions
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_REJECTED = "ORDER_REJECTED"
    ORDER_TIMEOUT = "ORDER_TIMEOUT"
    SLIPPAGE_ABOVE = "SLIPPAGE_ABOVE"
    
    # Time conditions
    TIME_OF_DAY = "TIME_OF_DAY"
    MARKET_OPEN = "MARKET_OPEN"
    MARKET_CLOSE = "MARKET_CLOSE"
    SCHEDULED = "SCHEDULED"
    
    # Market conditions
    VOLATILITY_SPIKE = "VOLATILITY_SPIKE"
    VOLUME_SPIKE = "VOLUME_SPIKE"
    SPREAD_WIDENING = "SPREAD_WIDENING"
    
    # Custom
    CUSTOM = "CUSTOM"


@dataclass
class AlertCondition:
    """A condition that triggers an alert."""
    condition_type: ConditionType
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Threshold parameters
    threshold: Optional[float] = None
    comparison: str = ">"  # ">", "<", ">=", "<=", "==", "!="
    
    # Price conditions
    price_level: Optional[float] = None
    change_pct: Optional[float] = None
    
    # Time conditions
    start_time: Optional[time] = None
    end_time: Optional[time] = None
    days: List[int] = field(default_factory=list)  # 0=Monday
    
    # Symbol filter
    symbol: Optional[str] = None
    
    # Custom condition
    custom_function: Optional[Callable[[Dict[str, Any]], bool]] = None
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """
        Evaluate the condition against current context.
        
        Args:
            context: Current state (prices, signals, positions, etc.)
            
        Returns:
            True if condition is met
        """
        try:
            # Price conditions
            if self.condition_type == ConditionType.PRICE_ABOVE:
                price = context.get("price", 0)
                return price > (self.threshold or self.price_level or 0)
                
            elif self.condition_type == ConditionType.PRICE_BELOW:
                price = context.get("price", 0)
                return price < (self.threshold or self.price_level or 0)
                
            elif self.condition_type == ConditionType.PRICE_CHANGE_PCT:
                change = context.get("price_change_pct", 0)
                threshold = self.change_pct or self.threshold or 0
                return abs(change) >= threshold
                
            elif self.condition_type == ConditionType.PRICE_CROSS_LEVEL:
                prev_price = context.get("prev_price", 0)
                price = context.get("price", 0)
                level = self.price_level or self.threshold or 0
                return (prev_price < level <= price) or (prev_price > level >= price)
            
            # Signal conditions
            elif self.condition_type == ConditionType.SIGNAL_GENERATED:
                signal = context.get("signal", {})
                return signal.get("generated", False)
                
            elif self.condition_type == ConditionType.SIGNAL_DIRECTION:
                signal = context.get("signal", {})
                direction = signal.get("direction", "NEUTRAL")
                required = self.parameters.get("direction", "LONG")
                return direction == required
                
            elif self.condition_type == ConditionType.SIGNAL_CONFIDENCE_ABOVE:
                signal = context.get("signal", {})
                confidence = signal.get("confidence", 0)
                return confidence >= (self.threshold or 60)
                
            elif self.condition_type == ConditionType.SIGNAL_CHANGE:
                prev_signal = context.get("prev_signal", {})
                signal = context.get("signal", {})
                return prev_signal.get("direction") != signal.get("direction")
            
            # Risk conditions
            elif self.condition_type == ConditionType.VAR_ABOVE:
                var = context.get("var", 0)
                return var > (self.threshold or 0)
                
            elif self.condition_type == ConditionType.DRAWDOWN_ABOVE:
                drawdown = context.get("drawdown", 0)
                return drawdown > (self.threshold or 0.05)
                
            elif self.condition_type == ConditionType.EXPOSURE_ABOVE:
                exposure = context.get("gross_exposure", 0)
                return exposure > (self.threshold or 0)
                
            elif self.condition_type == ConditionType.CONCENTRATION_ABOVE:
                concentration = context.get("concentration", 0)
                return concentration > (self.threshold or 0.4)
                
            elif self.condition_type == ConditionType.LIMIT_BREACH:
                breaches = context.get("limit_breaches", [])
                return len(breaches) > 0
            
            # Position conditions
            elif self.condition_type == ConditionType.POSITION_SIZE_ABOVE:
                position = context.get("position", {})
                size = abs(position.get("quantity", 0))
                return size > (self.threshold or 0)
                
            elif self.condition_type == ConditionType.POSITION_OPENED:
                return context.get("position_opened", False)
                
            elif self.condition_type == ConditionType.POSITION_CLOSED:
                return context.get("position_closed", False)
                
            elif self.condition_type == ConditionType.POSITION_PNL_ABOVE:
                position = context.get("position", {})
                pnl = position.get("unrealized_pnl", 0)
                return pnl > (self.threshold or 0)
                
            elif self.condition_type == ConditionType.POSITION_PNL_BELOW:
                position = context.get("position", {})
                pnl = position.get("unrealized_pnl", 0)
                return pnl < (self.threshold or 0)
            
            # Execution conditions
            elif self.condition_type == ConditionType.ORDER_FILLED:
                return context.get("order_filled", False)
                
            elif self.condition_type == ConditionType.ORDER_REJECTED:
                return context.get("order_rejected", False)
                
            elif self.condition_type == ConditionType.SLIPPAGE_ABOVE:
                slippage = context.get("slippage_bps", 0)
                return slippage > (self.threshold or 10)
            
            # Time conditions
            elif self.condition_type == ConditionType.TIME_OF_DAY:
                now = datetime.now().time()
                if self.start_time and self.end_time:
                    return self.start_time <= now <= self.end_time
                return True
                
            elif self.condition_type == ConditionType.MARKET_OPEN:
                return context.get("market_just_opened", False)
                
            elif self.condition_type == ConditionType.MARKET_CLOSE:
                return context.get("market_closing_soon", False)
            
            # Market conditions
            elif self.condition_type == ConditionType.VOLATILITY_SPIKE:
                vol = context.get("current_volatility", 0)
                avg_vol = context.get("average_volatility", vol)
                threshold = self.threshold or 2.0
                if avg_vol > 0:
                    return vol / avg_vol > threshold
                return False
                
            elif self.condition_type == ConditionType.VOLUME_SPIKE:
                volume = context.get("current_volume", 0)
                avg_volume = context.get("average_volume", volume)
                threshold = self.threshold or 2.0
                if avg_volume > 0:
                    return volume / avg_volume > threshold
                return False
                
            elif self.condition_type == ConditionType.SPREAD_WIDENING:
                spread = context.get("bid_ask_spread", 0)
                avg_spread = context.get("average_spread", spread)
                threshold = self.threshold or 2.0
                if avg_spread > 0:
                    return spread / avg_spread > threshold
                return False
            
            # Custom condition
            elif self.condition_type == ConditionType.CUSTOM:
                if self.custom_function:
                    return self.custom_function(context)
                return False
                
            return False
            
        except Exception as e:
            logger.error(f"Condition evaluation error: {e}")
            return False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "condition_type": self.condition_type.value,
            "threshold": self.threshold,
            "comparison": self.comparison,
            "price_level": self.price_level,
            "change_pct": self.change_pct,
            "symbol": self.symbol,
            "parameters": self.parameters,
        }


@dataclass
class AlertTrigger:
    """Record of a triggered alert."""
    trigger_id: str
    rule_id: str
    timestamp: datetime
    severity: AlertSeverity
    category: AlertCategory
    title: str
    message: str
    context: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "trigger_id": self.trigger_id,
            "rule_id": self.rule_id,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "category": self.category.value,
            "title": self.title,
            "message": self.message,
            "acknowledged": self.acknowledged,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
        }


@dataclass
class AlertConfig:
    """Configuration for an alert rule."""
    rule_id: str
    name: str
    description: str = ""
    
    # Alert properties
    severity: AlertSeverity = AlertSeverity.MEDIUM
    category: AlertCategory = AlertCategory.CUSTOM
    
    # Conditions (all must be met - AND logic)
    conditions: List[AlertCondition] = field(default_factory=list)
    
    # Message template
    title_template: str = "Alert: {name}"
    message_template: str = "{description}"
    
    # Notification settings
    enabled: bool = True
    channels: List[str] = field(default_factory=lambda: ["email"])  # email, telegram, slack, sms
    
    # Rate limiting
    cooldown_minutes: int = 5
    max_alerts_per_hour: int = 10
    max_alerts_per_day: int = 50
    
    # Escalation
    escalate_after_minutes: int = 30
    escalation_severity: AlertSeverity = AlertSeverity.HIGH
    escalation_channels: List[str] = field(default_factory=list)
    
    # Time restrictions
    active_hours_start: Optional[time] = None
    active_hours_end: Optional[time] = None
    active_days: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])  # Monday-Friday
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Statistics
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0


@dataclass
class AlertRule:
    """
    An alert rule that evaluates conditions and generates alerts.
    """
    config: AlertConfig
    
    # Runtime state
    _triggers_this_hour: int = field(default=0, init=False)
    _triggers_today: int = field(default=0, init=False)
    _last_hour_reset: datetime = field(default_factory=datetime.now, init=False)
    _last_day_reset: datetime = field(default_factory=datetime.now, init=False)
    
    @property
    def is_active(self) -> bool:
        """Check if rule is currently active."""
        if not self.config.enabled:
            return False
        
        now = datetime.now()
        
        # Check active days
        if now.weekday() not in self.config.active_days:
            return False
        
        # Check active hours
        if self.config.active_hours_start and self.config.active_hours_end:
            current_time = now.time()
            if not (self.config.active_hours_start <= current_time <= self.config.active_hours_end):
                return False
        
        return True
    
    def evaluate(self, context: Dict[str, Any]) -> Optional[AlertTrigger]:
        """
        Evaluate conditions and generate alert if triggered.
        
        Args:
            context: Current state context
            
        Returns:
            AlertTrigger if conditions met, None otherwise
        """
        if not self.is_active:
            return None
        
        # Check rate limits
        if not self._check_rate_limits():
            return None
        
        # Check cooldown
        if not self._check_cooldown():
            return None
        
        # Evaluate all conditions
        if not all(cond.evaluate(context) for cond in self.config.conditions):
            return None
        
        # Generate alert
        trigger = self._create_trigger(context)
        
        # Update statistics
        self._update_stats()
        
        return trigger
    
    def _check_rate_limits(self) -> bool:
        """Check if we're within rate limits."""
        now = datetime.now()
        
        # Reset hourly counter
        if (now - self._last_hour_reset).total_seconds() > 3600:
            self._triggers_this_hour = 0
            self._last_hour_reset = now
        
        # Reset daily counter
        if now.date() != self._last_day_reset.date():
            self._triggers_today = 0
            self._last_day_reset = now
        
        # Check limits
        if self._triggers_this_hour >= self.config.max_alerts_per_hour:
            return False
        if self._triggers_today >= self.config.max_alerts_per_day:
            return False
        
        return True
    
    def _check_cooldown(self) -> bool:
        """Check if cooldown period has passed."""
        if not self.config.last_triggered:
            return True
        
        elapsed = datetime.now() - self.config.last_triggered
        cooldown = timedelta(minutes=self.config.cooldown_minutes)
        
        return elapsed >= cooldown
    
    def _create_trigger(self, context: Dict[str, Any]) -> AlertTrigger:
        """Create an alert trigger."""
        # Format message from template
        title = self.config.title_template.format(
            name=self.config.name,
            **context,
        )
        
        message = self.config.message_template.format(
            description=self.config.description,
            **{k: v for k, v in context.items() if not callable(v)},
        )
        
        return AlertTrigger(
            trigger_id=f"ALERT-{uuid.uuid4().hex[:8]}",
            rule_id=self.config.rule_id,
            timestamp=datetime.now(),
            severity=self.config.severity,
            category=self.config.category,
            title=title,
            message=message,
            context=context,
        )
    
    def _update_stats(self):
        """Update trigger statistics."""
        self.config.last_triggered = datetime.now()
        self.config.trigger_count += 1
        self._triggers_this_hour += 1
        self._triggers_today += 1
    
    def needs_escalation(self, trigger: AlertTrigger) -> bool:
        """Check if alert needs escalation."""
        if trigger.acknowledged:
            return False
        
        if not self.config.escalation_channels:
            return False
        
        elapsed = datetime.now() - trigger.timestamp
        escalation_time = timedelta(minutes=self.config.escalate_after_minutes)
        
        return elapsed >= escalation_time
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "rule_id": self.config.rule_id,
            "name": self.config.name,
            "description": self.config.description,
            "severity": self.config.severity.value,
            "category": self.config.category.value,
            "enabled": self.config.enabled,
            "channels": self.config.channels,
            "conditions": [c.to_dict() for c in self.config.conditions],
            "trigger_count": self.config.trigger_count,
            "last_triggered": self.config.last_triggered.isoformat() if self.config.last_triggered else None,
        }


# Factory functions for common alert rules
def create_price_alert(
    name: str,
    symbol: str,
    price_level: float,
    direction: str = "above",
    severity: AlertSeverity = AlertSeverity.MEDIUM,
) -> AlertConfig:
    """Create a price level alert."""
    condition_type = ConditionType.PRICE_ABOVE if direction == "above" else ConditionType.PRICE_BELOW
    
    return AlertConfig(
        rule_id=f"ALERT-{uuid.uuid4().hex[:8]}",
        name=name,
        description=f"Alert when {symbol} price goes {direction} {price_level}",
        severity=severity,
        category=AlertCategory.PRICE,
        conditions=[
            AlertCondition(
                condition_type=condition_type,
                price_level=price_level,
                symbol=symbol,
            )
        ],
        title_template=f"Price Alert: {symbol}",
        message_template=f"{symbol} price is now {{price}}, which is {direction} {price_level}",
    )


def create_risk_alert(
    name: str,
    risk_type: str,  # "var", "drawdown", "exposure"
    threshold: float,
    severity: AlertSeverity = AlertSeverity.HIGH,
) -> AlertConfig:
    """Create a risk breach alert."""
    condition_map = {
        "var": ConditionType.VAR_ABOVE,
        "drawdown": ConditionType.DRAWDOWN_ABOVE,
        "exposure": ConditionType.EXPOSURE_ABOVE,
        "concentration": ConditionType.CONCENTRATION_ABOVE,
    }
    
    return AlertConfig(
        rule_id=f"ALERT-{uuid.uuid4().hex[:8]}",
        name=name,
        description=f"Alert when {risk_type} exceeds {threshold}",
        severity=severity,
        category=AlertCategory.RISK,
        conditions=[
            AlertCondition(
                condition_type=condition_map.get(risk_type, ConditionType.CUSTOM),
                threshold=threshold,
            )
        ],
        title_template=f"Risk Alert: {risk_type.upper()} Threshold",
        message_template=f"Current {risk_type} has exceeded the limit of {threshold}",
    )


def create_signal_alert(
    name: str,
    direction: str,  # "LONG", "SHORT"
    min_confidence: float = 70,
    severity: AlertSeverity = AlertSeverity.MEDIUM,
) -> AlertConfig:
    """Create a signal alert."""
    return AlertConfig(
        rule_id=f"ALERT-{uuid.uuid4().hex[:8]}",
        name=name,
        description=f"Alert on {direction} signal with confidence >= {min_confidence}%",
        severity=severity,
        category=AlertCategory.SIGNAL,
        conditions=[
            AlertCondition(
                condition_type=ConditionType.SIGNAL_DIRECTION,
                parameters={"direction": direction},
            ),
            AlertCondition(
                condition_type=ConditionType.SIGNAL_CONFIDENCE_ABOVE,
                threshold=min_confidence,
            ),
        ],
        title_template=f"Signal Alert: {direction}",
        message_template=f"New {direction} signal generated with {{signal.confidence}}% confidence",
    )


def create_pnl_alert(
    name: str,
    threshold: float,
    alert_type: str = "loss",  # "loss", "profit"
    severity: AlertSeverity = AlertSeverity.HIGH,
) -> AlertConfig:
    """Create a P&L alert."""
    condition_type = ConditionType.POSITION_PNL_BELOW if alert_type == "loss" else ConditionType.POSITION_PNL_ABOVE
    threshold_val = -abs(threshold) if alert_type == "loss" else abs(threshold)
    
    return AlertConfig(
        rule_id=f"ALERT-{uuid.uuid4().hex[:8]}",
        name=name,
        description=f"Alert when P&L {'drops below' if alert_type == 'loss' else 'exceeds'} ${threshold:,.0f}",
        severity=severity,
        category=AlertCategory.PNL,
        conditions=[
            AlertCondition(
                condition_type=condition_type,
                threshold=threshold_val,
            )
        ],
        title_template=f"P&L Alert: {'Loss' if alert_type == 'loss' else 'Profit'} Threshold",
        message_template=f"Current P&L is ${{pnl:,.0f}}, {'below' if alert_type == 'loss' else 'above'} ${threshold:,.0f} threshold",
    )
