"""
Alert Engine
============
Core alert evaluation and notification dispatch engine.

Features:
- Rule evaluation against market context
- Multi-channel notification dispatch
- Alert state management
- Escalation handling
"""

import json
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from .rules import (
    AlertCategory,
    AlertConfig,
    AlertRule,
    AlertSeverity,
    AlertTrigger,
)

logger = logging.getLogger(__name__)


class AlertStatus(Enum):
    """Status of an alert."""
    ACTIVE = "ACTIVE"           # Alert is active
    ACKNOWLEDGED = "ACKNOWLEDGED"  # Alert has been acknowledged
    RESOLVED = "RESOLVED"       # Alert has been resolved
    ESCALATED = "ESCALATED"     # Alert has been escalated
    EXPIRED = "EXPIRED"         # Alert has expired


@dataclass
class AlertEvent:
    """An alert event with full context."""
    trigger: AlertTrigger
    rule: AlertRule
    status: AlertStatus = AlertStatus.ACTIVE
    channels_notified: list[str] = field(default_factory=list)
    notification_errors: list[str] = field(default_factory=list)
    escalated: bool = False
    escalated_at: datetime | None = None

    def to_dict(self) -> dict:
        return {
            "trigger": self.trigger.to_dict(),
            "rule_name": self.rule.config.name,
            "status": self.status.value,
            "channels_notified": self.channels_notified,
            "notification_errors": self.notification_errors,
            "escalated": self.escalated,
        }


@dataclass
class AlertEngineConfig:
    """Configuration for the alert engine."""
    # Storage
    storage_path: str = "data/alerts"

    # Evaluation
    evaluation_interval_seconds: int = 10

    # Defaults
    default_cooldown_minutes: int = 5
    default_max_alerts_per_hour: int = 20
    default_max_alerts_per_day: int = 100

    # Escalation
    escalation_check_interval_seconds: int = 60
    auto_escalate: bool = True

    # Retention
    alert_retention_days: int = 30

    # Global controls
    enabled: bool = True
    muted: bool = False
    mute_until: datetime | None = None


class AlertEngine:
    """
    Central alert engine for rule evaluation and notification dispatch.

    Manages alert rules, evaluates conditions, dispatches notifications,
    and handles escalation.
    """

    def __init__(
        self,
        config: AlertEngineConfig | None = None,
        notification_callback: Callable[[AlertEvent], None] | None = None,
    ):
        self.config = config or AlertEngineConfig()
        self._notification_callback = notification_callback

        # Rule storage
        self._rules: dict[str, AlertRule] = {}

        # Active alerts
        self._active_alerts: dict[str, AlertEvent] = {}
        self._alert_history: list[AlertEvent] = []

        # Notification channels
        self._channels: dict[str, Any] = {}

        # State
        self._is_running = False
        self._lock = threading.Lock()
        self._last_evaluation: datetime | None = None
        self._last_escalation_check: datetime | None = None

        # Storage path
        self._storage_path = Path(self.config.storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)

        # Load saved state
        self._load_state()

    # =========================================================================
    # Rule Management
    # =========================================================================

    def add_rule(self, config: AlertConfig) -> AlertRule:
        """
        Add a new alert rule.

        Args:
            config: Alert rule configuration

        Returns:
            Created AlertRule
        """
        with self._lock:
            rule = AlertRule(config=config)
            self._rules[config.rule_id] = rule
            self._save_state()

            logger.info(f"Added alert rule: {config.name} ({config.rule_id})")
            return rule

    def remove_rule(self, rule_id: str) -> bool:
        """Remove an alert rule."""
        with self._lock:
            if rule_id in self._rules:
                del self._rules[rule_id]
                self._save_state()
                logger.info(f"Removed alert rule: {rule_id}")
                return True
            return False

    def update_rule(self, rule_id: str, updates: dict[str, Any]) -> AlertRule | None:
        """Update an existing rule."""
        with self._lock:
            if rule_id not in self._rules:
                return None

            rule = self._rules[rule_id]
            config = rule.config

            # Update allowed fields
            if "enabled" in updates:
                config.enabled = updates["enabled"]
            if "severity" in updates:
                config.severity = AlertSeverity(updates["severity"])
            if "channels" in updates:
                config.channels = updates["channels"]
            if "cooldown_minutes" in updates:
                config.cooldown_minutes = updates["cooldown_minutes"]

            config.updated_at = datetime.now()
            self._save_state()

            return rule

    def get_rule(self, rule_id: str) -> AlertRule | None:
        """Get a rule by ID."""
        return self._rules.get(rule_id)

    def get_rules(
        self,
        enabled: bool | None = None,
        category: AlertCategory | None = None,
        severity: AlertSeverity | None = None,
    ) -> list[AlertRule]:
        """Get rules matching filters."""
        rules = list(self._rules.values())

        if enabled is not None:
            rules = [r for r in rules if r.config.enabled == enabled]

        if category:
            rules = [r for r in rules if r.config.category == category]

        if severity:
            rules = [r for r in rules if r.config.severity == severity]

        return rules

    def enable_rule(self, rule_id: str) -> bool:
        """Enable a rule."""
        rule = self.get_rule(rule_id)
        if rule:
            rule.config.enabled = True
            self._save_state()
            return True
        return False

    def disable_rule(self, rule_id: str) -> bool:
        """Disable a rule."""
        rule = self.get_rule(rule_id)
        if rule:
            rule.config.enabled = False
            self._save_state()
            return True
        return False

    # =========================================================================
    # Alert Evaluation
    # =========================================================================

    def evaluate(self, context: dict[str, Any]) -> list[AlertEvent]:
        """
        Evaluate all rules against current context.

        Args:
            context: Current market/system context

        Returns:
            List of triggered alert events
        """
        if not self.config.enabled:
            return []

        if self._is_muted():
            return []

        triggered = []

        with self._lock:
            for rule in self._rules.values():
                if not rule.is_active:
                    continue

                trigger = rule.evaluate(context)

                if trigger:
                    event = self._create_event(trigger, rule)
                    triggered.append(event)

                    # Store active alert
                    self._active_alerts[trigger.trigger_id] = event

                    # Dispatch notifications
                    self._dispatch_notifications(event)

        self._last_evaluation = datetime.now()

        if triggered:
            logger.info(f"Triggered {len(triggered)} alerts")

        return triggered

    def _create_event(self, trigger: AlertTrigger, rule: AlertRule) -> AlertEvent:
        """Create an alert event."""
        return AlertEvent(
            trigger=trigger,
            rule=rule,
            status=AlertStatus.ACTIVE,
        )

    def _is_muted(self) -> bool:
        """Check if alerts are globally muted."""
        if self.config.muted:
            if self.config.mute_until and datetime.now() >= self.config.mute_until:
                self.config.muted = False
                self.config.mute_until = None
                return False
            return True
        return False

    # =========================================================================
    # Notification Dispatch
    # =========================================================================

    def register_channel(self, name: str, channel: Any):
        """Register a notification channel."""
        self._channels[name] = channel
        logger.info(f"Registered notification channel: {name}")

    def _dispatch_notifications(self, event: AlertEvent):
        """Dispatch notifications to configured channels."""
        channels = event.rule.config.channels

        for channel_name in channels:
            try:
                if channel_name in self._channels:
                    channel = self._channels[channel_name]
                    channel.send(event)
                    event.channels_notified.append(channel_name)
                else:
                    logger.warning(f"Channel not registered: {channel_name}")

            except Exception as e:
                error_msg = f"Failed to send to {channel_name}: {e}"
                event.notification_errors.append(error_msg)
                logger.error(error_msg)

        # Call custom callback if provided
        if self._notification_callback:
            try:
                self._notification_callback(event)
            except Exception as e:
                logger.error(f"Notification callback error: {e}")

    # =========================================================================
    # Alert State Management
    # =========================================================================

    def acknowledge(
        self,
        alert_id: str,
        acknowledged_by: str = "system",
        note: str | None = None,
    ) -> bool:
        """Acknowledge an alert."""
        with self._lock:
            if alert_id in self._active_alerts:
                event = self._active_alerts[alert_id]
                event.trigger.acknowledged = True
                event.trigger.acknowledged_at = datetime.now()
                event.trigger.acknowledged_by = acknowledged_by
                event.status = AlertStatus.ACKNOWLEDGED

                logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
                self._save_state()
                return True
            return False

    def resolve(self, alert_id: str, note: str | None = None) -> bool:
        """Resolve an alert."""
        with self._lock:
            if alert_id in self._active_alerts:
                event = self._active_alerts[alert_id]
                event.status = AlertStatus.RESOLVED

                # Move to history
                self._alert_history.append(event)
                del self._active_alerts[alert_id]

                logger.info(f"Alert resolved: {alert_id}")
                self._save_state()
                return True
            return False

    def check_escalations(self):
        """Check for alerts that need escalation."""
        if not self.config.auto_escalate:
            return

        now = datetime.now()

        with self._lock:
            for _alert_id, event in list(self._active_alerts.items()):
                if event.escalated:
                    continue

                if event.rule.needs_escalation(event.trigger):
                    self._escalate_alert(event)

        self._last_escalation_check = now

    def _escalate_alert(self, event: AlertEvent):
        """Escalate an alert."""
        event.escalated = True
        event.escalated_at = datetime.now()
        event.status = AlertStatus.ESCALATED

        # Update severity
        event.trigger.severity = event.rule.config.escalation_severity

        # Send to escalation channels
        escalation_channels = event.rule.config.escalation_channels

        for channel_name in escalation_channels:
            try:
                if channel_name in self._channels:
                    channel = self._channels[channel_name]
                    # Send with escalation flag
                    channel.send(event, escalated=True)
                    event.channels_notified.append(f"{channel_name}(escalated)")
            except Exception as e:
                event.notification_errors.append(f"Escalation to {channel_name}: {e}")

        logger.warning(f"Alert escalated: {event.trigger.trigger_id}")

    def get_active_alerts(
        self,
        severity: AlertSeverity | None = None,
        category: AlertCategory | None = None,
    ) -> list[AlertEvent]:
        """Get active alerts."""
        alerts = list(self._active_alerts.values())

        if severity:
            alerts = [a for a in alerts if a.trigger.severity == severity]

        if category:
            alerts = [a for a in alerts if a.trigger.category == category]

        # Sort by severity and time
        severity_order = {
            AlertSeverity.CRITICAL: 0,
            AlertSeverity.HIGH: 1,
            AlertSeverity.MEDIUM: 2,
            AlertSeverity.LOW: 3,
            AlertSeverity.INFO: 4,
        }

        alerts.sort(key=lambda a: (severity_order.get(a.trigger.severity, 5), a.trigger.timestamp))

        return alerts

    def get_alert_history(
        self,
        limit: int = 100,
        since: datetime | None = None,
    ) -> list[AlertEvent]:
        """Get alert history."""
        history = self._alert_history

        if since:
            history = [a for a in history if a.trigger.timestamp >= since]

        return history[-limit:]

    # =========================================================================
    # Global Controls
    # =========================================================================

    def mute(self, duration_minutes: int | None = None):
        """Mute all alerts."""
        self.config.muted = True
        if duration_minutes:
            self.config.mute_until = datetime.now() + timedelta(minutes=duration_minutes)
        logger.info("Alerts muted" + (f" for {duration_minutes} minutes" if duration_minutes else ""))

    def unmute(self):
        """Unmute all alerts."""
        self.config.muted = False
        self.config.mute_until = None
        logger.info("Alerts unmuted")

    def enable(self):
        """Enable alert engine."""
        self.config.enabled = True
        logger.info("Alert engine enabled")

    def disable(self):
        """Disable alert engine."""
        self.config.enabled = False
        logger.info("Alert engine disabled")

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_statistics(self) -> dict[str, Any]:
        """Get alert engine statistics."""
        now = datetime.now()
        today = now.date()
        now.replace(minute=0, second=0, microsecond=0)

        # Count by severity
        severity_counts = {}
        for event in self._active_alerts.values():
            sev = event.trigger.severity.value
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        # Count by category
        category_counts = {}
        for event in self._active_alerts.values():
            cat = event.trigger.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1

        # Alerts today
        alerts_today = sum(
            1 for a in self._alert_history
            if a.trigger.timestamp.date() == today
        ) + len(self._active_alerts)

        # Acknowledgment rate
        total_historical = len(self._alert_history)
        acknowledged = sum(1 for a in self._alert_history if a.trigger.acknowledged)
        ack_rate = acknowledged / total_historical if total_historical > 0 else 0

        return {
            "total_rules": len(self._rules),
            "active_rules": len([r for r in self._rules.values() if r.config.enabled]),
            "active_alerts": len(self._active_alerts),
            "alerts_today": alerts_today,
            "alerts_history": len(self._alert_history),
            "by_severity": severity_counts,
            "by_category": category_counts,
            "acknowledgment_rate": ack_rate,
            "last_evaluation": self._last_evaluation.isoformat() if self._last_evaluation else None,
            "enabled": self.config.enabled,
            "muted": self.config.muted,
            "mute_until": self.config.mute_until.isoformat() if self.config.mute_until else None,
        }

    # =========================================================================
    # Persistence
    # =========================================================================

    def _load_state(self):
        """Load saved state."""
        rules_path = self._storage_path / "rules.json"

        if rules_path.exists():
            try:
                with open(rules_path) as f:
                    data = json.load(f)

                for rule_data in data.get("rules", []):
                    config = self._deserialize_rule_config(rule_data)
                    self._rules[config.rule_id] = AlertRule(config=config)

                logger.info(f"Loaded {len(self._rules)} alert rules")

            except Exception as e:
                logger.error(f"Failed to load alert rules: {e}")

    def _save_state(self):
        """Save current state."""
        rules_path = self._storage_path / "rules.json"

        try:
            rules_data = []
            for rule in self._rules.values():
                rules_data.append(self._serialize_rule_config(rule.config))

            with open(rules_path, "w") as f:
                json.dump({"rules": rules_data}, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save alert rules: {e}")

    def _serialize_rule_config(self, config: AlertConfig) -> dict:
        """Serialize rule configuration."""
        return {
            "rule_id": config.rule_id,
            "name": config.name,
            "description": config.description,
            "severity": config.severity.value,
            "category": config.category.value,
            "enabled": config.enabled,
            "channels": config.channels,
            "cooldown_minutes": config.cooldown_minutes,
            "max_alerts_per_hour": config.max_alerts_per_hour,
            "max_alerts_per_day": config.max_alerts_per_day,
            "title_template": config.title_template,
            "message_template": config.message_template,
            "tags": config.tags,
            "conditions": [c.to_dict() for c in config.conditions],
            "trigger_count": config.trigger_count,
            "last_triggered": config.last_triggered.isoformat() if config.last_triggered else None,
        }

    def _deserialize_rule_config(self, data: dict) -> AlertConfig:
        """Deserialize rule configuration."""
        from .rules import AlertCondition, ConditionType

        conditions = []
        for cond_data in data.get("conditions", []):
            conditions.append(AlertCondition(
                condition_type=ConditionType(cond_data["condition_type"]),
                threshold=cond_data.get("threshold"),
                price_level=cond_data.get("price_level"),
                change_pct=cond_data.get("change_pct"),
                symbol=cond_data.get("symbol"),
                parameters=cond_data.get("parameters", {}),
            ))

        last_triggered = data.get("last_triggered")
        if last_triggered:
            last_triggered = datetime.fromisoformat(last_triggered)

        return AlertConfig(
            rule_id=data["rule_id"],
            name=data["name"],
            description=data.get("description", ""),
            severity=AlertSeverity(data.get("severity", "MEDIUM")),
            category=AlertCategory(data.get("category", "CUSTOM")),
            enabled=data.get("enabled", True),
            channels=data.get("channels", ["email"]),
            cooldown_minutes=data.get("cooldown_minutes", 5),
            max_alerts_per_hour=data.get("max_alerts_per_hour", 10),
            max_alerts_per_day=data.get("max_alerts_per_day", 50),
            title_template=data.get("title_template", "Alert: {name}"),
            message_template=data.get("message_template", "{description}"),
            tags=data.get("tags", []),
            conditions=conditions,
            trigger_count=data.get("trigger_count", 0),
            last_triggered=last_triggered,
        )
