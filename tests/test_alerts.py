"""
Tests for Alerts Module
=======================
Tests for alert rules, engine, channels, and scheduling.
"""

import shutil
import tempfile
import uuid
from datetime import datetime, time
from pathlib import Path
from typing import Any

import pytest

from core.alerts import (
    AlertEngine,
    AlertHistory,
    AlertRecord,
    AlertRule,
    ReportConfig,
    ReportScheduler,
)
from core.alerts.channels.base import ChannelConfig, NotificationChannel
from core.alerts.rules import (
    AlertCategory,
    AlertCondition,
    AlertSeverity,
    AlertTrigger,
    ConditionType,
    create_price_alert,
    create_risk_alert,
)
from core.alerts.scheduler import ReportFrequency, ReportType


class TestAlertRules:
    """Tests for alert rules."""

    def test_alert_condition_creation(self):
        """Test creating an alert condition."""
        condition = AlertCondition(
            condition_type=ConditionType.PRICE_ABOVE,
            threshold=100.0,
            symbol="WTI",
        )

        assert condition.condition_type == ConditionType.PRICE_ABOVE
        assert condition.threshold == 100.0
        assert condition.symbol == "WTI"

    def test_alert_condition_evaluate_price_above(self):
        """Test evaluating a PRICE_ABOVE condition."""
        condition = AlertCondition(
            condition_type=ConditionType.PRICE_ABOVE,
            threshold=80.0,
        )

        # Price above threshold - should trigger
        assert condition.evaluate({"price": 85.0}) is True
        # Price below threshold - should not trigger
        assert condition.evaluate({"price": 75.0}) is False

    def test_alert_condition_evaluate_price_below(self):
        """Test evaluating a PRICE_BELOW condition."""
        condition = AlertCondition(
            condition_type=ConditionType.PRICE_BELOW,
            threshold=70.0,
        )

        # Price below threshold - should trigger
        assert condition.evaluate({"price": 65.0}) is True
        # Price above threshold - should not trigger
        assert condition.evaluate({"price": 75.0}) is False

    def test_alert_trigger_creation(self):
        """Test creating an alert trigger."""
        trigger = AlertTrigger(
            trigger_id=f"ALERT-{uuid.uuid4().hex[:8]}",
            rule_id="test_rule",
            timestamp=datetime.now(),
            severity=AlertSeverity.HIGH,
            category=AlertCategory.PRICE,
            title="Test Alert",
            message="Price exceeded threshold",
        )

        assert trigger.rule_id == "test_rule"
        assert trigger.severity == AlertSeverity.HIGH
        assert trigger.acknowledged is False

    def test_create_price_alert(self):
        """Test price alert factory function."""
        config = create_price_alert(
            name="Test Price Alert",
            symbol="Brent",
            price_level=80.0,
            direction="above",
            severity=AlertSeverity.HIGH,
        )

        assert config.name == "Test Price Alert"
        assert config.severity == AlertSeverity.HIGH
        assert config.category == AlertCategory.PRICE
        assert len(config.conditions) == 1
        assert config.conditions[0].condition_type == ConditionType.PRICE_ABOVE

    def test_create_risk_alert(self):
        """Test risk alert factory function."""
        config = create_risk_alert(
            name="Test Risk Alert",
            risk_type="var",
            threshold=100000,
            severity=AlertSeverity.CRITICAL,
        )

        assert config.name == "Test Risk Alert"
        assert config.category == AlertCategory.RISK
        assert config.severity == AlertSeverity.CRITICAL
        assert len(config.conditions) == 1

    def test_alert_rule_creation(self):
        """Test creating an AlertRule."""
        config = create_price_alert(
            name="Test",
            symbol="WTI",
            price_level=75.0,
        )
        rule = AlertRule(config=config)

        assert rule.config.name == "Test"
        assert rule.is_active  # Should be active by default

    def test_alert_rule_to_dict(self):
        """Test alert rule serialization."""
        config = create_price_alert(
            name="Test",
            symbol="WTI",
            price_level=75.0,
        )
        rule = AlertRule(config=config)

        data = rule.to_dict()

        assert data["name"] == "Test"
        assert "conditions" in data
        assert "severity" in data

    def test_alert_rule_evaluate(self):
        """Test alert rule evaluation."""
        config = create_price_alert(
            name="WTI Breakout",
            symbol="WTI",
            price_level=80.0,
            direction="above",
        )
        rule = AlertRule(config=config)

        # Price above threshold - should trigger
        trigger = rule.evaluate({"price": 85.0})
        assert trigger is not None
        assert trigger.category == AlertCategory.PRICE


class TestAlertEngine:
    """Tests for alert engine."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp = tempfile.mkdtemp()
        yield Path(temp)
        shutil.rmtree(temp)

    @pytest.fixture
    def engine(self, temp_dir):
        """Create alert engine for tests."""
        from core.alerts.engine import AlertEngineConfig
        config = AlertEngineConfig(storage_path=str(temp_dir / "alerts"))
        return AlertEngine(config=config)

    def test_engine_initialization(self, engine):
        """Test engine initialization."""
        assert engine is not None

    def test_add_rule(self, engine):
        """Test adding a rule to the engine."""
        config = create_price_alert(
            name="Test Alert",
            symbol="WTI",
            price_level=80.0,
        )

        # AlertEngine.add_rule expects AlertConfig, not AlertRule
        engine.add_rule(config)

        rules = engine.get_rules()
        assert len(rules) >= 1

    def test_remove_rule(self, engine):
        """Test removing a rule from the engine."""
        config = create_price_alert(
            name="Test Alert",
            symbol="WTI",
            price_level=80.0,
        )

        engine.add_rule(config)
        initial_count = len(engine.get_rules())

        engine.remove_rule(config.rule_id)

        assert len(engine.get_rules()) == initial_count - 1

    def test_get_rules(self, engine):
        """Test getting all rules."""
        config1 = create_price_alert("Rule 1", "WTI", 80.0)
        config2 = create_price_alert("Rule 2", "Brent", 85.0)

        engine.add_rule(config1)
        engine.add_rule(config2)

        rules = engine.get_rules()

        assert len(rules) >= 2


class TestAlertHistory:
    """Tests for alert history."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp = tempfile.mkdtemp()
        yield Path(temp)
        shutil.rmtree(temp)

    @pytest.fixture
    def history(self, temp_dir):
        """Create alert history for tests."""
        return AlertHistory(db_path=str(temp_dir / "alerts.db"))

    def test_record_alert(self, history):
        """Test recording an alert."""
        record = AlertRecord(
            record_id="alert1",
            trigger_id="trigger1",
            rule_id="rule1",
            rule_name="Test Rule",
            severity=AlertSeverity.HIGH,
            category=AlertCategory.PRICE,
            title="Test Alert",
            message="Price exceeded threshold",
            created_at=datetime.now(),
        )

        history.add(record)

        # Verify it was recorded
        retrieved = history.get("alert1")
        assert retrieved is not None
        assert retrieved.title == "Test Alert"

    def test_query_alerts(self, history):
        """Test querying alerts."""
        # Add multiple alerts
        for i in range(5):
            record = AlertRecord(
                record_id=f"alert{i}",
                trigger_id=f"trigger{i}",
                rule_id="rule1",
                rule_name="Test Rule",
                severity=AlertSeverity.MEDIUM,
                category=AlertCategory.PRICE,
                title=f"Alert {i}",
                message="Test",
                created_at=datetime.now(),
            )
            history.add(record)

        # Query all
        alerts = history.query(limit=10)
        assert len(alerts) == 5

    def test_update_alert_status(self, history):
        """Test updating alert status."""
        record = AlertRecord(
            record_id="alert1",
            trigger_id="trigger1",
            rule_id="rule1",
            rule_name="Test Rule",
            severity=AlertSeverity.HIGH,
            category=AlertCategory.PRICE,
            title="Test Alert",
            message="Test",
            created_at=datetime.now(),
        )

        history.add(record)

        # Update status
        history.update("alert1", {"acknowledged": True, "acknowledged_at": datetime.now()})

        retrieved = history.get("alert1")
        assert retrieved.acknowledged is True


class TestReportScheduler:
    """Tests for report scheduler."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp = tempfile.mkdtemp()
        yield Path(temp)
        shutil.rmtree(temp)

    @pytest.fixture
    def scheduler(self, temp_dir):
        """Create scheduler for tests."""
        return ReportScheduler(storage_path=str(temp_dir))

    def test_add_report(self, scheduler):
        """Test adding a scheduled report."""
        config = ReportConfig(
            report_id="daily_pnl",
            name="Daily P&L Report",
            report_type=ReportType.DAILY_PNL,
            frequency=ReportFrequency.DAILY,
            schedule_time=time(18, 0),
            enabled=True,
        )

        scheduler.add_report(config)

        report = scheduler.get_report("daily_pnl")
        assert report is not None

    def test_get_reports(self, scheduler):
        """Test listing scheduled reports."""
        config1 = ReportConfig(
            report_id="report1",
            name="Report 1",
            report_type=ReportType.DAILY_PNL,
        )
        config2 = ReportConfig(
            report_id="report2",
            name="Report 2",
            report_type=ReportType.DAILY_RISK,
        )

        scheduler.add_report(config1)
        scheduler.add_report(config2)

        reports = scheduler.get_reports()
        assert len(reports) == 2


class MockNotificationChannelForTests(NotificationChannel):
    """Mock notification channel for testing."""

    def __init__(self):
        config = ChannelConfig(name="mock")
        super().__init__(config)
        self.sent_messages = []

    def send(self, event: Any, escalated: bool = False) -> bool:
        """Send a notification - mock implementation."""
        self.sent_messages.append({"event": event, "escalated": escalated})
        self._record_send()
        return True

    def test(self) -> bool:
        """Test the channel - mock implementation."""
        return True


class TestNotificationChannels:
    """Tests for notification channels."""

    def test_mock_channel_send(self):
        """Test sending via mock channel."""
        channel = MockNotificationChannelForTests()

        # Create a simple event-like object
        event = {"title": "Test Alert", "message": "This is a test message"}

        success = channel.send(event)

        assert success is True
        assert len(channel.sent_messages) == 1
        assert channel.sent_messages[0]["event"]["title"] == "Test Alert"

    def test_channel_test_method(self):
        """Test channel test method."""
        channel = MockNotificationChannelForTests()

        # Test should return True for mock
        result = channel.test()
        assert result is True

    def test_channel_statistics(self):
        """Test channel statistics."""
        channel = MockNotificationChannelForTests()

        # Send a few messages
        for i in range(3):
            channel.send({"title": f"Test {i}"})

        stats = channel.get_statistics()
        assert stats["send_count"] == 3
        assert stats["name"] == "mock"
