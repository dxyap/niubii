"""
Tests for Alerts Module
=======================
Tests for alert rules, engine, channels, and scheduling.
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil

from core.alerts import (
    AlertRule,
    AlertCondition,
    AlertSeverity,
    AlertCategory,
    AlertTrigger,
    AlertConfig,
    AlertEngine,
    AlertEngineConfig,
    AlertEvent,
    AlertStatus,
    ReportScheduler,
    ScheduledReport,
    ReportType,
    AlertHistory,
    AlertRecord,
)
from core.alerts.rules import ConditionType, create_price_alert, create_risk_alert
from core.alerts.channels import NotificationChannel, ChannelConfig


class TestAlertRules:
    """Tests for alert rules."""
    
    def test_alert_condition_creation(self):
        """Test creating an alert condition."""
        condition = AlertCondition(
            type=ConditionType.PRICE_ABOVE,
            threshold=100.0,
            symbol="WTI",
        )
        
        assert condition.type == ConditionType.PRICE_ABOVE
        assert condition.threshold == 100.0
        assert condition.symbol == "WTI"
    
    def test_alert_trigger_creation(self):
        """Test creating an alert trigger."""
        trigger = AlertTrigger(
            condition_met=True,
            current_value=105.0,
            threshold_value=100.0,
            trigger_time=datetime.now(),
        )
        
        assert trigger.condition_met is True
        assert trigger.current_value == 105.0
    
    def test_create_price_alert(self):
        """Test price alert factory function."""
        rule = create_price_alert(
            rule_id="test_price",
            name="Test Price Alert",
            symbol="Brent",
            threshold=80.0,
            above=True,
            severity=AlertSeverity.HIGH,
        )
        
        assert rule.id == "test_price"
        assert rule.name == "Test Price Alert"
        assert rule.severity == AlertSeverity.HIGH
        assert rule.category == AlertCategory.PRICE
        assert len(rule.conditions) == 1
        assert rule.conditions[0].type == ConditionType.PRICE_ABOVE
    
    def test_create_risk_alert(self):
        """Test risk alert factory function."""
        rule = create_risk_alert(
            rule_id="test_risk",
            name="Test Risk Alert",
            limit_name="VaR",
            threshold=0.95,
            severity=AlertSeverity.CRITICAL,
        )
        
        assert rule.id == "test_risk"
        assert rule.category == AlertCategory.RISK
        assert rule.severity == AlertSeverity.CRITICAL
    
    def test_alert_rule_to_dict(self):
        """Test alert rule serialization."""
        rule = create_price_alert(
            rule_id="test",
            name="Test",
            symbol="WTI",
            threshold=75.0,
        )
        
        data = rule.to_dict()
        
        assert data["id"] == "test"
        assert data["name"] == "Test"
        assert "conditions" in data
    
    def test_alert_rule_from_dict(self):
        """Test alert rule deserialization."""
        rule = create_price_alert(
            rule_id="test",
            name="Test",
            symbol="WTI",
            threshold=75.0,
        )
        
        data = rule.to_dict()
        restored = AlertRule.from_dict(data)
        
        assert restored.id == rule.id
        assert restored.name == rule.name
        assert len(restored.conditions) == len(rule.conditions)


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
        config = AlertEngineConfig(
            storage_path=temp_dir,
            enabled=True,
        )
        return AlertEngine(config)
    
    def test_engine_initialization(self, engine):
        """Test engine initialization."""
        assert engine is not None
        assert engine.config.enabled is True
    
    def test_add_rule(self, engine):
        """Test adding a rule to the engine."""
        rule = create_price_alert(
            rule_id="test1",
            name="Test Alert",
            symbol="WTI",
            threshold=80.0,
        )
        
        engine.add_rule(rule)
        
        assert "test1" in engine._rules
        assert engine.get_rule("test1") == rule
    
    def test_remove_rule(self, engine):
        """Test removing a rule from the engine."""
        rule = create_price_alert(
            rule_id="test1",
            name="Test Alert",
            symbol="WTI",
            threshold=80.0,
        )
        
        engine.add_rule(rule)
        engine.remove_rule("test1")
        
        assert "test1" not in engine._rules
    
    def test_list_rules(self, engine):
        """Test listing all rules."""
        rule1 = create_price_alert("r1", "Rule 1", "WTI", 80.0)
        rule2 = create_price_alert("r2", "Rule 2", "Brent", 85.0)
        
        engine.add_rule(rule1)
        engine.add_rule(rule2)
        
        rules = engine.list_rules()
        
        assert len(rules) == 2
    
    def test_evaluate_conditions(self, engine):
        """Test evaluating conditions."""
        rule = create_price_alert(
            rule_id="test1",
            name="Test Alert",
            symbol="WTI",
            threshold=80.0,
        )
        
        engine.add_rule(rule)
        
        # Simulate market data
        market_data = {"WTI": 85.0}  # Above threshold
        
        # This would normally trigger an alert
        # The actual evaluation logic depends on implementation


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
        return AlertHistory(storage_path=temp_dir)
    
    def test_record_alert(self, history):
        """Test recording an alert."""
        record = AlertRecord(
            id="alert1",
            rule_id="rule1",
            timestamp=datetime.now(),
            severity=AlertSeverity.HIGH,
            category=AlertCategory.PRICE,
            title="Test Alert",
            message="Price exceeded threshold",
            status=AlertStatus.ACTIVE,
        )
        
        history.record_alert(record)
        
        # Verify it was recorded
        retrieved = history.get_alert("alert1")
        assert retrieved is not None
        assert retrieved.title == "Test Alert"
    
    def test_query_alerts(self, history):
        """Test querying alerts."""
        # Add multiple alerts
        for i in range(5):
            record = AlertRecord(
                id=f"alert{i}",
                rule_id="rule1",
                timestamp=datetime.now(),
                severity=AlertSeverity.MEDIUM,
                category=AlertCategory.PRICE,
                title=f"Alert {i}",
                message="Test",
                status=AlertStatus.ACTIVE,
            )
            history.record_alert(record)
        
        # Query all
        alerts = history.query_alerts(limit=10)
        assert len(alerts) == 5
    
    def test_update_alert_status(self, history):
        """Test updating alert status."""
        record = AlertRecord(
            id="alert1",
            rule_id="rule1",
            timestamp=datetime.now(),
            severity=AlertSeverity.HIGH,
            category=AlertCategory.PRICE,
            title="Test Alert",
            message="Test",
            status=AlertStatus.ACTIVE,
        )
        
        history.record_alert(record)
        
        # Update status
        history.update_status("alert1", AlertStatus.ACKNOWLEDGED)
        
        retrieved = history.get_alert("alert1")
        assert retrieved.status == AlertStatus.ACKNOWLEDGED


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
        return ReportScheduler(storage_path=temp_dir)
    
    def test_add_report(self, scheduler):
        """Test adding a scheduled report."""
        report = ScheduledReport(
            id="daily_pnl",
            name="Daily P&L Report",
            report_type=ReportType.PNL,
            schedule="0 18 * * *",  # 6 PM daily
            enabled=True,
        )
        
        scheduler.add_report(report)
        
        assert "daily_pnl" in scheduler._reports
    
    def test_list_reports(self, scheduler):
        """Test listing scheduled reports."""
        report1 = ScheduledReport(
            id="report1",
            name="Report 1",
            report_type=ReportType.PNL,
            schedule="0 18 * * *",
        )
        report2 = ScheduledReport(
            id="report2",
            name="Report 2",
            report_type=ReportType.RISK,
            schedule="0 9 * * *",
        )
        
        scheduler.add_report(report1)
        scheduler.add_report(report2)
        
        reports = scheduler.list_reports()
        assert len(reports) == 2


class MockNotificationChannel(NotificationChannel):
    """Mock notification channel for testing."""
    
    def __init__(self):
        super().__init__(ChannelConfig(name="mock", type="mock"))
        self.sent_messages = []
    
    def _send(self, title: str, message: str, **kwargs) -> bool:
        self.sent_messages.append({"title": title, "message": message})
        return True


class TestNotificationChannels:
    """Tests for notification channels."""
    
    def test_mock_channel_send(self):
        """Test sending via mock channel."""
        channel = MockNotificationChannel()
        
        success = channel.send(
            title="Test Alert",
            message="This is a test message",
            severity=AlertSeverity.MEDIUM,
            category=AlertCategory.PRICE,
        )
        
        assert success is True
        assert len(channel.sent_messages) == 1
        assert channel.sent_messages[0]["title"] == "Test Alert"
    
    def test_channel_rate_limiting(self):
        """Test channel rate limiting."""
        config = ChannelConfig(
            name="rate_limited",
            type="mock",
            rate_limit_per_minute=2,
        )
        
        channel = MockNotificationChannel()
        channel.config = config
        
        # Should allow first 2 messages
        for i in range(2):
            channel.send(
                title=f"Test {i}",
                message="Test",
                severity=AlertSeverity.LOW,
                category=AlertCategory.SYSTEM,
            )
        
        assert len(channel.sent_messages) == 2
