"""
Alerts & Notifications Module
=============================
Multi-channel alert system for trading signals, risk breaches, and market events.

This module provides:
- Alert rules engine for configurable conditions and triggers
- Multi-channel notifications (Email, Telegram, Slack, SMS)
- Scheduled reports for daily/weekly summaries
- Alert history and audit logging
- Alert escalation for critical unacknowledged alerts

Phase 7 Implementation - Alerts & Notifications
"""

from .channels import (
    ChannelConfig,
    EmailChannel,
    NotificationChannel,
    SlackChannel,
    SMSChannel,
    TelegramChannel,
)
from .engine import (
    AlertEngine,
    AlertEngineConfig,
    AlertEvent,
    AlertStatus,
)
from .history import (
    AlertAuditLog,
    AlertHistory,
    AlertRecord,
)
from .rules import (
    AlertCategory,
    AlertCondition,
    AlertConfig,
    AlertRule,
    AlertSeverity,
    AlertTrigger,
)
from .scheduler import (
    ReportConfig,
    ReportScheduler,
    ReportType,
    ScheduledReport,
)

__all__ = [
    # Rules
    "AlertRule",
    "AlertCondition",
    "AlertSeverity",
    "AlertCategory",
    "AlertTrigger",
    "AlertConfig",
    # Engine
    "AlertEngine",
    "AlertEvent",
    "AlertStatus",
    "AlertEngineConfig",
    # Channels
    "NotificationChannel",
    "EmailChannel",
    "TelegramChannel",
    "SlackChannel",
    "SMSChannel",
    "ChannelConfig",
    # Scheduler
    "ReportScheduler",
    "ScheduledReport",
    "ReportType",
    "ReportConfig",
    # History
    "AlertHistory",
    "AlertRecord",
    "AlertAuditLog",
]
