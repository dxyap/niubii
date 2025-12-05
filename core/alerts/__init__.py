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

from .rules import (
    AlertRule,
    AlertCondition,
    AlertSeverity,
    AlertCategory,
    AlertTrigger,
    AlertConfig,
)

from .engine import (
    AlertEngine,
    AlertEvent,
    AlertStatus,
    AlertEngineConfig,
)

from .channels import (
    NotificationChannel,
    EmailChannel,
    TelegramChannel,
    SlackChannel,
    SMSChannel,
    ChannelConfig,
)

from .scheduler import (
    ReportScheduler,
    ScheduledReport,
    ReportType,
    ReportConfig,
)

from .history import (
    AlertHistory,
    AlertRecord,
    AlertAuditLog,
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
