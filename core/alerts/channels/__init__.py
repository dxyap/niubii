"""
Notification Channels
=====================
Multi-channel notification delivery for alerts.

Supported Channels:
- Email: SMTP-based email notifications
- Telegram: Telegram bot notifications
- Slack: Slack webhook integration
- SMS: SMS via Twilio
"""

from .base import NotificationChannel, ChannelConfig, ChannelStatus
from .email import EmailChannel, EmailConfig
from .telegram import TelegramChannel, TelegramConfig
from .slack import SlackChannel, SlackConfig
from .sms import SMSChannel, SMSConfig

__all__ = [
    # Base
    "NotificationChannel",
    "ChannelConfig",
    "ChannelStatus",
    # Email
    "EmailChannel",
    "EmailConfig",
    # Telegram
    "TelegramChannel",
    "TelegramConfig",
    # Slack
    "SlackChannel",
    "SlackConfig",
    # SMS
    "SMSChannel",
    "SMSConfig",
]
