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

from .base import ChannelConfig, ChannelStatus, NotificationChannel
from .email import EmailChannel, EmailConfig
from .slack import SlackChannel, SlackConfig
from .sms import SMSChannel, SMSConfig
from .telegram import TelegramChannel, TelegramConfig

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
