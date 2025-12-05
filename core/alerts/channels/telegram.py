"""
Telegram Notification Channel
=============================
Telegram bot notifications for alerts.
"""

import json
import logging
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from .base import ChannelConfig, NotificationChannel

logger = logging.getLogger(__name__)


@dataclass
class TelegramConfig(ChannelConfig):
    """Configuration for Telegram notifications."""
    # Bot settings
    bot_token: str = ""

    # Chat IDs to send to
    chat_ids: list[str] = field(default_factory=list)

    # Escalation chat IDs
    escalation_chat_ids: list[str] = field(default_factory=list)

    # Message settings
    parse_mode: str = "HTML"  # HTML or Markdown
    disable_web_preview: bool = True
    disable_notification: bool = False

    # For escalated alerts
    escalation_disable_notification: bool = False


class TelegramChannel(NotificationChannel):
    """
    Telegram notification channel using Bot API.

    Supports:
    - Multiple chat IDs
    - HTML and Markdown formatting
    - Silent notifications
    - Escalation to additional chats
    """

    API_BASE = "https://api.telegram.org/bot"

    def __init__(self, config: TelegramConfig):
        super().__init__(config)
        self.telegram_config = config

    def send(self, event: Any, escalated: bool = False) -> bool:
        """Send Telegram notification."""
        if not self.is_enabled:
            return False

        if not self._check_rate_limit():
            return False

        if not self.telegram_config.bot_token:
            logger.warning("Telegram bot token not configured")
            return False

        # Get alert details
        trigger = event.trigger

        if not self._should_send(trigger.severity.value, trigger.category.value):
            return False

        # Build chat ID list
        chat_ids = self.telegram_config.chat_ids.copy()
        if escalated:
            chat_ids.extend(self.telegram_config.escalation_chat_ids)

        if not chat_ids:
            logger.warning("No Telegram chat IDs configured")
            return False

        try:
            # Create message
            message = self._create_message(event, escalated)

            # Send to all chats
            success_count = 0
            for chat_id in chat_ids:
                if self._send_message(chat_id, message, escalated):
                    success_count += 1

            if success_count > 0:
                self._record_send()
                logger.info(f"Telegram sent to {success_count}/{len(chat_ids)} chats")
                return True

            return False

        except Exception as e:
            error_msg = f"Failed to send Telegram: {e}"
            self._record_error(error_msg)
            logger.error(error_msg)
            return False

    def test(self) -> bool:
        """Test Telegram configuration."""
        if not self.telegram_config.bot_token:
            logger.error("Telegram bot token not configured")
            return False

        try:
            # Test bot by getting info
            url = f"{self.API_BASE}{self.telegram_config.bot_token}/getMe"

            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())

                if data.get("ok"):
                    bot_info = data.get("result", {})
                    logger.info(f"Telegram bot test successful: @{bot_info.get('username')}")
                    return True
                else:
                    logger.error(f"Telegram test failed: {data.get('description')}")
                    return False

        except Exception as e:
            logger.error(f"Telegram channel test failed: {e}")
            return False

    def _create_message(self, event: Any, escalated: bool) -> str:
        """Create Telegram message."""
        trigger = event.trigger

        severity_emoji = {
            "INFO": "ℹ️",
            "LOW": "📝",
            "MEDIUM": "⚠️",
            "HIGH": "🔶",
            "CRITICAL": "🔴",
        }.get(trigger.severity.value, "📢")

        category_emoji = {
            "SIGNAL": "📡",
            "RISK": "🛡️",
            "PRICE": "💰",
            "POSITION": "📊",
            "EXECUTION": "⚡",
            "SYSTEM": "🖥️",
            "MARKET": "📈",
            "PNL": "💵",
            "COMPLIANCE": "📋",
        }.get(trigger.category.value, "🔔")

        if self.telegram_config.parse_mode == "HTML":
            return self._create_html_message(event, escalated, severity_emoji, category_emoji)
        else:
            return self._create_markdown_message(event, escalated, severity_emoji, category_emoji)

    def _create_html_message(
        self,
        event: Any,
        escalated: bool,
        severity_emoji: str,
        category_emoji: str,
    ) -> str:
        """Create HTML formatted message."""
        trigger = event.trigger

        parts = []

        if escalated:
            parts.append("🚨 <b>ESCALATED ALERT</b> 🚨")
            parts.append("")

        parts.append(f"{severity_emoji} <b>{trigger.title}</b>")
        parts.append("")
        parts.append(f"<b>Severity:</b> {trigger.severity.value}")
        parts.append(f"<b>Category:</b> {category_emoji} {trigger.category.value}")
        parts.append("")
        parts.append(trigger.message)
        parts.append("")
        parts.append(f"📅 <i>{trigger.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</i>")
        parts.append(f"🔖 <code>{trigger.trigger_id}</code>")

        return "\n".join(parts)

    def _create_markdown_message(
        self,
        event: Any,
        escalated: bool,
        severity_emoji: str,
        category_emoji: str,
    ) -> str:
        """Create Markdown formatted message."""
        trigger = event.trigger

        parts = []

        if escalated:
            parts.append("🚨 *ESCALATED ALERT* 🚨")
            parts.append("")

        parts.append(f"{severity_emoji} *{trigger.title}*")
        parts.append("")
        parts.append(f"*Severity:* {trigger.severity.value}")
        parts.append(f"*Category:* {category_emoji} {trigger.category.value}")
        parts.append("")
        parts.append(trigger.message)
        parts.append("")
        parts.append(f"📅 _{trigger.timestamp.strftime('%Y-%m-%d %H:%M:%S')}_")
        parts.append(f"🔖 `{trigger.trigger_id}`")

        return "\n".join(parts)

    def _send_message(self, chat_id: str, message: str, escalated: bool) -> bool:
        """Send message to a chat."""
        try:
            url = f"{self.API_BASE}{self.telegram_config.bot_token}/sendMessage"

            disable_notification = (
                self.telegram_config.escalation_disable_notification if escalated
                else self.telegram_config.disable_notification
            )

            data = {
                "chat_id": chat_id,
                "text": message,
                "parse_mode": self.telegram_config.parse_mode,
                "disable_web_page_preview": self.telegram_config.disable_web_preview,
                "disable_notification": disable_notification,
            }

            encoded_data = urllib.parse.urlencode(data).encode()
            req = urllib.request.Request(url, data=encoded_data)

            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode())
                return result.get("ok", False)

        except Exception as e:
            logger.error(f"Failed to send to chat {chat_id}: {e}")
            return False
