"""
Slack Notification Channel
==========================
Slack webhook integration for alerts.
"""

import json
import logging
import urllib.request
from dataclasses import dataclass
from typing import Any

from .base import ChannelConfig, NotificationChannel

logger = logging.getLogger(__name__)


@dataclass
class SlackConfig(ChannelConfig):
    """Configuration for Slack notifications."""
    # Webhook URLs
    webhook_url: str = ""

    # Escalation webhook
    escalation_webhook_url: str = ""

    # Channel settings
    default_channel: str = ""  # Override channel if webhook allows
    username: str = "Oil Trading Bot"
    icon_emoji: str = ":chart_with_upwards_trend:"

    # Message settings
    include_footer: bool = True
    include_context: bool = True


class SlackChannel(NotificationChannel):
    """
    Slack notification channel using incoming webhooks.

    Supports:
    - Rich message formatting with attachments
    - Color-coded severity
    - Message actions
    - Escalation to different channels
    """

    def __init__(self, config: SlackConfig):
        super().__init__(config)
        self.slack_config = config

    def send(self, event: Any, escalated: bool = False) -> bool:
        """Send Slack notification."""
        if not self.is_enabled:
            return False

        if not self._check_rate_limit():
            return False

        webhook_url = (
            self.slack_config.escalation_webhook_url if escalated
            else self.slack_config.webhook_url
        )

        if not webhook_url:
            # Fall back to main webhook
            webhook_url = self.slack_config.webhook_url

        if not webhook_url:
            logger.warning("Slack webhook URL not configured")
            return False

        # Get alert details
        trigger = event.trigger

        if not self._should_send(trigger.severity.value, trigger.category.value):
            return False

        try:
            # Create payload
            payload = self._create_payload(event, escalated)

            # Send to webhook
            if self._send_webhook(webhook_url, payload):
                self._record_send()
                logger.info("Slack notification sent")
                return True

            return False

        except Exception as e:
            error_msg = f"Failed to send Slack: {e}"
            self._record_error(error_msg)
            logger.error(error_msg)
            return False

    def test(self) -> bool:
        """Test Slack configuration."""
        if not self.slack_config.webhook_url:
            logger.error("Slack webhook URL not configured")
            return False

        try:
            payload = {
                "text": "🔔 Test notification from Oil Trading Alerts",
                "username": self.slack_config.username,
                "icon_emoji": self.slack_config.icon_emoji,
            }

            if self._send_webhook(self.slack_config.webhook_url, payload):
                logger.info("Slack channel test successful")
                return True

            return False

        except Exception as e:
            logger.error(f"Slack channel test failed: {e}")
            return False

    def _create_payload(self, event: Any, escalated: bool) -> dict:
        """Create Slack message payload."""
        trigger = event.trigger

        # Severity color
        severity_colors = {
            "INFO": "#17a2b8",
            "LOW": "#28a745",
            "MEDIUM": "#ffc107",
            "HIGH": "#fd7e14",
            "CRITICAL": "#dc3545",
        }
        color = severity_colors.get(trigger.severity.value, "#6c757d")

        # Severity emoji
        severity_emoji = {
            "INFO": ":information_source:",
            "LOW": ":white_check_mark:",
            "MEDIUM": ":warning:",
            "HIGH": ":large_orange_diamond:",
            "CRITICAL": ":red_circle:",
        }.get(trigger.severity.value, ":bell:")

        # Category emoji
        category_emoji = {
            "SIGNAL": ":satellite_antenna:",
            "RISK": ":shield:",
            "PRICE": ":moneybag:",
            "POSITION": ":bar_chart:",
            "EXECUTION": ":zap:",
            "SYSTEM": ":desktop_computer:",
            "MARKET": ":chart_with_upwards_trend:",
            "PNL": ":dollar:",
            "COMPLIANCE": ":clipboard:",
        }.get(trigger.category.value, ":bell:")

        # Build attachment
        attachment = {
            "color": color,
            "fallback": f"[{trigger.severity.value}] {trigger.title}",
            "title": trigger.title,
            "text": trigger.message,
            "fields": [
                {
                    "title": "Severity",
                    "value": f"{severity_emoji} {trigger.severity.value}",
                    "short": True,
                },
                {
                    "title": "Category",
                    "value": f"{category_emoji} {trigger.category.value}",
                    "short": True,
                },
            ],
            "ts": int(trigger.timestamp.timestamp()),
        }

        # Add context if enabled
        if self.slack_config.include_context:
            attachment["fields"].append({
                "title": "Rule",
                "value": event.rule.config.name,
                "short": True,
            })
            attachment["fields"].append({
                "title": "Alert ID",
                "value": f"`{trigger.trigger_id}`",
                "short": True,
            })

        # Add footer if enabled
        if self.slack_config.include_footer:
            attachment["footer"] = "Oil Trading Dashboard"
            attachment["footer_icon"] = "https://raw.githubusercontent.com/streamlit/streamlit/develop/frontend/public/favicon.png"

        # Build payload
        payload = {
            "username": self.slack_config.username,
            "icon_emoji": self.slack_config.icon_emoji,
            "attachments": [attachment],
        }

        # Escalation header
        if escalated:
            payload["text"] = ":rotating_light: *ESCALATED ALERT* :rotating_light:"

        # Optional channel override
        if self.slack_config.default_channel:
            payload["channel"] = self.slack_config.default_channel

        return payload

    def _send_webhook(self, url: str, payload: dict) -> bool:
        """Send payload to webhook URL."""
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
            )

            with urllib.request.urlopen(req, timeout=10) as response:
                result = response.read().decode()
                return result == "ok"

        except urllib.error.HTTPError as e:
            logger.error(f"Slack webhook error: {e.code} - {e.read().decode()}")
            return False
        except Exception as e:
            logger.error(f"Slack webhook error: {e}")
            return False
