"""
SMS Notification Channel
========================
SMS notifications via Twilio.
"""

import base64
import logging
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from .base import ChannelConfig, NotificationChannel

logger = logging.getLogger(__name__)


@dataclass
class SMSConfig(ChannelConfig):
    """Configuration for SMS notifications."""
    # Twilio credentials
    account_sid: str = ""
    auth_token: str = ""

    # Phone numbers
    from_number: str = ""
    to_numbers: list[str] = field(default_factory=list)

    # Escalation phone numbers
    escalation_numbers: list[str] = field(default_factory=list)

    # Message settings
    max_message_length: int = 160  # SMS limit
    include_alert_id: bool = False


class SMSChannel(NotificationChannel):
    """
    SMS notification channel using Twilio.

    Supports:
    - Multiple recipients
    - Message truncation for SMS limits
    - Escalation to additional numbers
    """

    TWILIO_API_BASE = "https://api.twilio.com/2010-04-01"

    def __init__(self, config: SMSConfig):
        super().__init__(config)
        self.sms_config = config

    def send(self, event: Any, escalated: bool = False) -> bool:
        """Send SMS notification."""
        if not self.is_enabled:
            return False

        if not self._check_rate_limit():
            return False

        if not all([
            self.sms_config.account_sid,
            self.sms_config.auth_token,
            self.sms_config.from_number,
        ]):
            logger.warning("Twilio credentials not configured")
            return False

        # Get alert details
        trigger = event.trigger

        if not self._should_send(trigger.severity.value, trigger.category.value):
            return False

        # Build recipient list
        recipients = self.sms_config.to_numbers.copy()
        if escalated:
            recipients.extend(self.sms_config.escalation_numbers)

        if not recipients:
            logger.warning("No SMS recipients configured")
            return False

        try:
            # Create message
            message = self._create_message(event, escalated)

            # Send to all recipients
            success_count = 0
            for phone in recipients:
                if self._send_sms(phone, message):
                    success_count += 1

            if success_count > 0:
                self._record_send()
                logger.info(f"SMS sent to {success_count}/{len(recipients)} recipients")
                return True

            return False

        except Exception as e:
            error_msg = f"Failed to send SMS: {e}"
            self._record_error(error_msg)
            logger.error(error_msg)
            return False

    def test(self) -> bool:
        """Test SMS configuration."""
        if not all([
            self.sms_config.account_sid,
            self.sms_config.auth_token,
        ]):
            logger.error("Twilio credentials not configured")
            return False

        try:
            # Test API connection
            url = f"{self.TWILIO_API_BASE}/Accounts/{self.sms_config.account_sid}.json"

            req = self._create_request(url)

            with urllib.request.urlopen(req, timeout=10) as response:
                response.read().decode()
                logger.info("SMS channel test successful")
                return True

        except Exception as e:
            logger.error(f"SMS channel test failed: {e}")
            return False

    def _create_message(self, event: Any, escalated: bool) -> str:
        """Create SMS message."""
        trigger = event.trigger

        severity_abbrev = {
            "INFO": "INFO",
            "LOW": "LOW",
            "MEDIUM": "MED",
            "HIGH": "HIGH",
            "CRITICAL": "CRIT",
        }.get(trigger.severity.value, "ALERT")

        parts = []

        if escalated:
            parts.append("⚠️ESCALATED")

        parts.append(f"[{severity_abbrev}]")
        parts.append(trigger.title[:50])  # Truncate title

        # Build message
        message = " ".join(parts)

        # Add alert ID if enabled and space permits
        if self.sms_config.include_alert_id:
            alert_suffix = f" ID:{trigger.trigger_id[:8]}"
            if len(message) + len(alert_suffix) <= self.sms_config.max_message_length:
                message += alert_suffix

        # Truncate if needed
        if len(message) > self.sms_config.max_message_length:
            message = message[:self.sms_config.max_message_length - 3] + "..."

        return message

    def _create_request(self, url: str, data: dict | None = None) -> urllib.request.Request:
        """Create authenticated request."""
        if data:
            encoded_data = urllib.parse.urlencode(data).encode()
            req = urllib.request.Request(url, data=encoded_data)
        else:
            req = urllib.request.Request(url)

        # Add Basic Auth
        credentials = f"{self.sms_config.account_sid}:{self.sms_config.auth_token}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()
        req.add_header("Authorization", f"Basic {encoded_credentials}")

        return req

    def _send_sms(self, to_number: str, message: str) -> bool:
        """Send SMS to a phone number."""
        try:
            url = (
                f"{self.TWILIO_API_BASE}/Accounts/{self.sms_config.account_sid}"
                f"/Messages.json"
            )

            data = {
                "From": self.sms_config.from_number,
                "To": to_number,
                "Body": message,
            }

            req = self._create_request(url, data)

            with urllib.request.urlopen(req, timeout=10) as response:
                response.read().decode()
                return response.status == 201

        except urllib.error.HTTPError as e:
            logger.error(f"SMS send error to {to_number}: {e.code}")
            return False
        except Exception as e:
            logger.error(f"SMS send error to {to_number}: {e}")
            return False
