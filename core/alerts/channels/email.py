"""
Email Notification Channel
==========================
SMTP-based email notifications for alerts.
"""

import logging
import smtplib
import ssl
from dataclasses import dataclass, field
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

from .base import ChannelConfig, NotificationChannel

logger = logging.getLogger(__name__)


@dataclass
class EmailConfig(ChannelConfig):
    """Configuration for email notifications."""
    # SMTP settings
    smtp_server: str = ""
    smtp_port: int = 587
    use_tls: bool = True
    use_ssl: bool = False

    # Authentication
    username: str = ""
    password: str = ""

    # Email settings
    from_address: str = ""
    from_name: str = "Oil Trading Alerts"

    # Recipients
    recipients: list[str] = field(default_factory=list)
    cc_recipients: list[str] = field(default_factory=list)

    # Escalation recipients
    escalation_recipients: list[str] = field(default_factory=list)

    # Template settings
    use_html: bool = True
    subject_prefix: str = "[OIL TRADING]"


class EmailChannel(NotificationChannel):
    """
    Email notification channel using SMTP.

    Supports:
    - Plain text and HTML emails
    - Multiple recipients with CC
    - TLS/SSL encryption
    - Escalation to additional recipients
    """

    def __init__(self, config: EmailConfig):
        super().__init__(config)
        self.email_config = config

    def send(self, event: Any, escalated: bool = False) -> bool:
        """Send email notification."""
        if not self.is_enabled:
            return False

        if not self._check_rate_limit():
            return False

        # Get alert details
        trigger = event.trigger

        if not self._should_send(trigger.severity.value, trigger.category.value):
            return False

        # Build recipient list
        recipients = self.email_config.recipients.copy()
        if escalated:
            recipients.extend(self.email_config.escalation_recipients)

        if not recipients:
            logger.warning("No email recipients configured")
            return False

        try:
            # Create message
            msg = self._create_message(event, escalated)

            # Send via SMTP
            self._send_smtp(msg, recipients)

            self._record_send()
            logger.info(f"Email sent to {len(recipients)} recipients")
            return True

        except Exception as e:
            error_msg = f"Failed to send email: {e}"
            self._record_error(error_msg)
            logger.error(error_msg)
            return False

    def test(self) -> bool:
        """Test email configuration."""
        if not self.email_config.smtp_server:
            logger.error("SMTP server not configured")
            return False

        try:
            # Test SMTP connection
            context = ssl.create_default_context()

            if self.email_config.use_ssl:
                server = smtplib.SMTP_SSL(
                    self.email_config.smtp_server,
                    self.email_config.smtp_port,
                    context=context,
                )
            else:
                server = smtplib.SMTP(
                    self.email_config.smtp_server,
                    self.email_config.smtp_port,
                )
                if self.email_config.use_tls:
                    server.starttls(context=context)

            if self.email_config.username and self.email_config.password:
                server.login(self.email_config.username, self.email_config.password)

            server.quit()

            logger.info("Email channel test successful")
            return True

        except Exception as e:
            logger.error(f"Email channel test failed: {e}")
            return False

    def _create_message(self, event: Any, escalated: bool) -> MIMEMultipart:
        """Create email message."""
        trigger = event.trigger

        # Create message
        msg = MIMEMultipart("alternative")

        # Subject
        severity_emoji = {
            "INFO": "ℹ️",
            "LOW": "📝",
            "MEDIUM": "⚠️",
            "HIGH": "🔶",
            "CRITICAL": "🔴",
        }.get(trigger.severity.value, "📢")

        prefix = self.email_config.subject_prefix
        escalation_text = "[ESCALATED] " if escalated else ""
        msg["Subject"] = f"{prefix} {escalation_text}{severity_emoji} {trigger.title}"

        msg["From"] = f"{self.email_config.from_name} <{self.email_config.from_address}>"
        msg["To"] = ", ".join(self.email_config.recipients)

        if self.email_config.cc_recipients:
            msg["Cc"] = ", ".join(self.email_config.cc_recipients)

        # Plain text body
        text_body = self._format_message(
            title=trigger.title,
            message=trigger.message,
            severity=trigger.severity.value,
            category=trigger.category.value,
            timestamp=trigger.timestamp,
            escalated=escalated,
        )
        msg.attach(MIMEText(text_body, "plain"))

        # HTML body
        if self.email_config.use_html:
            html_body = self._create_html_body(event, escalated)
            msg.attach(MIMEText(html_body, "html"))

        return msg

    def _create_html_body(self, event: Any, escalated: bool) -> str:
        """Create HTML email body."""
        trigger = event.trigger

        severity_colors = {
            "INFO": "#17a2b8",
            "LOW": "#28a745",
            "MEDIUM": "#ffc107",
            "HIGH": "#fd7e14",
            "CRITICAL": "#dc3545",
        }

        color = severity_colors.get(trigger.severity.value, "#6c757d")

        escalation_banner = ""
        if escalated:
            escalation_banner = """
            <div style="background-color: #dc3545; color: white; padding: 10px; text-align: center; margin-bottom: 20px;">
                🚨 ESCALATED ALERT - IMMEDIATE ATTENTION REQUIRED 🚨
            </div>
            """

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f4f4f4; }}
                .container {{ max-width: 600px; margin: 0 auto; background: white; border-radius: 8px; overflow: hidden; }}
                .header {{ background-color: {color}; color: white; padding: 20px; }}
                .content {{ padding: 20px; }}
                .severity-badge {{ display: inline-block; padding: 4px 12px; border-radius: 4px; background-color: {color}; color: white; font-weight: bold; }}
                .meta {{ color: #666; font-size: 12px; margin-top: 20px; padding-top: 10px; border-top: 1px solid #eee; }}
                .category {{ background-color: #e9ecef; padding: 4px 8px; border-radius: 4px; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                {escalation_banner}
                <div class="header">
                    <h2 style="margin: 0;">{trigger.title}</h2>
                </div>
                <div class="content">
                    <p><span class="severity-badge">{trigger.severity.value}</span> <span class="category">{trigger.category.value}</span></p>
                    <p>{trigger.message}</p>
                    <div class="meta">
                        <p>📅 Time: {trigger.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")}</p>
                        <p>🔖 Alert ID: {trigger.trigger_id}</p>
                        <p>📋 Rule: {event.rule.config.name}</p>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """

        return html

    def _send_smtp(self, msg: MIMEMultipart, recipients: list[str]):
        """Send message via SMTP."""
        context = ssl.create_default_context()

        if self.email_config.use_ssl:
            server = smtplib.SMTP_SSL(
                self.email_config.smtp_server,
                self.email_config.smtp_port,
                context=context,
            )
        else:
            server = smtplib.SMTP(
                self.email_config.smtp_server,
                self.email_config.smtp_port,
            )
            if self.email_config.use_tls:
                server.starttls(context=context)

        try:
            if self.email_config.username and self.email_config.password:
                server.login(self.email_config.username, self.email_config.password)

            server.sendmail(
                self.email_config.from_address,
                recipients,
                msg.as_string(),
            )
        finally:
            server.quit()
