"""
Base Notification Channel
=========================
Abstract base class for notification channels.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ChannelStatus(Enum):
    """Channel status."""
    ACTIVE = "ACTIVE"
    DISABLED = "DISABLED"
    ERROR = "ERROR"
    RATE_LIMITED = "RATE_LIMITED"


@dataclass
class ChannelConfig:
    """Base configuration for notification channels."""
    name: str
    enabled: bool = True
    
    # Rate limiting
    max_per_minute: int = 10
    max_per_hour: int = 100
    
    # Retry settings
    max_retries: int = 3
    retry_delay_seconds: int = 5
    
    # Formatting
    include_timestamp: bool = True
    include_severity: bool = True
    max_message_length: int = 4000
    
    # Filtering
    min_severity: str = "LOW"  # Only send alerts >= this severity
    categories: List[str] = field(default_factory=list)  # Empty = all categories


class NotificationChannel(ABC):
    """
    Abstract base class for notification channels.
    
    All notification channels must implement:
    - send(): Send a notification
    - test(): Test the channel connection
    """
    
    def __init__(self, config: ChannelConfig):
        self.config = config
        self._status = ChannelStatus.ACTIVE
        self._last_error: Optional[str] = None
        self._send_count = 0
        self._error_count = 0
        self._last_send: Optional[datetime] = None
        
        # Rate limiting
        self._minute_sends: List[datetime] = []
        self._hour_sends: List[datetime] = []
    
    @property
    def name(self) -> str:
        return self.config.name
    
    @property
    def is_enabled(self) -> bool:
        return self.config.enabled and self._status == ChannelStatus.ACTIVE
    
    @property
    def status(self) -> ChannelStatus:
        return self._status
    
    @abstractmethod
    def send(self, event: Any, escalated: bool = False) -> bool:
        """
        Send a notification.
        
        Args:
            event: Alert event to send
            escalated: Whether this is an escalated alert
            
        Returns:
            True if send was successful
        """
        pass
    
    @abstractmethod
    def test(self) -> bool:
        """
        Test the channel connection.
        
        Returns:
            True if test was successful
        """
        pass
    
    def _check_rate_limit(self) -> bool:
        """Check if within rate limits."""
        now = datetime.now()
        
        # Clean old entries
        minute_ago = now.timestamp() - 60
        hour_ago = now.timestamp() - 3600
        
        self._minute_sends = [t for t in self._minute_sends if t.timestamp() > minute_ago]
        self._hour_sends = [t for t in self._hour_sends if t.timestamp() > hour_ago]
        
        # Check limits
        if len(self._minute_sends) >= self.config.max_per_minute:
            logger.warning(f"Channel {self.name}: Rate limited (per minute)")
            self._status = ChannelStatus.RATE_LIMITED
            return False
        
        if len(self._hour_sends) >= self.config.max_per_hour:
            logger.warning(f"Channel {self.name}: Rate limited (per hour)")
            self._status = ChannelStatus.RATE_LIMITED
            return False
        
        if self._status == ChannelStatus.RATE_LIMITED:
            self._status = ChannelStatus.ACTIVE
        
        return True
    
    def _record_send(self):
        """Record a successful send."""
        now = datetime.now()
        self._minute_sends.append(now)
        self._hour_sends.append(now)
        self._send_count += 1
        self._last_send = now
    
    def _record_error(self, error: str):
        """Record an error."""
        self._error_count += 1
        self._last_error = error
        
        if self._error_count >= 5:
            self._status = ChannelStatus.ERROR
    
    def _format_message(
        self,
        title: str,
        message: str,
        severity: str,
        category: str,
        timestamp: datetime,
        escalated: bool = False,
    ) -> str:
        """Format a notification message."""
        parts = []
        
        if escalated:
            parts.append("🚨 ESCALATED ALERT 🚨")
            parts.append("")
        
        if self.config.include_severity:
            severity_emoji = {
                "INFO": "ℹ️",
                "LOW": "📝",
                "MEDIUM": "⚠️",
                "HIGH": "🔶",
                "CRITICAL": "🔴",
            }.get(severity, "📢")
            parts.append(f"{severity_emoji} [{severity}] {title}")
        else:
            parts.append(title)
        
        parts.append("")
        parts.append(message)
        
        if self.config.include_timestamp:
            parts.append("")
            parts.append(f"📅 {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        
        full_message = "\n".join(parts)
        
        # Truncate if needed
        if len(full_message) > self.config.max_message_length:
            full_message = full_message[:self.config.max_message_length - 3] + "..."
        
        return full_message
    
    def _should_send(self, severity: str, category: str) -> bool:
        """Check if alert should be sent based on filters."""
        severity_order = ["INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
        
        # Check severity
        min_idx = severity_order.index(self.config.min_severity)
        current_idx = severity_order.index(severity) if severity in severity_order else 0
        if current_idx < min_idx:
            return False
        
        # Check category
        if self.config.categories and category not in self.config.categories:
            return False
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get channel statistics."""
        return {
            "name": self.name,
            "status": self._status.value,
            "enabled": self.config.enabled,
            "send_count": self._send_count,
            "error_count": self._error_count,
            "last_send": self._last_send.isoformat() if self._last_send else None,
            "last_error": self._last_error,
        }
    
    def reset_error_state(self):
        """Reset error state."""
        self._status = ChannelStatus.ACTIVE
        self._error_count = 0
        self._last_error = None
    
    def enable(self):
        """Enable the channel."""
        self.config.enabled = True
        self.reset_error_state()
    
    def disable(self):
        """Disable the channel."""
        self.config.enabled = False
        self._status = ChannelStatus.DISABLED
