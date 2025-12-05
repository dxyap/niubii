"""
Audit Logging Module
====================
Comprehensive audit trail for security and compliance.
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import threading

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of audit events."""
    # Authentication
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    PASSWORD_CHANGE = "password_change"
    SESSION_EXPIRED = "session_expired"
    TOKEN_CREATED = "token_created"
    TOKEN_REFRESHED = "token_refreshed"
    
    # User management
    USER_CREATED = "user_created"
    USER_UPDATED = "user_updated"
    USER_DEACTIVATED = "user_deactivated"
    USER_ROLE_CHANGED = "user_role_changed"
    
    # Trading
    ORDER_CREATED = "order_created"
    ORDER_MODIFIED = "order_modified"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_EXECUTED = "order_executed"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    
    # Risk
    RISK_LIMIT_SET = "risk_limit_set"
    RISK_LIMIT_BREACHED = "risk_limit_breached"
    RISK_LIMIT_OVERRIDE = "risk_limit_override"
    
    # System
    SYSTEM_STARTED = "system_started"
    SYSTEM_STOPPED = "system_stopped"
    CONFIG_CHANGED = "config_changed"
    ALERT_TRIGGERED = "alert_triggered"
    
    # Data access
    DATA_ACCESSED = "data_accessed"
    DATA_EXPORTED = "data_exported"
    REPORT_GENERATED = "report_generated"
    
    # ML/Research
    MODEL_TRAINED = "model_trained"
    MODEL_DEPLOYED = "model_deployed"
    PREDICTION_MADE = "prediction_made"
    
    # Backtest
    BACKTEST_STARTED = "backtest_started"
    BACKTEST_COMPLETED = "backtest_completed"


class AuditSeverity(Enum):
    """Severity levels for audit events."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Audit event record."""
    id: str
    timestamp: datetime
    event_type: AuditEventType
    severity: AuditSeverity
    user_id: Optional[str]
    username: Optional[str]
    action: str
    resource: Optional[str]
    resource_id: Optional[str]
    details: Dict[str, Any]
    ip_address: Optional[str]
    user_agent: Optional[str]
    success: bool
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "user_id": self.user_id,
            "username": self.username,
            "action": self.action,
            "resource": self.resource,
            "resource_id": self.resource_id,
            "details": self.details,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "success": self.success,
            "error_message": self.error_message,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "AuditEvent":
        return cls(
            id=data["id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            event_type=AuditEventType(data["event_type"]),
            severity=AuditSeverity(data["severity"]),
            user_id=data.get("user_id"),
            username=data.get("username"),
            action=data["action"],
            resource=data.get("resource"),
            resource_id=data.get("resource_id"),
            details=data.get("details", {}),
            ip_address=data.get("ip_address"),
            user_agent=data.get("user_agent"),
            success=data.get("success", True),
            error_message=data.get("error_message"),
        )


@dataclass
class AuditConfig:
    """Audit logging configuration."""
    enabled: bool = True
    storage_path: Path = field(default_factory=lambda: Path("data/audit"))
    db_file: str = "audit.db"
    log_file: str = "audit.log"
    retention_days: int = 365
    log_to_file: bool = True
    log_to_db: bool = True
    log_to_console: bool = False
    include_details: bool = True
    mask_sensitive: bool = True
    sensitive_fields: List[str] = field(default_factory=lambda: [
        "password", "token", "secret", "api_key", "credit_card"
    ])


class AuditLogger:
    """
    Audit Logger.
    
    Records all significant system events for compliance and security.
    """
    
    def __init__(self, config: Optional[AuditConfig] = None):
        self.config = config or AuditConfig()
        
        # Ensure storage directory exists
        self.config.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Thread lock for database operations
        self._lock = threading.Lock()
        
        # Initialize database
        if self.config.log_to_db:
            self._init_db()
        
        # Initialize file logger
        if self.config.log_to_file:
            self._init_file_logger()
    
    def _init_db(self):
        """Initialize SQLite database."""
        db_path = self.config.storage_path / self.config.db_file
        
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_events (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    user_id TEXT,
                    username TEXT,
                    action TEXT NOT NULL,
                    resource TEXT,
                    resource_id TEXT,
                    details TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    success INTEGER,
                    error_message TEXT
                )
            """)
            
            # Create indexes
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_events(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_event_type ON audit_events(event_type)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_id ON audit_events(user_id)
            """)
            
            conn.commit()
    
    def _init_file_logger(self):
        """Initialize file logger."""
        log_path = self.config.storage_path / self.config.log_file
        
        self._file_handler = logging.FileHandler(log_path)
        self._file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(message)s')
        )
    
    def _generate_id(self) -> str:
        """Generate unique event ID."""
        import secrets
        return secrets.token_hex(16)
    
    def _mask_sensitive(self, data: Dict) -> Dict:
        """Mask sensitive fields in data."""
        if not self.config.mask_sensitive:
            return data
        
        masked = {}
        
        for key, value in data.items():
            key_lower = key.lower()
            
            if any(s in key_lower for s in self.config.sensitive_fields):
                masked[key] = "***MASKED***"
            elif isinstance(value, dict):
                masked[key] = self._mask_sensitive(value)
            else:
                masked[key] = value
        
        return masked
    
    def log(
        self,
        event_type: AuditEventType,
        action: str,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        resource: Optional[str] = None,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        severity: Optional[AuditSeverity] = None,
    ) -> AuditEvent:
        """
        Log an audit event.
        
        Args:
            event_type: Type of event
            action: Description of action taken
            user_id: User ID (if applicable)
            username: Username (if applicable)
            resource: Resource being accessed
            resource_id: Resource identifier
            details: Additional details
            ip_address: Client IP address
            user_agent: Client user agent
            success: Whether action was successful
            error_message: Error message if failed
            severity: Event severity
            
        Returns:
            Created audit event
        """
        if not self.config.enabled:
            return None
        
        # Determine severity
        if severity is None:
            if not success:
                severity = AuditSeverity.ERROR
            elif event_type in [
                AuditEventType.RISK_LIMIT_BREACHED,
                AuditEventType.LOGIN_FAILURE,
            ]:
                severity = AuditSeverity.WARNING
            else:
                severity = AuditSeverity.INFO
        
        # Mask sensitive data
        masked_details = self._mask_sensitive(details or {})
        
        event = AuditEvent(
            id=self._generate_id(),
            timestamp=datetime.now(),
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            username=username,
            action=action,
            resource=resource,
            resource_id=resource_id,
            details=masked_details if self.config.include_details else {},
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            error_message=error_message,
        )
        
        # Log to database
        if self.config.log_to_db:
            self._log_to_db(event)
        
        # Log to file
        if self.config.log_to_file:
            self._log_to_file(event)
        
        # Log to console
        if self.config.log_to_console:
            self._log_to_console(event)
        
        return event
    
    def _log_to_db(self, event: AuditEvent):
        """Log event to database."""
        db_path = self.config.storage_path / self.config.db_file
        
        with self._lock:
            try:
                with sqlite3.connect(str(db_path)) as conn:
                    conn.execute("""
                        INSERT INTO audit_events (
                            id, timestamp, event_type, severity, user_id, username,
                            action, resource, resource_id, details, ip_address,
                            user_agent, success, error_message
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        event.id,
                        event.timestamp.isoformat(),
                        event.event_type.value,
                        event.severity.value,
                        event.user_id,
                        event.username,
                        event.action,
                        event.resource,
                        event.resource_id,
                        json.dumps(event.details),
                        event.ip_address,
                        event.user_agent,
                        1 if event.success else 0,
                        event.error_message,
                    ))
                    conn.commit()
            except Exception as e:
                logger.error(f"Failed to log audit event to database: {e}")
    
    def _log_to_file(self, event: AuditEvent):
        """Log event to file."""
        log_line = (
            f"[{event.severity.value.upper()}] "
            f"{event.event_type.value} | "
            f"user={event.username or 'system'} | "
            f"action={event.action} | "
            f"resource={event.resource or 'N/A'} | "
            f"success={event.success}"
        )
        
        if event.error_message:
            log_line += f" | error={event.error_message}"
        
        try:
            self._file_handler.emit(
                logging.LogRecord(
                    name="audit",
                    level=logging.INFO,
                    pathname="",
                    lineno=0,
                    msg=log_line,
                    args=(),
                    exc_info=None,
                )
            )
        except Exception as e:
            logger.error(f"Failed to log audit event to file: {e}")
    
    def _log_to_console(self, event: AuditEvent):
        """Log event to console."""
        logger.info(
            f"AUDIT: {event.event_type.value} | "
            f"user={event.username or 'system'} | "
            f"action={event.action}"
        )
    
    def query(
        self,
        event_types: Optional[List[AuditEventType]] = None,
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        success_only: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AuditEvent]:
        """
        Query audit events.
        
        Args:
            event_types: Filter by event types
            user_id: Filter by user ID
            start_date: Filter by start date
            end_date: Filter by end date
            success_only: Filter by success status
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of matching audit events
        """
        db_path = self.config.storage_path / self.config.db_file
        
        query = "SELECT * FROM audit_events WHERE 1=1"
        params = []
        
        if event_types:
            placeholders = ",".join("?" * len(event_types))
            query += f" AND event_type IN ({placeholders})"
            params.extend(et.value for et in event_types)
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())
        
        if success_only is not None:
            query += " AND success = ?"
            params.append(1 if success_only else 0)
        
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        events = []
        
        try:
            with sqlite3.connect(str(db_path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, params)
                
                for row in cursor.fetchall():
                    events.append(AuditEvent(
                        id=row["id"],
                        timestamp=datetime.fromisoformat(row["timestamp"]),
                        event_type=AuditEventType(row["event_type"]),
                        severity=AuditSeverity(row["severity"]),
                        user_id=row["user_id"],
                        username=row["username"],
                        action=row["action"],
                        resource=row["resource"],
                        resource_id=row["resource_id"],
                        details=json.loads(row["details"]) if row["details"] else {},
                        ip_address=row["ip_address"],
                        user_agent=row["user_agent"],
                        success=bool(row["success"]),
                        error_message=row["error_message"],
                    ))
        except Exception as e:
            logger.error(f"Failed to query audit events: {e}")
        
        return events
    
    def get_user_activity(
        self,
        user_id: str,
        days: int = 30,
    ) -> Dict[str, Any]:
        """
        Get user activity summary.
        
        Args:
            user_id: User ID
            days: Number of days to analyze
            
        Returns:
            Activity summary
        """
        start_date = datetime.now() - timedelta(days=days)
        events = self.query(user_id=user_id, start_date=start_date, limit=1000)
        
        summary = {
            "total_events": len(events),
            "successful": sum(1 for e in events if e.success),
            "failed": sum(1 for e in events if not e.success),
            "by_type": {},
            "by_day": {},
            "last_activity": None,
        }
        
        for event in events:
            # By type
            type_key = event.event_type.value
            summary["by_type"][type_key] = summary["by_type"].get(type_key, 0) + 1
            
            # By day
            day_key = event.timestamp.strftime("%Y-%m-%d")
            summary["by_day"][day_key] = summary["by_day"].get(day_key, 0) + 1
            
            # Last activity
            if summary["last_activity"] is None or event.timestamp > summary["last_activity"]:
                summary["last_activity"] = event.timestamp
        
        return summary
    
    def get_security_events(
        self,
        days: int = 7,
    ) -> List[AuditEvent]:
        """
        Get security-related events.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            List of security events
        """
        security_types = [
            AuditEventType.LOGIN_FAILURE,
            AuditEventType.PASSWORD_CHANGE,
            AuditEventType.USER_ROLE_CHANGED,
            AuditEventType.RISK_LIMIT_OVERRIDE,
            AuditEventType.RISK_LIMIT_BREACHED,
        ]
        
        return self.query(
            event_types=security_types,
            start_date=datetime.now() - timedelta(days=days),
            limit=500,
        )
    
    def cleanup_old_events(self) -> int:
        """
        Remove events older than retention period.
        
        Returns:
            Number of events deleted
        """
        db_path = self.config.storage_path / self.config.db_file
        cutoff_date = datetime.now() - timedelta(days=self.config.retention_days)
        
        try:
            with sqlite3.connect(str(db_path)) as conn:
                cursor = conn.execute(
                    "DELETE FROM audit_events WHERE timestamp < ?",
                    (cutoff_date.isoformat(),)
                )
                deleted = cursor.rowcount
                conn.commit()
                
                logger.info(f"Cleaned up {deleted} old audit events")
                return deleted
        except Exception as e:
            logger.error(f"Failed to cleanup old audit events: {e}")
            return 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get audit log statistics."""
        db_path = self.config.storage_path / self.config.db_file
        
        stats = {
            "total_events": 0,
            "events_by_type": {},
            "events_by_severity": {},
            "success_rate": 0,
            "db_size_mb": 0,
        }
        
        try:
            with sqlite3.connect(str(db_path)) as conn:
                # Total events
                cursor = conn.execute("SELECT COUNT(*) FROM audit_events")
                stats["total_events"] = cursor.fetchone()[0]
                
                # By type
                cursor = conn.execute(
                    "SELECT event_type, COUNT(*) FROM audit_events GROUP BY event_type"
                )
                stats["events_by_type"] = dict(cursor.fetchall())
                
                # By severity
                cursor = conn.execute(
                    "SELECT severity, COUNT(*) FROM audit_events GROUP BY severity"
                )
                stats["events_by_severity"] = dict(cursor.fetchall())
                
                # Success rate
                cursor = conn.execute(
                    "SELECT AVG(success) FROM audit_events"
                )
                result = cursor.fetchone()[0]
                stats["success_rate"] = result if result else 0
            
            # Database size
            stats["db_size_mb"] = db_path.stat().st_size / (1024 * 1024)
        
        except Exception as e:
            logger.error(f"Failed to get audit statistics: {e}")
        
        return stats


# Convenience functions
def log_login(
    audit_logger: AuditLogger,
    username: str,
    success: bool,
    ip_address: Optional[str] = None,
    error: Optional[str] = None,
):
    """Log login attempt."""
    audit_logger.log(
        event_type=AuditEventType.LOGIN_SUCCESS if success else AuditEventType.LOGIN_FAILURE,
        action=f"User login attempt for {username}",
        username=username,
        ip_address=ip_address,
        success=success,
        error_message=error,
    )


def log_trade(
    audit_logger: AuditLogger,
    user_id: str,
    username: str,
    action: str,
    order_id: str,
    details: Dict[str, Any],
):
    """Log trading action."""
    event_types = {
        "create": AuditEventType.ORDER_CREATED,
        "modify": AuditEventType.ORDER_MODIFIED,
        "cancel": AuditEventType.ORDER_CANCELLED,
        "execute": AuditEventType.ORDER_EXECUTED,
    }
    
    audit_logger.log(
        event_type=event_types.get(action, AuditEventType.ORDER_CREATED),
        action=f"Order {action}: {order_id}",
        user_id=user_id,
        username=username,
        resource="order",
        resource_id=order_id,
        details=details,
    )
