"""
Alert History & Audit Log
=========================
Persistent storage and auditing for alerts.

Features:
- SQLite-based alert storage
- Full audit trail
- Query and filtering
- Statistics and analytics
"""

import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from .rules import AlertSeverity, AlertCategory, AlertTrigger

logger = logging.getLogger(__name__)


@dataclass
class AlertRecord:
    """A persistent alert record."""
    record_id: str
    trigger_id: str
    rule_id: str
    rule_name: str
    
    # Alert details
    severity: AlertSeverity
    category: AlertCategory
    title: str
    message: str
    
    # Timestamps
    created_at: datetime
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    escalated_at: Optional[datetime] = None
    
    # Status
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    resolved: bool = False
    resolved_by: Optional[str] = None
    escalated: bool = False
    
    # Notification info
    channels_notified: List[str] = field(default_factory=list)
    notification_errors: List[str] = field(default_factory=list)
    
    # Context
    context: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "record_id": self.record_id,
            "trigger_id": self.trigger_id,
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "severity": self.severity.value,
            "category": self.category.value,
            "title": self.title,
            "message": self.message,
            "created_at": self.created_at.isoformat(),
            "acknowledged": self.acknowledged,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "acknowledged_by": self.acknowledged_by,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "escalated": self.escalated,
            "channels_notified": self.channels_notified,
        }


class AlertHistory:
    """
    Persistent storage for alert history.
    
    Uses SQLite for efficient querying and storage.
    """
    
    def __init__(self, db_path: str = "data/alerts/alert_history.db"):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._lock = threading.Lock()
        self._init_database()
    
    def _init_database(self):
        """Initialize the database schema."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    record_id TEXT PRIMARY KEY,
                    trigger_id TEXT NOT NULL,
                    rule_id TEXT NOT NULL,
                    rule_name TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    category TEXT NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT,
                    created_at TIMESTAMP NOT NULL,
                    acknowledged BOOLEAN DEFAULT FALSE,
                    acknowledged_at TIMESTAMP,
                    acknowledged_by TEXT,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolved_at TIMESTAMP,
                    resolved_by TEXT,
                    escalated BOOLEAN DEFAULT FALSE,
                    escalated_at TIMESTAMP,
                    channels_notified TEXT,
                    notification_errors TEXT,
                    context TEXT,
                    notes TEXT
                )
            """)
            
            # Create indexes for common queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_created_at 
                ON alerts(created_at DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_severity 
                ON alerts(severity)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_category 
                ON alerts(category)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_rule_id 
                ON alerts(rule_id)
            """)
            
            conn.commit()
    
    def add(self, record: AlertRecord):
        """Add an alert record."""
        import json
        
        with self._lock:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute("""
                    INSERT INTO alerts (
                        record_id, trigger_id, rule_id, rule_name,
                        severity, category, title, message,
                        created_at, acknowledged, acknowledged_at, acknowledged_by,
                        resolved, resolved_at, resolved_by, escalated, escalated_at,
                        channels_notified, notification_errors, context, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.record_id,
                    record.trigger_id,
                    record.rule_id,
                    record.rule_name,
                    record.severity.value,
                    record.category.value,
                    record.title,
                    record.message,
                    record.created_at.isoformat(),
                    record.acknowledged,
                    record.acknowledged_at.isoformat() if record.acknowledged_at else None,
                    record.acknowledged_by,
                    record.resolved,
                    record.resolved_at.isoformat() if record.resolved_at else None,
                    record.resolved_by,
                    record.escalated,
                    record.escalated_at.isoformat() if record.escalated_at else None,
                    json.dumps(record.channels_notified),
                    json.dumps(record.notification_errors),
                    json.dumps(record.context),
                    record.notes,
                ))
                conn.commit()
    
    def update(self, record_id: str, updates: Dict[str, Any]):
        """Update an alert record."""
        allowed_fields = {
            "acknowledged", "acknowledged_at", "acknowledged_by",
            "resolved", "resolved_at", "resolved_by",
            "escalated", "escalated_at", "notes",
        }
        
        # Filter to allowed fields
        safe_updates = {k: v for k, v in updates.items() if k in allowed_fields}
        
        if not safe_updates:
            return
        
        # Build update query
        set_clauses = []
        values = []
        
        for field, value in safe_updates.items():
            set_clauses.append(f"{field} = ?")
            if isinstance(value, datetime):
                values.append(value.isoformat())
            else:
                values.append(value)
        
        values.append(record_id)
        
        with self._lock:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    f"UPDATE alerts SET {', '.join(set_clauses)} WHERE record_id = ?",
                    values
                )
                conn.commit()
    
    def get(self, record_id: str) -> Optional[AlertRecord]:
        """Get an alert record by ID."""
        with self._lock:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM alerts WHERE record_id = ?",
                    (record_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    return self._row_to_record(dict(row))
                return None
    
    def query(
        self,
        severity: Optional[AlertSeverity] = None,
        category: Optional[AlertCategory] = None,
        rule_id: Optional[str] = None,
        acknowledged: Optional[bool] = None,
        resolved: Optional[bool] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AlertRecord]:
        """Query alert records with filters."""
        conditions = []
        values = []
        
        if severity:
            conditions.append("severity = ?")
            values.append(severity.value)
        
        if category:
            conditions.append("category = ?")
            values.append(category.value)
        
        if rule_id:
            conditions.append("rule_id = ?")
            values.append(rule_id)
        
        if acknowledged is not None:
            conditions.append("acknowledged = ?")
            values.append(acknowledged)
        
        if resolved is not None:
            conditions.append("resolved = ?")
            values.append(resolved)
        
        if since:
            conditions.append("created_at >= ?")
            values.append(since.isoformat())
        
        if until:
            conditions.append("created_at <= ?")
            values.append(until.isoformat())
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        with self._lock:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    f"""
                    SELECT * FROM alerts 
                    WHERE {where_clause}
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    values + [limit, offset]
                )
                
                return [self._row_to_record(dict(row)) for row in cursor.fetchall()]
    
    def count(
        self,
        severity: Optional[AlertSeverity] = None,
        category: Optional[AlertCategory] = None,
        since: Optional[datetime] = None,
    ) -> int:
        """Count alerts matching filters."""
        conditions = []
        values = []
        
        if severity:
            conditions.append("severity = ?")
            values.append(severity.value)
        
        if category:
            conditions.append("category = ?")
            values.append(category.value)
        
        if since:
            conditions.append("created_at >= ?")
            values.append(since.isoformat())
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        with self._lock:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.execute(
                    f"SELECT COUNT(*) FROM alerts WHERE {where_clause}",
                    values
                )
                return cursor.fetchone()[0]
    
    def get_statistics(
        self,
        since: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Get alert statistics."""
        since = since or (datetime.now() - timedelta(days=30))
        
        with self._lock:
            with sqlite3.connect(self._db_path) as conn:
                # Total count
                total = conn.execute(
                    "SELECT COUNT(*) FROM alerts WHERE created_at >= ?",
                    (since.isoformat(),)
                ).fetchone()[0]
                
                # By severity
                severity_counts = {}
                cursor = conn.execute("""
                    SELECT severity, COUNT(*) 
                    FROM alerts 
                    WHERE created_at >= ?
                    GROUP BY severity
                """, (since.isoformat(),))
                for row in cursor:
                    severity_counts[row[0]] = row[1]
                
                # By category
                category_counts = {}
                cursor = conn.execute("""
                    SELECT category, COUNT(*) 
                    FROM alerts 
                    WHERE created_at >= ?
                    GROUP BY category
                """, (since.isoformat(),))
                for row in cursor:
                    category_counts[row[0]] = row[1]
                
                # Acknowledgment stats
                acked = conn.execute(
                    "SELECT COUNT(*) FROM alerts WHERE created_at >= ? AND acknowledged = TRUE",
                    (since.isoformat(),)
                ).fetchone()[0]
                
                # Resolution stats
                resolved = conn.execute(
                    "SELECT COUNT(*) FROM alerts WHERE created_at >= ? AND resolved = TRUE",
                    (since.isoformat(),)
                ).fetchone()[0]
                
                # Average response time
                cursor = conn.execute("""
                    SELECT AVG(
                        (julianday(acknowledged_at) - julianday(created_at)) * 24 * 60
                    )
                    FROM alerts 
                    WHERE created_at >= ? AND acknowledged_at IS NOT NULL
                """, (since.isoformat(),))
                avg_response = cursor.fetchone()[0] or 0
                
                return {
                    "total": total,
                    "by_severity": severity_counts,
                    "by_category": category_counts,
                    "acknowledged": acked,
                    "resolved": resolved,
                    "acknowledgment_rate": acked / total if total > 0 else 0,
                    "resolution_rate": resolved / total if total > 0 else 0,
                    "avg_response_minutes": round(avg_response, 1),
                    "period_start": since.isoformat(),
                    "period_end": datetime.now().isoformat(),
                }
    
    def cleanup(self, days: int = 30):
        """Remove old alert records."""
        cutoff = datetime.now() - timedelta(days=days)
        
        with self._lock:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM alerts WHERE created_at < ? AND resolved = TRUE",
                    (cutoff.isoformat(),)
                )
                deleted = cursor.rowcount
                conn.commit()
                
                if deleted > 0:
                    logger.info(f"Cleaned up {deleted} old alert records")
    
    def _row_to_record(self, row: Dict) -> AlertRecord:
        """Convert database row to AlertRecord."""
        import json
        
        def parse_datetime(val):
            if val:
                return datetime.fromisoformat(val)
            return None
        
        def parse_json(val):
            if val:
                return json.loads(val)
            return []
        
        return AlertRecord(
            record_id=row["record_id"],
            trigger_id=row["trigger_id"],
            rule_id=row["rule_id"],
            rule_name=row["rule_name"],
            severity=AlertSeverity(row["severity"]),
            category=AlertCategory(row["category"]),
            title=row["title"],
            message=row["message"] or "",
            created_at=parse_datetime(row["created_at"]),
            acknowledged=bool(row["acknowledged"]),
            acknowledged_at=parse_datetime(row.get("acknowledged_at")),
            acknowledged_by=row.get("acknowledged_by"),
            resolved=bool(row.get("resolved", False)),
            resolved_at=parse_datetime(row.get("resolved_at")),
            resolved_by=row.get("resolved_by"),
            escalated=bool(row.get("escalated", False)),
            escalated_at=parse_datetime(row.get("escalated_at")),
            channels_notified=parse_json(row.get("channels_notified")),
            notification_errors=parse_json(row.get("notification_errors")),
            context=parse_json(row.get("context")) if row.get("context") else {},
            notes=row.get("notes", ""),
        )


class AlertAuditLog:
    """
    Audit log for all alert-related actions.
    
    Records all changes and actions for compliance and debugging.
    """
    
    def __init__(self, log_path: str = "data/alerts/audit.log"):
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._lock = threading.Lock()
    
    def log(
        self,
        action: str,
        alert_id: Optional[str] = None,
        rule_id: Optional[str] = None,
        user: str = "system",
        details: Optional[Dict] = None,
    ):
        """Log an audit entry."""
        import json
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "alert_id": alert_id,
            "rule_id": rule_id,
            "user": user,
            "details": details or {},
        }
        
        with self._lock:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
    
    def log_alert_created(self, trigger: AlertTrigger, rule_name: str):
        """Log alert creation."""
        self.log(
            action="ALERT_CREATED",
            alert_id=trigger.trigger_id,
            rule_id=trigger.rule_id,
            details={
                "rule_name": rule_name,
                "severity": trigger.severity.value,
                "category": trigger.category.value,
                "title": trigger.title,
            }
        )
    
    def log_alert_acknowledged(
        self,
        alert_id: str,
        user: str,
        note: Optional[str] = None,
    ):
        """Log alert acknowledgment."""
        self.log(
            action="ALERT_ACKNOWLEDGED",
            alert_id=alert_id,
            user=user,
            details={"note": note} if note else None,
        )
    
    def log_alert_resolved(
        self,
        alert_id: str,
        user: str,
        note: Optional[str] = None,
    ):
        """Log alert resolution."""
        self.log(
            action="ALERT_RESOLVED",
            alert_id=alert_id,
            user=user,
            details={"note": note} if note else None,
        )
    
    def log_alert_escalated(self, alert_id: str, from_severity: str, to_severity: str):
        """Log alert escalation."""
        self.log(
            action="ALERT_ESCALATED",
            alert_id=alert_id,
            details={
                "from_severity": from_severity,
                "to_severity": to_severity,
            }
        )
    
    def log_notification_sent(
        self,
        alert_id: str,
        channel: str,
        success: bool,
        error: Optional[str] = None,
    ):
        """Log notification send attempt."""
        self.log(
            action="NOTIFICATION_SENT",
            alert_id=alert_id,
            details={
                "channel": channel,
                "success": success,
                "error": error,
            }
        )
    
    def log_rule_created(self, rule_id: str, rule_name: str, user: str = "system"):
        """Log rule creation."""
        self.log(
            action="RULE_CREATED",
            rule_id=rule_id,
            user=user,
            details={"rule_name": rule_name},
        )
    
    def log_rule_updated(
        self,
        rule_id: str,
        changes: Dict[str, Any],
        user: str = "system",
    ):
        """Log rule update."""
        self.log(
            action="RULE_UPDATED",
            rule_id=rule_id,
            user=user,
            details={"changes": changes},
        )
    
    def log_rule_deleted(self, rule_id: str, user: str = "system"):
        """Log rule deletion."""
        self.log(
            action="RULE_DELETED",
            rule_id=rule_id,
            user=user,
        )
    
    def get_entries(
        self,
        since: Optional[datetime] = None,
        action: Optional[str] = None,
        alert_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """Get audit log entries."""
        import json
        
        entries = []
        
        with self._lock:
            if not self._log_path.exists():
                return []
            
            with open(self._log_path) as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        
                        # Filter
                        if since:
                            entry_time = datetime.fromisoformat(entry["timestamp"])
                            if entry_time < since:
                                continue
                        
                        if action and entry.get("action") != action:
                            continue
                        
                        if alert_id and entry.get("alert_id") != alert_id:
                            continue
                        
                        entries.append(entry)
                        
                    except json.JSONDecodeError:
                        continue
        
        # Return most recent first, with limit
        return list(reversed(entries[-limit:]))
