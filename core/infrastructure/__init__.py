"""
Infrastructure Module
=====================
Production hardening, authentication, monitoring, and deployment support.

Phase 9 Implementation:
- Authentication & Authorization
- Role-Based Access Control (RBAC)
- Audit Logging
- Health Checks & Metrics
- Database Migrations
"""

from .audit import (
    AuditConfig,
    AuditEvent,
    AuditEventType,
    AuditLogger,
)
from .auth import (
    AccessToken,
    AuthConfig,
    AuthManager,
    Permission,
    Role,
    Session,
    User,
)
from .monitoring import (
    HealthChecker,
    HealthStatus,
    MetricsCollector,
    MonitoringConfig,
)
from .rbac import (
    PermissionSet,
    RBACManager,
    RoleDefinition,
)

__all__ = [
    # Auth
    "AuthManager",
    "User",
    "Role",
    "Permission",
    "AuthConfig",
    "Session",
    "AccessToken",
    # RBAC
    "RBACManager",
    "RoleDefinition",
    "PermissionSet",
    # Audit
    "AuditLogger",
    "AuditEvent",
    "AuditEventType",
    "AuditConfig",
    # Monitoring
    "HealthChecker",
    "HealthStatus",
    "MetricsCollector",
    "MonitoringConfig",
]
