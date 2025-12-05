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

from .auth import (
    AuthManager,
    User,
    Role,
    Permission,
    AuthConfig,
    Session,
    AccessToken,
)

from .rbac import (
    RBACManager,
    RoleDefinition,
    PermissionSet,
)

from .audit import (
    AuditLogger,
    AuditEvent,
    AuditEventType,
    AuditConfig,
)

from .monitoring import (
    HealthChecker,
    HealthStatus,
    MetricsCollector,
    MonitoringConfig,
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
