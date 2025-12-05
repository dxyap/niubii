"""
Tests for Infrastructure Module
===============================
Tests for authentication, RBAC, audit logging, and monitoring.
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil

from core.infrastructure import (
    AuthManager,
    User,
    Role,
    Permission,
    AuthConfig,
    Session,
    RBACManager,
    PermissionSet,
    AuditLogger,
    AuditEvent,
    AuditEventType,
    AuditConfig,
    HealthChecker,
    HealthStatus,
    MetricsCollector,
    MonitoringConfig,
)


class TestAuthManager:
    """Tests for authentication manager."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp = tempfile.mkdtemp()
        yield Path(temp)
        shutil.rmtree(temp)
    
    @pytest.fixture
    def auth_manager(self, temp_dir):
        """Create auth manager for tests."""
        config = AuthConfig(storage_path=temp_dir)
        return AuthManager(config)
    
    def test_create_user(self, auth_manager):
        """Test user creation."""
        user = auth_manager.create_user(
            username="testuser",
            email="test@example.com",
            password="SecurePass123!",
            roles=[Role.TRADER],
        )
        
        assert user is not None
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert Role.TRADER in user.roles
    
    def test_duplicate_username(self, auth_manager):
        """Test duplicate username prevention."""
        auth_manager.create_user(
            username="testuser",
            email="test1@example.com",
            password="SecurePass123!",
        )
        
        with pytest.raises(ValueError, match="already exists"):
            auth_manager.create_user(
                username="testuser",
                email="test2@example.com",
                password="SecurePass123!",
            )
    
    def test_password_validation(self, auth_manager):
        """Test password strength validation."""
        with pytest.raises(ValueError, match="at least"):
            auth_manager.create_user(
                username="testuser",
                email="test@example.com",
                password="weak",  # Too short
            )
    
    def test_authenticate_success(self, auth_manager):
        """Test successful authentication."""
        auth_manager.create_user(
            username="testuser",
            email="test@example.com",
            password="SecurePass123!",
        )
        
        session = auth_manager.authenticate("testuser", "SecurePass123!")
        
        assert session is not None
        assert session.is_active is True
    
    def test_authenticate_failure(self, auth_manager):
        """Test failed authentication."""
        auth_manager.create_user(
            username="testuser",
            email="test@example.com",
            password="SecurePass123!",
        )
        
        session = auth_manager.authenticate("testuser", "WrongPassword!")
        
        assert session is None
    
    def test_session_validation(self, auth_manager):
        """Test session validation."""
        auth_manager.create_user(
            username="testuser",
            email="test@example.com",
            password="SecurePass123!",
        )
        
        session = auth_manager.authenticate("testuser", "SecurePass123!")
        user = auth_manager.validate_session(session.id)
        
        assert user is not None
        assert user.username == "testuser"
    
    def test_logout(self, auth_manager):
        """Test logout."""
        auth_manager.create_user(
            username="testuser",
            email="test@example.com",
            password="SecurePass123!",
        )
        
        session = auth_manager.authenticate("testuser", "SecurePass123!")
        result = auth_manager.logout(session.id)
        
        assert result is True
        
        # Session should no longer be valid
        user = auth_manager.validate_session(session.id)
        assert user is None
    
    def test_change_password(self, auth_manager):
        """Test password change."""
        user = auth_manager.create_user(
            username="testuser",
            email="test@example.com",
            password="SecurePass123!",
        )
        
        result = auth_manager.change_password(
            user.id,
            "SecurePass123!",
            "NewSecurePass456!",
        )
        
        assert result is True
        
        # Old password should fail
        session = auth_manager.authenticate("testuser", "SecurePass123!")
        assert session is None
        
        # New password should work
        session = auth_manager.authenticate("testuser", "NewSecurePass456!")
        assert session is not None
    
    def test_account_lockout(self, auth_manager):
        """Test account lockout after failed attempts."""
        auth_manager.config.lockout_attempts = 3
        
        auth_manager.create_user(
            username="testuser",
            email="test@example.com",
            password="SecurePass123!",
        )
        
        # Fail multiple times
        for _ in range(3):
            auth_manager.authenticate("testuser", "WrongPassword!")
        
        # Account should be locked
        session = auth_manager.authenticate("testuser", "SecurePass123!")
        assert session is None
    
    def test_access_token_creation(self, auth_manager):
        """Test access token creation."""
        user = auth_manager.create_user(
            username="testuser",
            email="test@example.com",
            password="SecurePass123!",
        )
        
        token = auth_manager.create_access_token(user, scopes=["read", "write"])
        
        assert token is not None
        assert token.user_id == user.id
        assert "read" in token.scopes


class TestRBACManager:
    """Tests for RBAC manager."""
    
    @pytest.fixture
    def rbac(self):
        """Create RBAC manager for tests."""
        return RBACManager()
    
    @pytest.fixture
    def test_user(self):
        """Create test user."""
        return User(
            id="test123",
            username="testuser",
            email="test@example.com",
            password_hash="hash",
            roles=[Role.TRADER],
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
    
    def test_get_role_permissions(self, rbac):
        """Test getting permissions for a role."""
        perms = rbac.get_role_permissions(Role.ADMIN)
        
        assert len(perms.permissions) > 0
        assert Permission.MANAGE_USERS in perms.permissions
    
    def test_get_user_permissions(self, rbac, test_user):
        """Test getting all permissions for a user."""
        perms = rbac.get_user_permissions(test_user)
        
        assert len(perms.permissions) > 0
    
    def test_check_permission_granted(self, rbac, test_user):
        """Test checking a granted permission."""
        result = rbac.check_permission(test_user, Permission.VIEW_DASHBOARD)
        
        assert result is True
    
    def test_check_permission_denied(self, rbac, test_user):
        """Test checking a denied permission."""
        result = rbac.check_permission(test_user, Permission.MANAGE_USERS)
        
        assert result is False
    
    def test_admin_has_all_permissions(self, rbac):
        """Test that admin has all permissions."""
        admin_user = User(
            id="admin123",
            username="admin",
            email="admin@example.com",
            password_hash="hash",
            roles=[Role.ADMIN],
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        
        # Check all permissions
        for perm in Permission:
            assert rbac.check_permission(admin_user, perm) is True
    
    def test_can_access_resource(self, rbac, test_user):
        """Test resource access check."""
        assert rbac.can_access_resource(test_user, "dashboard") is True
        assert rbac.can_access_resource(test_user, "users") is False


class TestAuditLogger:
    """Tests for audit logger."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp = tempfile.mkdtemp()
        yield Path(temp)
        shutil.rmtree(temp)
    
    @pytest.fixture
    def audit_logger(self, temp_dir):
        """Create audit logger for tests."""
        config = AuditConfig(
            storage_path=temp_dir,
            log_to_console=False,
        )
        return AuditLogger(config)
    
    def test_log_event(self, audit_logger):
        """Test logging an event."""
        event = audit_logger.log(
            event_type=AuditEventType.LOGIN_SUCCESS,
            action="User login",
            username="testuser",
            ip_address="192.168.1.1",
        )
        
        assert event is not None
        assert event.event_type == AuditEventType.LOGIN_SUCCESS
    
    def test_query_events(self, audit_logger):
        """Test querying events."""
        # Log multiple events
        for i in range(5):
            audit_logger.log(
                event_type=AuditEventType.DATA_ACCESSED,
                action=f"Access {i}",
                username="testuser",
            )
        
        events = audit_logger.query(limit=10)
        
        assert len(events) == 5
    
    def test_query_by_type(self, audit_logger):
        """Test querying by event type."""
        audit_logger.log(
            event_type=AuditEventType.LOGIN_SUCCESS,
            action="Login",
            username="user1",
        )
        audit_logger.log(
            event_type=AuditEventType.ORDER_CREATED,
            action="Create order",
            username="user2",
        )
        
        events = audit_logger.query(
            event_types=[AuditEventType.LOGIN_SUCCESS],
        )
        
        assert len(events) == 1
        assert events[0].event_type == AuditEventType.LOGIN_SUCCESS
    
    def test_sensitive_data_masking(self, audit_logger):
        """Test sensitive data masking."""
        event = audit_logger.log(
            event_type=AuditEventType.LOGIN_SUCCESS,
            action="Login",
            details={"password": "secret123", "username": "testuser"},
        )
        
        assert event.details.get("password") == "***MASKED***"
        assert event.details.get("username") == "testuser"
    
    def test_user_activity_summary(self, audit_logger):
        """Test user activity summary."""
        for i in range(10):
            audit_logger.log(
                event_type=AuditEventType.DATA_ACCESSED,
                action=f"Access {i}",
                user_id="user123",
                username="testuser",
            )
        
        summary = audit_logger.get_user_activity("user123", days=7)
        
        assert summary["total_events"] == 10


class TestHealthChecker:
    """Tests for health checker."""
    
    @pytest.fixture
    def health_checker(self):
        """Create health checker for tests."""
        return HealthChecker()
    
    def test_checker_creation(self, health_checker):
        """Test checker initialization."""
        assert health_checker is not None
    
    def test_register_check(self, health_checker):
        """Test registering a health check."""
        health_checker.register(
            name="test_check",
            check_fn=lambda: True,
            description="Test check",
        )
        
        assert "test_check" in health_checker._checks
    
    def test_run_check(self, health_checker):
        """Test running a health check."""
        health_checker.register(
            name="always_healthy",
            check_fn=lambda: True,
        )
        
        result = health_checker.check("always_healthy")
        
        assert result.status == HealthStatus.HEALTHY
    
    def test_failing_check(self, health_checker):
        """Test a failing health check."""
        health_checker.register(
            name="always_unhealthy",
            check_fn=lambda: False,
            critical=True,
        )
        
        result = health_checker.check("always_unhealthy")
        
        assert result.status == HealthStatus.UNHEALTHY
    
    def test_overall_status(self, health_checker):
        """Test overall health status."""
        health_checker.register("check1", lambda: True)
        health_checker.register("check2", lambda: True)
        
        status = health_checker.get_overall_status()
        
        assert status == HealthStatus.HEALTHY
    
    def test_health_summary(self, health_checker):
        """Test health summary."""
        health_checker.register("check1", lambda: True)
        health_checker.register("check2", lambda: True)
        
        summary = health_checker.get_health_summary()
        
        assert "status" in summary
        assert "checks" in summary
        assert "summary" in summary


class TestMetricsCollector:
    """Tests for metrics collector."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp = tempfile.mkdtemp()
        yield Path(temp)
        shutil.rmtree(temp)
    
    @pytest.fixture
    def collector(self, temp_dir):
        """Create metrics collector for tests."""
        config = MonitoringConfig(storage_path=temp_dir)
        return MetricsCollector(config)
    
    def test_collector_creation(self, collector):
        """Test collector initialization."""
        assert collector is not None
    
    def test_increment_counter(self, collector):
        """Test incrementing a counter."""
        collector.increment("app_requests_total")
        collector.increment("app_requests_total")
        
        metric = collector.get_metric("app_requests_total")
        
        assert metric.value == 2
    
    def test_set_gauge(self, collector):
        """Test setting a gauge."""
        collector.set("app_active_sessions", 42)
        
        metric = collector.get_metric("app_active_sessions")
        
        assert metric.value == 42
    
    def test_observe_histogram(self, collector):
        """Test observing histogram values."""
        collector.observe("app_request_duration_seconds", 0.05)
        collector.observe("app_request_duration_seconds", 0.15)
        collector.observe("app_request_duration_seconds", 0.25)
        
        metric = collector.get_metric("app_request_duration_seconds")
        
        assert metric.count == 3
        assert metric.sum == pytest.approx(0.45, rel=0.01)
    
    def test_prometheus_output(self, collector):
        """Test Prometheus format output."""
        collector.increment("app_requests_total")
        collector.set("app_active_sessions", 10)
        
        output = collector.get_prometheus_output()
        
        assert "app_requests_total" in output
        assert "app_active_sessions" in output
