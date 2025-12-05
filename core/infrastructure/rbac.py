"""
Role-Based Access Control (RBAC)
================================
Role and permission management for authorization.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .auth import Role, Permission, User

logger = logging.getLogger(__name__)


@dataclass
class PermissionSet:
    """Set of permissions for a role."""
    permissions: Set[Permission] = field(default_factory=set)
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if permission is granted."""
        return permission in self.permissions
    
    def grant(self, permission: Permission):
        """Grant a permission."""
        self.permissions.add(permission)
    
    def revoke(self, permission: Permission):
        """Revoke a permission."""
        self.permissions.discard(permission)
    
    def to_list(self) -> List[str]:
        """Convert to list of permission names."""
        return [p.value for p in self.permissions]


@dataclass
class RoleDefinition:
    """Definition of a role with its permissions."""
    role: Role
    permissions: PermissionSet
    description: str = ""
    is_system_role: bool = True
    
    def to_dict(self) -> Dict:
        return {
            "role": self.role.value,
            "permissions": self.permissions.to_list(),
            "description": self.description,
            "is_system_role": self.is_system_role,
        }


class RBACManager:
    """
    Role-Based Access Control Manager.
    
    Manages role definitions and permission checking.
    """
    
    # Default role permissions
    DEFAULT_PERMISSIONS = {
        Role.ADMIN: {
            Permission.VIEW_DASHBOARD,
            Permission.VIEW_ANALYTICS,
            Permission.VIEW_SIGNALS,
            Permission.VIEW_RISK,
            Permission.VIEW_RESEARCH,
            Permission.EXECUTE_TRADES,
            Permission.MANAGE_ORDERS,
            Permission.VIEW_BLOTTER,
            Permission.MANAGE_POSITIONS,
            Permission.SET_RISK_LIMITS,
            Permission.OVERRIDE_LIMITS,
            Permission.MANAGE_USERS,
            Permission.MANAGE_ROLES,
            Permission.VIEW_AUDIT_LOGS,
            Permission.MANAGE_ALERTS,
            Permission.MANAGE_SYSTEM,
            Permission.TRAIN_MODELS,
            Permission.MANAGE_RESEARCH,
            Permission.RUN_BACKTEST,
            Permission.MANAGE_STRATEGIES,
        },
        Role.TRADER: {
            Permission.VIEW_DASHBOARD,
            Permission.VIEW_ANALYTICS,
            Permission.VIEW_SIGNALS,
            Permission.VIEW_RISK,
            Permission.EXECUTE_TRADES,
            Permission.MANAGE_ORDERS,
            Permission.VIEW_BLOTTER,
            Permission.MANAGE_POSITIONS,
            Permission.RUN_BACKTEST,
        },
        Role.ANALYST: {
            Permission.VIEW_DASHBOARD,
            Permission.VIEW_ANALYTICS,
            Permission.VIEW_SIGNALS,
            Permission.VIEW_RISK,
            Permission.VIEW_RESEARCH,
            Permission.VIEW_BLOTTER,
            Permission.TRAIN_MODELS,
            Permission.MANAGE_RESEARCH,
            Permission.RUN_BACKTEST,
            Permission.MANAGE_STRATEGIES,
        },
        Role.RISK_MANAGER: {
            Permission.VIEW_DASHBOARD,
            Permission.VIEW_ANALYTICS,
            Permission.VIEW_SIGNALS,
            Permission.VIEW_RISK,
            Permission.VIEW_BLOTTER,
            Permission.SET_RISK_LIMITS,
            Permission.OVERRIDE_LIMITS,
            Permission.MANAGE_ALERTS,
            Permission.VIEW_AUDIT_LOGS,
        },
        Role.VIEWER: {
            Permission.VIEW_DASHBOARD,
            Permission.VIEW_ANALYTICS,
            Permission.VIEW_SIGNALS,
            Permission.VIEW_RISK,
        },
    }
    
    def __init__(self):
        self._role_definitions: Dict[Role, RoleDefinition] = {}
        self._initialize_default_roles()
    
    def _initialize_default_roles(self):
        """Initialize default role definitions."""
        for role, permissions in self.DEFAULT_PERMISSIONS.items():
            self._role_definitions[role] = RoleDefinition(
                role=role,
                permissions=PermissionSet(permissions),
                description=self._get_role_description(role),
                is_system_role=True,
            )
    
    def _get_role_description(self, role: Role) -> str:
        """Get description for a role."""
        descriptions = {
            Role.ADMIN: "Full system access with all permissions",
            Role.TRADER: "Trading and order management capabilities",
            Role.ANALYST: "Research, analytics, and model development",
            Role.RISK_MANAGER: "Risk monitoring and limit management",
            Role.VIEWER: "Read-only access to dashboards",
        }
        return descriptions.get(role, "")
    
    def get_role_permissions(self, role: Role) -> PermissionSet:
        """
        Get permissions for a role.
        
        Args:
            role: Role to get permissions for
            
        Returns:
            Permission set for the role
        """
        definition = self._role_definitions.get(role)
        
        if definition:
            return definition.permissions
        
        return PermissionSet()
    
    def get_user_permissions(self, user: User) -> PermissionSet:
        """
        Get all permissions for a user based on their roles.
        
        Args:
            user: User to get permissions for
            
        Returns:
            Combined permission set from all user roles
        """
        all_permissions: Set[Permission] = set()
        
        for role in user.roles:
            role_perms = self.get_role_permissions(role)
            all_permissions.update(role_perms.permissions)
        
        return PermissionSet(all_permissions)
    
    def check_permission(self, user: User, permission: Permission) -> bool:
        """
        Check if user has a specific permission.
        
        Args:
            user: User to check
            permission: Permission to check
            
        Returns:
            True if user has the permission
        """
        user_permissions = self.get_user_permissions(user)
        return user_permissions.has_permission(permission)
    
    def check_permissions(self, user: User, permissions: List[Permission]) -> bool:
        """
        Check if user has all specified permissions.
        
        Args:
            user: User to check
            permissions: List of permissions to check
            
        Returns:
            True if user has all permissions
        """
        user_permissions = self.get_user_permissions(user)
        return all(user_permissions.has_permission(p) for p in permissions)
    
    def check_any_permission(self, user: User, permissions: List[Permission]) -> bool:
        """
        Check if user has any of the specified permissions.
        
        Args:
            user: User to check
            permissions: List of permissions to check
            
        Returns:
            True if user has at least one permission
        """
        user_permissions = self.get_user_permissions(user)
        return any(user_permissions.has_permission(p) for p in permissions)
    
    def get_role_definition(self, role: Role) -> Optional[RoleDefinition]:
        """Get role definition."""
        return self._role_definitions.get(role)
    
    def list_roles(self) -> List[RoleDefinition]:
        """List all role definitions."""
        return list(self._role_definitions.values())
    
    def list_permissions(self) -> List[Permission]:
        """List all available permissions."""
        return list(Permission)
    
    def update_role_permissions(
        self,
        role: Role,
        permissions: Set[Permission],
    ) -> bool:
        """
        Update permissions for a role.
        
        Args:
            role: Role to update
            permissions: New permission set
            
        Returns:
            True if updated successfully
        """
        definition = self._role_definitions.get(role)
        
        if not definition:
            return False
        
        if definition.is_system_role and role == Role.ADMIN:
            logger.warning("Cannot modify admin role permissions")
            return False
        
        definition.permissions = PermissionSet(permissions)
        logger.info(f"Updated permissions for role: {role.value}")
        return True
    
    def get_required_permissions(self, resource: str) -> List[Permission]:
        """
        Get required permissions for a resource/page.
        
        Args:
            resource: Resource identifier
            
        Returns:
            List of required permissions
        """
        resource_permissions = {
            "dashboard": [Permission.VIEW_DASHBOARD],
            "analytics": [Permission.VIEW_ANALYTICS],
            "signals": [Permission.VIEW_SIGNALS],
            "risk": [Permission.VIEW_RISK],
            "research": [Permission.VIEW_RESEARCH],
            "trading": [Permission.EXECUTE_TRADES],
            "blotter": [Permission.VIEW_BLOTTER],
            "positions": [Permission.MANAGE_POSITIONS],
            "users": [Permission.MANAGE_USERS],
            "audit": [Permission.VIEW_AUDIT_LOGS],
            "alerts": [Permission.MANAGE_ALERTS],
            "backtest": [Permission.RUN_BACKTEST],
            "ml": [Permission.TRAIN_MODELS],
        }
        
        return resource_permissions.get(resource.lower(), [Permission.VIEW_DASHBOARD])
    
    def can_access_resource(self, user: User, resource: str) -> bool:
        """
        Check if user can access a resource.
        
        Args:
            user: User to check
            resource: Resource identifier
            
        Returns:
            True if user can access the resource
        """
        required = self.get_required_permissions(resource)
        return self.check_any_permission(user, required)


def require_permission(permission: Permission):
    """
    Decorator to require a permission for a function.
    
    Usage:
        @require_permission(Permission.EXECUTE_TRADES)
        def execute_trade(user, ...):
            ...
    """
    def decorator(func):
        def wrapper(user: User, *args, **kwargs):
            rbac = RBACManager()
            
            if not rbac.check_permission(user, permission):
                raise PermissionError(
                    f"User {user.username} does not have permission: {permission.value}"
                )
            
            return func(user, *args, **kwargs)
        
        return wrapper
    return decorator


def require_role(role: Role):
    """
    Decorator to require a role for a function.
    
    Usage:
        @require_role(Role.ADMIN)
        def admin_function(user, ...):
            ...
    """
    def decorator(func):
        def wrapper(user: User, *args, **kwargs):
            if role not in user.roles:
                raise PermissionError(
                    f"User {user.username} does not have role: {role.value}"
                )
            
            return func(user, *args, **kwargs)
        
        return wrapper
    return decorator
