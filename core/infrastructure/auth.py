"""
Authentication Module
=====================
User authentication, session management, and token handling.
"""

import hashlib
import json
import logging
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class AuthProvider(Enum):
    """Authentication provider types."""
    LOCAL = "local"
    OAUTH2 = "oauth2"
    LDAP = "ldap"
    SSO = "sso"


class Role(Enum):
    """User roles."""
    ADMIN = "admin"
    TRADER = "trader"
    ANALYST = "analyst"
    RISK_MANAGER = "risk_manager"
    VIEWER = "viewer"


class Permission(Enum):
    """System permissions."""
    # Dashboard access
    VIEW_DASHBOARD = "view_dashboard"
    VIEW_ANALYTICS = "view_analytics"
    VIEW_SIGNALS = "view_signals"
    VIEW_RISK = "view_risk"
    VIEW_RESEARCH = "view_research"

    # Trading permissions
    EXECUTE_TRADES = "execute_trades"
    MANAGE_ORDERS = "manage_orders"
    VIEW_BLOTTER = "view_blotter"
    MANAGE_POSITIONS = "manage_positions"

    # Risk management
    SET_RISK_LIMITS = "set_risk_limits"
    OVERRIDE_LIMITS = "override_limits"

    # Administration
    MANAGE_USERS = "manage_users"
    MANAGE_ROLES = "manage_roles"
    VIEW_AUDIT_LOGS = "view_audit_logs"
    MANAGE_ALERTS = "manage_alerts"
    MANAGE_SYSTEM = "manage_system"

    # ML/Research
    TRAIN_MODELS = "train_models"
    MANAGE_RESEARCH = "manage_research"

    # Backtest
    RUN_BACKTEST = "run_backtest"
    MANAGE_STRATEGIES = "manage_strategies"


@dataclass
class AuthConfig:
    """Authentication configuration."""
    provider: AuthProvider = AuthProvider.LOCAL
    session_duration_hours: int = 8
    token_expiry_hours: int = 24
    refresh_token_days: int = 7
    max_sessions_per_user: int = 5
    password_min_length: int = 8
    require_special_chars: bool = True
    require_numbers: bool = True
    lockout_attempts: int = 5
    lockout_duration_minutes: int = 30
    storage_path: Path = field(default_factory=lambda: Path("data/auth"))
    secret_key: str | None = None

    # OAuth2 settings
    oauth2_client_id: str | None = None
    oauth2_client_secret: str | None = None
    oauth2_authorize_url: str | None = None
    oauth2_token_url: str | None = None

    # LDAP settings
    ldap_server: str | None = None
    ldap_base_dn: str | None = None


@dataclass
class User:
    """User account."""
    id: str
    username: str
    email: str
    password_hash: str
    roles: list[Role]
    created_at: datetime
    updated_at: datetime
    last_login: datetime | None = None
    is_active: bool = True
    is_locked: bool = False
    failed_attempts: int = 0
    locked_until: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "password_hash": self.password_hash,
            "roles": [r.value for r in self.roles],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "is_active": self.is_active,
            "is_locked": self.is_locked,
            "failed_attempts": self.failed_attempts,
            "locked_until": self.locked_until.isoformat() if self.locked_until else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "User":
        return cls(
            id=data["id"],
            username=data["username"],
            email=data["email"],
            password_hash=data["password_hash"],
            roles=[Role(r) for r in data.get("roles", [])],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            last_login=datetime.fromisoformat(data["last_login"]) if data.get("last_login") else None,
            is_active=data.get("is_active", True),
            is_locked=data.get("is_locked", False),
            failed_attempts=data.get("failed_attempts", 0),
            locked_until=datetime.fromisoformat(data["locked_until"]) if data.get("locked_until") else None,
            metadata=data.get("metadata", {}),
        )


@dataclass
class Session:
    """User session."""
    id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    ip_address: str | None = None
    user_agent: str | None = None
    is_active: bool = True

    def is_expired(self) -> bool:
        return datetime.now() > self.expires_at

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "is_active": self.is_active,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        return cls(
            id=data["id"],
            user_id=data["user_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]),
            ip_address=data.get("ip_address"),
            user_agent=data.get("user_agent"),
            is_active=data.get("is_active", True),
        )


@dataclass
class AccessToken:
    """Access token for API authentication."""
    token: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    scopes: list[str] = field(default_factory=list)
    refresh_token: str | None = None

    def is_expired(self) -> bool:
        return datetime.now() > self.expires_at

    def to_dict(self) -> dict:
        return {
            "token": self.token,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "scopes": self.scopes,
            "refresh_token": self.refresh_token,
        }


class AuthManager:
    """
    Authentication Manager.

    Handles user authentication, session management, and access tokens.
    """

    def __init__(self, config: AuthConfig | None = None):
        self.config = config or AuthConfig()

        # Ensure storage directory exists
        self.config.storage_path.mkdir(parents=True, exist_ok=True)

        # Secret key for token signing
        self._secret_key = self.config.secret_key or self._load_or_generate_secret()

        # In-memory caches
        self._users: dict[str, User] = {}
        self._sessions: dict[str, Session] = {}
        self._tokens: dict[str, AccessToken] = {}

        # Load persisted data
        self._load_data()

    def _load_or_generate_secret(self) -> str:
        """Load or generate secret key."""
        secret_file = self.config.storage_path / ".secret"

        if secret_file.exists():
            return secret_file.read_text().strip()

        secret = secrets.token_hex(32)
        secret_file.write_text(secret)
        return secret

    def _hash_password(self, password: str) -> str:
        """Hash password using SHA-256 with salt."""
        salt = secrets.token_hex(16)
        hash_input = f"{salt}{password}{self._secret_key}"
        password_hash = hashlib.sha256(hash_input.encode()).hexdigest()
        return f"{salt}${password_hash}"

    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash."""
        try:
            salt, stored_hash = password_hash.split("$")
            hash_input = f"{salt}{password}{self._secret_key}"
            computed_hash = hashlib.sha256(hash_input.encode()).hexdigest()
            return secrets.compare_digest(computed_hash, stored_hash)
        except Exception:
            return False

    def _validate_password(self, password: str) -> tuple[bool, str]:
        """Validate password strength."""
        if len(password) < self.config.password_min_length:
            return False, f"Password must be at least {self.config.password_min_length} characters"

        if self.config.require_numbers and not any(c.isdigit() for c in password):
            return False, "Password must contain at least one number"

        if self.config.require_special_chars:
            special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
            if not any(c in special_chars for c in password):
                return False, "Password must contain at least one special character"

        return True, ""

    def _generate_token(self) -> str:
        """Generate a secure token."""
        return secrets.token_urlsafe(32)

    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        roles: list[Role] | None = None,
    ) -> User:
        """
        Create a new user.

        Args:
            username: Username
            email: Email address
            password: Plain text password
            roles: List of roles (default: [VIEWER])

        Returns:
            Created user
        """
        # Check if username exists
        for user in self._users.values():
            if user.username == username:
                raise ValueError(f"Username '{username}' already exists")
            if user.email == email:
                raise ValueError(f"Email '{email}' already exists")

        # Validate password
        valid, error = self._validate_password(password)
        if not valid:
            raise ValueError(error)

        now = datetime.now()

        user = User(
            id=secrets.token_hex(16),
            username=username,
            email=email,
            password_hash=self._hash_password(password),
            roles=roles or [Role.VIEWER],
            created_at=now,
            updated_at=now,
        )

        self._users[user.id] = user
        self._save_data()

        logger.info(f"Created user: {username}")
        return user

    def authenticate(
        self,
        username: str,
        password: str,
        ip_address: str | None = None,
        user_agent: str | None = None,
    ) -> Session | None:
        """
        Authenticate user and create session.

        Args:
            username: Username
            password: Plain text password
            ip_address: Client IP address
            user_agent: Client user agent

        Returns:
            Session if authentication successful, None otherwise
        """
        # Find user
        user = None
        for u in self._users.values():
            if u.username == username:
                user = u
                break

        if not user:
            logger.warning(f"Authentication failed: user not found - {username}")
            return None

        # Check if locked
        if user.is_locked:
            if user.locked_until and datetime.now() > user.locked_until:
                user.is_locked = False
                user.failed_attempts = 0
                user.locked_until = None
            else:
                logger.warning(f"Authentication failed: account locked - {username}")
                return None

        # Verify password
        if not self._verify_password(password, user.password_hash):
            user.failed_attempts += 1

            if user.failed_attempts >= self.config.lockout_attempts:
                user.is_locked = True
                user.locked_until = datetime.now() + timedelta(minutes=self.config.lockout_duration_minutes)
                logger.warning(f"Account locked due to failed attempts: {username}")

            self._save_data()
            logger.warning(f"Authentication failed: invalid password - {username}")
            return None

        # Check if active
        if not user.is_active:
            logger.warning(f"Authentication failed: account inactive - {username}")
            return None

        # Reset failed attempts
        user.failed_attempts = 0
        user.last_login = datetime.now()

        # Create session
        now = datetime.now()
        session = Session(
            id=secrets.token_urlsafe(32),
            user_id=user.id,
            created_at=now,
            expires_at=now + timedelta(hours=self.config.session_duration_hours),
            ip_address=ip_address,
            user_agent=user_agent,
        )

        # Limit sessions per user
        user_sessions = [s for s in self._sessions.values() if s.user_id == user.id and s.is_active]
        if len(user_sessions) >= self.config.max_sessions_per_user:
            # Invalidate oldest session
            oldest = min(user_sessions, key=lambda s: s.created_at)
            oldest.is_active = False

        self._sessions[session.id] = session
        self._save_data()

        logger.info(f"User authenticated: {username}")
        return session

    def validate_session(self, session_id: str) -> User | None:
        """
        Validate session and return user.

        Args:
            session_id: Session ID

        Returns:
            User if session valid, None otherwise
        """
        session = self._sessions.get(session_id)

        if not session or not session.is_active or session.is_expired():
            return None

        return self._users.get(session.user_id)

    def create_access_token(
        self,
        user: User,
        scopes: list[str] | None = None,
    ) -> AccessToken:
        """
        Create an access token for API authentication.

        Args:
            user: User
            scopes: Token scopes

        Returns:
            Access token
        """
        now = datetime.now()

        token = AccessToken(
            token=self._generate_token(),
            user_id=user.id,
            created_at=now,
            expires_at=now + timedelta(hours=self.config.token_expiry_hours),
            scopes=scopes or [],
            refresh_token=self._generate_token(),
        )

        self._tokens[token.token] = token
        self._save_data()

        return token

    def validate_token(self, token: str) -> User | None:
        """
        Validate access token and return user.

        Args:
            token: Access token

        Returns:
            User if token valid, None otherwise
        """
        access_token = self._tokens.get(token)

        if not access_token or access_token.is_expired():
            return None

        return self._users.get(access_token.user_id)

    def refresh_token(self, refresh_token: str) -> AccessToken | None:
        """
        Refresh an access token.

        Args:
            refresh_token: Refresh token

        Returns:
            New access token if valid, None otherwise
        """
        # Find token by refresh token
        old_token = None
        for t in self._tokens.values():
            if t.refresh_token == refresh_token:
                old_token = t
                break

        if not old_token:
            return None

        user = self._users.get(old_token.user_id)
        if not user:
            return None

        # Create new token
        new_token = self.create_access_token(user, old_token.scopes)

        # Invalidate old token
        del self._tokens[old_token.token]
        self._save_data()

        return new_token

    def logout(self, session_id: str) -> bool:
        """
        Logout user and invalidate session.

        Args:
            session_id: Session ID

        Returns:
            True if session was invalidated
        """
        session = self._sessions.get(session_id)

        if session:
            session.is_active = False
            self._save_data()
            logger.info(f"User logged out: session {session_id}")
            return True

        return False

    def change_password(
        self,
        user_id: str,
        old_password: str,
        new_password: str,
    ) -> bool:
        """
        Change user password.

        Args:
            user_id: User ID
            old_password: Current password
            new_password: New password

        Returns:
            True if password changed
        """
        user = self._users.get(user_id)

        if not user:
            return False

        if not self._verify_password(old_password, user.password_hash):
            return False

        valid, error = self._validate_password(new_password)
        if not valid:
            raise ValueError(error)

        user.password_hash = self._hash_password(new_password)
        user.updated_at = datetime.now()
        self._save_data()

        logger.info(f"Password changed for user: {user.username}")
        return True

    def get_user(self, user_id: str) -> User | None:
        """Get user by ID."""
        return self._users.get(user_id)

    def get_user_by_username(self, username: str) -> User | None:
        """Get user by username."""
        for user in self._users.values():
            if user.username == username:
                return user
        return None

    def list_users(self) -> list[User]:
        """List all users."""
        return list(self._users.values())

    def update_user_roles(self, user_id: str, roles: list[Role]) -> bool:
        """Update user roles."""
        user = self._users.get(user_id)

        if not user:
            return False

        user.roles = roles
        user.updated_at = datetime.now()
        self._save_data()

        return True

    def deactivate_user(self, user_id: str) -> bool:
        """Deactivate user account."""
        user = self._users.get(user_id)

        if not user:
            return False

        user.is_active = False
        user.updated_at = datetime.now()

        # Invalidate all sessions
        for session in self._sessions.values():
            if session.user_id == user_id:
                session.is_active = False

        self._save_data()
        logger.info(f"User deactivated: {user.username}")
        return True

    def _load_data(self):
        """Load persisted data."""
        users_file = self.config.storage_path / "users.json"
        sessions_file = self.config.storage_path / "sessions.json"

        if users_file.exists():
            try:
                with open(users_file) as f:
                    data = json.load(f)
                    self._users = {
                        uid: User.from_dict(udata)
                        for uid, udata in data.items()
                    }
            except Exception as e:
                logger.error(f"Error loading users: {e}")

        if sessions_file.exists():
            try:
                with open(sessions_file) as f:
                    data = json.load(f)
                    self._sessions = {
                        sid: Session.from_dict(sdata)
                        for sid, sdata in data.items()
                    }
            except Exception as e:
                logger.error(f"Error loading sessions: {e}")

    def _save_data(self):
        """Save data to disk."""
        users_file = self.config.storage_path / "users.json"
        sessions_file = self.config.storage_path / "sessions.json"

        try:
            with open(users_file, "w") as f:
                json.dump(
                    {uid: u.to_dict() for uid, u in self._users.items()},
                    f,
                    indent=2,
                )

            with open(sessions_file, "w") as f:
                json.dump(
                    {sid: s.to_dict() for sid, s in self._sessions.items()},
                    f,
                    indent=2,
                )
        except Exception as e:
            logger.error(f"Error saving auth data: {e}")

    def create_default_admin(self, password: str = "admin123!") -> User:
        """Create default admin user if none exists."""
        # Check if any admin exists
        for user in self._users.values():
            if Role.ADMIN in user.roles:
                return user

        return self.create_user(
            username="admin",
            email="admin@localhost",
            password=password,
            roles=[Role.ADMIN],
        )
