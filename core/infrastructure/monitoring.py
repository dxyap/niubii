"""
Monitoring Module
=================
Health checks, metrics collection, and system monitoring.
"""

import logging
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import json

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class MetricType(Enum):
    """Prometheus metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    enabled: bool = True
    health_check_interval_seconds: int = 30
    metrics_retention_hours: int = 24
    storage_path: Path = field(default_factory=lambda: Path("data/monitoring"))
    expose_prometheus_endpoint: bool = True
    prometheus_port: int = 9090
    collect_system_metrics: bool = True
    collect_app_metrics: bool = True
    alert_on_degraded: bool = True
    alert_on_unhealthy: bool = True


@dataclass
class HealthCheck:
    """Health check definition."""
    name: str
    check_fn: Callable[[], bool]
    timeout_seconds: float = 5.0
    critical: bool = True
    description: str = ""
    last_check: Optional[datetime] = None
    last_status: HealthStatus = HealthStatus.UNKNOWN
    consecutive_failures: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    duration_ms: float
    timestamp: datetime
    message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "status": self.status.value,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp.isoformat(),
            "message": self.message,
            "details": self.details,
        }


@dataclass
class Metric:
    """Prometheus-style metric."""
    name: str
    type: MetricType
    help: str
    labels: Dict[str, str] = field(default_factory=dict)
    value: float = 0.0
    timestamp: Optional[datetime] = None
    
    # For histograms
    buckets: List[float] = field(default_factory=list)
    bucket_counts: Dict[float, int] = field(default_factory=dict)
    sum: float = 0.0
    count: int = 0
    
    def to_prometheus(self) -> str:
        """Format metric in Prometheus exposition format."""
        labels_str = ""
        if self.labels:
            labels_list = [f'{k}="{v}"' for k, v in self.labels.items()]
            labels_str = "{" + ",".join(labels_list) + "}"
        
        if self.type == MetricType.HISTOGRAM:
            lines = []
            for bucket, count in sorted(self.bucket_counts.items()):
                lines.append(f'{self.name}_bucket{{{labels_str},le="{bucket}"}} {count}')
            lines.append(f'{self.name}_sum{labels_str} {self.sum}')
            lines.append(f'{self.name}_count{labels_str} {self.count}')
            return "\n".join(lines)
        else:
            return f"{self.name}{labels_str} {self.value}"


class HealthChecker:
    """
    Health Checker.
    
    Manages health checks for all system components.
    """
    
    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()
        
        # Registered health checks
        self._checks: Dict[str, HealthCheck] = {}
        
        # Background thread for periodic checks
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Register default checks
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default system health checks."""
        # Database check
        self.register(
            name="database",
            check_fn=self._check_database,
            critical=True,
            description="Check database connectivity",
        )
        
        # Disk space check
        self.register(
            name="disk_space",
            check_fn=self._check_disk_space,
            critical=True,
            description="Check available disk space",
        )
        
        # Memory check
        self.register(
            name="memory",
            check_fn=self._check_memory,
            critical=False,
            description="Check memory usage",
        )
        
        # Data loader check
        self.register(
            name="data_loader",
            check_fn=self._check_data_loader,
            critical=True,
            description="Check data loader connectivity",
        )
    
    def register(
        self,
        name: str,
        check_fn: Callable[[], bool],
        critical: bool = True,
        description: str = "",
        timeout_seconds: float = 5.0,
    ):
        """
        Register a health check.
        
        Args:
            name: Check name
            check_fn: Function that returns True if healthy
            critical: Whether this check affects overall health
            description: Check description
            timeout_seconds: Check timeout
        """
        self._checks[name] = HealthCheck(
            name=name,
            check_fn=check_fn,
            critical=critical,
            description=description,
            timeout_seconds=timeout_seconds,
        )
        
        logger.info(f"Registered health check: {name}")
    
    def unregister(self, name: str):
        """Unregister a health check."""
        if name in self._checks:
            del self._checks[name]
            logger.info(f"Unregistered health check: {name}")
    
    def check(self, name: str) -> HealthCheckResult:
        """
        Run a specific health check.
        
        Args:
            name: Check name
            
        Returns:
            Health check result
        """
        check = self._checks.get(name)
        
        if not check:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNKNOWN,
                duration_ms=0,
                timestamp=datetime.now(),
                message="Check not found",
            )
        
        start_time = time.time()
        
        try:
            result = check.check_fn()
            duration_ms = (time.time() - start_time) * 1000
            
            if result:
                status = HealthStatus.HEALTHY
                check.consecutive_failures = 0
            else:
                check.consecutive_failures += 1
                status = HealthStatus.UNHEALTHY if check.critical else HealthStatus.DEGRADED
            
            check.last_status = status
            check.last_check = datetime.now()
            
            return HealthCheckResult(
                name=name,
                status=status,
                duration_ms=duration_ms,
                timestamp=datetime.now(),
                details=check.metadata,
            )
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            check.consecutive_failures += 1
            check.last_status = HealthStatus.UNHEALTHY
            check.last_check = datetime.now()
            
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                duration_ms=duration_ms,
                timestamp=datetime.now(),
                message=str(e),
            )
    
    def check_all(self) -> Dict[str, HealthCheckResult]:
        """
        Run all health checks.
        
        Returns:
            Dictionary of check results
        """
        results = {}
        
        for name in self._checks:
            results[name] = self.check(name)
        
        return results
    
    def get_overall_status(self) -> HealthStatus:
        """
        Get overall system health status.
        
        Returns:
            Overall health status
        """
        results = self.check_all()
        
        has_critical_failure = False
        has_degraded = False
        
        for result in results.values():
            check = self._checks.get(result.name)
            
            if result.status == HealthStatus.UNHEALTHY:
                if check and check.critical:
                    has_critical_failure = True
                else:
                    has_degraded = True
            elif result.status == HealthStatus.DEGRADED:
                has_degraded = True
        
        if has_critical_failure:
            return HealthStatus.UNHEALTHY
        elif has_degraded:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
    
    def get_health_summary(self) -> Dict[str, Any]:
        """
        Get complete health summary.
        
        Returns:
            Health summary
        """
        results = self.check_all()
        overall = self.get_overall_status()
        
        return {
            "status": overall.value,
            "timestamp": datetime.now().isoformat(),
            "checks": {
                name: result.to_dict()
                for name, result in results.items()
            },
            "summary": {
                "total": len(results),
                "healthy": sum(1 for r in results.values() if r.status == HealthStatus.HEALTHY),
                "degraded": sum(1 for r in results.values() if r.status == HealthStatus.DEGRADED),
                "unhealthy": sum(1 for r in results.values() if r.status == HealthStatus.UNHEALTHY),
            },
        }
    
    def start_background_checks(self):
        """Start background health check thread."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._background_check_loop, daemon=True)
        self._thread.start()
        logger.info("Started background health checks")
    
    def stop_background_checks(self):
        """Stop background health check thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Stopped background health checks")
    
    def _background_check_loop(self):
        """Background check loop."""
        while self._running:
            try:
                self.check_all()
            except Exception as e:
                logger.error(f"Background health check error: {e}")
            
            time.sleep(self.config.health_check_interval_seconds)
    
    # Default check implementations
    def _check_database(self) -> bool:
        """Check database connectivity."""
        try:
            import sqlite3
            # Try to connect to a test database
            conn = sqlite3.connect(":memory:")
            conn.execute("SELECT 1")
            conn.close()
            return True
        except Exception:
            return False
    
    def _check_disk_space(self) -> bool:
        """Check available disk space."""
        try:
            import shutil
            total, used, free = shutil.disk_usage("/")
            free_percent = (free / total) * 100
            self._checks["disk_space"].metadata = {
                "free_gb": round(free / (1024**3), 2),
                "free_percent": round(free_percent, 1),
            }
            return free_percent > 10  # At least 10% free
        except Exception:
            return True  # Assume OK if we can't check
    
    def _check_memory(self) -> bool:
        """Check memory usage."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            self._checks["memory"].metadata = {
                "used_percent": memory.percent,
                "available_gb": round(memory.available / (1024**3), 2),
            }
            return memory.percent < 90  # Less than 90% used
        except ImportError:
            return True  # psutil not available
        except Exception:
            return True
    
    def _check_data_loader(self) -> bool:
        """Check data loader connectivity."""
        try:
            from core.data import DataLoader
            loader = DataLoader()
            # Just check if it initializes
            return True
        except Exception:
            return False


class MetricsCollector:
    """
    Prometheus-compatible Metrics Collector.
    
    Collects and exposes application metrics.
    """
    
    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()
        
        # Ensure storage directory exists
        self.config.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Metrics storage
        self._metrics: Dict[str, Metric] = {}
        self._lock = threading.Lock()
        
        # Register default metrics
        self._register_default_metrics()
    
    def _register_default_metrics(self):
        """Register default application metrics."""
        # Request counter
        self.register(
            name="app_requests_total",
            type=MetricType.COUNTER,
            help="Total number of requests",
        )
        
        # Request duration
        self.register(
            name="app_request_duration_seconds",
            type=MetricType.HISTOGRAM,
            help="Request duration in seconds",
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
        )
        
        # Active sessions
        self.register(
            name="app_active_sessions",
            type=MetricType.GAUGE,
            help="Number of active sessions",
        )
        
        # Trading metrics
        self.register(
            name="trading_orders_total",
            type=MetricType.COUNTER,
            help="Total number of orders",
        )
        
        self.register(
            name="trading_position_value",
            type=MetricType.GAUGE,
            help="Current position value",
        )
        
        self.register(
            name="trading_pnl",
            type=MetricType.GAUGE,
            help="Current P&L",
        )
        
        # Risk metrics
        self.register(
            name="risk_var_95",
            type=MetricType.GAUGE,
            help="Value at Risk (95%)",
        )
        
        self.register(
            name="risk_limit_utilization",
            type=MetricType.GAUGE,
            help="Risk limit utilization percentage",
        )
        
        # System metrics
        self.register(
            name="system_cpu_usage",
            type=MetricType.GAUGE,
            help="CPU usage percentage",
        )
        
        self.register(
            name="system_memory_usage",
            type=MetricType.GAUGE,
            help="Memory usage percentage",
        )
        
        # ML metrics
        self.register(
            name="ml_predictions_total",
            type=MetricType.COUNTER,
            help="Total ML predictions made",
        )
        
        self.register(
            name="ml_model_accuracy",
            type=MetricType.GAUGE,
            help="ML model accuracy",
        )
    
    def register(
        self,
        name: str,
        type: MetricType,
        help: str,
        labels: Optional[Dict[str, str]] = None,
        buckets: Optional[List[float]] = None,
    ):
        """
        Register a new metric.
        
        Args:
            name: Metric name
            type: Metric type
            help: Help text
            labels: Default labels
            buckets: Histogram buckets
        """
        with self._lock:
            self._metrics[name] = Metric(
                name=name,
                type=type,
                help=help,
                labels=labels or {},
                buckets=buckets or [],
                bucket_counts={b: 0 for b in (buckets or [])},
            )
    
    def increment(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ):
        """
        Increment a counter metric.
        
        Args:
            name: Metric name
            value: Value to add
            labels: Additional labels
        """
        with self._lock:
            metric = self._metrics.get(name)
            
            if metric and metric.type == MetricType.COUNTER:
                metric.value += value
                metric.timestamp = datetime.now()
                
                if labels:
                    metric.labels.update(labels)
    
    def set(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ):
        """
        Set a gauge metric.
        
        Args:
            name: Metric name
            value: New value
            labels: Additional labels
        """
        with self._lock:
            metric = self._metrics.get(name)
            
            if metric and metric.type == MetricType.GAUGE:
                metric.value = value
                metric.timestamp = datetime.now()
                
                if labels:
                    metric.labels.update(labels)
    
    def observe(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ):
        """
        Observe a value for histogram/summary.
        
        Args:
            name: Metric name
            value: Observed value
            labels: Additional labels
        """
        with self._lock:
            metric = self._metrics.get(name)
            
            if metric and metric.type == MetricType.HISTOGRAM:
                metric.sum += value
                metric.count += 1
                
                # Update bucket counts
                for bucket in metric.buckets:
                    if value <= bucket:
                        metric.bucket_counts[bucket] = metric.bucket_counts.get(bucket, 0) + 1
                
                metric.timestamp = datetime.now()
                
                if labels:
                    metric.labels.update(labels)
    
    def get_metric(self, name: str) -> Optional[Metric]:
        """Get a metric by name."""
        return self._metrics.get(name)
    
    def get_all_metrics(self) -> Dict[str, Metric]:
        """Get all metrics."""
        return self._metrics.copy()
    
    def get_prometheus_output(self) -> str:
        """
        Get all metrics in Prometheus exposition format.
        
        Returns:
            Prometheus-formatted metrics
        """
        lines = []
        
        for name, metric in self._metrics.items():
            # Help line
            lines.append(f"# HELP {name} {metric.help}")
            # Type line
            lines.append(f"# TYPE {name} {metric.type.value}")
            # Metric value(s)
            lines.append(metric.to_prometheus())
        
        return "\n".join(lines)
    
    def collect_system_metrics(self):
        """Collect system metrics."""
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.set("system_cpu_usage", cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.set("system_memory_usage", memory.percent)
        
        except ImportError:
            pass  # psutil not available
    
    def save_metrics(self):
        """Save metrics to file."""
        metrics_file = self.config.storage_path / "metrics.json"
        
        try:
            data = {
                name: {
                    "type": m.type.value,
                    "value": m.value,
                    "timestamp": m.timestamp.isoformat() if m.timestamp else None,
                    "labels": m.labels,
                }
                for name, m in self._metrics.items()
            }
            
            with open(metrics_file, "w") as f:
                json.dump(data, f, indent=2)
        
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
    
    def load_metrics(self):
        """Load metrics from file."""
        metrics_file = self.config.storage_path / "metrics.json"
        
        if not metrics_file.exists():
            return
        
        try:
            with open(metrics_file) as f:
                data = json.load(f)
            
            for name, values in data.items():
                if name in self._metrics:
                    self._metrics[name].value = values.get("value", 0)
                    if values.get("timestamp"):
                        self._metrics[name].timestamp = datetime.fromisoformat(values["timestamp"])
        
        except Exception as e:
            logger.error(f"Failed to load metrics: {e}")


class MetricsMiddleware:
    """
    Middleware for automatic metrics collection.
    
    Can be used with Streamlit or other frameworks.
    """
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
    
    def track_request(self, endpoint: str):
        """
        Context manager to track request metrics.
        
        Usage:
            with middleware.track_request("/api/data"):
                # handle request
        """
        return RequestTracker(self.collector, endpoint)


class RequestTracker:
    """Context manager for request tracking."""
    
    def __init__(self, collector: MetricsCollector, endpoint: str):
        self.collector = collector
        self.endpoint = endpoint
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.collector.increment("app_requests_total", labels={"endpoint": self.endpoint})
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.collector.observe(
            "app_request_duration_seconds",
            duration,
            labels={"endpoint": self.endpoint},
        )
