"""
Report Scheduler
================
Scheduled report generation and delivery.

Features:
- Daily/weekly P&L summaries
- Risk reports
- Signal performance reports
- Custom report scheduling
"""

import json
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ReportType(Enum):
    """Types of scheduled reports."""
    DAILY_PNL = "DAILY_PNL"
    WEEKLY_PNL = "WEEKLY_PNL"
    MONTHLY_PNL = "MONTHLY_PNL"
    DAILY_RISK = "DAILY_RISK"
    WEEKLY_RISK = "WEEKLY_RISK"
    SIGNAL_PERFORMANCE = "SIGNAL_PERFORMANCE"
    POSITION_SUMMARY = "POSITION_SUMMARY"
    TRADE_BLOTTER = "TRADE_BLOTTER"
    MARKET_OVERVIEW = "MARKET_OVERVIEW"
    CUSTOM = "CUSTOM"


class ReportFrequency(Enum):
    """Report frequency."""
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"
    CUSTOM = "CUSTOM"


@dataclass
class ReportConfig:
    """Configuration for a scheduled report."""
    report_id: str
    name: str
    report_type: ReportType

    # Scheduling
    frequency: ReportFrequency = ReportFrequency.DAILY
    schedule_time: time = time(18, 0)  # 6 PM
    schedule_days: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])  # Mon-Fri

    # For monthly reports
    schedule_day_of_month: int = 1

    # Delivery
    channels: list[str] = field(default_factory=lambda: ["email"])
    recipients: list[str] = field(default_factory=list)

    # Content settings
    include_charts: bool = True
    include_details: bool = True
    date_range_days: int = 1  # For daily, 7 for weekly, etc.

    # Custom report generator
    generator_function: Callable[[], dict] | None = None

    # Status
    enabled: bool = True
    last_run: datetime | None = None
    next_run: datetime | None = None
    run_count: int = 0

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    tags: list[str] = field(default_factory=list)


@dataclass
class ScheduledReport:
    """A scheduled report instance."""
    config: ReportConfig

    def should_run(self, now: datetime | None = None) -> bool:
        """Check if report should run now."""
        if not self.config.enabled:
            return False

        now = now or datetime.now()
        current_time = now.time()
        current_day = now.weekday()

        # Check day
        if self.config.frequency == ReportFrequency.MONTHLY:
            if now.day != self.config.schedule_day_of_month:
                return False
        elif current_day not in self.config.schedule_days:
            return False

        # Check time (within 1 minute window)
        schedule_minutes = self.config.schedule_time.hour * 60 + self.config.schedule_time.minute
        current_minutes = current_time.hour * 60 + current_time.minute

        if abs(current_minutes - schedule_minutes) > 1:
            return False

        # Check if already run today
        return not (self.config.last_run and self.config.last_run.date() == now.date())

    def calculate_next_run(self) -> datetime:
        """Calculate next run time."""
        now = datetime.now()

        if self.config.frequency == ReportFrequency.DAILY:
            # Find next scheduled day
            for days_ahead in range(7):
                next_date = now.date() + timedelta(days=days_ahead)
                if next_date.weekday() in self.config.schedule_days:
                    next_run = datetime.combine(next_date, self.config.schedule_time)
                    if next_run > now:
                        return next_run

            # Default to tomorrow
            return datetime.combine(now.date() + timedelta(days=1), self.config.schedule_time)

        elif self.config.frequency == ReportFrequency.WEEKLY:
            # Find next scheduled day (assuming first day in list is the report day)
            target_day = self.config.schedule_days[0] if self.config.schedule_days else 0
            days_ahead = (target_day - now.weekday() + 7) % 7
            if days_ahead == 0 and now.time() >= self.config.schedule_time:
                days_ahead = 7
            next_date = now.date() + timedelta(days=days_ahead)
            return datetime.combine(next_date, self.config.schedule_time)

        elif self.config.frequency == ReportFrequency.MONTHLY:
            # Next month on schedule_day_of_month
            if now.day < self.config.schedule_day_of_month:
                next_date = now.replace(day=self.config.schedule_day_of_month)
            else:
                # Next month
                if now.month == 12:
                    next_date = now.replace(year=now.year + 1, month=1, day=self.config.schedule_day_of_month)
                else:
                    next_date = now.replace(month=now.month + 1, day=self.config.schedule_day_of_month)
            return datetime.combine(next_date.date(), self.config.schedule_time)

        # Default
        return datetime.combine(now.date() + timedelta(days=1), self.config.schedule_time)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "report_id": self.config.report_id,
            "name": self.config.name,
            "report_type": self.config.report_type.value,
            "frequency": self.config.frequency.value,
            "schedule_time": self.config.schedule_time.strftime("%H:%M"),
            "channels": self.config.channels,
            "enabled": self.config.enabled,
            "last_run": self.config.last_run.isoformat() if self.config.last_run else None,
            "next_run": self.config.next_run.isoformat() if self.config.next_run else None,
            "run_count": self.config.run_count,
        }


class ReportScheduler:
    """
    Scheduler for automated report generation and delivery.

    Manages report schedules, generates content, and dispatches to channels.
    """

    def __init__(
        self,
        storage_path: str = "data/reports",
        report_generators: dict[ReportType, Callable] | None = None,
    ):
        self._reports: dict[str, ScheduledReport] = {}
        self._generators = report_generators or {}
        self._channels: dict[str, Any] = {}

        self._storage_path = Path(storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)

        self._is_running = False
        self._lock = threading.Lock()
        self._last_check: datetime | None = None

        # Load saved schedules
        self._load_schedules()

    # =========================================================================
    # Report Management
    # =========================================================================

    def add_report(self, config: ReportConfig) -> ScheduledReport:
        """Add a scheduled report."""
        with self._lock:
            report = ScheduledReport(config=config)
            report.config.next_run = report.calculate_next_run()
            self._reports[config.report_id] = report
            self._save_schedules()

            logger.info(f"Added scheduled report: {config.name} ({config.report_id})")
            return report

    def remove_report(self, report_id: str) -> bool:
        """Remove a scheduled report."""
        with self._lock:
            if report_id in self._reports:
                del self._reports[report_id]
                self._save_schedules()
                logger.info(f"Removed scheduled report: {report_id}")
                return True
            return False

    def update_report(self, report_id: str, updates: dict) -> ScheduledReport | None:
        """Update a scheduled report."""
        with self._lock:
            if report_id not in self._reports:
                return None

            report = self._reports[report_id]
            config = report.config

            if "enabled" in updates:
                config.enabled = updates["enabled"]
            if "schedule_time" in updates:
                config.schedule_time = updates["schedule_time"]
            if "channels" in updates:
                config.channels = updates["channels"]

            report.config.next_run = report.calculate_next_run()
            self._save_schedules()

            return report

    def get_report(self, report_id: str) -> ScheduledReport | None:
        """Get a report by ID."""
        return self._reports.get(report_id)

    def get_reports(
        self,
        enabled: bool | None = None,
        report_type: ReportType | None = None,
    ) -> list[ScheduledReport]:
        """Get reports matching filters."""
        reports = list(self._reports.values())

        if enabled is not None:
            reports = [r for r in reports if r.config.enabled == enabled]

        if report_type:
            reports = [r for r in reports if r.config.report_type == report_type]

        return reports

    # =========================================================================
    # Execution
    # =========================================================================

    def check_and_run(self) -> list[str]:
        """Check for due reports and run them."""
        now = datetime.now()
        executed = []

        with self._lock:
            for report_id, report in self._reports.items():
                if report.should_run(now):
                    try:
                        self._run_report(report)
                        executed.append(report_id)

                        # Update state
                        report.config.last_run = now
                        report.config.run_count += 1
                        report.config.next_run = report.calculate_next_run()

                    except Exception as e:
                        logger.error(f"Failed to run report {report_id}: {e}")

        self._last_check = now

        if executed:
            self._save_schedules()
            logger.info(f"Executed {len(executed)} scheduled reports")

        return executed

    def run_report_now(self, report_id: str) -> bool:
        """Run a specific report immediately."""
        with self._lock:
            if report_id not in self._reports:
                return False

            report = self._reports[report_id]

            try:
                self._run_report(report)
                report.config.last_run = datetime.now()
                report.config.run_count += 1
                self._save_schedules()

                logger.info(f"Manually ran report: {report_id}")
                return True

            except Exception as e:
                logger.error(f"Failed to run report {report_id}: {e}")
                return False

    def _run_report(self, report: ScheduledReport):
        """Execute a report."""
        config = report.config

        # Generate report content
        content = self._generate_content(report)

        # Send to channels
        for channel_name in config.channels:
            if channel_name in self._channels:
                try:
                    self._channels[channel_name].send_report(content)
                except Exception as e:
                    logger.error(f"Failed to send report to {channel_name}: {e}")

    def _generate_content(self, report: ScheduledReport) -> dict:
        """Generate report content."""
        config = report.config
        report_type = config.report_type

        # Check for custom generator
        if config.generator_function:
            return config.generator_function()

        # Check for registered generator
        if report_type in self._generators:
            return self._generators[report_type]()

        # Default content
        return self._generate_default_content(report)

    def _generate_default_content(self, report: ScheduledReport) -> dict:
        """Generate default report content."""
        config = report.config
        now = datetime.now()

        return {
            "title": config.name,
            "report_type": config.report_type.value,
            "generated_at": now.isoformat(),
            "period_start": (now - timedelta(days=config.date_range_days)).isoformat(),
            "period_end": now.isoformat(),
            "content": {
                "message": f"Report: {config.name}",
                "type": config.report_type.value,
            },
        }

    # =========================================================================
    # Report Generators
    # =========================================================================

    def register_generator(self, report_type: ReportType, generator: Callable[[], dict]):
        """Register a report generator."""
        self._generators[report_type] = generator
        logger.info(f"Registered generator for {report_type.value}")

    def register_channel(self, name: str, channel: Any):
        """Register a notification channel."""
        self._channels[name] = channel
        logger.info(f"Registered channel for reports: {name}")

    # =========================================================================
    # Built-in Generators
    # =========================================================================

    @staticmethod
    def generate_daily_pnl_report() -> dict:
        """Generate daily P&L report content."""
        now = datetime.now()

        return {
            "title": "Daily P&L Report",
            "report_type": "DAILY_PNL",
            "generated_at": now.isoformat(),
            "sections": [
                {
                    "title": "Portfolio Summary",
                    "content": "Today's trading summary and P&L breakdown."
                },
                {
                    "title": "Position Summary",
                    "content": "Current open positions and unrealized P&L."
                },
                {
                    "title": "Trade Activity",
                    "content": "Trades executed today with realized P&L."
                },
            ],
        }

    @staticmethod
    def generate_risk_report() -> dict:
        """Generate risk report content."""
        now = datetime.now()

        return {
            "title": "Risk Report",
            "report_type": "DAILY_RISK",
            "generated_at": now.isoformat(),
            "sections": [
                {
                    "title": "VaR Summary",
                    "content": "Portfolio VaR metrics and limit utilization."
                },
                {
                    "title": "Exposure Analysis",
                    "content": "Gross and net exposure breakdown."
                },
                {
                    "title": "Concentration",
                    "content": "Position concentration by instrument and strategy."
                },
            ],
        }

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_statistics(self) -> dict:
        """Get scheduler statistics."""
        total = len(self._reports)
        enabled = len([r for r in self._reports.values() if r.config.enabled])

        # Next scheduled
        upcoming = sorted(
            [r for r in self._reports.values() if r.config.enabled and r.config.next_run],
            key=lambda r: r.config.next_run,
        )

        next_report = upcoming[0] if upcoming else None

        return {
            "total_reports": total,
            "enabled_reports": enabled,
            "disabled_reports": total - enabled,
            "next_scheduled": next_report.to_dict() if next_report else None,
            "last_check": self._last_check.isoformat() if self._last_check else None,
        }

    # =========================================================================
    # Persistence
    # =========================================================================

    def _load_schedules(self):
        """Load saved schedules."""
        schedule_path = self._storage_path / "schedules.json"

        if schedule_path.exists():
            try:
                with open(schedule_path) as f:
                    data = json.load(f)

                for report_data in data.get("reports", []):
                    config = self._deserialize_config(report_data)
                    report = ScheduledReport(config=config)
                    report.config.next_run = report.calculate_next_run()
                    self._reports[config.report_id] = report

                logger.info(f"Loaded {len(self._reports)} scheduled reports")

            except Exception as e:
                logger.error(f"Failed to load schedules: {e}")

    def _save_schedules(self):
        """Save schedules."""
        schedule_path = self._storage_path / "schedules.json"

        try:
            reports_data = [
                self._serialize_config(r.config)
                for r in self._reports.values()
            ]

            with open(schedule_path, "w") as f:
                json.dump({"reports": reports_data}, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save schedules: {e}")

    def _serialize_config(self, config: ReportConfig) -> dict:
        """Serialize report configuration."""
        return {
            "report_id": config.report_id,
            "name": config.name,
            "report_type": config.report_type.value,
            "frequency": config.frequency.value,
            "schedule_time": config.schedule_time.strftime("%H:%M"),
            "schedule_days": config.schedule_days,
            "schedule_day_of_month": config.schedule_day_of_month,
            "channels": config.channels,
            "recipients": config.recipients,
            "enabled": config.enabled,
            "include_charts": config.include_charts,
            "include_details": config.include_details,
            "date_range_days": config.date_range_days,
            "last_run": config.last_run.isoformat() if config.last_run else None,
            "run_count": config.run_count,
            "tags": config.tags,
        }

    def _deserialize_config(self, data: dict) -> ReportConfig:
        """Deserialize report configuration."""
        schedule_time = time.fromisoformat(data["schedule_time"])
        last_run = data.get("last_run")
        if last_run:
            last_run = datetime.fromisoformat(last_run)

        return ReportConfig(
            report_id=data["report_id"],
            name=data["name"],
            report_type=ReportType(data.get("report_type", "CUSTOM")),
            frequency=ReportFrequency(data.get("frequency", "DAILY")),
            schedule_time=schedule_time,
            schedule_days=data.get("schedule_days", [0, 1, 2, 3, 4]),
            schedule_day_of_month=data.get("schedule_day_of_month", 1),
            channels=data.get("channels", ["email"]),
            recipients=data.get("recipients", []),
            enabled=data.get("enabled", True),
            include_charts=data.get("include_charts", True),
            include_details=data.get("include_details", True),
            date_range_days=data.get("date_range_days", 1),
            last_run=last_run,
            run_count=data.get("run_count", 0),
            tags=data.get("tags", []),
        )
