"""
Risk Monitoring
===============
Real-time risk monitoring and alerting.
"""

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd


@dataclass
class RiskAlert:
    """Represents a risk alert."""
    alert_id: str
    alert_type: str
    severity: str  # WARNING, CRITICAL, BREACH
    message: str
    metric_name: str
    current_value: float
    limit_value: float
    utilization_pct: float
    timestamp: datetime
    acknowledged: bool = False

    def to_dict(self) -> dict:
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type,
            "severity": self.severity,
            "message": self.message,
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "limit_value": self.limit_value,
            "utilization_pct": self.utilization_pct,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged,
        }


class RiskMonitor:
    """
    Real-time risk monitoring and alerting.

    Features:
    - Continuous risk metric monitoring
    - Alert generation and management
    - Risk dashboard data aggregation
    """

    def __init__(self):
        """Initialize risk monitor."""
        self.alerts: list[RiskAlert] = []
        self.alert_counter = 0

        # Risk metrics history
        self.var_history: list[dict] = []
        self.pnl_history: list[dict] = []
        self.exposure_history: list[dict] = []

    def generate_alert(
        self,
        alert_type: str,
        severity: str,
        metric_name: str,
        current_value: float,
        limit_value: float,
        utilization_pct: float
    ) -> RiskAlert:
        """
        Generate a risk alert.

        Args:
            alert_type: Type of alert
            severity: Alert severity
            metric_name: Name of the metric
            current_value: Current metric value
            limit_value: Limit value
            utilization_pct: Utilization percentage

        Returns:
            Generated alert
        """
        self.alert_counter += 1

        message = f"{metric_name} at {utilization_pct:.1f}% of limit ({current_value:,.0f} / {limit_value:,.0f})"

        alert = RiskAlert(
            alert_id=f"ALERT-{datetime.now().strftime('%Y%m%d')}-{self.alert_counter:04d}",
            alert_type=alert_type,
            severity=severity,
            message=message,
            metric_name=metric_name,
            current_value=current_value,
            limit_value=limit_value,
            utilization_pct=utilization_pct,
            timestamp=datetime.now(),
        )

        self.alerts.append(alert)
        return alert

    def check_and_alert(
        self,
        metric_name: str,
        current_value: float,
        limit_value: float,
        alert_type: str = "limit"
    ) -> RiskAlert | None:
        """
        Check metric and generate alert if needed.

        Args:
            metric_name: Name of the metric
            current_value: Current value
            limit_value: Limit value
            alert_type: Type of alert

        Returns:
            Alert if generated, None otherwise
        """
        if limit_value <= 0:
            return None

        utilization = current_value / limit_value * 100

        if utilization >= 100:
            severity = "BREACH"
        elif utilization >= 90:
            severity = "CRITICAL"
        elif utilization >= 75:
            severity = "WARNING"
        else:
            return None

        return self.generate_alert(
            alert_type=alert_type,
            severity=severity,
            metric_name=metric_name,
            current_value=current_value,
            limit_value=limit_value,
            utilization_pct=utilization,
        )

    def get_active_alerts(self, severity: str | None = None) -> list[dict]:
        """
        Get active (unacknowledged) alerts.

        Args:
            severity: Optional severity filter

        Returns:
            List of active alerts
        """
        active = [a for a in self.alerts if not a.acknowledged]

        if severity:
            active = [a for a in active if a.severity == severity]

        return [a.to_dict() for a in active]

    def acknowledge_alert(self, alert_id: str) -> bool:
        """
        Acknowledge an alert.

        Args:
            alert_id: Alert ID to acknowledge

        Returns:
            True if acknowledged, False if not found
        """
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False

    def record_var(self, var_value: float, portfolio_value: float) -> None:
        """Record VaR observation."""
        self.var_history.append({
            "timestamp": datetime.now(),
            "var": var_value,
            "portfolio_value": portfolio_value,
            "var_pct": var_value / portfolio_value * 100 if portfolio_value > 0 else 0,
        })

        # Keep last 1000 observations
        if len(self.var_history) > 1000:
            self.var_history = self.var_history[-1000:]

    def record_pnl(self, pnl: float, cumulative_pnl: float) -> None:
        """Record P&L observation."""
        self.pnl_history.append({
            "timestamp": datetime.now(),
            "pnl": pnl,
            "cumulative_pnl": cumulative_pnl,
        })

        if len(self.pnl_history) > 1000:
            self.pnl_history = self.pnl_history[-1000:]

    def calculate_drawdown(self) -> dict:
        """
        Calculate current drawdown metrics.

        Returns:
            Drawdown metrics
        """
        if not self.pnl_history:
            return {
                "current_drawdown": 0,
                "max_drawdown": 0,
                "drawdown_duration": 0,
            }

        cumulative_pnl = [h["cumulative_pnl"] for h in self.pnl_history]

        # Running maximum
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = cumulative_pnl - running_max

        current_drawdown = drawdown[-1] if drawdown.size > 0 else 0
        max_drawdown = np.min(drawdown) if drawdown.size > 0 else 0

        # Drawdown duration (periods since last high)
        if running_max[-1] == cumulative_pnl[-1]:
            duration = 0
        else:
            # Find last peak
            peaks = np.where(np.array(cumulative_pnl) == np.array(running_max))[0]
            duration = len(cumulative_pnl) - peaks[-1] - 1 if len(peaks) > 0 else len(cumulative_pnl)

        return {
            "current_drawdown": round(current_drawdown, 2),
            "max_drawdown": round(max_drawdown, 2),
            "drawdown_duration": duration,
            "at_high_water_mark": current_drawdown == 0,
        }

    def get_risk_summary(
        self,
        positions: dict[str, dict],
        current_var: float,
        var_limit: float
    ) -> dict:
        """
        Get comprehensive risk summary.

        Args:
            positions: Current positions
            current_var: Current VaR
            var_limit: VaR limit

        Returns:
            Risk summary dictionary
        """
        # Calculate exposure
        gross_exposure = 0
        net_exposure = 0

        for ticker, pos in positions.items():
            contract_type = ticker[:2]
            multiplier = 1000 if contract_type in ["CL", "CO"] else 42000
            value = pos["quantity"] * pos["price"] * multiplier

            gross_exposure += abs(value)
            net_exposure += value

        # Get drawdown
        drawdown = self.calculate_drawdown()

        # Count alerts
        active_alerts = [a for a in self.alerts if not a.acknowledged]
        critical_alerts = len([a for a in active_alerts if a.severity == "CRITICAL"])
        breach_alerts = len([a for a in active_alerts if a.severity == "BREACH"])

        return {
            "var": {
                "current": round(current_var, 2),
                "limit": var_limit,
                "utilization_pct": round(current_var / var_limit * 100, 1) if var_limit > 0 else 0,
            },
            "exposure": {
                "gross": round(gross_exposure, 2),
                "net": round(net_exposure, 2),
                "direction": "Long" if net_exposure > 0 else "Short" if net_exposure < 0 else "Flat",
            },
            "drawdown": drawdown,
            "alerts": {
                "total_active": len(active_alerts),
                "critical": critical_alerts,
                "breach": breach_alerts,
            },
            "positions": len(positions),
            "timestamp": datetime.now().isoformat(),
        }

    def get_position_risk_breakdown(
        self,
        positions: dict[str, dict]
    ) -> pd.DataFrame:
        """
        Get risk breakdown by position.

        Args:
            positions: Current positions

        Returns:
            DataFrame with position risk metrics
        """
        data = []

        total_exposure = sum(
            abs(pos["quantity"] * pos["price"] * (1000 if ticker[:2] in ["CL", "CO"] else 42000))
            for ticker, pos in positions.items()
        )

        for ticker, pos in positions.items():
            contract_type = ticker[:2]
            multiplier = 1000 if contract_type in ["CL", "CO"] else 42000

            notional = pos["quantity"] * pos["price"] * multiplier
            weight = abs(notional) / total_exposure * 100 if total_exposure > 0 else 0

            # Simplified contribution to risk
            risk_contrib = weight * 0.02  # Assume 2% daily vol

            data.append({
                "ticker": ticker,
                "quantity": pos["quantity"],
                "price": pos["price"],
                "notional": round(notional, 2),
                "weight_pct": round(weight, 1),
                "direction": "Long" if pos["quantity"] > 0 else "Short",
                "risk_contribution_pct": round(risk_contrib, 2),
            })

        return pd.DataFrame(data)
