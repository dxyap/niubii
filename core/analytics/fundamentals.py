"""
Fundamental Analysis
====================
Analysis of oil market fundamentals including inventory, OPEC, and refinery data.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd


class FundamentalAnalyzer:
    """
    Fundamental analysis for oil markets.

    Features:
    - EIA inventory analysis
    - OPEC production monitoring
    - Refinery turnaround tracking
    - Supply/demand balance
    """

    # Historical averages (for percentile calculations)
    INVENTORY_STATS = {
        "us_crude": {"mean": 440, "std": 30, "min": 380, "max": 540},  # MMbbl
        "cushing": {"mean": 40, "std": 15, "min": 20, "max": 80},
        "gasoline": {"mean": 230, "std": 15, "min": 200, "max": 265},
        "distillate": {"mean": 140, "std": 20, "min": 100, "max": 180},
    }

    def __init__(self):
        """Initialize fundamental analyzer."""
        pass

    def analyze_inventory(
        self,
        current_level: float,
        change: float,
        expectation: float,
        inventory_type: str = "us_crude"
    ) -> dict:
        """
        Analyze inventory report.

        Args:
            current_level: Current inventory level (MMbbl)
            change: Week-over-week change
            expectation: Expected change
            inventory_type: Type of inventory

        Returns:
            Inventory analysis
        """
        surprise = change - expectation

        # Get historical stats
        stats = self.INVENTORY_STATS.get(inventory_type, self.INVENTORY_STATS["us_crude"])

        # Calculate percentile
        zscore = (current_level - stats["mean"]) / stats["std"]
        percentile = 50 + 34.13 * np.sign(zscore) * (1 - np.exp(-abs(zscore)))
        percentile = max(0, min(100, percentile))

        # Interpret surprise
        if surprise < -2:
            surprise_signal = "Bullish (Larger draw than expected)"
        elif surprise < -0.5:
            surprise_signal = "Slightly Bullish"
        elif surprise > 2:
            surprise_signal = "Bearish (Larger build than expected)"
        elif surprise > 0.5:
            surprise_signal = "Slightly Bearish"
        else:
            surprise_signal = "Neutral (In line with expectations)"

        # Level assessment
        if percentile < 20:
            level_signal = "Very Low - Bullish"
        elif percentile < 40:
            level_signal = "Below Average - Slightly Bullish"
        elif percentile > 80:
            level_signal = "Very High - Bearish"
        elif percentile > 60:
            level_signal = "Above Average - Slightly Bearish"
        else:
            level_signal = "Normal Range"

        return {
            "current_level": round(current_level, 1),
            "change": round(change, 1),
            "expectation": round(expectation, 1),
            "surprise": round(surprise, 1),
            "surprise_signal": surprise_signal,
            "percentile": round(percentile, 1),
            "level_signal": level_signal,
            "zscore": round(zscore, 2),
            "vs_5yr_avg": round(current_level - stats["mean"], 1),
        }

    def analyze_cushing_stocks(
        self,
        current_level: float,
        tank_capacity: float = 76.0
    ) -> dict:
        """
        Analyze Cushing storage levels.

        Args:
            current_level: Current stocks (MMbbl)
            tank_capacity: Total tank capacity

        Returns:
            Cushing analysis
        """
        utilization = (current_level / tank_capacity) * 100

        # Critical levels
        if utilization < 25:
            status = "Critical Low - Strong Backwardation Signal"
            risk_level = "High"
        elif utilization < 40:
            status = "Low - Supportive of Backwardation"
            risk_level = "Medium"
        elif utilization > 80:
            status = "Near Capacity - Contango/Storage Play"
            risk_level = "High"
        elif utilization > 60:
            status = "Elevated - Watch for Builds"
            risk_level = "Medium"
        else:
            status = "Normal Operating Range"
            risk_level = "Low"

        # Days of supply estimate (assuming ~400k bpd throughput)
        throughput = 0.4  # MMbbl/day
        days_of_supply = current_level / throughput

        return {
            "current_level": round(current_level, 1),
            "tank_capacity": tank_capacity,
            "utilization_pct": round(utilization, 1),
            "status": status,
            "risk_level": risk_level,
            "available_capacity": round(tank_capacity - current_level, 1),
            "days_of_supply": round(days_of_supply, 1),
        }

    def analyze_opec_compliance(
        self,
        production_data: pd.DataFrame
    ) -> dict:
        """
        Analyze OPEC+ production compliance.

        Args:
            production_data: DataFrame with quota and actual production

        Returns:
            OPEC compliance analysis
        """
        if production_data.empty:
            return {"overall_compliance": 0, "analysis": "No data"}

        total_quota = production_data["quota_mbpd"].sum()
        total_actual = production_data["actual_mbpd"].sum()

        overall_compliance = (1 - (total_actual - total_quota) / total_quota) * 100
        overall_compliance = max(0, min(150, overall_compliance))

        # Identify over/under producers
        production_data["deviation"] = production_data["actual_mbpd"] - production_data["quota_mbpd"]
        over_producers = production_data[production_data["deviation"] > 0.1]
        under_producers = production_data[production_data["deviation"] < -0.1]

        # Market impact assessment
        total_deviation = total_actual - total_quota
        if total_deviation > 0.5:
            market_impact = "Bearish - Significant Overproduction"
        elif total_deviation > 0.2:
            market_impact = "Slightly Bearish - Mild Overproduction"
        elif total_deviation < -0.5:
            market_impact = "Bullish - Significant Underproduction"
        elif total_deviation < -0.2:
            market_impact = "Slightly Bullish - Mild Underproduction"
        else:
            market_impact = "Neutral - Good Compliance"

        return {
            "overall_compliance_pct": round(overall_compliance, 1),
            "total_quota_mbpd": round(total_quota, 2),
            "total_actual_mbpd": round(total_actual, 2),
            "deviation_mbpd": round(total_deviation, 2),
            "over_producers": over_producers["country"].tolist() if not over_producers.empty else [],
            "under_producers": under_producers["country"].tolist() if not under_producers.empty else [],
            "market_impact": market_impact,
        }

    def analyze_turnaround_impact(
        self,
        turnaround_data: pd.DataFrame,
        analysis_date: datetime = None
    ) -> dict:
        """
        Analyze refinery turnaround impact on demand.

        Args:
            turnaround_data: DataFrame with turnaround schedules
            analysis_date: Date for analysis (default: now)

        Returns:
            Turnaround impact analysis
        """
        if analysis_date is None:
            analysis_date = datetime.now()

        if turnaround_data.empty:
            return {"current_offline": 0, "impact": "None"}

        # Ensure datetime columns
        turnaround_data["start_date"] = pd.to_datetime(turnaround_data["start_date"])
        turnaround_data["end_date"] = pd.to_datetime(turnaround_data["end_date"])

        # Current offline capacity
        current_offline = turnaround_data[
            (turnaround_data["start_date"] <= analysis_date) &
            (turnaround_data["end_date"] >= analysis_date)
        ]["capacity_kbpd"].sum()

        # Upcoming (next 30 days)
        upcoming_30d = turnaround_data[
            (turnaround_data["start_date"] > analysis_date) &
            (turnaround_data["start_date"] <= analysis_date + timedelta(days=30))
        ]
        upcoming_capacity = upcoming_30d["capacity_kbpd"].sum()

        # Regional breakdown
        regional = turnaround_data[
            (turnaround_data["start_date"] <= analysis_date + timedelta(days=30)) &
            (turnaround_data["end_date"] >= analysis_date)
        ].groupby("region")["capacity_kbpd"].sum().to_dict()

        # Impact assessment
        total_offline = current_offline + upcoming_capacity
        if total_offline > 2000:
            impact = "Very High - Significant crude demand reduction"
        elif total_offline > 1000:
            impact = "High - Notable crude demand impact"
        elif total_offline > 500:
            impact = "Moderate - Some crude demand impact"
        else:
            impact = "Low - Minimal market impact"

        return {
            "current_offline_kbpd": round(current_offline, 0),
            "upcoming_30d_kbpd": round(upcoming_capacity, 0),
            "total_impact_kbpd": round(total_offline, 0),
            "regional_breakdown": regional,
            "impact_assessment": impact,
            "num_turnarounds": len(turnaround_data[
                (turnaround_data["start_date"] <= analysis_date + timedelta(days=30)) &
                (turnaround_data["end_date"] >= analysis_date)
            ]),
        }

    def calculate_days_of_supply(
        self,
        inventory: float,
        demand: float
    ) -> dict:
        """
        Calculate days of supply.

        Args:
            inventory: Current inventory (MMbbl)
            demand: Daily demand (MMbbl/day)

        Returns:
            Days of supply metrics
        """
        if demand <= 0:
            return {"days_of_supply": float("inf")}

        days = inventory / demand

        # Assessment
        if days < 20:
            assessment = "Critical Low - Supply Concerns"
        elif days < 25:
            assessment = "Low - Below Normal"
        elif days > 35:
            assessment = "High - Ample Supply"
        elif days > 30:
            assessment = "Elevated - Above Normal"
        else:
            assessment = "Normal Range"

        return {
            "days_of_supply": round(days, 1),
            "inventory_mmb": round(inventory, 1),
            "demand_mmbd": round(demand, 2),
            "assessment": assessment,
        }

    def generate_fundamental_summary(
        self,
        inventory_data: dict,
        opec_data: pd.DataFrame,
        turnaround_data: pd.DataFrame
    ) -> dict:
        """
        Generate comprehensive fundamental summary.

        Args:
            inventory_data: Inventory metrics
            opec_data: OPEC production data
            turnaround_data: Turnaround schedule

        Returns:
            Complete fundamental summary
        """
        # Inventory analysis
        inv_analysis = self.analyze_inventory(
            current_level=inventory_data.get("level", 430),
            change=inventory_data.get("change", -1.5),
            expectation=inventory_data.get("expectation", -1.0),
        )

        # OPEC analysis
        opec_analysis = self.analyze_opec_compliance(opec_data)

        # Turnaround analysis
        turnaround_analysis = self.analyze_turnaround_impact(turnaround_data)

        # Overall assessment
        signals = []
        if "Bullish" in inv_analysis["surprise_signal"]:
            signals.append(1)
        elif "Bearish" in inv_analysis["surprise_signal"]:
            signals.append(-1)
        else:
            signals.append(0)

        if "Bullish" in opec_analysis["market_impact"]:
            signals.append(1)
        elif "Bearish" in opec_analysis["market_impact"]:
            signals.append(-1)
        else:
            signals.append(0)

        avg_signal = np.mean(signals)
        if avg_signal > 0.3:
            overall = "Net Bullish Fundamentals"
        elif avg_signal < -0.3:
            overall = "Net Bearish Fundamentals"
        else:
            overall = "Mixed Fundamentals"

        return {
            "inventory": inv_analysis,
            "opec": opec_analysis,
            "turnarounds": turnaround_analysis,
            "overall_assessment": overall,
            "timestamp": datetime.now().isoformat(),
        }
