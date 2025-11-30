"""
Term Structure / Curves Analysis
================================
Analysis of futures curves, roll yield, and term structure signals.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from scipy import stats


class CurveAnalyzer:
    """
    Futures curve analysis for oil markets.
    
    Features:
    - Curve shape classification (contango/backwardation)
    - Roll yield calculations
    - Calendar spread analysis
    - Historical percentile rankings
    """
    
    def __init__(self):
        """Initialize curve analyzer."""
        self._historical_slopes = []
    
    def analyze_curve(self, curve_df: pd.DataFrame) -> Dict:
        """
        Comprehensive curve analysis.
        
        Args:
            curve_df: DataFrame with columns [month, price, days_to_expiry]
            
        Returns:
            Dictionary of curve metrics
        """
        prices = curve_df["price"].values
        months = curve_df["month"].values
        
        # Curve shape
        front = prices[0]
        back = prices[-1]
        mid = prices[len(prices) // 2]
        
        # Calculate slopes
        slope_front = (prices[2] - prices[0]) / 2 if len(prices) > 2 else 0
        slope_back = (prices[-1] - prices[-3]) / 2 if len(prices) > 3 else 0
        overall_slope = (back - front) / (len(prices) - 1)
        
        # Curvature (second derivative)
        if len(prices) >= 3:
            curvature = np.mean(np.diff(np.diff(prices)))
        else:
            curvature = 0
        
        # Classify structure
        if overall_slope > 0.15:
            structure = "Strong Contango"
        elif overall_slope > 0.05:
            structure = "Contango"
        elif overall_slope < -0.15:
            structure = "Strong Backwardation"
        elif overall_slope < -0.05:
            structure = "Backwardation"
        else:
            structure = "Flat"
        
        # Roll yield (annualized)
        m1_m2_spread = prices[0] - prices[1] if len(prices) > 1 else 0
        days_to_roll = curve_df["days_to_expiry"].iloc[0] if "days_to_expiry" in curve_df else 30
        roll_yield_annual = (m1_m2_spread / prices[0]) * (365 / max(days_to_roll, 1)) * 100
        
        return {
            "structure": structure,
            "front_price": round(front, 2),
            "back_price": round(back, 2),
            "overall_slope": round(overall_slope, 4),
            "front_slope": round(slope_front, 4),
            "back_slope": round(slope_back, 4),
            "curvature": round(curvature, 4),
            "m1_m2_spread": round(m1_m2_spread, 2),
            "roll_yield_annual_pct": round(roll_yield_annual, 2),
            "num_months": len(prices),
        }
    
    def calculate_calendar_spreads(
        self,
        curve_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate all calendar spreads from curve.
        
        Args:
            curve_df: Curve data
            
        Returns:
            DataFrame of calendar spreads
        """
        spreads = []
        prices = curve_df["price"].values
        
        # Standard spreads: M1-M2, M1-M3, M1-M6, M1-M12
        standard_spreads = [(0, 1), (0, 2), (0, 5), (0, 11)]
        
        for front_idx, back_idx in standard_spreads:
            if back_idx < len(prices):
                spread_value = prices[front_idx] - prices[back_idx]
                spreads.append({
                    "spread_name": f"M{front_idx+1}-M{back_idx+1}",
                    "front_month": front_idx + 1,
                    "back_month": back_idx + 1,
                    "spread_value": round(spread_value, 2),
                    "front_price": round(prices[front_idx], 2),
                    "back_price": round(prices[back_idx], 2),
                })
        
        return pd.DataFrame(spreads)
    
    def calculate_roll_yield(
        self,
        curve_df: pd.DataFrame,
        holding_period_days: int = 30
    ) -> Dict:
        """
        Calculate expected roll yield for different holding periods.
        
        Args:
            curve_df: Curve data
            holding_period_days: Holding period in days
            
        Returns:
            Dictionary of roll yield metrics
        """
        if len(curve_df) < 2:
            return {"roll_yield": 0, "roll_cost": 0}
        
        front_price = curve_df["price"].iloc[0]
        second_price = curve_df["price"].iloc[1]
        
        # Simple roll yield
        roll_cost = second_price - front_price
        roll_yield = -roll_cost  # Positive when in backwardation
        
        # Annualized
        roll_yield_annual = roll_yield * (365 / holding_period_days)
        roll_yield_pct = (roll_yield / front_price) * 100
        roll_yield_annual_pct = roll_yield_pct * (365 / holding_period_days)
        
        return {
            "roll_cost": round(roll_cost, 2),
            "roll_yield": round(roll_yield, 2),
            "roll_yield_pct": round(roll_yield_pct, 2),
            "roll_yield_annual": round(roll_yield_annual, 2),
            "roll_yield_annual_pct": round(roll_yield_annual_pct, 2),
            "curve_carry": "Positive" if roll_yield > 0 else "Negative",
        }
    
    def get_curve_percentile(
        self,
        current_slope: float,
        historical_slopes: List[float]
    ) -> float:
        """
        Calculate percentile rank of current curve slope.
        
        Args:
            current_slope: Current curve slope
            historical_slopes: List of historical slopes
            
        Returns:
            Percentile rank (0-100)
        """
        if not historical_slopes:
            return 50.0
        
        percentile = stats.percentileofscore(historical_slopes, current_slope)
        return round(percentile, 1)
    
    def detect_curve_regime(
        self,
        curve_history: pd.DataFrame
    ) -> Dict:
        """
        Detect current curve regime and regime changes.
        
        Args:
            curve_history: Historical curve slopes
            
        Returns:
            Regime information
        """
        if curve_history.empty:
            return {"regime": "Unknown", "confidence": 0}
        
        slopes = curve_history["slope"].values
        
        # Moving averages for regime detection
        ma_20 = np.mean(slopes[-20:]) if len(slopes) >= 20 else np.mean(slopes)
        ma_60 = np.mean(slopes[-60:]) if len(slopes) >= 60 else np.mean(slopes)
        current = slopes[-1]
        
        # Regime classification
        if current > 0.1 and ma_20 > 0.05:
            regime = "Strong Contango"
            confidence = min(abs(current / 0.2) * 100, 100)
        elif current > 0.02:
            regime = "Contango"
            confidence = min(abs(current / 0.1) * 100, 100)
        elif current < -0.1 and ma_20 < -0.05:
            regime = "Strong Backwardation"
            confidence = min(abs(current / 0.2) * 100, 100)
        elif current < -0.02:
            regime = "Backwardation"
            confidence = min(abs(current / 0.1) * 100, 100)
        else:
            regime = "Transitional"
            confidence = 50
        
        # Detect regime change
        regime_change = False
        if len(slopes) > 5:
            prev_regime = "contango" if np.mean(slopes[-10:-5]) > 0.02 else "backwardation"
            curr_regime = "contango" if np.mean(slopes[-5:]) > 0.02 else "backwardation"
            regime_change = prev_regime != curr_regime
        
        return {
            "regime": regime,
            "confidence": round(confidence, 1),
            "current_slope": round(current, 4),
            "ma_20_slope": round(ma_20, 4),
            "ma_60_slope": round(ma_60, 4),
            "regime_change_detected": regime_change,
        }
    
    def generate_curve_chart_data(
        self,
        curve_df: pd.DataFrame,
        historical_curves: Optional[List[pd.DataFrame]] = None
    ) -> Dict:
        """
        Prepare data for curve visualization.
        
        Args:
            curve_df: Current curve data
            historical_curves: Optional list of historical curves for comparison
            
        Returns:
            Chart-ready data structure
        """
        chart_data = {
            "current": {
                "months": curve_df["month"].tolist(),
                "prices": curve_df["price"].tolist(),
                "label": "Current",
            },
            "historical": [],
        }
        
        if historical_curves:
            for i, hist_curve in enumerate(historical_curves[:5]):  # Max 5 historical
                chart_data["historical"].append({
                    "months": hist_curve["month"].tolist(),
                    "prices": hist_curve["price"].tolist(),
                    "label": f"Historical {i+1}",
                })
        
        return chart_data
