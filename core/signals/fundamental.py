"""
Fundamental Signal Generation
=============================
Trading signals based on oil market fundamentals.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional


class FundamentalSignals:
    """
    Fundamental signal generation for oil trading.
    
    Signal Types:
    - Inventory surprise signals
    - OPEC compliance signals
    - Refinery turnaround signals
    - Supply/demand balance signals
    - Term structure signals
    """
    
    def __init__(self):
        """Initialize fundamental signal generator."""
        pass
    
    def inventory_surprise_signal(
        self,
        actual_change: float,
        expected_change: float,
        current_level: float,
        historical_mean: float = 440
    ) -> Dict:
        """
        Generate signal from inventory surprise.
        
        Args:
            actual_change: Actual inventory change
            expected_change: Expected change
            current_level: Current inventory level
            historical_mean: Historical mean level
            
        Returns:
            Signal dictionary
        """
        surprise = actual_change - expected_change
        level_vs_avg = current_level - historical_mean
        
        # Surprise signal
        if surprise < -3:
            surprise_signal = "STRONG_BUY"
            surprise_confidence = min(abs(surprise) * 15, 90)
        elif surprise < -1:
            surprise_signal = "BUY"
            surprise_confidence = min(abs(surprise) * 20, 70)
        elif surprise > 3:
            surprise_signal = "STRONG_SELL"
            surprise_confidence = min(abs(surprise) * 15, 90)
        elif surprise > 1:
            surprise_signal = "SELL"
            surprise_confidence = min(abs(surprise) * 20, 70)
        else:
            surprise_signal = "NEUTRAL"
            surprise_confidence = 50
        
        # Level signal
        if level_vs_avg < -30:
            level_signal = "BUY"
            level_confidence = 70
        elif level_vs_avg < -15:
            level_signal = "MILD_BUY"
            level_confidence = 55
        elif level_vs_avg > 30:
            level_signal = "SELL"
            level_confidence = 70
        elif level_vs_avg > 15:
            level_signal = "MILD_SELL"
            level_confidence = 55
        else:
            level_signal = "NEUTRAL"
            level_confidence = 50
        
        # Combine signals
        signal_scores = {"STRONG_BUY": 2, "BUY": 1, "MILD_BUY": 0.5, 
                        "STRONG_SELL": -2, "SELL": -1, "MILD_SELL": -0.5, "NEUTRAL": 0}
        
        combined_score = (
            signal_scores[surprise_signal] * 0.7 +
            signal_scores[level_signal] * 0.3
        )
        
        if combined_score > 0.8:
            final_signal = "LONG"
        elif combined_score < -0.8:
            final_signal = "SHORT"
        else:
            final_signal = "NEUTRAL"
        
        return {
            "signal": final_signal,
            "confidence": round((surprise_confidence * 0.7 + level_confidence * 0.3), 1),
            "surprise": round(surprise, 2),
            "surprise_signal": surprise_signal,
            "level_vs_avg": round(level_vs_avg, 1),
            "level_signal": level_signal,
            "actual_change": round(actual_change, 2),
            "expected_change": round(expected_change, 2),
            "current_level": round(current_level, 1),
        }
    
    def opec_compliance_signal(
        self,
        overall_compliance: float,
        production_deviation: float
    ) -> Dict:
        """
        Generate signal from OPEC compliance data.
        
        Args:
            overall_compliance: Compliance percentage
            production_deviation: Production vs quota (mbpd)
            
        Returns:
            Signal dictionary
        """
        # Compliance signal
        if overall_compliance > 105:
            # Over-compliance (bullish)
            signal = "LONG"
            confidence = min((overall_compliance - 100) * 10, 80)
            description = "OPEC+ over-complying with cuts"
        elif overall_compliance < 90:
            # Significant non-compliance (bearish)
            signal = "SHORT"
            confidence = min((100 - overall_compliance) * 5, 80)
            description = "Significant OPEC+ non-compliance"
        elif overall_compliance < 95:
            # Mild non-compliance
            signal = "MILD_SHORT"
            confidence = 55
            description = "Some OPEC+ quota breaches"
        else:
            signal = "NEUTRAL"
            confidence = 50
            description = "OPEC+ compliance near target"
        
        # Production deviation impact
        if production_deviation < -0.5:
            signal = "LONG" if signal != "SHORT" else "NEUTRAL"
            confidence = max(confidence, 65)
        elif production_deviation > 0.5:
            signal = "SHORT" if signal != "LONG" else "NEUTRAL"
            confidence = max(confidence, 65)
        
        return {
            "signal": signal,
            "confidence": round(confidence, 1),
            "compliance_pct": round(overall_compliance, 1),
            "deviation_mbpd": round(production_deviation, 2),
            "description": description,
        }
    
    def term_structure_signal(
        self,
        m1_m2_spread: float,
        m1_m12_spread: float,
        curve_slope: float
    ) -> Dict:
        """
        Generate signal from term structure.
        
        Args:
            m1_m2_spread: Front month - second month spread
            m1_m12_spread: Front month - 12th month spread
            curve_slope: Overall curve slope
            
        Returns:
            Signal dictionary
        """
        # Backwardation is typically bullish (tight market)
        # Contango is typically bearish (oversupplied)
        
        if curve_slope < -0.15:
            structure_signal = "LONG"
            confidence = min(abs(curve_slope) * 300, 85)
            description = "Strong backwardation - tight market"
        elif curve_slope < -0.05:
            structure_signal = "MILD_LONG"
            confidence = 60
            description = "Moderate backwardation"
        elif curve_slope > 0.15:
            structure_signal = "SHORT"
            confidence = min(curve_slope * 300, 85)
            description = "Strong contango - oversupplied market"
        elif curve_slope > 0.05:
            structure_signal = "MILD_SHORT"
            confidence = 60
            description = "Moderate contango"
        else:
            structure_signal = "NEUTRAL"
            confidence = 50
            description = "Flat curve - balanced market"
        
        # Calendar spread momentum
        if m1_m2_spread > 0.5:
            spread_signal = "LONG"
        elif m1_m2_spread < -0.5:
            spread_signal = "SHORT"
        else:
            spread_signal = "NEUTRAL"
        
        # Combine
        final_signal = structure_signal.replace("MILD_", "")
        
        return {
            "signal": final_signal,
            "confidence": round(confidence, 1),
            "m1_m2_spread": round(m1_m2_spread, 2),
            "m1_m12_spread": round(m1_m12_spread, 2),
            "curve_slope": round(curve_slope, 4),
            "spread_signal": spread_signal,
            "description": description,
        }
    
    def crack_spread_signal(
        self,
        current_crack: float,
        historical_mean: float = 25,
        historical_std: float = 8
    ) -> Dict:
        """
        Generate signal from crack spread levels.
        
        Args:
            current_crack: Current crack spread
            historical_mean: Historical mean
            historical_std: Historical standard deviation
            
        Returns:
            Signal dictionary
        """
        zscore = (current_crack - historical_mean) / historical_std
        percentile = 50 + 34.13 * np.sign(zscore) * (1 - np.exp(-abs(zscore)))
        
        # High crack spreads generally supportive of crude (refiners buying)
        if zscore > 1.5:
            signal = "LONG"
            confidence = min(zscore * 30, 80)
            description = "Very high refining margins - strong crude demand"
        elif zscore > 0.5:
            signal = "MILD_LONG"
            confidence = 60
            description = "Above average refining margins"
        elif zscore < -1.5:
            signal = "SHORT"
            confidence = min(abs(zscore) * 30, 80)
            description = "Very low refining margins - weak crude demand"
        elif zscore < -0.5:
            signal = "MILD_SHORT"
            confidence = 60
            description = "Below average refining margins"
        else:
            signal = "NEUTRAL"
            confidence = 50
            description = "Normal refining margins"
        
        return {
            "signal": signal.replace("MILD_", ""),
            "confidence": round(confidence, 1),
            "current_crack": round(current_crack, 2),
            "zscore": round(zscore, 2),
            "percentile": round(percentile, 1),
            "description": description,
        }
    
    def turnaround_signal(
        self,
        offline_capacity_kbpd: float,
        upcoming_capacity_kbpd: float
    ) -> Dict:
        """
        Generate signal from refinery turnaround data.
        
        Args:
            offline_capacity_kbpd: Current offline capacity
            upcoming_capacity_kbpd: Capacity going offline in next 30 days
            
        Returns:
            Signal dictionary
        """
        total_impact = offline_capacity_kbpd + upcoming_capacity_kbpd
        
        # High turnaround activity reduces crude demand (bearish)
        if total_impact > 2000:
            signal = "SHORT"
            confidence = 70
            description = "Heavy turnaround season - reduced crude demand"
        elif total_impact > 1000:
            signal = "MILD_SHORT"
            confidence = 60
            description = "Moderate turnaround activity"
        elif total_impact < 300:
            signal = "LONG"
            confidence = 60
            description = "Light turnarounds - strong refinery demand"
        else:
            signal = "NEUTRAL"
            confidence = 50
            description = "Normal turnaround levels"
        
        return {
            "signal": signal.replace("MILD_", ""),
            "confidence": round(confidence, 1),
            "offline_kbpd": round(offline_capacity_kbpd, 0),
            "upcoming_kbpd": round(upcoming_capacity_kbpd, 0),
            "total_impact_kbpd": round(total_impact, 0),
            "description": description,
        }
    
    def generate_composite_fundamental_signal(
        self,
        inventory_data: Dict,
        opec_data: Dict,
        curve_data: Dict,
        crack_spread: float,
        turnaround_data: Dict
    ) -> Dict:
        """
        Generate composite fundamental signal.
        
        Args:
            inventory_data: Inventory metrics
            opec_data: OPEC compliance data
            curve_data: Curve/term structure data
            crack_spread: Current crack spread
            turnaround_data: Turnaround impact data
            
        Returns:
            Composite signal dictionary
        """
        # Generate individual signals
        inv_signal = self.inventory_surprise_signal(
            actual_change=inventory_data.get("change", 0),
            expected_change=inventory_data.get("expectation", 0),
            current_level=inventory_data.get("level", 430)
        )
        
        opec_signal = self.opec_compliance_signal(
            overall_compliance=opec_data.get("compliance", 95),
            production_deviation=opec_data.get("deviation", 0)
        )
        
        term_signal = self.term_structure_signal(
            m1_m2_spread=curve_data.get("m1_m2_spread", 0),
            m1_m12_spread=curve_data.get("m1_m12_spread", 0),
            curve_slope=curve_data.get("slope", 0)
        )
        
        crack_signal = self.crack_spread_signal(crack_spread)
        
        turn_signal = self.turnaround_signal(
            offline_capacity_kbpd=turnaround_data.get("offline", 500),
            upcoming_capacity_kbpd=turnaround_data.get("upcoming", 500)
        )
        
        # Weight signals
        signal_scores = {"LONG": 1, "SHORT": -1, "NEUTRAL": 0}
        weights = {
            "inventory": 0.30,
            "opec": 0.20,
            "term_structure": 0.20,
            "crack": 0.15,
            "turnaround": 0.15,
        }
        
        total_score = (
            weights["inventory"] * signal_scores.get(inv_signal["signal"], 0) * (inv_signal["confidence"] / 100) +
            weights["opec"] * signal_scores.get(opec_signal["signal"], 0) * (opec_signal["confidence"] / 100) +
            weights["term_structure"] * signal_scores.get(term_signal["signal"], 0) * (term_signal["confidence"] / 100) +
            weights["crack"] * signal_scores.get(crack_signal["signal"], 0) * (crack_signal["confidence"] / 100) +
            weights["turnaround"] * signal_scores.get(turn_signal["signal"], 0) * (turn_signal["confidence"] / 100)
        )
        
        # Determine composite signal
        if total_score > 0.3:
            composite_signal = "LONG"
        elif total_score < -0.3:
            composite_signal = "SHORT"
        else:
            composite_signal = "NEUTRAL"
        
        return {
            "signal": composite_signal,
            "confidence": round(min(abs(total_score) * 100 + 20, 95), 1),
            "score": round(total_score, 3),
            "components": {
                "inventory": inv_signal,
                "opec": opec_signal,
                "term_structure": term_signal,
                "crack_spread": crack_signal,
                "turnaround": turn_signal,
            },
            "timestamp": datetime.now().isoformat(),
        }
