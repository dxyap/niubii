"""
Spread Analysis
===============
Analysis of inter-commodity and crack spreads.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from scipy import stats


class SpreadAnalyzer:
    """
    Spread analysis for oil markets.
    
    Features:
    - WTI-Brent spread analysis
    - Crack spread calculations (3-2-1, 2-1-1, etc.)
    - Regional differentials
    - Historical percentile rankings
    """
    
    # Contract specifications for spread calculations
    BARREL_CONVERSION = {
        "CL": 1,      # WTI - already in $/bbl
        "CO": 1,      # Brent - already in $/bbl
        "XB": 42,     # RBOB - $/gal to $/bbl (42 gal/bbl)
        "HO": 42,     # Heating Oil - $/gal to $/bbl
        "QS": 7.45,   # Gasoil - $/tonne to $/bbl (approx)
    }
    
    def __init__(self):
        """Initialize spread analyzer."""
        pass
    
    def calculate_wti_brent_spread(
        self,
        wti_price: float,
        brent_price: float
    ) -> Dict:
        """
        Analyze WTI-Brent spread.
        
        Args:
            wti_price: WTI price
            brent_price: Brent price
            
        Returns:
            Spread analysis
        """
        spread = wti_price - brent_price
        spread_pct = (spread / brent_price) * 100
        
        # Historical context (typical range -8 to +2)
        if spread > 0:
            direction = "WTI Premium"
        elif spread < -5:
            direction = "Wide Brent Premium"
        else:
            direction = "Normal Brent Premium"
        
        return {
            "spread": round(spread, 2),
            "spread_pct": round(spread_pct, 2),
            "wti_price": round(wti_price, 2),
            "brent_price": round(brent_price, 2),
            "direction": direction,
        }
    
    def calculate_crack_spread(
        self,
        crude_price: float,
        gasoline_price: float,
        distillate_price: float,
        crack_type: str = "3-2-1"
    ) -> Dict:
        """
        Calculate crack spread.
        
        Args:
            crude_price: Crude oil price ($/bbl)
            gasoline_price: Gasoline price ($/gal or $/bbl)
            distillate_price: Distillate price ($/gal or $/bbl)
            crack_type: Type of crack spread
            
        Returns:
            Crack spread analysis
        """
        # Convert to $/bbl if needed (assuming inputs are $/gal for products)
        if gasoline_price < 10:  # Likely in $/gal
            gasoline_bbl = gasoline_price * 42
        else:
            gasoline_bbl = gasoline_price
            
        if distillate_price < 10:
            distillate_bbl = distillate_price * 42
        else:
            distillate_bbl = distillate_price
        
        # Calculate crack based on type
        if crack_type == "3-2-1":
            # 3-2-1: 3 barrels crude -> 2 barrels gasoline + 1 barrel distillate
            crack = (2 * gasoline_bbl + distillate_bbl - 3 * crude_price) / 3
            description = "3-2-1 (2 gas + 1 dist - 3 crude)"
            
        elif crack_type == "2-1-1":
            # 2-1-1: 2 barrels crude -> 1 barrel gasoline + 1 barrel distillate
            crack = (gasoline_bbl + distillate_bbl - 2 * crude_price) / 2
            description = "2-1-1 (1 gas + 1 dist - 2 crude)"
            
        elif crack_type == "gasoline":
            # Simple gasoline crack
            crack = gasoline_bbl - crude_price
            description = "Gasoline Crack (gas - crude)"
            
        elif crack_type == "heating_oil":
            # Simple heating oil crack
            crack = distillate_bbl - crude_price
            description = "Heating Oil Crack (HO - crude)"
            
        else:
            crack = (2 * gasoline_bbl + distillate_bbl - 3 * crude_price) / 3
            description = "3-2-1 Default"
        
        # Refining margin percentage
        margin_pct = (crack / crude_price) * 100
        
        return {
            "crack_spread": round(crack, 2),
            "crack_type": crack_type,
            "description": description,
            "crude_price": round(crude_price, 2),
            "gasoline_bbl": round(gasoline_bbl, 2),
            "distillate_bbl": round(distillate_bbl, 2),
            "margin_pct": round(margin_pct, 2),
        }
    
    def analyze_regional_differentials(
        self,
        prices: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Analyze regional price differentials.
        
        Args:
            prices: Dictionary of regional prices
            
        Returns:
            DataFrame of differentials
        """
        # Default benchmark is Brent
        benchmark = prices.get("Brent", prices.get("brent", 77.0))
        
        differentials = []
        
        # Common differentials
        diff_pairs = [
            ("WTI", "Brent", "WTI-Brent"),
            ("Dubai", "Brent", "Dubai-Brent"),
            ("WCS", "WTI", "WCS-WTI"),
            ("Mars", "WTI", "Mars-WTI"),
            ("LLS", "WTI", "LLS-WTI"),
        ]
        
        for grade1, grade2, name in diff_pairs:
            price1 = prices.get(grade1, prices.get(grade1.lower()))
            price2 = prices.get(grade2, prices.get(grade2.lower()))
            
            if price1 is not None and price2 is not None:
                diff = price1 - price2
                differentials.append({
                    "differential": name,
                    "grade_1": grade1,
                    "grade_2": grade2,
                    "price_1": round(price1, 2),
                    "price_2": round(price2, 2),
                    "spread": round(diff, 2),
                })
        
        return pd.DataFrame(differentials)
    
    def calculate_spread_zscore(
        self,
        current_spread: float,
        historical_spreads: pd.Series,
        lookback: int = 60
    ) -> Dict:
        """
        Calculate z-score of spread vs historical.
        
        Args:
            current_spread: Current spread value
            historical_spreads: Historical spread values
            lookback: Lookback period in days
            
        Returns:
            Z-score analysis
        """
        if len(historical_spreads) < lookback:
            lookback = len(historical_spreads)
        
        recent = historical_spreads.tail(lookback)
        mean = recent.mean()
        std = recent.std()
        
        if std == 0:
            zscore = 0
        else:
            zscore = (current_spread - mean) / std
        
        # Signal interpretation
        if zscore > 2:
            signal = "Extremely Wide - Potential Short"
        elif zscore > 1:
            signal = "Wide - Watch for Mean Reversion"
        elif zscore < -2:
            signal = "Extremely Narrow - Potential Long"
        elif zscore < -1:
            signal = "Narrow - Watch for Mean Reversion"
        else:
            signal = "Normal Range"
        
        return {
            "current": round(current_spread, 2),
            "mean": round(mean, 2),
            "std": round(std, 2),
            "zscore": round(zscore, 2),
            "percentile": round(stats.percentileofscore(recent, current_spread), 1),
            "signal": signal,
            "lookback_days": lookback,
        }
    
    def get_seasonal_pattern(
        self,
        spread_name: str,
        historical_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extract seasonal pattern from spread history.
        
        Args:
            spread_name: Name of spread
            historical_data: Historical spread data with date index
            
        Returns:
            DataFrame with seasonal averages by month
        """
        if historical_data.empty:
            return pd.DataFrame()
        
        # Add month column
        df = historical_data.copy()
        df["month"] = df.index.month
        
        # Calculate monthly statistics
        seasonal = df.groupby("month").agg({
            "spread": ["mean", "std", "min", "max", "median"]
        }).round(2)
        
        seasonal.columns = ["mean", "std", "min", "max", "median"]
        seasonal["month_name"] = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
        ]
        
        return seasonal
    
    def generate_spread_summary(
        self,
        wti: float,
        brent: float,
        rbob: float,
        ho: float
    ) -> Dict:
        """
        Generate comprehensive spread summary.
        
        Args:
            wti: WTI price
            brent: Brent price
            rbob: RBOB price ($/gal)
            ho: Heating oil price ($/gal)
            
        Returns:
            Complete spread summary
        """
        wti_brent = self.calculate_wti_brent_spread(wti, brent)
        crack_321 = self.calculate_crack_spread(wti, rbob, ho, "3-2-1")
        gas_crack = self.calculate_crack_spread(wti, rbob, ho, "gasoline")
        ho_crack = self.calculate_crack_spread(wti, rbob, ho, "heating_oil")
        
        return {
            "wti_brent": wti_brent,
            "crack_321": crack_321,
            "gasoline_crack": gas_crack,
            "heating_oil_crack": ho_crack,
            "timestamp": datetime.now().isoformat(),
        }
