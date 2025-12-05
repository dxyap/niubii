"""
Positioning Data
=================
COT reports and speculator positioning analysis.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import random

logger = logging.getLogger(__name__)


@dataclass
class PositioningConfig:
    """Configuration for positioning data."""
    provider: str = "mock"  # cftc, ice, bloomberg
    api_key: Optional[str] = None
    commodities: List[str] = field(default_factory=lambda: [
        "WTI", "Brent", "RBOB", "Heating_Oil", "Natural_Gas"
    ])
    refresh_hours: int = 168  # Weekly COT reports


@dataclass
class COTData:
    """Commitment of Traders data."""
    commodity: str
    report_date: datetime
    # Commercial positions (hedgers)
    commercial_long: int
    commercial_short: int
    commercial_net: int
    # Non-commercial (speculators/managed money)
    non_commercial_long: int
    non_commercial_short: int
    non_commercial_net: int
    # Spreads
    non_commercial_spreads: int
    # Open interest
    open_interest: int
    # Change from previous week
    commercial_net_change: int
    non_commercial_net_change: int
    oi_change: int
    
    def to_dict(self) -> Dict:
        return {
            "commodity": self.commodity,
            "report_date": self.report_date.isoformat(),
            "commercial_long": self.commercial_long,
            "commercial_short": self.commercial_short,
            "commercial_net": self.commercial_net,
            "non_commercial_long": self.non_commercial_long,
            "non_commercial_short": self.non_commercial_short,
            "non_commercial_net": self.non_commercial_net,
            "non_commercial_spreads": self.non_commercial_spreads,
            "open_interest": self.open_interest,
            "commercial_net_change": self.commercial_net_change,
            "non_commercial_net_change": self.non_commercial_net_change,
            "oi_change": self.oi_change,
        }


class PositioningData:
    """
    Positioning data provider.
    
    Monitors COT reports and speculator positions.
    """
    
    # Base positioning values for each commodity
    BASE_POSITIONS = {
        "WTI": {
            "open_interest": 2200000,  # Contracts
            "commercial_net_typical": -150000,  # Net short (hedgers)
            "spec_net_range": (-50000, 300000),  # Speculators range
        },
        "Brent": {
            "open_interest": 1800000,
            "commercial_net_typical": -120000,
            "spec_net_range": (-40000, 250000),
        },
        "RBOB": {
            "open_interest": 350000,
            "commercial_net_typical": -30000,
            "spec_net_range": (-20000, 80000),
        },
        "Heating_Oil": {
            "open_interest": 400000,
            "commercial_net_typical": -35000,
            "spec_net_range": (-25000, 90000),
        },
        "Natural_Gas": {
            "open_interest": 1500000,
            "commercial_net_typical": -100000,
            "spec_net_range": (-150000, 150000),
        },
    }
    
    # Extremes for percentile calculations
    HISTORICAL_EXTREMES = {
        "WTI": {"spec_net_min": -100000, "spec_net_max": 450000},
        "Brent": {"spec_net_min": -80000, "spec_net_max": 350000},
        "RBOB": {"spec_net_min": -40000, "spec_net_max": 120000},
        "Heating_Oil": {"spec_net_min": -45000, "spec_net_max": 130000},
        "Natural_Gas": {"spec_net_min": -250000, "spec_net_max": 200000},
    }
    
    def __init__(self, config: Optional[PositioningConfig] = None):
        self.config = config or PositioningConfig()
        
        # Data cache
        self._cot_data: Dict[str, COTData] = {}
        self._last_update: Optional[datetime] = None
    
    def get_latest_cot(
        self,
        commodity: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get latest COT report data.
        
        Args:
            commodity: Specific commodity to query
            
        Returns:
            COT data
        """
        if self.config.provider == "mock":
            return self._generate_mock_cot(commodity)
        
        return self._generate_mock_cot(commodity)
    
    def get_positioning_history(
        self,
        commodity: str,
        weeks: int = 52,
    ) -> List[Dict]:
        """
        Get historical positioning data.
        
        Args:
            commodity: Commodity to analyze
            weeks: Number of weeks of history
            
        Returns:
            Historical COT data
        """
        if commodity not in self.BASE_POSITIONS:
            return []
        
        base = self.BASE_POSITIONS[commodity]
        history = []
        
        # Random walk for spec net position
        spec_net = random.uniform(*base["spec_net_range"])
        
        for i in range(weeks):
            date = datetime.now() - timedelta(weeks=weeks - i)
            
            # Random walk
            change = random.uniform(-15000, 15000)
            spec_net = max(
                base["spec_net_range"][0],
                min(base["spec_net_range"][1], spec_net + change)
            )
            
            history.append({
                "date": date.isoformat(),
                "non_commercial_net": round(spec_net),
                "open_interest": round(base["open_interest"] * random.uniform(0.9, 1.1)),
            })
        
        return history
    
    def get_managed_money_positions(self) -> Dict[str, Any]:
        """
        Get managed money (hedge fund) positions.
        
        Returns:
            Managed money positioning data
        """
        positions = {}
        
        for commodity in self.config.commodities:
            if commodity not in self.BASE_POSITIONS:
                continue
            
            base = self.BASE_POSITIONS[commodity]
            spec_net = random.uniform(*base["spec_net_range"])
            
            # Calculate percentile
            extremes = self.HISTORICAL_EXTREMES.get(commodity, {"spec_net_min": -100000, "spec_net_max": 300000})
            percentile = (spec_net - extremes["spec_net_min"]) / (extremes["spec_net_max"] - extremes["spec_net_min"]) * 100
            percentile = max(0, min(100, percentile))
            
            positions[commodity] = {
                "net_contracts": round(spec_net),
                "percentile": round(percentile, 1),
                "stance": "extremely_long" if percentile > 80 else (
                    "long" if percentile > 60 else (
                        "neutral" if percentile > 40 else (
                            "short" if percentile > 20 else "extremely_short"
                        )
                    )
                ),
                "week_change": round(random.uniform(-20000, 20000)),
            }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "report_date": (datetime.now() - timedelta(days=datetime.now().weekday() + 3)).strftime("%Y-%m-%d"),
            "positions": positions,
        }
    
    def get_commercial_hedging(self) -> Dict[str, Any]:
        """
        Get commercial hedging positions.
        
        Returns:
            Commercial hedging data
        """
        hedging = {}
        
        for commodity in self.config.commodities:
            if commodity not in self.BASE_POSITIONS:
                continue
            
            base = self.BASE_POSITIONS[commodity]
            commercial_net = base["commercial_net_typical"] * random.uniform(0.7, 1.3)
            
            hedging[commodity] = {
                "net_contracts": round(commercial_net),
                "hedging_ratio": round(abs(commercial_net) / base["open_interest"] * 100, 1),
                "week_change": round(random.uniform(-10000, 10000)),
                "vs_3m_avg": round(random.uniform(-20, 20), 1),
            }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "hedging": hedging,
        }
    
    def get_open_interest_analysis(self) -> Dict[str, Any]:
        """
        Analyze open interest trends.
        
        Returns:
            Open interest analysis
        """
        analysis = {}
        
        for commodity in self.config.commodities:
            if commodity not in self.BASE_POSITIONS:
                continue
            
            base = self.BASE_POSITIONS[commodity]
            current_oi = base["open_interest"] * random.uniform(0.9, 1.1)
            
            analysis[commodity] = {
                "open_interest": round(current_oi),
                "week_change": round(random.uniform(-50000, 50000)),
                "week_change_pct": round(random.uniform(-5, 5), 1),
                "vs_year_avg_pct": round(random.uniform(-15, 15), 1),
                "trend": random.choice(["increasing", "decreasing", "stable"]),
            }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "analysis": analysis,
        }
    
    def calculate_positioning_signal(self) -> Dict[str, Any]:
        """
        Calculate trading signal from positioning data.
        
        Extreme positioning can be contrarian indicators.
        
        Returns:
            Signal based on positioning
        """
        managed_money = self.get_managed_money_positions()
        positions = managed_money.get("positions", {})
        
        # Focus on WTI and Brent
        oil_positions = [
            positions.get(c) for c in ["WTI", "Brent"]
            if c in positions
        ]
        
        if not oil_positions:
            return {
                "signal": "neutral",
                "confidence": 30,
                "rationale": "Insufficient positioning data",
                "timestamp": datetime.now().isoformat(),
            }
        
        avg_percentile = sum(p["percentile"] for p in oil_positions) / len(oil_positions)
        
        # Contrarian signals at extremes
        if avg_percentile > 85:
            signal = "bearish"  # Crowded long = contrarian short
            confidence = min(80, avg_percentile - 20)
            rationale = "Extreme long positioning suggests crowded trade, potential for reversal"
        elif avg_percentile < 15:
            signal = "bullish"  # Crowded short = contrarian long
            confidence = min(80, 40 - avg_percentile)
            rationale = "Extreme short positioning suggests capitulation, potential for reversal"
        elif avg_percentile > 70:
            signal = "cautious_bullish"
            confidence = 55
            rationale = "Elevated long positioning, momentum intact but watch for crowding"
        elif avg_percentile < 30:
            signal = "cautious_bearish"
            confidence = 55
            rationale = "Depressed positioning, sentiment weak but potential for washout"
        else:
            signal = "neutral"
            confidence = 50
            rationale = "Positioning within normal range, no contrarian signal"
        
        return {
            "signal": signal,
            "confidence": round(confidence),
            "rationale": rationale,
            "avg_percentile": round(avg_percentile, 1),
            "timestamp": datetime.now().isoformat(),
        }
    
    def get_positioning_extremes(self) -> Dict[str, Any]:
        """
        Identify commodities at positioning extremes.
        
        Returns:
            Commodities at extreme positioning levels
        """
        managed_money = self.get_managed_money_positions()
        positions = managed_money.get("positions", {})
        
        extremes = {
            "extremely_long": [],
            "extremely_short": [],
        }
        
        for commodity, data in positions.items():
            if data["percentile"] > 80:
                extremes["extremely_long"].append({
                    "commodity": commodity,
                    "percentile": data["percentile"],
                    "net_contracts": data["net_contracts"],
                })
            elif data["percentile"] < 20:
                extremes["extremely_short"].append({
                    "commodity": commodity,
                    "percentile": data["percentile"],
                    "net_contracts": data["net_contracts"],
                })
        
        return {
            "timestamp": datetime.now().isoformat(),
            "extremes": extremes,
            "has_extremes": bool(extremes["extremely_long"] or extremes["extremely_short"]),
        }
    
    def _generate_mock_cot(self, commodity: Optional[str] = None) -> Dict[str, Any]:
        """Generate mock COT data."""
        now = datetime.now()
        # COT reports are released on Fridays for Tuesday's data
        report_date = now - timedelta(days=now.weekday() + 3)
        
        commodities_to_fetch = (
            [commodity] if commodity and commodity in self.BASE_POSITIONS
            else list(self.BASE_POSITIONS.keys())
        )
        
        cot_data = {}
        
        for comm in commodities_to_fetch:
            base = self.BASE_POSITIONS[comm]
            
            # Generate positions
            oi = round(base["open_interest"] * random.uniform(0.9, 1.1))
            
            # Commercial (hedgers) - typically net short
            commercial_long = round(oi * random.uniform(0.15, 0.25))
            commercial_short = commercial_long + round(abs(base["commercial_net_typical"]) * random.uniform(0.8, 1.2))
            
            # Non-commercial (speculators)
            spec_net = random.uniform(*base["spec_net_range"])
            non_commercial_long = round(oi * random.uniform(0.2, 0.35))
            non_commercial_short = non_commercial_long - round(spec_net)
            non_commercial_short = max(0, non_commercial_short)
            
            spreads = round(oi * random.uniform(0.1, 0.2))
            
            cot_data[comm] = {
                "report_date": report_date.strftime("%Y-%m-%d"),
                "open_interest": oi,
                "commercial": {
                    "long": commercial_long,
                    "short": commercial_short,
                    "net": commercial_long - commercial_short,
                    "net_change": round(random.uniform(-10000, 10000)),
                },
                "non_commercial": {
                    "long": non_commercial_long,
                    "short": non_commercial_short,
                    "net": non_commercial_long - non_commercial_short,
                    "net_change": round(random.uniform(-15000, 15000)),
                    "spreads": spreads,
                },
                "spec_percentile": round(
                    (spec_net - self.HISTORICAL_EXTREMES[comm]["spec_net_min"]) /
                    (self.HISTORICAL_EXTREMES[comm]["spec_net_max"] - self.HISTORICAL_EXTREMES[comm]["spec_net_min"]) * 100
                ),
            }
        
        return {
            "timestamp": now.isoformat(),
            "report_date": report_date.strftime("%Y-%m-%d"),
            "source": "positioning",
            "provider": self.config.provider,
            "data": cot_data,
        }
