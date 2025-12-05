"""
Satellite Data
==============
Satellite imagery analysis for oil storage monitoring.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import random

logger = logging.getLogger(__name__)


@dataclass
class SatelliteConfig:
    """Configuration for satellite data."""
    provider: str = "mock"  # orbital_insight, kayrros, etc.
    api_key: Optional[str] = None
    locations: List[str] = field(default_factory=lambda: [
        "cushing_ok", "rotterdam", "singapore", "fujairah"
    ])
    refresh_hours: int = 24


@dataclass
class StorageTankData:
    """Storage tank observation from satellite."""
    location: str
    tank_id: Optional[str]
    observation_date: datetime
    utilization_pct: float
    capacity_barrels: int
    estimated_volume: int
    confidence: float
    source: str = "satellite"
    
    def to_dict(self) -> Dict:
        return {
            "location": self.location,
            "tank_id": self.tank_id,
            "observation_date": self.observation_date.isoformat(),
            "utilization_pct": self.utilization_pct,
            "capacity_barrels": self.capacity_barrels,
            "estimated_volume": self.estimated_volume,
            "confidence": self.confidence,
        }


class SatelliteData:
    """
    Satellite imagery data provider.
    
    Monitors oil storage tank levels using satellite imagery.
    """
    
    # Major storage locations and estimated capacities
    LOCATIONS = {
        "cushing_ok": {
            "name": "Cushing, Oklahoma",
            "capacity_mb": 76,  # Million barrels
            "importance": "WTI delivery point",
        },
        "rotterdam": {
            "name": "Rotterdam, Netherlands",
            "capacity_mb": 35,
            "importance": "European hub",
        },
        "singapore": {
            "name": "Singapore",
            "capacity_mb": 45,
            "importance": "Asian trading hub",
        },
        "fujairah": {
            "name": "Fujairah, UAE",
            "capacity_mb": 42,
            "importance": "Middle East hub",
        },
        "houston": {
            "name": "Houston, Texas",
            "capacity_mb": 65,
            "importance": "US Gulf Coast",
        },
    }
    
    def __init__(self, config: Optional[SatelliteConfig] = None):
        self.config = config or SatelliteConfig()
        
        # Data cache
        self._data: Dict[str, List[StorageTankData]] = {}
        self._last_update: Optional[datetime] = None
    
    def get_latest_observations(
        self,
        location: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get latest storage observations.
        
        Args:
            location: Specific location to query
            
        Returns:
            Storage observations
        """
        if self.config.provider == "mock":
            return self._generate_mock_data(location)
        
        # Would integrate with real satellite data API
        return self._generate_mock_data(location)
    
    def get_storage_trends(
        self,
        location: str,
        days: int = 30,
    ) -> List[Dict]:
        """
        Get storage trends over time.
        
        Args:
            location: Location to analyze
            days: Number of days of history
            
        Returns:
            Historical storage data
        """
        # Generate mock historical data
        from datetime import timedelta
        
        trends = []
        base_utilization = random.uniform(55, 70)
        
        for i in range(days):
            date = datetime.now() - timedelta(days=days - i)
            
            # Random walk for utilization
            change = random.uniform(-2, 2)
            base_utilization = max(30, min(95, base_utilization + change))
            
            trends.append({
                "date": date.isoformat(),
                "utilization_pct": round(base_utilization, 1),
                "change_daily": round(change, 2),
            })
        
        return trends
    
    def get_global_summary(self) -> Dict[str, Any]:
        """
        Get global storage summary.
        
        Returns:
            Summary of global storage levels
        """
        all_observations = self.get_latest_observations()
        locations = all_observations.get("locations", {})
        
        if not locations:
            return {"error": "No data available"}
        
        total_capacity = sum(
            self.LOCATIONS[loc]["capacity_mb"]
            for loc in locations
            if loc in self.LOCATIONS
        )
        
        total_volume = sum(
            self.LOCATIONS[loc]["capacity_mb"] * loc_data.get("utilization_pct", 50) / 100
            for loc, loc_data in locations.items()
            if loc in self.LOCATIONS
        )
        
        return {
            "timestamp": datetime.now().isoformat(),
            "global_utilization_pct": round(total_volume / total_capacity * 100, 1) if total_capacity > 0 else 0,
            "total_capacity_mb": total_capacity,
            "estimated_volume_mb": round(total_volume, 1),
            "locations_observed": len(locations),
            "by_location": {
                loc: data
                for loc, data in locations.items()
            },
        }
    
    def calculate_storage_signal(self) -> Dict[str, Any]:
        """
        Calculate trading signal from storage data.
        
        Returns:
            Signal based on storage levels
        """
        summary = self.get_global_summary()
        
        global_util = summary.get("global_utilization_pct", 50)
        
        # High storage = bearish, low storage = bullish
        if global_util < 40:
            signal = "bullish"
            confidence = min(90, (50 - global_util) * 2)
            rationale = "Low storage levels indicate tight supply"
        elif global_util > 75:
            signal = "bearish"
            confidence = min(90, (global_util - 60) * 2)
            rationale = "High storage levels indicate surplus"
        else:
            signal = "neutral"
            confidence = 50
            rationale = "Storage levels within normal range"
        
        return {
            "signal": signal,
            "confidence": round(confidence),
            "rationale": rationale,
            "global_utilization": global_util,
            "timestamp": datetime.now().isoformat(),
        }
    
    def _generate_mock_data(self, location: Optional[str] = None) -> Dict[str, Any]:
        """Generate mock satellite data."""
        now = datetime.now()
        
        locations_to_fetch = (
            [location] if location and location in self.LOCATIONS
            else list(self.LOCATIONS.keys())
        )
        
        location_data = {}
        
        for loc in locations_to_fetch:
            loc_info = self.LOCATIONS[loc]
            utilization = random.uniform(45, 85)
            
            location_data[loc] = {
                "name": loc_info["name"],
                "utilization_pct": round(utilization, 1),
                "capacity_mb": loc_info["capacity_mb"],
                "estimated_volume_mb": round(loc_info["capacity_mb"] * utilization / 100, 2),
                "change_week_pct": round(random.uniform(-5, 5), 2),
                "tanks_observed": random.randint(50, 200),
                "confidence": round(random.uniform(0.85, 0.98), 2),
                "last_observation": now.isoformat(),
            }
        
        return {
            "timestamp": now.isoformat(),
            "source": "satellite",
            "provider": self.config.provider,
            "locations": location_data,
        }
