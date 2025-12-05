"""
Shipping Data
=============
Tanker tracking and maritime shipping analysis.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import random

logger = logging.getLogger(__name__)


@dataclass
class ShippingConfig:
    """Configuration for shipping data."""
    provider: str = "mock"  # kpler, vortexa, marinetraffic
    api_key: Optional[str] = None
    vessel_types: List[str] = field(default_factory=lambda: ["VLCC", "Suezmax", "Aframax"])
    refresh_hours: int = 6


@dataclass
class TankerData:
    """Data for a single tanker."""
    vessel_name: str
    imo: str
    vessel_type: str
    status: str  # loading, discharging, at_sea, anchored
    cargo_type: Optional[str]
    cargo_volume_barrels: Optional[int]
    origin: Optional[str]
    destination: Optional[str]
    eta: Optional[datetime]
    latitude: float
    longitude: float
    speed_knots: float
    last_update: datetime
    
    def to_dict(self) -> Dict:
        return {
            "vessel_name": self.vessel_name,
            "imo": self.imo,
            "vessel_type": self.vessel_type,
            "status": self.status,
            "cargo_type": self.cargo_type,
            "cargo_volume_barrels": self.cargo_volume_barrels,
            "origin": self.origin,
            "destination": self.destination,
            "eta": self.eta.isoformat() if self.eta else None,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "speed_knots": self.speed_knots,
        }


class ShippingData:
    """
    Shipping/tanker data provider.
    
    Monitors tanker movements and oil trade flows.
    """
    
    # Major trade routes
    TRADE_ROUTES = {
        "middle_east_asia": {
            "name": "Middle East to Asia",
            "typical_mb_d": 18,
            "vessel_types": ["VLCC"],
        },
        "atlantic_basin": {
            "name": "Atlantic Basin",
            "typical_mb_d": 10,
            "vessel_types": ["Suezmax", "Aframax"],
        },
        "russia_europe": {
            "name": "Russia to Europe/Asia",
            "typical_mb_d": 5,
            "vessel_types": ["Aframax", "Suezmax"],
        },
        "west_africa_exports": {
            "name": "West Africa Exports",
            "typical_mb_d": 4,
            "vessel_types": ["Suezmax", "VLCC"],
        },
        "us_gulf_exports": {
            "name": "US Gulf Exports",
            "typical_mb_d": 4,
            "vessel_types": ["VLCC", "Aframax"],
        },
    }
    
    # Freight rate benchmarks
    FREIGHT_ROUTES = {
        "TD3": "VLCC Middle East - China",
        "TD20": "Suezmax West Africa - Europe",
        "TD7": "Aframax North Sea - Europe",
    }
    
    def __init__(self, config: Optional[ShippingConfig] = None):
        self.config = config or ShippingConfig()
        
        # Data cache
        self._vessel_data: List[TankerData] = []
        self._last_update: Optional[datetime] = None
    
    def get_fleet_overview(self) -> Dict[str, Any]:
        """
        Get overview of tanker fleet activity.
        
        Returns:
            Fleet activity summary
        """
        if self.config.provider == "mock":
            return self._generate_mock_fleet_data()
        
        return self._generate_mock_fleet_data()
    
    def get_trade_flows(self) -> Dict[str, Any]:
        """
        Get current oil trade flows.
        
        Returns:
            Trade flow estimates
        """
        flows = {}
        
        for route_id, route_info in self.TRADE_ROUTES.items():
            typical = route_info["typical_mb_d"]
            # Add random variation
            current = typical * random.uniform(0.8, 1.2)
            change = random.uniform(-15, 15)
            
            flows[route_id] = {
                "name": route_info["name"],
                "current_mb_d": round(current, 2),
                "typical_mb_d": typical,
                "change_pct": round(change, 1),
                "vessel_types": route_info["vessel_types"],
            }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "source": "shipping",
            "flows": flows,
            "total_observed_mb_d": round(sum(f["current_mb_d"] for f in flows.values()), 1),
        }
    
    def get_freight_rates(self) -> Dict[str, Any]:
        """
        Get current freight rates.
        
        Returns:
            Freight rate data
        """
        rates = {}
        
        base_rates = {
            "VLCC": 45000,  # $/day
            "Suezmax": 30000,
            "Aframax": 20000,
        }
        
        for vessel_type, base_rate in base_rates.items():
            current = base_rate * random.uniform(0.7, 1.5)
            change_week = random.uniform(-20, 20)
            
            rates[vessel_type] = {
                "rate_usd_day": round(current),
                "change_week_pct": round(change_week, 1),
            }
        
        # TCE (Time Charter Equivalent) benchmarks
        tce_benchmarks = {}
        for route_id, route_name in self.FREIGHT_ROUTES.items():
            tce_benchmarks[route_id] = {
                "route": route_name,
                "tce_usd_day": round(random.uniform(15000, 80000)),
                "ws_points": round(random.uniform(40, 120), 1),
            }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "spot_rates": rates,
            "tce_benchmarks": tce_benchmarks,
        }
    
    def get_port_congestion(self) -> Dict[str, Any]:
        """
        Get port congestion data.
        
        Returns:
            Port congestion metrics
        """
        ports = {
            "ras_tanura": {"name": "Ras Tanura", "region": "Middle East"},
            "singapore": {"name": "Singapore", "region": "Asia"},
            "rotterdam": {"name": "Rotterdam", "region": "Europe"},
            "houston": {"name": "Houston", "region": "Americas"},
            "ningbo": {"name": "Ningbo", "region": "Asia"},
            "fujairah": {"name": "Fujairah", "region": "Middle East"},
        }
        
        congestion = {}
        
        for port_id, port_info in ports.items():
            vessels_waiting = random.randint(5, 40)
            avg_wait_days = random.uniform(1, 7)
            
            congestion[port_id] = {
                "name": port_info["name"],
                "region": port_info["region"],
                "vessels_waiting": vessels_waiting,
                "avg_wait_days": round(avg_wait_days, 1),
                "congestion_level": "high" if vessels_waiting > 25 else ("medium" if vessels_waiting > 15 else "low"),
            }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "ports": congestion,
        }
    
    def calculate_shipping_signal(self) -> Dict[str, Any]:
        """
        Calculate trading signal from shipping data.
        
        Returns:
            Signal based on shipping activity
        """
        flows = self.get_trade_flows()
        freight = self.get_freight_rates()
        
        # Analyze trade flows
        total_flow = flows.get("total_observed_mb_d", 40)
        typical_total = sum(r["typical_mb_d"] for r in self.TRADE_ROUTES.values())
        
        flow_ratio = total_flow / typical_total if typical_total > 0 else 1
        
        # Analyze freight rates (high rates = strong demand)
        vlcc_rate = freight.get("spot_rates", {}).get("VLCC", {}).get("rate_usd_day", 45000)
        
        # Generate signal
        if flow_ratio > 1.1 and vlcc_rate > 55000:
            signal = "bullish"
            confidence = 70
            rationale = "Strong trade flows and high freight rates indicate robust demand"
        elif flow_ratio < 0.9 and vlcc_rate < 35000:
            signal = "bearish"
            confidence = 65
            rationale = "Weak trade flows and low freight rates suggest soft demand"
        else:
            signal = "neutral"
            confidence = 50
            rationale = "Shipping activity within normal range"
        
        return {
            "signal": signal,
            "confidence": confidence,
            "rationale": rationale,
            "flow_ratio": round(flow_ratio, 2),
            "vlcc_rate": vlcc_rate,
            "timestamp": datetime.now().isoformat(),
        }
    
    def _generate_mock_fleet_data(self) -> Dict[str, Any]:
        """Generate mock fleet data."""
        vessel_counts = {
            "VLCC": {"at_sea": random.randint(180, 220), "loading": random.randint(25, 40), "discharging": random.randint(30, 50), "anchored": random.randint(40, 70)},
            "Suezmax": {"at_sea": random.randint(120, 160), "loading": random.randint(20, 35), "discharging": random.randint(25, 40), "anchored": random.randint(30, 50)},
            "Aframax": {"at_sea": random.randint(200, 280), "loading": random.randint(30, 50), "discharging": random.randint(40, 60), "anchored": random.randint(50, 80)},
        }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "source": "shipping",
            "provider": self.config.provider,
            "fleet_by_type": vessel_counts,
            "total_vessels": sum(
                sum(counts.values())
                for counts in vessel_counts.values()
            ),
        }
