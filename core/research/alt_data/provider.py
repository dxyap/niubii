"""
Alternative Data Provider
=========================
Unified interface for alternative data sources.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class DataSourceType(Enum):
    """Alternative data source types."""
    SATELLITE = "satellite"
    SHIPPING = "shipping"
    POSITIONING = "positioning"
    REFINERY = "refinery"
    WEATHER = "weather"
    SOCIAL = "social"


@dataclass
class DataSourceConfig:
    """Configuration for a data source."""
    source_type: DataSourceType
    enabled: bool = True
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    refresh_interval_hours: int = 24
    cache_enabled: bool = True
    
    # Credentials
    username: Optional[str] = None
    password: Optional[str] = None


class AlternativeDataProvider:
    """
    Unified provider for alternative data sources.
    
    Aggregates data from multiple alternative sources and
    provides a consistent interface for accessing insights.
    """
    
    def __init__(self, configs: Optional[Dict[DataSourceType, DataSourceConfig]] = None):
        self.configs = configs or {}
        
        # Data caches
        self._cache: Dict[str, Any] = {}
        self._last_fetch: Dict[str, datetime] = {}
        
        # Initialize providers (lazy load)
        self._providers: Dict[DataSourceType, Any] = {}
    
    def get_latest_data(
        self,
        source_type: DataSourceType,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Get latest data from a source.
        
        Args:
            source_type: Type of data source
            **kwargs: Source-specific parameters
            
        Returns:
            Data from the source
        """
        if source_type not in self.configs or not self.configs[source_type].enabled:
            return self._get_mock_data(source_type, **kwargs)
        
        # Check cache
        cache_key = f"{source_type.value}_{str(kwargs)}"
        if cache_key in self._cache:
            last_fetch = self._last_fetch.get(cache_key)
            config = self.configs[source_type]
            
            if last_fetch:
                hours_elapsed = (datetime.now() - last_fetch).total_seconds() / 3600
                if hours_elapsed < config.refresh_interval_hours:
                    return self._cache[cache_key]
        
        # Fetch fresh data
        try:
            if source_type == DataSourceType.SATELLITE:
                data = self._fetch_satellite_data(**kwargs)
            elif source_type == DataSourceType.SHIPPING:
                data = self._fetch_shipping_data(**kwargs)
            elif source_type == DataSourceType.POSITIONING:
                data = self._fetch_positioning_data(**kwargs)
            else:
                data = self._get_mock_data(source_type, **kwargs)
            
            # Cache
            self._cache[cache_key] = data
            self._last_fetch[cache_key] = datetime.now()
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch {source_type.value} data: {e}")
            return self._get_mock_data(source_type, **kwargs)
    
    def get_aggregated_view(self) -> Dict[str, Any]:
        """
        Get aggregated view from all enabled sources.
        
        Returns:
            Aggregated alternative data insights
        """
        insights = {
            "timestamp": datetime.now().isoformat(),
            "sources": [],
        }
        
        # Satellite data
        try:
            satellite = self.get_latest_data(DataSourceType.SATELLITE)
            insights["satellite_storage"] = satellite
            insights["sources"].append("satellite")
        except Exception as e:
            logger.debug(f"Satellite data unavailable: {e}")
        
        # Shipping data
        try:
            shipping = self.get_latest_data(DataSourceType.SHIPPING)
            insights["shipping_activity"] = shipping
            insights["sources"].append("shipping")
        except Exception as e:
            logger.debug(f"Shipping data unavailable: {e}")
        
        # Positioning data
        try:
            positioning = self.get_latest_data(DataSourceType.POSITIONING)
            insights["market_positioning"] = positioning
            insights["sources"].append("positioning")
        except Exception as e:
            logger.debug(f"Positioning data unavailable: {e}")
        
        # Generate overall signal
        insights["signal"] = self._generate_aggregate_signal(insights)
        
        return insights
    
    def _fetch_satellite_data(self, **kwargs) -> Dict[str, Any]:
        """Fetch satellite imagery data."""
        # Would integrate with satellite data providers
        # (Orbital Insight, Kayrros, etc.)
        return self._get_mock_data(DataSourceType.SATELLITE, **kwargs)
    
    def _fetch_shipping_data(self, **kwargs) -> Dict[str, Any]:
        """Fetch shipping/tanker data."""
        # Would integrate with AIS data providers
        # (MarineTraffic, Kpler, etc.)
        return self._get_mock_data(DataSourceType.SHIPPING, **kwargs)
    
    def _fetch_positioning_data(self, **kwargs) -> Dict[str, Any]:
        """Fetch positioning/COT data."""
        # Would fetch from CFTC or data vendors
        return self._get_mock_data(DataSourceType.POSITIONING, **kwargs)
    
    def _get_mock_data(
        self,
        source_type: DataSourceType,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate mock data for testing."""
        import random
        
        base_timestamp = datetime.now().isoformat()
        
        if source_type == DataSourceType.SATELLITE:
            return {
                "timestamp": base_timestamp,
                "source": "satellite",
                "data_type": "storage_levels",
                "locations": {
                    "cushing_ok": {
                        "utilization_pct": round(random.uniform(45, 75), 1),
                        "change_week": round(random.uniform(-3, 3), 2),
                        "tanks_observed": 145,
                    },
                    "rotterdam": {
                        "utilization_pct": round(random.uniform(60, 85), 1),
                        "change_week": round(random.uniform(-2, 2), 2),
                        "tanks_observed": 89,
                    },
                    "singapore": {
                        "utilization_pct": round(random.uniform(70, 90), 1),
                        "change_week": round(random.uniform(-2, 2), 2),
                        "tanks_observed": 112,
                    },
                },
                "global_utilization": round(random.uniform(55, 75), 1),
                "signal": "neutral" if random.random() > 0.3 else ("bullish" if random.random() > 0.5 else "bearish"),
            }
        
        elif source_type == DataSourceType.SHIPPING:
            return {
                "timestamp": base_timestamp,
                "source": "shipping",
                "data_type": "tanker_activity",
                "vlcc_activity": {
                    "at_sea": random.randint(150, 250),
                    "loading": random.randint(20, 40),
                    "discharging": random.randint(25, 45),
                    "anchored": random.randint(30, 60),
                },
                "trade_flows": {
                    "middle_east_asia": round(random.uniform(15, 25), 1),  # mb/d
                    "atlantic_basin": round(random.uniform(8, 15), 1),
                    "russia_exports": round(random.uniform(3, 6), 1),
                },
                "freight_rates": {
                    "vlcc_td3": round(random.uniform(20, 80), 0),  # $/day (thousands)
                    "suezmax": round(random.uniform(15, 50), 0),
                    "aframax": round(random.uniform(10, 35), 0),
                },
                "signal": "neutral",
            }
        
        elif source_type == DataSourceType.POSITIONING:
            return {
                "timestamp": base_timestamp,
                "source": "cot_report",
                "data_type": "market_positioning",
                "crude_oil": {
                    "managed_money_net": random.randint(-50000, 150000),
                    "producer_hedger_net": random.randint(-200000, -50000),
                    "swap_dealer_net": random.randint(-100000, 50000),
                    "spec_long_pct": round(random.uniform(50, 80), 1),
                    "change_week": random.randint(-15000, 15000),
                },
                "gasoline": {
                    "managed_money_net": random.randint(-20000, 60000),
                    "spec_long_pct": round(random.uniform(45, 75), 1),
                },
                "heating_oil": {
                    "managed_money_net": random.randint(-15000, 40000),
                    "spec_long_pct": round(random.uniform(40, 70), 1),
                },
                "signal": "bullish" if random.random() > 0.6 else ("bearish" if random.random() > 0.5 else "neutral"),
            }
        
        else:
            return {
                "timestamp": base_timestamp,
                "source": source_type.value,
                "data_type": "unknown",
                "message": "Mock data - configure API for real data",
            }
    
    def _generate_aggregate_signal(self, insights: Dict[str, Any]) -> Dict[str, Any]:
        """Generate aggregate signal from all sources."""
        signals = []
        
        if "satellite_storage" in insights:
            sat_signal = insights["satellite_storage"].get("signal", "neutral")
            signals.append({"source": "satellite", "signal": sat_signal, "weight": 0.3})
        
        if "shipping_activity" in insights:
            ship_signal = insights["shipping_activity"].get("signal", "neutral")
            signals.append({"source": "shipping", "signal": ship_signal, "weight": 0.3})
        
        if "market_positioning" in insights:
            pos_signal = insights["market_positioning"].get("signal", "neutral")
            signals.append({"source": "positioning", "signal": pos_signal, "weight": 0.4})
        
        if not signals:
            return {"direction": "neutral", "confidence": 0}
        
        # Calculate weighted signal
        signal_values = {"bullish": 1, "neutral": 0, "bearish": -1}
        
        weighted_sum = sum(
            signal_values.get(s["signal"], 0) * s["weight"]
            for s in signals
        )
        total_weight = sum(s["weight"] for s in signals)
        
        if total_weight > 0:
            score = weighted_sum / total_weight
        else:
            score = 0
        
        if score > 0.3:
            direction = "bullish"
        elif score < -0.3:
            direction = "bearish"
        else:
            direction = "neutral"
        
        return {
            "direction": direction,
            "score": round(score, 2),
            "confidence": round(min(abs(score) * 100, 100), 0),
            "contributing_sources": len(signals),
        }
    
    def get_data_freshness(self) -> Dict[str, Any]:
        """Get freshness of cached data."""
        freshness = {}
        
        for key, last_fetch in self._last_fetch.items():
            age_hours = (datetime.now() - last_fetch).total_seconds() / 3600
            freshness[key] = {
                "last_fetch": last_fetch.isoformat(),
                "age_hours": round(age_hours, 2),
            }
        
        return freshness
    
    def clear_cache(self):
        """Clear all cached data."""
        self._cache.clear()
        self._last_fetch.clear()
