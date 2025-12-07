"""
Regional Oil Inventory Data
===========================
Real-time oil storage inventory monitoring via Bloomberg data.

Uses Bloomberg Index tickers for:
- ARA (Amsterdam-Rotterdam-Antwerp): ARACRS, ARAGSS, ARAGAS, ARAFLS
- Fujairah: FUJLDS, FUJMDS, FUJHDS
- Singapore: MASGLST, MASGMST, MASGHST
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SatelliteConfig:
    """Configuration for regional inventory data."""
    provider: str = "bloomberg"  # bloomberg is the production provider
    api_key: str | None = None  # Not needed for Bloomberg
    locations: list[str] = field(default_factory=lambda: [
        "cushing_ok", "rotterdam", "singapore", "fujairah"
    ])
    refresh_hours: int = 24


@dataclass
class StorageTankData:
    """Storage observation data."""
    location: str
    tank_id: str | None
    observation_date: datetime
    utilization_pct: float
    capacity_barrels: int
    estimated_volume: int
    confidence: float
    source: str = "bloomberg"

    def to_dict(self) -> dict:
        return {
            "location": self.location,
            "tank_id": self.tank_id,
            "observation_date": self.observation_date.isoformat(),
            "utilization_pct": self.utilization_pct,
            "capacity_barrels": self.capacity_barrels,
            "estimated_volume": self.estimated_volume,
            "confidence": self.confidence,
        }


# =============================================================================
# BLOOMBERG INVENTORY TICKERS - FILL IN YOUR TICKERS HERE
# =============================================================================
# Instructions:
#   1. Search Bloomberg for the relevant inventory/stock indices
#   2. Replace the placeholder strings below with actual Bloomberg tickers
#   3. Format: "TICKER Index" (e.g., "ARACRS Index")
#   4. Units are typically in thousand barrels (kb)
#
# Suggested tickers to search in Bloomberg:
#   - ARA: Search "ARA crude stocks", "ARA gasoline", "ARA gasoil", "ARA fuel oil"
#   - Fujairah: Search "Fujairah light distillates", "FOIZ stocks", "FEDCom"
#   - Singapore: Search "Singapore oil stocks", "MAS light distillates"
# =============================================================================

BLOOMBERG_INVENTORY_TICKERS = {
    # -------------------------------------------------------------------------
    # ARA (Amsterdam-Rotterdam-Antwerp) - European Hub
    # -------------------------------------------------------------------------
    "ara": {
        "crude": "",       # TODO: Add ARA Crude Stocks ticker (e.g., "ARACRS Index")
        "gasoline": "",    # TODO: Add ARA Gasoline Stocks ticker (e.g., "ARAGSS Index")
        "gasoil": "",      # TODO: Add ARA Gasoil/Diesel Stocks ticker (e.g., "ARAGAS Index")
        "fuel_oil": "",    # TODO: Add ARA Fuel Oil Stocks ticker (e.g., "ARAFLS Index")
    },

    # -------------------------------------------------------------------------
    # Fujairah - Middle East Hub (UAE)
    # -------------------------------------------------------------------------
    "fujairah": {
        "light": "",       # TODO: Add Fujairah Light Distillates ticker (e.g., "FUJLDS Index")
        "middle": "",      # TODO: Add Fujairah Middle Distillates ticker (e.g., "FUJMDS Index")
        "heavy": "",       # TODO: Add Fujairah Heavy Distillates/Residue ticker (e.g., "FUJHDS Index")
    },

    # -------------------------------------------------------------------------
    # Singapore - Asian Hub
    # -------------------------------------------------------------------------
    "singapore": {
        "light": "",       # TODO: Add Singapore Light Distillates ticker (e.g., "MASGLST Index")
        "middle": "",      # TODO: Add Singapore Middle Distillates ticker (e.g., "MASGMST Index")
        "heavy": "",       # TODO: Add Singapore Heavy/Fuel Oil ticker (e.g., "MASGHST Index")
    },
}

# Estimated capacities in million barrels (for utilization calculations)
STORAGE_CAPACITIES_MB = {
    "ara": {
        "total": 50,  # ARA total capacity ~50 MMbbl
        "crude": 25,
        "products": 25,
    },
    "fujairah": {
        "total": 42,  # Fujairah capacity ~42 MMbbl
        "crude": 20,
        "products": 22,
    },
    "singapore": {
        "total": 85,  # Singapore capacity ~85 MMbbl (major Asian hub)
        "crude": 45,
        "products": 40,
    },
}


class SatelliteData:
    """
    Regional oil inventory data provider.

    Production mode: Uses Bloomberg API for real inventory data.
    Monitors oil storage levels in ARA, Fujairah, and Singapore.
    """

    # Major storage locations and estimated capacities
    LOCATIONS = {
        "cushing_ok": {
            "name": "Cushing, Oklahoma",
            "capacity_mb": 76,  # Million barrels
            "importance": "WTI delivery point",
        },
        "rotterdam": {
            "name": "ARA (Amsterdam-Rotterdam-Antwerp)",
            "capacity_mb": 50,
            "importance": "European hub",
        },
        "singapore": {
            "name": "Singapore",
            "capacity_mb": 85,
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

    def __init__(self, config: SatelliteConfig | None = None):
        self.config = config or SatelliteConfig()

        # Data cache
        self._data: dict[str, list[StorageTankData]] = {}
        self._last_update: datetime | None = None

        # Bloomberg client (lazy initialization)
        self._bloomberg_client = None

    def _get_bloomberg_client(self):
        """Get or create Bloomberg client."""
        if self._bloomberg_client is None:
            try:
                from core.data.bloomberg import BloombergClient
                self._bloomberg_client = BloombergClient()
            except Exception as e:
                logger.error(f"Failed to initialize Bloomberg client: {e}")
                return None
        return self._bloomberg_client

    def get_latest_observations(
        self,
        location: str | None = None,
    ) -> dict[str, Any]:
        """
        Get latest storage observations from Bloomberg.

        Args:
            location: Specific location to query

        Returns:
            Storage observations with real data
        """
        return self._fetch_bloomberg_inventory_data(location)

    def _fetch_bloomberg_inventory_data(self, location: str | None = None) -> dict[str, Any]:
        """
        Fetch real inventory data from Bloomberg.

        Returns:
            Dictionary with inventory data for each location
        """
        client = self._get_bloomberg_client()
        now = datetime.now()

        if client is None or not client.connected:
            logger.warning("Bloomberg not connected - inventory data unavailable")
            return {
                "timestamp": now.isoformat(),
                "source": "bloomberg",
                "provider": "bloomberg",
                "locations": {},
                "error": "Bloomberg connection unavailable",
            }

        locations_to_fetch = (
            [location] if location and location in ["ara", "fujairah", "singapore", "rotterdam"]
            else ["ara", "fujairah", "singapore"]
        )

        # Map rotterdam to ara for backwards compatibility
        if "rotterdam" in locations_to_fetch:
            locations_to_fetch.remove("rotterdam")
            if "ara" not in locations_to_fetch:
                locations_to_fetch.append("ara")

        location_data = {}

        for loc in locations_to_fetch:
            try:
                loc_result = self._fetch_location_data(client, loc)
                if loc_result:
                    location_data[loc] = loc_result
                    # Also add under "rotterdam" key for backwards compatibility
                    if loc == "ara":
                        location_data["rotterdam"] = loc_result
            except Exception as e:
                logger.error(f"Failed to fetch {loc} inventory data: {e}")
                continue

        return {
            "timestamp": now.isoformat(),
            "source": "bloomberg",
            "provider": "bloomberg",
            "locations": location_data,
        }

    def _fetch_location_data(self, client, location: str) -> dict[str, Any] | None:
        """
        Fetch inventory data for a specific location from Bloomberg.

        Args:
            client: Bloomberg client
            location: Location key (ara, fujairah, singapore)

        Returns:
            Dictionary with inventory data for the location
        """
        tickers = BLOOMBERG_INVENTORY_TICKERS.get(location)
        if not tickers:
            return None

        # Filter out empty tickers (not yet configured)
        configured_tickers = {k: v for k, v in tickers.items() if v}

        if not configured_tickers:
            logger.warning(
                f"No Bloomberg tickers configured for {location}. "
                f"Please add tickers in core/research/alt_data/satellite.py"
            )
            return None

        capacity = STORAGE_CAPACITIES_MB.get(location, {})
        total_capacity = capacity.get("total", 50)

        try:
            # Get all configured tickers for this location
            ticker_list = list(configured_tickers.values())
            fields = ["PX_LAST", "CHG_PCT_1W"]

            prices_df = client.get_prices(ticker_list, fields)

            if prices_df is None or prices_df.empty:
                logger.warning(f"No data returned for {location}")
                return None

            # Calculate total inventory (convert from kb to MMbbl)
            total_inventory_kb = 0
            product_breakdown = {}
            weekly_change_sum = 0
            count = 0

            for product_key, ticker in configured_tickers.items():
                if ticker in prices_df.index:
                    row = prices_df.loc[ticker]
                    volume_kb = row.get("PX_LAST", 0)
                    weekly_change = row.get("CHG_PCT_1W", 0)

                    if volume_kb is not None and not (hasattr(volume_kb, '__iter__') and not volume_kb):
                        total_inventory_kb += float(volume_kb)
                        product_breakdown[product_key] = {
                            "volume_kb": float(volume_kb),
                            "volume_mmb": float(volume_kb) / 1000,
                            "weekly_change_pct": float(weekly_change) if weekly_change else 0,
                        }
                        if weekly_change:
                            weekly_change_sum += float(weekly_change)
                            count += 1

            # Convert to million barrels
            total_inventory_mmb = total_inventory_kb / 1000

            # Calculate utilization
            utilization = (total_inventory_mmb / total_capacity) * 100 if total_capacity > 0 else 0

            # Average weekly change
            avg_weekly_change = weekly_change_sum / count if count > 0 else 0

            location_names = {
                "ara": "ARA (Amsterdam-Rotterdam-Antwerp)",
                "fujairah": "Fujairah, UAE",
                "singapore": "Singapore",
            }

            return {
                "name": location_names.get(location, location),
                "utilization_pct": round(utilization, 1),
                "capacity_mb": total_capacity,
                "estimated_volume_mb": round(total_inventory_mmb, 2),
                "change_week_pct": round(avg_weekly_change, 2),
                "tanks_observed": len(product_breakdown),
                "confidence": 0.95,  # High confidence with Bloomberg data
                "last_observation": datetime.now().isoformat(),
                "source": "bloomberg",
                "products": product_breakdown,
            }

        except Exception as e:
            logger.error(f"Error fetching {location} data from Bloomberg: {e}")
            return None

    def get_storage_trends(
        self,
        location: str,
        days: int = 30,
    ) -> list[dict]:
        """
        Get storage trends over time from Bloomberg historical data.

        Args:
            location: Location to analyze
            days: Number of days of history

        Returns:
            Historical storage data
        """
        client = self._get_bloomberg_client()
        if client is None or not client.connected:
            logger.warning("Bloomberg not connected - cannot fetch historical trends")
            return []

        # Map location to Bloomberg ticker
        location_key = "ara" if location in ["rotterdam", "ara"] else location

        tickers = BLOOMBERG_INVENTORY_TICKERS.get(location_key)
        if not tickers:
            return []

        # Filter out empty tickers
        configured_tickers = {k: v for k, v in tickers.items() if v}
        if not configured_tickers:
            logger.warning(f"No Bloomberg tickers configured for {location_key}")
            return []

        try:
            from datetime import timedelta

            # Use the first configured ticker for trend data
            ticker = list(configured_tickers.values())[0]
            start_date = datetime.now() - timedelta(days=days)

            hist_data = client.get_historical(
                ticker,
                start_date=start_date,
                fields=["PX_LAST"]
            )

            if hist_data is None or hist_data.empty:
                return []

            trends = []
            prev_value = None

            for date, row in hist_data.iterrows():
                value = row.get("PX_LAST")
                if value is not None:
                    daily_change = 0
                    if prev_value is not None:
                        daily_change = ((value - prev_value) / prev_value) * 100 if prev_value > 0 else 0
                    prev_value = value

                    # Convert kb to utilization percentage
                    capacity = STORAGE_CAPACITIES_MB.get(location_key, {})
                    total_capacity_kb = capacity.get("total", 50) * 1000
                    utilization = (value / total_capacity_kb) * 100 if total_capacity_kb > 0 else 0

                    trends.append({
                        "date": date.isoformat() if hasattr(date, 'isoformat') else str(date),
                        "utilization_pct": round(utilization, 1),
                        "change_daily": round(daily_change, 2),
                        "volume_kb": value,
                    })

            return trends

        except Exception as e:
            logger.error(f"Error fetching historical trends: {e}")
            return []

    def get_global_summary(self) -> dict[str, Any]:
        """
        Get global storage summary.

        Returns:
            Summary of global storage levels
        """
        all_observations = self.get_latest_observations()
        locations = all_observations.get("locations", {})

        if not locations:
            return {"error": "No data available"}

        # Exclude duplicate rotterdam entry (it's the same as ara)
        unique_locations = {k: v for k, v in locations.items() if k != "rotterdam"}

        total_capacity = sum(
            loc_data.get("capacity_mb", 0)
            for loc_data in unique_locations.values()
        )

        total_volume = sum(
            loc_data.get("estimated_volume_mb", 0)
            for loc_data in unique_locations.values()
        )

        return {
            "timestamp": datetime.now().isoformat(),
            "global_utilization_pct": round(total_volume / total_capacity * 100, 1) if total_capacity > 0 else 0,
            "total_capacity_mb": total_capacity,
            "estimated_volume_mb": round(total_volume, 1),
            "locations_observed": len(unique_locations),
            "by_location": dict(unique_locations.items()),
            "source": "bloomberg",
        }

    def calculate_storage_signal(self) -> dict[str, Any]:
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
            "source": "bloomberg",
        }

    def get_product_breakdown(self, location: str) -> dict[str, Any] | None:
        """
        Get detailed product breakdown for a location.

        Args:
            location: Location key (ara, fujairah, singapore)

        Returns:
            Detailed product inventory breakdown
        """
        observations = self.get_latest_observations(location)
        locations = observations.get("locations", {})

        # Handle rotterdam -> ara mapping
        location_key = "ara" if location == "rotterdam" else location
        loc_data = locations.get(location_key) or locations.get(location)

        if loc_data:
            return loc_data.get("products")

        return None
