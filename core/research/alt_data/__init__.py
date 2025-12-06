"""
Alternative Data Module
=======================
Alternative data sources for oil market analysis.

Data sources:
- Satellite imagery (storage tank levels)
- Shipping/tanker tracking
- Positioning data (COT, speculator positions)
"""

from .positioning import (
    COTData,
    PositioningConfig,
    PositioningData,
)
from .provider import (
    AlternativeDataProvider,
    DataSourceConfig,
)
from .satellite import (
    SatelliteConfig,
    SatelliteData,
    StorageTankData,
)
from .shipping import (
    ShippingConfig,
    ShippingData,
    TankerData,
)

__all__ = [
    "AlternativeDataProvider",
    "DataSourceConfig",
    "SatelliteData",
    "StorageTankData",
    "SatelliteConfig",
    "ShippingData",
    "TankerData",
    "ShippingConfig",
    "PositioningData",
    "COTData",
    "PositioningConfig",
]
