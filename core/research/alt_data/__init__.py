"""
Alternative Data Module
=======================
Alternative data sources for oil market analysis.

Data sources:
- Satellite imagery (storage tank levels)
- Shipping/tanker tracking
- Positioning data (COT, speculator positions)
"""

from .provider import (
    AlternativeDataProvider,
    DataSourceConfig,
)

from .satellite import (
    SatelliteData,
    StorageTankData,
    SatelliteConfig,
)

from .shipping import (
    ShippingData,
    TankerData,
    ShippingConfig,
)

from .positioning import (
    PositioningData,
    COTData,
    PositioningConfig,
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
