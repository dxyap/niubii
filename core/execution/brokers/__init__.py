"""
Broker Integration
==================
Broker interfaces for order execution.

Provides:
- Base broker interface
- Simulated broker for paper trading
- Interactive Brokers integration (future)
"""

from .base import Broker, BrokerConfig, BrokerStatus, ExecutionReport
from .simulator import SimulatedBroker, SimulatorConfig

__all__ = [
    "Broker",
    "BrokerConfig",
    "BrokerStatus",
    "ExecutionReport",
    "SimulatedBroker",
    "SimulatorConfig",
]
