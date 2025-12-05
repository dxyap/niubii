"""
Execution Module
================
Order management, execution algorithms, and automated trading.

This module provides:
- Order Management System (OMS) for order lifecycle management
- Position sizing algorithms (Kelly, volatility targeting, risk parity)
- Execution algorithms (TWAP, VWAP, implementation shortfall)
- Broker integration layer (simulator + real broker interfaces)
- Paper trading mode for strategy testing
- Automation rules engine for signal-to-order conversion

Phase 6 Implementation - Execution & Automation
"""

from .algorithms import (
    AlgorithmConfig,
    ExecutionAlgorithm,
    ExecutionSlice,
    TWAPAlgorithm,
    VWAPAlgorithm,
)
from .automation import (
    AutomationEngine,
    AutomationRule,
    RuleAction,
    RuleCondition,
    RuleConfig,
    RuleStatus,
)
from .oms import (
    Order,
    OrderEvent,
    OrderManager,
    OrderSide,
    OrderStatus,
    OrderType,
    OrderUpdate,
    TimeInForce,
)
from .paper_trading import (
    PaperTradingConfig,
    PaperTradingEngine,
    SimulatedFill,
)
from .sizing import (
    FixedFractional,
    KellyCriterion,
    PositionSizer,
    RiskParity,
    SizingConfig,
    SizingMethod,
    VolatilityTargeting,
)

__all__ = [
    # OMS
    "Order",
    "OrderStatus",
    "OrderType",
    "OrderSide",
    "TimeInForce",
    "OrderManager",
    "OrderEvent",
    "OrderUpdate",
    # Sizing
    "PositionSizer",
    "SizingMethod",
    "KellyCriterion",
    "VolatilityTargeting",
    "RiskParity",
    "FixedFractional",
    "SizingConfig",
    # Algorithms
    "ExecutionAlgorithm",
    "TWAPAlgorithm",
    "VWAPAlgorithm",
    "AlgorithmConfig",
    "ExecutionSlice",
    # Paper Trading
    "PaperTradingEngine",
    "PaperTradingConfig",
    "SimulatedFill",
    # Automation
    "AutomationRule",
    "RuleCondition",
    "RuleAction",
    "AutomationEngine",
    "RuleConfig",
    "RuleStatus",
]
