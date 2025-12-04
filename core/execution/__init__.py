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

from .oms import (
    Order,
    OrderStatus,
    OrderType,
    OrderSide,
    TimeInForce,
    OrderManager,
    OrderEvent,
    OrderUpdate,
)

from .sizing import (
    PositionSizer,
    SizingMethod,
    KellyCriterion,
    VolatilityTargeting,
    RiskParity,
    FixedFractional,
    SizingConfig,
)

from .algorithms import (
    ExecutionAlgorithm,
    TWAPAlgorithm,
    VWAPAlgorithm,
    AlgorithmConfig,
    ExecutionSlice,
)

from .paper_trading import (
    PaperTradingEngine,
    PaperTradingConfig,
    SimulatedFill,
)

from .automation import (
    AutomationRule,
    RuleCondition,
    RuleAction,
    AutomationEngine,
    RuleConfig,
    RuleStatus,
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
