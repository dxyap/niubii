"""
Backtesting Module
==================
Comprehensive backtesting framework for trading strategy development.

Components:
- engine: Main backtesting engine
- strategy: Strategy base classes and examples
- execution: Order execution simulation
- costs: Transaction cost models
- metrics: Performance metrics
- optimization: Parameter optimization
- reporting: Report generation

Example:
    from core.backtest import (
        BacktestEngine, BacktestConfig,
        MACrossoverStrategy, StrategyConfig,
        run_backtest
    )
    
    # Create strategy
    strategy = MACrossoverStrategy(fast_period=10, slow_period=30)
    
    # Run backtest
    result = run_backtest(strategy, historical_data)
    
    # View results
    print(f"Sharpe: {result.metrics.sharpe_ratio:.2f}")
    print(f"Return: {result.metrics.total_return_pct:.2f}%")
"""

# Core engine
from .engine import (
    BacktestEngine,
    BacktestConfig,
    BacktestResult,
    run_backtest,
    quick_backtest,
)

# Strategy framework
from .strategy import (
    Strategy,
    StrategyConfig,
    Signal,
    Order,
    Position,
    BuyAndHoldStrategy,
    MACrossoverStrategy,
    RSIMeanReversionStrategy,
    BollingerBandStrategy,
    MomentumStrategy,
    CalendarSpreadStrategy,
    CompositeStrategy,
    create_strategy_from_signals,
)

# Execution
from .execution import (
    ExecutionSimulator,
    PositionManager,
    Fill,
    OrderRecord,
    OrderStatus,
)

# Costs
from .costs import (
    CostModel,
    CostModelConfig,
    SimpleCostModel,
    VolatilityAdjustedCostModel,
    MarketImpactCostModel,
    TieredCommissionModel,
    TransactionCosts,
    OrderSide,
    create_cost_comparison,
)

# Metrics
from .metrics import (
    MetricsCalculator,
    PerformanceMetrics,
    calculate_drawdown_series,
    calculate_monthly_returns,
    compare_strategies,
    TRADING_DAYS_PER_YEAR,
)

# Optimization
from .optimization import (
    StrategyOptimizer,
    OptimizationConfig,
    OptimizationResult,
    ParameterGrid,
    sensitivity_analysis,
    monte_carlo_analysis,
)

# Reporting
from .reporting import (
    generate_summary_report,
    generate_html_report,
    create_equity_chart,
    create_drawdown_chart,
    create_monthly_heatmap,
    create_returns_distribution,
    create_trade_analysis_chart,
    create_full_report,
    compare_results_chart,
)

__all__ = [
    # Engine
    "BacktestEngine",
    "BacktestConfig",
    "BacktestResult",
    "run_backtest",
    "quick_backtest",
    # Strategy
    "Strategy",
    "StrategyConfig",
    "Signal",
    "Order",
    "Position",
    "BuyAndHoldStrategy",
    "MACrossoverStrategy",
    "RSIMeanReversionStrategy",
    "BollingerBandStrategy",
    "MomentumStrategy",
    "CalendarSpreadStrategy",
    "CompositeStrategy",
    "create_strategy_from_signals",
    # Execution
    "ExecutionSimulator",
    "PositionManager",
    "Fill",
    "OrderRecord",
    "OrderStatus",
    # Costs
    "CostModel",
    "CostModelConfig",
    "SimpleCostModel",
    "VolatilityAdjustedCostModel",
    "MarketImpactCostModel",
    "TieredCommissionModel",
    "TransactionCosts",
    "OrderSide",
    "create_cost_comparison",
    # Metrics
    "MetricsCalculator",
    "PerformanceMetrics",
    "calculate_drawdown_series",
    "calculate_monthly_returns",
    "compare_strategies",
    "TRADING_DAYS_PER_YEAR",
    # Optimization
    "StrategyOptimizer",
    "OptimizationConfig",
    "OptimizationResult",
    "ParameterGrid",
    "sensitivity_analysis",
    "monte_carlo_analysis",
    # Reporting
    "generate_summary_report",
    "generate_html_report",
    "create_equity_chart",
    "create_drawdown_chart",
    "create_monthly_heatmap",
    "create_returns_distribution",
    "create_trade_analysis_chart",
    "create_full_report",
    "compare_results_chart",
]
