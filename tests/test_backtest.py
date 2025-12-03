"""
Tests for Backtesting Module
============================
Comprehensive tests for the backtesting framework.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict

# Import all modules
from core.backtest import (
    # Engine
    BacktestEngine, BacktestConfig, BacktestResult,
    run_backtest, quick_backtest,
    # Strategies
    Strategy, StrategyConfig, Signal, Order, Position,
    BuyAndHoldStrategy, MACrossoverStrategy,
    RSIMeanReversionStrategy, BollingerBandStrategy,
    MomentumStrategy, create_strategy_from_signals,
    # Execution
    ExecutionSimulator, PositionManager, Fill, OrderStatus,
    # Costs
    CostModel, CostModelConfig, SimpleCostModel,
    VolatilityAdjustedCostModel, MarketImpactCostModel,
    TransactionCosts, OrderSide,
    # Metrics
    MetricsCalculator, PerformanceMetrics,
    calculate_drawdown_series, calculate_monthly_returns,
    compare_strategies,
    # Optimization
    StrategyOptimizer, OptimizationConfig, ParameterGrid,
    sensitivity_analysis,
    # Reporting
    generate_summary_report, create_equity_chart,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n = 252  # ~1 year of trading days
    
    dates = pd.date_range(start="2024-01-01", periods=n, freq="B")
    
    # Generate prices with trend and noise
    base_price = 75.0
    returns = np.random.normal(0.0002, 0.02, n)
    prices = base_price * np.cumprod(1 + returns)
    
    # Generate OHLC
    high_mult = 1 + np.abs(np.random.normal(0.005, 0.003, n))
    low_mult = 1 - np.abs(np.random.normal(0.005, 0.003, n))
    
    data = pd.DataFrame({
        "PX_OPEN": prices * (1 + np.random.normal(0, 0.002, n)),
        "PX_HIGH": prices * high_mult,
        "PX_LOW": prices * low_mult,
        "PX_LAST": prices,
        "PX_VOLUME": np.random.randint(50000, 200000, n),
    }, index=dates)
    
    # Ensure OHLC consistency
    data["PX_HIGH"] = data[["PX_OPEN", "PX_HIGH", "PX_LAST"]].max(axis=1)
    data["PX_LOW"] = data[["PX_OPEN", "PX_LOW", "PX_LAST"]].min(axis=1)
    
    return data


@pytest.fixture
def sample_equity_curve() -> pd.Series:
    """Generate sample equity curve."""
    np.random.seed(42)
    n = 252
    dates = pd.date_range(start="2024-01-01", periods=n, freq="B")
    
    initial = 1_000_000
    returns = np.random.normal(0.0003, 0.01, n)
    equity = initial * np.cumprod(1 + returns)
    
    return pd.Series(equity, index=dates)


@pytest.fixture
def sample_trades() -> pd.DataFrame:
    """Generate sample trade data."""
    np.random.seed(42)
    n = 50
    
    pnl = np.random.normal(500, 2000, n)
    
    return pd.DataFrame({
        "timestamp": pd.date_range(start="2024-01-01", periods=n, freq="5D"),
        "symbol": ["CL1"] * n,
        "side": np.random.choice(["BUY", "SELL"], n),
        "quantity": np.random.randint(1, 10, n),
        "price": np.random.uniform(70, 80, n),
        "pnl": pnl,
    })


# =============================================================================
# METRICS TESTS
# =============================================================================

class TestMetricsCalculator:
    """Tests for MetricsCalculator."""
    
    def test_calculate_all_basic(self, sample_equity_curve, sample_trades):
        """Test basic metrics calculation."""
        calc = MetricsCalculator()
        metrics = calc.calculate_all(sample_equity_curve, sample_trades)
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.trading_days == len(sample_equity_curve)
        assert metrics.total_return != 0
        assert not np.isnan(metrics.sharpe_ratio)
    
    def test_sharpe_ratio(self, sample_equity_curve):
        """Test Sharpe ratio calculation."""
        calc = MetricsCalculator(risk_free_rate=0.05)
        metrics = calc.calculate_all(sample_equity_curve)
        
        # Sharpe should be reasonable
        assert -5 < metrics.sharpe_ratio < 5
    
    def test_sortino_ratio(self, sample_equity_curve):
        """Test Sortino ratio calculation."""
        calc = MetricsCalculator()
        metrics = calc.calculate_all(sample_equity_curve)
        
        # Sortino should be >= Sharpe (theoretically)
        # In practice can vary based on return distribution
        assert not np.isnan(metrics.sortino_ratio)
    
    def test_max_drawdown(self, sample_equity_curve):
        """Test max drawdown calculation."""
        calc = MetricsCalculator()
        metrics = calc.calculate_all(sample_equity_curve)
        
        # Drawdown should be negative or zero
        assert metrics.max_drawdown <= 0
    
    def test_trade_metrics(self, sample_equity_curve, sample_trades):
        """Test trade-level metrics."""
        calc = MetricsCalculator()
        metrics = calc.calculate_all(sample_equity_curve, sample_trades)
        
        assert metrics.total_trades == len(sample_trades)
        assert 0 <= metrics.win_rate <= 100


class TestDrawdownCalculations:
    """Tests for drawdown calculations."""
    
    def test_calculate_drawdown_series(self, sample_equity_curve):
        """Test drawdown series calculation."""
        dd_df = calculate_drawdown_series(sample_equity_curve)
        
        assert "drawdown" in dd_df.columns
        assert "peak" in dd_df.columns
        assert dd_df["drawdown"].max() <= 0
    
    def test_calculate_monthly_returns(self, sample_equity_curve):
        """Test monthly returns calculation."""
        monthly = calculate_monthly_returns(sample_equity_curve)
        
        assert not monthly.empty
        assert "Year" in monthly.columns


# =============================================================================
# COST MODEL TESTS
# =============================================================================

class TestCostModels:
    """Tests for transaction cost models."""
    
    def test_simple_cost_model(self):
        """Test SimpleCostModel."""
        config = CostModelConfig(
            commission_per_contract=2.50,
            slippage_ticks=1.0,
            tick_size=0.01,
            contract_multiplier=1000,
        )
        model = SimpleCostModel(config)
        
        costs = model.calculate_costs(
            price=75.0,
            quantity=10,
            side=OrderSide.BUY
        )
        
        assert isinstance(costs, TransactionCosts)
        assert costs.commission == 25.0  # 10 * 2.50
        assert costs.slippage > 0
        assert costs.total > 0
    
    def test_volatility_adjusted_model(self):
        """Test VolatilityAdjustedCostModel."""
        model = VolatilityAdjustedCostModel()
        
        # Higher vol should mean higher costs
        costs_low_vol = model.calculate_costs(
            price=75.0, quantity=10, side=OrderSide.BUY,
            volatility=0.01
        )
        costs_high_vol = model.calculate_costs(
            price=75.0, quantity=10, side=OrderSide.BUY,
            volatility=0.04
        )
        
        assert costs_high_vol.slippage > costs_low_vol.slippage
    
    def test_market_impact_model(self):
        """Test MarketImpactCostModel."""
        model = MarketImpactCostModel()
        
        costs = model.calculate_costs(
            price=75.0, quantity=100, side=OrderSide.BUY,
            volatility=0.02, avg_daily_volume=50000
        )
        
        assert costs.market_impact > 0


# =============================================================================
# STRATEGY TESTS
# =============================================================================

class TestStrategies:
    """Tests for strategy implementations."""
    
    def test_buy_and_hold(self, sample_ohlcv_data):
        """Test BuyAndHoldStrategy."""
        strategy = BuyAndHoldStrategy()
        position = Position(symbol="CL1")
        
        signal = strategy.generate_signal(
            datetime.now(), sample_ohlcv_data, position
        )
        
        assert signal == Signal.LONG
    
    def test_ma_crossover(self, sample_ohlcv_data):
        """Test MACrossoverStrategy."""
        strategy = MACrossoverStrategy(fast_period=10, slow_period=30)
        position = Position(symbol="CL1")
        
        signal = strategy.generate_signal(
            datetime.now(), sample_ohlcv_data, position
        )
        
        assert signal in [Signal.LONG, Signal.SHORT, Signal.HOLD]
    
    def test_rsi_strategy(self, sample_ohlcv_data):
        """Test RSIMeanReversionStrategy."""
        strategy = RSIMeanReversionStrategy(rsi_period=14)
        position = Position(symbol="CL1")
        
        signal = strategy.generate_signal(
            datetime.now(), sample_ohlcv_data, position
        )
        
        assert signal in [Signal.LONG, Signal.SHORT, Signal.FLAT, Signal.HOLD]
    
    def test_bollinger_band_strategy(self, sample_ohlcv_data):
        """Test BollingerBandStrategy."""
        strategy = BollingerBandStrategy(period=20, num_std=2.0)
        position = Position(symbol="CL1")
        
        signal = strategy.generate_signal(
            datetime.now(), sample_ohlcv_data, position
        )
        
        assert signal in [Signal.LONG, Signal.SHORT, Signal.FLAT, Signal.HOLD]
    
    def test_momentum_strategy(self, sample_ohlcv_data):
        """Test MomentumStrategy."""
        strategy = MomentumStrategy(lookback=20)
        position = Position(symbol="CL1")
        
        signal = strategy.generate_signal(
            datetime.now(), sample_ohlcv_data, position
        )
        
        assert signal in [Signal.LONG, Signal.SHORT, Signal.HOLD]
    
    def test_custom_signal_strategy(self, sample_ohlcv_data):
        """Test create_strategy_from_signals."""
        def signal_func(data):
            # Simple: long if price > 20-day MA
            ma = data["PX_LAST"].rolling(20).mean()
            return (data["PX_LAST"] > ma).astype(int) * 2 - 1
        
        strategy = create_strategy_from_signals(signal_func, "CustomTest")
        position = Position(symbol="CL1")
        
        signal = strategy.generate_signal(
            datetime.now(), sample_ohlcv_data, position
        )
        
        assert signal in [Signal.LONG, Signal.SHORT, Signal.HOLD]


# =============================================================================
# EXECUTION TESTS
# =============================================================================

class TestExecution:
    """Tests for execution simulation."""
    
    def test_execution_simulator_market_order(self):
        """Test market order execution."""
        simulator = ExecutionSimulator()
        
        order = Order(
            timestamp=datetime.now(),
            symbol="CL1",
            side="BUY",
            quantity=10,
            order_type="MARKET"
        )
        
        record = simulator.submit_order(order)
        assert record.status == OrderStatus.PENDING
        
        fills = simulator.process_bar(
            datetime.now(),
            {"open": 75.0, "high": 76.0, "low": 74.0, "close": 75.5}
        )
        
        assert len(fills) == 1
        assert fills[0].quantity == 10
    
    def test_position_manager(self):
        """Test PositionManager."""
        pm = PositionManager(contract_multiplier=1000)
        
        fill = Fill(
            order_id="ORD-001",
            timestamp=datetime.now(),
            symbol="CL1",
            side="BUY",
            quantity=10,
            price=75.0,
            commission=25.0,
            slippage=10.0,
        )
        
        position = pm.update_position(fill, current_price=76.0)
        
        assert position.quantity == 10
        assert position.avg_price == 75.0
        assert position.unrealized_pnl == 10000  # (76-75) * 10 * 1000


# =============================================================================
# ENGINE TESTS
# =============================================================================

class TestBacktestEngine:
    """Tests for BacktestEngine."""
    
    def test_run_backtest(self, sample_ohlcv_data):
        """Test basic backtest run."""
        config = BacktestConfig(initial_capital=1_000_000)
        engine = BacktestEngine(config)
        
        strategy = BuyAndHoldStrategy()
        result = engine.run(strategy, sample_ohlcv_data, "CL1")
        
        assert isinstance(result, BacktestResult)
        assert len(result.equity_curve) > 0
        assert result.metrics.trading_days > 0
    
    def test_run_ma_strategy(self, sample_ohlcv_data):
        """Test MA crossover backtest."""
        strategy = MACrossoverStrategy(fast_period=10, slow_period=30)
        result = run_backtest(
            strategy, sample_ohlcv_data,
            initial_capital=1_000_000
        )
        
        assert isinstance(result, BacktestResult)
        assert result.strategy_name == "MACrossover(10,30)"
    
    def test_quick_backtest(self, sample_ohlcv_data):
        """Test quick_backtest function."""
        # Create simple signal
        prices = sample_ohlcv_data["PX_LAST"]
        signals = (prices > prices.rolling(20).mean()).astype(int)
        
        result = quick_backtest(
            signal_series=signals,
            price_series=prices,
            initial_capital=1_000_000,
            position_size=5
        )
        
        assert isinstance(result, BacktestResult)
    
    def test_multiple_strategies(self, sample_ohlcv_data):
        """Test running multiple strategies."""
        config = BacktestConfig(initial_capital=1_000_000)
        engine = BacktestEngine(config)
        
        strategies = [
            BuyAndHoldStrategy(),
            MACrossoverStrategy(fast_period=10, slow_period=30),
            RSIMeanReversionStrategy(rsi_period=14),
        ]
        
        results = engine.run_multiple(strategies, sample_ohlcv_data, "CL1")
        
        assert len(results) == 3
        for name, result in results.items():
            assert isinstance(result, BacktestResult)


# =============================================================================
# OPTIMIZATION TESTS
# =============================================================================

class TestOptimization:
    """Tests for optimization module."""
    
    def test_parameter_grid(self):
        """Test ParameterGrid generation."""
        grid = ParameterGrid({
            "fast_period": [5, 10, 15],
            "slow_period": [20, 30],
        })
        
        assert len(grid) == 6  # 3 * 2
        
        combos = list(grid)
        assert {"fast_period": 5, "slow_period": 20} in combos
    
    def test_strategy_optimizer(self, sample_ohlcv_data):
        """Test StrategyOptimizer."""
        optimizer = StrategyOptimizer(
            strategy_class=MACrossoverStrategy,
            param_grid={
                "fast_period": [5, 10],
                "slow_period": [20, 30],
            },
            config=OptimizationConfig(
                target_metric="sharpe_ratio",
                min_trades=5,
            )
        )
        
        result = optimizer.optimize(sample_ohlcv_data, "CL1")
        
        assert result.best_params is not None
        assert not result.all_results.empty
    
    def test_sensitivity_analysis(self, sample_ohlcv_data):
        """Test sensitivity analysis."""
        result = sensitivity_analysis(
            strategy_class=MACrossoverStrategy,
            base_params={"fast_period": 10, "slow_period": 30},
            param_to_vary="fast_period",
            param_values=[5, 10, 15, 20],
            data=sample_ohlcv_data,
            symbol="CL1"
        )
        
        assert len(result) == 4
        assert "sharpe" in result.columns


# =============================================================================
# REPORTING TESTS
# =============================================================================

class TestReporting:
    """Tests for reporting module."""
    
    def test_generate_summary_report(self, sample_ohlcv_data):
        """Test summary report generation."""
        strategy = BuyAndHoldStrategy()
        result = run_backtest(strategy, sample_ohlcv_data)
        
        report = generate_summary_report(result)
        
        assert isinstance(report, str)
        assert "BACKTEST REPORT" in report
        assert "Sharpe Ratio" in report
    
    def test_create_equity_chart(self, sample_ohlcv_data):
        """Test equity chart creation."""
        strategy = BuyAndHoldStrategy()
        result = run_backtest(strategy, sample_ohlcv_data)
        
        fig = create_equity_chart(result)
        
        # If plotly is available, we should get a figure
        # If not, None is acceptable
        if fig is not None:
            assert hasattr(fig, "data")


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for the full backtest workflow."""
    
    def test_full_workflow(self, sample_ohlcv_data):
        """Test complete backtest workflow."""
        # 1. Create strategy
        strategy = MACrossoverStrategy(
            fast_period=10, 
            slow_period=30,
            config=StrategyConfig(position_size=5)
        )
        
        # 2. Configure cost model
        cost_config = CostModelConfig(
            commission_per_contract=2.50,
            slippage_ticks=1.0,
        )
        cost_model = SimpleCostModel(cost_config)
        
        # 3. Configure backtest
        bt_config = BacktestConfig(
            initial_capital=1_000_000,
            commission_per_contract=2.50,
            slippage_pct=0.01,
        )
        
        # 4. Run backtest
        engine = BacktestEngine(bt_config, cost_model)
        result = engine.run(strategy, sample_ohlcv_data, "CL1")
        
        # 5. Verify results
        assert result.metrics.trading_days > 0
        assert len(result.equity_curve) > 0
        
        # 6. Generate report
        report = generate_summary_report(result)
        assert len(report) > 0
        
        # 7. Calculate additional metrics
        calc = MetricsCalculator()
        metrics = calc.calculate_all(result.equity_curve, result.trades)
        assert not np.isnan(metrics.sharpe_ratio)
    
    def test_compare_strategies_workflow(self, sample_ohlcv_data):
        """Test comparing multiple strategies."""
        strategies = [
            BuyAndHoldStrategy(),
            MACrossoverStrategy(10, 30),
            RSIMeanReversionStrategy(14),
        ]
        
        engine = BacktestEngine()
        results = engine.run_multiple(strategies, sample_ohlcv_data)
        
        # Compare
        comparison = engine.compare_strategies(results)
        
        assert len(comparison) == 3
        assert "sharpe_ratio" in comparison.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
