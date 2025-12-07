"""
Execution Module Tests
======================
Tests for the execution system.
"""

import os
import tempfile

import pytest

from core.execution.algorithms import (
    AlgorithmConfig,
    AlgorithmType,
    ImplementationShortfall,
    POVAlgorithm,
    TWAPAlgorithm,
    VWAPAlgorithm,
    get_execution_algorithm,
)
from core.execution.brokers import BrokerStatus, SimulatedBroker, SimulatorConfig

# Import execution modules
from core.execution.oms import (
    Order,
    OrderManager,
    OrderSide,
    OrderStatus,
    OrderType,
    OrderUpdate,
)
from core.execution.paper_trading import PaperTradingConfig, PaperTradingEngine
from core.execution.sizing import (
    ATRBasedSizing,
    FixedFractional,
    FixedSizer,
    KellyCriterion,
    SizingConfig,
    SizingMethod,
    SizingResult,
    VaRBasedSizing,
    VolatilityTargeting,
    calculate_optimal_size,
    get_position_sizer,
)


# =============================================================================
# ORDER MANAGEMENT SYSTEM TESTS
# =============================================================================
class TestOrderManagementSystem:
    """Tests for the Order Management System."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_orders.db")
            yield db_path

    @pytest.fixture
    def oms(self, temp_db):
        """Create OMS instance for testing."""
        return OrderManager(db_path=temp_db)

    def test_create_market_order(self, oms):
        """Test creating a market order."""
        order = oms.create_order(
            symbol="CL1",
            side=OrderSide.BUY,
            quantity=5,
            order_type=OrderType.MARKET,
            strategy="test_strategy",
        )

        assert order is not None
        assert order.order_id.startswith("ORD-")
        assert order.symbol == "CL1"
        assert order.side == OrderSide.BUY
        assert order.quantity == 5
        assert order.status == OrderStatus.CREATED

    def test_create_limit_order(self, oms):
        """Test creating a limit order."""
        order = oms.create_order(
            symbol="CO1",
            side=OrderSide.SELL,
            quantity=10,
            order_type=OrderType.LIMIT,
            limit_price=78.50,
        )

        assert order.order_type == OrderType.LIMIT
        assert order.limit_price == 78.50

    def test_order_validation_fails_for_limit_without_price(self, oms):
        """Test that limit orders require a price."""
        with pytest.raises(ValueError, match="Limit price required"):
            oms.create_order(
                symbol="CL1",
                side=OrderSide.BUY,
                quantity=5,
                order_type=OrderType.LIMIT,
            )

    def test_submit_order(self, oms):
        """Test submitting an order."""
        order = oms.create_order(
            symbol="CL1",
            side=OrderSide.BUY,
            quantity=5,
        )

        submitted = oms.submit_order(order.order_id)

        assert submitted.status == OrderStatus.PENDING
        assert submitted.submitted_at is not None

    def test_process_fill(self, oms):
        """Test processing an order fill."""
        order = oms.create_order(
            symbol="CL1",
            side=OrderSide.BUY,
            quantity=5,
        )
        oms.submit_order(order.order_id)

        # Update to working status
        oms.update_order(order.order_id, OrderUpdate(status=OrderStatus.WORKING))

        # Process fill
        filled = oms.process_fill(
            order_id=order.order_id,
            fill_quantity=5,
            fill_price=75.50,
            commission=12.50,
        )

        assert filled.status == OrderStatus.FILLED
        assert filled.filled_quantity == 5
        assert filled.avg_fill_price == 75.50
        assert filled.commission == 12.50

    def test_partial_fill(self, oms):
        """Test partial fill processing."""
        order = oms.create_order(
            symbol="CL1",
            side=OrderSide.BUY,
            quantity=10,
        )
        oms.submit_order(order.order_id)
        oms.update_order(order.order_id, OrderUpdate(status=OrderStatus.WORKING))

        # First partial fill
        partial = oms.process_fill(order.order_id, 4, 75.00, 10.00)

        assert partial.status == OrderStatus.PARTIALLY_FILLED
        assert partial.filled_quantity == 4
        assert partial.remaining_quantity == 6

        # Second fill completes order
        complete = oms.process_fill(order.order_id, 6, 75.20, 15.00)

        assert complete.status == OrderStatus.FILLED
        assert complete.filled_quantity == 10

    def test_cancel_order(self, oms):
        """Test cancelling an order."""
        order = oms.create_order(symbol="CL1", side=OrderSide.BUY, quantity=5)
        oms.submit_order(order.order_id)
        oms.update_order(order.order_id, OrderUpdate(status=OrderStatus.WORKING))

        oms.cancel_order(order.order_id)
        cancelled = oms.confirm_cancel(order.order_id)

        assert cancelled.status == OrderStatus.CANCELLED

    def test_get_active_orders(self, oms):
        """Test retrieving active orders."""
        # Create multiple orders
        order1 = oms.create_order(symbol="CL1", side=OrderSide.BUY, quantity=5)
        oms.create_order(symbol="CO1", side=OrderSide.SELL, quantity=3)

        oms.submit_order(order1.order_id)
        oms.update_order(order1.order_id, OrderUpdate(status=OrderStatus.WORKING))

        active = oms.get_active_orders()

        assert len(active) == 1
        assert active[0].order_id == order1.order_id

    def test_order_statistics(self, oms):
        """Test order statistics calculation."""
        # Create and fill some orders
        for _i in range(3):
            order = oms.create_order(symbol="CL1", side=OrderSide.BUY, quantity=5)
            oms.submit_order(order.order_id)
            oms.update_order(order.order_id, OrderUpdate(status=OrderStatus.WORKING))
            oms.process_fill(order.order_id, 5, 75.0, 12.50)

        stats = oms.get_statistics()

        assert stats["total_orders"] == 3
        assert stats["filled"] == 3
        assert stats["fill_rate"] == 100.0


# =============================================================================
# POSITION SIZING TESTS
# =============================================================================
class TestPositionSizing:
    """Tests for position sizing algorithms."""

    @pytest.fixture
    def base_config(self):
        """Base sizing configuration."""
        return SizingConfig(
            account_value=1_000_000,
            max_position_pct=0.25,
            max_position_contracts=50,
            risk_per_trade_pct=0.02,
            target_volatility=0.15,
        )

    def test_fixed_sizer(self, base_config):
        """Test fixed position sizing."""
        base_config.max_position_contracts = 100  # Increase limit for this test
        base_config.max_position_pct = 0.80  # Increase limit to allow 10 contracts
        sizer = FixedSizer(base_config, fixed_contracts=10)
        result = sizer.calculate_size(price=75.0)

        assert result.contracts == 10
        assert result.method == SizingMethod.FIXED

    def test_fixed_fractional(self, base_config):
        """Test fixed fractional sizing."""
        sizer = FixedFractional(base_config)
        result = sizer.calculate_size(price=75.0, volatility=0.25, stop_loss_pct=0.02)

        assert result.contracts > 0
        assert result.method == SizingMethod.FIXED_FRACTIONAL
        assert result.risk_amount <= base_config.account_value * base_config.risk_per_trade_pct * 1.5

    def test_kelly_criterion(self, base_config):
        """Test Kelly criterion sizing."""
        base_config.kelly_fraction = 0.25
        sizer = KellyCriterion(base_config)

        result = sizer.calculate_size(
            price=75.0,
            volatility=0.25,
            win_rate=0.55,
            avg_win_loss_ratio=1.5,
        )

        assert result.contracts > 0
        assert result.method == SizingMethod.KELLY
        assert "Kelly" in result.rationale

    def test_volatility_targeting(self, base_config):
        """Test volatility targeting sizing."""
        base_config.target_volatility = 0.15
        base_config.max_position_pct = 0.80  # Allow larger positions for this test
        sizer = VolatilityTargeting(base_config)

        # Low volatility asset = larger position
        result_low_vol = sizer.calculate_size(price=75.0, volatility=0.10)

        # High volatility asset = smaller position
        result_high_vol = sizer.calculate_size(price=75.0, volatility=0.40)

        # Low vol should give larger position (before limits)
        assert result_low_vol.contracts >= result_high_vol.contracts

    def test_atr_based_sizing(self, base_config):
        """Test ATR-based sizing."""
        sizer = ATRBasedSizing(base_config, atr_multiplier=2.0)
        result = sizer.calculate_size(price=75.0, volatility=0.25, atr=1.5)

        assert result.contracts > 0
        assert result.method == SizingMethod.ATR_BASED

    def test_var_based_sizing(self, base_config):
        """Test VaR-based sizing."""
        sizer = VaRBasedSizing(base_config, max_var_pct=0.02)
        result = sizer.calculate_size(price=75.0, volatility=0.25)

        assert result.contracts > 0
        assert result.method == SizingMethod.VAR_BASED

    def test_position_limits_applied(self, base_config):
        """Test that position limits are applied."""
        base_config.max_position_contracts = 5  # Set a low limit
        base_config.max_position_pct = 0.80  # Don't let pct limit kick in first
        sizer = VolatilityTargeting(base_config)

        # Very low vol would normally give huge position
        result = sizer.calculate_size(price=75.0, volatility=0.05)

        assert result.contracts <= 5
        # Either we hit the limit and have adjustments, or we naturally stayed below
        assert result.contracts > 0

    def test_get_position_sizer_factory(self, base_config):
        """Test position sizer factory function."""
        base_config.method = SizingMethod.KELLY
        sizer = get_position_sizer(base_config)

        assert isinstance(sizer, KellyCriterion)

    def test_calculate_optimal_size_convenience(self):
        """Test convenience function."""
        result = calculate_optimal_size(
            price=75.0,
            volatility=0.25,
            account_value=1_000_000,
            method=SizingMethod.VOLATILITY_TARGET,
        )

        assert isinstance(result, SizingResult)
        assert result.contracts > 0


# =============================================================================
# EXECUTION ALGORITHM TESTS
# =============================================================================
class TestExecutionAlgorithms:
    """Tests for execution algorithms."""

    @pytest.fixture
    def sample_order(self):
        """Create sample order for testing."""
        return Order(
            order_id="TEST-001",
            symbol="CL1",
            side=OrderSide.BUY,
            quantity=20,
        )

    def test_twap_generates_schedule(self, sample_order):
        """Test TWAP schedule generation."""
        config = AlgorithmConfig(
            algo_type=AlgorithmType.TWAP,
            duration_minutes=60,
            num_slices=12,
            randomize_timing=False,
            randomize_size=False,
        )

        algo = TWAPAlgorithm(config)
        slices = algo.generate_schedule(sample_order, current_price=75.0)

        assert len(slices) == 12
        total_qty = sum(s.quantity for s in slices)
        assert total_qty == 20

    def test_vwap_generates_volume_weighted_schedule(self, sample_order):
        """Test VWAP schedule follows volume profile."""
        config = AlgorithmConfig(
            algo_type=AlgorithmType.VWAP,
            duration_minutes=120,
            num_slices=8,
            volume_profile=[0.15, 0.12, 0.09, 0.08, 0.10, 0.12, 0.18, 0.16],
            randomize_timing=False,
            randomize_size=False,
        )

        algo = VWAPAlgorithm(config)
        slices = algo.generate_schedule(sample_order, current_price=75.0)

        assert len(slices) == 8
        # First slice should be larger (15% of volume)
        assert slices[0].quantity >= slices[3].quantity  # Bigger than midday

    def test_pov_generates_schedule(self, sample_order):
        """Test POV schedule generation."""
        config = AlgorithmConfig(
            algo_type=AlgorithmType.POV,
            duration_minutes=60,
            num_slices=6,
            participation_rate=0.10,
        )

        algo = POVAlgorithm(config)
        slices = algo.generate_schedule(sample_order, current_price=75.0)

        assert len(slices) <= 6
        total_qty = sum(s.quantity for s in slices)
        assert total_qty == 20

    def test_is_front_loads_with_high_urgency(self, sample_order):
        """Test Implementation Shortfall front-loads with high urgency."""
        config_aggressive = AlgorithmConfig(
            algo_type=AlgorithmType.IS,
            duration_minutes=60,
            num_slices=6,
            urgency=0.9,
            randomize_size=False,
        )

        config_passive = AlgorithmConfig(
            algo_type=AlgorithmType.IS,
            duration_minutes=60,
            num_slices=6,
            urgency=0.1,
            randomize_size=False,
        )

        algo_agg = ImplementationShortfall(config_aggressive)
        algo_pas = ImplementationShortfall(config_passive)

        slices_agg = algo_agg.generate_schedule(sample_order, current_price=75.0)
        slices_pas = algo_pas.generate_schedule(sample_order, current_price=75.0)

        # Aggressive should have larger first slice
        assert slices_agg[0].quantity >= slices_pas[0].quantity

    def test_algorithm_progress_tracking(self, sample_order):
        """Test algorithm progress tracking."""
        config = AlgorithmConfig(num_slices=4, randomize_size=False)
        algo = TWAPAlgorithm(config)
        slices = algo.generate_schedule(sample_order, current_price=75.0)

        # Mark some slices as executed
        algo.mark_slice_executed(slices[0].slice_id, slices[0].quantity, 75.10)
        algo.mark_slice_executed(slices[1].slice_id, slices[1].quantity, 75.05)

        progress = algo.get_progress()

        assert progress.completed_slices == 2
        assert progress.pending_slices == 2
        assert progress.executed_quantity == slices[0].quantity + slices[1].quantity

    def test_get_execution_algorithm_factory(self):
        """Test algorithm factory function."""
        config = AlgorithmConfig(algo_type=AlgorithmType.VWAP)
        algo = get_execution_algorithm(config)

        assert isinstance(algo, VWAPAlgorithm)


# =============================================================================
# BROKER SIMULATOR TESTS
# =============================================================================
class TestSimulatedBroker:
    """Tests for the simulated broker."""

    @pytest.fixture
    def broker(self):
        """Create simulated broker for testing."""
        config = SimulatorConfig(
            initial_capital=1_000_000,
            slippage_bps=1.0,
            commission_per_contract=2.50,
        )
        return SimulatedBroker(config)

    def test_broker_connects(self, broker):
        """Test broker connection."""
        assert broker.is_connected
        assert broker.status == BrokerStatus.CONNECTED

    def test_submit_market_order(self, broker):
        """Test submitting a market order."""
        broker.set_price("CL1", 75.0)

        order_id = broker.submit_order(
            symbol="CL1",
            side="BUY",
            quantity=5,
            order_type="MARKET",
        )

        assert order_id is not None
        assert order_id.startswith("SIM-")

        status = broker.get_order_status(order_id)
        assert status["status"] == "FILLED"

    def test_submit_limit_order(self, broker):
        """Test submitting a limit order."""
        broker.set_price("CL1", 75.0)

        # Limit below market - should not fill
        order_id = broker.submit_order(
            symbol="CL1",
            side="BUY",
            quantity=5,
            order_type="LIMIT",
            limit_price=74.0,
        )

        status = broker.get_order_status(order_id)
        assert status["status"] == "WORKING"

        # Price drops to limit - should fill
        broker.set_price("CL1", 73.5)
        broker.process_pending_orders()

        status = broker.get_order_status(order_id)
        assert status["status"] == "FILLED"

    def test_position_tracking(self, broker):
        """Test position tracking after fills."""
        broker.set_price("CL1", 75.0)

        # Buy 5 contracts
        broker.submit_order("CL1", "BUY", 5, "MARKET")

        positions = broker.get_positions()
        assert "CL1" in positions
        assert positions["CL1"]["quantity"] == 5

    def test_pnl_calculation(self, broker):
        """Test P&L calculation."""
        broker.set_price("CL1", 75.0)
        broker.submit_order("CL1", "BUY", 5, "MARKET")

        # Price goes up
        broker.set_price("CL1", 77.0)

        positions = broker.get_positions()
        # Should have unrealized profit
        assert positions["CL1"]["unrealized_pnl"] > 0

    def test_cancel_order(self, broker):
        """Test order cancellation."""
        broker.set_price("CL1", 75.0)

        order_id = broker.submit_order(
            symbol="CL1",
            side="BUY",
            quantity=5,
            order_type="LIMIT",
            limit_price=74.0,
        )

        success = broker.cancel_order(order_id)
        assert success

        status = broker.get_order_status(order_id)
        assert status["status"] == "CANCELLED"

    def test_account_info(self, broker):
        """Test account information."""
        account = broker.get_account_info()

        assert account["initial_capital"] == 1_000_000
        assert account["nav"] == 1_000_000  # Before any trades

    def test_reset(self, broker):
        """Test broker reset."""
        broker.set_price("CL1", 75.0)
        broker.submit_order("CL1", "BUY", 5, "MARKET")

        broker.reset()

        positions = broker.get_positions()
        assert len(positions) == 0

        account = broker.get_account_info()
        assert account["nav"] == 1_000_000


# =============================================================================
# PAPER TRADING ENGINE TESTS
# =============================================================================
class TestPaperTradingEngine:
    """Tests for the paper trading engine."""

    @pytest.fixture
    def engine(self):
        """Create paper trading engine for testing."""
        config = PaperTradingConfig(
            initial_capital=1_000_000,
            max_position_per_symbol=50,
        )
        return PaperTradingEngine(config)

    def test_start_session(self, engine):
        """Test starting a paper trading session."""
        session_id = engine.start_session()

        assert session_id is not None
        assert engine._is_active

    def test_submit_order(self, engine):
        """Test submitting an order."""
        engine.start_session()
        engine.update_prices({"CL1": 75.0})

        order = engine.submit_order(
            symbol="CL1",
            side="BUY",
            quantity=5,
            order_type="MARKET",
            strategy="test",
        )

        assert order is not None
        assert order.order_id is not None

    def test_position_tracking(self, engine):
        """Test position tracking."""
        engine.start_session()
        engine.update_prices({"CL1": 75.0})

        engine.submit_order("CL1", "BUY", 5, "MARKET")

        positions = engine.get_positions()
        assert "CL1" in positions

    def test_pnl_summary(self, engine):
        """Test P&L summary."""
        engine.start_session()
        engine.update_prices({"CL1": 75.0})

        engine.submit_order("CL1", "BUY", 5, "MARKET")
        engine.update_prices({"CL1": 77.0})

        summary = engine.get_pnl_summary()

        assert "total_pnl" in summary
        assert "return_pct" in summary

    def test_stop_session(self, engine):
        """Test stopping a session."""
        engine.start_session()
        engine.update_prices({"CL1": 75.0})
        engine.submit_order("CL1", "BUY", 5, "MARKET")

        session = engine.stop_session()

        assert session is not None
        assert not engine._is_active


# =============================================================================
# INTEGRATION TESTS
# =============================================================================
class TestExecutionIntegration:
    """Integration tests for execution system."""

    def test_full_order_lifecycle(self):
        """Test complete order lifecycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "orders.db")
            oms = OrderManager(db_path=db_path)
            broker = SimulatedBroker()
            broker.set_price("CL1", 75.0)

            # Create order
            order = oms.create_order(
                symbol="CL1",
                side=OrderSide.BUY,
                quantity=5,
            )

            # Submit
            oms.submit_order(order.order_id)

            # Send to broker
            broker_order_id = broker.submit_order(
                symbol=order.symbol,
                side=order.side.value,
                quantity=order.quantity,
                internal_order_id=order.order_id,
            )

            # Update OMS with broker ID
            oms.update_order(order.order_id, OrderUpdate(
                broker_order_id=broker_order_id,
                status=OrderStatus.WORKING,
            ))

            # Broker fills order
            broker_status = broker.get_order_status(broker_order_id)

            # Update OMS with fill
            oms.process_fill(
                order.order_id,
                broker_status["filled_quantity"],
                broker_status["avg_fill_price"],
            )

            # Verify final state
            final_order = oms.get_order(order.order_id)
            assert final_order.status == OrderStatus.FILLED
            assert final_order.filled_quantity == 5

    def test_algorithm_with_broker(self):
        """Test execution algorithm with broker simulation."""
        broker = SimulatedBroker()
        broker.set_price("CL1", 75.0)

        order = Order(
            order_id="TEST-001",
            symbol="CL1",
            side=OrderSide.BUY,
            quantity=12,
        )

        config = AlgorithmConfig(
            num_slices=4,
            randomize_timing=False,
            randomize_size=False,
        )
        algo = TWAPAlgorithm(config)
        slices = algo.generate_schedule(order, current_price=75.0)

        # Execute slices
        for slice_ in slices:
            broker_id = broker.submit_order(
                symbol=slice_.parent_order_id,
                side="BUY",
                quantity=slice_.quantity,
            )

            status = broker.get_order_status(broker_id)
            algo.mark_slice_executed(
                slice_.slice_id,
                status["filled_quantity"],
                status["avg_fill_price"],
            )

        progress = algo.get_progress()
        assert progress.executed_quantity == 12
        assert progress.pct_complete == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
