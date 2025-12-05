"""
Tests for Signals Module
========================
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.signals.aggregator import SignalAggregator
from core.signals.fundamental import FundamentalSignals
from core.signals.technical import TechnicalSignals


class TestTechnicalSignals:
    """Tests for technical signal generation."""

    @pytest.fixture
    def sample_prices(self):
        """Generate sample price series."""
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        prices = 72.50 * np.exp(np.cumsum(np.random.normal(0.0002, 0.015, 100)))
        return pd.Series(prices, index=dates)

    def test_ma_crossover_signal(self, sample_prices):
        """Test MA crossover signal."""
        signals = TechnicalSignals()
        result = signals.ma_crossover_signal(sample_prices)

        assert 'signal' in result
        assert result['signal'] in ['BUY', 'SELL', 'HOLD']
        assert 'confidence' in result
        assert 0 <= result['confidence'] <= 100

    def test_rsi_signal(self, sample_prices):
        """Test RSI signal."""
        signals = TechnicalSignals()
        result = signals.rsi_signal(sample_prices)

        assert 'signal' in result
        assert 'rsi' in result
        assert 0 <= result['rsi'] <= 100

    def test_bollinger_band_signal(self, sample_prices):
        """Test Bollinger Band signal."""
        signals = TechnicalSignals()
        result = signals.bollinger_band_signal(sample_prices)

        assert 'signal' in result
        assert 'upper_band' in result
        assert 'lower_band' in result
        assert result['upper_band'] > result['lower_band']

    def test_composite_signal(self, sample_prices):
        """Test composite technical signal."""
        signals = TechnicalSignals()
        result = signals.generate_composite_signal(sample_prices)

        assert 'signal' in result
        assert result['signal'] in ['LONG', 'SHORT', 'NEUTRAL']
        assert 'confidence' in result
        assert 'components' in result


class TestFundamentalSignals:
    """Tests for fundamental signal generation."""

    def test_inventory_surprise_signal(self):
        """Test inventory surprise signal."""
        signals = FundamentalSignals()
        result = signals.inventory_surprise_signal(
            actual_change=-2.5,
            expected_change=-1.0,
            current_level=425.0
        )

        assert 'signal' in result
        assert result['signal'] in ['LONG', 'SHORT', 'NEUTRAL']
        assert 'surprise' in result
        assert result['surprise'] == pytest.approx(-1.5, rel=0.01)

    def test_opec_compliance_signal(self):
        """Test OPEC compliance signal."""
        signals = FundamentalSignals()
        result = signals.opec_compliance_signal(
            overall_compliance=94.0,
            production_deviation=0.3
        )

        assert 'signal' in result
        assert 'description' in result

    def test_term_structure_signal(self):
        """Test term structure signal."""
        signals = FundamentalSignals()
        result = signals.term_structure_signal(
            m1_m2_spread=0.35,
            m1_m12_spread=1.8,
            curve_slope=0.15
        )

        assert 'signal' in result
        assert 'description' in result

    def test_composite_fundamental_signal(self):
        """Test composite fundamental signal."""
        signals = FundamentalSignals()
        result = signals.generate_composite_fundamental_signal(
            inventory_data={"level": 430, "change": -2.1, "expectation": -1.5},
            opec_data={"compliance": 94, "deviation": 0.3},
            curve_data={"m1_m2_spread": 0.35, "m1_m12_spread": 1.8, "slope": 0.15},
            crack_spread=28.5,
            turnaround_data={"offline": 800, "upcoming": 600}
        )

        assert 'signal' in result
        assert result['signal'] in ['LONG', 'SHORT', 'NEUTRAL']
        assert 'components' in result


class TestSignalAggregator:
    """Tests for signal aggregation."""

    def test_aggregate_signals(self):
        """Test signal aggregation."""
        aggregator = SignalAggregator()

        tech_signal = {"signal": "LONG", "confidence": 70}
        fund_signal = {"signal": "LONG", "confidence": 65}

        result = aggregator.aggregate_signals(
            technical_signal=tech_signal,
            fundamental_signal=fund_signal,
            instrument="CL1 Comdty",
            current_price=72.50
        )

        assert result.direction in ['LONG', 'SHORT', 'NEUTRAL']
        assert result.confidence > 0
        assert result.entry_price == 72.50
        assert result.stop_loss < result.target_price

    def test_get_signal_performance(self):
        """Test signal performance tracking."""
        aggregator = SignalAggregator()

        # Generate some test signals
        for _ in range(5):
            aggregator.aggregate_signals(
                technical_signal={"signal": "LONG", "confidence": 70},
                fundamental_signal={"signal": "LONG", "confidence": 65},
                instrument="CL1 Comdty",
                current_price=72.50
            )

        perf = aggregator.get_signal_performance()

        assert 'total_signals' in perf
        assert perf['total_signals'] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
