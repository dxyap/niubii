"""
Tests for Risk Module
=====================
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.risk.limits import RiskLimits
from core.risk.monitor import RiskMonitor
from core.risk.var import VaRCalculator


class TestVaRCalculator:
    """Tests for VaR calculations."""

    @pytest.fixture
    def sample_positions(self):
        """Create sample positions."""
        return {
            "CL1 Comdty": {"quantity": 45, "price": 72.50},
            "CO1 Comdty": {"quantity": -15, "price": 77.20},
        }

    @pytest.fixture
    def sample_returns(self):
        """Generate sample returns."""
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
        returns = pd.DataFrame({
            "CL1 Comdty": np.random.normal(0.0002, 0.02, 252),
            "CO1 Comdty": np.random.normal(0.0002, 0.02, 252),
        }, index=dates)
        return returns

    def test_parametric_var(self, sample_positions, sample_returns):
        """Test parametric VaR calculation."""
        calc = VaRCalculator(confidence_level=0.95)
        result = calc.calculate_parametric_var(sample_positions, sample_returns)

        assert 'var' in result
        assert result['var'] > 0
        assert 'portfolio_value' in result
        assert result['method'] == 'parametric'

    def test_historical_var(self, sample_positions, sample_returns):
        """Test historical VaR calculation."""
        calc = VaRCalculator(confidence_level=0.95)
        result = calc.calculate_historical_var(sample_positions, sample_returns)

        assert 'var' in result
        assert result['var'] >= 0
        assert result['method'] == 'historical'

    def test_expected_shortfall(self, sample_positions, sample_returns):
        """Test expected shortfall calculation."""
        calc = VaRCalculator(confidence_level=0.95)
        result = calc.calculate_expected_shortfall(sample_positions, sample_returns)

        assert 'cvar' in result
        assert 'var' in result
        assert result['cvar'] >= result['var']  # CVaR should be >= VaR

    def test_parametric_var_handles_missing_returns(self, sample_positions, sample_returns):
        """Parametric VaR should fall back to assumed vol when data missing."""
        positions = dict(sample_positions)
        positions["HO1 Comdty"] = {"quantity": 20, "price": 2.45}

        calc = VaRCalculator(confidence_level=0.95)
        result = calc.calculate_parametric_var(positions, sample_returns)

        assert result["var"] > 0
        assert result["method"] == "parametric"

    def test_parametric_var_uses_correlation_matrix(self, sample_returns):
        """Ensure optional correlation matrix is incorporated."""
        positions = {
            "CL1 Comdty": {"quantity": 30, "price": 72.5},
            "CO1 Comdty": {"quantity": -20, "price": 77.0},
            "XB1 Comdty": {"quantity": 15, "price": 2.1},
        }
        extended_returns = sample_returns.copy()
        extended_returns["XB1 Comdty"] = np.random.normal(0.0001, 0.03, len(sample_returns))

        correlation = pd.DataFrame(
            [
                [1.0, 0.85, 0.40],
                [0.85, 1.0, 0.35],
                [0.40, 0.35, 1.0],
            ],
            index=list(positions.keys()),
            columns=list(positions.keys()),
        )

        calc = VaRCalculator(confidence_level=0.95)
        result = calc.calculate_parametric_var(positions, extended_returns, correlation)

        assert result["var"] > 0

    def test_stress_test(self, sample_positions):
        """Test stress testing."""
        calc = VaRCalculator()

        scenarios = {
            "oil_shock_up": {"factors": {"crude_oil": 0.10}},
            "oil_shock_down": {"factors": {"crude_oil": -0.10}},
        }

        results = calc.run_stress_test(sample_positions, scenarios)

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 2


class TestRiskLimits:
    """Tests for risk limits."""

    def test_position_limit_check(self, tmp_path):
        """Test position limit checking."""
        limits = RiskLimits(config_path=str(tmp_path / "limits.yaml"))

        result = limits.check_position_limit(
            ticker="CL1 Comdty",
            current_quantity=50,
            proposed_quantity=10,
            price=72.50
        )

        assert 'status' in result
        assert 'approved' in result
        assert 'contract_utilization_pct' in result

    def test_var_limit_check(self, tmp_path):
        """Test VaR limit checking."""
        limits = RiskLimits(config_path=str(tmp_path / "limits.yaml"))

        result = limits.check_var_limit(
            current_var=250000,
            proposed_var_impact=50000
        )

        assert 'status' in result
        assert 'approved' in result
        assert 'utilization_pct' in result

    def test_concentration_check(self, tmp_path):
        """Test concentration limit checking."""
        limits = RiskLimits(config_path=str(tmp_path / "limits.yaml"))

        positions = {
            "CL1 Comdty": {"quantity": 45, "price": 72.50},
            "CO1 Comdty": {"quantity": 15, "price": 77.20},
        }

        result = limits.check_concentration(positions, portfolio_value=5000000)

        assert 'status' in result
        assert 'concentrations' in result


class TestRiskMonitor:
    """Tests for risk monitoring."""

    def test_generate_alert(self):
        """Test alert generation."""
        monitor = RiskMonitor()

        alert = monitor.generate_alert(
            alert_type="limit",
            severity="WARNING",
            metric_name="VaR",
            current_value=280000,
            limit_value=375000,
            utilization_pct=75
        )

        assert alert.severity == "WARNING"
        assert alert.metric_name == "VaR"

    def test_check_and_alert(self):
        """Test automatic alert generation."""
        monitor = RiskMonitor()

        # Should generate warning (> 75%)
        alert = monitor.check_and_alert(
            metric_name="VaR",
            current_value=300000,
            limit_value=375000
        )

        assert alert is not None
        assert alert.severity in ["WARNING", "CRITICAL", "BREACH"]

    def test_calculate_drawdown(self):
        """Test drawdown calculation."""
        monitor = RiskMonitor()

        # Record some P&L
        cumulative = 0
        for pnl in [1000, 2000, -500, 3000, -2000, 1000]:
            cumulative += pnl
            monitor.record_pnl(pnl, cumulative)

        drawdown = monitor.calculate_drawdown()

        assert 'current_drawdown' in drawdown
        assert 'max_drawdown' in drawdown

    def test_get_risk_summary(self):
        """Test risk summary generation."""
        monitor = RiskMonitor()

        positions = {
            "CL1 Comdty": {"quantity": 45, "price": 72.50},
        }

        summary = monitor.get_risk_summary(
            positions=positions,
            current_var=250000,
            var_limit=375000
        )

        assert 'var' in summary
        assert 'exposure' in summary
        assert 'alerts' in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
