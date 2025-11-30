"""
Tests for Analytics Module
==========================
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.analytics.curves import CurveAnalyzer
from core.analytics.spreads import SpreadAnalyzer
from core.analytics.fundamentals import FundamentalAnalyzer


class TestCurveAnalyzer:
    """Tests for curve analysis."""
    
    @pytest.fixture
    def sample_curve(self):
        """Create sample curve data."""
        return pd.DataFrame({
            'month': range(1, 13),
            'price': [72.50 + i * 0.15 for i in range(12)],
            'days_to_expiry': [30 * i for i in range(1, 13)],
            'ticker': [f'CL{i} Comdty' for i in range(1, 13)],
        })
    
    def test_analyze_curve(self, sample_curve):
        """Test curve analysis."""
        analyzer = CurveAnalyzer()
        result = analyzer.analyze_curve(sample_curve)
        
        assert 'structure' in result
        assert 'overall_slope' in result
        assert 'm1_m2_spread' in result
        assert 'roll_yield_annual_pct' in result
    
    def test_calculate_calendar_spreads(self, sample_curve):
        """Test calendar spread calculation."""
        analyzer = CurveAnalyzer()
        spreads = analyzer.calculate_calendar_spreads(sample_curve)
        
        assert isinstance(spreads, pd.DataFrame)
        assert len(spreads) > 0
        assert 'spread_value' in spreads.columns
    
    def test_calculate_roll_yield(self, sample_curve):
        """Test roll yield calculation."""
        analyzer = CurveAnalyzer()
        roll = analyzer.calculate_roll_yield(sample_curve)
        
        assert 'roll_cost' in roll
        assert 'roll_yield' in roll
        assert 'roll_yield_annual_pct' in roll


class TestSpreadAnalyzer:
    """Tests for spread analysis."""
    
    def test_wti_brent_spread(self):
        """Test WTI-Brent spread calculation."""
        analyzer = SpreadAnalyzer()
        result = analyzer.calculate_wti_brent_spread(72.50, 77.20)
        
        assert 'spread' in result
        assert result['spread'] == pytest.approx(-4.70, rel=0.01)
        assert 'direction' in result
    
    def test_crack_spread_321(self):
        """Test 3-2-1 crack spread calculation."""
        analyzer = SpreadAnalyzer()
        result = analyzer.calculate_crack_spread(
            crude_price=72.50,
            gasoline_price=2.20,
            distillate_price=2.50,
            crack_type="3-2-1"
        )
        
        assert 'crack_spread' in result
        assert result['crack_spread'] > 0
        assert 'margin_pct' in result
    
    def test_spread_zscore(self):
        """Test spread z-score calculation."""
        analyzer = SpreadAnalyzer()
        
        # Create sample historical spreads
        historical = pd.Series(np.random.normal(-4.5, 1.0, 100))
        
        result = analyzer.calculate_spread_zscore(
            current_spread=-6.0,
            historical_spreads=historical
        )
        
        assert 'zscore' in result
        assert 'percentile' in result
        assert 'signal' in result


class TestFundamentalAnalyzer:
    """Tests for fundamental analysis."""
    
    def test_analyze_inventory(self):
        """Test inventory analysis."""
        analyzer = FundamentalAnalyzer()
        result = analyzer.analyze_inventory(
            current_level=430.0,
            change=-2.1,
            expectation=-1.5
        )
        
        assert 'surprise' in result
        assert result['surprise'] == pytest.approx(-0.6, rel=0.01)
        assert 'surprise_signal' in result
        assert 'percentile' in result
    
    def test_analyze_cushing_stocks(self):
        """Test Cushing stock analysis."""
        analyzer = FundamentalAnalyzer()
        result = analyzer.analyze_cushing_stocks(
            current_level=25.0,
            tank_capacity=76.0
        )
        
        assert 'utilization_pct' in result
        assert result['utilization_pct'] == pytest.approx(32.9, rel=0.1)
        assert 'status' in result
    
    def test_analyze_opec_compliance(self):
        """Test OPEC compliance analysis."""
        analyzer = FundamentalAnalyzer()
        
        opec_data = pd.DataFrame({
            'country': ['Saudi Arabia', 'Russia', 'Iraq'],
            'quota_mbpd': [9.0, 9.5, 4.0],
            'actual_mbpd': [8.98, 9.45, 4.25],
            'compliance_pct': [102, 95, 62],
        })
        
        result = analyzer.analyze_opec_compliance(opec_data)
        
        assert 'overall_compliance_pct' in result
        assert 'deviation_mbpd' in result
        assert 'market_impact' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
