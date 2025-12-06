"""
Machine Learning Module Tests
=============================
Tests for feature engineering, model training, and predictions.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

# Import ML modules
from core.ml.features import FeatureConfig, FeatureEngineer

# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n_days = 300
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='B')

    # Generate realistic price data
    base_price = 75.0
    returns = np.random.normal(0.0001, 0.02, n_days)
    prices = base_price * np.exp(np.cumsum(returns))

    # Generate OHLC from close
    data = pd.DataFrame({
        'PX_OPEN': prices * (1 + np.random.normal(0, 0.003, n_days)),
        'PX_HIGH': prices * (1 + np.abs(np.random.normal(0.005, 0.003, n_days))),
        'PX_LOW': prices * (1 - np.abs(np.random.normal(0.005, 0.003, n_days))),
        'PX_LAST': prices,
        'PX_VOLUME': np.random.randint(50000, 200000, n_days),
        'OPEN_INT': np.random.randint(100000, 300000, n_days),
    }, index=dates)

    # Ensure OHLC consistency
    data['PX_HIGH'] = data[['PX_OPEN', 'PX_HIGH', 'PX_LAST']].max(axis=1)
    data['PX_LOW'] = data[['PX_OPEN', 'PX_LOW', 'PX_LAST']].min(axis=1)

    return data


@pytest.fixture
def minimal_ohlcv_data():
    """Generate minimal OHLCV data (insufficient for features)."""
    dates = pd.date_range(end=datetime.now(), periods=50, freq='B')
    return pd.DataFrame({
        'PX_OPEN': [75.0] * 50,
        'PX_HIGH': [76.0] * 50,
        'PX_LOW': [74.0] * 50,
        'PX_LAST': [75.5] * 50,
        'PX_VOLUME': [100000] * 50,
    }, index=dates)


# =============================================================================
# FEATURE ENGINEERING TESTS
# =============================================================================

class TestFeatureConfig:
    """Tests for FeatureConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = FeatureConfig()

        assert config.target_horizon == 5
        assert config.target_type == "direction"
        assert config.min_periods == 200
        assert config.rsi_window == 14
        assert config.bb_window == 20
        assert config.bb_std == 2.0

    def test_custom_config(self):
        """Test custom configuration values."""
        config = FeatureConfig(
            target_horizon=10,
            target_type="return",
            min_periods=100,
        )

        assert config.target_horizon == 10
        assert config.target_type == "return"
        assert config.min_periods == 100


class TestFeatureEngineer:
    """Tests for FeatureEngineer class."""

    def test_initialization(self):
        """Test feature engineer initialization."""
        engineer = FeatureEngineer()

        assert engineer.config is not None
        assert len(engineer.feature_names) == 0  # No features created yet

    def test_custom_config_initialization(self):
        """Test initialization with custom config."""
        config = FeatureConfig(target_horizon=10)
        engineer = FeatureEngineer(config)

        assert engineer.config.target_horizon == 10

    def test_create_features(self, sample_ohlcv_data):
        """Test feature creation from OHLCV data."""
        engineer = FeatureEngineer()
        features = engineer.create_features(sample_ohlcv_data)

        # Should have created features
        assert not features.empty
        assert len(features) > 0

        # Should have feature names
        assert len(engineer.feature_names) > 0

        # Should have target column
        assert 'target_direction' in features.columns

    def test_feature_count(self, sample_ohlcv_data):
        """Test that expected number of features are created."""
        engineer = FeatureEngineer()
        engineer.create_features(sample_ohlcv_data)

        # Should have 50+ features
        assert len(engineer.feature_names) >= 50

    def test_no_nan_in_output(self, sample_ohlcv_data):
        """Test that output has no NaN values."""
        engineer = FeatureEngineer()
        features = engineer.create_features(sample_ohlcv_data)

        # All feature columns should have no NaN
        feature_cols = [c for c in features.columns if not c.startswith('target_')]
        for col in feature_cols:
            assert features[col].notna().all(), f"NaN found in {col}"

    def test_insufficient_data(self, minimal_ohlcv_data):
        """Test handling of insufficient data."""
        engineer = FeatureEngineer()
        features = engineer.create_features(minimal_ohlcv_data)

        # Should return empty DataFrame
        assert features.empty

    def test_price_features(self, sample_ohlcv_data):
        """Test price-based feature creation."""
        engineer = FeatureEngineer()
        features = engineer.create_features(sample_ohlcv_data)

        # Check for expected price features
        price_features = [c for c in features.columns if 'price' in c.lower()]
        assert len(price_features) > 0

        # Check for lag features
        lag_features = [c for c in features.columns if 'lag' in c]
        assert len(lag_features) > 0

    def test_return_features(self, sample_ohlcv_data):
        """Test return-based feature creation."""
        engineer = FeatureEngineer()
        features = engineer.create_features(sample_ohlcv_data)

        # Check for return features
        return_features = [c for c in features.columns if 'return' in c.lower() and not c.startswith('target')]
        assert len(return_features) > 0

    def test_ma_features(self, sample_ohlcv_data):
        """Test moving average feature creation."""
        engineer = FeatureEngineer()
        features = engineer.create_features(sample_ohlcv_data)

        # Check for MA features
        ma_features = [c for c in features.columns if 'ma_' in c]
        assert len(ma_features) > 0

        # Check for crossover features
        cross_features = [c for c in features.columns if 'cross' in c]
        assert len(cross_features) > 0

    def test_volatility_features(self, sample_ohlcv_data):
        """Test volatility feature creation."""
        engineer = FeatureEngineer()
        features = engineer.create_features(sample_ohlcv_data)

        # Check for volatility features
        vol_features = [c for c in features.columns if 'volatility' in c or 'atr' in c]
        assert len(vol_features) > 0

    def test_momentum_features(self, sample_ohlcv_data):
        """Test momentum feature creation."""
        engineer = FeatureEngineer()
        features = engineer.create_features(sample_ohlcv_data)

        # Check for RSI
        assert 'rsi_14' in features.columns

        # RSI should be between 0 and 100
        assert features['rsi_14'].min() >= 0
        assert features['rsi_14'].max() <= 100

        # Check for MACD
        macd_features = [c for c in features.columns if 'macd' in c]
        assert len(macd_features) > 0

    def test_volume_features(self, sample_ohlcv_data):
        """Test volume feature creation."""
        engineer = FeatureEngineer()
        features = engineer.create_features(sample_ohlcv_data)

        # Check for volume features
        vol_features = [c for c in features.columns if 'volume' in c.lower()]
        assert len(vol_features) > 0

    def test_bollinger_features(self, sample_ohlcv_data):
        """Test Bollinger Band feature creation."""
        engineer = FeatureEngineer()
        features = engineer.create_features(sample_ohlcv_data)

        # Check for BB features
        bb_features = [c for c in features.columns if 'bb_' in c]
        assert len(bb_features) > 0

        # BB position should be roughly between 0 and 1 (can exceed in extremes)
        assert features['bb_position'].median() > 0
        assert features['bb_position'].median() < 1

    def test_calendar_features(self, sample_ohlcv_data):
        """Test calendar feature creation."""
        engineer = FeatureEngineer()
        features = engineer.create_features(sample_ohlcv_data)

        # Check for calendar features
        assert 'day_of_week' in features.columns
        assert 'month' in features.columns

        # Day of week should be 0-4 (business days)
        assert features['day_of_week'].min() >= 0
        assert features['day_of_week'].max() <= 4

    def test_target_direction(self, sample_ohlcv_data):
        """Test direction target creation."""
        config = FeatureConfig(target_type="direction", target_horizon=5)
        engineer = FeatureEngineer(config)
        features = engineer.create_features(sample_ohlcv_data)

        assert 'target_direction' in features.columns

        # Should be binary
        unique_vals = features['target_direction'].unique()
        assert set(unique_vals).issubset({0.0, 1.0})

    def test_target_return(self, sample_ohlcv_data):
        """Test return target creation."""
        config = FeatureConfig(target_type="return", target_horizon=5)
        engineer = FeatureEngineer(config)
        features = engineer.create_features(sample_ohlcv_data)

        assert 'target_return' in features.columns

        # Returns should be reasonable
        assert features['target_return'].abs().max() < 1.0  # Less than 100%

    def test_exclude_target(self, sample_ohlcv_data):
        """Test feature creation without target."""
        engineer = FeatureEngineer()
        features = engineer.create_features(sample_ohlcv_data, include_target=False)

        # Should not have target columns
        target_cols = [c for c in features.columns if c.startswith('target_')]
        assert len(target_cols) == 0

    def test_feature_importance_template(self, sample_ohlcv_data):
        """Test feature category mapping."""
        engineer = FeatureEngineer()
        engineer.create_features(sample_ohlcv_data)

        categories = engineer.get_feature_importance_template()

        # Should have categories for all features
        assert len(categories) == len(engineer.feature_names)

        # Check expected categories exist
        category_values = set(categories.values())
        expected_categories = {'Price', 'Returns', 'Moving Averages', 'Volatility', 'Momentum'}
        assert expected_categories.issubset(category_values)

    def test_reproducibility(self, sample_ohlcv_data):
        """Test that feature creation is deterministic."""
        engineer = FeatureEngineer()

        features1 = engineer.create_features(sample_ohlcv_data)
        features2 = engineer.create_features(sample_ohlcv_data)

        pd.testing.assert_frame_equal(features1, features2)


class TestFeatureEngineerEdgeCases:
    """Edge case tests for FeatureEngineer."""

    def test_missing_volume_column(self, sample_ohlcv_data):
        """Test handling of missing volume data."""
        data = sample_ohlcv_data.drop(columns=['PX_VOLUME'])

        engineer = FeatureEngineer()
        features = engineer.create_features(data)

        # Should still work without volume
        assert not features.empty

        # Should not have volume features
        vol_features = [c for c in features.columns if 'volume' in c.lower()]
        assert len(vol_features) == 0

    def test_missing_open_interest(self, sample_ohlcv_data):
        """Test handling of missing open interest data."""
        data = sample_ohlcv_data.drop(columns=['OPEN_INT'])

        engineer = FeatureEngineer()
        features = engineer.create_features(data)

        # Should still work without OI
        assert not features.empty

        # Should not have OI features
        oi_features = [c for c in features.columns if 'oi_' in c]
        assert len(oi_features) == 0

    def test_non_datetime_index(self, sample_ohlcv_data):
        """Test handling of non-datetime index."""
        data = sample_ohlcv_data.reset_index(drop=True)

        engineer = FeatureEngineer()
        features = engineer.create_features(data)

        # Should still work
        assert not features.empty

        # Should not have calendar features
        calendar_features = [c for c in features.columns if 'day_of_week' in c or 'month' in c]
        assert len(calendar_features) == 0

    def test_constant_prices(self):
        """Test handling of constant price data."""
        dates = pd.date_range(end=datetime.now(), periods=300, freq='B')
        data = pd.DataFrame({
            'PX_OPEN': [75.0] * 300,
            'PX_HIGH': [75.0] * 300,
            'PX_LOW': [75.0] * 300,
            'PX_LAST': [75.0] * 300,
            'PX_VOLUME': [100000] * 300,
        }, index=dates)

        engineer = FeatureEngineer()
        features = engineer.create_features(data)

        # Should handle gracefully (may have fewer valid rows due to NaN/inf)
        # The key is it shouldn't crash
        assert isinstance(features, pd.DataFrame)
