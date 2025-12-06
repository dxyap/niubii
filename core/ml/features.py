"""
Feature Engineering Pipeline
============================
Create ML features from price, volume, and fundamental data.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""

    # Price-based features
    price_lags: list[int] = field(default_factory=lambda: [1, 2, 3, 5, 10, 21])
    return_windows: list[int] = field(default_factory=lambda: [1, 5, 10, 21, 63])

    # Moving averages
    ma_windows: list[int] = field(default_factory=lambda: [5, 10, 20, 50, 100, 200])

    # Volatility features
    volatility_windows: list[int] = field(default_factory=lambda: [5, 10, 21, 63])

    # Momentum features
    momentum_windows: list[int] = field(default_factory=lambda: [5, 10, 21])
    rsi_window: int = 14

    # Volume features
    volume_ma_windows: list[int] = field(default_factory=lambda: [5, 10, 20])

    # Bollinger Bands
    bb_window: int = 20
    bb_std: float = 2.0

    # Target configuration
    target_horizon: int = 5  # Days ahead to predict
    target_type: str = "direction"  # "direction", "return", "volatility"

    # Feature selection
    min_periods: int = 200  # Minimum data points required


class FeatureEngineer:
    """
    Feature engineering pipeline for ML models.

    Creates technical and derived features from OHLCV data.
    """

    def __init__(self, config: FeatureConfig | None = None):
        """Initialize feature engineer with configuration."""
        self.config = config or FeatureConfig()
        self._feature_names: list[str] = []

    @property
    def feature_names(self) -> list[str]:
        """Get list of feature names."""
        return self._feature_names.copy()

    def create_features(
        self,
        df: pd.DataFrame,
        include_target: bool = True,
    ) -> pd.DataFrame:
        """
        Create all features from OHLCV data.

        Args:
            df: DataFrame with OHLCV data (columns: PX_OPEN, PX_HIGH, PX_LOW, PX_LAST, PX_VOLUME)
            include_target: Whether to include target variable

        Returns:
            DataFrame with features and optionally target
        """
        if len(df) < self.config.min_periods:
            logger.warning(f"Insufficient data: {len(df)} < {self.config.min_periods}")
            return pd.DataFrame()

        # Standardize column names
        df = self._standardize_columns(df)

        features = pd.DataFrame(index=df.index)

        # Price features
        features = self._add_price_features(df, features)

        # Return features
        features = self._add_return_features(df, features)

        # Moving average features
        features = self._add_ma_features(df, features)

        # Volatility features
        features = self._add_volatility_features(df, features)

        # Momentum features
        features = self._add_momentum_features(df, features)

        # Volume features
        if 'volume' in df.columns:
            features = self._add_volume_features(df, features)

        # Open interest features
        if 'open_interest' in df.columns:
            features = self._add_oi_features(df, features)

        # Bollinger Band features
        features = self._add_bollinger_features(df, features)

        # Calendar features
        features = self._add_calendar_features(df, features)

        # Target variable
        if include_target:
            features = self._add_target(df, features)

        # Store feature names (excluding target)
        self._feature_names = [c for c in features.columns if not c.startswith('target_')]

        # Drop rows with NaN
        features = features.dropna()

        logger.info(f"Created {len(self._feature_names)} features from {len(df)} rows")

        return features

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names."""
        df = df.copy()

        column_map = {
            'PX_OPEN': 'open',
            'PX_HIGH': 'high',
            'PX_LOW': 'low',
            'PX_LAST': 'close',
            'PX_CLOSE': 'close',
            'PX_VOLUME': 'volume',
            'OPEN_INT': 'open_interest',
        }

        df.columns = [column_map.get(c, c.lower()) for c in df.columns]
        return df

    def _add_price_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        close = df['close']

        # Lagged prices (as returns from current)
        for lag in self.config.price_lags:
            features[f'price_lag_{lag}'] = close.pct_change(lag)

        # Price relative to high/low
        if 'high' in df.columns and 'low' in df.columns:
            features['price_range'] = (df['high'] - df['low']) / close
            features['price_position'] = (close - df['low']) / (df['high'] - df['low'] + 1e-10)

        # Gap from previous close
        if 'open' in df.columns:
            features['overnight_gap'] = (df['open'] - close.shift(1)) / close.shift(1)

        return features

    def _add_return_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add return-based features."""
        close = df['close']

        # Simple returns over various windows
        for window in self.config.return_windows:
            ret = close.pct_change(window)
            features[f'return_{window}d'] = ret

            # Z-score of returns
            features[f'return_{window}d_zscore'] = (
                ret - ret.rolling(63).mean()
            ) / (ret.rolling(63).std() + 1e-10)

        # Log returns
        log_returns = np.log(close / close.shift(1))
        features['log_return_1d'] = log_returns

        # Cumulative returns
        features['cum_return_5d'] = close.pct_change(5)
        features['cum_return_21d'] = close.pct_change(21)

        return features

    def _add_ma_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add moving average features."""
        close = df['close']

        mas = {}
        for window in self.config.ma_windows:
            ma = close.rolling(window).mean()
            mas[window] = ma

            # Price relative to MA
            features[f'price_ma_{window}_ratio'] = close / ma

            # MA slope
            features[f'ma_{window}_slope'] = ma.pct_change(5)

        # MA crossover signals
        if 5 in mas and 20 in mas:
            features['ma_5_20_cross'] = (mas[5] > mas[20]).astype(float)
            features['ma_5_20_distance'] = (mas[5] - mas[20]) / close

        if 10 in mas and 50 in mas:
            features['ma_10_50_cross'] = (mas[10] > mas[50]).astype(float)
            features['ma_10_50_distance'] = (mas[10] - mas[50]) / close

        if 50 in mas and 200 in mas:
            features['ma_50_200_cross'] = (mas[50] > mas[200]).astype(float)
            features['ma_50_200_distance'] = (mas[50] - mas[200]) / close

        return features

    def _add_volatility_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features."""
        close = df['close']
        log_returns = np.log(close / close.shift(1))

        for window in self.config.volatility_windows:
            # Realized volatility (annualized)
            vol = log_returns.rolling(window).std() * np.sqrt(252)
            features[f'volatility_{window}d'] = vol

            # Volatility of volatility
            features[f'vol_of_vol_{window}d'] = vol.rolling(window).std()

        # Volatility ratio (short-term vs long-term)
        if 5 in self.config.volatility_windows and 21 in self.config.volatility_windows:
            vol_5 = log_returns.rolling(5).std()
            vol_21 = log_returns.rolling(21).std()
            features['volatility_ratio_5_21'] = vol_5 / (vol_21 + 1e-10)

        # Parkinson volatility (uses high-low range)
        if 'high' in df.columns and 'low' in df.columns:
            hl_ratio = np.log(df['high'] / df['low'])
            parkinson = hl_ratio ** 2 / (4 * np.log(2))
            features['parkinson_vol_21d'] = np.sqrt(parkinson.rolling(21).mean() * 252)

        # ATR (Average True Range)
        if 'high' in df.columns and 'low' in df.columns:
            tr = pd.concat([
                df['high'] - df['low'],
                abs(df['high'] - close.shift(1)),
                abs(df['low'] - close.shift(1))
            ], axis=1).max(axis=1)
            features['atr_14'] = tr.rolling(14).mean() / close

        return features

    def _add_momentum_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add momentum features."""
        close = df['close']

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(self.config.rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.config.rsi_window).mean()
        rs = gain / (loss + 1e-10)
        features['rsi_14'] = 100 - (100 / (1 + rs))

        # Normalized RSI (z-score)
        features['rsi_14_zscore'] = (
            features['rsi_14'] - features['rsi_14'].rolling(63).mean()
        ) / (features['rsi_14'].rolling(63).std() + 1e-10)

        # Rate of Change (ROC)
        for window in self.config.momentum_windows:
            features[f'roc_{window}d'] = close.pct_change(window)

        # MACD
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9, adjust=False).mean()
        features['macd'] = macd / close
        features['macd_signal'] = signal / close
        features['macd_histogram'] = (macd - signal) / close

        # Stochastic oscillator
        if 'high' in df.columns and 'low' in df.columns:
            low_14 = df['low'].rolling(14).min()
            high_14 = df['high'].rolling(14).max()
            features['stoch_k'] = 100 * (close - low_14) / (high_14 - low_14 + 1e-10)
            features['stoch_d'] = features['stoch_k'].rolling(3).mean()

        # Williams %R
        if 'high' in df.columns and 'low' in df.columns:
            features['williams_r'] = -100 * (high_14 - close) / (high_14 - low_14 + 1e-10)

        return features

    def _add_volume_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        volume = df['volume']
        close = df['close']

        # Volume MAs and ratios
        for window in self.config.volume_ma_windows:
            vol_ma = volume.rolling(window).mean()
            features[f'volume_ma_{window}_ratio'] = volume / (vol_ma + 1e-10)

        # Volume trend
        features['volume_trend'] = volume.rolling(10).mean() / (volume.rolling(50).mean() + 1e-10)

        # On-Balance Volume (simplified)
        obv = (np.sign(close.diff()) * volume).cumsum()
        features['obv_normalized'] = obv / obv.rolling(50).std()

        # Volume-price trend
        features['volume_price_trend'] = (
            (close.pct_change() * volume).rolling(10).sum() /
            (volume.rolling(10).sum() + 1e-10)
        )

        # High volume days
        vol_std = volume.rolling(50).std()
        vol_mean = volume.rolling(50).mean()
        features['volume_zscore'] = (volume - vol_mean) / (vol_std + 1e-10)

        return features

    def _add_oi_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add open interest features."""
        oi = df['open_interest']
        close = df['close']

        # OI change
        features['oi_change_1d'] = oi.pct_change()
        features['oi_change_5d'] = oi.pct_change(5)

        # OI relative to MA
        oi_ma = oi.rolling(20).mean()
        features['oi_ma_ratio'] = oi / (oi_ma + 1e-10)

        # OI-price divergence
        price_change = close.pct_change(5)
        oi_change = oi.pct_change(5)
        features['oi_price_divergence'] = oi_change - price_change

        return features

    def _add_bollinger_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add Bollinger Band features."""
        close = df['close']

        ma = close.rolling(self.config.bb_window).mean()
        std = close.rolling(self.config.bb_window).std()

        upper = ma + self.config.bb_std * std
        lower = ma - self.config.bb_std * std

        # Bollinger Band position
        features['bb_position'] = (close - lower) / (upper - lower + 1e-10)

        # Bollinger Band width (volatility indicator)
        features['bb_width'] = (upper - lower) / ma

        # Distance from bands
        features['bb_upper_distance'] = (upper - close) / close
        features['bb_lower_distance'] = (close - lower) / close

        return features

    def _add_calendar_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add calendar/time-based features."""
        if not isinstance(df.index, pd.DatetimeIndex):
            return features

        # Day of week (Monday=0, Friday=4)
        features['day_of_week'] = df.index.dayofweek

        # Month
        features['month'] = df.index.month

        # Is month end
        features['is_month_end'] = df.index.is_month_end.astype(float)

        # Is quarter end
        features['is_quarter_end'] = df.index.is_quarter_end.astype(float)

        # Days to month end
        features['days_to_month_end'] = (
            pd.to_datetime(df.index.to_period('M').to_timestamp('M')) - df.index
        ).days

        return features

    def _add_target(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add target variable for supervised learning."""
        close = df['close']
        horizon = self.config.target_horizon

        if self.config.target_type == "direction":
            # Binary classification: 1 if price goes up, 0 if down
            future_return = close.shift(-horizon) / close - 1
            features['target_direction'] = (future_return > 0).astype(float)
            features['target_return'] = future_return  # Also include actual return

        elif self.config.target_type == "return":
            # Regression: predict actual return
            features['target_return'] = close.shift(-horizon) / close - 1

        elif self.config.target_type == "volatility":
            # Predict future volatility
            log_returns = np.log(close / close.shift(1))
            future_vol = log_returns.shift(-horizon).rolling(horizon).std() * np.sqrt(252)
            features['target_volatility'] = future_vol

        return features

    def get_feature_importance_template(self) -> dict[str, str]:
        """Get template mapping feature names to categories."""
        categories = {}

        for name in self._feature_names:
            if 'price' in name or 'gap' in name:
                categories[name] = 'Price'
            elif 'return' in name:
                categories[name] = 'Returns'
            elif 'ma_' in name:
                categories[name] = 'Moving Averages'
            elif 'volatility' in name or 'vol' in name or 'atr' in name or 'parkinson' in name:
                categories[name] = 'Volatility'
            elif 'rsi' in name or 'roc' in name or 'macd' in name or 'stoch' in name or 'williams' in name:
                categories[name] = 'Momentum'
            elif 'volume' in name or 'obv' in name:
                categories[name] = 'Volume'
            elif 'oi_' in name:
                categories[name] = 'Open Interest'
            elif 'bb_' in name:
                categories[name] = 'Bollinger Bands'
            elif 'day' in name or 'month' in name or 'quarter' in name:
                categories[name] = 'Calendar'
            else:
                categories[name] = 'Other'

        return categories
