"""
Factor Model
=============
Factor decomposition and analysis for oil returns.

Features:
- Risk factor identification
- Factor exposure calculation
- Return attribution
- Factor-based risk analysis
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RiskFactor(Enum):
    """Common risk factors for oil markets."""
    MARKET = "market"              # Broad commodity/oil market
    DOLLAR = "dollar"              # USD strength
    EQUITIES = "equities"          # Equity market correlation
    RATES = "rates"                # Interest rate sensitivity
    VOLATILITY = "volatility"      # Volatility factor
    MOMENTUM = "momentum"          # Trend/momentum
    VALUE = "value"                # Mean reversion
    CARRY = "carry"                # Convenience yield/carry
    SPREAD = "spread"              # Crude spread factor
    SEASONAL = "seasonal"          # Seasonality


@dataclass
class FactorConfig:
    """Configuration for factor model."""
    # Factors to include
    factors: list[RiskFactor] = field(default_factory=lambda: [
        RiskFactor.MARKET,
        RiskFactor.MOMENTUM,
        RiskFactor.VOLATILITY,
    ])

    # Estimation settings
    lookback_days: int = 252
    min_observations: int = 60

    # Factor construction
    momentum_lookback: int = 20
    volatility_lookback: int = 20

    # Regularization
    use_regularization: bool = True
    regularization_strength: float = 0.01


@dataclass
class FactorDecomposition:
    """Result of factor decomposition."""
    timestamp: datetime
    asset: str

    # Factor exposures (betas)
    exposures: dict[str, float]

    # Factor returns contribution
    factor_returns: dict[str, float]

    # Residual (unexplained)
    residual: float

    # Model statistics
    r_squared: float
    adjusted_r_squared: float

    # Standard errors
    exposure_std_errors: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "asset": self.asset,
            "exposures": self.exposures,
            "factor_returns": self.factor_returns,
            "residual": round(self.residual, 6),
            "r_squared": round(self.r_squared, 4),
        }


class FactorModel:
    """
    Factor model for oil return decomposition.

    Decomposes returns into systematic factor exposures
    and idiosyncratic components.
    """

    def __init__(self, config: FactorConfig | None = None):
        self.config = config or FactorConfig()

        # Factor data cache
        self._factor_data: dict[RiskFactor, pd.Series] = {}

        # Model cache
        self._model_cache: dict[str, Any] = {}

    def construct_factors(
        self,
        price_data: dict[str, pd.Series],
        returns: pd.Series | None = None,
    ) -> dict[RiskFactor, pd.Series]:
        """
        Construct factor time series.

        Args:
            price_data: Dictionary of price series (WTI, Brent, SPX, DXY, etc.)
            returns: Optional return series for the asset being analyzed

        Returns:
            Dictionary of factor returns
        """
        factors = {}

        # Market factor (WTI or average oil return)
        if "WTI" in price_data:
            wti_returns = price_data["WTI"].pct_change()
            factors[RiskFactor.MARKET] = wti_returns

        # Dollar factor (inverted DXY returns)
        if "DXY" in price_data:
            dxy_returns = price_data["DXY"].pct_change()
            factors[RiskFactor.DOLLAR] = -dxy_returns  # Negative correlation

        # Equity factor
        if "SPX" in price_data:
            spx_returns = price_data["SPX"].pct_change()
            factors[RiskFactor.EQUITIES] = spx_returns

        # Momentum factor (past N-day return)
        if returns is not None:
            momentum = returns.rolling(self.config.momentum_lookback).mean()
            factors[RiskFactor.MOMENTUM] = momentum
        elif "WTI" in price_data:
            wti_returns = price_data["WTI"].pct_change()
            momentum = wti_returns.rolling(self.config.momentum_lookback).mean()
            factors[RiskFactor.MOMENTUM] = momentum

        # Volatility factor (change in volatility)
        if returns is not None:
            vol = returns.rolling(self.config.volatility_lookback).std()
            vol_change = vol.pct_change()
            factors[RiskFactor.VOLATILITY] = vol_change
        elif "WTI" in price_data:
            wti_returns = price_data["WTI"].pct_change()
            vol = wti_returns.rolling(self.config.volatility_lookback).std()
            vol_change = vol.pct_change()
            factors[RiskFactor.VOLATILITY] = vol_change

        # Spread factor (WTI-Brent spread change)
        if "WTI" in price_data and "Brent" in price_data:
            spread = price_data["WTI"] - price_data["Brent"]
            spread_return = spread.pct_change()
            factors[RiskFactor.SPREAD] = spread_return

        # Value factor (deviation from moving average)
        if "WTI" in price_data:
            wti = price_data["WTI"]
            ma = wti.rolling(50).mean()
            value = (wti / ma - 1)  # Distance from MA
            factors[RiskFactor.VALUE] = value

        self._factor_data = factors
        return factors

    def decompose_returns(
        self,
        returns: pd.Series,
        factor_data: dict[RiskFactor, pd.Series] | None = None,
        asset_name: str = "Asset",
    ) -> FactorDecomposition:
        """
        Decompose returns into factor contributions.

        Args:
            returns: Return series to decompose
            factor_data: Factor returns (uses cached if not provided)
            asset_name: Name of the asset

        Returns:
            FactorDecomposition result
        """
        factors = factor_data or self._factor_data

        if not factors:
            raise ValueError("No factor data. Call construct_factors first.")

        # Align data
        df = pd.DataFrame({"returns": returns})

        for factor, series in factors.items():
            if factor in self.config.factors:
                df[factor.value] = series

        df = df.dropna()

        if len(df) < self.config.min_observations:
            raise ValueError(f"Insufficient data: {len(df)} < {self.config.min_observations}")

        # Run regression
        y = df["returns"].values
        X = df.drop(columns=["returns"]).values
        factor_names = list(df.columns[1:])

        # Add constant
        X = np.column_stack([np.ones(len(X)), X])

        # OLS with optional regularization
        if self.config.use_regularization:
            # Ridge regression
            lambda_val = self.config.regularization_strength
            XtX = X.T @ X + lambda_val * np.eye(X.shape[1])
            Xty = X.T @ y
            betas = np.linalg.solve(XtX, Xty)
        else:
            # Standard OLS
            betas = np.linalg.lstsq(X, y, rcond=None)[0]

        # Calculate fitted values and residuals
        y_hat = X @ betas
        residuals = y - y_hat

        # R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Adjusted R-squared
        n = len(y)
        p = len(betas) - 1  # Exclude constant
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1) if n > p + 1 else r_squared

        # Standard errors
        mse = ss_res / (n - p - 1) if n > p + 1 else ss_res / n
        var_beta = mse * np.linalg.inv(X.T @ X)
        std_errors = np.sqrt(np.diag(var_beta))

        # Extract results
        exposures = {name: float(betas[i + 1]) for i, name in enumerate(factor_names)}
        exposure_std_errors = {name: float(std_errors[i + 1]) for i, name in enumerate(factor_names)}

        # Calculate factor contributions
        last_idx = len(df) - 1
        factor_returns = {}

        for i, name in enumerate(factor_names):
            factor_ret = df.iloc[last_idx][name]
            contribution = float(betas[i + 1] * factor_ret)
            factor_returns[name] = contribution

        return FactorDecomposition(
            timestamp=datetime.now(),
            asset=asset_name,
            exposures=exposures,
            factor_returns=factor_returns,
            residual=float(residuals[-1]),
            r_squared=float(r_squared),
            adjusted_r_squared=float(adj_r_squared),
            exposure_std_errors=exposure_std_errors,
        )

    def rolling_decomposition(
        self,
        returns: pd.Series,
        factor_data: dict[RiskFactor, pd.Series],
        window: int = 60,
    ) -> pd.DataFrame:
        """
        Calculate rolling factor decomposition.

        Args:
            returns: Return series
            factor_data: Factor returns
            window: Rolling window

        Returns:
            DataFrame with rolling exposures
        """
        results = []

        for i in range(window, len(returns)):
            window_returns = returns.iloc[i - window:i]
            window_factors = {
                f: s.iloc[i - window:i]
                for f, s in factor_data.items()
            }

            try:
                decomp = self.decompose_returns(
                    window_returns,
                    window_factors,
                    asset_name="rolling",
                )

                result = {
                    "date": returns.index[i],
                    "r_squared": decomp.r_squared,
                    **decomp.exposures,
                }
                results.append(result)

            except Exception as e:
                logger.debug(f"Rolling decomposition failed at {i}: {e}")

        return pd.DataFrame(results).set_index("date")

    def get_factor_statistics(
        self,
        factor_data: dict[RiskFactor, pd.Series] | None = None,
    ) -> dict[str, Any]:
        """
        Get statistics for each factor.

        Returns:
            Factor statistics
        """
        factors = factor_data or self._factor_data

        stats = {}

        for factor, series in factors.items():
            clean_series = series.dropna()

            if len(clean_series) < 10:
                continue

            stats[factor.value] = {
                "mean": float(clean_series.mean()),
                "std": float(clean_series.std()),
                "annualized_vol": float(clean_series.std() * np.sqrt(252)),
                "skew": float(clean_series.skew()),
                "kurtosis": float(clean_series.kurtosis()),
                "min": float(clean_series.min()),
                "max": float(clean_series.max()),
                "observations": len(clean_series),
            }

        return stats

    def calculate_factor_correlation_matrix(
        self,
        factor_data: dict[RiskFactor, pd.Series] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix between factors.

        Returns:
            Correlation matrix
        """
        factors = factor_data or self._factor_data

        df = pd.DataFrame({f.value: s for f, s in factors.items()}).dropna()

        return df.corr()


def create_mock_factor_data(days: int = 252) -> dict[str, pd.Series]:
    """Create mock data for factor model testing."""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=days, freq="D")

    # Create correlated factors
    market = np.random.normal(0.0001, 0.02, days)

    # WTI strongly correlated with market
    wti_noise = np.random.normal(0, 0.01, days)
    wti = market + wti_noise

    # Brent highly correlated with WTI
    brent = wti + np.random.normal(0, 0.005, days)

    # SPX moderately correlated
    spx = market * 0.3 + np.random.normal(0, 0.01, days)

    # DXY negatively correlated
    dxy = -market * 0.4 + np.random.normal(0, 0.005, days)

    # Convert to prices
    wti_prices = 70 * np.cumprod(1 + wti)
    brent_prices = 73 * np.cumprod(1 + brent)
    spx_prices = 4500 * np.cumprod(1 + spx)
    dxy_prices = 104 * np.cumprod(1 + dxy)

    return {
        "WTI": pd.Series(wti_prices, index=dates),
        "Brent": pd.Series(brent_prices, index=dates),
        "SPX": pd.Series(spx_prices, index=dates),
        "DXY": pd.Series(dxy_prices, index=dates),
    }
