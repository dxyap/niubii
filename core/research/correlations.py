"""
Correlation Analyzer
====================
Cross-asset correlation analysis for oil markets.

Features:
- Rolling correlation calculations
- Cross-asset correlation matrices
- Correlation regime detection
- Correlation breakdown alerts
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from core.data.loader import DataLoader

logger = logging.getLogger(__name__)


class CorrelationMethod(Enum):
    """Correlation calculation methods."""
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"


class CorrelationRegime(Enum):
    """Correlation regime states."""
    HIGH_POSITIVE = "high_positive"      # > 0.7
    MODERATE_POSITIVE = "moderate_positive"  # 0.3 to 0.7
    LOW = "low"                          # -0.3 to 0.3
    MODERATE_NEGATIVE = "moderate_negative"  # -0.7 to -0.3
    HIGH_NEGATIVE = "high_negative"      # < -0.7


@dataclass
class CrossAssetCorrelation:
    """Correlation between two assets."""
    asset1: str
    asset2: str
    correlation: float
    method: CorrelationMethod
    window_days: int
    observations: int
    start_date: datetime
    end_date: datetime
    regime: CorrelationRegime
    pvalue: float | None = None

    def to_dict(self) -> dict:
        return {
            "asset1": self.asset1,
            "asset2": self.asset2,
            "correlation": round(self.correlation, 4),
            "method": self.method.value,
            "window_days": self.window_days,
            "observations": self.observations,
            "regime": self.regime.value,
            "pvalue": round(self.pvalue, 4) if self.pvalue else None,
        }


@dataclass
class RollingCorrelation:
    """Rolling correlation time series."""
    asset1: str
    asset2: str
    correlations: list[float]
    dates: list[datetime]
    window_days: int
    method: CorrelationMethod
    current_regime: CorrelationRegime
    regime_changes: list[dict]  # List of regime change events

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "date": self.dates,
            "correlation": self.correlations,
        }).set_index("date")


class CorrelationAnalyzer:
    """
    Cross-asset correlation analyzer.

    Analyzes correlations between oil and other asset classes:
    - Equity indices (S&P 500, energy stocks)
    - Currencies (USD, EUR, emerging markets)
    - Interest rates (treasuries, TIPS)
    - Other commodities (natural gas, gold)
    """

    # Default asset pairs to analyze
    DEFAULT_PAIRS = [
        ("WTI", "Brent"),
        ("WTI", "SPX"),       # S&P 500
        ("WTI", "DXY"),       # US Dollar Index
        ("WTI", "XLE"),       # Energy ETF
        ("WTI", "Gold"),
        ("WTI", "NatGas"),
        ("WTI", "UST10Y"),    # 10-Year Treasury
        ("Brent", "EURUSD"),
    ]

    # Asset name to Bloomberg ticker mapping
    ASSET_TICKERS = {
        # Oil & Energy
        "WTI": "CL1 Comdty",
        "Brent": "CO1 Comdty",
        "NatGas": "NG1 Comdty",
        "Natural_Gas": "NG1 Comdty",
        # Indices
        "SPX": "SPX Index",
        "SP500": "SPX Index",
        "VIX": "VIX Index",
        "DXY": "DXY Index",
        "Dollar": "DXY Index",
        # Commodities
        "Gold": "GC1 Comdty",
        # Currencies
        "EURUSD": "EURUSD Curncy",
        # Fixed Income
        "UST10Y": "USGG10YR Index",
        # ETFs
        "XLE": "XLE US Equity",
    }

    # Asset name aliases (normalize to canonical names)
    ASSET_ALIASES = {
        "Dollar": "DXY",
        "SP500": "SPX",
        "Natural_Gas": "NatGas",
    }

    def __init__(
        self,
        method: CorrelationMethod = CorrelationMethod.PEARSON,
        default_window: int = 60,
        data_loader: DataLoader | None = None,
    ):
        self.method = method
        self.default_window = default_window
        self._data_loader = data_loader

        # Cache for correlation matrices
        self._cache: dict[str, Any] = {}
        # Cache for price data (keyed by asset name)
        self._price_data_cache: dict[str, pd.Series] = {}

    def set_data_loader(self, data_loader: DataLoader) -> None:
        """Set the data loader for fetching real market data."""
        self._data_loader = data_loader
        # Clear cache when data loader changes
        self._price_data_cache.clear()

    def _get_bloomberg_ticker(self, asset: str) -> str | None:
        """Get Bloomberg ticker for an asset name."""
        # First normalize the asset name
        normalized = self.ASSET_ALIASES.get(asset, asset)
        # Then get the ticker
        return self.ASSET_TICKERS.get(normalized) or self.ASSET_TICKERS.get(asset)

    def _get_price_data(self, assets: list[str], days: int = 365) -> dict[str, pd.Series]:
        """
        Get price data for given assets from Bloomberg.

        Args:
            assets: List of asset names
            days: Number of days of data

        Returns:
            Dictionary mapping asset names to price series
        """
        if self._data_loader is None:
            logger.warning("No data loader configured - cannot fetch price data")
            return {}

        result = {}
        start_date = datetime.now() - timedelta(days=days)
        end_date = datetime.now()

        for asset in assets:
            # Check cache first
            cache_key = f"{asset}_{days}"
            if cache_key in self._price_data_cache:
                result[asset] = self._price_data_cache[cache_key]
                continue

            # Get Bloomberg ticker
            ticker = self._get_bloomberg_ticker(asset)
            if not ticker:
                logger.warning(f"No Bloomberg ticker mapping for asset: {asset}")
                continue

            try:
                # Fetch historical data from Bloomberg
                df = self._data_loader.get_historical(
                    ticker,
                    start_date=start_date,
                    end_date=end_date,
                    frequency="daily"
                )

                if df is not None and not df.empty:
                    # Extract close price (PX_LAST)
                    if "PX_LAST" in df.columns:
                        price_series = df["PX_LAST"].astype(float)
                    elif len(df.columns) == 1:
                        price_series = df.iloc[:, 0].astype(float)
                    else:
                        # Try common column names
                        for col in ["close", "Close", "CLOSE", "PX_LAST"]:
                            if col in df.columns:
                                price_series = df[col].astype(float)
                                break
                        else:
                            price_series = df.iloc[:, 0].astype(float)

                    # Store in cache and result
                    self._price_data_cache[cache_key] = price_series
                    result[asset] = price_series
                    logger.debug(f"Fetched {len(price_series)} days of data for {asset} ({ticker})")
                else:
                    logger.warning(f"No data returned for {asset} ({ticker})")

            except Exception as e:
                logger.warning(f"Failed to fetch data for {asset} ({ticker}): {e}")

        return result

    def calculate_correlation(
        self,
        returns1: np.ndarray,
        returns2: np.ndarray,
        method: CorrelationMethod | None = None,
    ) -> tuple[float, float]:
        """
        Calculate correlation between two return series.

        Args:
            returns1: First return series
            returns2: Second return series
            method: Correlation method

        Returns:
            Tuple of (correlation, p-value)
        """
        method = method or self.method

        # Remove NaN values
        mask = ~(np.isnan(returns1) | np.isnan(returns2))
        r1 = returns1[mask]
        r2 = returns2[mask]

        if len(r1) < 10:
            return 0.0, 1.0

        if method == CorrelationMethod.PEARSON:
            from scipy import stats
            corr, pvalue = stats.pearsonr(r1, r2)
        elif method == CorrelationMethod.SPEARMAN:
            from scipy import stats
            corr, pvalue = stats.spearmanr(r1, r2)
        elif method == CorrelationMethod.KENDALL:
            from scipy import stats
            corr, pvalue = stats.kendalltau(r1, r2)
        else:
            # Fallback to numpy
            corr = np.corrcoef(r1, r2)[0, 1]
            pvalue = None

        return float(corr), float(pvalue) if pvalue is not None else None

    def calculate_pair_correlation(
        self,
        prices1: pd.Series,
        prices2: pd.Series,
        asset1: str,
        asset2: str,
        window_days: int | None = None,
    ) -> CrossAssetCorrelation:
        """
        Calculate correlation for an asset pair.

        Args:
            prices1: Price series for first asset
            prices2: Price series for second asset
            asset1: Name of first asset
            asset2: Name of second asset
            window_days: Lookback window in days

        Returns:
            CrossAssetCorrelation result
        """
        window = window_days or self.default_window

        # Align series
        aligned = pd.DataFrame({
            asset1: prices1,
            asset2: prices2,
        }).dropna()

        # Use last N days
        if len(aligned) > window:
            aligned = aligned.tail(window)

        # Calculate returns
        returns1 = aligned[asset1].pct_change().dropna().values
        returns2 = aligned[asset2].pct_change().dropna().values

        # Calculate correlation
        corr, pvalue = self.calculate_correlation(returns1, returns2)

        # Determine regime
        regime = self._get_regime(corr)

        return CrossAssetCorrelation(
            asset1=asset1,
            asset2=asset2,
            correlation=corr,
            method=self.method,
            window_days=window,
            observations=len(returns1),
            start_date=aligned.index[0],
            end_date=aligned.index[-1],
            regime=regime,
            pvalue=pvalue,
        )

    def calculate_rolling_correlation(
        self,
        prices1_or_asset1: pd.Series | str,
        prices2_or_asset2: pd.Series | str,
        asset1: str | None = None,
        asset2: str | None = None,
        window_days: int = 30,
        window: int | None = None,
        days: int = 365,
    ) -> RollingCorrelation | list:
        """
        Calculate rolling correlation.

        Args:
            prices1_or_asset1: Price series for first asset, or asset name (str)
            prices2_or_asset2: Price series for second asset, or asset name (str)
            asset1: Name of first asset (optional if prices1_or_asset1 is str)
            asset2: Name of second asset (optional if prices2_or_asset2 is str)
            window_days: Rolling window in days
            window: Alias for window_days (for compatibility)
            days: Number of days of data to use (only when using asset names)

        Returns:
            RollingCorrelation with time series, or list of correlation data points
        """
        # Handle window alias
        if window is not None:
            window_days = window

        # Check if called with asset names (strings) instead of price series
        if isinstance(prices1_or_asset1, str) and isinstance(prices2_or_asset2, str):
            asset1_name = prices1_or_asset1
            asset2_name = prices2_or_asset2
            # Fetch price data
            price_data = self._get_price_data([asset1_name, asset2_name], days=days)
            if asset1_name not in price_data or asset2_name not in price_data:
                return []
            prices1 = price_data[asset1_name]
            prices2 = price_data[asset2_name]
        else:
            prices1 = prices1_or_asset1
            prices2 = prices2_or_asset2
            asset1_name = asset1 if asset1 else "Asset1"
            asset2_name = asset2 if asset2 else "Asset2"

        # Align series
        aligned = pd.DataFrame({
            asset1_name: prices1,
            asset2_name: prices2,
        }).dropna()

        if aligned.empty or len(aligned) < window_days:
            return []

        # Calculate returns
        returns = aligned.pct_change().dropna()

        if returns.empty:
            return []

        # Rolling correlation
        rolling_corr = returns[asset1_name].rolling(window=window_days).corr(returns[asset2_name])
        rolling_corr = rolling_corr.dropna()

        if rolling_corr.empty:
            return []

        # Detect regime changes
        regime_changes = []
        prev_regime = None

        correlations = rolling_corr.values.tolist()
        dates = rolling_corr.index.tolist()

        for _i, (date, corr) in enumerate(zip(dates, correlations)):
            regime = self._get_regime(corr)
            if prev_regime is not None and regime != prev_regime:
                regime_changes.append({
                    "date": date,
                    "from_regime": prev_regime.value,
                    "to_regime": regime.value,
                    "correlation": corr,
                })
            prev_regime = regime

        current_regime = self._get_regime(correlations[-1]) if correlations else CorrelationRegime.LOW

        return RollingCorrelation(
            asset1=asset1_name,
            asset2=asset2_name,
            correlations=correlations,
            dates=dates,
            window_days=window_days,
            method=self.method,
            current_regime=current_regime,
            regime_changes=regime_changes,
        )

    def calculate_correlation_matrix(
        self,
        price_data: dict[str, pd.Series] | list[str],
        window_days: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix for multiple assets.

        Args:
            price_data: Dictionary of asset name to price series,
                        or list of asset names (will fetch data internally)
            window_days: Lookback window

        Returns:
            Correlation matrix as DataFrame
        """
        window = window_days or self.default_window

        # Handle list of asset names - fetch price data
        if isinstance(price_data, list):
            price_data = self._get_price_data(price_data)
            if not price_data:
                return pd.DataFrame()

        # Align all series
        df = pd.DataFrame(price_data).dropna()

        if df.empty:
            return pd.DataFrame()

        if len(df) > window:
            df = df.tail(window)

        # Calculate returns
        returns = df.pct_change().dropna()

        if returns.empty:
            return pd.DataFrame()

        # Correlation matrix
        if self.method == CorrelationMethod.PEARSON:
            corr_matrix = returns.corr(method="pearson")
        elif self.method == CorrelationMethod.SPEARMAN:
            corr_matrix = returns.corr(method="spearman")
        else:
            corr_matrix = returns.corr(method="kendall")

        return corr_matrix

    def detect_correlation_breakdown(
        self,
        rolling_corr: RollingCorrelation,
        threshold_std: float = 2.0,
    ) -> list[dict]:
        """
        Detect correlation breakdown events.

        Args:
            rolling_corr: Rolling correlation result
            threshold_std: Standard deviation threshold for breakdown

        Returns:
            List of breakdown events
        """
        correlations = np.array(rolling_corr.correlations)
        dates = rolling_corr.dates

        if len(correlations) < 30:
            return []

        # Calculate rolling mean and std
        mean = np.mean(correlations)
        std = np.std(correlations)

        breakdowns = []

        for i in range(len(correlations)):
            deviation = abs(correlations[i] - mean) / std

            if deviation > threshold_std:
                breakdowns.append({
                    "date": dates[i],
                    "correlation": correlations[i],
                    "mean": mean,
                    "deviation_std": deviation,
                    "direction": "spike" if correlations[i] > mean else "breakdown",
                })

        return breakdowns

    def get_correlation_summary(
        self,
        price_data: dict[str, pd.Series],
        base_asset: str = "WTI",
    ) -> dict[str, Any]:
        """
        Get summary of correlations with a base asset.

        Args:
            price_data: Dictionary of asset prices
            base_asset: Base asset to correlate against

        Returns:
            Summary of correlations
        """
        if base_asset not in price_data:
            return {"error": f"Base asset {base_asset} not in data"}

        correlations = []

        for asset, prices in price_data.items():
            if asset == base_asset:
                continue

            try:
                result = self.calculate_pair_correlation(
                    price_data[base_asset],
                    prices,
                    base_asset,
                    asset,
                )
                correlations.append(result.to_dict())
            except Exception as e:
                logger.warning(f"Failed to calculate correlation for {asset}: {e}")

        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        # Find highest positive and negative
        positive = [c for c in correlations if c["correlation"] > 0]
        negative = [c for c in correlations if c["correlation"] < 0]

        return {
            "base_asset": base_asset,
            "all_correlations": correlations,
            "highest_positive": positive[0] if positive else None,
            "highest_negative": negative[0] if negative else None,
            "count": len(correlations),
            "generated_at": datetime.now().isoformat(),
        }

    def detect_regime(
        self,
        asset1: str,
        asset2: str,
        window_days: int = 60,
    ) -> dict[str, Any]:
        """
        Detect the current correlation regime between two assets.

        Args:
            asset1: Name of first asset
            asset2: Name of second asset
            window_days: Lookback window for correlation calculation

        Returns:
            Dictionary with regime information:
            - regime: Current regime name
            - current_correlation: Current correlation value
            - regime_strength: Confidence/strength of regime (0-100%)
            - days_in_regime: Estimated days in current regime
        """
        # Get rolling correlation data
        rolling_result = self.calculate_rolling_correlation(
            asset1, asset2,
            window_days=window_days,
            days=365
        )

        if not rolling_result or (isinstance(rolling_result, list) and len(rolling_result) == 0):
            return {
                "regime": "Unknown",
                "current_correlation": 0.0,
                "regime_strength": 50.0,
                "days_in_regime": 0,
            }

        # Handle both RollingCorrelation object and list
        if isinstance(rolling_result, RollingCorrelation):
            correlations = rolling_result.correlations
            current_regime = rolling_result.current_regime
            regime_changes = rolling_result.regime_changes
        else:
            return {
                "regime": "Unknown",
                "current_correlation": 0.0,
                "regime_strength": 50.0,
                "days_in_regime": 0,
            }

        if not correlations:
            return {
                "regime": "Unknown",
                "current_correlation": 0.0,
                "regime_strength": 50.0,
                "days_in_regime": 0,
            }

        current_corr = correlations[-1]

        # Calculate regime strength based on how extreme the correlation is
        # Higher absolute correlation = higher strength
        regime_strength = min(100, abs(current_corr) * 100 + 30)

        # Calculate days in current regime
        days_in_regime = 0
        if regime_changes:
            last_change = regime_changes[-1]
            # Estimate days since last regime change
            days_in_regime = len(correlations) - next(
                (i for i, (d, _) in enumerate(zip(rolling_result.dates, correlations))
                 if d >= last_change["date"]),
                len(correlations)
            )
        else:
            # No regime changes, assume entire period is current regime
            days_in_regime = len(correlations)

        # Format regime name for display
        regime_name = current_regime.value.replace("_", " ").title()

        return {
            "regime": regime_name,
            "current_correlation": current_corr,
            "regime_strength": regime_strength,
            "days_in_regime": max(0, days_in_regime),
        }

    def _get_regime(self, correlation: float) -> CorrelationRegime:
        """Determine correlation regime."""
        if correlation > 0.7:
            return CorrelationRegime.HIGH_POSITIVE
        elif correlation > 0.3:
            return CorrelationRegime.MODERATE_POSITIVE
        elif correlation < -0.7:
            return CorrelationRegime.HIGH_NEGATIVE
        elif correlation < -0.3:
            return CorrelationRegime.MODERATE_NEGATIVE
        else:
            return CorrelationRegime.LOW


def create_mock_price_data(days: int = 252) -> dict[str, pd.Series]:
    """Create mock price data for testing."""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=days, freq="D")

    # Base WTI prices with trend and volatility
    wti_returns = np.random.normal(0.0001, 0.02, days)
    wti_prices = 70 * np.cumprod(1 + wti_returns)

    # Brent highly correlated with WTI
    brent_noise = np.random.normal(0, 0.005, days)
    brent_returns = wti_returns * 0.95 + brent_noise
    brent_prices = 73 * np.cumprod(1 + brent_returns)

    # SPX moderately correlated
    spx_noise = np.random.normal(0, 0.01, days)
    spx_returns = wti_returns * 0.3 + spx_noise
    spx_prices = 4500 * np.cumprod(1 + spx_returns)

    # DXY negatively correlated
    dxy_noise = np.random.normal(0, 0.005, days)
    dxy_returns = -wti_returns * 0.4 + dxy_noise
    dxy_prices = 104 * np.cumprod(1 + dxy_returns)

    # Gold uncorrelated
    gold_returns = np.random.normal(0.0001, 0.01, days)
    gold_prices = 1950 * np.cumprod(1 + gold_returns)

    # VIX (volatility index) - negatively correlated with market
    vix_noise = np.random.normal(0, 0.03, days)
    vix_returns = -spx_returns * 0.5 + vix_noise
    vix_prices = 18 * np.cumprod(1 + vix_returns)
    vix_prices = np.clip(vix_prices, 10, 80)  # VIX typically ranges 10-80

    # Natural Gas - moderately correlated with oil
    natgas_noise = np.random.normal(0, 0.025, days)
    natgas_returns = wti_returns * 0.4 + natgas_noise
    natgas_prices = 2.5 * np.cumprod(1 + natgas_returns)

    return {
        "WTI": pd.Series(wti_prices, index=dates),
        "Brent": pd.Series(brent_prices, index=dates),
        "SPX": pd.Series(spx_prices, index=dates),
        "DXY": pd.Series(dxy_prices, index=dates),
        "Gold": pd.Series(gold_prices, index=dates),
        "VIX": pd.Series(vix_prices, index=dates),
        "NatGas": pd.Series(natgas_prices, index=dates),
    }
