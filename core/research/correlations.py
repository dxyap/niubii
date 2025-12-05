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

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

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
    pvalue: Optional[float] = None
    
    def to_dict(self) -> Dict:
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
    correlations: List[float]
    dates: List[datetime]
    window_days: int
    method: CorrelationMethod
    current_regime: CorrelationRegime
    regime_changes: List[Dict]  # List of regime change events
    
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
    
    def __init__(
        self,
        method: CorrelationMethod = CorrelationMethod.PEARSON,
        default_window: int = 60,
    ):
        self.method = method
        self.default_window = default_window
        
        # Cache for correlation matrices
        self._cache: Dict[str, Any] = {}
    
    def calculate_correlation(
        self,
        returns1: np.ndarray,
        returns2: np.ndarray,
        method: Optional[CorrelationMethod] = None,
    ) -> Tuple[float, float]:
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
        window_days: Optional[int] = None,
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
        prices1: pd.Series,
        prices2: pd.Series,
        asset1: str,
        asset2: str,
        window_days: int = 30,
    ) -> RollingCorrelation:
        """
        Calculate rolling correlation.
        
        Args:
            prices1: Price series for first asset
            prices2: Price series for second asset
            asset1: Name of first asset
            asset2: Name of second asset
            window_days: Rolling window in days
            
        Returns:
            RollingCorrelation with time series
        """
        # Align series
        aligned = pd.DataFrame({
            asset1: prices1,
            asset2: prices2,
        }).dropna()
        
        # Calculate returns
        returns = aligned.pct_change().dropna()
        
        # Rolling correlation
        rolling_corr = returns[asset1].rolling(window=window_days).corr(returns[asset2])
        rolling_corr = rolling_corr.dropna()
        
        # Detect regime changes
        regime_changes = []
        prev_regime = None
        
        correlations = rolling_corr.values.tolist()
        dates = rolling_corr.index.tolist()
        
        for i, (date, corr) in enumerate(zip(dates, correlations)):
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
            asset1=asset1,
            asset2=asset2,
            correlations=correlations,
            dates=dates,
            window_days=window_days,
            method=self.method,
            current_regime=current_regime,
            regime_changes=regime_changes,
        )
    
    def calculate_correlation_matrix(
        self,
        price_data: Dict[str, pd.Series],
        window_days: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix for multiple assets.
        
        Args:
            price_data: Dictionary of asset name to price series
            window_days: Lookback window
            
        Returns:
            Correlation matrix as DataFrame
        """
        window = window_days or self.default_window
        
        # Align all series
        df = pd.DataFrame(price_data).dropna()
        
        if len(df) > window:
            df = df.tail(window)
        
        # Calculate returns
        returns = df.pct_change().dropna()
        
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
    ) -> List[Dict]:
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
        price_data: Dict[str, pd.Series],
        base_asset: str = "WTI",
    ) -> Dict[str, Any]:
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


def create_mock_price_data(days: int = 252) -> Dict[str, pd.Series]:
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
    
    return {
        "WTI": pd.Series(wti_prices, index=dates),
        "Brent": pd.Series(brent_prices, index=dates),
        "SPX": pd.Series(spx_prices, index=dates),
        "DXY": pd.Series(dxy_prices, index=dates),
        "Gold": pd.Series(gold_prices, index=dates),
    }
