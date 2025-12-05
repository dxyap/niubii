"""
Regime Detection
================
Market regime identification using statistical methods.

Features:
- Hidden Markov Model regime detection
- Volatility regime identification
- Trend regime classification
- Regime transition alerts
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime states."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRISIS = "crisis"


class VolatilityRegime(Enum):
    """Volatility regime states."""
    LOW = "low"           # < 15% annualized
    NORMAL = "normal"     # 15-30%
    HIGH = "high"         # 30-50%
    EXTREME = "extreme"   # > 50%


@dataclass
class RegimeConfig:
    """Configuration for regime detection."""
    # Number of regimes to detect
    n_regimes: int = 3
    
    # Lookback for calculations
    lookback_days: int = 252
    
    # Volatility thresholds (annualized)
    vol_low_threshold: float = 0.15
    vol_high_threshold: float = 0.30
    vol_extreme_threshold: float = 0.50
    
    # Trend thresholds
    trend_threshold: float = 0.1  # 10% annualized
    
    # HMM settings
    hmm_iterations: int = 100
    hmm_tolerance: float = 0.01
    
    # Smoothing
    smooth_window: int = 5


@dataclass
class RegimeTransition:
    """A regime transition event."""
    timestamp: datetime
    from_regime: MarketRegime
    to_regime: MarketRegime
    confidence: float
    trigger: str  # What caused the transition
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "from_regime": self.from_regime.value,
            "to_regime": self.to_regime.value,
            "confidence": round(self.confidence, 2),
            "trigger": self.trigger,
        }


class RegimeDetector:
    """
    Market regime detector using multiple methods.
    
    Methods:
    - Rule-based (volatility and trend thresholds)
    - Statistical (rolling statistics)
    - HMM (Hidden Markov Model) when available
    """
    
    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()
        
        # Regime history
        self._regime_history: List[Tuple[datetime, MarketRegime]] = []
        self._transitions: List[RegimeTransition] = []
        
        # HMM model (lazy loaded)
        self._hmm_model = None
        self._hmm_fitted = False
    
    def detect_current_regime(
        self,
        prices: pd.Series,
        returns: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Detect the current market regime.
        
        Args:
            prices: Price series
            returns: Optional pre-calculated returns
            
        Returns:
            Current regime with confidence
        """
        if returns is None:
            returns = prices.pct_change().dropna()
        
        # Use last N days
        lookback = min(self.config.lookback_days, len(returns))
        recent_returns = returns.tail(lookback)
        recent_prices = prices.tail(lookback + 1)
        
        # Calculate metrics
        volatility = self._calculate_volatility(recent_returns)
        trend = self._calculate_trend(recent_prices)
        vol_regime = self._classify_volatility_regime(volatility)
        
        # Determine market regime
        regime, confidence, trigger = self._classify_regime(volatility, trend, recent_returns)
        
        # Check for regime change
        if self._regime_history:
            last_regime = self._regime_history[-1][1]
            if regime != last_regime:
                transition = RegimeTransition(
                    timestamp=datetime.now(),
                    from_regime=last_regime,
                    to_regime=regime,
                    confidence=confidence,
                    trigger=trigger,
                )
                self._transitions.append(transition)
        
        # Update history
        self._regime_history.append((datetime.now(), regime))
        
        return {
            "regime": regime.value,
            "confidence": round(confidence, 2),
            "volatility_regime": vol_regime.value,
            "volatility_annualized": round(volatility * np.sqrt(252), 4),
            "trend": round(trend, 4),
            "trigger": trigger,
            "lookback_days": lookback,
            "timestamp": datetime.now().isoformat(),
        }
    
    def detect_regime_timeseries(
        self,
        prices: pd.Series,
        window: int = 30,
    ) -> pd.DataFrame:
        """
        Detect regimes over time.
        
        Args:
            prices: Price series
            window: Rolling window for regime detection
            
        Returns:
            DataFrame with regime classifications over time
        """
        returns = prices.pct_change().dropna()
        
        results = []
        
        for i in range(window, len(returns)):
            window_returns = returns.iloc[i - window:i]
            window_prices = prices.iloc[i - window:i + 1]
            
            vol = self._calculate_volatility(window_returns)
            trend = self._calculate_trend(window_prices)
            regime, confidence, _ = self._classify_regime(vol, trend, window_returns)
            vol_regime = self._classify_volatility_regime(vol)
            
            results.append({
                "date": returns.index[i],
                "regime": regime.value,
                "vol_regime": vol_regime.value,
                "volatility": vol * np.sqrt(252),
                "trend": trend,
                "confidence": confidence,
            })
        
        return pd.DataFrame(results).set_index("date")
    
    def fit_hmm(
        self,
        returns: pd.Series,
        n_regimes: Optional[int] = None,
    ) -> bool:
        """
        Fit Hidden Markov Model for regime detection.
        
        Args:
            returns: Return series
            n_regimes: Number of regimes
            
        Returns:
            True if fitting succeeded
        """
        try:
            from hmmlearn import hmm
        except ImportError:
            logger.warning("hmmlearn not available, using rule-based regime detection")
            return False
        
        n_regimes = n_regimes or self.config.n_regimes
        
        # Prepare data
        X = returns.values.reshape(-1, 1)
        X = X[~np.isnan(X).any(axis=1)]
        
        if len(X) < 50:
            logger.warning("Insufficient data for HMM fitting")
            return False
        
        try:
            model = hmm.GaussianHMM(
                n_components=n_regimes,
                covariance_type="diag",
                n_iter=self.config.hmm_iterations,
                tol=self.config.hmm_tolerance,
            )
            
            model.fit(X)
            self._hmm_model = model
            self._hmm_fitted = True
            
            logger.info(f"HMM fitted with {n_regimes} regimes")
            return True
            
        except Exception as e:
            logger.error(f"HMM fitting failed: {e}")
            return False
    
    def predict_with_hmm(
        self,
        returns: pd.Series,
    ) -> pd.DataFrame:
        """
        Predict regimes using fitted HMM.
        
        Args:
            returns: Return series
            
        Returns:
            DataFrame with predicted regimes
        """
        if not self._hmm_fitted:
            raise ValueError("HMM not fitted. Call fit_hmm first.")
        
        X = returns.values.reshape(-1, 1)
        
        # Predict hidden states
        hidden_states = self._hmm_model.predict(X)
        
        # Get state probabilities
        probs = self._hmm_model.predict_proba(X)
        
        results = pd.DataFrame({
            "date": returns.index,
            "state": hidden_states,
            "confidence": probs.max(axis=1),
        }).set_index("date")
        
        # Add state characteristics
        for state in range(self.config.n_regimes):
            mask = hidden_states == state
            state_returns = returns.values[mask]
            if len(state_returns) > 0:
                results.loc[mask, "state_mean"] = np.mean(state_returns)
                results.loc[mask, "state_vol"] = np.std(state_returns) * np.sqrt(252)
        
        return results
    
    def get_transition_matrix(self) -> Optional[np.ndarray]:
        """Get regime transition matrix from HMM."""
        if self._hmm_fitted:
            return self._hmm_model.transmat_
        return None
    
    def get_regime_statistics(
        self,
        prices: pd.Series,
    ) -> Dict[str, Any]:
        """
        Get statistics about regime occurrences.
        
        Args:
            prices: Price series
            
        Returns:
            Regime statistics
        """
        regimes_df = self.detect_regime_timeseries(prices)
        
        # Count by regime
        regime_counts = regimes_df["regime"].value_counts().to_dict()
        
        # Average duration
        current_regime = None
        durations = []
        current_duration = 0
        
        for regime in regimes_df["regime"]:
            if regime == current_regime:
                current_duration += 1
            else:
                if current_regime is not None:
                    durations.append((current_regime, current_duration))
                current_regime = regime
                current_duration = 1
        
        if current_regime is not None:
            durations.append((current_regime, current_duration))
        
        # Calculate average duration per regime
        avg_durations = {}
        for regime in MarketRegime:
            regime_durations = [d for r, d in durations if r == regime.value]
            if regime_durations:
                avg_durations[regime.value] = np.mean(regime_durations)
        
        # Performance by regime
        returns = prices.pct_change()
        performance = {}
        
        for regime in regimes_df["regime"].unique():
            mask = regimes_df["regime"] == regime
            regime_returns = returns.loc[mask.index[mask]]
            if len(regime_returns) > 0:
                performance[regime] = {
                    "avg_return": float(np.mean(regime_returns)),
                    "volatility": float(np.std(regime_returns) * np.sqrt(252)),
                    "sharpe": float(np.mean(regime_returns) / np.std(regime_returns) * np.sqrt(252)) if np.std(regime_returns) > 0 else 0,
                }
        
        return {
            "regime_counts": regime_counts,
            "average_durations": avg_durations,
            "performance_by_regime": performance,
            "total_observations": len(regimes_df),
            "current_regime": regimes_df["regime"].iloc[-1] if len(regimes_df) > 0 else None,
        }
    
    def get_recent_transitions(self, limit: int = 10) -> List[Dict]:
        """Get recent regime transitions."""
        return [t.to_dict() for t in self._transitions[-limit:]]
    
    def _calculate_volatility(self, returns: pd.Series) -> float:
        """Calculate realized volatility."""
        return float(returns.std())
    
    def _calculate_trend(self, prices: pd.Series) -> float:
        """Calculate trend as annualized return."""
        if len(prices) < 2:
            return 0.0
        
        total_return = prices.iloc[-1] / prices.iloc[0] - 1
        days = len(prices)
        annualized = total_return * (252 / days)
        
        return float(annualized)
    
    def _classify_volatility_regime(self, volatility: float) -> VolatilityRegime:
        """Classify volatility regime."""
        annualized_vol = volatility * np.sqrt(252)
        
        if annualized_vol > self.config.vol_extreme_threshold:
            return VolatilityRegime.EXTREME
        elif annualized_vol > self.config.vol_high_threshold:
            return VolatilityRegime.HIGH
        elif annualized_vol < self.config.vol_low_threshold:
            return VolatilityRegime.LOW
        else:
            return VolatilityRegime.NORMAL
    
    def _classify_regime(
        self,
        volatility: float,
        trend: float,
        returns: pd.Series,
    ) -> Tuple[MarketRegime, float, str]:
        """
        Classify market regime based on multiple factors.
        
        Returns:
            Tuple of (regime, confidence, trigger)
        """
        annualized_vol = volatility * np.sqrt(252)
        
        # Check for crisis conditions
        drawdown = self._calculate_max_drawdown(returns)
        if drawdown > 0.15 and annualized_vol > 0.40:
            return MarketRegime.CRISIS, 0.9, "high_drawdown_and_volatility"
        
        # Check volatility extremes
        if annualized_vol > self.config.vol_extreme_threshold:
            return MarketRegime.HIGH_VOLATILITY, 0.85, "extreme_volatility"
        
        if annualized_vol < self.config.vol_low_threshold * 0.7:
            return MarketRegime.LOW_VOLATILITY, 0.8, "very_low_volatility"
        
        # Check trends
        if trend > self.config.trend_threshold:
            confidence = min(0.9, 0.5 + abs(trend))
            return MarketRegime.TRENDING_UP, confidence, "positive_trend"
        
        if trend < -self.config.trend_threshold:
            confidence = min(0.9, 0.5 + abs(trend))
            return MarketRegime.TRENDING_DOWN, confidence, "negative_trend"
        
        # Default to ranging
        return MarketRegime.RANGING, 0.6, "no_clear_trend"
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return float(abs(drawdown.min()))


def create_mock_regime_data(days: int = 504) -> pd.Series:
    """Create mock price data with different regimes for testing."""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=days, freq="D")
    
    prices = [100.0]
    
    for i in range(1, days):
        # Create different regime periods
        if i < 100:
            # Trending up
            drift = 0.001
            vol = 0.015
        elif i < 200:
            # Ranging
            drift = 0.0
            vol = 0.01
        elif i < 280:
            # High volatility
            drift = 0.0
            vol = 0.035
        elif i < 350:
            # Trending down
            drift = -0.001
            vol = 0.02
        else:
            # Recovery/normal
            drift = 0.0005
            vol = 0.018
        
        ret = np.random.normal(drift, vol)
        prices.append(prices[-1] * (1 + ret))
    
    return pd.Series(prices, index=dates)
