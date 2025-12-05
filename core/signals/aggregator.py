"""
Signal Aggregation
==================
Combines signals from multiple sources into actionable trading signals.
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """Represents a trading signal."""
    signal_id: str
    instrument: str
    direction: str  # LONG, SHORT, NEUTRAL
    confidence: float
    entry_price: float
    stop_loss: float
    target_price: float
    time_horizon: str
    source: str
    drivers: list[dict]
    timestamp: datetime

    def to_dict(self) -> dict:
        return {
            "signal_id": self.signal_id,
            "instrument": self.instrument,
            "direction": self.direction,
            "confidence": self.confidence,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "target_price": self.target_price,
            "time_horizon": self.time_horizon,
            "source": self.source,
            "drivers": self.drivers,
            "timestamp": self.timestamp.isoformat(),
        }


class SignalAggregator:
    """
    Aggregates signals from multiple sources.

    Features:
    - Weighted signal combination
    - Confidence adjustment
    - Signal filtering and validation
    - Historical signal tracking
    """

    def __init__(self):
        """Initialize signal aggregator."""
        self.signal_history: list[TradingSignal] = []

        # Default source weights
        self.source_weights = {
            "technical": 0.35,
            "fundamental": 0.40,
            "ml_model": 0.25,
        }

    def aggregate_signals(
        self,
        technical_signal: dict,
        fundamental_signal: dict,
        ml_signal: dict | None = None,
        instrument: str = "CL1 Comdty",
        current_price: float = 72.50
    ) -> TradingSignal:
        """
        Aggregate signals from multiple sources.

        Args:
            technical_signal: Technical analysis signal
            fundamental_signal: Fundamental analysis signal
            ml_signal: Optional ML model signal
            instrument: Trading instrument
            current_price: Current market price

        Returns:
            Aggregated trading signal
        """
        # Score mapping
        score_map = {"LONG": 1, "SHORT": -1, "NEUTRAL": 0}

        # Calculate weighted score
        tech_score = score_map.get(technical_signal.get("signal", "NEUTRAL"), 0)
        tech_conf = technical_signal.get("confidence", 50) / 100

        fund_score = score_map.get(fundamental_signal.get("signal", "NEUTRAL"), 0)
        fund_conf = fundamental_signal.get("confidence", 50) / 100

        if ml_signal:
            ml_score = score_map.get(ml_signal.get("signal", "NEUTRAL"), 0)
            ml_conf = ml_signal.get("confidence", 50) / 100

            total_score = (
                self.source_weights["technical"] * tech_score * tech_conf +
                self.source_weights["fundamental"] * fund_score * fund_conf +
                self.source_weights["ml_model"] * ml_score * ml_conf
            )
            total_weight = sum(self.source_weights.values())
        else:
            # Redistribute ML weight
            adj_tech_weight = self.source_weights["technical"] / (1 - self.source_weights["ml_model"])
            adj_fund_weight = self.source_weights["fundamental"] / (1 - self.source_weights["ml_model"])

            total_score = (
                adj_tech_weight * tech_score * tech_conf +
                adj_fund_weight * fund_score * fund_conf
            )
            total_weight = adj_tech_weight + adj_fund_weight

        normalized_score = total_score / total_weight if total_weight > 0 else 0

        # Determine direction
        if normalized_score > 0.25:
            direction = "LONG"
        elif normalized_score < -0.25:
            direction = "SHORT"
        else:
            direction = "NEUTRAL"

        # Calculate confidence
        confidence = min(abs(normalized_score) * 100 + 30, 95)

        # Calculate entry, stop, and target
        volatility = current_price * 0.015  # Assume 1.5% daily volatility

        if direction == "LONG":
            entry_price = current_price
            stop_loss = round(current_price - (2 * volatility), 2)
            target_price = round(current_price + (3 * volatility), 2)
        elif direction == "SHORT":
            entry_price = current_price
            stop_loss = round(current_price + (2 * volatility), 2)
            target_price = round(current_price - (3 * volatility), 2)
        else:
            entry_price = current_price
            stop_loss = round(current_price - (2 * volatility), 2)
            target_price = round(current_price + (2 * volatility), 2)

        # Compile drivers
        drivers = []

        # Technical drivers
        if "components" in technical_signal:
            for comp_name, comp_data in technical_signal["components"].items():
                if comp_data.get("signal") not in ["NEUTRAL", "HOLD", "RANGE_BOUND"]:
                    drivers.append({
                        "source": f"Technical: {comp_name}",
                        "signal": comp_data.get("signal"),
                        "weight": round(self.source_weights["technical"] / 5 * 100, 1),
                    })
        else:
            drivers.append({
                "source": "Technical Analysis",
                "signal": technical_signal.get("signal"),
                "weight": round(self.source_weights["technical"] * 100, 1),
            })

        # Fundamental drivers
        if "components" in fundamental_signal:
            for comp_name, comp_data in fundamental_signal["components"].items():
                if comp_data.get("signal") not in ["NEUTRAL"]:
                    drivers.append({
                        "source": f"Fundamental: {comp_name}",
                        "signal": comp_data.get("signal"),
                        "weight": round(self.source_weights["fundamental"] / 5 * 100, 1),
                    })
        else:
            drivers.append({
                "source": "Fundamental Analysis",
                "signal": fundamental_signal.get("signal"),
                "weight": round(self.source_weights["fundamental"] * 100, 1),
            })

        # ML driver
        if ml_signal:
            drivers.append({
                "source": "ML Model",
                "signal": ml_signal.get("signal"),
                "weight": round(self.source_weights["ml_model"] * 100, 1),
            })

        # Create signal object
        signal = TradingSignal(
            signal_id=f"SIG-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8]}",
            instrument=instrument,
            direction=direction,
            confidence=round(confidence, 1),
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_price=target_price,
            time_horizon="5-10 Days",
            source="Composite",
            drivers=drivers,
            timestamp=datetime.now(),
        )

        # Store in history
        self.signal_history.append(signal)

        return signal

    def get_signal_performance(self, lookback: int = 30) -> dict:
        """
        Calculate signal performance statistics.

        Args:
            lookback: Number of recent signals to analyze

        Returns:
            Performance metrics
        """
        if not self.signal_history:
            return {
                "total_signals": 0,
                "win_rate": 0,
                "avg_confidence": 0,
            }

        recent_signals = self.signal_history[-lookback:]

        total = len(recent_signals)
        long_signals = len([s for s in recent_signals if s.direction == "LONG"])
        short_signals = len([s for s in recent_signals if s.direction == "SHORT"])
        neutral_signals = len([s for s in recent_signals if s.direction == "NEUTRAL"])

        avg_confidence = np.mean([s.confidence for s in recent_signals])

        return {
            "total_signals": total,
            "long_signals": long_signals,
            "short_signals": short_signals,
            "neutral_signals": neutral_signals,
            "avg_confidence": round(avg_confidence, 1),
            "signal_history": [s.to_dict() for s in recent_signals[-10:]],
        }

    def filter_signals(
        self,
        min_confidence: float = 60,
        direction_filter: str | None = None
    ) -> list[TradingSignal]:
        """
        Filter signals by criteria.

        Args:
            min_confidence: Minimum confidence threshold
            direction_filter: Optional direction filter

        Returns:
            Filtered signals
        """
        filtered = [
            s for s in self.signal_history
            if s.confidence >= min_confidence
        ]

        if direction_filter:
            filtered = [s for s in filtered if s.direction == direction_filter]

        return filtered

    def get_latest_signals(self, n: int = 5) -> list[dict]:
        """
        Get latest N signals.

        Args:
            n: Number of signals

        Returns:
            List of signal dictionaries
        """
        return [s.to_dict() for s in self.signal_history[-n:]]

    def set_source_weights(self, weights: dict[str, float]) -> None:
        """
        Update source weights.

        Args:
            weights: New weight dictionary
        """
        total = sum(weights.values())
        self.source_weights = {k: v/total for k, v in weights.items()}


class MLSignalGenerator:
    """
    ML-based signal generator.

    Integrates machine learning models with the signal aggregation pipeline.
    """

    def __init__(self, model_path: Path | None = None):
        """
        Initialize ML signal generator.

        Args:
            model_path: Path to saved model file
        """
        self.model_path = model_path
        self.prediction_service = None
        self._is_available = False

        # Try to initialize ML components
        self._initialize()

    def _initialize(self) -> None:
        """Initialize ML components."""
        try:
            from core.ml import PredictionService

            if self.model_path and Path(self.model_path).exists():
                self.prediction_service = PredictionService(self.model_path)
                self._is_available = self.prediction_service.is_ready
                logger.info(f"ML signal generator initialized with model: {self.model_path}")
            else:
                # Try to find latest model
                model_dir = Path("models")
                if model_dir.exists():
                    model_files = sorted(model_dir.glob("*.pkl"))
                    if model_files:
                        self.model_path = model_files[-1]
                        self.prediction_service = PredictionService(self.model_path)
                        self._is_available = self.prediction_service.is_ready
                        logger.info(f"ML signal generator auto-loaded model: {self.model_path}")

        except ImportError as e:
            logger.warning(f"ML modules not available: {e}")
            self._is_available = False
        except Exception as e:
            logger.warning(f"Could not initialize ML signal generator: {e}")
            self._is_available = False

    @property
    def is_available(self) -> bool:
        """Check if ML signals are available."""
        return self._is_available

    def generate_signal(
        self,
        historical_data: pd.DataFrame,
        ticker: str = "CO1 Comdty",
    ) -> dict[str, Any]:
        """
        Generate ML signal from historical data.

        Args:
            historical_data: OHLCV DataFrame with sufficient history
            ticker: Instrument ticker

        Returns:
            Signal dict compatible with SignalAggregator
        """
        if not self._is_available:
            return {
                "signal": "NEUTRAL",
                "confidence": 0,
                "error": "ML model not available",
                "source": "ml_model",
            }

        try:
            # Get prediction
            result = self.prediction_service.predict(historical_data)

            # Map signal format
            signal_map = {
                "BULLISH": "LONG",
                "BEARISH": "SHORT",
                "NEUTRAL": "NEUTRAL",
            }

            ml_signal = result.get("signal", "NEUTRAL")

            return {
                "signal": signal_map.get(ml_signal, "NEUTRAL"),
                "confidence": result.get("confidence", 0) * 100,
                "probability_up": result.get("probability_up", 0.5),
                "probability_down": result.get("probability_down", 0.5),
                "horizon": result.get("horizon", 5),
                "source": "ml_model",
                "model_type": result.get("model_type", "unknown"),
                "timestamp": result.get("timestamp", datetime.now().isoformat()),
            }

        except Exception as e:
            logger.warning(f"ML signal generation failed: {e}")
            return {
                "signal": "NEUTRAL",
                "confidence": 0,
                "error": str(e),
                "source": "ml_model",
            }

    def get_signal_with_data_loader(
        self,
        data_loader: Any,
        ticker: str = "CO1 Comdty",
        lookback_days: int = 365,
    ) -> dict[str, Any]:
        """
        Generate ML signal using data loader.

        Args:
            data_loader: DataLoader instance
            ticker: Bloomberg ticker
            lookback_days: Days of history to use

        Returns:
            Signal dict
        """
        if not self._is_available:
            return {
                "signal": "NEUTRAL",
                "confidence": 0,
                "error": "ML model not available",
                "source": "ml_model",
            }

        try:
            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)

            df = data_loader.get_historical(ticker, start_date, end_date)

            if df is None or df.empty:
                return {
                    "signal": "NEUTRAL",
                    "confidence": 0,
                    "error": f"No data for {ticker}",
                    "source": "ml_model",
                }

            return self.generate_signal(df, ticker)

        except Exception as e:
            logger.warning(f"ML signal generation with data loader failed: {e}")
            return {
                "signal": "NEUTRAL",
                "confidence": 0,
                "error": str(e),
                "source": "ml_model",
            }

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model."""
        if not self._is_available:
            return {"status": "not_available"}

        return self.prediction_service.get_model_info()
