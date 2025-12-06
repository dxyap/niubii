"""
Ensemble Models
===============
Combine multiple models for robust predictions.
"""

import logging
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .gradient_boost import GradientBoostModel, ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfig:
    """Configuration for ensemble models."""

    # Ensemble method
    method: str = "weighted_average"  # "weighted_average", "voting", "stacking"

    # Model configurations
    use_xgboost: bool = True
    use_lightgbm: bool = True
    use_sklearn: bool = True

    # Weights (if using weighted_average)
    weights: dict[str, float] | None = None

    # Diversity settings
    use_different_features: bool = False
    feature_fraction: float = 0.8


class EnsembleModel:
    """
    Ensemble of gradient boosting models.

    Combines XGBoost, LightGBM, and sklearn models for robust predictions.
    """

    def __init__(self, config: EnsembleConfig | None = None):
        """Initialize ensemble with configuration."""
        self.config = config or EnsembleConfig()
        self.models: dict[str, GradientBoostModel] = {}
        self.weights: dict[str, float] = {}
        self.feature_names: list[str] = []
        self.metadata: dict[str, Any] = {}
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """Check if ensemble has been trained."""
        return self._is_fitted

    def _create_models(self, task: str = "classification"):
        """Create component models."""
        self.models = {}

        if self.config.use_xgboost:
            self.models['xgboost'] = GradientBoostModel(
                ModelConfig(model_type='xgboost', task=task)
            )

        if self.config.use_lightgbm:
            self.models['lightgbm'] = GradientBoostModel(
                ModelConfig(model_type='lightgbm', task=task)
            )

        if self.config.use_sklearn:
            self.models['sklearn'] = GradientBoostModel(
                ModelConfig(model_type='sklearn', task=task)
            )

        # Initialize weights
        n_models = len(self.models)
        if self.config.weights:
            self.weights = self.config.weights
        else:
            # Equal weights by default
            self.weights = dict.fromkeys(self.models, 1.0 / n_models)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        task: str = "classification",
    ) -> 'EnsembleModel':
        """
        Train all models in the ensemble.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Optional validation features
            y_val: Optional validation targets
            task: "classification" or "regression"

        Returns:
            Self for chaining
        """
        self._create_models(task)
        self.feature_names = list(X_train.columns)

        self.metadata = {
            'trained_at': datetime.now().isoformat(),
            'n_train_samples': len(X_train),
            'n_features': len(self.feature_names),
            'task': task,
            'n_models': len(self.models),
        }

        val_scores = {}

        for name, model in self.models.items():
            logger.info(f"Training {name}...")

            try:
                model.fit(X_train, y_train, X_val, y_val)

                # Evaluate on validation set for weight calibration
                if X_val is not None:
                    metrics = model.evaluate(X_val, y_val)
                    val_scores[name] = metrics.get('roc_auc', metrics.get('r2', 0.5))

            except Exception as e:
                logger.warning(f"Failed to train {name}: {e}")
                continue

        # Update weights based on validation performance
        if val_scores and self.config.method == "weighted_average":
            self._calibrate_weights(val_scores)

        self._is_fitted = True
        self.metadata['model_scores'] = val_scores
        self.metadata['weights'] = self.weights

        logger.info(f"Ensemble trained with {len(self.models)} models")
        return self

    def _calibrate_weights(self, scores: dict[str, float]):
        """Calibrate model weights based on validation scores."""
        total = sum(scores.values())
        if total > 0:
            self.weights = {name: score / total for name, score in scores.items()}

        logger.info(f"Calibrated weights: {self.weights}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make ensemble predictions.

        Args:
            X: Features DataFrame

        Returns:
            Ensemble predictions
        """
        if not self._is_fitted:
            raise ValueError("Ensemble must be fitted before predictions")

        if self.config.method == "weighted_average":
            return self._predict_weighted_average(X)
        elif self.config.method == "voting":
            return self._predict_voting(X)
        else:
            return self._predict_weighted_average(X)

    def _predict_weighted_average(self, X: pd.DataFrame) -> np.ndarray:
        """Weighted average of predictions."""
        predictions = []
        weights = []

        for name, model in self.models.items():
            if model.is_fitted:
                try:
                    pred = model.predict(X)
                    predictions.append(pred)
                    weights.append(self.weights.get(name, 1.0))
                except Exception as e:
                    logger.warning(f"Prediction failed for {name}: {e}")

        if not predictions:
            raise ValueError("No models produced predictions")

        # Weighted average
        weights = np.array(weights)
        weights = weights / weights.sum()

        ensemble_pred = np.average(predictions, axis=0, weights=weights)

        # Round for classification
        task = list(self.models.values())[0].config.task
        if task == "classification":
            ensemble_pred = (ensemble_pred > 0.5).astype(int)

        return ensemble_pred

    def _predict_voting(self, X: pd.DataFrame) -> np.ndarray:
        """Majority voting for classification."""
        predictions = []

        for name, model in self.models.items():
            if model.is_fitted:
                try:
                    pred = model.predict(X)
                    predictions.append(pred)
                except Exception as e:
                    logger.warning(f"Prediction failed for {name}: {e}")

        if not predictions:
            raise ValueError("No models produced predictions")

        # Majority vote
        stacked = np.stack(predictions, axis=0)
        ensemble_pred = np.round(np.mean(stacked, axis=0)).astype(int)

        return ensemble_pred

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get ensemble prediction probabilities.

        Args:
            X: Features DataFrame

        Returns:
            Probability array
        """
        if not self._is_fitted:
            raise ValueError("Ensemble must be fitted before predictions")

        probas = []
        weights = []

        for name, model in self.models.items():
            if model.is_fitted:
                try:
                    proba = model.predict_proba(X)
                    probas.append(proba)
                    weights.append(self.weights.get(name, 1.0))
                except Exception as e:
                    logger.warning(f"Proba prediction failed for {name}: {e}")

        if not probas:
            raise ValueError("No models produced probability predictions")

        # Weighted average of probabilities
        weights = np.array(weights)
        weights = weights / weights.sum()

        ensemble_proba = np.average(probas, axis=0, weights=weights)

        return ensemble_proba

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        """
        Evaluate ensemble and individual models.

        Args:
            X: Features
            y: True labels

        Returns:
            Dict of metrics for ensemble and each model
        """
        results = {'ensemble': {}, 'models': {}}

        # Ensemble metrics
        predictions = self.predict(X)

        task = list(self.models.values())[0].config.task

        if task == "classification":
            from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

            results['ensemble']['accuracy'] = accuracy_score(y, predictions)
            results['ensemble']['f1'] = f1_score(y, predictions, zero_division=0)

            try:
                proba = self.predict_proba(X)[:, 1]
                results['ensemble']['roc_auc'] = roc_auc_score(y, proba)
            except Exception:
                pass
        else:
            from sklearn.metrics import mean_squared_error, r2_score

            results['ensemble']['rmse'] = np.sqrt(mean_squared_error(y, predictions))
            results['ensemble']['r2'] = r2_score(y, predictions)

        # Individual model metrics
        for name, model in self.models.items():
            if model.is_fitted:
                results['models'][name] = model.evaluate(X, y)

        return results

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get aggregated feature importance across all models.

        Args:
            top_n: Number of top features

        Returns:
            DataFrame with feature importance
        """
        importance_dfs = []

        for name, model in self.models.items():
            if model.is_fitted:
                df = model.get_feature_importance(top_n=len(self.feature_names))
                df['model'] = name
                importance_dfs.append(df)

        if not importance_dfs:
            return pd.DataFrame(columns=['feature', 'importance'])

        combined = pd.concat(importance_dfs, ignore_index=True)

        # Average importance across models
        avg_importance = combined.groupby('feature')['importance'].mean().reset_index()
        avg_importance = avg_importance.sort_values('importance', ascending=False)

        return avg_importance.head(top_n)

    def get_signal(
        self,
        X: pd.DataFrame,
        threshold: float = 0.5,
    ) -> dict[str, Any]:
        """
        Generate ensemble trading signal.

        Args:
            X: Features for latest data point
            threshold: Probability threshold

        Returns:
            Signal dict with direction, confidence, and model agreement
        """
        if not self._is_fitted:
            raise ValueError("Ensemble must be fitted before generating signals")

        # Get individual model signals
        model_signals = []

        for name, model in self.models.items():
            if model.is_fitted:
                try:
                    signal = model.get_signal(X, threshold)
                    signal['model'] = name
                    model_signals.append(signal)
                except Exception:
                    continue

        if not model_signals:
            return {'signal': 'NEUTRAL', 'confidence': 0, 'error': 'No model signals'}

        # Ensemble signal from probabilities
        proba = self.predict_proba(X)
        prob_up = proba[:, 1][-1] if len(proba.shape) > 1 else proba[-1]

        # Calculate agreement
        signals = [s['signal'] for s in model_signals]
        bullish_count = sum(1 for s in signals if s == 'BULLISH')
        bearish_count = sum(1 for s in signals if s == 'BEARISH')

        agreement = max(bullish_count, bearish_count) / len(signals)

        # Determine ensemble signal
        if prob_up > threshold + 0.1:
            signal = "BULLISH"
            confidence = min((prob_up - 0.5) * 2 * agreement, 1.0)
        elif prob_up < threshold - 0.1:
            signal = "BEARISH"
            confidence = min((0.5 - prob_up) * 2 * agreement, 1.0)
        else:
            signal = "NEUTRAL"
            confidence = 1.0 - abs(prob_up - 0.5) * 2

        return {
            'signal': signal,
            'confidence': round(confidence, 3),
            'probability_up': round(prob_up, 3),
            'probability_down': round(1 - prob_up, 3),
            'model_agreement': round(agreement, 3),
            'bullish_models': bullish_count,
            'bearish_models': bearish_count,
            'neutral_models': len(signals) - bullish_count - bearish_count,
            'model_signals': model_signals,
        }

    def save(self, path: str | Path) -> None:
        """Save ensemble to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            'config': self.config,
            'models': self.models,
            'weights': self.weights,
            'feature_names': self.feature_names,
            'metadata': self.metadata,
            'is_fitted': self._is_fitted,
        }

        with open(path, 'wb') as f:
            pickle.dump(state, f)

        logger.info(f"Ensemble saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> 'EnsembleModel':
        """Load ensemble from disk."""
        path = Path(path)

        with open(path, 'rb') as f:
            state = pickle.load(f)

        instance = cls(config=state['config'])
        instance.models = state['models']
        instance.weights = state['weights']
        instance.feature_names = state['feature_names']
        instance.metadata = state['metadata']
        instance._is_fitted = state['is_fitted']

        logger.info(f"Ensemble loaded from {path}")
        return instance
