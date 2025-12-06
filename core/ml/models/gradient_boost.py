"""
Gradient Boosting Models
========================
XGBoost and LightGBM models for oil price prediction.
"""

import logging
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for gradient boosting models."""

    # Model type
    model_type: str = "xgboost"  # "xgboost" or "lightgbm"
    task: str = "classification"  # "classification" or "regression"

    # Core hyperparameters
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    min_child_weight: int = 1
    subsample: float = 0.8
    colsample_bytree: float = 0.8

    # Regularization
    reg_alpha: float = 0.0  # L1 regularization
    reg_lambda: float = 1.0  # L2 regularization

    # Early stopping
    early_stopping_rounds: int = 20
    eval_metric: str = "auc"  # "auc", "logloss", "rmse", "mae"

    # Other settings
    random_state: int = 42
    n_jobs: int = -1
    verbose: bool = False


class GradientBoostModel:
    """
    Gradient boosting model wrapper for XGBoost/LightGBM.

    Provides a unified interface for training, prediction, and evaluation.
    """

    def __init__(self, config: ModelConfig | None = None):
        """Initialize model with configuration."""
        self.config = config or ModelConfig()
        self.model = None
        self.feature_names: list[str] = []
        self.feature_importance: dict[str, float] = {}
        self.training_history: dict[str, list[float]] = {}
        self.metadata: dict[str, Any] = {}
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """Check if model has been trained."""
        return self._is_fitted

    def _create_model(self):
        """Create the underlying model based on config."""
        params = self._get_params()

        if self.config.model_type == "xgboost":
            try:
                import xgboost as xgb

                if self.config.task == "classification":
                    self.model = xgb.XGBClassifier(**params)
                else:
                    self.model = xgb.XGBRegressor(**params)

            except ImportError:
                logger.warning("XGBoost not available, falling back to sklearn")
                self._create_sklearn_fallback()

        elif self.config.model_type == "lightgbm":
            try:
                import lightgbm as lgb

                if self.config.task == "classification":
                    self.model = lgb.LGBMClassifier(**params)
                else:
                    self.model = lgb.LGBMRegressor(**params)

            except ImportError:
                logger.warning("LightGBM not available, falling back to sklearn")
                self._create_sklearn_fallback()
        else:
            self._create_sklearn_fallback()

    def _create_sklearn_fallback(self):
        """Create sklearn gradient boosting as fallback."""
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

        params = {
            'n_estimators': self.config.n_estimators,
            'max_depth': self.config.max_depth,
            'learning_rate': self.config.learning_rate,
            'subsample': self.config.subsample,
            'random_state': self.config.random_state,
        }

        if self.config.task == "classification":
            self.model = GradientBoostingClassifier(**params)
        else:
            self.model = GradientBoostingRegressor(**params)

        self.config.model_type = "sklearn"
        logger.info("Using sklearn GradientBoosting as fallback")

    def _get_params(self) -> dict[str, Any]:
        """Get model parameters from config."""
        params = {
            'n_estimators': self.config.n_estimators,
            'max_depth': self.config.max_depth,
            'learning_rate': self.config.learning_rate,
            'subsample': self.config.subsample,
            'colsample_bytree': self.config.colsample_bytree,
            'reg_alpha': self.config.reg_alpha,
            'reg_lambda': self.config.reg_lambda,
            'random_state': self.config.random_state,
            'n_jobs': self.config.n_jobs,
        }

        if self.config.model_type == "xgboost":
            params['min_child_weight'] = self.config.min_child_weight
            params['use_label_encoder'] = False
            if self.config.task == "classification":
                params['eval_metric'] = 'logloss'
                params['objective'] = 'binary:logistic'

        elif self.config.model_type == "lightgbm":
            params['min_child_samples'] = self.config.min_child_weight
            params['verbose'] = -1

        return params

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> 'GradientBoostModel':
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Optional validation features
            y_val: Optional validation targets

        Returns:
            Self for chaining
        """
        self._create_model()
        self.feature_names = list(X_train.columns)

        # Record metadata
        self.metadata = {
            'trained_at': datetime.now().isoformat(),
            'n_train_samples': len(X_train),
            'n_features': len(self.feature_names),
            'model_type': self.config.model_type,
            'task': self.config.task,
        }

        try:
            if X_val is not None and y_val is not None:
                # Train with early stopping
                if self.config.model_type in ["xgboost", "lightgbm"]:
                    eval_set = [(X_train, y_train), (X_val, y_val)]

                    self.model.fit(
                        X_train, y_train,
                        eval_set=eval_set,
                        verbose=self.config.verbose,
                    )

                    # Store training history
                    if hasattr(self.model, 'evals_result'):
                        self.training_history = self.model.evals_result()
                else:
                    self.model.fit(X_train, y_train)
            else:
                self.model.fit(X_train, y_train)

            # Extract feature importance
            self._extract_feature_importance()

            self._is_fitted = True
            self.metadata['n_val_samples'] = len(X_val) if X_val is not None else 0

            logger.info(f"Model trained successfully on {len(X_train)} samples")

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features DataFrame

        Returns:
            Predictions array
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Ensure columns match training data
        X = X[self.feature_names]

        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities (classification only).

        Args:
            X: Features DataFrame

        Returns:
            Probability array (n_samples, n_classes)
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        if self.config.task != "classification":
            raise ValueError("predict_proba only available for classification")

        X = X[self.feature_names]
        return self.model.predict_proba(X)

    def _extract_feature_importance(self):
        """Extract and store feature importance."""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            self.feature_importance = dict(zip(self.feature_names, importances))

            # Sort by importance
            self.feature_importance = dict(
                sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
            )

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance as DataFrame.

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.feature_importance:
            return pd.DataFrame(columns=['feature', 'importance'])

        df = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in self.feature_importance.items()
        ])

        return df.head(top_n)

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X: Features
            y: True labels/values

        Returns:
            Dict of metrics
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before evaluation")

        predictions = self.predict(X)

        metrics = {}

        if self.config.task == "classification":
            from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score, roc_auc_score

            metrics['accuracy'] = accuracy_score(y, predictions)
            metrics['precision'] = precision_score(y, predictions, zero_division=0)
            metrics['recall'] = recall_score(y, predictions, zero_division=0)
            metrics['f1'] = f1_score(y, predictions, zero_division=0)

            try:
                proba = self.predict_proba(X)[:, 1]
                metrics['roc_auc'] = roc_auc_score(y, proba)
                metrics['log_loss'] = log_loss(y, proba)
            except Exception:
                pass

        else:  # regression
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

            metrics['rmse'] = np.sqrt(mean_squared_error(y, predictions))
            metrics['mae'] = mean_absolute_error(y, predictions)
            metrics['r2'] = r2_score(y, predictions)

        return metrics

    def save(self, path: str | Path) -> None:
        """
        Save model to disk.

        Args:
            path: File path for saving
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            'model': self.model,
            'config': self.config,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'training_history': self.training_history,
            'metadata': self.metadata,
            'is_fitted': self._is_fitted,
        }

        with open(path, 'wb') as f:
            pickle.dump(state, f)

        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> 'GradientBoostModel':
        """
        Load model from disk.

        Args:
            path: File path to load from

        Returns:
            Loaded model instance
        """
        path = Path(path)

        with open(path, 'rb') as f:
            state = pickle.load(f)

        instance = cls(config=state['config'])
        instance.model = state['model']
        instance.feature_names = state['feature_names']
        instance.feature_importance = state['feature_importance']
        instance.training_history = state['training_history']
        instance.metadata = state['metadata']
        instance._is_fitted = state['is_fitted']

        logger.info(f"Model loaded from {path}")
        return instance

    def get_signal(
        self,
        X: pd.DataFrame,
        threshold: float = 0.5,
    ) -> dict[str, Any]:
        """
        Generate trading signal from model prediction.

        Args:
            X: Features for latest data point
            threshold: Probability threshold for signal

        Returns:
            Signal dict with direction, confidence, and metadata
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before generating signals")

        if self.config.task == "classification":
            proba = self.predict_proba(X)

            # Get probability of positive class (price going up)
            prob_up = proba[:, 1][-1] if len(proba.shape) > 1 else proba[-1]

            # Determine signal
            if prob_up > threshold + 0.1:
                signal = "BULLISH"
                confidence = min((prob_up - 0.5) * 2, 1.0)
            elif prob_up < threshold - 0.1:
                signal = "BEARISH"
                confidence = min((0.5 - prob_up) * 2, 1.0)
            else:
                signal = "NEUTRAL"
                confidence = 1.0 - abs(prob_up - 0.5) * 2

            return {
                'signal': signal,
                'confidence': round(confidence, 3),
                'probability_up': round(prob_up, 3),
                'probability_down': round(1 - prob_up, 3),
                'model_type': self.config.model_type,
                'horizon': self.metadata.get('horizon', 5),
            }

        else:  # regression
            prediction = self.predict(X)[-1]

            if prediction > 0.01:
                signal = "BULLISH"
                confidence = min(abs(prediction) * 10, 1.0)
            elif prediction < -0.01:
                signal = "BEARISH"
                confidence = min(abs(prediction) * 10, 1.0)
            else:
                signal = "NEUTRAL"
                confidence = 0.5

            return {
                'signal': signal,
                'confidence': round(confidence, 3),
                'predicted_return': round(prediction, 4),
                'model_type': self.config.model_type,
                'horizon': self.metadata.get('horizon', 5),
            }
