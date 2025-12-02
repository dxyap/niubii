"""
Machine Learning Module
=======================
ML-powered signal generation for oil trading.

Provides:
- Feature engineering pipeline
- Gradient boosting models (XGBoost/LightGBM)
- Model training and prediction
- Performance monitoring and drift detection
"""

from .features import FeatureEngineer, FeatureConfig
from .training import ModelTrainer, TrainingConfig
from .prediction import PredictionService
from .monitoring import ModelMonitor, DriftDetector

__all__ = [
    "FeatureEngineer",
    "FeatureConfig",
    "ModelTrainer", 
    "TrainingConfig",
    "PredictionService",
    "ModelMonitor",
    "DriftDetector",
]
