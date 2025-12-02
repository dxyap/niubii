"""
Model Training Pipeline
=======================
End-to-end training pipeline for ML models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import logging
import json

from .features import FeatureEngineer, FeatureConfig
from .models import GradientBoostModel, ModelConfig, EnsembleModel

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""
    
    # Data split
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Walk-forward validation
    use_walk_forward: bool = True
    walk_forward_windows: int = 5
    
    # Cross-validation
    use_cv: bool = False
    cv_folds: int = 5
    
    # Feature engineering
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    
    # Model configuration
    model_config: ModelConfig = field(default_factory=ModelConfig)
    use_ensemble: bool = True
    
    # Output
    model_dir: str = "models"
    save_features: bool = True


class ModelTrainer:
    """
    End-to-end model training pipeline.
    
    Handles:
    - Data preparation and splitting
    - Feature engineering
    - Model training with validation
    - Walk-forward backtesting
    - Model persistence
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        """Initialize trainer with configuration."""
        self.config = config or TrainingConfig()
        self.feature_engineer = FeatureEngineer(self.config.feature_config)
        self.model = None
        self.training_results: Dict[str, Any] = {}
        self.feature_data: Optional[pd.DataFrame] = None
    
    def prepare_data(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare and split data for training.
        
        Args:
            df: Raw OHLCV DataFrame
            
        Returns:
            Tuple of (train, validation, test) DataFrames with features
        """
        logger.info(f"Preparing data from {len(df)} rows")
        
        # Create features
        feature_df = self.feature_engineer.create_features(df, include_target=True)
        
        if feature_df.empty:
            raise ValueError("Feature engineering produced empty DataFrame")
        
        self.feature_data = feature_df
        logger.info(f"Created {len(self.feature_engineer.feature_names)} features")
        
        # Time-based split (important for time series!)
        n = len(feature_df)
        train_end = int(n * self.config.train_ratio)
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))
        
        train_df = feature_df.iloc[:train_end]
        val_df = feature_df.iloc[train_end:val_end]
        test_df = feature_df.iloc[val_end:]
        
        logger.info(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        
        return train_df, val_df, test_df
    
    def train(
        self,
        df: pd.DataFrame,
        target_col: str = "target_direction",
    ) -> Dict[str, Any]:
        """
        Train model on data.
        
        Args:
            df: Raw OHLCV DataFrame
            target_col: Name of target column
            
        Returns:
            Training results dict
        """
        logger.info("Starting training pipeline")
        
        # Prepare data
        train_df, val_df, test_df = self.prepare_data(df)
        
        # Get feature columns
        feature_cols = self.feature_engineer.feature_names
        
        # Prepare X, y
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_val = val_df[feature_cols]
        y_val = val_df[target_col]
        X_test = test_df[feature_cols]
        y_test = test_df[target_col]
        
        # Train model
        if self.config.use_ensemble:
            self.model = EnsembleModel()
            self.model.fit(X_train, y_train, X_val, y_val, 
                          task=self.config.model_config.task)
        else:
            self.model = GradientBoostModel(self.config.model_config)
            self.model.fit(X_train, y_train, X_val, y_val)
        
        # Evaluate
        train_metrics = self.model.evaluate(X_train, y_train)
        val_metrics = self.model.evaluate(X_val, y_val)
        test_metrics = self.model.evaluate(X_test, y_test)
        
        # Store results
        self.training_results = {
            'trained_at': datetime.now().isoformat(),
            'n_samples': len(train_df) + len(val_df) + len(test_df),
            'n_features': len(feature_cols),
            'feature_names': feature_cols,
            'target': target_col,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'feature_importance': self.model.get_feature_importance(20).to_dict('records'),
        }
        
        logger.info(f"Training complete. Test accuracy: {test_metrics.get('accuracy', 'N/A')}")
        
        return self.training_results
    
    def walk_forward_train(
        self,
        df: pd.DataFrame,
        target_col: str = "target_direction",
    ) -> Dict[str, Any]:
        """
        Walk-forward training and validation.
        
        Simulates real trading by training on past data and testing on future.
        
        Args:
            df: Raw OHLCV DataFrame
            target_col: Name of target column
            
        Returns:
            Walk-forward results
        """
        logger.info("Starting walk-forward training")
        
        # Create features
        feature_df = self.feature_engineer.create_features(df, include_target=True)
        
        if feature_df.empty:
            raise ValueError("Feature engineering produced empty DataFrame")
        
        self.feature_data = feature_df
        feature_cols = self.feature_engineer.feature_names
        
        n = len(feature_df)
        window_size = n // (self.config.walk_forward_windows + 1)
        
        fold_results = []
        all_predictions = []
        all_actuals = []
        
        for i in range(self.config.walk_forward_windows):
            # Define train/test split for this fold
            train_end = (i + 1) * window_size
            test_end = min((i + 2) * window_size, n)
            
            train_df = feature_df.iloc[:train_end]
            test_df = feature_df.iloc[train_end:test_end]
            
            if len(train_df) < 100 or len(test_df) < 10:
                continue
            
            # Split train into train/val
            val_size = int(len(train_df) * 0.2)
            actual_train = train_df.iloc[:-val_size]
            val_df = train_df.iloc[-val_size:]
            
            X_train = actual_train[feature_cols]
            y_train = actual_train[target_col]
            X_val = val_df[feature_cols]
            y_val = val_df[target_col]
            X_test = test_df[feature_cols]
            y_test = test_df[target_col]
            
            # Train model
            if self.config.use_ensemble:
                model = EnsembleModel()
                model.fit(X_train, y_train, X_val, y_val,
                         task=self.config.model_config.task)
            else:
                model = GradientBoostModel(self.config.model_config)
                model.fit(X_train, y_train, X_val, y_val)
            
            # Evaluate
            test_metrics = model.evaluate(X_test, y_test)
            predictions = model.predict(X_test)
            
            fold_results.append({
                'fold': i + 1,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'metrics': test_metrics,
            })
            
            all_predictions.extend(predictions.tolist())
            all_actuals.extend(y_test.tolist())
            
            logger.info(f"Fold {i+1}: accuracy={test_metrics.get('accuracy', 'N/A'):.4f}")
        
        # Aggregate results
        avg_metrics = {}
        for metric in fold_results[0]['metrics'].keys():
            values = [f['metrics'][metric] for f in fold_results if metric in f['metrics']]
            avg_metrics[metric] = np.mean(values)
            avg_metrics[f'{metric}_std'] = np.std(values)
        
        # Keep the last trained model
        self.model = model
        
        self.training_results = {
            'method': 'walk_forward',
            'n_folds': len(fold_results),
            'fold_results': fold_results,
            'avg_metrics': avg_metrics,
            'total_predictions': len(all_predictions),
            'feature_names': feature_cols,
        }
        
        logger.info(f"Walk-forward complete. Avg accuracy: {avg_metrics.get('accuracy', 'N/A'):.4f}")
        
        return self.training_results
    
    def save_model(self, path: Optional[Union[str, Path]] = None) -> Path:
        """
        Save trained model to disk.
        
        Args:
            path: Optional path override
            
        Returns:
            Path where model was saved
        """
        if self.model is None:
            raise ValueError("No model to save. Train first.")
        
        if path is None:
            model_dir = Path(self.config.model_dir)
            model_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = model_dir / f"model_{timestamp}.pkl"
        else:
            path = Path(path)
        
        self.model.save(path)
        
        # Save training results
        results_path = path.with_suffix('.json')
        with open(results_path, 'w') as f:
            # Convert numpy types for JSON serialization
            results = self._serialize_results(self.training_results)
            json.dump(results, f, indent=2)
        
        logger.info(f"Model and results saved to {path}")
        return path
    
    def load_model(self, path: Union[str, Path]) -> None:
        """
        Load model from disk.
        
        Args:
            path: Path to model file
        """
        path = Path(path)
        
        if self.config.use_ensemble:
            self.model = EnsembleModel.load(path)
        else:
            self.model = GradientBoostModel.load(path)
        
        # Load training results if available
        results_path = path.with_suffix('.json')
        if results_path.exists():
            with open(results_path, 'r') as f:
                self.training_results = json.load(f)
        
        logger.info(f"Model loaded from {path}")
    
    def _serialize_results(self, obj: Any) -> Any:
        """Convert numpy types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._serialize_results(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_results(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance from trained model."""
        if self.model is None:
            raise ValueError("No model trained")
        
        return self.model.get_feature_importance(top_n)
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive training report.
        
        Returns:
            Report dict with all training information
        """
        if not self.training_results:
            raise ValueError("No training results available")
        
        report = {
            'summary': {
                'trained_at': self.training_results.get('trained_at'),
                'n_samples': self.training_results.get('n_samples'),
                'n_features': self.training_results.get('n_features'),
            },
            'performance': {},
            'feature_importance': self.training_results.get('feature_importance', []),
        }
        
        # Add metrics
        if 'test_metrics' in self.training_results:
            report['performance'] = self.training_results['test_metrics']
        elif 'avg_metrics' in self.training_results:
            report['performance'] = self.training_results['avg_metrics']
        
        # Add category breakdown
        categories = self.feature_engineer.get_feature_importance_template()
        importance_df = self.model.get_feature_importance(len(categories))
        
        if not importance_df.empty:
            importance_df['category'] = importance_df['feature'].map(categories)
            category_importance = importance_df.groupby('category')['importance'].sum()
            report['category_importance'] = category_importance.to_dict()
        
        return report
