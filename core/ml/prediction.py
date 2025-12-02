"""
Prediction Service
==================
Real-time ML predictions for trading signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
import logging

from .features import FeatureEngineer, FeatureConfig
from .models import GradientBoostModel, EnsembleModel

logger = logging.getLogger(__name__)


class PredictionService:
    """
    Real-time prediction service for trading signals.
    
    Manages:
    - Model loading and caching
    - Feature computation
    - Signal generation
    - Prediction history
    """
    
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        feature_config: Optional[FeatureConfig] = None,
    ):
        """
        Initialize prediction service.
        
        Args:
            model_path: Path to saved model
            feature_config: Feature engineering configuration
        """
        self.model = None
        self.model_path = model_path
        self.feature_engineer = FeatureEngineer(feature_config or FeatureConfig())
        self.prediction_history: List[Dict[str, Any]] = []
        self._last_features: Optional[pd.DataFrame] = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, path: Union[str, Path]) -> None:
        """
        Load model from disk.
        
        Args:
            path: Path to model file
        """
        path = Path(path)
        
        if not path.exists():
            logger.warning(f"Model file not found: {path}")
            return
        
        try:
            # Try loading as ensemble first
            self.model = EnsembleModel.load(path)
            logger.info(f"Loaded ensemble model from {path}")
        except Exception:
            try:
                self.model = GradientBoostModel.load(path)
                logger.info(f"Loaded single model from {path}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                self.model = None
        
        self.model_path = path
    
    @property
    def is_ready(self) -> bool:
        """Check if service is ready for predictions."""
        return self.model is not None and self.model.is_fitted
    
    def predict(
        self,
        df: pd.DataFrame,
        return_features: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate prediction from market data.
        
        Args:
            df: OHLCV DataFrame (needs enough history for features)
            return_features: Whether to include features in response
            
        Returns:
            Prediction dict with signal and confidence
        """
        if not self.is_ready:
            return {
                'signal': 'UNAVAILABLE',
                'confidence': 0,
                'error': 'Model not loaded or not fitted',
                'timestamp': datetime.now().isoformat(),
            }
        
        try:
            # Create features (without target)
            features = self.feature_engineer.create_features(df, include_target=False)
            
            if features.empty:
                return {
                    'signal': 'UNAVAILABLE',
                    'confidence': 0,
                    'error': 'Insufficient data for features',
                    'timestamp': datetime.now().isoformat(),
                }
            
            self._last_features = features
            
            # Get latest features
            X = features.iloc[[-1]]
            
            # Ensure features match model
            model_features = self.model.feature_names
            missing = set(model_features) - set(X.columns)
            if missing:
                logger.warning(f"Missing features: {missing}")
                # Fill missing with zeros (not ideal but allows prediction)
                for feat in missing:
                    X[feat] = 0
            
            X = X[model_features]
            
            # Generate signal
            signal = self.model.get_signal(X)
            signal['timestamp'] = datetime.now().isoformat()
            signal['data_date'] = str(features.index[-1])
            
            # Add to history
            self.prediction_history.append(signal)
            if len(self.prediction_history) > 1000:
                self.prediction_history = self.prediction_history[-1000:]
            
            if return_features:
                signal['features'] = X.iloc[0].to_dict()
            
            return signal
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                'signal': 'ERROR',
                'confidence': 0,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
            }
    
    def predict_batch(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate predictions for entire DataFrame.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            DataFrame with predictions for each row
        """
        if not self.is_ready:
            raise ValueError("Model not loaded or not fitted")
        
        # Create features
        features = self.feature_engineer.create_features(df, include_target=False)
        
        if features.empty:
            raise ValueError("Feature engineering produced empty DataFrame")
        
        # Ensure features match model
        model_features = self.model.feature_names
        X = features[model_features]
        
        # Get predictions
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X)
            predictions = pd.DataFrame(index=features.index)
            predictions['prob_up'] = proba[:, 1]
            predictions['prob_down'] = proba[:, 0]
            predictions['prediction'] = (proba[:, 1] > 0.5).astype(int)
        else:
            pred = self.model.predict(X)
            predictions = pd.DataFrame(index=features.index)
            predictions['prediction'] = pred
        
        return predictions
    
    def get_signal_for_ticker(
        self,
        data_loader: Any,
        ticker: str = "CO1 Comdty",
        lookback_days: int = 365,
    ) -> Dict[str, Any]:
        """
        Get ML signal for a specific ticker using data loader.
        
        Args:
            data_loader: DataLoader instance
            ticker: Bloomberg ticker
            lookback_days: Days of history to use
            
        Returns:
            Signal dict
        """
        from datetime import datetime, timedelta
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        try:
            df = data_loader.get_historical(ticker, start_date, end_date)
            
            if df is None or df.empty:
                return {
                    'signal': 'UNAVAILABLE',
                    'confidence': 0,
                    'error': f'No data available for {ticker}',
                    'ticker': ticker,
                    'timestamp': datetime.now().isoformat(),
                }
            
            signal = self.predict(df)
            signal['ticker'] = ticker
            
            return signal
            
        except Exception as e:
            return {
                'signal': 'ERROR',
                'confidence': 0,
                'error': str(e),
                'ticker': ticker,
                'timestamp': datetime.now().isoformat(),
            }
    
    def get_prediction_history(
        self,
        last_n: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get recent prediction history.
        
        Args:
            last_n: Number of recent predictions
            
        Returns:
            List of prediction dicts
        """
        return self.prediction_history[-last_n:]
    
    def get_signal_statistics(self) -> Dict[str, Any]:
        """
        Get statistics on recent signals.
        
        Returns:
            Dict with signal statistics
        """
        if not self.prediction_history:
            return {'total': 0}
        
        signals = [p['signal'] for p in self.prediction_history if 'signal' in p]
        confidences = [p['confidence'] for p in self.prediction_history if 'confidence' in p]
        
        stats = {
            'total': len(signals),
            'bullish': sum(1 for s in signals if s == 'BULLISH'),
            'bearish': sum(1 for s in signals if s == 'BEARISH'),
            'neutral': sum(1 for s in signals if s == 'NEUTRAL'),
            'avg_confidence': np.mean(confidences) if confidences else 0,
            'last_signal': self.prediction_history[-1] if self.prediction_history else None,
        }
        
        return stats
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded model."""
        if not self.is_ready:
            return {'status': 'not_loaded'}
        
        info = {
            'status': 'loaded',
            'model_path': str(self.model_path) if self.model_path else None,
            'n_features': len(self.model.feature_names),
            'feature_names': self.model.feature_names[:10],  # First 10
        }
        
        if hasattr(self.model, 'metadata'):
            info['metadata'] = self.model.metadata
        
        return info
    
    def clear_history(self) -> None:
        """Clear prediction history."""
        self.prediction_history = []
        logger.info("Prediction history cleared")
