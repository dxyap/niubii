"""
Model Monitoring
================
Track model performance and detect drift.
"""

import json
import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MonitoringConfig:
    """Configuration for model monitoring."""

    # Performance tracking
    performance_window: int = 100  # Number of predictions to track
    min_samples_for_metrics: int = 20

    # Drift detection thresholds
    psi_threshold: float = 0.2  # Population Stability Index
    accuracy_drop_threshold: float = 0.1  # Alert if accuracy drops by this much

    # Feature drift
    feature_drift_threshold: float = 0.3  # KS statistic threshold

    # Alerting
    enable_alerts: bool = True
    alert_cooldown_hours: int = 4


class ModelMonitor:
    """
    Monitor model performance over time.

    Tracks:
    - Prediction accuracy vs. actual outcomes
    - Confidence calibration
    - Signal distribution changes
    - Feature drift
    """

    def __init__(self, config: MonitoringConfig | None = None):
        """Initialize monitor with configuration."""
        self.config = config or MonitoringConfig()

        # Performance tracking
        self.predictions: deque = deque(maxlen=self.config.performance_window)
        self.actuals: deque = deque(maxlen=self.config.performance_window)
        self.confidences: deque = deque(maxlen=self.config.performance_window)
        self.timestamps: deque = deque(maxlen=self.config.performance_window)

        # Baseline metrics (from training)
        self.baseline_metrics: dict[str, float] = {}
        self.baseline_feature_stats: dict[str, dict[str, float]] = {}

        # Drift detection
        self.drift_alerts: list[dict[str, Any]] = []
        self.last_alert_time: datetime | None = None

        # Performance history
        self.daily_metrics: list[dict[str, Any]] = []

    def set_baseline(
        self,
        metrics: dict[str, float],
        feature_stats: dict[str, dict[str, float]] | None = None,
    ) -> None:
        """
        Set baseline metrics from training.

        Args:
            metrics: Training metrics (accuracy, f1, etc.)
            feature_stats: Feature statistics (mean, std for each feature)
        """
        self.baseline_metrics = metrics

        if feature_stats:
            self.baseline_feature_stats = feature_stats

        logger.info(f"Set baseline metrics: {metrics}")

    def record_prediction(
        self,
        prediction: int,
        confidence: float,
        timestamp: datetime | None = None,
    ) -> None:
        """
        Record a new prediction.

        Args:
            prediction: Predicted class (0 or 1)
            confidence: Model confidence
            timestamp: Prediction timestamp
        """
        self.predictions.append(prediction)
        self.confidences.append(confidence)
        self.timestamps.append(timestamp or datetime.now())
        self.actuals.append(None)  # Placeholder until actual is known

    def record_actual(
        self,
        actual: int,
        index: int = -1,
    ) -> None:
        """
        Record actual outcome for a prediction.

        Args:
            actual: Actual class (0 or 1)
            index: Index of prediction to update (-1 for most recent)
        """
        if len(self.actuals) > 0:
            actuals_list = list(self.actuals)
            actuals_list[index] = actual
            self.actuals = deque(actuals_list, maxlen=self.config.performance_window)

    def record_batch(
        self,
        predictions: list[int],
        actuals: list[int],
        confidences: list[float],
    ) -> None:
        """
        Record a batch of predictions with outcomes.

        Args:
            predictions: List of predictions
            actuals: List of actual outcomes
            confidences: List of confidence scores
        """
        now = datetime.now()
        for pred, actual, conf in zip(predictions, actuals, confidences):
            self.predictions.append(pred)
            self.actuals.append(actual)
            self.confidences.append(conf)
            self.timestamps.append(now)

    def get_performance_metrics(self) -> dict[str, float]:
        """
        Calculate current performance metrics.

        Returns:
            Dict of performance metrics
        """
        # Filter to predictions with known actuals
        pairs = [
            (p, a) for p, a in zip(self.predictions, self.actuals)
            if a is not None
        ]

        if len(pairs) < self.config.min_samples_for_metrics:
            return {'status': 'insufficient_data', 'n_samples': len(pairs)}

        predictions = [p for p, a in pairs]
        actuals = [a for p, a in pairs]

        # Calculate metrics
        correct = sum(1 for p, a in zip(predictions, actuals) if p == a)
        accuracy = correct / len(pairs)

        # Precision and recall
        tp = sum(1 for p, a in zip(predictions, actuals) if p == 1 and a == 1)
        fp = sum(1 for p, a in zip(predictions, actuals) if p == 1 and a == 0)
        fn = sum(1 for p, a in zip(predictions, actuals) if p == 0 and a == 1)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics = {
            'accuracy': round(accuracy, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'n_samples': len(pairs),
            'timestamp': datetime.now().isoformat(),
        }

        # Compare to baseline
        if self.baseline_metrics:
            baseline_acc = self.baseline_metrics.get('accuracy', 0)
            metrics['accuracy_vs_baseline'] = round(accuracy - baseline_acc, 4)

            # Check for significant drop
            if accuracy < baseline_acc - self.config.accuracy_drop_threshold:
                metrics['alert'] = 'Performance degradation detected'
                self._create_alert('performance_drop', metrics)

        return metrics

    def get_confidence_calibration(self) -> dict[str, Any]:
        """
        Check if confidence scores are well-calibrated.

        Returns:
            Calibration metrics
        """
        pairs = [
            (c, a) for c, a in zip(self.confidences, self.actuals)
            if a is not None
        ]

        if len(pairs) < self.config.min_samples_for_metrics:
            return {'status': 'insufficient_data'}

        confidences = [c for c, a in pairs]
        actuals = [a for c, a in pairs]

        # Bin confidences
        bins = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
        calibration = []

        for low, high in bins:
            bin_mask = [(low <= c < high) for c in confidences]
            bin_confidences = [c for c, m in zip(confidences, bin_mask) if m]
            bin_actuals = [a for a, m in zip(actuals, bin_mask) if m]

            if bin_confidences:
                expected = np.mean(bin_confidences)
                observed = np.mean(bin_actuals)
                calibration.append({
                    'bin': f'{low:.1f}-{high:.1f}',
                    'expected': round(expected, 3),
                    'observed': round(observed, 3),
                    'gap': round(expected - observed, 3),
                    'count': len(bin_confidences),
                })

        # Calculate Expected Calibration Error (ECE)
        total_samples = sum(c['count'] for c in calibration)
        ece = sum(
            c['count'] / total_samples * abs(c['gap'])
            for c in calibration
        ) if total_samples > 0 else 0

        return {
            'calibration_bins': calibration,
            'ece': round(ece, 4),
            'is_well_calibrated': ece < 0.1,
        }

    def get_signal_distribution(self) -> dict[str, Any]:
        """
        Get distribution of signals.

        Returns:
            Signal distribution statistics
        """
        if not self.predictions:
            return {'status': 'no_predictions'}

        predictions = list(self.predictions)

        bullish = sum(1 for p in predictions if p == 1)
        bearish = sum(1 for p in predictions if p == 0)
        total = len(predictions)

        return {
            'total': total,
            'bullish': bullish,
            'bearish': bearish,
            'bullish_pct': round(bullish / total * 100, 1) if total > 0 else 0,
            'bearish_pct': round(bearish / total * 100, 1) if total > 0 else 0,
            'avg_confidence': round(np.mean(list(self.confidences)), 3) if self.confidences else 0,
        }

    def _create_alert(self, alert_type: str, data: dict[str, Any]) -> None:
        """Create a monitoring alert."""
        if not self.config.enable_alerts:
            return

        # Check cooldown
        if self.last_alert_time:
            cooldown = timedelta(hours=self.config.alert_cooldown_hours)
            if datetime.now() - self.last_alert_time < cooldown:
                return

        alert = {
            'type': alert_type,
            'timestamp': datetime.now().isoformat(),
            'data': data,
        }

        self.drift_alerts.append(alert)
        self.last_alert_time = datetime.now()

        logger.warning(f"Monitoring alert: {alert_type} - {data}")

    def get_alerts(self, since: datetime | None = None) -> list[dict[str, Any]]:
        """
        Get monitoring alerts.

        Args:
            since: Only get alerts after this time

        Returns:
            List of alerts
        """
        if since is None:
            return self.drift_alerts

        return [
            a for a in self.drift_alerts
            if datetime.fromisoformat(a['timestamp']) > since
        ]

    def get_summary(self) -> dict[str, Any]:
        """
        Get comprehensive monitoring summary.

        Returns:
            Summary dict
        """
        return {
            'performance': self.get_performance_metrics(),
            'calibration': self.get_confidence_calibration(),
            'signal_distribution': self.get_signal_distribution(),
            'n_alerts': len(self.drift_alerts),
            'recent_alerts': self.drift_alerts[-5:],
        }

    def save_state(self, path: Path) -> None:
        """Save monitor state to disk."""
        state = {
            'baseline_metrics': self.baseline_metrics,
            'daily_metrics': self.daily_metrics,
            'drift_alerts': self.drift_alerts,
        }

        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self, path: Path) -> None:
        """Load monitor state from disk."""
        if path.exists():
            with open(path) as f:
                state = json.load(f)

            self.baseline_metrics = state.get('baseline_metrics', {})
            self.daily_metrics = state.get('daily_metrics', [])
            self.drift_alerts = state.get('drift_alerts', [])


class DriftDetector:
    """
    Detect data and concept drift.

    Uses statistical tests to identify when:
    - Feature distributions change (data drift)
    - Relationship between features and target changes (concept drift)
    """

    def __init__(self, psi_threshold: float = 0.2, ks_threshold: float = 0.3):
        """
        Initialize drift detector.

        Args:
            psi_threshold: PSI threshold for significant drift
            ks_threshold: KS statistic threshold for feature drift
        """
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold

        self.reference_data: pd.DataFrame | None = None
        self.reference_stats: dict[str, dict[str, float]] = {}

    def set_reference(self, data: pd.DataFrame) -> None:
        """
        Set reference data for drift comparison.

        Args:
            data: Training/reference data
        """
        self.reference_data = data.copy()

        # Calculate reference statistics
        for col in data.columns:
            self.reference_stats[col] = {
                'mean': data[col].mean(),
                'std': data[col].std(),
                'min': data[col].min(),
                'max': data[col].max(),
                'q25': data[col].quantile(0.25),
                'q50': data[col].quantile(0.50),
                'q75': data[col].quantile(0.75),
            }

        logger.info(f"Set reference data with {len(data)} samples, {len(data.columns)} features")

    def detect_drift(
        self,
        current_data: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Detect drift between reference and current data.

        Args:
            current_data: Current/production data

        Returns:
            Drift analysis results
        """
        if self.reference_data is None:
            return {'status': 'no_reference_data'}

        results = {
            'overall_drift': False,
            'drifted_features': [],
            'feature_drift_scores': {},
            'timestamp': datetime.now().isoformat(),
        }

        common_cols = list(set(self.reference_data.columns) & set(current_data.columns))

        for col in common_cols:
            ref_col = self.reference_data[col].dropna()
            cur_col = current_data[col].dropna()

            if len(ref_col) == 0 or len(cur_col) == 0:
                continue

            # Calculate KS statistic
            ks_stat = self._ks_test(ref_col, cur_col)

            # Calculate PSI
            psi = self._calculate_psi(ref_col, cur_col)

            results['feature_drift_scores'][col] = {
                'ks_statistic': round(ks_stat, 4),
                'psi': round(psi, 4),
                'is_drifted': ks_stat > self.ks_threshold or psi > self.psi_threshold,
            }

            if results['feature_drift_scores'][col]['is_drifted']:
                results['drifted_features'].append(col)

        # Overall drift if more than 10% of features drifted
        drift_ratio = len(results['drifted_features']) / len(common_cols) if common_cols else 0
        results['drift_ratio'] = round(drift_ratio, 3)
        results['overall_drift'] = drift_ratio > 0.1

        return results

    def _ks_test(self, ref: pd.Series, cur: pd.Series) -> float:
        """Calculate Kolmogorov-Smirnov statistic."""
        try:
            from scipy import stats
            stat, _ = stats.ks_2samp(ref, cur)
            return stat
        except ImportError:
            # Fallback: simple distribution comparison
            ref_sorted = np.sort(ref)
            cur_sorted = np.sort(cur)

            n1 = len(ref_sorted)
            n2 = len(cur_sorted)

            np.arange(1, n1 + 1) / n1
            np.arange(1, n2 + 1) / n2

            # Simple approximation
            ref_mean = np.mean(ref)
            cur_mean = np.mean(cur)
            ref_std = np.std(ref)

            if ref_std > 0:
                return abs(ref_mean - cur_mean) / ref_std
            return 0

    def _calculate_psi(
        self,
        ref: pd.Series,
        cur: pd.Series,
        n_bins: int = 10,
    ) -> float:
        """Calculate Population Stability Index."""
        # Create bins from reference data
        try:
            pd.qcut(ref, n_bins, duplicates='drop')
        except ValueError:
            # Fall back to equal-width bins
            np.linspace(ref.min(), ref.max(), n_bins + 1)

        # Calculate proportions
        ref_counts = pd.cut(ref, bins=n_bins).value_counts(normalize=True)
        cur_counts = pd.cut(cur, bins=n_bins).value_counts(normalize=True)

        # Align indices
        ref_props = ref_counts.reindex(ref_counts.index | cur_counts.index, fill_value=0.0001)
        cur_props = cur_counts.reindex(ref_counts.index | cur_counts.index, fill_value=0.0001)

        # PSI calculation
        psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))

        return abs(psi)

    def get_feature_stats_comparison(
        self,
        current_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compare feature statistics between reference and current.

        Args:
            current_data: Current data

        Returns:
            DataFrame with statistics comparison
        """
        if self.reference_data is None:
            return pd.DataFrame()

        comparisons = []

        for col in self.reference_stats:
            if col not in current_data.columns:
                continue

            ref_stats = self.reference_stats[col]
            cur_mean = current_data[col].mean()
            cur_std = current_data[col].std()

            comparisons.append({
                'feature': col,
                'ref_mean': ref_stats['mean'],
                'cur_mean': cur_mean,
                'mean_change': cur_mean - ref_stats['mean'],
                'mean_change_pct': (cur_mean - ref_stats['mean']) / (ref_stats['mean'] + 1e-10) * 100,
                'ref_std': ref_stats['std'],
                'cur_std': cur_std,
            })

        return pd.DataFrame(comparisons)
