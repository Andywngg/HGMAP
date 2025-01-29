from prometheus_client import Counter, Histogram, Gauge, Summary
import numpy as np
from typing import Dict, Any
import time
import logging

logger = logging.getLogger(__name__)

# Prediction metrics
PREDICTION_COUNTER = Counter(
    'predictions_total',
    'Total number of predictions made',
    ['result']  # 'success' or 'error'
)

PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Time spent processing predictions',
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
)

MODEL_ACCURACY = Gauge(
    'model_accuracy_rolling_window',
    'Model accuracy over the last rolling window'
)

FEATURE_IMPORTANCE = Gauge(
    'feature_importance',
    'Feature importance scores',
    ['feature_name']
)

# System metrics
MEMORY_USAGE = Gauge(
    'memory_usage_bytes',
    'Current memory usage in bytes'
)

CPU_USAGE = Gauge(
    'cpu_usage_percent',
    'Current CPU usage percentage'
)

# Data quality metrics
FEATURE_DRIFT = Gauge(
    'feature_drift_score',
    'Feature drift detection score'
)

PREDICTION_CLASS_RATIO = Gauge(
    'prediction_class_ratio',
    'Ratio of positive to negative predictions'
)

# Error tracking
ERROR_COUNTER = Counter(
    'errors_total',
    'Total number of errors',
    ['error_type']
)

class MetricsTracker:
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.predictions = []
        self.true_labels = []
        self.feature_importance_history = {}
        
    def track_prediction(self, prediction_time: float, success: bool = True):
        """Track prediction latency and count."""
        PREDICTION_LATENCY.observe(prediction_time)
        PREDICTION_COUNTER.labels(
            result='success' if success else 'error'
        ).inc()
    
    def update_accuracy(self, true_label: int, predicted_label: int):
        """Update rolling window accuracy."""
        self.predictions.append(predicted_label)
        self.true_labels.append(true_label)
        
        # Keep only the last window_size predictions
        if len(self.predictions) > self.window_size:
            self.predictions.pop(0)
            self.true_labels.pop(0)
        
        # Calculate accuracy
        if self.predictions:
            accuracy = np.mean(
                np.array(self.predictions) == np.array(self.true_labels)
            )
            MODEL_ACCURACY.set(accuracy)
    
    def track_feature_importance(self, importance_dict: Dict[str, float]):
        """Track feature importance scores."""
        for feature, score in importance_dict.items():
            FEATURE_IMPORTANCE.labels(feature_name=feature).set(score)
            
            # Track history for drift detection
            if feature not in self.feature_importance_history:
                self.feature_importance_history[feature] = []
            
            self.feature_importance_history[feature].append(score)
            if len(self.feature_importance_history[feature]) > self.window_size:
                self.feature_importance_history[feature].pop(0)
    
    def track_system_metrics(self, memory_usage: float, cpu_usage: float):
        """Track system resource usage."""
        MEMORY_USAGE.set(memory_usage)
        CPU_USAGE.set(cpu_usage)
    
    def track_error(self, error_type: str):
        """Track errors by type."""
        ERROR_COUNTER.labels(error_type=error_type).inc()
    
    def calculate_feature_drift(self):
        """Calculate feature importance drift score."""
        if not self.feature_importance_history:
            return 0.0
        
        drift_scores = []
        for feature, history in self.feature_importance_history.items():
            if len(history) > 1:
                # Calculate coefficient of variation
                drift = np.std(history) / np.mean(history) if np.mean(history) != 0 else 0
                drift_scores.append(drift)
        
        avg_drift = np.mean(drift_scores) if drift_scores else 0
        FEATURE_DRIFT.set(avg_drift)
        return avg_drift
    
    def update_prediction_ratio(self):
        """Update the ratio of positive to negative predictions."""
        if self.predictions:
            ratio = np.mean(self.predictions)
            PREDICTION_CLASS_RATIO.set(ratio)
    
    def track_prediction_with_metadata(
        self,
        start_time: float,
        true_label: int,
        predicted_label: int,
        feature_importance: Dict[str, float],
        system_metrics: Dict[str, float]
    ):
        """Track all metrics for a single prediction."""
        try:
            # Calculate prediction time
            prediction_time = time.time() - start_time
            
            # Track basic metrics
            self.track_prediction(prediction_time)
            self.update_accuracy(true_label, predicted_label)
            
            # Track feature importance
            self.track_feature_importance(feature_importance)
            
            # Track system metrics
            self.track_system_metrics(
                system_metrics.get('memory_usage', 0),
                system_metrics.get('cpu_usage', 0)
            )
            
            # Update prediction ratio
            self.update_prediction_ratio()
            
            # Calculate feature drift
            self.calculate_feature_drift()
            
        except Exception as e:
            logger.error(f"Error tracking metrics: {str(e)}")
            self.track_error('metrics_tracking_error')
            raise

def get_current_metrics() -> Dict[str, Any]:
    """Get current values of all metrics."""
    return {
        'model_accuracy': float(MODEL_ACCURACY._value.get()),
        'prediction_count': float(PREDICTION_COUNTER._value.get()),
        'average_latency': float(PREDICTION_LATENCY._sum.get() / 
                               max(PREDICTION_COUNTER._value.get(), 1)),
        'feature_drift': float(FEATURE_DRIFT._value.get()),
        'memory_usage': float(MEMORY_USAGE._value.get()),
        'cpu_usage': float(CPU_USAGE._value.get()),
        'error_count': float(ERROR_COUNTER._value.get())
    } 