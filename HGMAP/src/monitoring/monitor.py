import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import roc_auc_score, precision_recall_curve
from scipy import stats
import logging
import json
import time
from datetime import datetime

class ModelMonitor:
    def __init__(
        self,
        model,
        feature_names: List[str],
        monitoring_config: Optional[Dict] = None
    ):
        self.model = model
        self.feature_names = feature_names
        self.monitoring_config = monitoring_config or {}
        self.baseline_stats = {}
        self.current_stats = {}
        self.alerts = []
        
        # Setup logging
        logging.basicConfig(
            filename='model_monitoring.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def compute_distribution_stats(
        self,
        data: np.ndarray
    ) -> Dict:
        """Compute distribution statistics for features"""
        stats_dict = {}
        
        for i, feature in enumerate(self.feature_names):
            feature_data = data[:, i]
            stats_dict[feature] = {
                'mean': float(np.mean(feature_data)),
                'std': float(np.std(feature_data)),
                'median': float(np.median(feature_data)),
                'q1': float(np.percentile(feature_data, 25)),
                'q3': float(np.percentile(feature_data, 75)),
                'ks_test': None  # Will be computed when comparing
            }
            
        return stats_dict
    
    def set_baseline(
        self,
        baseline_data: np.ndarray,
        baseline_predictions: Optional[np.ndarray] = None
    ):
        """Set baseline statistics for monitoring"""
        self.baseline_stats = {
            'feature_stats': self.compute_distribution_stats(baseline_data),
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(baseline_data)
        }
        
        if baseline_predictions is not None:
            self.baseline_stats['prediction_stats'] = {
                'mean': float(np.mean(baseline_predictions)),
                'std': float(np.std(baseline_predictions))
            }
        
        # Save baseline stats
        with open('baseline_stats.json', 'w') as f:
            json.dump(self.baseline_stats, f, indent=2)
    
    def check_data_drift(
        self,
        current_data: np.ndarray,
        threshold: float = 0.05
    ) -> Dict:
        """Check for data drift using statistical tests"""
        drift_results = {}
        
        for i, feature in enumerate(self.feature_names):
            baseline_data = current_data[:, i]
            current_data_feature = current_data[:, i]
            
            # Perform Kolmogorov-Smirnov test
            ks_statistic, p_value = stats.ks_2samp(
                baseline_data,
                current_data_feature
            )
            
            drift_detected = p_value < threshold
            if drift_detected:
                self.logger.warning(
                    f"Data drift detected for feature {feature}"
                )
                self.alerts.append({
                    'timestamp': datetime.now().isoformat(),
                    'type': 'data_drift',
                    'feature': feature,
                    'p_value': float(p_value)
                })
            
            drift_results[feature] = {
                'ks_statistic': float(ks_statistic),
                'p_value': float(p_value),
                'drift_detected': drift_detected
            }
            
        return drift_results
    
    def check_performance_drift(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float = 0.1
    ) -> Dict:
        """Monitor model performance drift"""
        current_auc = roc_auc_score(y_true, y_pred)
        baseline_auc = self.baseline_stats.get('performance_metrics', {}).get('auc')
        
        if baseline_auc is not None:
            performance_drift = abs(current_auc - baseline_auc) > threshold
            if performance_drift:
                self.logger.warning("Performance drift detected")
                self.alerts.append({
                    'timestamp': datetime.now().isoformat(),
                    'type': 'performance_drift',
                    'current_auc': float(current_auc),
                    'baseline_auc': float(baseline_auc)
                })
        
        return {
            'current_auc': float(current_auc),
            'performance_drift': performance_drift if baseline_auc is not None else None
        }
    
    def monitor_predictions(
        self,
        predictions: np.ndarray,
        threshold: float = 0.1
    ) -> Dict:
        """Monitor prediction distribution changes"""
        current_mean = np.mean(predictions)
        current_std = np.std(predictions)
        
        baseline_mean = self.baseline_stats.get('prediction_stats', {}).get('mean')
        baseline_std = self.baseline_stats.get('prediction_stats', {}).get('std')
        
        if baseline_mean is not None and baseline_std is not None:
            mean_shift = abs(current_mean - baseline_mean) > threshold * baseline_std
            if mean_shift:
                self.logger.warning("Significant shift in prediction distribution")
                self.alerts.append({
                    'timestamp': datetime.now().isoformat(),
                    'type': 'prediction_drift',
                    'current_mean': float(current_mean),
                    'baseline_mean': float(baseline_mean)
                })
        
        return {
            'current_mean': float(current_mean),
            'current_std': float(current_std),
            'mean_shift': mean_shift if baseline_mean is not None else None
        }
    
    def generate_monitoring_report(self) -> Dict:
        """Generate comprehensive monitoring report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'alerts': self.alerts,
            'drift_summary': {
                'n_features_drifted': len([
                    alert for alert in self.alerts 
                    if alert['type'] == 'data_drift'
                ]),
                'performance_alerts': len([
                    alert for alert in self.alerts 
                    if alert['type'] == 'performance_drift'
                ])
            },
            'current_stats': self.current_stats
        }
        
        # Save report
        with open(f'monitoring_report_{int(time.time())}.json', 'w') as f:
            json.dump(report, f, indent=2)
            
        return report 