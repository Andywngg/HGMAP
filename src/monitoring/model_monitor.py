import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class ModelMonitor:
    """Monitor model performance and data drift."""
    
    def __init__(
        self,
        model_dir: str = "data/models",
        monitoring_dir: str = "data/monitoring",
        reference_data_path: Optional[str] = None
    ):
        self.model_dir = Path(model_dir)
        self.monitoring_dir = Path(monitoring_dir)
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)
        
        # Load reference data if provided
        if reference_data_path:
            self.reference_data = pd.read_csv(reference_data_path)
        else:
            self.reference_data = None
        
        # Initialize monitoring metrics
        self.performance_history = []
        self.data_drift_history = []
        
        # Load existing history if available
        self._load_history()
    
    def _load_history(self):
        """Load existing monitoring history."""
        history_file = self.monitoring_dir / "monitoring_history.json"
        if history_file.exists():
            with open(history_file, 'r') as f:
                history = json.load(f)
                self.performance_history = history.get('performance', [])
                self.data_drift_history = history.get('data_drift', [])
    
    def _save_history(self):
        """Save monitoring history."""
        history = {
            'performance': self.performance_history,
            'data_drift': self.data_drift_history
        }
        
        with open(self.monitoring_dir / "monitoring_history.json", 'w') as f:
            json.dump(history, f, indent=4)
    
    def calculate_performance_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict[str, float]:
        """Calculate model performance metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_prob),
            'timestamp': datetime.now().isoformat()
        }
        
        return metrics
    
    def detect_data_drift(
        self,
        current_data: pd.DataFrame,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """Detect data drift using Kolmogorov-Smirnov test."""
        if self.reference_data is None:
            raise ValueError("Reference data not provided")
        
        if feature_names is None:
            feature_names = current_data.columns
        
        drift_metrics = {}
        for feature in feature_names:
            if feature in self.reference_data.columns:
                # Perform KS test
                statistic, p_value = ks_2samp(
                    self.reference_data[feature],
                    current_data[feature]
                )
                
                drift_metrics[feature] = {
                    'ks_statistic': float(statistic),
                    'p_value': float(p_value),
                    'is_drift': p_value < 0.05
                }
        
        return drift_metrics
    
    def monitor_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        current_data: Optional[pd.DataFrame] = None
    ):
        """Monitor predictions and update history."""
        # Calculate performance metrics
        metrics = self.calculate_performance_metrics(y_true, y_pred, y_prob)
        self.performance_history.append(metrics)
        
        # Detect data drift if current data is provided
        if current_data is not None and self.reference_data is not None:
            drift_metrics = self.detect_data_drift(current_data)
            self.data_drift_history.append({
                'timestamp': datetime.now().isoformat(),
                'metrics': drift_metrics
            })
        
        # Save updated history
        self._save_history()
        
        # Generate monitoring report
        self.generate_monitoring_report()
    
    def generate_monitoring_report(self):
        """Generate monitoring report with visualizations."""
        if not self.performance_history:
            return
        
        # Create performance trend plots
        metrics_df = pd.DataFrame(self.performance_history)
        metrics_df['timestamp'] = pd.to_datetime(metrics_df['timestamp'])
        
        plt.figure(figsize=(12, 6))
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            plt.plot(metrics_df['timestamp'], metrics_df[metric], label=metric)
        
        plt.title('Model Performance Trends')
        plt.xlabel('Time')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.monitoring_dir / "performance_trends.png")
        plt.close()
        
        # Generate data drift heatmap if available
        if self.data_drift_history:
            latest_drift = self.data_drift_history[-1]['metrics']
            drift_data = {
                feature: metrics['ks_statistic']
                for feature, metrics in latest_drift.items()
            }
            
            plt.figure(figsize=(10, 6))
            sns.heatmap(
                pd.DataFrame([drift_data]).T,
                cmap='YlOrRd',
                annot=True,
                fmt='.3f'
            )
            plt.title('Feature Drift Analysis (KS Statistic)')
            plt.tight_layout()
            plt.savefig(self.monitoring_dir / "data_drift_heatmap.png")
            plt.close()
        
        # Save summary report
        latest_metrics = self.performance_history[-1]
        summary = {
            'latest_performance': {
                k: v for k, v in latest_metrics.items() if k != 'timestamp'
            },
            'performance_trend': {
                metric: metrics_df[metric].tolist()[-5:]  # Last 5 measurements
                for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            },
            'data_drift_summary': {
                feature: metrics['is_drift']
                for feature, metrics in latest_drift.items()
            } if self.data_drift_history else {}
        }
        
        with open(self.monitoring_dir / "monitoring_summary.json", 'w') as f:
            json.dump(summary, f, indent=4)
        
        logger.info(f"Generated monitoring report in {self.monitoring_dir}")
        
        # Check for performance degradation
        if latest_metrics['accuracy'] < 0.9:
            logger.warning("Model performance has dropped below 90% accuracy threshold")
        
        return summary 