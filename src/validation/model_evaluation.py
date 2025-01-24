import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Tuple

from src.data.processor import MicrobiomeDataProcessor
from src.model.advanced_ensemble import HyperEnsemble

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Evaluate model performance on microbiome data"""
    
    SUPPORTED_DISEASES = {
        'ibd': 'Inflammatory Bowel Disease',
        'cd': "Crohn's Disease",
        'uc': 'Ulcerative Colitis',
        't2d': 'Type 2 Diabetes',
        'obesity': 'Obesity',
        'crc': 'Colorectal Cancer'
    }
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.processor = MicrobiomeDataProcessor()
        self.results_dir = self.data_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def evaluate_disease_prediction(
        self,
        disease: str,
        test_size: float = 0.2,
        n_splits: int = 5
    ) -> Dict:
        """Evaluate model performance for specific disease prediction"""
        if disease not in self.SUPPORTED_DISEASES:
            raise ValueError(f"Disease {disease} not supported. Supported diseases: {list(self.SUPPORTED_DISEASES.keys())}")
        
        logger.info(f"Evaluating model for {self.SUPPORTED_DISEASES[disease]}")
        
        # Load disease-specific data
        abundances, metadata = self._load_disease_data(disease)
        
        # Prepare data
        X = abundances.values
        y = metadata[f'{disease}_status'].values
        
        # Initialize metrics storage
        metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'auc_roc': [],
            'feature_importance': None
        }
        
        # Cross-validation
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            logger.info(f"Processing fold {fold + 1}/{n_splits}")
            
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train model
            model = self._train_model(X_train, y_train)
            
            # Get predictions
            y_pred = model.predict_proba(X_test)[:, 1]
            y_pred_binary = (y_pred > 0.5).astype(int)
            
            # Calculate metrics
            metrics['accuracy'].append(accuracy_score(y_test, y_pred_binary))
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred_binary, average='binary'
            )
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['f1'].append(f1)
            metrics['auc_roc'].append(roc_auc_score(y_test, y_pred))
            
            # Get feature importance for last fold
            if fold == n_splits - 1:
                metrics['feature_importance'] = model.get_feature_importance(X_test)
        
        # Calculate average metrics
        for key in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']:
            metrics[f'avg_{key}'] = np.mean(metrics[key])
            metrics[f'std_{key}'] = np.std(metrics[key])
        
        # Save results
        self._save_results(metrics, disease)
        
        return metrics
    
    def _train_model(self, X: np.ndarray, y: np.ndarray) -> HyperEnsemble:
        """Train the model"""
        from torch import nn
        
        # Define base models
        base_models = [
            nn.Sequential(
                nn.Linear(X.shape[1], 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
                nn.Softmax(dim=1)
            ),
            nn.Sequential(
                nn.Linear(X.shape[1], 256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 2),
                nn.Softmax(dim=1)
            )
        ]
        
        # Initialize and train ensemble
        model = HyperEnsemble(base_models=base_models)
        model.fit(X, y)
        
        return model
    
    def _load_disease_data(
        self,
        disease: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load disease-specific data"""
        # Check cache first
        cached_data = self.processor.load_processed_data(f"{disease}_data")
        if cached_data is not None:
            return cached_data
        
        # Load and combine relevant datasets
        datasets = []
        
        # Load American Gut Project data
        agp_data = self.processor.load_american_gut(
            self.data_dir / "american_gut"
        )
        datasets.append(agp_data)
        
        # Load HMP data
        hmp_data = self.processor.load_hmp(
            self.data_dir / "hmp"
        )
        datasets.append(hmp_data)
        
        # Combine datasets
        abundances, metadata = self.processor.combine_datasets(datasets)
        
        # Cache the processed data
        self.processor.save_processed_data_to_cache(
            abundances, metadata, f"{disease}_data"
        )
        
        return abundances, metadata
    
    def _save_results(self, metrics: Dict, disease: str) -> None:
        """Save evaluation results"""
        results_file = self.results_dir / f"{disease}_evaluation.txt"
        
        with open(results_file, 'w') as f:
            f.write(f"Evaluation Results for {self.SUPPORTED_DISEASES[disease]}\n")
            f.write("=" * 50 + "\n\n")
            
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']:
                f.write(f"Average {metric}: {metrics[f'avg_{metric}']:.3f} ± {metrics[f'std_{metric}']:.3f}\n")
        
        # Plot feature importance
        if metrics['feature_importance'] is not None:
            self._plot_feature_importance(metrics['feature_importance'], disease)
    
    def _plot_feature_importance(
        self,
        importance: Dict[str, np.ndarray],
        disease: str
    ) -> None:
        """Plot feature importance"""
        plt.figure(figsize=(12, 6))
        
        # Average importance across models
        avg_importance = np.mean([imp for imp in importance.values()], axis=0)
        feature_names = [f"Feature {i}" for i in range(len(avg_importance))]
        
        # Sort by importance
        sorted_idx = np.argsort(avg_importance)
        pos = np.arange(sorted_idx.shape[0]) + .5
        
        plt.barh(pos, avg_importance[sorted_idx])
        plt.yticks(pos, np.array(feature_names)[sorted_idx])
        plt.xlabel('Mean SHAP Value')
        plt.title(f'Feature Importance for {self.SUPPORTED_DISEASES[disease]}')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / f"{disease}_feature_importance.png")
        plt.close()

if __name__ == '__main__':
    # Example usage
    evaluator = ModelEvaluator()
    
    # Evaluate for each supported disease
    for disease in evaluator.SUPPORTED_DISEASES:
        try:
            metrics = evaluator.evaluate_disease_prediction(disease)
            logger.info(f"Evaluation completed for {disease}")
            logger.info(f"Average accuracy: {metrics['avg_accuracy']:.3f}")
            logger.info(f"Average AUC-ROC: {metrics['avg_auc_roc']:.3f}")
        except Exception as e:
            logger.error(f"Error evaluating {disease}: {e}") 