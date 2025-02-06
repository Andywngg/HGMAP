#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for model evaluation metrics and visualization.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union, Optional
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve
)

class ModelEvaluator:
    """Class for evaluating model performance and generating visualizations."""
    
    def __init__(
        self,
        metrics: List[str],
        threshold: float = 0.5,
        output_dir: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the ModelEvaluator.
        
        Args:
            metrics: List of metric names to compute
            threshold: Classification threshold for binary predictions
            output_dir: Directory to save evaluation results
        """
        self.metrics = metrics
        self.threshold = threshold
        self.output_dir = Path(output_dir) if output_dir else None
        
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate_and_save(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        model_name: str
    ) -> Dict[str, float]:
        """
        Evaluate model performance and save results.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities
            model_name: Name of the model being evaluated
        
        Returns:
            Dictionary of computed metrics
        """
        metrics_dict = {}
        
        # Calculate metrics
        for metric in self.metrics:
            if metric == "accuracy":
                metrics_dict[metric] = accuracy_score(y_true, y_pred)
            elif metric == "precision":
                metrics_dict[metric] = precision_score(y_true, y_pred, average='weighted')
            elif metric == "recall":
                metrics_dict[metric] = recall_score(y_true, y_pred, average='weighted')
            elif metric == "f1":
                metrics_dict[metric] = f1_score(y_true, y_pred, average='weighted')
            elif metric == "roc_auc":
                metrics_dict[metric] = roc_auc_score(y_true, y_prob)
            elif metric == "average_precision":
                metrics_dict[metric] = average_precision_score(y_true, y_prob)
        
        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Generate classification report
        class_report = classification_report(y_true, y_pred, output_dict=True)
        
        if self.output_dir:
            # Save metrics
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metrics_file = self.output_dir / f"{model_name}_metrics_{timestamp}.json"
            
            with open(metrics_file, 'w') as f:
                json.dump({
                    'metrics': metrics_dict,
                    'classification_report': class_report
                }, f, indent=4)
            
            # Plot and save confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive']
            )
            plt.title(f'Confusion Matrix - {model_name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(self.output_dir / f"{model_name}_confusion_matrix_{timestamp}.png")
            plt.close()
            
            # Plot and save ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {metrics_dict["roc_auc"]:.3f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_name}')
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(self.output_dir / f"{model_name}_roc_curve_{timestamp}.png")
            plt.close()
            
            # Plot and save Precision-Recall curve
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, label=f'PR curve (AP = {metrics_dict["average_precision"]:.3f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - {model_name}')
            plt.legend(loc="lower left")
            plt.tight_layout()
            plt.savefig(self.output_dir / f"{model_name}_pr_curve_{timestamp}.png")
            plt.close()
        
        return metrics_dict
    
    def compare_models(
        self,
        results: List[Dict[str, Union[str, Dict[str, float]]]],
        output_file: Optional[Union[str, Path]] = None
    ) -> pd.DataFrame:
        """
        Compare performance metrics across multiple models.
        
        Args:
            results: List of dictionaries containing model names and their metrics
            output_file: Path to save comparison results
        
        Returns:
            DataFrame with model comparison
        """
        comparison_df = pd.DataFrame([
            {
                'model': result['model_name'],
                **result['metrics']
            }
            for result in results
        ])
        
        if output_file:
            comparison_df.to_csv(output_file, index=False)
            
            # Plot comparison
            plt.figure(figsize=(12, 6))
            comparison_df.set_index('model').plot(kind='bar')
            plt.title('Model Performance Comparison')
            plt.xlabel('Model')
            plt.ylabel('Score')
            plt.xticks(rotation=45)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(str(Path(output_file).with_suffix('.png')))
            plt.close()
        
        return comparison_df 