#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)
import shap
import joblib
from pathlib import Path
import logging
from typing import Dict, Any, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation and visualization."""
    
    def __init__(self, model_dir: str = 'models'):
        self.model_dir = Path(model_dir)
        self.load_model()
        
    def load_model(self):
        """Load the trained model and scalers."""
        self.model = joblib.load(self.model_dir / 'best_model.joblib')
        self.scalers = joblib.load(self.model_dir / 'scalers.joblib')
        
        feature_importance = pd.read_csv(self.model_dir / 'feature_importance.csv')
        self.feature_names = feature_importance['feature'].tolist()
    
    def prepare_data(self, X: pd.DataFrame) -> np.ndarray:
        """Prepare data using saved scalers."""
        return self.scalers['standard'].transform(X)
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            labels: list = None, output_dir: str = 'evaluation'):
        """Plot confusion matrix heatmap."""
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues',
                   xticklabels=labels if labels else 'auto',
                   yticklabels=labels if labels else 'auto')
        
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / 'confusion_matrix.png')
        plt.close()
    
    def plot_roc_curves(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                       labels: list = None, output_dir: str = 'evaluation'):
        """Plot ROC curves for each class."""
        plt.figure(figsize=(10, 8))
        
        if y_pred_proba.shape[1] == 2:  # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
            
        else:  # Multi-class
            for i in range(y_pred_proba.shape[1]):
                fpr, tpr, _ = roc_curve(y_true == i, y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                
                label = labels[i] if labels else f'Class {i}'
                plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / 'roc_curves.png')
        plt.close()
    
    def plot_precision_recall_curves(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                   labels: list = None, output_dir: str = 'evaluation'):
        """Plot precision-recall curves for each class."""
        plt.figure(figsize=(10, 8))
        
        if y_pred_proba.shape[1] == 2:  # Binary classification
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba[:, 1])
            pr_auc = auc(recall, precision)
            
            plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.2f})')
            
        else:  # Multi-class
            for i in range(y_pred_proba.shape[1]):
                precision, recall, _ = precision_recall_curve(y_true == i, y_pred_proba[:, i])
                pr_auc = auc(recall, precision)
                
                label = labels[i] if labels else f'Class {i}'
                plt.plot(recall, precision, label=f'{label} (AUC = {pr_auc:.2f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc="lower left")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / 'precision_recall_curves.png')
        plt.close()
    
    def plot_feature_importance(self, X: pd.DataFrame, output_dir: str = 'evaluation'):
        """Plot SHAP feature importance."""
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)
        
        plt.figure(figsize=(12, 8))
        if isinstance(shap_values, list):  # Multi-class
            shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        else:  # Binary
            shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / 'feature_importance.png')
        plt.close()
    
    def evaluate(self, X: pd.DataFrame, y: np.ndarray, labels: list = None) -> Dict[str, Any]:
        """Comprehensive model evaluation."""
        # Prepare data
        X_prepared = self.prepare_data(X)
        
        # Get predictions
        y_pred = self.model.predict(X_prepared)
        y_pred_proba = self.model.predict_proba(X_prepared)
        
        # Generate classification report
        report = classification_report(y, y_pred, target_names=labels if labels else None)
        logger.info("\nClassification Report:\n" + report)
        
        # Plot evaluation metrics
        self.plot_confusion_matrix(y, y_pred, labels)
        self.plot_roc_curves(y, y_pred_proba, labels)
        self.plot_precision_recall_curves(y, y_pred_proba, labels)
        self.plot_feature_importance(X)
        
        # Save results
        results = {
            'classification_report': report,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_dir: str = 'evaluation'):
        """Save evaluation results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save classification report
        with open(output_dir / 'classification_report.txt', 'w') as f:
            f.write(results['classification_report'])
        
        # Save predictions and probabilities
        np.save(output_dir / 'predictions.npy', results['predictions'])
        np.save(output_dir / 'probabilities.npy', results['probabilities']) 