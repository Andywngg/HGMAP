#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import shap
import joblib
from pathlib import Path
import logging
from typing import Tuple, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MicrobiomeModelTrainer:
    """Advanced model trainer for microbiome classification."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.best_model = None
        self.feature_importance = None
        self.shap_values = None
        
    def prepare_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data with scaling and SMOTE."""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply SMOTE for class balancing
        smote = SMOTE(random_state=self.random_state)
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
        
        self.scalers['standard'] = scaler
        return X_resampled, y_resampled
    
    def train_base_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train multiple base models."""
        models = {
            'rf': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            'gb': GradientBoostingClassifier(n_estimators=100, random_state=self.random_state),
            'xgb': xgb.XGBClassifier(n_estimators=100, random_state=self.random_state),
            'lgb': lgb.LGBMClassifier(n_estimators=100, random_state=self.random_state)
        }
        
        results = {}
        for name, model in models.items():
            # Perform 10-fold cross validation
            cv_results = cross_validate(
                model, X, y,
                cv=10,
                scoring={
                    'accuracy': 'accuracy',
                    'precision': 'precision_weighted',
                    'recall': 'recall_weighted',
                    'f1': 'f1_weighted',
                    'roc_auc': 'roc_auc_ovr_weighted'
                },
                return_train_score=True
            )
            
            results[name] = {
                'cv_accuracy': cv_results['test_accuracy'].mean(),
                'cv_precision': cv_results['test_precision'].mean(),
                'cv_recall': cv_results['test_recall'].mean(),
                'cv_f1': cv_results['test_f1'].mean(),
                'cv_roc_auc': cv_results['test_roc_auc'].mean(),
                'model': model
            }
            
            logger.info(f"{name} CV Results:")
            logger.info(f"Accuracy: {cv_results['test_accuracy'].mean():.4f} ± {cv_results['test_accuracy'].std():.4f}")
            logger.info(f"Precision: {cv_results['test_precision'].mean():.4f} ± {cv_results['test_precision'].std():.4f}")
            logger.info(f"Recall: {cv_results['test_recall'].mean():.4f} ± {cv_results['test_recall'].std():.4f}")
            logger.info(f"F1: {cv_results['test_f1'].mean():.4f} ± {cv_results['test_f1'].std():.4f}")
            logger.info(f"ROC AUC: {cv_results['test_roc_auc'].mean():.4f} ± {cv_results['test_roc_auc'].std():.4f}")
        
        return results
    
    def select_best_model(self, results: Dict[str, Any]) -> Tuple[str, Any]:
        """Select the best performing model based on ROC AUC."""
        best_score = 0
        best_name = None
        
        for name, result in results.items():
            if result['cv_roc_auc'] > best_score:
                best_score = result['cv_roc_auc']
                best_name = name
        
        return best_name, results[best_name]['model']
    
    def calculate_feature_importance(self, model: Any, X: pd.DataFrame):
        """Calculate feature importance using SHAP values."""
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        if isinstance(shap_values, list):
            # For multi-class, take mean absolute value across classes
            shap_values = np.abs(np.array(shap_values)).mean(axis=0)
        
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': np.abs(shap_values).mean(axis=0)
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        self.feature_importance = feature_importance
        self.shap_values = shap_values
        
        return feature_importance
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Main training pipeline."""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        # Prepare data
        X_train_prepared, y_train_prepared = self.prepare_data(X_train, y_train)
        X_test_scaled = self.scalers['standard'].transform(X_test)
        
        # Train base models
        results = self.train_base_models(X_train_prepared, y_train_prepared)
        
        # Select best model
        best_name, best_model = self.select_best_model(results)
        logger.info(f"Best model: {best_name}")
        
        # Fit best model on full training data
        best_model.fit(X_train_prepared, y_train_prepared)
        
        # Calculate feature importance
        self.calculate_feature_importance(best_model, pd.DataFrame(X_train_prepared, columns=X.columns))
        
        # Final evaluation on test set
        y_pred = best_model.predict(X_test_scaled)
        y_pred_proba = best_model.predict_proba(X_test_scaled)
        
        test_results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        }
        
        logger.info("Test Set Results:")
        for metric, value in test_results.items():
            logger.info(f"{metric}: {value:.4f}")
        
        self.best_model = best_model
        return test_results
    
    def save_model(self, model_dir: str = 'models'):
        """Save the trained model and associated artifacts."""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.best_model, model_dir / 'best_model.joblib')
        joblib.dump(self.scalers, model_dir / 'scalers.joblib')
        
        if self.feature_importance is not None:
            self.feature_importance.to_csv(model_dir / 'feature_importance.csv', index=False) 