#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import shap
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HyperEnsemble(BaseEstimator, ClassifierMixin):
    """Advanced ensemble model with stacking and automated hyperparameter tuning."""
    
    def __init__(
        self,
        n_folds: int = 5,
        use_probabilities: bool = True,
        random_state: int = 42,
        base_models_config: Optional[Dict] = None
    ):
        self.n_folds = n_folds
        self.use_probabilities = use_probabilities
        self.random_state = random_state
        self.base_models_config = base_models_config or self._get_default_models()
        self.base_models = {}
        self.meta_model = None
        self.scaler = StandardScaler()
        self.feature_importance_ = None
        self.shap_values_ = None
        
    def _get_default_models(self) -> Dict:
        """Get default configuration for base models."""
        return {
            'rf': {
                'model': RandomForestClassifier(
                    n_estimators=1000,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    max_features='sqrt',
                    bootstrap=True,
                    random_state=self.random_state,
                    n_jobs=-1
                ),
                'weight': 1.0
            },
            'xgb': {
                'model': xgb.XGBClassifier(
                    n_estimators=1000,
                    max_depth=6,
                    learning_rate=0.01,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=self.random_state,
                    n_jobs=-1
                ),
                'weight': 1.0
            },
            'lgb': {
                'model': lgb.LGBMClassifier(
                    n_estimators=1000,
                    num_leaves=31,
                    learning_rate=0.01,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=self.random_state,
                    n_jobs=-1
                ),
                'weight': 1.0
            },
            'gb': {
                'model': GradientBoostingClassifier(
                    n_estimators=1000,
                    max_depth=6,
                    learning_rate=0.01,
                    subsample=0.8,
                    random_state=self.random_state
                ),
                'weight': 1.0
            }
        }
    
    def _generate_meta_features(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        is_train: bool = True
    ) -> np.ndarray:
        """Generate meta-features through cross-validation predictions."""
        meta_features = np.zeros((X.shape[0], len(self.base_models_config) * (2 if self.use_probabilities else 1)))
        
        if is_train:
            # Use cross-validation to get out-of-fold predictions
            kf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            
            for train_idx, val_idx in kf.split(X, y):
                X_train_fold = X[train_idx]
                y_train_fold = y[train_idx]
                X_val_fold = X[val_idx]
                
                col_idx = 0
                for name, model_config in self.base_models_config.items():
                    model = model_config['model']
                    model.fit(X_train_fold, y_train_fold)
                    
                    if self.use_probabilities:
                        proba = model.predict_proba(X_val_fold)
                        meta_features[val_idx, col_idx:col_idx + 2] = proba
                        col_idx += 2
                    else:
                        pred = model.predict(X_val_fold)
                        meta_features[val_idx, col_idx] = pred
                        col_idx += 1
        else:
            # For test data, use models as is
            col_idx = 0
            for name, model in self.base_models.items():
                if self.use_probabilities:
                    proba = model.predict_proba(X)
                    meta_features[:, col_idx:col_idx + 2] = proba
                    col_idx += 2
                else:
                    pred = model.predict(X)
                    meta_features[:, col_idx] = pred
                    col_idx += 1
        
        return meta_features
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'HyperEnsemble':
        """Fit the stacking ensemble."""
        logger.info("Fitting HyperEnsemble...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply SMOTE for class balancing
        smote = SMOTE(random_state=self.random_state)
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
        
        # Generate meta-features
        meta_features = self._generate_meta_features(X_resampled, y_resampled, is_train=True)
        
        # Fit base models on full training data
        for name, model_config in self.base_models_config.items():
            logger.info(f"Fitting {name} model...")
            model = model_config['model']
            model.fit(X_resampled, y_resampled)
            self.base_models[name] = model
        
        # Fit meta-model
        logger.info("Fitting meta-model...")
        self.meta_model = lgb.LGBMClassifier(
            n_estimators=200,
            num_leaves=31,
            learning_rate=0.01,
            random_state=self.random_state
        )
        self.meta_model.fit(meta_features, y_resampled)
        
        # Calculate feature importance and SHAP values
        self._calculate_feature_importance(X_scaled)
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for X."""
        X_scaled = self.scaler.transform(X)
        meta_features = self._generate_meta_features(X_scaled, is_train=False)
        return self.meta_model.predict_proba(meta_features)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for X."""
        return np.argmax(self.predict_proba(X), axis=1)
    
    def _calculate_feature_importance(self, X: np.ndarray) -> None:
        """Calculate feature importance using SHAP values."""
        logger.info("Calculating SHAP values...")
        
        # Use the Random Forest model for SHAP calculations
        explainer = shap.TreeExplainer(self.base_models['rf'])
        self.shap_values_ = explainer.shap_values(X)
        
        if isinstance(self.shap_values_, list):
            # For binary classification, take the second class
            self.feature_importance_ = np.abs(self.shap_values_[1]).mean(0)
        else:
            self.feature_importance_ = np.abs(self.shap_values_).mean(0)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate the model using multiple metrics."""
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_prob)
        }
        
        return metrics
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        scoring: str = 'roc_auc'
    ) -> Dict[str, float]:
        """Perform cross-validation."""
        cv_scores = cross_val_score(
            self,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )
        
        return {
            'mean_cv_score': cv_scores.mean(),
            'std_cv_score': cv_scores.std(),
            'cv_scores': cv_scores
        }
    
    def save(self, model_dir: str) -> None:
        """Save the trained model and its components."""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self, model_dir / 'ensemble_model.joblib')
        joblib.dump(self.scaler, model_dir / 'scaler.joblib')
        
        if self.feature_importance_ is not None:
            np.save(model_dir / 'feature_importance.npy', self.feature_importance_)
        
        if self.shap_values_ is not None:
            np.save(model_dir / 'shap_values.npy', self.shap_values_)
    
    @classmethod
    def load(cls, model_dir: str) -> 'HyperEnsemble':
        """Load a trained model and its components."""
        model_dir = Path(model_dir)
        model = joblib.load(model_dir / 'ensemble_model.joblib')
        
        if (model_dir / 'feature_importance.npy').exists():
            model.feature_importance_ = np.load(model_dir / 'feature_importance.npy')
        
        if (model_dir / 'shap_values.npy').exists():
            model.shap_values_ = np.load(model_dir / 'shap_values.npy')
        
        return model

if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    
    # Generate sample data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    # Initialize and train model
    model = HyperEnsemble()
    model.fit(X, y)
    
    # Evaluate
    metrics = model.evaluate(X, y)
    print("\nModel Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Cross-validate
    cv_results = model.cross_validate(X, y)
    print(f"\nCross-validation ROC-AUC: {cv_results['mean_cv_score']:.4f} ± {cv_results['std_cv_score']:.4f}") 