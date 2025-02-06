import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import logging
import shap

class HyperEnsemble(BaseEstimator, ClassifierMixin):
    """Advanced ensemble model with stacking and automated hyperparameter tuning."""
    
    def __init__(
        self,
        n_folds: int = 5,
        random_state: int = 42,
        use_probabilities: bool = True,
        base_models_config: Optional[Dict] = None
    ):
        self.n_folds = n_folds
        self.random_state = random_state
        self.use_probabilities = use_probabilities
        self.base_models_config = base_models_config or self._get_default_models()
        self.base_models = {}
        self.meta_model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.shap_values = None
        
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
                'model': XGBClassifier(
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
                'model': LGBMClassifier(
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
        y: np.ndarray,
        models: Dict,
        train: bool = True
    ) -> np.ndarray:
        """Generate meta-features through cross-validation predictions."""
        meta_features = np.zeros((X.shape[0], len(models) * (2 if self.use_probabilities else 1)))
        
        if train:
            # Use cross-validation to get out-of-fold predictions
            kf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            
            for train_idx, val_idx in kf.split(X, y):
                X_train_fold = X[train_idx]
                y_train_fold = y[train_idx]
                X_val_fold = X[val_idx]
                
                col_idx = 0
                for name, model_config in models.items():
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
            for name, model_config in models.items():
                if self.use_probabilities:
                    proba = model_config['model'].predict_proba(X)
                    meta_features[:, col_idx:col_idx + 2] = proba
                    col_idx += 2
                else:
                    pred = model_config['model'].predict(X)
                    meta_features[:, col_idx] = pred
                    col_idx += 1
        
        return meta_features
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'HyperEnsemble':
        """Fit the stacking ensemble."""
        logging.info("Fitting HyperEnsemble...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Generate meta-features
        meta_features = self._generate_meta_features(X_scaled, y, self.base_models_config, train=True)
        
        # Fit base models on full training data
        for name, model_config in self.base_models_config.items():
            logging.info(f"Fitting {name} model...")
            model = model_config['model']
            model.fit(X_scaled, y)
            self.base_models[name] = model
        
        # Fit meta-model
        logging.info("Fitting meta-model...")
        self.meta_model = XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.01,
            random_state=self.random_state
        )
        self.meta_model.fit(meta_features, y)
        
        # Calculate feature importance
        self._calculate_feature_importance(X_scaled, y)
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for X."""
        X_scaled = self.scaler.transform(X)
        meta_features = self._generate_meta_features(X_scaled, None, self.base_models, train=False)
        return self.meta_model.predict_proba(meta_features)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for X."""
        return np.argmax(self.predict_proba(X), axis=1)
    
    def _calculate_feature_importance(self, X: np.ndarray, y: np.ndarray) -> None:
        """Calculate feature importance using SHAP values."""
        logging.info("Calculating SHAP values...")
        
        # Use the Random Forest model for SHAP calculations
        explainer = shap.TreeExplainer(self.base_models['rf'])
        self.shap_values = explainer.shap_values(X)
        
        if isinstance(self.shap_values, list):
            # For binary classification, take the second class
            self.feature_importance = np.abs(self.shap_values[1]).mean(0)
        else:
            self.feature_importance = np.abs(self.shap_values).mean(0)
    
    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Get feature importance based on SHAP values."""
        if self.feature_importance is None:
            raise ValueError("Model must be fitted before getting feature importance")
        
        importance_df = pd.DataFrame({
            'feature': feature_names if feature_names else [f'feature_{i}' for i in range(len(self.feature_importance))],
            'importance': self.feature_importance
        })
        return importance_df.sort_values('importance', ascending=False)
    
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
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Perform cross-validation."""
        cv_scores = cross_val_score(
            self,
            X,
            y,
            cv=self.n_folds,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        return {
            'mean_cv_score': cv_scores.mean(),
            'std_cv_score': cv_scores.std(),
            'cv_scores': cv_scores
        } 