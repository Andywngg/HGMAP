import numpy as np
import torch
from typing import List, Dict
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

class HyperEnsemble:
    def __init__(
        self,
        base_models: List[torch.nn.Module],
        n_folds: int = 5,
        meta_learner_type: str = 'weighted'
    ):
        self.base_models = base_models
        self.n_folds = n_folds
        self.meta_learner_type = meta_learner_type
        
        # Initialize boosting models
        self.boosting_models = {
            'xgb': XGBClassifier(
                n_estimators=1000,
                learning_rate=0.01,
                max_depth=7,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method='gpu_hist'  # GPU acceleration
            ),
            'lgb': LGBMClassifier(
                n_estimators=1000,
                learning_rate=0.01,
                num_leaves=31,
                feature_fraction=0.8,
                device='gpu'  # GPU acceleration
            ),
            'catboost': CatBoostClassifier(
                iterations=1000,
                learning_rate=0.01,
                depth=7,
                task_type='GPU'  # GPU acceleration
            )
        }
        
        # Dynamic weight optimization
        self.model_weights = None
        self.meta_features = None
        
    def _optimize_weights(self, val_preds: List[np.ndarray], y_val: np.ndarray):
        """Optimize ensemble weights using differential evolution"""
        from scipy.optimize import differential_evolution
        
        def objective(weights):
            # Normalize weights
            weights = weights / np.sum(weights)
            # Compute weighted average
            weighted_pred = np.zeros_like(val_preds[0])
            for w, p in zip(weights, val_preds):
                weighted_pred += w * p
            return -roc_auc_score(y_val, weighted_pred)
        
        bounds = [(0, 1)] * len(self.base_models)
        result = differential_evolution(
            objective,
            bounds,
            mutation=(0.5, 1),
            recombination=0.7,
            popsize=20
        )
        return result.x / np.sum(result.x)
    
    def _generate_meta_features(self, X: np.ndarray, y: np.ndarray = None):
        """Generate meta-features using k-fold predictions"""
        meta_features = np.zeros((X.shape[0], len(self.base_models) * 2))
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True)
        
        for i, model in enumerate(self.base_models):
            fold_preds = []
            for train_idx, val_idx in skf.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train = y[train_idx]
                
                # Train model on fold
                model.fit(X_train, y_train)
                
                # Generate predictions
                val_pred = model.predict_proba(X_val)
                fold_preds.append((val_idx, val_pred))
            
            # Combine fold predictions
            all_preds = np.zeros((X.shape[0], 2))
            for idx, pred in fold_preds:
                all_preds[idx] = pred
                
            # Store meta-features
            meta_features[:, i*2:(i+1)*2] = all_preds
            
        return meta_features
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the hyper-ensemble"""
        # Generate meta-features
        self.meta_features = self._generate_meta_features(X, y)
        
        # Train boosting models on meta-features
        for name, model in self.boosting_models.items():
            model.fit(
                self.meta_features,
                y,
                eval_metric=['auc', 'logloss'],
                early_stopping_rounds=50,
                verbose=False
            )
        
        # Train final meta-learner
        if self.meta_learner_type == 'weighted':
            # Optimize weights using validation predictions
            val_preds = [model.predict_proba(X)[:, 1] for model in self.base_models]
            self.model_weights = self._optimize_weights(val_preds, y)
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate probability predictions"""
        # Get base model predictions
        base_preds = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            base_preds[:, i] = model.predict_proba(X)[:, 1]
        
        # Get boosting model predictions
        boost_preds = np.zeros((X.shape[0], len(self.boosting_models)))
        for i, model in enumerate(self.boosting_models.values()):
            boost_preds[:, i] = model.predict_proba(X)[:, 1]
        
        # Combine predictions
        if self.meta_learner_type == 'weighted':
            final_pred = np.zeros(X.shape[0])
            # Weighted average of base models
            final_pred += np.average(base_preds, weights=self.model_weights, axis=1) * 0.5
            # Average of boosting models
            final_pred += np.mean(boost_preds, axis=1) * 0.5
        else:
            # Simple average
            final_pred = np.mean(np.concatenate([base_preds, boost_preds], axis=1), axis=1)
        
        return np.column_stack((1 - final_pred, final_pred))

    def explain_prediction(self, X: np.ndarray, sample_idx: int = None) -> Dict:
        """Generate SHAP explanations for predictions"""
        import shap
        
        explanations = {}
        
        # Explain base models
        for i, model in enumerate(self.base_models):
            if hasattr(model, 'predict_proba'):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # For binary classification, take positive class
                explanations[f'base_model_{i}'] = {
                    'shap_values': shap_values,
                    'expected_value': explainer.expected_value if isinstance(explainer.expected_value, float) 
                                    else explainer.expected_value[1]
                }
        
        # Explain boosting models
        for name, model in self.boosting_models.items():
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            explanations[name] = {
                'shap_values': shap_values,
                'expected_value': explainer.expected_value if isinstance(explainer.expected_value, float) 
                                else explainer.expected_value[1]
            }
        
        # If sample_idx is provided, return explanation for specific sample
        if sample_idx is not None:
            for model_name in explanations:
                explanations[model_name]['shap_values'] = \
                    explanations[model_name]['shap_values'][sample_idx]
        
        return explanations

    def get_feature_importance(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get global feature importance using SHAP values"""
        explanations = self.explain_prediction(X)
        
        feature_importance = {}
        for model_name, explanation in explanations.items():
            # Calculate mean absolute SHAP values for each feature
            importance = np.abs(explanation['shap_values']).mean(axis=0)
            feature_importance[model_name] = importance
            
        return feature_importance 