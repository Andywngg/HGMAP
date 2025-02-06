"""
Advanced model training module for microbiome-based disease prediction.
Implements ensemble learning with hyperparameter optimization and SHAP explanations.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import logging
from typing import Dict, List, Tuple, Optional, Union

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
import xgboost as xgb
import lightgbm as lgb
import shap
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import optuna
from optuna.trial import Trial

class AdvancedModelTrainer:
    """Advanced model trainer implementing ensemble learning with hyperparameter optimization."""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        n_splits: int = 10,
        random_state: int = 42,
        n_trials: int = 100,
        use_smote: bool = True
    ):
        """
        Initialize the advanced model trainer.
        
        Args:
            data_dir: Directory for saving model artifacts
            n_splits: Number of cross-validation splits
            random_state: Random seed for reproducibility
            n_trials: Number of optimization trials
            use_smote: Whether to use SMOTE for class imbalance
        """
        self.data_dir = Path(data_dir)
        self.models_dir = self.data_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.n_splits = n_splits
        self.random_state = random_state
        self.n_trials = n_trials
        self.use_smote = use_smote
        
        self.logger = logging.getLogger(__name__)
        self.base_models = {}
        self.best_model = None
        self.feature_names = None
        
    def _get_base_models(self) -> Dict:
        """Initialize base models with default parameters."""
        return {
            'rf': RandomForestClassifier(
                random_state=self.random_state,
                class_weight='balanced'
            ),
            'gb': GradientBoostingClassifier(
                random_state=self.random_state
            ),
            'xgb': xgb.XGBClassifier(
                objective='binary:logistic',
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='logloss',
                scale_pos_weight=1  # Will be adjusted based on class distribution
            ),
            'lgb': lgb.LGBMClassifier(
                random_state=self.random_state,
                verbose=-1,
                is_unbalance=True
            )
        }
    
    def _create_pipeline(self, model, X: np.ndarray, y: np.ndarray) -> ImbPipeline:
        """Create a pipeline with SMOTE if enabled."""
        if self.use_smote:
            # Calculate class distribution
            class_counts = np.bincount(y)
            scale_pos_weight = class_counts[0] / class_counts[1]
            
            # Adjust model parameters for class imbalance
            if isinstance(model, xgb.XGBClassifier):
                model.set_params(scale_pos_weight=scale_pos_weight)
            
            return ImbPipeline([
                ('smote', SMOTE(random_state=self.random_state)),
                ('model', model)
            ])
        return ImbPipeline([('model', model)])
    
    def optimize_hyperparameters(
        self, 
        trial: Trial,
        model_name: str,
        X: np.ndarray,
        y: np.ndarray
    ) -> float:
        """Optimize hyperparameters for a specific model using Optuna."""
        if model_name == 'rf':
            model = RandomForestClassifier(
                n_estimators=trial.suggest_int('n_estimators', 100, 1000),
                max_depth=trial.suggest_int('max_depth', 3, 30),
                min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
                random_state=self.random_state,
                class_weight='balanced'
            )
        elif model_name == 'gb':
            model = GradientBoostingClassifier(
                n_estimators=trial.suggest_int('n_estimators', 100, 1000),
                learning_rate=trial.suggest_float('learning_rate', 1e-4, 1.0, log=True),
                max_depth=trial.suggest_int('max_depth', 3, 30),
                min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                random_state=self.random_state
            )
        elif model_name == 'xgb':
            model = xgb.XGBClassifier(
                objective='binary:logistic',
                n_estimators=trial.suggest_int('n_estimators', 100, 1000),
                max_depth=trial.suggest_int('max_depth', 3, 30),
                learning_rate=trial.suggest_float('learning_rate', 1e-4, 1.0, log=True),
                subsample=trial.suggest_float('subsample', 0.5, 1.0),
                colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        elif model_name == 'lgb':
            model = lgb.LGBMClassifier(
                n_estimators=trial.suggest_int('n_estimators', 100, 1000),
                learning_rate=trial.suggest_float('learning_rate', 1e-4, 1.0, log=True),
                max_depth=trial.suggest_int('max_depth', 3, 30),
                num_leaves=trial.suggest_int('num_leaves', 20, 100),
                random_state=self.random_state,
                verbose=-1,
                is_unbalance=True
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        # Create pipeline with SMOTE
        pipeline = self._create_pipeline(model, X, y)
        
        # Perform cross-validation
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        return scores.mean()
    
    def calculate_shap_values(self, model, X: np.ndarray) -> np.ndarray:
        """Calculate SHAP values for model interpretability."""
        try:
            import shap
            if isinstance(model, (RandomForestClassifier, GradientBoostingClassifier)):
                explainer = shap.TreeExplainer(model)
            elif isinstance(model, xgb.XGBClassifier):
                explainer = shap.TreeExplainer(model)
            elif isinstance(model, lgb.LGBMClassifier):
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X, 100))
            
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For binary classification
                
            return shap_values
        except ImportError:
            self.logger.warning("SHAP not installed. Skipping feature importance calculation.")
            return None
        except Exception as e:
            self.logger.warning(f"Could not calculate SHAP values: {str(e)}")
            return None
    
    def evaluate_model(self, model, X: np.ndarray, y: np.ndarray, model_name: str) -> Dict:
        """Evaluate model performance with multiple metrics."""
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_prob)
        }
        
        # Calculate confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        self.logger.info(f"\n{model_name.upper()} Performance Metrics:")
        for metric, value in metrics.items():
            self.logger.info(f"{metric}: {value:.4f}")
        
        self.logger.info("\nConfusion Matrix:")
        self.logger.info(f"{cm}")
        
        # Calculate and save SHAP values
        if hasattr(model, 'named_steps'):
            model_step = model.named_steps['model']
        else:
            model_step = model
            
        shap_values = self.calculate_shap_values(model_step, X)
        if shap_values is not None:
            joblib.dump(shap_values, self.models_dir / f"{model_name}_shap_values.joblib")
        
        return metrics
    
    def train_base_models(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict:
        """
        Train and optimize base models.
        
        Args:
            X: Training features
            y: Target labels
            
        Returns:
            Dictionary of optimized base models
        """
        optimized_models = {}
        
        for model_name in self._get_base_models().keys():
            self.logger.info(f"Optimizing {model_name}...")
            
            study = optuna.create_study(direction='maximize')
            study.optimize(
                lambda trial: self.optimize_hyperparameters(trial, model_name, X, y),
                n_trials=self.n_trials
            )
            
            # Create model with best parameters
            if model_name == 'rf':
                model = RandomForestClassifier(
                    **study.best_params,
                    random_state=self.random_state,
                    class_weight='balanced'
                )
            elif model_name == 'gb':
                model = GradientBoostingClassifier(
                    **study.best_params,
                    random_state=self.random_state
                )
            elif model_name == 'xgb':
                model = xgb.XGBClassifier(
                    objective='binary:logistic',
                    **study.best_params,
                    random_state=self.random_state,
                    use_label_encoder=False,
                    eval_metric='logloss'
                )
            elif model_name == 'lgb':
                model = lgb.LGBMClassifier(
                    **study.best_params,
                    random_state=self.random_state,
                    verbose=-1,
                    is_unbalance=True
                )
            
            # Create pipeline with SMOTE
            pipeline = self._create_pipeline(model, X, y)
            
            # Fit model on full dataset
            pipeline.fit(X, y)
            optimized_models[model_name] = pipeline
            
            # Save model
            joblib.dump(pipeline, self.models_dir / f"{model_name}_model.joblib")
            
            # Evaluate model
            self.evaluate_model(pipeline, X, y, model_name)
        
        return optimized_models
    
    def train_stacking_ensemble(
        self,
        X: np.ndarray,
        y: np.ndarray,
        base_models: Dict
    ) -> StackingClassifier:
        """
        Train stacking ensemble using optimized base models.
        
        Args:
            X: Training features
            y: Target labels
            base_models: Dictionary of trained base models
            
        Returns:
            Trained stacking ensemble model
        """
        estimators = [(name, model) for name, model in base_models.items()]
        
        stacking = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000),
            cv=self.n_splits,
            n_jobs=-1
        )
        
        # Create pipeline with SMOTE
        pipeline = self._create_pipeline(stacking, X, y)
        pipeline.fit(X, y)
        
        # Evaluate ensemble
        metrics = self.evaluate_model(pipeline, X, y, "ensemble")
        
        # Check if model meets accuracy requirement
        if metrics['accuracy'] < 0.9:
            self.logger.warning("Model accuracy is below 90% threshold")
        
        return pipeline
    
    def train_and_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Train and evaluate the complete model pipeline.
        
        Args:
            X: Training features
            y: Target labels
            feature_names: Optional list of feature names
            
        Returns:
            Dictionary of trained models
        """
        self.logger.info("Starting model training pipeline...")
        
        # Store feature names
        self.feature_names = feature_names
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Save scaler
        joblib.dump(scaler, self.models_dir / "scaler.joblib")
        
        # Train and optimize base models
        self.base_models = self.train_base_models(X_scaled, y)
        
        # Train stacking ensemble
        self.best_model = self.train_stacking_ensemble(X_scaled, y, self.base_models)
        
        # Save ensemble model
        joblib.dump(self.best_model, self.models_dir / "ensemble_model.joblib")
        
        # Save feature names if provided
        if feature_names is not None:
            with open(self.models_dir / "feature_names.txt", "w") as f:
                f.write("\n".join(feature_names))
        
        self.logger.info("Model training completed successfully.")
        
        # Return all models
        return {
            'base_models': self.base_models,
            'ensemble_model': self.best_model
        }
    
    def predict(
        self,
        X: np.ndarray,
        return_proba: bool = False,
        return_shap: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Optional[np.ndarray]]]:
        """
        Make predictions using the trained ensemble model.
        
        Args:
            X: Features to predict
            return_proba: Whether to return probability estimates
            return_shap: Whether to return SHAP values
            
        Returns:
            Predictions or tuple of predictions and SHAP values
        """
        if self.best_model is None:
            raise ValueError("Model has not been trained yet")
            
        # Scale features
        scaler = joblib.load(self.models_dir / "scaler.joblib")
        X_scaled = scaler.transform(X)
        
        if return_shap:
            if hasattr(self.best_model, 'named_steps'):
                model_step = self.best_model.named_steps['model']
            else:
                model_step = self.best_model
            shap_values = self.calculate_shap_values(model_step, X_scaled)
        else:
            shap_values = None
        
        if return_proba:
            predictions = self.best_model.predict_proba(X_scaled)[:, 1]
        else:
            predictions = self.best_model.predict(X_scaled)
            
        if return_shap:
            return predictions, shap_values
        return predictions 