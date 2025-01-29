import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Train and evaluate machine learning models for microbiome data"""
    
    def __init__(
        self,
        random_state: int = 42,
        n_jobs: int = -1,
        model_dir: str = "models"
    ):
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.models = {}
        self.metrics = {}
        
    def prepare_data(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training"""
        X_train, X_test, y_train, y_test = train_test_split(
            features, target,
            test_size=test_size,
            random_state=self.random_state,
            stratify=target
        )
        return X_train, X_test, y_train, y_test
    
    def train_base_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Dict:
        """Train base models"""
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=self.random_state
            )
        }
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Cross validation
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=5, scoring='roc_auc'
            )
            
            logger.info(f"{name} CV ROC AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            
            self.models[name] = model
        
        return self.models
    
    def evaluate_model(
        self,
        model_name: str,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        """Evaluate model performance"""
        model = self.models[model_name]
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        logger.info(f"\nModel: {model_name}")
        logger.info(f"ROC AUC: {metrics['roc_auc']:.3f}")
        logger.info("\nClassification Report:")
        logger.info(metrics['classification_report'])
        
        self.metrics[model_name] = metrics
        return metrics
    
    def save_model(
        self,
        model_name: str,
        feature_names: Optional[list] = None
    ) -> None:
        """Save trained model and metrics"""
        model_path = self.model_dir / model_name
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(
            self.models[model_name],
            model_path / "model.joblib"
        )
        
        # Save metrics
        with open(model_path / "metrics.json", 'w') as f:
            json.dump(self.metrics[model_name], f, indent=2)
        
        # Save feature names
        if feature_names is not None:
            pd.Series(feature_names).to_csv(
                model_path / "feature_names.csv",
                index=False
            )
        
        logger.info(f"Saved model and metrics to {model_path}")
    
    def train_and_evaluate(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        test_size: float = 0.2
    ) -> Dict:
        """Train and evaluate all models"""
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(
            features, target, test_size
        )
        
        # Train models
        self.train_base_models(X_train, y_train)
        
        # Evaluate models
        results = {}
        for model_name in self.models:
            metrics = self.evaluate_model(model_name, X_test, y_test)
            results[model_name] = metrics
            
            # Save model and metrics
            self.save_model(model_name, features.columns.tolist())
        
        return results 