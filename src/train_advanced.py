import torch
import numpy as np
from sklearn.model_selection import train_test_split
from src.data.advanced_preprocessing import AdvancedPreprocessor
from src.features.microbiome_features import MicrobiomeFeatureEngineer
from src.model.losses import MicrobiomeLoss
from src.model.transfer import TransferLearningModule
from src.optimization.hyperopt import AdvancedHyperOptimizer
from src.model.advanced_ensemble import HyperEnsemble
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import shap

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdvancedTrainingPipeline:
    def __init__(
        self,
        data_path: str,
        source_data_paths: List[str] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.data_path = data_path
        self.source_data_paths = source_data_paths or []
        
    def load_and_preprocess_data(self):
        """Load and preprocess the data"""
        # Load main dataset
        data = pd.read_excel(self.data_path)
        X = data.drop(columns=['target']).values
        y = data['target'].values
        
        # Load source datasets for transfer learning
        source_datasets = []
        for path in self.source_data_paths:
            source_data = pd.read_excel(path)
            source_datasets.append({
                'X': source_data.drop(columns=['target']).values,
                'y': source_data['target'].values
            })
        
        # Advanced preprocessing
        preprocessor = AdvancedPreprocessor(
            n_components=50,
            use_umap=True,
            use_pca=True,
            use_kernel_pca=True
        )
        X_preprocessed = preprocessor.fit_transform(X)
        
        # Feature engineering
        feature_engineer = MicrobiomeFeatureEngineer(
            min_prevalence=0.1,
            min_abundance=0.001,
            n_clusters=20
        )
        X_engineered = feature_engineer.fit_transform(
            X_preprocessed,
            feature_names=data.columns[:-1].tolist()
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_engineered, y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )
        
        return {
            'train': (X_train, y_train),
            'test': (X_test, y_test),
            'source_datasets': source_datasets
        }
    
    def optimize_hyperparameters(self, X_train, y_train):
        """Optimize model hyperparameters"""
        optimizer = AdvancedHyperOptimizer(
            model_builder=self.build_model,
            n_trials=100,
            cv_folds=5,
            metric='auc'
        )
        
        best_params, best_score = optimizer.optimize(X_train, y_train)
        print(f"Best validation score: {best_score}")
        return best_params
    
    def build_model(self, **params):
        """Build model with given parameters"""
        model = HyperEnsemble(**params)
        return model
    
    def train_model(self, data_dict, hyperparameters):
        """Train the model with all improvements"""
        X_train, y_train = data_dict['train']
        X_test, y_test = data_dict['test']
        
        # Initialize model with best hyperparameters
        model = self.build_model(**hyperparameters)
        
        # Initialize transfer learning
        transfer_module = TransferLearningModule(
            base_model=model,
            source_datasets=data_dict['source_datasets'],
            n_epochs=50,
            lr=1e-4
        )
        
        # Apply transfer learning
        model = transfer_module.apply_transfer_learning(
            target_data={'X': X_train, 'y': y_train}
        )
        
        # Initialize custom loss
        criterion = MicrobiomeLoss(
            alpha=0.5,
            beta=0.3,
            gamma=0.2,
            focal_gamma=2.0,
            class_weights=torch.tensor(
                [1.0, (y_train == 0).sum() / (y_train == 1).sum()]
            ).to(self.device)
        )
        
        # Train with custom loss
        model.train_with_custom_loss(
            X_train, y_train,
            criterion=criterion,
            n_epochs=100,
            batch_size=32,
            learning_rate=1e-4
        )
        
        return model, (X_test, y_test)
    
    def evaluate_model(self, model, test_data):
        """Evaluate the trained model"""
        X_test, y_test = test_data
        
        # Get predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_prob[:, 1]),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        
        return metrics
    
    def run_pipeline(self):
        """Run the complete training pipeline"""
        # Load and preprocess data
        print("Loading and preprocessing data...")
        data_dict = self.load_and_preprocess_data()
        
        # Optimize hyperparameters
        print("Optimizing hyperparameters...")
        best_params = self.optimize_hyperparameters(
            data_dict['train'][0],
            data_dict['train'][1]
        )
        
        # Train model
        print("Training model with all improvements...")
        model, test_data = self.train_model(data_dict, best_params)
        
        # Evaluate
        print("Evaluating model...")
        metrics = self.evaluate_model(model, test_data)
        
        print("\nFinal Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return model, metrics

class HyperEnsemble(BaseEstimator, ClassifierMixin):
    """Advanced ensemble model combining multiple base models with meta-learning."""
    
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
        self.base_models_config = base_models_config or self.get_default_models()
        self.base_models = {}
        self.meta_model = None
        self.feature_importance_ = None
        self.shap_values_ = None
        self.scaler = StandardScaler()
    
    def get_default_models(self) -> Dict:
        """Get default configuration for base models."""
        return {
            'rf': {
                'model': RandomForestClassifier,
                'params': {
                    'n_estimators': 500,
                    'max_depth': 15,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'n_jobs': -1,
                    'random_state': self.random_state,
                    'class_weight': 'balanced'
                }
            },
            'xgb': {
                'model': xgb.XGBClassifier,
                'params': {
                    'n_estimators': 500,
                    'max_depth': 8,
                    'learning_rate': 0.01,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'min_child_weight': 3,
                    'n_jobs': -1,
                    'random_state': self.random_state,
                    'scale_pos_weight': 1
                }
            },
            'lgb': {
                'model': lgb.LGBMClassifier,
                'params': {
                    'n_estimators': 500,
                    'num_leaves': 31,
                    'max_depth': 8,
                    'learning_rate': 0.01,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'min_child_samples': 20,
                    'n_jobs': -1,
                    'random_state': self.random_state,
                    'class_weight': 'balanced'
                }
            },
            'gb': {
                'model': GradientBoostingClassifier,
                'params': {
                    'n_estimators': 500,
                    'max_depth': 8,
                    'learning_rate': 0.01,
                    'subsample': 0.8,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': self.random_state
                }
            }
        }
    
    def _generate_meta_features(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        is_train: bool = True
    ) -> np.ndarray:
        """Generate meta-features using cross-validation predictions."""
        if is_train:
            meta_features = np.zeros((X.shape[0], len(self.base_models_config)))
            kf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            
            for i, (name, config) in enumerate(self.base_models_config.items()):
                cv_preds = np.zeros(X.shape[0])
                
                for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
                    model = config['model'](**config['params'])
                    model.fit(X[train_idx], y[train_idx])
                    
                    if self.use_probabilities:
                        cv_preds[val_idx] = model.predict_proba(X[val_idx])[:, 1]
                    else:
                        cv_preds[val_idx] = model.predict(X[val_idx])
                    
                    if fold == 0:
                        self.base_models[name] = model
                
                meta_features[:, i] = cv_preds
        else:
            meta_features = np.zeros((X.shape[0], len(self.base_models)))
            for i, (name, model) in enumerate(self.base_models.items()):
                if self.use_probabilities:
                    meta_features[:, i] = model.predict_proba(X)[:, 1]
                else:
                    meta_features[:, i] = model.predict(X)
        
        return meta_features
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'HyperEnsemble':
        """Fit the ensemble model."""
        logger.info("Fitting ensemble model...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Generate meta-features
        meta_features = self._generate_meta_features(X_scaled, y, is_train=True)
        
        # Fit meta-model
        self.meta_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state
        )
        self.meta_model.fit(meta_features, y)
        
        # Calculate feature importance
        self._calculate_feature_importance(X_scaled)
        
        logger.info("Model fitting completed")
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        X_scaled = self.scaler.transform(X)
        meta_features = self._generate_meta_features(X_scaled, is_train=False)
        return self.meta_model.predict_proba(meta_features)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.predict_proba(X)[:, 1] > 0.5
    
    def _calculate_feature_importance(self, X: np.ndarray) -> None:
        """Calculate feature importance using SHAP values."""
        logger.info("Calculating feature importance...")
        
        # Get feature importance from base models
        importance_matrix = np.zeros((X.shape[1], len(self.base_models)))
        for i, (name, model) in enumerate(self.base_models.items()):
            if hasattr(model, 'feature_importances_'):
                importance_matrix[:, i] = model.feature_importances_
        
        # Average feature importance across models
        self.feature_importance_ = importance_matrix.mean(axis=1)
        
        # Calculate SHAP values for the first base model (as an example)
        first_model = list(self.base_models.values())[0]
        explainer = shap.TreeExplainer(first_model)
        self.shap_values_ = explainer.shap_values(X)
        
        logger.info("Feature importance calculation completed")
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'auc_roc': roc_auc_score(y, y_prob)
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
            'std_cv_score': cv_scores.std()
        }

def main():
    """Run the model training process."""
    try:
        # Set up paths
        data_dir = Path("data/features")
        output_dir = Path("models")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load engineered features
        features_df = pd.read_csv(data_dir / "engineered_features.csv", index_col=0)
        
        # Prepare data
        X = features_df.drop(columns=['target']).values
        y = features_df['target'].values
        
        # Initialize and train model
        model = HyperEnsemble(n_folds=5, random_state=42)
        model.fit(X, y)
        
        # Evaluate model
        metrics = model.evaluate(X, y)
        cv_results = model.cross_validate(X, y)
        
        logger.info("\nModel Performance:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        logger.info(f"\nCross-validation ROC-AUC: {cv_results['mean_cv_score']:.4f} ± {cv_results['std_cv_score']:.4f}")
        
        # Save model and results
        import joblib
        joblib.dump(model, output_dir / "ensemble_model.joblib")
        
        logger.info("Model training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in model training process: {str(e)}")
        raise

if __name__ == "__main__":
    main() 