import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    VotingClassifier
)
from sklearn.inspection import permutation_importance
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    make_scorer,
    balanced_accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.processor_final import MicrobiomeProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_advanced_ensemble():
    """Create an advanced ensemble of diverse models"""
    # Base models with different architectures
    models = {
        "rf": RandomForestClassifier(
            n_estimators=2000,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        ),
        "xgb": XGBClassifier(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            scale_pos_weight=1,
            n_jobs=-1,
            random_state=42
        ),
        "lgb": LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.01,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary",
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        ),
        "et": ExtraTreesClassifier(
            n_estimators=1000,
            max_depth=15,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        ),
        "gb": GradientBoostingClassifier(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=7,
            subsample=0.8,
            random_state=42
        ),
        "svm": SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=42
        )
    }
    
    # Create voting classifier
    ensemble = VotingClassifier(
        estimators=[(name, model) for name, model in models.items()],
        voting="soft",
        n_jobs=-1
    )
    
    return ensemble

def create_parameter_distributions():
    """Create parameter distributions for random search"""
    return {
        "rf__n_estimators": randint(1000, 3000),
        "rf__max_depth": randint(10, 30),
        "rf__min_samples_split": randint(2, 10),
        "rf__min_samples_leaf": randint(1, 5),
        
        "xgb__n_estimators": randint(500, 2000),
        "xgb__learning_rate": uniform(0.001, 0.1),
        "xgb__max_depth": randint(3, 12),
        "xgb__subsample": uniform(0.6, 0.4),
        "xgb__colsample_bytree": uniform(0.6, 0.4),
        
        "lgb__n_estimators": randint(500, 2000),
        "lgb__learning_rate": uniform(0.001, 0.1),
        "lgb__num_leaves": randint(20, 50),
        "lgb__subsample": uniform(0.6, 0.4),
        "lgb__colsample_bytree": uniform(0.6, 0.4),
        
        "et__n_estimators": randint(500, 2000),
        "et__max_depth": randint(10, 30),
        
        "gb__n_estimators": randint(500, 2000),
        "gb__learning_rate": uniform(0.001, 0.1),
        "gb__max_depth": randint(3, 12),
        "gb__subsample": uniform(0.6, 0.4),
        
        "svm__C": uniform(0.1, 10),
        "svm__gamma": uniform(0.001, 0.1)
    }

def compute_feature_importance(model, X, y):
    """Compute and visualize feature importance using sklearn's permutation importance"""
    try:
        # Get feature importance from permutation importance
        result = permutation_importance(
            model, X, y,
            n_repeats=10,
            random_state=42,
            n_jobs=-1
        )
        
        importance = result.importances_mean
        std = result.importances_std
        
        feature_imp = pd.DataFrame({
            'feature': X.columns,
            'importance': importance,
            'std': std
        }).sort_values('importance', ascending=False)
        
        # Plot feature importance with error bars
        plt.figure(figsize=(12, 8))
        plt.errorbar(
            x=feature_imp['importance'][:20],
            y=range(20),
            xerr=feature_imp['std'][:20],
            fmt='o'
        )
        plt.yticks(range(20), feature_imp['feature'][:20])
        plt.xlabel('Permutation Importance')
        plt.title('Top 20 Most Important Features (with std)')
        plt.tight_layout()
        plt.savefig('data/results/feature_importance.png')
        plt.close()
        
        # Plot feature importance distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(data=feature_imp, x='importance', bins=50)
        plt.title('Feature Importance Distribution')
        plt.tight_layout()
        plt.savefig('data/results/feature_importance_dist.png')
        plt.close()
        
        return feature_imp
        
    except Exception as e:
        logger.error(f"Error in compute_feature_importance: {str(e)}")
        raise

def evaluate_model(model, X, y):
    """Comprehensive model evaluation with multiple metrics"""
    try:
        # Make predictions
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': balanced_accuracy_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_prob),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred)
        }
        
        # Generate confusion matrix
        cm = confusion_matrix(y, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('data/results/confusion_matrix.png')
        plt.close()
        
        # Save classification report
        report = classification_report(y, y_pred, output_dict=True)
        
        return metrics, report
        
    except Exception as e:
        logger.error(f"Error in evaluate_model: {str(e)}")
        raise

def train_and_evaluate():
    try:
        # Initialize data processor
        processor = MicrobiomeProcessor()
        
        # Load and prepare features
        logger.info("Loading and preparing features...")
        X, y = processor.prepare_data()
        
        logger.info(f"Final feature matrix shape: {X.shape}")
        logger.info(f"Class distribution:\n{pd.Series(y).value_counts()}")
        
        # Create pipeline with SMOTE
        pipeline = ImbPipeline([
            ("scaler", StandardScaler()),
            ("sampler", SMOTE(random_state=42)),
            ("ensemble", create_advanced_ensemble())
        ])
        
        # Set up cross-validation
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        
        # Create custom scorer that prioritizes balanced accuracy
        scorer = make_scorer(balanced_accuracy_score)
        
        # Perform random search
        logger.info("Performing random search for hyperparameter optimization...")
        random_search = RandomizedSearchCV(
            pipeline,
            param_distributions=create_parameter_distributions(),
            n_iter=100,
            cv=cv,
            scoring=scorer,
            n_jobs=-1,
            random_state=42,
            verbose=2
        )
        
        random_search.fit(X, y)
        
        logger.info(f"Best parameters: {random_search.best_params_}")
        logger.info(f"Best cross-validation score: {random_search.best_score_:.3f}")
        
        # Get cross-validation scores for best model
        cv_scores = cross_val_score(
            random_search.best_estimator_,
            X,
            y,
            cv=cv,
            scoring=scorer,
            n_jobs=-1
        )
        
        logger.info(f"Cross-validation scores: {cv_scores}")
        logger.info(f"Mean CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Evaluate model
        logger.info("Performing comprehensive model evaluation...")
        metrics, classification_report = evaluate_model(random_search.best_estimator_, X, y)
        
        # Analyze feature importance
        logger.info("Analyzing feature importance...")
        feature_importance = compute_feature_importance(random_search.best_estimator_, X, y)
        
        # Save results
        Path("data/results").mkdir(exist_ok=True)
        results = {
            "best_params": random_search.best_params_,
            "best_score": float(random_search.best_score_),
            "cv_scores": cv_scores.tolist(),
            "mean_cv_score": float(cv_scores.mean()),
            "std_cv_score": float(cv_scores.std()),
            "n_features": int(X.shape[1]),
            "n_samples": int(X.shape[0]),
            "metrics": metrics,
            "classification_report": classification_report,
            "feature_importance": feature_importance.to_dict()
        }
        
        with open("data/results/model_evaluation.json", "w") as f:
            json.dump(results, f, indent=4)
        
        # Save best model
        joblib.dump(random_search.best_estimator_, "data/results/best_model.joblib")
        
        return random_search.best_estimator_, results
        
    except Exception as e:
        logger.error(f"Error in train_and_evaluate: {str(e)}")
        raise

if __name__ == "__main__":
    model, results = train_and_evaluate()