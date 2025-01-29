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
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import make_scorer, balanced_accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

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

def train_and_evaluate():
    try:
        # Initialize data processor
        processor = MicrobiomeProcessor()
        
        # Load and prepare features
        logger.info("Loading and preparing features...")
        X, y = processor.prepare_features()
        
        logger.info(f"Final feature matrix shape: {X.shape}")
        logger.info(f"Class distribution:\n{pd.Series(y).value_counts()}")
        
        # Create pipeline with SMOTE
        pipeline = ImbPipeline([
            ("scaler", StandardScaler()),
            ("sampler", SMOTE(random_state=42)),
            ("ensemble", create_advanced_ensemble())
        ])
        
        # Set up cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
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
        
        # Save results
        Path("data/results").mkdir(exist_ok=True)
        results = {
            "best_params": random_search.best_params_,
            "best_score": float(random_search.best_score_),
            "cv_scores": cv_scores.tolist(),
            "mean_cv_score": float(cv_scores.mean()),
            "std_cv_score": float(cv_scores.std()),
            "n_features": int(X.shape[1]),
            "n_samples": int(X.shape[0])
        }
        
        with open("data/results/model_evaluation.json", "w") as f:
            json.dump(results, f, indent=4)
        
        # Save best model
        best_model = random_search.best_estimator_
        best_model.fit(X, y)
        joblib.dump(best_model, "data/results/best_model.joblib")
        
        return best_model, results
        
    except Exception as e:
        logger.error(f"Error in train_and_evaluate: {str(e)}")
        raise

if __name__ == "__main__":
    model, results = train_and_evaluate() 