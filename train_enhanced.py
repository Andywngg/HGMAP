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
    VotingClassifier,
    StackingClassifier
)
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_advanced_ensemble():
    """Create an advanced stacking ensemble."""
    # Base models with optimized configurations
    base_models = [
        ('rf', RandomForestClassifier(
            n_estimators=2000,
            max_depth=20,
            min_samples_split=4,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        )),
        ('xgb', XGBClassifier(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1,
            n_jobs=-1,
            random_state=42
        )),
        ('lgb', LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.01,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        )),
        ('et', ExtraTreesClassifier(
            n_estimators=1000,
            max_depth=15,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        )),
        ('gb', GradientBoostingClassifier(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=7,
            subsample=0.8,
            random_state=42
        )),
        ('svm', SVC(
            kernel='rbf',
            probability=True,
            class_weight='balanced',
            random_state=42
        ))
    ]
    
    # Meta-classifier
    meta_classifier = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        num_leaves=31,
        class_weight='balanced',
        random_state=42
    )
    
    # Create stacking classifier
    stacking = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_classifier,
        cv=5,
        n_jobs=-1
    )
    
    return stacking

def create_parameter_distributions():
    """Create comprehensive parameter distributions for random search."""
    return {
        'rf__n_estimators': randint(1000, 3000),
        'rf__max_depth': randint(10, 30),
        'rf__min_samples_split': randint(2, 10),
        'rf__min_samples_leaf': randint(1, 5),
        
        'xgb__n_estimators': randint(500, 2000),
        'xgb__learning_rate': uniform(0.001, 0.1),
        'xgb__max_depth': randint(3, 12),
        'xgb__subsample': uniform(0.6, 0.4),
        'xgb__colsample_bytree': uniform(0.6, 0.4),
        'xgb__min_child_weight': randint(1, 7),
        'xgb__gamma': uniform(0, 0.5),
        
        'lgb__n_estimators': randint(500, 2000),
        'lgb__learning_rate': uniform(0.001, 0.1),
        'lgb__num_leaves': randint(20, 50),
        'lgb__subsample': uniform(0.6, 0.4),
        'lgb__colsample_bytree': uniform(0.6, 0.4),
        'lgb__min_child_samples': randint(10, 50),
        
        'et__n_estimators': randint(500, 2000),
        'et__max_depth': randint(10, 30),
        'et__min_samples_split': randint(2, 10),
        
        'gb__n_estimators': randint(500, 2000),
        'gb__learning_rate': uniform(0.001, 0.1),
        'gb__max_depth': randint(3, 12),
        'gb__subsample': uniform(0.6, 0.4),
        'gb__min_samples_split': randint(2, 10),
        
        'svm__C': uniform(0.1, 10),
        'svm__gamma': uniform(0.001, 0.1)
    }

def evaluate_model(model, X, y):
    """Comprehensive model evaluation with multiple metrics."""
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

def compute_diversity_metrics(abundance_matrix):
    """Compute comprehensive diversity metrics."""
    # Convert to proportions
    proportions = abundance_matrix.div(abundance_matrix.sum(axis=1), axis=0)
    
    # Shannon diversity
    shannon = -(proportions * np.log1p(proportions)).sum(axis=1)
    
    # Simpson diversity
    simpson = 1 - (proportions ** 2).sum(axis=1)
    
    # Species richness
    richness = (abundance_matrix > 0).sum(axis=1)
    
    # Pielou's evenness
    evenness = shannon / np.log1p(richness)
    
    # Berger-Parker dominance
    dominance = abundance_matrix.max(axis=1) / abundance_matrix.sum(axis=1)
    
    return pd.DataFrame({
        'shannon_diversity': shannon,
        'simpson_diversity': simpson,
        'species_richness': richness,
        'evenness': evenness,
        'dominance': dominance
    })

def process_abundance_data(abundance_df):
    """Process abundance data with advanced feature engineering."""
    try:
        # 1. Handle missing values
        abundance_matrix = abundance_df.fillna(0)
        
        # 2. Log transformation with pseudocount
        abundance_matrix = np.log1p(abundance_matrix)
        
        # 3. Compute diversity metrics
        diversity_features = compute_diversity_metrics(abundance_matrix)
        
        # 4. Combine features
        all_features = pd.concat([abundance_matrix, diversity_features], axis=1)
        
        # 5. Scale features
        scaler = StandardScaler()
        scaled_features = pd.DataFrame(
            scaler.fit_transform(all_features),
            columns=all_features.columns,
            index=all_features.index
        )
        
        return scaled_features
        
    except Exception as e:
        logger.error(f"Error in process_abundance_data: {str(e)}")
        raise

def train_and_evaluate():
    """Main training and evaluation function."""
    try:
        # Load data
        logger.info("Loading data...")
        abundance_train = pd.read_csv("data/processed/abundance_train.csv", index_col=0)
        targets = pd.read_csv("data/processed/targets.csv")
        
        # Fix sample IDs in targets
        targets.set_index(targets.columns[0], inplace=True)
        
        # Process features
        logger.info("Processing features...")
        X = process_abundance_data(abundance_train)
        
        # Convert indices to strings
        X.index = X.index.astype(str)
        targets.index = targets.index.astype(str)
        
        # Align samples between features and targets
        common_samples = X.index.intersection(targets.index)
        logger.info(f"Number of common samples: {len(common_samples)}")
        
        if len(common_samples) == 0:
            # Try to match sample IDs by removing potential prefixes/suffixes
            X_samples = set(X.index.str.extract(r'(\d+)', expand=False))
            target_samples = set(targets.index.str.extract(r'(\d+)', expand=False))
            
            # Create mapping between original and cleaned sample IDs
            X_mapping = {idx.str.extract(r'(\d+)', expand=False).iloc[0]: idx for idx in X.index}
            target_mapping = {idx.str.extract(r'(\d+)', expand=False).iloc[0]: idx for idx in targets.index}
            
            # Find common samples after cleaning
            common_clean_samples = X_samples.intersection(target_samples)
            logger.info(f"Number of common samples after cleaning: {len(common_clean_samples)}")
            
            # Get original sample IDs
            X_common = [X_mapping[s] for s in common_clean_samples]
            target_common = [target_mapping[s] for s in common_clean_samples]
            
            X = X.loc[X_common]
            y = targets.loc[target_common].iloc[:, 0]
        else:
            X = X.loc[common_samples]
            y = targets.loc[common_samples].iloc[:, 0]
        
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
        
        # Save results
        Path("data/results").mkdir(exist_ok=True)
        results = {
            "best_params": random_search.best_params_,
            "best_score": float(random_search.best_score_),
            "cv_scores": cv_scores.tolist(),
            "mean_cv_score": float(cv_scores.mean()),
            "std_cv_score": float(cv_scores.std()),
            "metrics": metrics,
            "classification_report": classification_report
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