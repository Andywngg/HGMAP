import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_numpy_types(obj):
    """Convert numpy types to Python types recursively."""
    if isinstance(obj, dict):
        return {str(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.int8, np.int16, np.int32, np.int64,
                         np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    else:
        return obj

def align_samples(X, y):
    """Ensure proper sample alignment between features and targets."""
    # Convert indices to strings
    X.index = X.index.astype(str)
    y.index = y.index.astype(str)
    
    # Standardize sample ID format
    X.index = X.index.map(lambda x: f"sample_{x}" if not x.startswith("sample_") else x)
    y.index = y.index.map(lambda x: f"sample_{x}" if not x.startswith("sample_") else x)
    
    # Get common samples
    common_samples = X.index.intersection(y.index)
    logger.info(f"Number of aligned samples: {len(common_samples)}")
    
    if len(common_samples) == 0:
        raise ValueError("No common samples found between features and targets!")
    
    return X.loc[common_samples], y.loc[common_samples]

def preprocess_features(X):
    """Advanced feature preprocessing."""
    # Log transform abundance data
    X_processed = np.log1p(X)
    
    # Add ratio features
    top_features = X.mean().nlargest(10).index
    for i, feat1 in enumerate(top_features):
        for feat2 in top_features[i+1:]:
            ratio_name = f"ratio_{feat1}_{feat2}"
            X_processed[ratio_name] = X[feat1] / (X[feat2] + 1e-6)
    
    # Add sum and variance features
    X_processed['total_abundance'] = X.sum(axis=1)
    X_processed['abundance_variance'] = X.var(axis=1)
    
    # Add presence/absence features
    X_processed['nonzero_features'] = (X > 0).sum(axis=1)
    
    return X_processed

def evaluate_model(model, X, y, label_encoder):
    """Model evaluation with multi-class metrics."""
    try:
        # Make predictions
        y_pred = model.predict(X)
        
        # Calculate metrics
        metrics = {
            'balanced_accuracy': balanced_accuracy_score(y, y_pred),
            'macro_precision': precision_score(y, y_pred, average='macro'),
            'macro_recall': recall_score(y, y_pred, average='macro'),
            'macro_f1': f1_score(y, y_pred, average='macro'),
            'weighted_f1': f1_score(y, y_pred, average='weighted')
        }
        
        # Save feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        feature_importance.to_csv('data/results/feature_importance.csv', index=False)
        
        # Save classification report
        report = classification_report(y, y_pred, 
                                    target_names=label_encoder.classes_,
                                    output_dict=True)
        
        return metrics, report
        
    except Exception as e:
        logger.error(f"Error in evaluate_model: {str(e)}")
        raise

def train_and_evaluate():
    """Main training and evaluation function with improved handling."""
    try:
        # Load data
        logger.info("Loading data...")
        abundance_train = pd.read_csv("data/processed/abundance_train.csv", index_col=0)
        targets = pd.read_csv("data/processed/targets.csv", index_col=0)
        
        # Process features
        logger.info("Processing features...")
        X = abundance_train.copy()
        y = targets.iloc[:, 0]
        
        # Align samples
        logger.info("Aligning samples...")
        X, y = align_samples(X, y)
        
        # Advanced feature preprocessing
        logger.info("Performing advanced feature preprocessing...")
        X_processed = preprocess_features(X)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_processed)
        X_scaled = pd.DataFrame(X_scaled, columns=X_processed.columns, index=X_processed.index)
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        logger.info(f"Class distribution:\n{pd.Series(y).value_counts()}")
        logger.info(f"Number of classes: {len(label_encoder.classes_)}")
        logger.info(f"Number of features: {X_scaled.shape[1]}")
        
        # Create base model for feature selection
        base_model = RandomForestClassifier(
            n_estimators=1000,
            class_weight='balanced',
            random_state=42
        )
        
        # Perform feature selection
        logger.info("Performing feature selection...")
        selector = SelectFromModel(base_model, prefit=False, threshold='mean')
        X_selected = selector.fit_transform(X_scaled, y_encoded)
        selected_features = X_scaled.columns[selector.get_support()].tolist()
        logger.info(f"Selected {len(selected_features)} features")
        
        # Define parameter grid for optimization
        param_grid = {
            'n_estimators': [1000, 2000],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 4, 6],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Create and configure the model
        model = RandomForestClassifier(
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        )
        
        # Perform grid search
        logger.info("Performing grid search...")
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=5,
            scoring='balanced_accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_selected, y_encoded)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.3f}")
        
        # Get cross-validation scores for best model
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            grid_search.best_estimator_,
            X_selected,
            y_encoded,
            cv=cv,
            scoring='balanced_accuracy'
        )
        
        logger.info(f"Cross-validation scores: {cv_scores}")
        logger.info(f"Mean CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Fit final model
        logger.info("Training final model...")
        final_model = grid_search.best_estimator_
        final_model.fit(X_selected, y_encoded)
        
        # Evaluate model
        logger.info("Performing comprehensive model evaluation...")
        metrics, classification_report = evaluate_model(
            final_model,
            pd.DataFrame(X_selected, columns=selected_features, index=X_scaled.index),
            y_encoded,
            label_encoder
        )
        
        # Save results
        Path("data/results").mkdir(exist_ok=True)
        
        results = {
            "cv_scores": cv_scores.tolist(),
            "mean_cv_score": cv_scores.mean(),
            "std_cv_score": cv_scores.std(),
            "best_params": grid_search.best_params_,
            "metrics": metrics,
            "classification_report": classification_report,
            "label_mapping": dict(zip(label_encoder.classes_, range(len(label_encoder.classes_)))),
            "selected_features": selected_features
        }
        
        # Convert all numpy types to Python types
        results = convert_numpy_types(results)
        
        with open("data/results/model_evaluation.json", "w") as f:
            json.dump(results, f, indent=4)
        
        # Save model and preprocessing objects
        joblib.dump(final_model, "data/results/best_model.joblib")
        joblib.dump(scaler, "data/results/scaler.joblib")
        joblib.dump(selector, "data/results/feature_selector.joblib")
        joblib.dump(label_encoder, "data/results/label_encoder.joblib")
        
        return final_model, results
        
    except Exception as e:
        logger.error(f"Error in train_and_evaluate: {str(e)}")
        raise

if __name__ == "__main__":
    model, results = train_and_evaluate() 