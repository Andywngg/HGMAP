#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script for running the microbiome analysis pipeline.
This script orchestrates the complete workflow from data preprocessing to model evaluation.
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

from features.engineer_features import MicrobiomeFeatureEngineer
from training.train_advanced import MicrobiomeModelTrainer
from evaluation.evaluate_model import ModelEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_and_preprocess_data(data_dir: Path) -> pd.DataFrame:
    """Load and preprocess the microbiome data."""
    logger.info("Loading and preprocessing data...")
    
    # Load abundance data
    abundance_files = list(data_dir.glob("*_converted.tsv"))
    if not abundance_files:
        raise FileNotFoundError("No converted abundance files found")
    
    # Combine abundance data
    abundance_dfs = []
    for file in abundance_files:
        df = pd.read_csv(file, sep='\t', index_col=0)
        # Extract taxonomy information
        taxonomy = df.index.str.split('|', expand=True)
        df.index = taxonomy[0]  # Use OTU ID as index
        abundance_dfs.append(df)
    
    abundance_df = pd.concat(abundance_dfs, axis=1)
    
    # Load metadata if available
    metadata_path = data_dir / "metadata.csv"
    if metadata_path.exists():
        metadata_df = pd.read_csv(metadata_path, index_col=0)
        # Merge abundance and metadata
        merged_df = abundance_df.merge(
            metadata_df,
            left_index=True,
            right_index=True,
            how='inner'
        )
    else:
        merged_df = abundance_df
        logger.warning("No metadata file found. Proceeding with abundance data only.")
    
    logger.info(f"Loaded data with shape: {merged_df.shape}")
    return merged_df

def prepare_features_and_labels(
    data_df: pd.DataFrame,
    target_col: str = 'disease_status'
) -> tuple:
    """Prepare features and labels from the data."""
    from features.microbiome_features import MicrobiomeFeatureEngineer
    
    logger.info("Preparing features and labels...")
    
    # Initialize feature engineer
    feature_engineer = MicrobiomeFeatureEngineer(
        n_pca_components=50,
        min_prevalence=0.1,
        min_abundance=0.001
    )
    
    # Separate features and target
    if target_col in data_df.columns:
        y = data_df[target_col]
        X = data_df.drop(columns=[target_col])
    else:
        logger.warning(f"Target column '{target_col}' not found. Using all columns as features.")
        X = data_df
        y = None
    
    # Engineer features
    X_engineered = feature_engineer.engineer_features(
        abundance_df=X,
        metadata_df=None  # We'll handle metadata separately if needed
    )
    
    # Encode labels if available
    if y is not None:
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
    else:
        y_encoded = None
        label_encoder = None
    
    logger.info(f"Prepared {X_engineered.shape[1]} features")
    return X_engineered, y_encoded, label_encoder

def train_and_evaluate_model(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2
) -> tuple:
    """Train and evaluate the model."""
    from model.advanced_ensemble import HyperEnsemble
    
    logger.info("Training and evaluating model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=42
    )
    
    # Apply SMOTE for class balancing
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    # Initialize and train model
    model = HyperEnsemble(n_folds=5, random_state=42)
    model.fit(X_train_balanced, y_train_balanced)
    
    # Evaluate model
    metrics = model.evaluate(X_test, y_test)
    cv_results = model.cross_validate(X, y)
    
    logger.info("\nTest Metrics:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    logger.info(f"\nCross-validation ROC-AUC: {cv_results['mean_cv_score']:.4f} ± {cv_results['std_cv_score']:.4f}")
    
    return model, metrics, cv_results

def save_results(
    model,
    metrics: dict,
    cv_results: dict,
    feature_names: list,
    output_dir: Path
) -> None:
    """Save model results and visualizations."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import shap
    
    logger.info("Saving results and generating visualizations...")
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # Save metrics
    results = {
        'metrics': metrics,
        'cv_results': {
            'mean_score': cv_results['mean_cv_score'],
            'std_score': cv_results['std_cv_score']
        }
    }
    
    import json
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=4)
    
    # Generate and save visualizations
    if hasattr(model, 'feature_importance_') and model.feature_importance_ is not None:
        # Feature importance plot
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importance_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(
            data=importance_df.head(20),
            x='importance',
            y='feature'
        )
        plt.title('Top 20 Most Important Features')
        plt.tight_layout()
        plt.savefig(viz_dir / 'feature_importance.png')
        plt.close()
        
        # SHAP summary plot if available
        if hasattr(model, 'shap_values_') and model.shap_values_ is not None:
            shap.summary_plot(
                model.shap_values_,
                feature_names=feature_names,
                show=False
            )
            plt.savefig(viz_dir / 'shap_summary.png')
            plt.close()
    
    logger.info(f"Saved results to {output_dir}")

def main():
    """Run the complete pipeline."""
    try:
        # Set up paths
        base_dir = Path(__file__).parent.parent
        data_dir = base_dir / "data" / "mgnify"
        output_dir = base_dir / "results"
        
        # 1. Load and preprocess data
        data_df = load_and_preprocess_data(data_dir)
        
        # 2. Prepare features and labels
        X, y, label_encoder = prepare_features_and_labels(data_df)
        
        if y is not None:
            # 3. Train and evaluate model
            model, metrics, cv_results = train_and_evaluate_model(X.values, y)
            
            # 4. Save results
            save_results(
                model,
                metrics,
                cv_results,
                feature_names=X.columns.tolist(),
                output_dir=output_dir
            )
        else:
            logger.warning("No labels available for training. Stopping after feature engineering.")
            # Save engineered features
            X.to_csv(output_dir / "engineered_features.csv")
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main() 