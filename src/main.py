#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Dict, Any

from features.engineer_features import MicrobiomeFeatureEngineer
from training.train_advanced import MicrobiomeModelTrainer
from evaluation.evaluate_model import ModelEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(data_dir: str = 'data/mgnify') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load abundance and metadata from processed MGnify data."""
    data_dir = Path(data_dir)
    
    # Load abundance data
    abundance_df = pd.read_csv(data_dir / 'abundance.csv', index_col=0)
    metadata_df = pd.read_csv(data_dir / 'metadata.csv', index_col=0)
    
    logger.info(f"Loaded abundance data with shape: {abundance_df.shape}")
    logger.info(f"Loaded metadata with shape: {metadata_df.shape}")
    
    return abundance_df, metadata_df

def prepare_labels(metadata_df: pd.DataFrame) -> Tuple[np.ndarray, list]:
    """Prepare labels from metadata."""
    # Extract health status
    health_status = metadata_df['study_name'].apply(lambda x: 'healthy' if 'healthy' in x.lower() 
                                                  else x.split('(')[-1].strip(')').strip())
    
    # Get unique labels
    unique_labels = sorted(health_status.unique())
    label_map = {label: i for i, label in enumerate(unique_labels)}
    
    # Convert to numeric labels
    y = health_status.map(label_map).values
    
    logger.info(f"Prepared labels with {len(unique_labels)} classes: {unique_labels}")
    return y, unique_labels

def main():
    """Main pipeline execution."""
    # Create output directories
    output_dirs = ['features', 'models', 'evaluation']
    for dir_name in output_dirs:
        Path(dir_name).mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading data...")
    abundance_df, metadata_df = load_data()
    
    # Prepare labels
    logger.info("Preparing labels...")
    y, labels = prepare_labels(metadata_df)
    
    # Engineer features
    logger.info("Engineering features...")
    feature_engineer = MicrobiomeFeatureEngineer(n_components=10)
    X = feature_engineer.engineer_features(abundance_df, abundance_df.index)
    feature_engineer.save_features(X)
    
    # Train model
    logger.info("Training model...")
    model_trainer = MicrobiomeModelTrainer()
    results = model_trainer.train(X, y)
    
    # Log training results
    logger.info("\nTraining Results:")
    for metric, value in results.items():
        logger.info(f"{metric}: {value:.4f}")
    
    # Save model
    logger.info("Saving model...")
    model_trainer.save_model()
    
    # Evaluate model
    logger.info("Evaluating model...")
    evaluator = ModelEvaluator()
    eval_results = evaluator.evaluate(X, y, labels)
    evaluator.save_results(eval_results)
    
    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    main() 