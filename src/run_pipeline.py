#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script for running the complete microbiome analysis pipeline.
This script orchestrates the entire workflow from data loading to model evaluation.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
import yaml
import joblib
from sklearn.model_selection import train_test_split

from .utils.config import load_config, create_directories, setup_logging
from .data.integrate_datasets import DataIntegrator
from .features.engineer_features import MicrobiomeFeatureEngineer
from .training.advanced_model import AdvancedModelTrainer
from .utils.metrics import ModelEvaluator
from .utils.visualization import VisualizationManager

def run_pipeline(config_path: Path) -> Dict[str, Any]:
    """Run the complete microbiome analysis pipeline."""
    # Load configuration
    config = load_config(config_path)
    
    # Set up directories and logging
    create_directories(config)
    setup_logging(config.logging)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting microbiome analysis pipeline")
    
    try:
        # Step 1: Data Integration
        logger.info("Step 1: Integrating datasets...")
        integrator = DataIntegrator(data_dir=config.data.input_dir)
        abundance_df, metadata_df = integrator.integrate_and_preprocess()
        
        # Step 2: Feature Engineering
        logger.info("Step 2: Engineering features...")
        feature_engineer = MicrobiomeFeatureEngineer(
            n_pca_components=config.feature_engineering.n_pca_components,
            min_prevalence=config.feature_engineering.min_prevalence,
            min_abundance=config.feature_engineering.min_abundance,
            taxonomy_levels=config.feature_engineering.taxonomy_levels
        )
        
        # Engineer features
        features_df = feature_engineer.engineer_features(
            abundance_df=abundance_df,
            metadata_df=metadata_df
        )
        
        # Step 3: Model Training
        logger.info("Step 3: Training models...")
        X = features_df.drop(columns=['health_status']).values
        y = (features_df['health_status'] == 'disease').astype(int).values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config.training.test_size,
            random_state=config.training.random_state,
            stratify=y
        )
        
        # Train models
        trainer = AdvancedModelTrainer(
            data_dir=config.data.output_dir,
            n_splits=config.training.n_splits,
            random_state=config.training.random_state,
            n_trials=config.training.n_trials
        )
        
        best_models = trainer.train_and_evaluate(X_train, y_train)
        
        # Save results
        trainer.save_results(
            X_train,
            feature_names=features_df.drop(columns=['health_status']).columns.tolist()
        )
        
        # Step 4: Model Evaluation
        logger.info("Step 4: Evaluating models...")
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_all(best_models, X_test, y_test)
        
        # Step 5: Visualization
        logger.info("Step 5: Generating visualizations...")
        visualizer = VisualizationManager(output_dir=config.visualization.output_dir)
        visualizer.create_all_plots(
            features_df=features_df,
            models=best_models,
            metrics=metrics
        )
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python run_pipeline.py <config_path>")
        sys.exit(1)
    
    config_path = Path(sys.argv[1])
    metrics = run_pipeline(config_path)
    
    print("\nFinal Model Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}") 