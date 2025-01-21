import os
import logging
from pathlib import Path
import torch
import numpy as np
from config.constants import *
from src.train_advanced import AdvancedTrainingPipeline
from src.model.uncertainty import UncertaintyQuantifier
from src.interpretation.explainer import AdvancedExplainer

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main():
    logger = setup_logging()
    logger.info("Starting advanced microbiome analysis pipeline")
    
    try:
        # Initialize training pipeline
        pipeline = AdvancedTrainingPipeline(
            data_path=TAXONOMY_FILES["SupplementaryData1"],
            source_data_paths=[TAXONOMY_FILES["SupplementaryData2"]],
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Train model
        logger.info("Training model...")
        model, metrics = pipeline.run_pipeline()
        
        # Ensure all models have consistent input/output interfaces
        assert all(hasattr(model, 'predict_proba') for model in pipeline.models.values())
        assert all(hasattr(model, 'fit') for model in pipeline.models.values())
        
        # Uncertainty quantification
        logger.info("Performing uncertainty quantification...")
        uncertainty_quantifier = UncertaintyQuantifier(
            model=model,
            n_samples=50,
            dropout_rate=0.3
        )
        
        # Model interpretation
        logger.info("Generating model explanations...")
        explainer = AdvancedExplainer(
            model=model,
            feature_names=pipeline.feature_names
        )
        
        # Save results
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # Save metrics
        np.save(results_dir / "metrics.npy", metrics)
        
        # Save model
        torch.save(model.state_dict(), results_dir / "model.pt")
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}", exc_info=True)
        raise

def verify_data_pipeline(X):
    # Preprocessing
    X_preprocessed = preprocessor.fit_transform(X)
    assert X_preprocessed.shape[0] == X.shape[0]
    
    # Feature engineering
    X_engineered = feature_engineer.fit_transform(X_preprocessed)
    assert X_engineered.shape[0] == X.shape[0]
    
    # Feature selection
    X_selected = feature_selector.fit_transform(X_engineered)
    assert X_selected.shape[0] == X.shape[0]
    
    return True

if __name__ == "__main__":
    main() 