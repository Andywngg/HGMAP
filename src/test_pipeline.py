#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to verify the complete microbiome analysis pipeline.
This script will:
1. Generate mock data
2. Process and engineer features
3. Train and evaluate models
4. Test the API
5. Verify monitoring
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_mock_data(
    n_samples: int = 1000,
    n_features: int = 200,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate mock microbiome data for testing."""
    logger.info("Generating mock data...")
    
    # Set random seed
    rng = np.random.default_rng(random_state)
    
    # Generate abundance data with more realistic distributions
    abundance_matrix = np.zeros((n_samples, n_features))
    for i in range(n_features):
        # Generate data with zero-inflation (common in microbiome data)
        zero_mask = rng.random(n_samples) < 0.7  # 70% zeros
        abundance_matrix[~zero_mask, i] = rng.lognormal(mean=0, sigma=2, size=np.sum(~zero_mask))
    
    # Create feature names with realistic taxonomic information
    feature_names = []
    phyla = ['p__Firmicutes', 'p__Bacteroidetes', 'p__Proteobacteria', 'p__Actinobacteria']
    genera = [f'g__Genus_{i}' for i in range(50)]  # 50 unique genera
    
    for i in range(n_features):
        phylum = rng.choice(phyla)
        genus = rng.choice(genera)
        species = f's__Species_{i}'
        feature_names.append(f'k__Bacteria;{phylum};c__Class;o__Order;f__Family;{genus};{species}')
    
    # Create abundance DataFrame
    abundance_df = pd.DataFrame(
        abundance_matrix,
        columns=feature_names,
        index=[f"Sample_{i}" for i in range(n_samples)]
    )
    
    # Generate metadata with realistic distributions
    metadata = {
        'sample_id': [f"Sample_{i}" for i in range(n_samples)],
        'age': rng.integers(18, 80, n_samples),
        'bmi': rng.normal(25, 5, n_samples).clip(15, 45),  # Realistic BMI range
        'sex': rng.choice(['M', 'F'], n_samples),
        'health_status': rng.choice(['healthy', 'disease'], n_samples, p=[0.6, 0.4])  # Slight class imbalance
    }
    metadata_df = pd.DataFrame(metadata)
    metadata_df.set_index('sample_id', inplace=True)
    
    return abundance_df, metadata_df

def setup_directories() -> Dict[str, Path]:
    """Create necessary directories for testing."""
    dirs = {
        'data': Path('data'),
        'raw': Path('data/raw'),
        'processed': Path('data/processed'),
        'models': Path('data/models'),
        'monitoring': Path('data/monitoring')
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs

def test_diversity_metrics(abundance_df: pd.DataFrame):
    """Test diversity metrics calculation."""
    from src.features.diversity_metrics import DiversityCalculator
    
    logger.info("Testing diversity metrics calculation...")
    calculator = DiversityCalculator()
    metrics = calculator.calculate_all_metrics(abundance_df)
    
    # Verify results
    assert 'alpha_diversity' in metrics
    assert 'beta_diversity' in metrics
    logger.info("Diversity metrics calculation successful")
    return metrics

def test_model_training(
    abundance_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    dirs: Dict[str, Path]
):
    """Test model training and evaluation."""
    from src.training.advanced_model import AdvancedModelTrainer
    
    logger.info("Testing model training...")
    
    # Prepare features
    from src.features.diversity_metrics import DiversityCalculator
    calculator = DiversityCalculator()
    diversity_features = calculator.calculate_all_metrics(abundance_df)
    
    # Combine features
    X = np.hstack([
        abundance_df.values,
        diversity_features['alpha_diversity'].values.reshape(-1, 1),
        diversity_features['beta_diversity']
    ])
    
    y = (metadata_df['health_status'] == 'disease').astype(int).values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create feature names
    feature_names = list(abundance_df.columns) + ['alpha_diversity'] + [f'beta_div_{i}' for i in range(diversity_features['beta_diversity'].shape[1])]
    
    # Train models
    trainer = AdvancedModelTrainer(
        data_dir=str(dirs['models']),
        n_splits=5,
        random_state=42,
        n_trials=10,  # Reduced for testing
        use_smote=True  # Enable SMOTE for class imbalance
    )
    
    models = trainer.train_and_evaluate(
        X_train,
        y_train,
        feature_names=feature_names
    )
    
    return models, X_test, y_test

def test_monitoring(
    models: Dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    dirs: Dict[str, Path]
):
    """Test model monitoring."""
    from src.monitoring.model_monitor import ModelMonitor
    
    logger.info("Testing model monitoring...")
    
    # Initialize monitor
    monitor = ModelMonitor(
        model_dir=str(dirs['models']),
        monitoring_dir=str(dirs['monitoring'])
    )
    
    # Monitor predictions
    monitor.monitor_predictions(
        y_test,
        models['ensemble_model'].predict(X_test),
        models['ensemble_model'].predict_proba(X_test)[:, 1],
        pd.DataFrame(X_test)
    )
    
    return monitor

def test_api(
    models: Dict,
    dirs: Dict[str, Path]
):
    """Test API functionality."""
    import uvicorn
    import requests
    from multiprocessing import Process
    from time import sleep
    
    logger.info("Testing API...")
    
    # Start API server in a separate process
    def run_api():
        uvicorn.run(
            "src.api.prediction_api:app",
            host="127.0.0.1",
            port=8000,
            log_level="info"
        )
    
    api_process = Process(target=run_api)
    api_process.start()
    sleep(2)  # Wait for server to start
    
    try:
        # Test API endpoints
        response = requests.get("http://127.0.0.1:8000/health")
        assert response.status_code == 200
        
        # Test prediction endpoint
        test_data = {
            "features": [0.1] * 200,  # Match number of features
            "explain": True
        }
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=test_data
        )
        assert response.status_code == 200
        
        logger.info("API tests successful")
    finally:
        api_process.terminate()

def main():
    """Run the complete pipeline test."""
    try:
        # Setup
        dirs = setup_directories()
        
        # Generate mock data
        abundance_df, metadata_df = generate_mock_data()
        
        # Save mock data
        abundance_df.to_csv(dirs['raw'] / "abundance.csv")
        metadata_df.to_csv(dirs['raw'] / "metadata.csv")
        
        # Test diversity metrics
        diversity_metrics = test_diversity_metrics(abundance_df)
        
        # Test model training
        models, X_test, y_test = test_model_training(abundance_df, metadata_df, dirs)
        
        # Test monitoring
        monitor = test_monitoring(models, X_test, y_test, dirs)
        
        # Test API
        test_api(models, dirs)
        
        logger.info("Pipeline test completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline test failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 