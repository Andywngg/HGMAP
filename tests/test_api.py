import pytest
from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
from pathlib import Path
import json
from src.api.main import app
from src.model.advanced_ensemble import HyperEnsemble
from src.data.processor import MicrobiomeDataProcessor

client = TestClient(app)

@pytest.fixture
def sample_data():
    """Generate sample data for testing"""
    np.random.seed(42)
    n_samples = 100
    n_features = 50
    
    # Generate random abundance data
    abundances = pd.DataFrame(
        np.random.negative_binomial(n=10, p=0.5, size=(n_samples, n_features)),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Generate random metadata
    metadata = pd.DataFrame({
        'age': np.random.normal(50, 15, n_samples),
        'sex': np.random.choice(['M', 'F'], n_samples),
        'bmi': np.random.normal(25, 5, n_samples),
        'disease_status': np.random.choice([0, 1], n_samples)
    })
    
    return abundances, metadata

@pytest.fixture
def trained_model(sample_data):
    """Create and train a model for testing"""
    abundances, metadata = sample_data
    model = HyperEnsemble()
    X = abundances.values
    y = metadata['disease_status'].values
    model.fit(X, y)
    return model

def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_root():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_predict_endpoint(trained_model):
    """Test prediction endpoint"""
    # Prepare test data
    test_data = {
        "taxa_abundances": {f"feature_{i}": float(np.random.random()) 
                           for i in range(50)},
        "metadata": {
            "age": 45,
            "sex": "M",
            "bmi": 24.5
        }
    }
    
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    
    result = response.json()
    assert "prediction" in result
    assert "probability" in result
    assert "confidence_score" in result
    assert "important_features" in result
    
    assert isinstance(result["prediction"], str)
    assert isinstance(result["probability"], float)
    assert isinstance(result["confidence_score"], float)
    assert isinstance(result["important_features"], list)

def test_model_info():
    """Test model info endpoint"""
    response = client.get("/model/info")
    assert response.status_code == 200
    
    info = response.json()
    assert "model_type" in info
    assert "feature_count" in info
    assert "features" in info

def test_invalid_input():
    """Test handling of invalid input"""
    invalid_data = {
        "taxa_abundances": {},  # Empty abundances
        "metadata": {}
    }
    
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422  # Validation error

def test_data_processor():
    """Test data processing pipeline"""
    processor = MicrobiomeDataProcessor()
    abundances, metadata = sample_data()
    
    # Test data processing
    processed_abundances, processed_metadata = processor._process_data(
        abundances, metadata
    )
    
    assert not processed_abundances.isna().any().any()
    assert not processed_metadata.isna().any().any()
    
    # Test data saving and loading
    processor.save_processed_data(
        processed_abundances,
        processed_metadata,
        "test_dataset"
    )
    
    loaded_abundances, loaded_metadata = processor.load_processed_data("test_dataset")
    
    pd.testing.assert_frame_equal(processed_abundances, loaded_abundances)
    pd.testing.assert_frame_equal(processed_metadata, loaded_metadata)

def test_model_persistence(trained_model, tmp_path):
    """Test model saving and loading"""
    import joblib
    
    # Save model
    model_path = tmp_path / "test_model.joblib"
    joblib.dump(trained_model, model_path)
    
    # Load model
    loaded_model = joblib.load(model_path)
    
    # Compare predictions
    X = np.random.random((10, 50))
    np.testing.assert_array_almost_equal(
        trained_model.predict_proba(X),
        loaded_model.predict_proba(X)
    )

def test_shap_explanations(trained_model):
    """Test SHAP explanations"""
    X = np.random.random((5, 50))
    explanations = trained_model.explain_prediction(X)
    
    assert isinstance(explanations, dict)
    for model_name, explanation in explanations.items():
        assert 'shap_values' in explanation
        assert 'expected_value' in explanation
        assert explanation['shap_values'].shape == X.shape

if __name__ == '__main__':
    pytest.main([__file__]) 