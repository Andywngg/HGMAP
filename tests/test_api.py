import pytest
from fastapi.testclient import TestClient
from src.api.main import app
import numpy as np

client = TestClient(app)

def test_root():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Microbiome Analysis API is running"}

def test_health():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_predict():
    """Test prediction endpoint with sample data"""
    # Sample microbiome data
    test_data = {
        "taxa_abundances": {
            "Bacteroides": 0.3,
            "Prevotella": 0.2,
            "Faecalibacterium": 0.15,
            "Roseburia": 0.1,
            "Bifidobacterium": 0.25
        },
        "metadata": {
            "age": "30",
            "gender": "F",
            "bmi": "22.5"
        }
    }
    
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    
    # Check response structure
    result = response.json()
    assert "prediction" in result
    assert "probability" in result
    assert "confidence_score" in result
    assert "important_features" in result
    
    # Check data types
    assert isinstance(result["prediction"], str)
    assert isinstance(result["probability"], float)
    assert isinstance(result["confidence_score"], float)
    assert isinstance(result["important_features"], list)
    
    # Check probability and confidence score ranges
    assert 0 <= result["probability"] <= 1
    assert 0 <= result["confidence_score"] <= 1

def test_model_info():
    """Test model info endpoint"""
    response = client.get("/model/info")
    assert response.status_code == 200
    
    result = response.json()
    assert "model_type" in result
    assert "feature_count" in result
    assert "features" in result
    
    assert isinstance(result["feature_count"], int)
    assert isinstance(result["features"], list)

def test_invalid_input():
    """Test prediction endpoint with invalid input"""
    invalid_data = {
        "taxa_abundances": {
            "Bacteroides": "invalid"  # Should be float
        }
    }
    
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422  # Validation error 