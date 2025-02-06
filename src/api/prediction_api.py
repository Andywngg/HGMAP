"""
FastAPI endpoint for microbiome-based disease predictions.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from typing import List, Dict, Optional
import logging

from ..training.advanced_model import AdvancedModelTrainer

# Initialize FastAPI app
app = FastAPI(
    title="Microbiome Disease Prediction API",
    description="API for predicting disease status based on microbiome data",
    version="1.0.0"
)

class PredictionRequest(BaseModel):
    """Request model for predictions."""
    features: List[float]
    feature_names: Optional[List[str]] = None
    explain: bool = False

class PredictionResponse(BaseModel):
    """Response model for predictions."""
    prediction: int
    probability: float
    confidence: str
    feature_importance: Optional[Dict[str, float]] = None
    shap_explanation: Optional[Dict[str, Dict[str, float]]] = None

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to the Microbiome Disease Prediction API",
        "version": "1.0.0",
        "status": "healthy"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make predictions based on microbiome features.
    
    Args:
        request: PredictionRequest object containing features
        
    Returns:
        PredictionResponse object containing prediction and metadata
    """
    try:
        # Convert features to numpy array
        features = np.array(request.features).reshape(1, -1)
        
        # Load model and scaler
        model_dir = Path("data/models")
        model = joblib.load(model_dir / "ensemble_model.joblib")
        scaler = joblib.load(model_dir / "scaler.joblib")
        
        # Scale features
        X_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        probabilities = model.predict_proba(X_scaled)[0]
        probability = probabilities[1] if prediction == 1 else probabilities[0]
        
        # Determine confidence level
        if probability >= 0.9:
            confidence = "very high"
        elif probability >= 0.7:
            confidence = "high"
        elif probability >= 0.5:
            confidence = "moderate"
        else:
            confidence = "low"
        
        # Get feature importance and SHAP explanation
        feature_importance = None
        shap_explanation = None
        
        if request.feature_names is not None:
            # Use first base model (Random Forest) for feature importance
            base_model = model.estimators_[0]
            if hasattr(base_model, 'feature_importances_'):
                importance = base_model.feature_importances_
                feature_importance = dict(zip(request.feature_names, importance))
            
            # Calculate SHAP values if requested
            if request.explain:
                try:
                    import shap
                    explainer = shap.TreeExplainer(base_model)
                    shap_values = explainer.shap_values(X_scaled)
                    
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]  # For binary classification
                    
                    # Get top contributing features
                    feature_contributions = {}
                    for idx in np.argsort(np.abs(shap_values[0]))[-10:]:  # Top 10 features
                        feature_contributions[request.feature_names[idx]] = float(shap_values[0][idx])
                    
                    shap_explanation = {
                        'feature_contributions': feature_contributions,
                        'base_value': float(explainer.expected_value if not isinstance(explainer.expected_value, list)
                                          else explainer.expected_value[1])
                    }
                except Exception as e:
                    logging.warning(f"SHAP explanation failed: {str(e)}")
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            confidence=confidence,
            feature_importance=feature_importance,
            shap_explanation=shap_explanation
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/model-info")
async def model_info():
    """Get information about the trained model."""
    try:
        model_dir = Path("data/models")
        model = joblib.load(model_dir / "ensemble_model.joblib")
        
        return {
            "model_type": "Stacking Ensemble",
            "base_models": [type(est).__name__ for name, est in model.estimators_],
            "final_estimator": type(model.final_estimator_).__name__,
            "feature_count": model.n_features_in_,
            "classes": model.classes_.tolist()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model info: {str(e)}"
        ) 