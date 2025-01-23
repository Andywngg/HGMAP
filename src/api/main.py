from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import sys
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from model.model import AdvancedEnsembleModel
from features.microbiome_features import MicrobiomeFeatureEngineer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Microbiome Analysis API",
    description="API for microbiome-based disease detection",
    version="1.0.0"
)

class MicrobiomeData(BaseModel):
    """Input data model for microbiome analysis"""
    taxa_abundances: Dict[str, float]
    metadata: Optional[Dict[str, str]] = None

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction: str
    probability: float
    confidence_score: float
    important_features: List[Dict[str, float]]

# Load model and feature engineer
MODEL_PATH = Path("models/ensemble_model.joblib")
FEATURE_ENGINEER_PATH = Path("models/feature_engineer.joblib")

try:
    model = joblib.load(MODEL_PATH)
    feature_engineer = joblib.load(FEATURE_ENGINEER_PATH)
    logger.info("Model and feature engineer loaded successfully")
except Exception as e:
    logger.error(f"Error loading model or feature engineer: {e}")
    model = None
    feature_engineer = None

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Microbiome Analysis API is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None or feature_engineer is None:
        raise HTTPException(status_code=503, detail="Model or feature engineer not loaded")
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: MicrobiomeData):
    """Make predictions based on microbiome data"""
    try:
        # Convert input data to correct format
        abundances = pd.Series(data.taxa_abundances)
        
        # Feature engineering
        features = feature_engineer.transform(abundances.values.reshape(1, -1))
        
        # Make prediction
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Get feature importance
        importance = model.feature_importances_
        feature_importance = [
            {"feature": name, "importance": float(score)}
            for name, score in zip(feature_engineer.feature_names_, importance)
        ]
        
        # Calculate confidence score
        confidence = float(max(probabilities))
        
        return PredictionResponse(
            prediction=str(prediction),
            probability=float(probabilities[1]),
            confidence_score=confidence,
            important_features=sorted(feature_importance, key=lambda x: x["importance"], reverse=True)[:10]
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info")
async def model_info():
    """Get information about the current model"""
    return {
        "model_type": type(model).__name__,
        "feature_count": len(feature_engineer.feature_names_),
        "features": feature_engineer.feature_names_
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 