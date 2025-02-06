from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import sys
import logging
import shap

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from model.model import AdvancedEnsembleModel
from features.microbiome_features import MicrobiomeFeatureEngineer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Microbiome Health Classifier API",
    description="API for predicting health status from microbiome data",
    version="1.0.0"
)

class MicrobiomeData(BaseModel):
    """Pydantic model for input data validation."""
    abundances: Dict[str, float]
    
    class Config:
        schema_extra = {
            "example": {
                "abundances": {
                    "Bacteroides": 0.25,
                    "Prevotella": 0.15,
                    "Faecalibacterium": 0.1,
                    "Roseburia": 0.05
                }
            }
        }

class PredictionResponse(BaseModel):
    """Pydantic model for prediction response."""
    prediction: int
    probability: float
    feature_importance: Dict[str, float]
    prediction_explanation: str

# Load model and feature engineer
MODEL_PATH = Path("models/ensemble_model.joblib")
FEATURE_NAMES_PATH = Path("models/feature_names.joblib")

try:
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    logger.info("Model and feature names loaded successfully")
except Exception as e:
    logger.error(f"Error loading model or feature names: {str(e)}")
    raise

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Microbiome Analysis API is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}

def validate_input_features(data: Dict[str, float]) -> pd.DataFrame:
    """Validate and prepare input features."""
    try:
        # Create DataFrame with all feature names
        df = pd.DataFrame(0, index=[0], columns=feature_names)
        
        # Fill in provided values
        for feature, value in data.items():
            if feature in feature_names:
                df.loc[0, feature] = value
            else:
                logger.warning(f"Unknown feature {feature} will be ignored")
        
        return df
        
    except Exception as e:
        logger.error(f"Error in input validation: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid input format: {str(e)}")

def get_feature_importance(model, input_data: pd.DataFrame) -> Dict[str, float]:
    """Calculate SHAP values for feature importance."""
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_data)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # For binary classification, use class 1
            
        importance_dict = dict(zip(feature_names, abs(shap_values[0])))
        return dict(sorted(importance_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:10])
        
    except Exception as e:
        logger.error(f"Error calculating feature importance: {str(e)}")
        return {}

def generate_prediction_explanation(prediction: int, probability: float, top_features: Dict[str, float]) -> str:
    """Generate a human-readable explanation of the prediction."""
    health_status = "healthy" if prediction == 0 else "non-healthy"
    confidence = probability if prediction == 1 else 1 - probability
    
    explanation = f"The microbiome profile suggests a {health_status} status with {confidence:.1%} confidence. "
    
    if top_features:
        top_3_features = list(top_features.items())[:3]
        explanation += "The most influential features were: "
        explanation += ", ".join(f"{feature} (impact: {abs(value):.3f})" 
                               for feature, value in top_3_features)
    
    return explanation

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: MicrobiomeData):
    """Endpoint for making predictions."""
    try:
        # Validate and prepare input
        input_df = validate_input_features(data.abundances)
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        
        # Get feature importance
        feature_importance = get_feature_importance(model, input_df)
        
        # Generate explanation
        explanation = generate_prediction_explanation(prediction, probability, feature_importance)
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            feature_importance=feature_importance,
            prediction_explanation=explanation
        )
    
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/features")
async def get_features():
    """Get list of accepted features."""
    return {"features": list(feature_names)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 