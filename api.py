from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize FastAPI app
app = FastAPI(
    title="Microbiome Health Analysis API",
    description="API for predicting health status based on microbiome data",
    version="1.0.0"
)

# Define input data model
class MicrobiomeData(BaseModel):
    shannon_diversity: float
    species_richness: float
    evenness: float
    abundance_data: dict[str, float]

class HealthPrediction(BaseModel):
    prediction: str
    probability: float
    model_used: str

class ModelArtifacts:
    def __init__(self):
        self.models_dir = Path("models")
        self.load_artifacts()
    
    def load_artifacts(self):
        """Load all necessary model artifacts."""
        try:
            # Load models
            self.rf_model = joblib.load(self.models_dir / "random_forest_model.joblib")
            self.gb_model = joblib.load(self.models_dir / "gradient_boosting_model.joblib")
            self.stacking_model = joblib.load(self.models_dir / "stacking_model.joblib")
            
            # Load scaler
            self.scaler = joblib.load(self.models_dir / "scaler.joblib")
            
            # Load feature names
            self.feature_names = pd.read_csv(self.models_dir / "feature_names.csv")['feature_name'].tolist()
            
            logging.info("Model artifacts loaded successfully")
        except Exception as e:
            logging.error(f"Error loading model artifacts: {str(e)}")
            raise RuntimeError("Failed to load model artifacts")

# Initialize model artifacts
artifacts = ModelArtifacts()

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Microbiome Health Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Make health predictions",
            "/models/info": "GET - Get model information"
        }
    }

@app.get("/models/info")
async def model_info():
    """Get information about the available models."""
    return {
        "models": [
            {
                "name": "Random Forest",
                "features": len(artifacts.feature_names),
                "type": "Ensemble"
            },
            {
                "name": "Gradient Boosting",
                "features": len(artifacts.feature_names),
                "type": "Ensemble"
            },
            {
                "name": "Stacking Classifier",
                "features": len(artifacts.feature_names),
                "type": "Meta-Ensemble"
            }
        ],
        "features": artifacts.feature_names
    }

@app.post("/predict", response_model=HealthPrediction)
async def predict(data: MicrobiomeData):
    """Make health predictions based on microbiome data."""
    try:
        # Prepare feature vector
        features = []
        
        # Add diversity metrics
        features.extend([
            data.shannon_diversity,
            data.species_richness,
            data.evenness
        ])
        
        # Add PCA components (zeros for now, as we need the full pipeline to compute them)
        features.extend([0] * 50)  # Placeholder for PCA components
        
        # Convert to numpy array and reshape
        X = np.array(features).reshape(1, -1)
        
        # Scale features
        X_scaled = artifacts.scaler.transform(X)
        
        # Make predictions with all models
        rf_pred = artifacts.rf_model.predict_proba(X_scaled)[0]
        gb_pred = artifacts.gb_model.predict_proba(X_scaled)[0]
        stack_pred = artifacts.stacking_model.predict_proba(X_scaled)[0]
        
        # Use the model with highest confidence
        predictions = [
            ("Random Forest", rf_pred),
            ("Gradient Boosting", gb_pred),
            ("Stacking Classifier", stack_pred)
        ]
        
        # Select model with highest confidence
        model_name, probs = max(predictions, key=lambda x: max(x[1]))
        prediction = "Non-healthy" if probs[1] > 0.5 else "Healthy"
        probability = max(probs)
        
        return HealthPrediction(
            prediction=prediction,
            probability=float(probability),
            model_used=model_name
        )
    
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 