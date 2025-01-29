#!/usr/bin/env python
# -*- coding: utf-8 -*-

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Microbiome Disease Classifier",
    description="API for classifying diseases based on microbiome data",
    version="1.0.0"
)

class PredictionInput(BaseModel):
    """Input data model for predictions."""
    features: Dict[str, float]

class PredictionOutput(BaseModel):
    """Output data model for predictions."""
    predicted_class: str
    probabilities: Dict[str, float]
    feature_importance: Dict[str, float]

class ModelService:
    """Service for loading and using the trained model."""
    
    def __init__(self, model_dir: str = 'models'):
        self.model_dir = Path(model_dir)
        self.load_model()
        
    def load_model(self):
        """Load the trained model and associated artifacts."""
        try:
            self.model = joblib.load(self.model_dir / 'best_model.joblib')
            self.scalers = joblib.load(self.model_dir / 'scalers.joblib')
            
            # Load feature names and labels
            feature_importance = pd.read_csv(self.model_dir / 'feature_importance.csv')
            self.feature_names = feature_importance['feature'].tolist()
            
            # Load class labels if available
            labels_path = self.model_dir / 'labels.txt'
            if labels_path.exists():
                with open(labels_path, 'r') as f:
                    self.labels = [line.strip() for line in f]
            else:
                self.labels = [f"Class_{i}" for i in range(self.model.n_classes_)]
                
            logger.info("Model and artifacts loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise RuntimeError("Failed to load model")
    
    def prepare_features(self, features: Dict[str, float]) -> np.ndarray:
        """Prepare features for prediction."""
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Reorder columns to match training data
        df = df[self.feature_names]
        
        # Scale features
        X_scaled = self.scalers['standard'].transform(df)
        
        return X_scaled
    
    def get_feature_importance(self, features: np.ndarray) -> Dict[str, float]:
        """Calculate feature importance for the prediction."""
        import shap
        
        try:
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(features)
            
            # Handle multi-class case
            if isinstance(shap_values, list):
                # Take mean absolute value across classes
                shap_values = np.abs(np.array(shap_values)).mean(axis=0)
            
            # Create feature importance dictionary
            importance_dict = {
                name: float(abs(value))
                for name, value in zip(self.feature_names, shap_values[0])
            }
            
            return importance_dict
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            return {}
    
    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Make a prediction."""
        try:
            # Prepare features
            X = self.prepare_features(features)
            
            # Get prediction and probabilities
            y_pred = self.model.predict(X)
            y_pred_proba = self.model.predict_proba(X)
            
            # Get feature importance
            feature_importance = self.get_feature_importance(X)
            
            # Prepare response
            result = {
                'predicted_class': self.labels[y_pred[0]],
                'probabilities': {
                    label: float(prob)
                    for label, prob in zip(self.labels, y_pred_proba[0])
                },
                'feature_importance': feature_importance
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise

# Initialize model service
model_service = ModelService()

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Microbiome Disease Classifier API",
        "version": "1.0.0"
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """Make a prediction."""
    try:
        result = model_service.predict(input_data.features)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") 