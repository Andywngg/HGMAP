from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import logging
import json
from src.data.processor_final import MicrobiomeProcessor

app = FastAPI(
    title="Microbiome Analysis API",
    description="API for microbiome analysis and disease prediction",
    version="1.0.0"
)

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the trained model
try:
    model = joblib.load("data/results/best_model.joblib")
    with open("data/results/model_evaluation.json", "r") as f:
        model_metrics = json.load(f)
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None
    model_metrics = None

class PredictionInput(BaseModel):
    """Input data schema for predictions"""
    features: Dict[str, float]
    sample_id: Optional[str] = None

class PredictionResponse(BaseModel):
    """Response schema for predictions"""
    sample_id: Optional[str]
    prediction: int
    probability: float
    prediction_explanation: Dict[str, Any]

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to the Microbiome Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "Make predictions on new samples",
            "/metrics": "Get model performance metrics",
            "/visualizations": "Get model visualization files",
            "/model-info": "Get detailed model information"
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: PredictionInput):
    """Make predictions on new samples
    
    Args:
        input_data: Input features for prediction
        
    Returns:
        Prediction results with probability and explanation
    """
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
            
        # Convert input features to DataFrame
        features_df = pd.DataFrame([input_data.features])
        
        # Make prediction
        prediction = int(model.predict(features_df)[0])
        probability = float(model.predict_proba(features_df)[0][1])
        
        # Get feature importance for explanation
        if hasattr(model, 'named_estimators_'):
            rf_model = model.named_estimators_['rf']
        else:
            rf_model = model
            
        importance = dict(zip(
            features_df.columns,
            rf_model.feature_importances_
        ))
        
        # Sort features by importance
        sorted_features = dict(sorted(
            importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5])
        
        return PredictionResponse(
            sample_id=input_data.sample_id,
            prediction=prediction,
            probability=probability,
            prediction_explanation={
                "top_features": sorted_features,
                "interpretation": "These features had the strongest influence on the prediction"
            }
        )
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get comprehensive model performance metrics"""
    try:
        if model_metrics is None:
            raise HTTPException(status_code=500, detail="Model metrics not loaded")
            
        return JSONResponse(content={
            "performance_metrics": {
                "accuracy": model_metrics["metrics"]["accuracy"],
                "roc_auc": model_metrics["metrics"]["roc_auc"],
                "precision": model_metrics["metrics"]["precision"],
                "recall": model_metrics["metrics"]["recall"],
                "f1": model_metrics["metrics"]["f1"]
            },
            "cross_validation": {
                "mean_score": model_metrics["mean_cv_score"],
                "std_score": model_metrics["std_cv_score"],
                "cv_scores": model_metrics["cv_scores"]
            },
            "classification_report": model_metrics["classification_report"]
        })
        
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/visualizations")
async def get_visualizations():
    """Get available model visualization files"""
    try:
        results_dir = Path("data/results")
        visualization_files = {
            "confusion_matrix": str(results_dir / "confusion_matrix.png"),
            "feature_importance": str(results_dir / "feature_importance.png")
        }
        
        # Check if files exist
        available_files = {}
        for name, path in visualization_files.items():
            if Path(path).exists():
                available_files[name] = f"/visualization/{name}"
                
        return JSONResponse(content={
            "available_visualizations": available_files,
            "message": "Use the specific visualization endpoints to download the files"
        })
        
    except Exception as e:
        logger.error(f"Error getting visualizations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/visualization/{viz_name}")
async def get_visualization(viz_name: str):
    """Get a specific visualization file
    
    Args:
        viz_name: Name of the visualization file
        
    Returns:
        The visualization file
    """
    try:
        viz_path = Path(f"data/results/{viz_name}.png")
        if not viz_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Visualization {viz_name} not found"
            )
            
        return FileResponse(
            str(viz_path),
            media_type="image/png",
            filename=f"{viz_name}.png"
        )
        
    except Exception as e:
        logger.error(f"Error getting visualization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
async def get_model_info():
    """Get detailed model information"""
    try:
        if model_metrics is None:
            raise HTTPException(status_code=500, detail="Model information not loaded")
            
        return JSONResponse(content={
            "model_parameters": model_metrics["best_params"],
            "feature_importance": model_metrics["feature_importance"],
            "dataset_info": {
                "n_samples": model_metrics["n_samples"],
                "n_features": model_metrics["n_features"]
            },
            "performance_summary": {
                "best_score": model_metrics["best_score"],
                "cross_validation": {
                    "mean": model_metrics["mean_cv_score"],
                    "std": model_metrics["std_cv_score"]
                }
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)