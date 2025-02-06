"""
SHAP-based model interpretation module for explaining predictions.
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Union
import joblib

class ShapExplainer:
    """Model interpretation using SHAP values."""
    
    def __init__(self, models_dir: Union[str, Path]):
        """
        Initialize SHAP explainer.
        
        Args:
            models_dir: Directory containing trained models
        """
        self.models_dir = Path(models_dir)
        self.explainers = {}
        self.shap_values = {}
        
    def load_models(self):
        """Load trained models for interpretation."""
        model_files = list(self.models_dir.glob("*_model.joblib"))
        self.models = {}
        
        for model_file in model_files:
            model_name = model_file.stem.replace("_model", "")
            self.models[model_name] = joblib.load(model_file)
    
    def calculate_shap_values(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        sample_size: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Calculate SHAP values for all models.
        
        Args:
            X: Features to explain
            feature_names: Optional list of feature names
            sample_size: Number of background samples for SHAP
            
        Returns:
            Dictionary of SHAP values for each model
        """
        # Load scaler
        scaler = joblib.load(self.models_dir / "scaler.joblib")
        X_scaled = scaler.transform(X)
        
        # Sample background data if needed
        if X_scaled.shape[0] > sample_size:
            background_inds = np.random.choice(X_scaled.shape[0], sample_size, replace=False)
            background = X_scaled[background_inds]
        else:
            background = X_scaled
        
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                # Create explainer
                if hasattr(model, 'feature_importances_'):  # Tree-based models
                    explainer = shap.TreeExplainer(model)
                else:  # Other models
                    explainer = shap.KernelExplainer(
                        model.predict_proba, background
                    )
                
                self.explainers[name] = explainer
                self.shap_values[name] = explainer.shap_values(X_scaled)
                
                # Generate visualization
                if feature_names is not None:
                    self._generate_shap_plots(name, X_scaled, feature_names)
        
        return self.shap_values
    
    def _generate_shap_plots(
        self,
        model_name: str,
        X: np.ndarray,
        feature_names: List[str]
    ):
        """Generate and save SHAP visualization plots."""
        plots_dir = self.models_dir / "shap_plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Summary plot
        plt.figure(figsize=(10, 6))
        shap_values = self.shap_values[model_name]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # For binary classification
            
        shap.summary_plot(
            shap_values,
            X,
            feature_names=feature_names,
            show=False
        )
        plt.tight_layout()
        plt.savefig(plots_dir / f"{model_name}_summary_plot.png")
        plt.close()
        
        # Feature importance plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            X,
            feature_names=feature_names,
            plot_type="bar",
            show=False
        )
        plt.tight_layout()
        plt.savefig(plots_dir / f"{model_name}_importance_plot.png")
        plt.close()
    
    def explain_prediction(
        self,
        X: np.ndarray,
        feature_names: List[str],
        top_k: int = 10
    ) -> Dict[str, Dict]:
        """
        Explain a specific prediction.
        
        Args:
            X: Single sample to explain
            feature_names: List of feature names
            top_k: Number of top features to include
            
        Returns:
            Dictionary containing feature contributions
        """
        explanations = {}
        
        # Load scaler
        scaler = joblib.load(self.models_dir / "scaler.joblib")
        X_scaled = scaler.transform(X)
        
        for name, explainer in self.explainers.items():
            # Calculate SHAP values for this prediction
            shap_values = explainer.shap_values(X_scaled)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For binary classification
            
            # Get feature contributions
            contributions = {}
            for idx in np.argsort(np.abs(shap_values[0]))[-top_k:]:
                contributions[feature_names[idx]] = float(shap_values[0][idx])
            
            explanations[name] = {
                'feature_contributions': contributions,
                'base_value': float(explainer.expected_value if not isinstance(explainer.expected_value, list) 
                                  else explainer.expected_value[1])
            }
        
        return explanations 