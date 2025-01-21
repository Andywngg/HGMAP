import shap
import matplotlib.pyplot as plt
import joblib
import pandas as pd

# Load model and data
stacked_model = joblib.load("stacked_model.pkl")
data = pd.read_csv("processed_microbiome_data.csv")
X_test = data.drop(columns=["target_column"])  # Replace with actual target column

# SHAP analysis
explainer = shap.Explainer(stacked_model, X_test)
shap_values = explainer(X_test)

# Global interpretability
shap.summary_plot(shap_values, X_test)

# Local interpretability for selected features
for feature in ["Shannon_Index", "Richness", "Evenness"]:
    shap.dependence_plot(feature, shap_values.values, X_test)
