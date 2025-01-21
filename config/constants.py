import os

# Configuration for file paths
BASE_DATA_DIR = "path/to/your/data/directory"  # Update this with your actual data directory

# Taxonomy data paths
TAXONOMY_FILES = {
    "SupplementaryData1": os.path.join(BASE_DATA_DIR, "SupplementaryData1.xlsx"),
    "SupplementaryData2": os.path.join(BASE_DATA_DIR, "SupplementaryData2.xlsx"),
    # Add more files as needed
}

# Define target column (you'll need to specify this based on your specific dataset)
TARGET_COLUMN = "GMWI2"  # Based on the Gut Microbiome Wellness Index 2 you mentioned

# Optional: Add configuration for model parameters
MODEL_PARAMS = {
    "random_forest_estimators": 300,
    "gradient_boosting_estimators": 200,
    "random_state": 42,
    "cv_splits": 5
}