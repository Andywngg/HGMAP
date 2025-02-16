"""
Microbiome Disease Prediction Pipeline
A comprehensive pipeline for microbiome-based disease prediction.
"""

from pathlib import Path

# Set project root directory
ROOT_DIR = Path(__file__).parent.parent

# Import main components
from .features.diversity_metrics import DiversityCalculator
from .training.advanced_model import AdvancedModelTrainer
from .monitoring.model_monitor import ModelMonitor
from .api.prediction_api import app

__version__ = "1.0.0"
__all__ = ['DiversityCalculator', 'AdvancedModelTrainer', 'ModelMonitor', 'app']

"""
HGMAP - AI-Based Microbiome and Gut Health Analysis for Early Disease Detection
"""

__version__ = "0.1.0" 