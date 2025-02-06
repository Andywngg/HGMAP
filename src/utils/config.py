from pathlib import Path
import yaml
from typing import Dict, Any
import logging
from pydantic import BaseModel, Field
from typing import List, Optional

logger = logging.getLogger(__name__)

class DataConfig(BaseModel):
    input_dir: str = Field(..., description="Directory containing raw input data")
    processed_dir: str = Field(..., description="Directory for processed data")
    features_dir: str = Field(..., description="Directory for engineered features")
    models_dir: str = Field(..., description="Directory for saved models")
    results_dir: str = Field(..., description="Directory for results and visualizations")

class FeatureEngineeringConfig(BaseModel):
    n_pca_components: int = Field(50, description="Number of PCA components")
    min_prevalence: float = Field(0.1, description="Minimum prevalence threshold")
    min_abundance: float = Field(0.001, description="Minimum abundance threshold")
    taxonomy_levels: List[str] = Field(
        ["phylum", "class", "order", "family", "genus", "species"],
        description="Taxonomy levels to consider"
    )
    diversity_metrics: List[str] = Field(
        [
            "richness",
            "shannon_diversity",
            "simpson_diversity",
            "pielou_evenness",
            "berger_parker_dominance",
            "effective_species"
        ],
        description="Diversity metrics to calculate"
    )

class BaseModelConfig(BaseModel):
    n_estimators: int = Field(500, description="Number of estimators")
    max_depth: int = Field(..., description="Maximum tree depth")
    learning_rate: Optional[float] = Field(None, description="Learning rate for boosting")
    subsample: Optional[float] = Field(None, description="Subsample ratio")
    min_samples_split: Optional[int] = Field(None, description="Minimum samples for split")
    min_samples_leaf: Optional[int] = Field(None, description="Minimum samples in leaf")
    class_weight: Optional[str] = Field(None, description="Class weight strategy")

class ModelConfig(BaseModel):
    n_folds: int = Field(5, description="Number of cross-validation folds")
    random_state: int = Field(42, description="Random seed")
    use_probabilities: bool = Field(True, description="Use probability predictions")
    base_models: Dict[str, BaseModelConfig] = Field(..., description="Base model configurations")
    meta_model: Dict[str, Any] = Field(..., description="Meta-model configuration")

class TrainingConfig(BaseModel):
    test_size: float = Field(0.2, description="Test set size")
    cv_folds: int = Field(5, description="Number of cross-validation folds")
    scoring: str = Field("roc_auc", description="Scoring metric")
    use_smote: bool = Field(True, description="Whether to use SMOTE")
    smote_params: Dict[str, Any] = Field(..., description="SMOTE parameters")

class EvaluationConfig(BaseModel):
    metrics: List[str] = Field(..., description="Evaluation metrics")
    threshold: float = Field(0.5, description="Classification threshold")

class VisualizationConfig(BaseModel):
    feature_importance: Dict[str, Any] = Field(..., description="Feature importance plot settings")
    shap: Dict[str, Any] = Field(..., description="SHAP plot settings")

class LoggingConfig(BaseModel):
    level: str = Field("INFO", description="Logging level")
    format: str = Field(..., description="Log message format")
    file: str = Field(..., description="Log file path")

class PipelineConfig(BaseModel):
    data: DataConfig
    feature_engineering: FeatureEngineeringConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    visualization: VisualizationConfig
    logging: LoggingConfig

def load_config(config_path: Path) -> PipelineConfig:
    """Load and validate configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config = PipelineConfig(**config_dict)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise

def create_directories(config: PipelineConfig) -> None:
    """Create necessary directories from configuration."""
    directories = [
        config.data.input_dir,
        config.data.processed_dir,
        config.data.features_dir,
        config.data.models_dir,
        config.data.results_dir,
        str(Path(config.logging.file).parent)
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {directory}")

def setup_logging(config: LoggingConfig) -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, config.level.upper()),
        format=config.format,
        handlers=[
            logging.FileHandler(config.file),
            logging.StreamHandler()
        ]
    )
    logger.info("Logging configured") 