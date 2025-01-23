import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import zscore

logger = logging.getLogger(__name__)

class MicrobiomeDataProcessor:
    """Process and prepare microbiome data for analysis"""
    
    def __init__(
        self,
        min_prevalence: float = 0.1,
        min_abundance: float = 0.001,
        random_state: int = 42
    ):
        self.min_prevalence = min_prevalence
        self.min_abundance = min_abundance
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_names_ = None
        
    def load_data(self, data_path: Path) -> pd.DataFrame:
        """Load data from various formats"""
        if data_path.suffix == '.csv':
            return pd.read_csv(data_path)
        elif data_path.suffix == '.biom':
            try:
                import biom
                table = biom.load_table(str(data_path))
                return pd.DataFrame(table.to_dataframe())
            except ImportError:
                logger.error("biom-format package required for .biom files")
                raise
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")
    
    def filter_taxa(self, X: pd.DataFrame) -> pd.DataFrame:
        """Filter taxa based on prevalence and abundance"""
        # Calculate prevalence and mean abundance
        prevalence = (X > 0).mean()
        mean_abundance = X.mean()
        
        # Filter based on thresholds
        keep_taxa = (prevalence >= self.min_prevalence) & (mean_abundance >= self.min_abundance)
        
        return X.loc[:, keep_taxa]
    
    def normalize_abundances(self, X: pd.DataFrame) -> pd.DataFrame:
        """Normalize abundance data"""
        # Add pseudo-count and perform log transformation
        X_norm = np.log1p(X + 1e-6)
        
        # Scale the data
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_norm),
            index=X_norm.index,
            columns=X_norm.columns
        )
        
        return X_scaled
    
    def process_metadata(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """Process metadata"""
        # Convert categorical variables
        categorical_cols = metadata.select_dtypes(include=['object']).columns
        metadata_processed = pd.get_dummies(metadata, columns=categorical_cols)
        
        # Handle numerical variables
        numerical_cols = metadata.select_dtypes(include=['float64', 'int64']).columns
        metadata_processed[numerical_cols] = metadata[numerical_cols].apply(zscore)
        
        return metadata_processed
    
    def prepare_data(
        self,
        abundance_path: Path,
        metadata_path: Optional[Path] = None,
        test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Prepare data for model training"""
        # Load abundance data
        X = self.load_data(abundance_path)
        
        # Filter and normalize abundances
        X = self.filter_taxa(X)
        X = self.normalize_abundances(X)
        self.feature_names_ = X.columns.tolist()
        
        # Load and process metadata if available
        metadata = None
        if metadata_path is not None:
            metadata = self.load_data(metadata_path)
            metadata = self.process_metadata(metadata)
        
        # Split data
        if metadata is not None:
            X_train, X_test, meta_train, meta_test = train_test_split(
                X, metadata,
                test_size=test_size,
                random_state=self.random_state
            )
            return X_train, X_test, meta_train, meta_test
        else:
            X_train, X_test = train_test_split(
                X,
                test_size=test_size,
                random_state=self.random_state
            )
            return X_train, X_test, None, None
    
    def save_processed_data(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        output_dir: Path,
        meta_train: Optional[pd.DataFrame] = None,
        meta_test: Optional[pd.DataFrame] = None
    ) -> None:
        """Save processed data"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        X_train.to_csv(output_dir / "abundance_train.csv")
        X_test.to_csv(output_dir / "abundance_test.csv")
        
        if meta_train is not None:
            meta_train.to_csv(output_dir / "metadata_train.csv")
        if meta_test is not None:
            meta_test.to_csv(output_dir / "metadata_test.csv")
        
        # Save feature names
        pd.Series(self.feature_names_).to_csv(
            output_dir / "feature_names.csv",
            index=False
        ) 