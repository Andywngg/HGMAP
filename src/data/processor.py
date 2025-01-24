import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
import biom
import skbio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MicrobiomeDataProcessor:
    """Process and prepare microbiome data for analysis"""
    
    def __init__(
        self,
        min_prevalence: float = 0.1,
        min_abundance: float = 0.001,
        random_state: int = 42,
        cache_dir: str = "data/cache"
    ):
        self.min_prevalence = min_prevalence
        self.min_abundance = min_abundance
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_names_ = None
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.label_encoders = {}
        
    def load_data(self, data_path: Path) -> pd.DataFrame:
        """Load data from various formats"""
        if data_path.suffix == '.csv':
            return pd.read_csv(data_path)
        elif data_path.suffix == '.biom':
            try:
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
    
    def load_american_gut(self, data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and process American Gut Project data"""
        logger.info("Loading American Gut Project data...")
        
        # Load abundance table
        abundance_path = Path(data_dir) / "abundance_table.biom"
        metadata_path = Path(data_dir) / "metadata.tsv"
        
        table = biom.load_table(str(abundance_path))
        abundances = pd.DataFrame(
            table.matrix_data.toarray(),
            index=table.ids(),
            columns=table.ids(axis='observation')
        )
        
        # Load metadata
        metadata = pd.read_csv(metadata_path, sep='\t', index_col=0)
        
        return self._process_data(abundances, metadata)
    
    def load_hmp(self, data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and process Human Microbiome Project data"""
        logger.info("Loading HMP data...")
        
        abundance_path = Path(data_dir) / "otu_table.biom"
        metadata_path = Path(data_dir) / "metadata.csv"
        
        table = biom.load_table(str(abundance_path))
        abundances = pd.DataFrame(
            table.matrix_data.toarray(),
            index=table.ids(),
            columns=table.ids(axis='observation')
        )
        
        metadata = pd.read_csv(metadata_path, index_col=0)
        
        return self._process_data(abundances, metadata)
    
    def _process_data(
        self,
        abundances: pd.DataFrame,
        metadata: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Process microbiome data"""
        # Filter low abundance features
        abundance_filter = abundances.sum() > abundances.sum().quantile(0.1)
        abundances = abundances.loc[:, abundance_filter]
        
        # Normalize abundances
        abundances = abundances.div(abundances.sum(axis=1), axis=0)
        
        # Log transform
        abundances = np.log1p(abundances)
        
        # Process metadata
        metadata = self._process_metadata(metadata)
        
        # Align samples
        common_samples = abundances.index.intersection(metadata.index)
        abundances = abundances.loc[common_samples]
        metadata = metadata.loc[common_samples]
        
        return abundances, metadata
    
    def _process_metadata(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """Process metadata"""
        # Select relevant columns
        relevant_columns = [
            'age', 'sex', 'bmi', 'disease_status',
            'antibiotics', 'diet_type'
        ]
        metadata = metadata[
            [col for col in relevant_columns if col in metadata.columns]
        ]
        
        # Handle categorical variables
        categorical_columns = metadata.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            metadata[col] = self.label_encoders[col].fit_transform(
                metadata[col].fillna('unknown')
            )
        
        # Handle numerical variables
        numerical_columns = metadata.select_dtypes(include=['float64', 'int64']).columns
        for col in numerical_columns:
            metadata[col] = metadata[col].fillna(metadata[col].median())
        
        return metadata
    
    def combine_datasets(
        self,
        datasets: List[Tuple[pd.DataFrame, pd.DataFrame]],
        feature_threshold: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Combine multiple datasets"""
        logger.info(f"Combining {len(datasets)} datasets...")
        
        all_abundances = []
        all_metadata = []
        
        # Combine abundances
        for abundances, metadata in datasets:
            all_abundances.append(abundances)
            all_metadata.append(metadata)
        
        combined_abundances = pd.concat(all_abundances, axis=0)
        combined_metadata = pd.concat(all_metadata, axis=0)
        
        # Filter features present in multiple datasets
        feature_presence = (combined_abundances > 0).sum() / len(combined_abundances)
        keep_features = feature_presence >= feature_threshold
        combined_abundances = combined_abundances.loc[:, keep_features]
        
        return combined_abundances, combined_metadata
    
    def save_processed_data_to_cache(
        self,
        abundances: pd.DataFrame,
        metadata: pd.DataFrame,
        dataset_name: str
    ):
        """Save processed data to cache"""
        cache_path = self.cache_dir / dataset_name
        cache_path.mkdir(parents=True, exist_ok=True)
        
        abundances.to_csv(cache_path / "abundances.csv")
        metadata.to_csv(cache_path / "metadata.csv")
        
    def load_processed_data(
        self,
        dataset_name: str
    ) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Load processed data from cache"""
        cache_path = self.cache_dir / dataset_name
        
        if not cache_path.exists():
            return None
            
        abundances = pd.read_csv(cache_path / "abundances.csv", index_col=0)
        metadata = pd.read_csv(cache_path / "metadata.csv", index_col=0)
        
        return abundances, metadata 