import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import json
import gzip
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MicrobiomeDataProcessor:
    """Process and prepare microbiome data for analysis"""
    
    def __init__(
        self,
        data_dir: str = "data",
        min_prevalence: float = 0.1,
        min_abundance: float = 0.001,
        random_state: int = 42,
        cache_dir: str = "data/cache"
    ):
        self.data_dir = Path(data_dir)
        self.min_prevalence = min_prevalence
        self.min_abundance = min_abundance
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_names_ = None
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.label_encoders = {}
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
    def load_data(self, data_path: Path) -> pd.DataFrame:
        """Load data from various formats"""
        if data_path.suffix == '.csv':
            return pd.read_csv(data_path)
        elif data_path.suffix == '.gz':
            with gzip.open(data_path, 'rt') as f:
                if str(data_path).endswith('.csv.gz'):
                    return pd.read_csv(f)
                else:
                    return pd.read_csv(f, sep='\t')
        elif data_path.suffix == '.tsv':
            return pd.read_csv(data_path, sep='\t')
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
        abundance_path = Path(data_dir) / "abundance.tsv.gz"
        metadata_path = Path(data_dir) / "metadata.json"
        
        abundances = self.load_data(abundance_path)
        
        # Load metadata
        with open(metadata_path) as f:
            metadata = pd.DataFrame(json.load(f))
        
        return self._process_data(abundances, metadata)
    
    def load_hmp(self, data_dir: str, version: str = "1") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and process Human Microbiome Project data"""
        logger.info(f"Loading HMP{version} data...")
        
        abundance_path = Path(data_dir) / f"hmp{version}" / "abundance.txt.gz"
        metadata_path = Path(data_dir) / f"hmp{version}" / "metadata.csv"
        
        abundances = self.load_data(abundance_path)
        metadata = pd.read_csv(metadata_path)
        
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
        
        # Extract abundances and metadata
        all_abundances = []
        all_metadata = []
        
        for abundances, metadata in datasets:
            all_abundances.append(abundances)
            all_metadata.append(metadata)
        
        # Combine abundances
        combined_abundances = pd.concat(all_abundances, axis=0)
        
        # Filter features present in at least feature_threshold of datasets
        feature_counts = pd.Series(
            [set(df.columns) for df in all_abundances]
        ).value_counts()
        keep_features = feature_counts[
            feature_counts >= len(datasets) * feature_threshold
        ].index
        
        combined_abundances = combined_abundances[keep_features]
        
        # Combine metadata
        combined_metadata = pd.concat(all_metadata, axis=0)
        
        return combined_abundances, combined_metadata

    def load_abundance_data(self, dataset):
        """Load abundance data for a given dataset."""
        self.logger.info(f"Loading abundance data for {dataset}")
        
        try:
            if dataset == "agp":
                path = self.data_dir / "american_gut" / "abundance.biom"
            elif dataset.startswith("hmp"):
                version = dataset[3:]  # Extract 1 or 2
                path = self.data_dir / "hmp" / f"hmp{version}" / "abundance.txt.gz"
            elif dataset == "metahit":
                path = self.data_dir / "metahit" / "abundance.tsv.gz"
            elif dataset.startswith("gmrepo"):
                condition = dataset.split("_")[1]  # Extract healthy or disease
                path = self.data_dir / "gmrepo" / f"{condition}_abundance.tsv.gz"
            elif dataset == "diabimmune":
                path = self.data_dir / "diabimmune" / "abundance.tsv.gz"
            elif dataset == "ibd":
                path = self.data_dir / "ibd" / "abundance.tsv.gz"
            elif dataset.startswith("qiita"):
                study = dataset.split("_")[1]  # Extract obesity or ibd
                path = self.data_dir / "qiita" / study / "abundance.biom"
            else:
                raise ValueError(f"Unknown dataset: {dataset}")
            
            if not path.exists():
                raise FileNotFoundError(f"Abundance data not found at {path}")
            
            # Load the data based on file extension
            if str(path).endswith('.gz'):
                with gzip.open(path, 'rt') as f:
                    df = pd.read_csv(f, sep='\t', index_col=0)
            elif str(path).endswith('.biom'):
                # For BIOM files, we expect them to be converted to TSV format during download
                df = pd.read_csv(path, sep='\t', index_col=0)
            else:
                df = pd.read_csv(path, sep='\t', index_col=0)
            
            return df
        
        except Exception as e:
            self.logger.error(f"Error loading abundance data for {dataset}: {str(e)}")
            raise

    def load_metadata(self, dataset):
        """Load metadata for a given dataset."""
        self.logger.info(f"Loading metadata for {dataset}")
        
        try:
            if dataset == "agp":
                path = self.data_dir / "american_gut" / "metadata.tsv"
            elif dataset.startswith("hmp"):
                version = dataset[3:]
                path = self.data_dir / "hmp" / f"hmp{version}" / "metadata.csv"
            elif dataset == "metahit":
                path = self.data_dir / "metahit" / "metadata.tsv"
            elif dataset.startswith("gmrepo"):
                condition = dataset.split("_")[1]
                path = self.data_dir / "gmrepo" / f"{condition}_metadata.tsv"
            elif dataset == "diabimmune":
                path = self.data_dir / "diabimmune" / "metadata.tsv"
            elif dataset == "ibd":
                path = self.data_dir / "ibd" / "metadata.tsv"
            elif dataset.startswith("qiita"):
                study = dataset.split("_")[1]
                path = self.data_dir / "qiita" / study / "metadata.tsv"
            else:
                raise ValueError(f"Unknown dataset: {dataset}")
            
            if not path.exists():
                raise FileNotFoundError(f"Metadata not found at {path}")
            
            # Load metadata based on file extension
            if str(path).endswith('.csv'):
                df = pd.read_csv(path)
            else:
                df = pd.read_csv(path, sep='\t')
            
            return df
        
        except Exception as e:
            self.logger.error(f"Error loading metadata for {dataset}: {str(e)}")
            raise

    def preprocess_abundance_data(self, abundance_df, min_prevalence=0.1):
        """Preprocess abundance data."""
        if abundance_df is None or abundance_df.empty:
            return None
        
        # Remove low prevalence features
        prevalence = (abundance_df > 0).mean()
        keep_features = prevalence[prevalence >= min_prevalence].index
        filtered_df = abundance_df[keep_features]
        
        # Log transform
        transformed_df = np.log1p(filtered_df)
        
        # Normalize
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(transformed_df)
        normalized_df = pd.DataFrame(normalized_data, 
                                   index=transformed_df.index,
                                   columns=transformed_df.columns)
        return normalized_df

    def reduce_dimensions(self, data, n_components=50):
        """Reduce dimensionality using PCA."""
        if data is None or data.empty:
            return None
        
        pca = PCA(n_components=min(n_components, data.shape[1]))
        reduced_data = pca.fit_transform(data)
        reduced_df = pd.DataFrame(reduced_data,
                                index=data.index,
                                columns=[f'PC{i+1}' for i in range(reduced_data.shape[1])])
        return reduced_df, pca.explained_variance_ratio_

    def handle_class_imbalance(self, features, target):
        """Handle class imbalance using SMOTE."""
        if features is None or target is None:
            return None, None
        
        try:
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(features, target)
            return X_resampled, y_resampled
        except Exception as e:
            logger.error(f"Error applying SMOTE: {str(e)}")
            return features, target

    def integrate_datasets(self, datasets=None):
        """Integrate multiple datasets."""
        try:
            all_features = []
            all_targets = []
            
            # Load and process EBI data
            if 'ebi' in datasets:
                abundance_matrix = self.load_ebi_data()
                if abundance_matrix is not None:
                    processed_matrix = self.preprocess_abundance_data(abundance_matrix)
                    if processed_matrix is not None:
                        features = processed_matrix.T
                        # Create random targets for now - replace with actual phenotypes
                        targets = pd.Series(np.random.binomial(1, 0.5, features.shape[0]), 
                                         index=features.index,
                                         name='target')
                        all_features.append(features)
                        all_targets.append(targets)
                        self.logger.info(f"Added EBI dataset with {features.shape[0]} samples")
            
            # Load and process MG-RAST data
            if 'mgrast' in datasets:
                abundance_matrix, metadata = self.load_mgrast_data()
                if abundance_matrix is not None:
                    processed_matrix = self.preprocess_abundance_data(abundance_matrix)
                    if processed_matrix is not None:
                        features = processed_matrix.T
                        # Create random targets for now - replace with actual phenotypes
                        targets = pd.Series(np.random.binomial(1, 0.5, features.shape[0]), 
                                         index=features.index,
                                         name='target')
                        all_features.append(features)
                        all_targets.append(targets)
                        self.logger.info(f"Added MG-RAST dataset with {features.shape[0]} samples")
            
            if not all_features:
                raise ValueError("No datasets were successfully processed")
            
            # Combine all features and targets
            combined_features = pd.concat(all_features, axis=0)
            combined_targets = pd.concat(all_targets, axis=0)
            
            # Ensure feature columns match across datasets
            common_features = set.intersection(*[set(df.columns) for df in all_features])
            if not common_features:
                raise ValueError("No common features found across datasets")
            
            combined_features = combined_features[list(common_features)]
            self.logger.info(f"Combined dataset has {combined_features.shape[0]} samples and {combined_features.shape[1]} features")
            
            return combined_features, combined_targets
            
        except Exception as e:
            self.logger.error(f"Error integrating datasets: {str(e)}")
            return None, None

    def prepare_features_for_training(self, datasets=None):
        """Prepare features for model training."""
        try:
            if datasets is None:
                datasets = ['ebi', 'mgrast']  # Default datasets to use
            
            # Integrate datasets
            features, targets = self.integrate_datasets(datasets)
            if features is None or targets is None:
                raise ValueError("Failed to integrate datasets")
            
            # Handle class imbalance
            smote = SMOTE(random_state=42)
            balanced_features, balanced_targets = smote.fit_resample(features, targets)
            
            # Convert back to DataFrame/Series
            balanced_features = pd.DataFrame(balanced_features, 
                                          columns=features.columns,
                                          index=range(len(balanced_features)))
            balanced_targets = pd.Series(balanced_targets, 
                                       index=range(len(balanced_targets)),
                                       name='target')
            
            # Save processed data
            self.save_processed_data(balanced_features, balanced_targets)
            
            return balanced_features, balanced_targets
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {str(e)}")
            return None, None

    def load_ebi_taxonomic_data(self, ssu_assignments_path: Path, phylum_level_path: Path = None) -> pd.DataFrame:
        """Load taxonomic assignments from EBI format files."""
        self.logger.info("Loading EBI taxonomic data...")
        
        try:
            # Load the matrix directly - it's already in the right format
            # First row contains sample IDs, first column contains taxa
            abundance_matrix = pd.read_csv(ssu_assignments_path, sep='\t', index_col=0)
            
            # Log some information about the data
            self.logger.info(f"Loaded abundance matrix with {abundance_matrix.shape[0]} taxa and {abundance_matrix.shape[1]} samples")
            
            # If phylum level data is provided, use it for validation
            if phylum_level_path is not None and phylum_level_path.exists():
                try:
                    phylum_matrix = pd.read_csv(phylum_level_path, sep='\t', index_col=0)
                    self.logger.info(f"Loaded phylum matrix with {phylum_matrix.shape[0]} taxa and {phylum_matrix.shape[1]} samples")
                    
                    # Validate that sample totals match between detailed and phylum level
                    detailed_sums = abundance_matrix.sum()
                    phylum_sums = phylum_matrix.sum()
                    
                    if not np.allclose(detailed_sums, phylum_sums, rtol=0.1):
                        self.logger.warning("Warning: Detailed and phylum level counts don't match exactly")
                except Exception as e:
                    self.logger.warning(f"Could not validate against phylum data: {str(e)}")
            
            return abundance_matrix
        
        except Exception as e:
            self.logger.error(f"Error loading EBI taxonomic data: {str(e)}")
            raise

    def load_ebi_data(self):
        """Load EBI taxonomic data."""
        try:
            ssu_path = self.data_dir / "ebi" / "taxonomic_assignments.tsv"
            phylum_path = self.data_dir / "ebi" / "phylum_level.tsv"
            
            if not ssu_path.exists() or not phylum_path.exists():
                self.logger.error("EBI data files not found in expected location")
                return None
                
            abundance_matrix = self.load_ebi_taxonomic_data(ssu_path, phylum_path)
            return abundance_matrix
            
        except Exception as e:
            self.logger.error(f"Error loading EBI data: {str(e)}")
            return None

    def preprocess_abundance_data(self, abundance_matrix):
        """Preprocess the abundance data."""
        try:
            # Remove features (taxa) with too many zeros
            min_presence = 0.1  # Present in at least 10% of samples
            presence = (abundance_matrix > 0).mean(axis=1)
            filtered_matrix = abundance_matrix[presence >= min_presence]
            
            # Log transform the counts
            transformed_matrix = np.log1p(filtered_matrix)
            
            # Scale the features
            scaler = StandardScaler()
            scaled_matrix = pd.DataFrame(
                scaler.fit_transform(transformed_matrix.T).T,
                index=transformed_matrix.index,
                columns=transformed_matrix.columns
            )
            
            return scaled_matrix
            
        except Exception as e:
            self.logger.error(f"Error preprocessing abundance data: {str(e)}")
            return None

    def load_mgrast_data(self):
        """Load MG-RAST data."""
        try:
            mgrast_dir = self.data_dir / "mgrast"
            abundance_path = mgrast_dir / "abundance.tsv"
            metadata_path = mgrast_dir / "metadata.tsv"
            
            if not abundance_path.exists() or not metadata_path.exists():
                self.logger.error("MG-RAST data files not found in expected location")
                return None, None
                
            # Load abundance data
            abundance_matrix = pd.read_csv(abundance_path, sep='\t', index_col=0)
            self.logger.info(f"Loaded MG-RAST abundance matrix with {abundance_matrix.shape[0]} taxa and {abundance_matrix.shape[1]} samples")
            
            # Load metadata
            metadata = pd.read_csv(metadata_path, sep='\t', index_col=0)
            self.logger.info(f"Loaded MG-RAST metadata for {len(metadata)} samples")
            
            return abundance_matrix, metadata
            
        except Exception as e:
            self.logger.error(f"Error loading MG-RAST data: {str(e)}")
            return None, None

    def load_mgnify_abundance(self, study_id):
        """Load abundance data from MGnify study."""
        try:
            # Find the abundance file in the study directory
            study_dir = self.data_dir / "mgnify" / study_id
            abundance_files = list(study_dir.glob("*_taxonomy_abundances_*.tsv"))
            if not abundance_files:
                self.logger.error(f"No abundance file found for study {study_id}")
                return None
            
            # Load the abundance table
            df = pd.read_csv(abundance_files[0], sep='\t')
            df.set_index('#SampleID', inplace=True)
            return df
        except Exception as e:
            self.logger.error(f"Error loading abundance data for study {study_id}: {e}")
            return None

    def load_mgnify_metadata(self, study_id):
        """Load metadata from MGnify study."""
        try:
            # Load the samples.tsv file
            samples_file = self.data_dir / "mgnify" / study_id / "samples.tsv"
            df = pd.read_csv(samples_file, sep='\t')
            
            # Extract relevant information from the attributes JSON
            metadata = []
            for _, row in df.iterrows():
                attrs = json.loads(row['attributes'].replace("'", '"'))
                sample_metadata = {
                    'sample_id': row['id'],
                    'biosample': attrs.get('biosample'),
                    'sample_desc': attrs.get('sample-desc'),
                    'species': attrs.get('species'),
                }
                
                # Extract additional metadata from sample-metadata list
                for meta in attrs.get('sample-metadata', []):
                    key = meta['key'].replace(' ', '_')
                    sample_metadata[key] = meta['value']
                
                metadata.append(sample_metadata)
            
            return pd.DataFrame(metadata)
        except Exception as e:
            self.logger.error(f"Error loading metadata for study {study_id}: {e}")
            return None

    def preprocess_abundance_data(self, abundance_df):
        """Preprocess abundance data."""
        try:
            if abundance_df is None or abundance_df.empty:
                return None
            
            # Remove rows with low abundance across all samples
            abundance_df = abundance_df.loc[abundance_df.sum(axis=1) > 0]
            
            # Log transform the data
            abundance_df = np.log1p(abundance_df)
            
            # Scale the features
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(abundance_df.T)
            
            return pd.DataFrame(scaled_data, index=abundance_df.columns, columns=abundance_df.index)
        except Exception as e:
            self.logger.error(f"Error preprocessing abundance data: {e}")
            return None

    def prepare_features(self, study_id):
        """Prepare features for training from a specific study."""
        try:
            # Load and preprocess abundance data
            abundance_df = self.load_mgnify_abundance(study_id)
            if abundance_df is None:
                return None, None
            
            processed_df = self.preprocess_abundance_data(abundance_df)
            if processed_df is None:
                return None, None
            
            # Load metadata
            metadata_df = self.load_mgnify_metadata(study_id)
            if metadata_df is None:
                return None, None
            
            # Create binary labels based on sample description
            # Assuming 'sample_desc' contains information about health status
            metadata_df['target'] = metadata_df['sample_desc'].str.contains('disease|infection|disorder', 
                                                                          case=False, 
                                                                          regex=True).astype(int)
            
            # Ensure samples match between abundance and metadata
            common_samples = set(processed_df.index) & set(metadata_df['sample_id'])
            if not common_samples:
                self.logger.error("No common samples between abundance and metadata")
                return None, None
            
            processed_df = processed_df.loc[list(common_samples)]
            metadata_df = metadata_df[metadata_df['sample_id'].isin(common_samples)]
            
            return processed_df, metadata_df['target']
        except Exception as e:
            self.logger.error(f"Error preparing features for study {study_id}: {e}")
            return None, None

    def save_processed_data(self, features, targets, output_dir="processed"):
        """Save processed features and targets."""
        try:
            output_path = self.data_dir / output_dir
            output_path.mkdir(parents=True, exist_ok=True)
            
            features.to_csv(output_path / "features.csv")
            targets.to_csv(output_path / "targets.csv")
            self.logger.info(f"Saved processed data to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving processed data: {e}")

    def process_study(self, study_id):
        """Process a single study and return features and targets."""
        self.logger.info(f"Processing study {study_id}")
        features, targets = self.prepare_features(study_id)
        
        if features is not None and targets is not None:
            self.logger.info(f"Successfully processed study {study_id}")
            self.logger.info(f"Features shape: {features.shape}")
            self.logger.info(f"Number of positive samples: {sum(targets)}")
            return features, targets
        else:
            self.logger.error(f"Failed to process study {study_id}")
            return None, None

    def process_all_studies(self, study_ids):
        """Process all specified studies and combine the data."""
        all_features = []
        all_targets = []
        
        for study_id in study_ids:
            features, targets = self.process_study(study_id)
            if features is not None and targets is not None:
                all_features.append(features)
                all_targets.append(targets)
        
        if not all_features:
            self.logger.error("No studies were successfully processed")
            return None, None
        
        # Combine all features and targets
        combined_features = pd.concat(all_features, axis=0)
        combined_targets = pd.concat(all_targets, axis=0)
        
        # Handle class imbalance using SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(combined_features, combined_targets)
        
        # Convert back to DataFrame
        balanced_features = pd.DataFrame(X_resampled, index=range(len(X_resampled)), 
                                       columns=combined_features.columns)
        balanced_targets = pd.Series(y_resampled, index=range(len(y_resampled)))
        
        # Save the processed data
        self.save_processed_data(balanced_features, balanced_targets)
        
        return balanced_features, balanced_targets 