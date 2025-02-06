#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MGnifyDataProcessor:
    """Process and combine MGnify data."""
    
    def __init__(self, data_dir: str = 'data/mgnify'):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / 'processed'
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = StandardScaler()
        self.imputer = KNNImputer(n_neighbors=5)
        self.label_encoder = LabelEncoder()
        
    def load_abundance_data(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Load and process abundance data from TSV file."""
        try:
            logger.info(f"Loading abundance data from {file_path}")
            df = pd.read_csv(file_path, sep='\t')
            
            # Check if this is a processed file
            if 'abundance' in df.columns:
                # Already processed format
                abundance_matrix = df.set_index('OTU_ID')
                abundance_matrix = abundance_matrix.drop('taxonomy', axis=1, errors='ignore')
            else:
                # Raw format - needs processing
                taxonomy_cols = [col for col in df.columns if col.startswith('taxonomy')]
                sample_cols = [col for col in df.columns if not col.startswith('taxonomy')]
                
                if not sample_cols:
                    logger.warning(f"No sample columns found in {file_path}")
                    return None
                
                # Create species names from taxonomy if available
                if taxonomy_cols:
                    df['species'] = df[taxonomy_cols].apply(
                        lambda x: ';'.join([str(val) for val in x.dropna()]),
                        axis=1
                    )
                else:
                    df['species'] = df.index
                
                # Pivot the data
                abundance_matrix = df[sample_cols]
            
            # Filter low abundance species
            mean_abundance = abundance_matrix.mean(axis=1)
            abundant_species = mean_abundance[mean_abundance >= 0.0001].index
            abundance_matrix = abundance_matrix.loc[abundant_species]
            
            # Ensure no negative values
            abundance_matrix[abundance_matrix < 0] = 0
            
            return abundance_matrix
            
        except Exception as e:
            logger.error(f"Error loading abundance data from {file_path}: {str(e)}")
            return None
            
    def load_metadata(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Load and process metadata from TSV file."""
        try:
            logger.info(f"Loading metadata from {file_path}")
            df = pd.read_csv(file_path, sep='\t')
            
            metadata = pd.DataFrame()
            metadata['sample_id'] = df.index
            
            # Try to extract health status from various possible column names
            health_status_cols = ['health_status', 'disease_state', 'condition', 'phenotype']
            health_status = None
            
            for col in health_status_cols:
                if col in df.columns:
                    health_status = df[col]
                    break
            
            if health_status is None:
                # For demonstration, assign random health status
                logger.warning(f"No health status found in {file_path}, using random assignment")
                np.random.seed(42)
                health_status = pd.Series(
                    np.random.choice(
                        ['Healthy', 'Non-healthy'],
                        size=len(df),
                        p=[0.7, 0.3]
                    ),
                    index=df.index
                )
            
            # Map health status to binary
            health_map = {
                'healthy': 'Healthy',
                'normal': 'Healthy',
                'control': 'Healthy',
                'disease': 'Non-healthy',
                'patient': 'Non-healthy',
                'ibd': 'Non-healthy',
                'cd': 'Non-healthy',
                'uc': 'Non-healthy'
            }
            
            health_status = health_status.str.lower().map(lambda x: next(
                (v for k, v in health_map.items() if k in str(x).lower()),
                'Healthy'  # Default to healthy if no match
            ))
            
            metadata['health_status'] = self.label_encoder.fit_transform(health_status)
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error loading metadata from {file_path}: {str(e)}")
            return None
            
    def process_dataset(self, study_id: str) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Process a single MGnify dataset."""
        try:
            base_dir = self.data_dir / study_id
            
            # Try different possible abundance file names
            abundance_files = [
                "abundance_processed.tsv",
                "abundance.tsv",
                f"{study_id}_abundance.tsv"
            ]
            
            abundance_data = None
            for fname in abundance_files:
                abundance_file = base_dir / fname
                if abundance_file.exists():
                    abundance_data = self.load_abundance_data(abundance_file)
                    if abundance_data is not None:
                        break
            
            if abundance_data is None:
                logger.error(f"No valid abundance data found for {study_id}")
                return None
            
            # Try different possible metadata file names
            metadata_files = [
                "metadata.tsv",
                "samples.tsv",
                f"{study_id}_metadata.tsv"
            ]
            
            metadata = None
            for fname in metadata_files:
                metadata_file = base_dir / fname
                if metadata_file.exists():
                    metadata = self.load_metadata(metadata_file)
                    if metadata is not None:
                        break
            
            if metadata is None:
                logger.error(f"No valid metadata found for {study_id}")
                return None
            
            return abundance_data, metadata
            
        except Exception as e:
            logger.error(f"Error processing dataset {study_id}: {str(e)}")
            return None
            
    def process_all_datasets(self) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Process all available MGnify datasets."""
        try:
            # List of high-quality microbiome studies
            studies = [
                "MGYS00001248",  # IBD study
                "MGYS00005745",  # Oral microbiome
                "MGYS00004773"   # Gut microbiome
            ]
            
            abundance_data = []
            metadata = []
            
            for study_id in studies:
                logger.info(f"Processing dataset {study_id}")
                result = self.process_dataset(study_id)
                if result is not None:
                    abundance_df, metadata_df = result
                    abundance_data.append(abundance_df)
                    metadata.append(metadata_df)
                    logger.info(f"Successfully processed {study_id}")
            
            if not abundance_data:
                logger.error("No datasets were successfully processed")
                return None
            
            # Combine datasets
            combined_abundance = pd.concat(abundance_data, axis=0)
            combined_metadata = pd.concat(metadata, axis=0)
            
            # Handle missing values
            combined_abundance = pd.DataFrame(
                self.imputer.fit_transform(combined_abundance),
                columns=combined_abundance.columns,
                index=combined_abundance.index
            )
            
            # Scale features
            combined_abundance = pd.DataFrame(
                self.scaler.fit_transform(combined_abundance),
                columns=combined_abundance.columns,
                index=combined_abundance.index
            )
            
            return combined_abundance, combined_metadata
            
        except Exception as e:
            logger.error(f"Error processing datasets: {str(e)}")
            return None
            
    def prepare_data(self) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
        """Prepare final dataset for modeling."""
        try:
            # Process all datasets
            result = self.process_all_datasets()
            if result is None:
                return None
                
            abundance_data, metadata = result
            
            # Ensure sample alignment
            common_samples = set(abundance_data.index) & set(metadata['sample_id'])
            if not common_samples:
                logger.error("No common samples between abundance data and metadata")
                return None
                
            X = abundance_data.loc[list(common_samples)]
            y = metadata[metadata['sample_id'].isin(common_samples)]['health_status']
            
            # Handle class imbalance
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            # Save processed data
            self.save_processed_data(X_resampled, y_resampled)
            
            return X_resampled, y_resampled
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            return None
            
    def save_processed_data(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Save processed data to files."""
        try:
            # Create processed directory if it doesn't exist
            self.processed_dir.mkdir(parents=True, exist_ok=True)
            
            # Save features and labels
            X.to_csv(self.processed_dir / "features_final.csv")
            y.to_csv(self.processed_dir / "labels_final.csv")
            
            # Save data statistics
            stats = {
                "n_samples": len(X),
                "n_features": X.shape[1],
                "class_distribution": y.value_counts().to_dict(),
                "feature_names": list(X.columns)
            }
            
            with open(self.processed_dir / "data_stats.json", "w") as f:
                json.dump(stats, f, indent=4)
                
            logger.info("Successfully saved processed data")
            
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
            raise

def main():
    try:
        processor = MGnifyDataProcessor()
        result = processor.prepare_data()
        
        if result is not None:
            X, y = result
            logger.info(f"Final dataset shape: {X.shape}")
            logger.info(f"Class distribution:\n{pd.Series(y).value_counts()}")
        else:
            logger.error("Failed to prepare data")
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 