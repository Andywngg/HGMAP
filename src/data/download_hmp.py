#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import gzip
import json
from tqdm import tqdm
import shutil
from typing import Tuple, Optional
from sklearn.impute import KNNImputer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HMPDownloader:
    """Download and process Human Microbiome Project data with validation and quality checks."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.hmp_dir = self.data_dir / "hmp1"  # Use hmp1 directory
        self.hmp_dir.mkdir(parents=True, exist_ok=True)
        
        # Quality control parameters - adjusted for mock data
        self.min_reads = 100  # Lowered for mock data
        self.min_features = 10  # Lowered for mock data
        self.min_prevalence = 0.001  # Lowered for mock data
    
    def download_abundance_data(self) -> pd.DataFrame:
        """Generate mock abundance data for testing."""
        logger.info("Downloading HMP abundance data...")
        
        abundance_file = self.hmp_dir / "abundance.tsv"
        
        try:
            if not abundance_file.exists():
                # Create mock data for testing
                logger.warning("Using mock data for testing...")
                n_samples = 80  # Smaller dataset than AGP
                n_features = 40  # Fewer features than AGP
                
                # Generate mock abundance data with realistic values
                rng = np.random.default_rng(43)  # Different seed than AGP
                abundance_matrix = rng.negative_binomial(n=2, p=0.3, size=(n_samples, n_features))
                abundance_matrix = abundance_matrix + 1  # Ensure no zeros
                
                # Create feature names with taxonomic information
                feature_names = []
                taxonomic_levels = ['k__Bacteria', 'p__Firmicutes', 'c__Clostridia', 
                                  'o__Clostridiales', 'f__Lachnospiraceae', 
                                  'g__Blautia', 's__unknown']
                
                for i in range(n_features):
                    # Randomly modify some levels for variety
                    modified_levels = taxonomic_levels.copy()
                    modified_levels[1] = np.random.choice(['p__Firmicutes', 'p__Bacteroidetes', 'p__Proteobacteria'])
                    modified_levels[-2] = f"g__Genus_{i}"
                    modified_levels[-1] = f"s__Species_{i}"
                    feature_names.append("; ".join(modified_levels))
                
                # Create DataFrame with sample names
                sample_names = [f"HMP_Sample_{i}" for i in range(n_samples)]
                abundance_df = pd.DataFrame(abundance_matrix, columns=feature_names, index=sample_names)
                
                # Save mock data
                abundance_df.to_csv(abundance_file, sep='\t')
                logger.info(f"Saved mock abundance data to {abundance_file}")
            else:
                abundance_df = pd.read_csv(abundance_file, sep='\t', index_col=0)
            
            # Quality control
            logger.info("Performing quality control...")
            
            # Filter low-count samples
            total_counts = abundance_df.sum(axis=1)
            abundance_df = abundance_df[total_counts >= self.min_reads]
            
            # Filter low-prevalence features
            prevalence = (abundance_df > 0).mean(axis=0)
            abundance_df = abundance_df.loc[:, prevalence >= self.min_prevalence]
            
            # Log transform
            abundance_df = np.log1p(abundance_df)
            
            logger.info(f"Final abundance matrix shape: {abundance_df.shape}")
            return abundance_df
            
        except Exception as e:
            logger.error(f"Error generating mock abundance data: {e}")
            raise
    
    def download_metadata(self) -> pd.DataFrame:
        """Generate mock metadata for testing."""
        logger.info("Downloading HMP metadata...")
        
        metadata_file = self.hmp_dir / "metadata.tsv"
        
        try:
            if not metadata_file.exists():
                # Create mock metadata for testing
                logger.warning("Using mock metadata for testing...")
                n_samples = 80  # Match with abundance data
                
                # Generate mock metadata with realistic values
                rng = np.random.default_rng(43)  # Same seed as abundance
                metadata = {
                    'sample_id': [f"HMP_Sample_{i}" for i in range(n_samples)],
                    'age': rng.integers(18, 80, n_samples),
                    'bmi': rng.normal(25, 5, n_samples),
                    'sex': rng.choice(['M', 'F'], n_samples),
                    'visit_number': rng.integers(1, 4, n_samples),
                    'health_status': rng.choice(['healthy', 'disease'], n_samples, p=[0.7, 0.3]),
                    'antibiotics_last_6months': rng.choice(['Yes', 'No'], n_samples, p=[0.2, 0.8]),
                    'diet_type': rng.choice(['Western', 'Mediterranean', 'Vegetarian'], n_samples)
                }
                
                metadata_df = pd.DataFrame(metadata)
                
                # Save mock metadata
                metadata_df.to_csv(metadata_file, sep='\t', index=False)
                logger.info(f"Saved mock metadata to {metadata_file}")
            else:
                metadata_df = pd.read_csv(metadata_file, sep='\t')
            
            logger.info(f"Processed metadata shape: {metadata_df.shape}")
            return metadata_df
            
        except Exception as e:
            logger.error(f"Error generating mock metadata: {e}")
            raise
    
    def process_and_validate_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Process and validate the complete dataset."""
        try:
            # Download and process data
            abundance_df = self.download_abundance_data()
            metadata_df = self.download_metadata()
            
            # Ensure sample alignment
            common_samples = set(abundance_df.index) & set(metadata_df['sample_id'])
            if not common_samples:
                raise ValueError("No common samples between abundance and metadata")
            
            abundance_df = abundance_df.loc[list(common_samples)]
            metadata_df = metadata_df[metadata_df['sample_id'].isin(common_samples)]
            
            # Validate data quality
            if abundance_df.isnull().sum().sum() > 0:
                raise ValueError("Abundance data contains missing values")
            
            if not all(abundance_df.sum(axis=1) > 0):
                raise ValueError("Some samples have zero total abundance")
            
            # Calculate class balance
            class_balance = metadata_df['health_status'].value_counts(normalize=True)
            logger.info(f"Class distribution:\n{class_balance}")
            
            if class_balance.min() < 0.1:
                logger.warning("Severe class imbalance detected")
            
            return abundance_df, metadata_df
            
        except Exception as e:
            logger.error(f"Error processing HMP data: {str(e)}")
            raise

if __name__ == "__main__":
    # Test the downloader
    downloader = HMPDownloader()
    abundance_df, metadata_df = downloader.process_and_validate_data()
    
    print("\nAbundance Data Summary:")
    print(abundance_df.describe())
    
    print("\nMetadata Summary:")
    print(metadata_df.describe())
    
    print("\nClass Distribution:")
    print(metadata_df['health_status'].value_counts(normalize=True)) 