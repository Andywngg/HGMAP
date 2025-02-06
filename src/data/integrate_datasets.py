#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Dict, List
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from .download_agp import AGPDownloader
from .download_hmp import HMPDownloader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataIntegrator:
    """Integrate and preprocess data from multiple microbiome sources."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.agp_downloader = AGPDownloader(data_dir)
        self.hmp_downloader = HMPDownloader(data_dir)
        
        # Quality control parameters
        self.min_samples = 50
        self.min_features = 10
        self.min_prevalence = 0.001
        self.min_abundance = 1e-5
    
    def download_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Download and combine data from all sources."""
        logger.info("Downloading data from all sources...")
        
        abundance_data = {}
        metadata_data = {}
        
        # Download AGP data
        logger.info("Processing AGP data...")
        try:
            agp_abundance, agp_metadata = self.agp_downloader.process_and_validate_data()
            abundance_data['agp'] = agp_abundance
            metadata_data['agp'] = agp_metadata
        except Exception as e:
            logger.warning(f"Error processing AGP data: {str(e)}")
        
        # Download HMP data
        logger.info("Processing HMP data...")
        try:
            hmp_abundance, hmp_metadata = self.hmp_downloader.process_and_validate_data()
            abundance_data['hmp'] = hmp_abundance
            metadata_data['hmp'] = hmp_metadata
        except Exception as e:
            logger.warning(f"Error processing HMP data: {str(e)}")
        
        # Combine datasets
        combined_abundance = pd.concat(abundance_data.values(), axis=0)
        combined_metadata = pd.concat(metadata_data.values(), axis=0)
        
        return combined_abundance, combined_metadata
    
    def preprocess_abundance_data(self, abundance_df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess abundance data with improved error handling and type conversion."""
        logger.info("Preprocessing abundance data...")
        
        try:
            # Ensure numeric conversion
            numeric_df = abundance_df.apply(pd.to_numeric, errors='coerce')
            
            # Replace infinite values with NaN
            numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
            
            # Calculate prevalence (presence/absence)
            prevalence = (numeric_df > self.min_abundance).astype(float).mean()
            abundant_features = prevalence[prevalence >= self.min_prevalence].index
            
            if len(abundant_features) < self.min_features:
                logger.warning(f"Only {len(abundant_features)} features passed prevalence filter")
                abundant_features = prevalence.nlargest(self.min_features).index
            
            filtered_df = numeric_df[abundant_features]
            
            # Log transform with pseudo-count
            transformed_df = np.log1p(filtered_df.clip(lower=0))
            
            # Impute missing values
            imputer = KNNImputer(n_neighbors=5)
            imputed_array = imputer.fit_transform(transformed_df)
            
            # Convert back to DataFrame
            processed_df = pd.DataFrame(
                imputed_array,
                index=transformed_df.index,
                columns=transformed_df.columns
            )
            
            # Scale features
            scaler = StandardScaler()
            scaled_array = scaler.fit_transform(processed_df)
            
            final_df = pd.DataFrame(
                scaled_array,
                index=processed_df.index,
                columns=processed_df.columns
            )
            
            logger.info(f"Processed abundance data shape: {final_df.shape}")
            return final_df
            
        except Exception as e:
            logger.error(f"Error preprocessing abundance data: {str(e)}")
            raise
    
    def preprocess_metadata(self, metadata_df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess metadata."""
        logger.info("Preprocessing metadata...")
        
        # Standardize health status values
        metadata_df['health_status'] = metadata_df['health_status'].map({
            'healthy': 'healthy',
            'control': 'healthy',
            'normal': 'healthy',
            'disease': 'disease',
            'patient': 'disease',
            'sick': 'disease'
        }).fillna('unknown')
        
        # Remove samples with unknown health status
        metadata_df = metadata_df[metadata_df['health_status'] != 'unknown']
        
        # Standardize sex values
        metadata_df['sex'] = metadata_df['sex'].map({
            'M': 'M',
            'Male': 'M',
            'male': 'M',
            'F': 'F',
            'Female': 'F',
            'female': 'F'
        }).fillna('unknown')
        
        return metadata_df
    
    def harmonize_features(self, abundance_dfs: List[pd.DataFrame]) -> pd.DataFrame:
        """Harmonize features across multiple datasets."""
        logger.info("Harmonizing features across datasets...")
        
        try:
            # Find common features
            common_features = set.intersection(*[set(df.columns) for df in abundance_dfs])
            
            if len(common_features) < self.min_features:
                logger.warning(f"Only {len(common_features)} common features found")
                
                # Use union of features instead and impute missing values
                all_features = set.union(*[set(df.columns) for df in abundance_dfs])
                logger.info(f"Using {len(all_features)} total features with imputation")
                
                # Create unified DataFrame with all features
                unified_df = pd.DataFrame(index=pd.Index([]), columns=sorted(all_features))
                
                for df in abundance_dfs:
                    temp_df = pd.DataFrame(0, index=df.index, columns=unified_df.columns)
                    temp_df[df.columns] = df
                    unified_df = pd.concat([unified_df, temp_df])
                
                return unified_df
            
            # Use only common features
            harmonized_dfs = [df[sorted(common_features)] for df in abundance_dfs]
            return pd.concat(harmonized_dfs)
            
        except Exception as e:
            logger.error(f"Error harmonizing features: {str(e)}")
            raise
    
    def integrate_and_preprocess(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Integrate and preprocess all data."""
        try:
            # Download and combine data
            abundance_df, metadata_df = self.download_all_data()
            
            # Ensure sufficient samples
            if len(abundance_df) < self.min_samples:
                raise ValueError(f"Insufficient samples: {len(abundance_df)}")
            
            # Ensure sufficient features
            if abundance_df.shape[1] < self.min_features:
                raise ValueError(f"Insufficient features: {abundance_df.shape[1]}")
            
            # Preprocess abundance data
            processed_abundance = self.preprocess_abundance_data(abundance_df)
            
            # Preprocess metadata
            processed_metadata = self.preprocess_metadata(metadata_df)
            
            # Ensure samples match
            common_samples = processed_abundance.index.intersection(processed_metadata.index)
            if len(common_samples) < self.min_samples:
                raise ValueError(f"Insufficient matched samples: {len(common_samples)}")
            
            final_abundance = processed_abundance.loc[common_samples]
            final_metadata = processed_metadata.loc[common_samples]
            
            # Save processed data
            final_abundance.to_csv(self.processed_dir / "abundance.csv")
            final_metadata.to_csv(self.processed_dir / "metadata.csv")
            
            return final_abundance, final_metadata
            
        except Exception as e:
            logger.error(f"Error in data integration: {str(e)}")
            raise

if __name__ == "__main__":
    # Test the integrator
    integrator = DataIntegrator()
    abundance_df, metadata_df = integrator.integrate_and_preprocess()
    
    print("\nProcessed Abundance Data Summary:")
    print(abundance_df.describe())
    
    print("\nProcessed Metadata Summary:")
    print(metadata_df.describe())
    
    print("\nClass Distribution:")
    print(metadata_df['health_status'].value_counts(normalize=True)) 