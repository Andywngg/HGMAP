import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataIntegrator:
    def __init__(self):
        self.processed_dir = Path("data/processed")
        self.integrated_dir = Path("data/integrated")
        self.integrated_dir.mkdir(exist_ok=True)
        
    def load_and_process_2020_data(self):
        """Load and process data from the 2020 study."""
        try:
            # Load metadata with correct header handling
            metadata_2020 = pd.read_csv(self.processed_dir / "41467_2020_18476_MOESM3_ESM.csv", header=[1,2])
            
            # Debug: print column names
            logging.info("2020 dataset columns:")
            for col in metadata_2020.columns:
                logging.info(f"  {col}")
            
            # Process metadata
            metadata_2020['study'] = '2020'
            metadata_2020['health_status'] = metadata_2020[('Subject Health Status', 'Unnamed: 7_level_1')].map(
                lambda x: 'Non-healthy' if pd.notna(x) and str(x).strip() == 'Non-healthy' else 'Healthy'
            )
            
            # Extract sample ID and ensure it's a string
            metadata_2020['sample_id'] = metadata_2020[('Sample Accession', 'Unnamed: 4_level_1')].astype(str)
            
            # Select and rename relevant columns
            selected_cols = {
                ('Subject Meta-datab', 'Age (Years)'): 'Age (Years)',
                ('Unnamed: 10_level_0', 'Sex'): 'Sex',
                ('Unnamed: 11_level_0', 'BMI (kgm²)'): 'BMI (kgm²)',
                ('Phenotypea', 'Unnamed: 8_level_1'): 'Phenotype'
            }
            
            # Create a new DataFrame with selected columns
            result_df = pd.DataFrame()
            result_df['study'] = metadata_2020['study']
            result_df['health_status'] = metadata_2020['health_status']
            result_df['sample_id'] = metadata_2020['sample_id']
            
            for old_col, new_col in selected_cols.items():
                result_df[new_col] = metadata_2020[old_col]
            
            # Since we don't have diversity metrics for 2020 data, we'll create empty features
            features_2020 = pd.DataFrame(
                index=result_df['sample_id'],
                columns=['gmwi2', 'gmwi', 'shannon_diversity', 'simpson_diversity', 'species_richness']
            )
            
            # Log data info
            logging.info("2020 dataset health status distribution:")
            logging.info(result_df['health_status'].value_counts())
            logging.info(f"2020 dataset samples: {len(result_df)}")
            logging.info(f"2020 dataset features shape: {features_2020.shape}")
            
            return result_df, features_2020
            
        except Exception as e:
            logging.error(f"Error processing 2020 data: {str(e)}")
            raise
    
    def load_and_process_2024_data(self):
        """Load and process data from the 2024 study."""
        try:
            # Load metadata with correct header row
            metadata_2024 = pd.read_csv(self.processed_dir / "41467_2024_51651_MOESM5_ESM.csv", skiprows=[0,2])
            
            # Load diversity metrics data, skipping the first row which is a description
            diversity_2024 = pd.read_csv(self.processed_dir / "41467_2024_51651_MOESM8_ESM.csv", skiprows=[0])
            
            # Process metadata
            metadata_2024['study'] = '2024'
            metadata_2024['health_status'] = metadata_2024['Subject health status (Healthy or Non-healthy)'].map(
                lambda x: str(x).strip()
            )
            metadata_2024['sample_id'] = metadata_2024['Sample accession'].astype(str)
            
            # Select relevant columns from metadata
            metadata_2024 = metadata_2024[[
                'study', 'health_status', 'sample_id', 'Age (Years)', 'Sex', 'BMI (kgm²)', 'Phenotype'
            ]]
            
            # Process diversity metrics
            diversity_2024['sample_id'] = diversity_2024['Sample accession'].astype(str)
            diversity_2024.set_index('sample_id', inplace=True)
            
            # Drop any rows where all values are missing (like the empty third row)
            diversity_2024 = diversity_2024.dropna(how='all')
            
            # Select relevant metrics and convert to numeric
            feature_cols = ['GMWI2', 'GMWI', 'Shannon Index', 'Simpson Index', 'Species richness']
            features_2024 = diversity_2024[feature_cols].apply(pd.to_numeric, errors='coerce')
            
            # Rename columns to match our schema
            features_2024.columns = [
                'gmwi2', 'gmwi', 'shannon_diversity', 'simpson_diversity', 'species_richness'
            ]
            
            # Log data info
            logging.info("2024 dataset health status distribution:")
            logging.info(metadata_2024['health_status'].value_counts())
            logging.info(f"2024 dataset samples: {len(metadata_2024)}")
            logging.info(f"2024 dataset features shape: {features_2024.shape}")
            
            return metadata_2024, features_2024
            
        except Exception as e:
            logging.error(f"Error processing 2024 data: {str(e)}")
            raise
    
    def calculate_diversity_metrics(self, abundance_data):
        """Calculate diversity metrics from abundance data."""
        try:
            # Convert abundances to proportions
            row_sums = abundance_data.sum(axis=1)
            proportions = abundance_data.div(row_sums, axis=0).fillna(0)
            
            # Shannon diversity
            log_proportions = np.log(proportions.replace(0, 1))
            shannon = -(proportions * log_proportions).sum(axis=1)
            
            # Species richness (number of non-zero species)
            richness = (abundance_data > 0).sum(axis=1)
            
            # Evenness (Shannon diversity / log(richness))
            evenness = shannon / np.log(richness.replace(0, 1))
            
            return pd.DataFrame({
                'shannon_diversity': shannon,
                'species_richness': richness,
                'evenness': evenness
            }, index=abundance_data.index)
        except Exception as e:
            logging.error(f"Error calculating diversity metrics: {str(e)}")
            raise
    
    def integrate_data(self):
        """Integrate data from both studies."""
        try:
            # Load data from both studies
            metadata_2020, features_2020 = self.load_and_process_2020_data()
            metadata_2024, features_2024 = self.load_and_process_2024_data()
            
            # Combine metadata and features
            metadata = pd.concat([metadata_2020, metadata_2024], ignore_index=True)
            features = pd.concat([features_2020, features_2024])
            
            # Log initial data shapes
            logging.info(f"Initial metadata shape: {metadata.shape}")
            logging.info(f"Initial features shape: {features.shape}")
            logging.info(f"Metadata sample IDs: {len(metadata['sample_id'].unique())}")
            logging.info(f"Feature sample IDs: {len(features.index.unique())}")
            
            # Since 2020 data doesn't have features, we'll only use 2024 data
            metadata = metadata[metadata['study'] == '2024']
            
            # Clean up features:
            # 1. Drop rows where all feature values are missing
            features = features.dropna(how='all')
            # 2. For duplicate sample IDs, keep the first non-null occurrence
            features = features.groupby(features.index).first()
            
            # Ensure features align with metadata
            features = features.loc[metadata['sample_id']]
            metadata = metadata[metadata['sample_id'].isin(features.index)]
            
            # Log class distribution
            health_status_counts = metadata['health_status'].value_counts()
            logging.info("Health status distribution before balancing:")
            logging.info(health_status_counts)
            
            # Check if we have both classes
            if len(health_status_counts) < 2:
                logging.error("Error: Dataset does not contain both healthy and non-healthy samples")
                logging.error(f"Health status distribution: {health_status_counts.to_dict()}")
                raise ValueError("Dataset must contain both healthy and non-healthy samples")
            
            # Ensure balanced classes
            min_class_size = min(health_status_counts)
            logging.info(f"Minimum class size: {min_class_size}")
            
            # Sample equal numbers from each class
            balanced_samples = []
            for status in ['Healthy', 'Non-healthy']:
                class_samples = metadata[metadata['health_status'] == status]
                if len(class_samples) > 0:
                    sampled = class_samples.sample(n=min(len(class_samples), min_class_size), random_state=42)
                    balanced_samples.append(sampled)
            
            # Combine balanced samples
            metadata = pd.concat(balanced_samples, ignore_index=True)
            
            # Set sample_id as index for metadata to ensure alignment
            metadata.set_index('sample_id', inplace=True)
            
            # Ensure both dataframes have the same index order
            features = features.loc[metadata.index]
            
            # Save integrated data
            metadata.to_csv(self.integrated_dir / "integrated_metadata.csv")
            features.to_csv(self.integrated_dir / "abundance_data.csv")
            
            logging.info(f"Integrated data saved successfully:")
            logging.info(f"Total samples: {len(metadata)}")
            logging.info(f"Total features: {features.shape[1]}")
            logging.info(f"Health status distribution: {metadata['health_status'].value_counts().to_dict()}")
            
        except Exception as e:
            logging.error(f"Error integrating data: {str(e)}")
            raise

if __name__ == "__main__":
    integrator = DataIntegrator()
    integrator.integrate_data() 