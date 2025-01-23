import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

class DataIntegrator:
    """Integrates and standardizes microbiome datasets from multiple sources"""
    
    def __init__(self):
        self.label_encoders = {}
        self.taxonomy_map = None
        self.metadata_columns = [
            'age_years', 'sex', 'bmi', 'health_status', 'phenotype',
            'geography', 'sequencing_platform'
        ]
    
    def load_taxonomy_reference(self, taxonomy_file: Path) -> pd.DataFrame:
        """Load and process taxonomy reference data"""
        try:
            taxonomy_df = pd.read_csv(taxonomy_file)
            # Clean column names
            taxonomy_df.columns = [col.strip().lower().replace(' ', '_') 
                                 for col in taxonomy_df.columns]
            
            # Create unique taxonomy ID
            taxonomy_df['taxonomy_id'] = taxonomy_df.apply(
                lambda x: f"{x['phylum']}|{x['family']}|{x['species']}",
                axis=1
            )
            
            self.taxonomy_map = taxonomy_df
            logger.info(f"Loaded taxonomy reference with {len(taxonomy_df)} species")
            return taxonomy_df
            
        except Exception as e:
            logger.error(f"Error loading taxonomy reference: {e}")
            raise
    
    def standardize_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize metadata columns"""
        try:
            # Create copy to avoid modifying original
            meta_df = df.copy()
            
            # Standardize column names
            meta_df.columns = [col.strip().lower().replace(' ', '_') 
                             for col in meta_df.columns]
            
            # Standardize age
            if 'age' in meta_df.columns or 'age_(years)' in meta_df.columns:
                age_col = 'age' if 'age' in meta_df.columns else 'age_(years)'
                meta_df['age_years'] = pd.to_numeric(meta_df[age_col], errors='coerce')
            
            # Standardize sex
            if 'sex' in meta_df.columns:
                meta_df['sex'] = meta_df['sex'].str.lower()
                meta_df['sex'] = meta_df['sex'].map({'male': 'M', 'female': 'F'})
            
            # Standardize BMI
            if 'bmi' in meta_df.columns or 'bmi_(kgm²)' in meta_df.columns:
                bmi_col = 'bmi' if 'bmi' in meta_df.columns else 'bmi_(kgm²)'
                meta_df['bmi'] = pd.to_numeric(meta_df[bmi_col], errors='coerce')
            
            # Standardize health status
            if 'subject_health_status' in meta_df.columns:
                meta_df['health_status'] = meta_df['subject_health_status'].str.lower()
            
            # Encode categorical variables
            categorical_cols = ['sex', 'health_status', 'phenotype', 
                              'geography', 'sequencing_platform']
            
            for col in categorical_cols:
                if col in meta_df.columns:
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                    # Handle missing values before encoding
                    meta_df[col] = meta_df[col].fillna('unknown')
                    meta_df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(meta_df[col])
            
            return meta_df
            
        except Exception as e:
            logger.error(f"Error standardizing metadata: {e}")
            raise
    
    def integrate_abundance_data(
        self,
        abundance_files: List[Path],
        taxonomy_ref: pd.DataFrame
    ) -> pd.DataFrame:
        """Integrate abundance data from multiple files"""
        try:
            abundance_dfs = []
            
            for file in abundance_files:
                # Load abundance data
                df = pd.read_csv(file)
                
                # Clean column names
                df.columns = [col.strip().lower().replace(' ', '_') 
                            for col in df.columns]
                
                # Match species to taxonomy reference
                common_species = set(df.columns) & set(taxonomy_ref['species'])
                
                # Keep only species present in taxonomy reference
                df = df[list(common_species)]
                
                abundance_dfs.append(df)
            
            # Combine all abundance data
            combined_df = pd.concat(abundance_dfs, axis=0, ignore_index=True)
            
            # Fill missing values with 0 (assuming missing means not detected)
            combined_df = combined_df.fillna(0)
            
            logger.info(f"Integrated abundance data with shape {combined_df.shape}")
            return combined_df
            
        except Exception as e:
            logger.error(f"Error integrating abundance data: {e}")
            raise
    
    def process_health_associations(self, associations_file: Path) -> pd.DataFrame:
        """Process health associations data"""
        try:
            assoc_df = pd.read_csv(associations_file)
            
            # Clean column names
            assoc_df.columns = [col.strip().lower().replace(' ', '_') 
                              for col in assoc_df.columns]
            
            # Create a mapping of species to their health associations
            species_health_map = {}
            for _, row in assoc_df.iterrows():
                species = row.get('species_name', '').strip()
                if species:
                    species_health_map[species] = {
                        'health_association': row.get('pathophysiological_association', ''),
                        'evidence': row.get('remarks', ''),
                        'reference': f"{row.get('author', '')} ({row.get('year', '')})"
                    }
            
            return pd.DataFrame.from_dict(species_health_map, orient='index')
            
        except Exception as e:
            logger.error(f"Error processing health associations: {e}")
            raise
    
    def integrate_all_data(
        self,
        data_dir: Path,
        output_dir: Path
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Integrate all available data sources"""
        try:
            # Load taxonomy reference
            taxonomy_ref = self.load_taxonomy_reference(
                data_dir / "41467_2020_18476_MOESM4_ESM.csv"
            )
            
            # Process metadata from both studies
            metadata_2020 = pd.read_csv(data_dir / "41467_2020_18476_MOESM3_ESM.csv")
            metadata_2024 = pd.read_csv(data_dir / "41467_2024_51651_MOESM5_ESM.csv")
            
            # Standardize metadata
            metadata_2020 = self.standardize_metadata(metadata_2020)
            metadata_2024 = self.standardize_metadata(metadata_2024)
            
            # Combine metadata
            combined_metadata = pd.concat(
                [metadata_2020, metadata_2024],
                axis=0,
                ignore_index=True
            )
            
            # Process health associations
            health_assoc = self.process_health_associations(
                data_dir / "41467_2020_18476_MOESM5_ESM.csv"
            )
            
            # Save processed data
            output_dir.mkdir(parents=True, exist_ok=True)
            
            combined_metadata.to_csv(output_dir / "integrated_metadata.csv", index=False)
            taxonomy_ref.to_csv(output_dir / "taxonomy_reference.csv", index=False)
            health_assoc.to_csv(output_dir / "health_associations.csv")
            
            logger.info("Data integration completed successfully")
            
            return combined_metadata, taxonomy_ref, health_assoc
            
        except Exception as e:
            logger.error(f"Error in data integration: {e}")
            raise 