import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import entropy
from skbio.diversity import alpha_diversity, beta_diversity
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class MicrobiomeFeatureEngineer:
    """Engineer features from microbiome abundance data"""
    
    def __init__(
        self,
        n_pca_components: int = 50,
        min_prevalence: float = 0.1,
        min_abundance: float = 0.001
    ):
        self.n_pca_components = n_pca_components
        self.min_prevalence = min_prevalence
        self.min_abundance = min_abundance
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_pca_components)
        self.feature_names_ = []
        
    def calculate_diversity_metrics(
        self,
        abundance_matrix: np.ndarray
    ) -> pd.DataFrame:
        """Calculate alpha and beta diversity metrics"""
        try:
            # Alpha diversity metrics
            shannon = alpha_diversity('shannon', abundance_matrix)
            simpson = alpha_diversity('simpson', abundance_matrix)
            observed_otus = alpha_diversity('observed_otus', abundance_matrix)
            
            # Beta diversity (sample-to-sample dissimilarity)
            beta_div = beta_diversity('braycurtis', abundance_matrix)
            
            # Combine metrics
            diversity_df = pd.DataFrame({
                'shannon_diversity': shannon,
                'simpson_diversity': simpson,
                'observed_species': observed_otus,
                'mean_beta_diversity': beta_div.mean(axis=1)
            })
            
            return diversity_df
            
        except Exception as e:
            logger.error(f"Error calculating diversity metrics: {e}")
            raise
    
    def calculate_ratios(
        self,
        abundance_df: pd.DataFrame,
        taxonomy_ref: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate important microbial ratios"""
        try:
            # Group species by phylum
            phylum_groups = taxonomy_ref.groupby('phylum')['species'].apply(list).to_dict()
            
            # Calculate phylum-level abundances
            phylum_abundances = {}
            for phylum, species_list in phylum_groups.items():
                common_species = list(set(species_list) & set(abundance_df.columns))
                if common_species:
                    phylum_abundances[phylum] = abundance_df[common_species].sum(axis=1)
            
            # Calculate key ratios
            ratios_df = pd.DataFrame()
            
            # Firmicutes to Bacteroidetes ratio
            if 'Firmicutes' in phylum_abundances and 'Bacteroidetes' in phylum_abundances:
                ratios_df['firmicutes_bacteroidetes_ratio'] = (
                    phylum_abundances['Firmicutes'] / 
                    phylum_abundances['Bacteroidetes'].replace(0, np.nan)
                )
            
            # Add other relevant ratios here
            
            return ratios_df
            
        except Exception as e:
            logger.error(f"Error calculating microbial ratios: {e}")
            raise
    
    def extract_functional_features(
        self,
        abundance_df: pd.DataFrame,
        health_assoc: pd.DataFrame
    ) -> pd.DataFrame:
        """Extract features based on functional associations"""
        try:
            functional_df = pd.DataFrame()
            
            # Calculate abundance of health-associated species
            health_prevalent = health_assoc[
                health_assoc['health_association'].str.contains('health', case=False, na=False)
            ].index
            
            disease_associated = health_assoc[
                health_assoc['health_association'].str.contains('disease|pathogen', case=False, na=False)
            ].index
            
            # Calculate summary metrics
            common_health = list(set(health_prevalent) & set(abundance_df.columns))
            common_disease = list(set(disease_associated) & set(abundance_df.columns))
            
            if common_health:
                functional_df['health_prevalent_abundance'] = abundance_df[common_health].sum(axis=1)
            if common_disease:
                functional_df['disease_associated_abundance'] = abundance_df[common_disease].sum(axis=1)
            
            return functional_df
            
        except Exception as e:
            logger.error(f"Error extracting functional features: {e}")
            raise
    
    def reduce_dimensions(
        self,
        abundance_matrix: np.ndarray
    ) -> np.ndarray:
        """Perform dimensionality reduction using PCA"""
        try:
            # Scale the data
            scaled_data = self.scaler.fit_transform(abundance_matrix)
            
            # Apply PCA
            pca_features = self.pca.fit_transform(scaled_data)
            
            # Store feature names
            self.feature_names_.extend([f'PC{i+1}' for i in range(self.n_pca_components)])
            
            return pca_features
            
        except Exception as e:
            logger.error(f"Error in dimensionality reduction: {e}")
            raise
    
    def engineer_features(
        self,
        abundance_df: pd.DataFrame,
        taxonomy_ref: pd.DataFrame,
        health_assoc: pd.DataFrame,
        metadata_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Engineer all features from the microbiome data"""
        try:
            feature_dfs = []
            
            # 1. Calculate diversity metrics
            logger.info("Calculating diversity metrics...")
            diversity_features = self.calculate_diversity_metrics(abundance_df.values)
            feature_dfs.append(diversity_features)
            
            # 2. Calculate microbial ratios
            logger.info("Calculating microbial ratios...")
            ratio_features = self.calculate_ratios(abundance_df, taxonomy_ref)
            feature_dfs.append(ratio_features)
            
            # 3. Extract functional features
            logger.info("Extracting functional features...")
            functional_features = self.extract_functional_features(abundance_df, health_assoc)
            feature_dfs.append(functional_features)
            
            # 4. Reduce dimensions of abundance data
            logger.info("Performing dimensionality reduction...")
            pca_features = self.reduce_dimensions(abundance_df.values)
            pca_df = pd.DataFrame(
                pca_features,
                columns=[f'PC{i+1}' for i in range(self.n_pca_components)]
            )
            feature_dfs.append(pca_df)
            
            # 5. Include metadata features if available
            if metadata_df is not None:
                # Select relevant metadata columns
                meta_features = metadata_df[[
                    col for col in metadata_df.columns 
                    if col.endswith('_encoded') or col in ['age_years', 'bmi']
                ]]
                feature_dfs.append(meta_features)
            
            # Combine all features
            combined_features = pd.concat(feature_dfs, axis=1)
            
            # Store feature names
            self.feature_names_ = combined_features.columns.tolist()
            
            logger.info(f"Generated {len(self.feature_names_)} features")
            return combined_features
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            raise
    
    def save_features(
        self,
        features_df: pd.DataFrame,
        output_dir: Path,
        prefix: str = "features"
    ) -> None:
        """Save engineered features"""
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save features
            features_df.to_csv(output_dir / f"{prefix}_matrix.csv", index=False)
            
            # Save feature names
            pd.Series(self.feature_names_).to_csv(
                output_dir / f"{prefix}_names.csv",
                index=False
            )
            
            logger.info(f"Saved features to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving features: {e}")
            raise 