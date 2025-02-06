import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import entropy
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
        
    def calculate_diversity_metrics(self, abundance_matrix: np.ndarray) -> pd.DataFrame:
        """Calculate various diversity metrics for each sample."""
        metrics = []
        
        for sample_abundances in abundance_matrix:
            # Filter out zeros for calculations
            nonzero_abundances = sample_abundances[sample_abundances > 0]
            total_abundance = np.sum(nonzero_abundances)
            
            # Normalize abundances
            if total_abundance > 0:
                normalized_abundances = nonzero_abundances / total_abundance
            else:
                normalized_abundances = nonzero_abundances
            
            # Calculate metrics
            sample_metrics = {
                # Richness (number of observed species)
                'richness': len(nonzero_abundances),
                
                # Shannon diversity
                'shannon_diversity': entropy(normalized_abundances) if len(normalized_abundances) > 0 else 0,
                
                # Simpson diversity
                'simpson_diversity': 1 - np.sum(normalized_abundances ** 2) if len(normalized_abundances) > 0 else 0,
                
                # Pielou's evenness
                'pielou_evenness': (entropy(normalized_abundances) / np.log(len(normalized_abundances))) 
                                  if len(normalized_abundances) > 1 else 0,
                
                # Berger-Parker dominance
                'berger_parker_dominance': np.max(normalized_abundances) if len(normalized_abundances) > 0 else 0,
                
                # Effective number of species (exponential of Shannon entropy)
                'effective_species': np.exp(entropy(normalized_abundances)) if len(normalized_abundances) > 0 else 0,
                
                # Additional metrics
                'total_abundance': total_abundance,
                'mean_abundance': np.mean(nonzero_abundances) if len(nonzero_abundances) > 0 else 0,
                'median_abundance': np.median(nonzero_abundances) if len(nonzero_abundances) > 0 else 0,
                'abundance_std': np.std(nonzero_abundances) if len(nonzero_abundances) > 1 else 0
            }
            metrics.append(sample_metrics)
        
        return pd.DataFrame(metrics)
    
    def filter_features(self, abundance_df: pd.DataFrame) -> pd.DataFrame:
        """Filter features based on prevalence and abundance thresholds."""
        # Calculate prevalence and mean abundance
        prevalence = (abundance_df > 0).mean()
        mean_abundance = abundance_df.mean()
        
        # Create masks for filtering
        prevalence_mask = prevalence >= self.min_prevalence
        abundance_mask = mean_abundance >= self.min_abundance
        
        # Apply filters
        filtered_df = abundance_df.loc[:, prevalence_mask & abundance_mask]
        
        logging.info(f"Filtered features from {abundance_df.shape[1]} to {filtered_df.shape[1]}")
        return filtered_df
    
    def calculate_ratios(self, abundance_df: pd.DataFrame, taxonomy_ref: pd.DataFrame) -> pd.DataFrame:
        """Calculate important microbial ratios."""
        ratios = pd.DataFrame(index=abundance_df.index)
        
        # Example ratios (customize based on domain knowledge)
        phylum_abundances = self._aggregate_by_taxonomy_level(abundance_df, taxonomy_ref, 'phylum')
        
        if 'Firmicutes' in phylum_abundances.columns and 'Bacteroidetes' in phylum_abundances.columns:
            ratios['firmicutes_bacteroidetes_ratio'] = (
                phylum_abundances['Firmicutes'] / 
                phylum_abundances['Bacteroidetes'].replace(0, np.nan)
            ).fillna(0)
        
        # Add more ratios
        if 'Proteobacteria' in phylum_abundances.columns:
            ratios['proteobacteria_ratio'] = phylum_abundances['Proteobacteria'] / phylum_abundances.sum()
        
        return ratios
    
    def _aggregate_by_taxonomy_level(
        self, 
        abundance_df: pd.DataFrame, 
        taxonomy_ref: pd.DataFrame,
        level: str
    ) -> pd.DataFrame:
        """Aggregate abundances by taxonomy level."""
        # Merge abundance with taxonomy
        merged_df = abundance_df.merge(
            taxonomy_ref[['otu_id', level]], 
            left_index=True, 
            right_on='otu_id'
        )
        
        # Aggregate by taxonomy level
        aggregated = merged_df.groupby(level).sum()
        return aggregated.T
    
    def calculate_beta_diversity(self, abundance_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate beta diversity metrics using Bray-Curtis dissimilarity."""
        # Normalize abundances
        abundance_matrix = abundance_df.values
        row_sums = abundance_matrix.sum(axis=1)
        normalized_matrix = abundance_matrix / row_sums[:, np.newaxis]
        
        # Calculate Bray-Curtis dissimilarity
        n_samples = abundance_matrix.shape[0]
        beta_div = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                # Bray-Curtis dissimilarity
                numerator = np.sum(np.abs(normalized_matrix[i] - normalized_matrix[j]))
                denominator = np.sum(normalized_matrix[i] + normalized_matrix[j])
                beta_div[i, j] = beta_div[j, i] = numerator / denominator if denominator > 0 else 0
        
        return pd.DataFrame(beta_div, index=abundance_df.index, columns=abundance_df.index)
    
    def engineer_features(
        self,
        abundance_df: pd.DataFrame,
        taxonomy_ref: pd.DataFrame,
        metadata_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Main feature engineering pipeline."""
        # Filter low prevalence/abundance features
        filtered_df = self.filter_features(abundance_df)
        
        # Calculate diversity metrics
        diversity_metrics = self.calculate_diversity_metrics(filtered_df.values)
        
        # Calculate taxonomic ratios
        ratio_features = self.calculate_ratios(filtered_df, taxonomy_ref)
        
        # Calculate beta diversity
        beta_div = self.calculate_beta_diversity(filtered_df)
        beta_div_features = pd.DataFrame({
            'mean_beta_div': beta_div.mean(),
            'std_beta_div': beta_div.std(),
            'max_beta_div': beta_div.max()
        }, index=filtered_df.index)
        
        # Perform PCA on filtered abundance data
        scaled_abundances = self.scaler.fit_transform(filtered_df)
        pca_features = pd.DataFrame(
            self.pca.fit_transform(scaled_abundances),
            index=filtered_df.index,
            columns=[f'PC{i+1}' for i in range(self.n_pca_components)]
        )
        
        # Calculate explained variance ratio
        explained_variance_ratio = pd.Series(
            self.pca.explained_variance_ratio_,
            index=[f'PC{i+1}_explained_var' for i in range(self.n_pca_components)]
        )
        
        # Combine all features
        feature_sets = [
            diversity_metrics,
            ratio_features,
            beta_div_features,
            pca_features,
            pd.DataFrame({'cumulative_variance_explained': np.cumsum(explained_variance_ratio)})
        ]
        
        if metadata_df is not None:
            feature_sets.append(metadata_df)
            
        combined_features = pd.concat(feature_sets, axis=1)
        
        # Log feature engineering summary
        logging.info(f"Engineered {combined_features.shape[1]} features for {combined_features.shape[0]} samples")
        
        return combined_features
    
    def save_features(self, features_df: pd.DataFrame, output_dir: str) -> None:
        """Save engineered features to disk."""
        output_path = Path(output_dir) / "engineered_features.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        features_df.to_csv(output_path, index=True)
        logging.info(f"Saved engineered features to {output_path}")
        
        # Save feature importance based on PCA
        if hasattr(self, 'pca') and hasattr(self.pca, 'components_'):
            importance_path = Path(output_dir) / "feature_importance.csv"
            feature_importance = pd.DataFrame(
                self.pca.components_.T,
                columns=[f'PC{i+1}' for i in range(self.n_pca_components)],
                index=self.feature_names_
            )
            feature_importance.to_csv(importance_path)
            logging.info(f"Saved feature importance to {importance_path}") 