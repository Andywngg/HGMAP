from pathlib import Path
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import entropy
from typing import Optional, Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MicrobiomeFeatureEngineer:
    """Feature engineering for microbiome data."""
    
    def __init__(
        self,
        n_pca_components: int = 50,
        min_prevalence: float = 0.1,
        min_abundance: float = 0.001,
        taxonomy_levels: Optional[List[str]] = None
    ):
        self.n_pca_components = n_pca_components
        self.min_prevalence = min_prevalence
        self.min_abundance = min_abundance
        self.taxonomy_levels = taxonomy_levels or [
            'phylum', 'class', 'order', 'family', 'genus', 'species'
        ]
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_pca_components)
        
    def calculate_diversity_metrics(self, abundance_matrix: pd.DataFrame) -> pd.DataFrame:
        """Calculate various diversity metrics."""
        metrics = {}
        
        # Richness (number of non-zero species)
        metrics['richness'] = (abundance_matrix > 0).sum(axis=1)
        
        # Shannon diversity
        metrics['shannon_diversity'] = abundance_matrix.apply(
            lambda x: entropy(x[x > 0]), axis=1
        )
        
        # Simpson diversity
        metrics['simpson_diversity'] = abundance_matrix.apply(
            lambda x: 1 - np.sum(np.square(x[x > 0])), axis=1
        )
        
        # Pielou's evenness
        metrics['pielou_evenness'] = metrics['shannon_diversity'] / np.log(metrics['richness'])
        
        # Berger-Parker dominance
        metrics['berger_parker_dominance'] = abundance_matrix.apply(
            lambda x: x.max() / x.sum(), axis=1
        )
        
        # Effective species (exp of Shannon entropy)
        metrics['effective_species'] = np.exp(metrics['shannon_diversity'])
        
        return pd.DataFrame(metrics, index=abundance_matrix.index)
    
    def filter_features(
        self,
        abundance_matrix: pd.DataFrame
    ) -> pd.DataFrame:
        """Filter features based on prevalence and abundance thresholds."""
        # Calculate prevalence
        prevalence = (abundance_matrix > 0).mean()
        
        # Calculate mean abundance
        mean_abundance = abundance_matrix.mean()
        
        # Filter based on thresholds
        keep_features = (prevalence >= self.min_prevalence) & (mean_abundance >= self.min_abundance)
        
        filtered_matrix = abundance_matrix.loc[:, keep_features]
        logger.info(f"Filtered features from {abundance_matrix.shape[1]} to {filtered_matrix.shape[1]}")
        
        return filtered_matrix
    
    def calculate_ratios(
        self,
        abundance_matrix: pd.DataFrame,
        taxonomy_ref: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate important microbial ratios."""
        ratios = {}
        
        # Firmicutes to Bacteroidetes ratio
        firmicutes = abundance_matrix.loc[:, taxonomy_ref['phylum'] == 'Firmicutes'].sum(axis=1)
        bacteroidetes = abundance_matrix.loc[:, taxonomy_ref['phylum'] == 'Bacteroidetes'].sum(axis=1)
        ratios['firmicutes_bacteroidetes_ratio'] = firmicutes / bacteroidetes
        
        # Prevotella to Bacteroides ratio
        prevotella = abundance_matrix.loc[:, taxonomy_ref['genus'] == 'Prevotella'].sum(axis=1)
        bacteroides = abundance_matrix.loc[:, taxonomy_ref['genus'] == 'Bacteroides'].sum(axis=1)
        ratios['prevotella_bacteroides_ratio'] = prevotella / bacteroides
        
        ratio_df = pd.DataFrame(ratios, index=abundance_matrix.index)
        ratio_df = ratio_df.replace([np.inf, -np.inf], np.nan)
        ratio_df = ratio_df.fillna(0)
        
        return ratio_df
    
    def _aggregate_by_taxonomy_level(
        self,
        abundance_matrix: pd.DataFrame,
        taxonomy_ref: pd.DataFrame,
        level: str
    ) -> pd.DataFrame:
        """Aggregate abundances by taxonomy level."""
        grouped = pd.DataFrame(index=abundance_matrix.index)
        
        for taxon in taxonomy_ref[level].unique():
            taxon_cols = taxonomy_ref[taxonomy_ref[level] == taxon].index
            grouped[f"{level}_{taxon}"] = abundance_matrix[taxon_cols].sum(axis=1)
        
        return grouped
    
    def engineer_features(
        self,
        abundance_df: pd.DataFrame,
        taxonomy_ref: pd.DataFrame,
        metadata_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Engineer features from microbiome data."""
        logger.info("Starting feature engineering process")
        
        # Filter features
        filtered_abundance = self.filter_features(abundance_df)
        
        # Calculate diversity metrics
        diversity_features = self.calculate_diversity_metrics(filtered_abundance)
        logger.info("Calculated diversity metrics")
        
        # Calculate important ratios
        ratio_features = self.calculate_ratios(filtered_abundance, taxonomy_ref)
        logger.info("Calculated microbial ratios")
        
        # Aggregate by taxonomy levels
        taxonomy_features = pd.DataFrame(index=filtered_abundance.index)
        for level in self.taxonomy_levels:
            level_features = self._aggregate_by_taxonomy_level(
                filtered_abundance, taxonomy_ref, level
            )
            taxonomy_features = pd.concat([taxonomy_features, level_features], axis=1)
        logger.info("Aggregated features by taxonomy levels")
        
        # Scale abundance data
        scaled_abundance = pd.DataFrame(
            self.scaler.fit_transform(filtered_abundance),
            index=filtered_abundance.index,
            columns=filtered_abundance.columns
        )
        
        # Apply PCA
        pca_features = pd.DataFrame(
            self.pca.fit_transform(scaled_abundance),
            index=filtered_abundance.index,
            columns=[f'PC{i+1}' for i in range(self.n_pca_components)]
        )
        logger.info(f"Applied PCA, explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        # Combine all features
        all_features = pd.concat([
            diversity_features,
            ratio_features,
            taxonomy_features,
            pca_features
        ], axis=1)
        
        # Add target variable if available in metadata
        if 'target' in metadata_df.columns:
            all_features['target'] = metadata_df['target']
        
        logger.info(f"Final feature matrix shape: {all_features.shape}")
        return all_features
    
    def save_features(
        self,
        features: pd.DataFrame,
        output_dir: Path,
        save_pca_components: bool = True
    ) -> None:
        """Save engineered features and PCA feature importance."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save features
        features.to_csv(output_dir / 'engineered_features.csv')
        
        # Save PCA components and explained variance
        if save_pca_components and hasattr(self, 'pca'):
            pca_results = {
                'components': self.pca.components_,
                'explained_variance_ratio': self.pca.explained_variance_ratio_,
                'singular_values': self.pca.singular_values_
            }
            np.savez(output_dir / 'pca_results.npz', **pca_results)
        
        logger.info(f"Saved engineered features to {output_dir}")

def main():
    """Run the feature engineering process"""
    try:
        # Set up paths
        data_dir = Path("data/integrated")
        output_dir = Path("data/features")
        
        logger.info("Starting feature engineering process...")
        
        # Load integrated data
        abundance_df = pd.read_csv(data_dir / "abundance_matrix.csv", index_col=0)
        taxonomy_ref = pd.read_csv(data_dir / "taxonomy_reference.csv", index_col=0)
        metadata_df = pd.read_csv(data_dir / "metadata.csv", index_col=0)
        
        # Initialize feature engineer
        feature_engineer = MicrobiomeFeatureEngineer(
            n_pca_components=50,
            min_prevalence=0.1,
            min_abundance=0.001
        )
        
        # Engineer features
        features_df = feature_engineer.engineer_features(
            abundance_df=abundance_df,
            taxonomy_ref=taxonomy_ref,
            metadata_df=metadata_df
        )
        
        # Save features
        feature_engineer.save_features(features_df, output_dir)
        
        logger.info("Feature engineering completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in feature engineering process: {str(e)}")
        raise

if __name__ == "__main__":
    main() 